"""Native backend: a Hugging Face MoE model that ships its own Engram-style module.

The model's own fusion layers (for Qwen3.8-Flash-Next: ``Qwen4ExpTextPLELayer``) stay in place and frozen. Only
their *table lookup* is redirected to WikiEngram's memory: the model's own table becomes the frozen base, the
personal delta is added on top, and rows are addressed with the variant's hashing, which is bit-exact with the
model's (``tests/test_variant_parity.py``). With an empty delta the model is therefore exactly the model as
shipped (``tests/test_native_qwen.py``).

Everything else - consolidation, sync, verification, fusion routing, generation - is shared with the retrofit
backend through :class:`~wikiengram.backends.hf.EngramCausalLM`.

Supported families: Qwen PLE (``Qwen4ExpForCausalLM`` / ``Qwen4ExpForConditionalGeneration``). DeepSeek-V4.1
has no trainable Hugging Face implementation yet (see docs/NATIVE_MOE.md).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from ..engram.memory import EngramMemory
from ..engram.tables import BaseTable, DenseTable, ShardedTable
from ..engram.variant import EngramVariant, detect_variant
from ..utils.device import resolve_device, resolve_dtype
from .hf import EngramCausalLM, _decoder_layers, offloaded_weights, param_checksum


class _MemoryLookup(nn.Module):
    """Stands in for a native n-gram embedding module: returns rows from WikiEngram's memory for the batch the
    owner has set, or defers to the original module when no Engram batch is active."""

    def __init__(self, owner: NativeEngramLM, layer_id: int, original: nn.Module):
        super().__init__()
        self.layer_id = layer_id
        self.original = original
        object.__setattr__(self, "_owner", owner)  # not a submodule (avoid a reference cycle in state_dict)

    def forward(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        owner = self._owner
        batch = owner._batch
        if batch is None or not owner.engram_enabled:
            if self.original.ngram_embedding.weight.device.type == "meta":
                raise RuntimeError("this model reads its n-gram table from the checkpoint; run it through "
                                   "NativeEngramLM (hidden_states / generate), which supplies the rows")
            return self.original(input_ids, *args, **kwargs)
        rows = batch.layers[self.layer_id]
        emb = rows.embeddings()  # [B, L, rows_per_position, row_dim]
        if emb.shape[:2] != tuple(input_ids.shape[:2]):
            raise RuntimeError(f"Engram batch covers {tuple(emb.shape[:2])} positions, model asked for "
                               f"{tuple(input_ids.shape[:2])}")
        return emb.flatten(-2).to(self.original.ngram_embedding.weight.dtype)


def _native_ple_layers(model: nn.Module) -> dict[int, nn.Module]:
    out = {}
    for i, layer in enumerate(_decoder_layers_any(model)):
        ple = getattr(layer, "ple", None)
        if ple is not None:
            out[i] = ple
    return out


def _decoder_layers_any(model: nn.Module) -> nn.ModuleList:
    try:
        return _decoder_layers(model)
    except TypeError:  # multimodal wrappers: model.model.language_model.layers
        base = getattr(model, getattr(model, "base_model_prefix", "model"))
        return base.language_model.layers


def fast_qsa_indexer(model: nn.Module) -> int:
    """Give Qwen3.8's QSA indexer an exact shortcut for sequences within its token budget.

    The indexer keeps, for each query, the ``budget / ratio`` best blocks of ``ratio`` keys plus the incomplete
    last block. When a query sees at most ``budget`` tokens, that is every visible token: the result is the
    attention mask itself. transformers' reference code still scores the blocks in a Python loop over every
    (sequence, query) pair, about 15 small GPU operations each, which made it the slowest part of a training step
    (about 120 s of 227 s at 168 sequences x 100 tokens). Longer sequences use the original code. Returns the
    number of indexer classes patched."""
    patched = 0
    for module in model.modules():
        cls = type(module)
        if not cls.__name__.endswith("QSAIndexer") or getattr(cls, "_wikiengram_original_forward", None):
            continue
        original = cls.forward

        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, _original=original):
            if attention_mask is None or attention_mask.shape[-1] // self.compress_ratio > self.block_topk:
                return _original(self, hidden_states, position_embeddings, attention_mask, past_key_values)
            if past_key_values is not None:  # keep the indexer's key cache complete for later, longer steps
                b, n, _ = hidden_states.shape
                keys = self.index_qk_proj(hidden_states)[..., self.index_n_heads * self.index_head_dim:]
                past_key_values.update_indexer(keys.reshape(b, n, -1, self.index_head_dim).squeeze(2),
                                               self.layer_idx)
            visible = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0
            if attention_mask.is_floating_point():
                return torch.where(visible, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
            return visible

        cls._wikiengram_original_forward = original
        cls.forward = forward
        patched += 1
    return patched


class NativeEngramLM(EngramCausalLM):
    """Wraps a frozen native Engram MoE model (Hugging Face implementation) for personal memory."""

    def __init__(self, model: nn.Module, tokenizer, variant: EngramVariant | None = None,
                 delta=None, device=None, name: str = "custom-native",
                 base_tables: dict[int, BaseTable] | None = None):
        """``base_tables``: readers for the model's own tables when they are not in memory (see
        :meth:`from_checkpoint`); by default the native table parameters are used."""
        self.device = resolve_device(device)
        self.name = name
        placed = getattr(model, "hf_device_map", None) or any(p.device.type == "meta" for p in model.parameters())
        self.model = (model if placed else model.to(self.device)).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        fast_qsa_indexer(self.model)
        self.streamer = None
        self.tokenizer = tokenizer
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        if variant is None:
            config = model.config.to_dict()
            cls = detect_variant(config)
            if cls is None:
                raise ValueError(f"{type(model).__name__}: no registered Engram variant recognises this config")
            variant = cls.from_model_config(config, tokenizer)
        self.variant = variant
        ples = _native_ple_layers(self.model)
        if tuple(sorted(ples)) != tuple(variant.layer_ids):
            raise ValueError(f"model has native Engram layers {sorted(ples)}, variant expects {variant.layer_ids}")
        self.engrams = nn.ModuleDict({str(lid): ples[lid] for lid in variant.layer_ids})
        tables = dict(base_tables or {})
        for lid, ple in ples.items():
            expected = (variant.num_rows[variant.layer_ids.index(lid)], variant.row_dim)
            if lid not in tables:
                tables[lid] = DenseTable(ple.ple_embedding.ngram_embedding.weight)
            if (tables[lid].num_rows, tables[lid].dim) != expected:
                raise ValueError(f"layer {lid}: native table {(tables[lid].num_rows, tables[lid].dim)} does not "
                                 f"match the variant {expected}")
            ple.ple_embedding = _MemoryLookup(self, lid, ple.ple_embedding)
        self.memory = EngramMemory(variant, tables, delta, self.device)
        self.engram_enabled = True
        self._batch = None
        self._hooks = []

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, delta=None, device: str | None = None,
                        dtype: str | None = None, cache_dir: str | Path | None = None, **load_kwargs
                        ) -> NativeEngramLM:
        """Load a native Engram model with transformers (any ``device_map`` / quantization kwargs pass through)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = resolve_device(device)
        torch_dtype = resolve_dtype(dtype, dev)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch_dtype, **load_kwargs)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, **load_kwargs)
        return cls(model, tokenizer, delta=delta, device=dev, name=model_name_or_path)

    @classmethod
    def from_checkpoint(cls, path: str | Path, delta=None, device: str | None = None, dtype: str | None = None,
                        expert_cache_gb: float = 0.0) -> NativeEngramLM:
        """Run a native Engram MoE model straight from its safetensors directory, whatever its size: resident
        non-expert weights, experts read on demand, the n-gram table read row by row (see
        :mod:`wikiengram.backends.checkpoint`). FP8 releases are dequantized on the fly."""
        from transformers import AutoTokenizer

        from ..utils.device import configure_threads
        from .checkpoint import load_streamed_model

        configure_threads()  # streaming is CPU-heavy: never oversubscribe a container's CPU quota
        model, ckpt, prefix = load_streamed_model(path, device, dtype, expert_cache_gb)
        tables = {lid: ShardedTable.from_checkpoint(ckpt.root, ckpt.weight_map,
                                                    f"{prefix}layers.{lid}.ple.ple_embedding.ngram_embedding")
                  for lid in _native_ple_layers(model)}
        return cls(model, AutoTokenizer.from_pretrained(path), delta=delta, device=device, name=str(path),
                   base_tables=tables)

    # ---------------------------------------------------------------------------------- no-delta paths
    # A native model always applies its own n-gram memory: "without Engram" means the shipped table without the
    # personal delta (identical to the shipped model, tests/test_native_qwen.py), never removing the module.
    def hidden_states(self, input_ids, attention_mask=None, batch=None):
        if batch is None:
            batch = self.build_batch(input_ids, attention_mask, use_delta=False)
        return super().hidden_states(input_ids, attention_mask, batch)

    def generate(self, prompts, max_new_tokens: int = 24, use_engram: bool = True, use_delta: bool = True, **kw):
        return super().generate(prompts, max_new_tokens, use_engram=True, use_delta=use_delta and use_engram, **kw)

    # ---------------------------------------------------------------------------------- parameter groups
    def _ple_param_ids(self) -> set[int]:
        return {id(p) for m in self.engrams.values() for p in m.parameters()}

    def backbone_params(self):
        ple = self._ple_param_ids()
        return [p for p in self.model.parameters() if id(p) not in ple]

    def fusion_params(self):
        return [p for m in self.engrams.values() for n, p in m.named_parameters()
                if not n.endswith("ngram_embedding.weight")]

    def base_checksum(self) -> str:
        tables = list(self.memory.base_tables.values())
        if all(isinstance(t, DenseTable) for t in tables):
            return param_checksum([t.weight for t in tables], offloaded=offloaded_weights(self.model))
        return hashlib.blake2b("|".join(t.checksum() for t in tables).encode(), digest_size=12).hexdigest()

    def base_identity(self) -> str:
        """Identity of the native base (fusion weights + sampled table bytes) recorded in every delta."""
        payload = f"{self.fusion_checksum()}|{self.base_checksum()}|{self.variant.fingerprint()}"
        return hashlib.blake2b(payload.encode(), digest_size=12).hexdigest()

    def save_base(self, path, info=None) -> None:
        raise TypeError("a native model ships its own Engram; there is no separate base to save")
