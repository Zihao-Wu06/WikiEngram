"""A frozen Hugging Face decoder with DeepSeek-V4.1-style Engram modules attached.

No small DeepSeek model with native Engram weights exists (the only one, DeepSeek-V4.1-Flash, is 552B + 196B
parameters). For laptops, CI and single-GPU work, WikiEngram attaches Engram modules ported from DeepSeek's
code to a frozen decoder (Qwen3, SmolLM2, Llama-style): the base Engram is trained once with the backbone
frozen (``wikiengram adapt``), then frozen too, and personal knowledge goes into a sparse delta. This mirrors
the V4.1 setting (frozen backbone + frozen pretrained Engram + personal delta) at a size that runs anywhere.

Engram is applied to the residual stream right before the chosen decoder blocks via forward pre-hooks, so the
backbone's own code and weights are untouched.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from ..engram.compression import load_or_build_token_map
from ..engram.delta import PersonalDelta
from ..engram.hashing import NgramHasher
from ..engram.layout import EngramLayout, EngramSpec
from ..engram.memory import EngramBatch, EngramMemory
from ..engram.module import EngramModule
from ..engram.tables import BaseTable, DenseTable, ZeroTable
from ..utils.device import resolve_device, resolve_dtype


@dataclass
class GenerationResult:
    texts: list[str]
    prompt_tokens: list[int]
    new_tokens: list[int]
    seconds: float
    first_token_seconds: float


@dataclass
class EngramBaseCheckpoint:
    """On-disk base Engram for a retrofit backbone: spec, fusion params and dense tables."""

    path: Path
    spec: EngramSpec
    backbone: str
    info: dict = field(default_factory=dict)

    @classmethod
    def read(cls, path: str | Path) -> EngramBaseCheckpoint:
        path = Path(path)
        meta = json.loads((path / "engram.json").read_text())
        return cls(path, EngramSpec.from_dict(meta["spec"]), meta["backbone"], meta.get("info", {}))


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    base = getattr(model, getattr(model, "base_model_prefix", "model"), None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise TypeError(f"{type(model).__name__}: cannot find decoder layers (expected model.<prefix>.layers)")
    return layers


def param_checksum(params) -> str:
    """Exact hash of the raw bytes of every parameter (bit-level frozen-ness check, ~1 s per GB)."""
    h = hashlib.blake2b(digest_size=12)
    for p in params:
        t = p.detach().to("cpu").contiguous()
        h.update(str((tuple(t.shape), str(t.dtype))).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


class EngramCausalLM:
    """Wraps a frozen ``AutoModelForCausalLM`` and injects Engram before ``spec.layer_ids`` blocks."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        spec: EngramSpec,
        base_tables: dict[int, BaseTable] | None = None,
        fusion_state: dict[str, torch.Tensor] | None = None,
        delta: PersonalDelta | None = None,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        name: str = "custom",
    ):
        self.device = resolve_device(device)
        self.name = name
        self.model = model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        token_map, size = load_or_build_token_map(tokenizer, cache_dir)
        self.spec = spec.with_compressed_vocab(size)
        if self.spec.hidden_size != model.config.hidden_size:
            raise ValueError(f"spec hidden_size {self.spec.hidden_size} != model {model.config.hidden_size}")
        n_layers = len(_decoder_layers(self.model))
        if max(self.spec.layer_ids) >= n_layers:
            raise ValueError(f"Engram layer ids {self.spec.layer_ids} out of range for {n_layers} layers")
        self.layout = EngramLayout.from_spec(self.spec)
        self.hasher = NgramHasher(self.layout, token_map)
        self.engrams = nn.ModuleDict({str(lid): EngramModule(self.spec, lid) for lid in self.spec.layer_ids})
        if fusion_state is not None:
            self.engrams.load_state_dict(fusion_state)
        self.engrams.to(self.device)
        for p in self.engrams.parameters():
            p.requires_grad_(False)
        if base_tables is None:
            base_tables = {
                lid: ZeroTable(n, self.spec.head_dim)
                for lid, n in zip(self.spec.layer_ids, self.layout.num_embeddings)
            }
        self.memory = EngramMemory(self.layout, base_tables, delta, self.device)
        self.engram_enabled = True
        self._batch: EngramBatch | None = None
        layers = _decoder_layers(self.model)
        self._hooks = [
            layers[lid].register_forward_pre_hook(self._make_hook(lid), with_kwargs=True)
            for lid in self.spec.layer_ids
        ]

    # ---------------------------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        spec: EngramSpec | None = None,
        base_path: str | Path | None = None,
        delta: PersonalDelta | None = None,
        device: str | None = None,
        dtype: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> EngramCausalLM:
        """Load a backbone; with ``base_path`` also load its trained base Engram (spec, tables, fusion)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = resolve_device(device)
        torch_dtype = resolve_dtype(dtype, dev)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch_dtype)
        except TypeError:  # transformers < 4.56 names it torch_dtype
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
        base_tables, fusion = None, None
        if base_path is not None:
            ckpt = EngramBaseCheckpoint.read(base_path)
            spec = ckpt.spec
            fusion = load_file(str(Path(base_path) / "fusion.safetensors"))
            base_tables = {
                lid: DenseTable.load(Path(base_path) / f"table_{lid}.safetensors", device="cpu")
                for lid in spec.layer_ids
            }
        if spec is None:
            raise ValueError("pass an EngramSpec or a base_path")
        return cls(model, tokenizer, spec, base_tables, fusion, delta, dev, cache_dir, name=model_name_or_path)

    def save_base(self, path: str | Path, info: dict | None = None) -> None:
        """Write the base Engram (fusion params + dense tables + spec) for this backbone."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file({k: v.detach().cpu().contiguous() for k, v in self.engrams.state_dict().items()},
                  str(path / "fusion.safetensors"))
        for lid, table in self.memory.base_tables.items():
            if not isinstance(table, DenseTable):
                raise TypeError("only dense base tables can be saved")
            table.save(path / f"table_{lid}.safetensors")
        meta = {
            "spec": self.spec.to_dict(),
            "layout_fingerprint": self.layout.fingerprint(),
            "backbone": self.name,
            "fusion_checksum": self.fusion_checksum(),
            "saved_at": time.time(),
            "info": info or {},
        }
        (path / "engram.json").write_text(json.dumps(meta, indent=2))

    # ---------------------------------------------------------------------------------- state
    @property
    def delta(self) -> PersonalDelta | None:
        return self.memory.delta

    def attach_delta(self, delta: PersonalDelta | None) -> None:
        if delta is not None and delta.layout_fingerprint != self.layout.fingerprint():
            raise ValueError("delta was trained for a different Engram layout")
        if delta is not None:
            delta.to(self.device)
        self.memory.delta = delta

    def new_delta(self) -> PersonalDelta:
        return PersonalDelta(self.spec.layer_ids, self.spec.head_dim, self.layout.fingerprint(), self.device,
                             meta={"backbone": self.name})

    def backbone_checksum(self) -> str:
        return param_checksum(self.model.parameters())

    def fusion_checksum(self) -> str:
        return param_checksum(self.engrams.parameters())

    def base_checksum(self) -> str:
        """Checksum of the frozen in-memory base tables (memory-mapped native tables are read-only files)."""
        return param_checksum([t.weight for t in self.memory.base_tables.values() if isinstance(t, DenseTable)])

    def num_backbone_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    # ---------------------------------------------------------------------------------- forward
    def _make_hook(self, lid: int):
        module = self.engrams[str(lid)]

        def hook(_layer, args, kwargs):
            if self._batch is None or not self.engram_enabled:
                return None
            rows = self._batch.layers[lid]
            if args:
                hidden = module(args[0], rows.embeddings(), self._batch.token_mask)
                return (hidden, *args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = module(kwargs["hidden_states"], rows.embeddings(), self._batch.token_mask)
            return args, kwargs

        return hook

    @contextlib.contextmanager
    def engram_inputs(self, batch: EngramBatch | None):
        self._batch = batch
        try:
            yield
        finally:
            self._batch = None

    def build_batch(self, input_ids, attention_mask=None, hash_slice: slice | None = None,
                    delta_mask: torch.Tensor | None = None, **kwargs) -> EngramBatch:
        """Hash the full sequence (exact n-gram history), then keep only ``hash_slice`` positions."""
        hashes = self.hasher(input_ids, attention_mask)
        if hash_slice is not None:
            hashes = hashes[:, hash_slice]
            if attention_mask is not None:
                attention_mask = attention_mask[:, hash_slice]
            if delta_mask is not None:
                delta_mask = delta_mask[:, hash_slice]
        return self.memory.build_batch(hashes, attention_mask, delta_mask=delta_mask, **kwargs)

    def hidden_states(self, input_ids, attention_mask=None, batch: EngramBatch | None = None) -> torch.Tensor:
        """Last hidden states ``[B, L, D]`` with Engram applied from ``batch``."""
        base = getattr(self.model, self.model.base_model_prefix)
        with self.engram_inputs(batch):
            out = base(input_ids=input_ids.to(self.device),
                       attention_mask=None if attention_mask is None else attention_mask.to(self.device),
                       use_cache=False)
        return out.last_hidden_state

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.get_output_embeddings()(hidden)

    def answer_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        batch: EngramBatch | None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Cross-entropy on positions where ``labels != -100`` (next-token shifted), lm_head only there."""
        hidden = self.hidden_states(input_ids, attention_mask, batch)
        shift_labels = labels[:, 1:].to(self.device)
        sel = shift_labels != -100
        h = hidden[:, :-1][sel]
        logits = self.logits(h).float()
        loss = nn.functional.cross_entropy(logits, shift_labels[sel], reduction="none")
        if reduction == "none":
            # per-sample mean
            counts = sel.sum(1)
            sample = torch.zeros(sel.shape[0], device=self.device).index_add_(
                0, sel.nonzero()[:, 0], loss)
            return sample / counts.clamp_min(1)
        return loss.mean()

    # ---------------------------------------------------------------------------------- generation
    def _newline_ids(self) -> set[int]:
        """Token ids whose text contains a newline (answers stop there). Computed once per tokenizer."""
        if getattr(self, "_newline_cache", None) is None:
            tok = self.tokenizer
            self._newline_cache = {i for i in range(len(tok)) if "\n" in tok.decode([i])}
        return self._newline_cache

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 24,
        use_engram: bool = True,
        use_delta: bool = True,
        stop_at_newline: bool = True,
        delta_start: list[int] | None = None,
    ) -> GenerationResult:
        """Greedy decoding with a KV cache. Engram hashes are recomputed from the full token history at every
        step, so n-grams that straddle the prompt/answer boundary are hashed exactly as in training.

        ``delta_start[b]``: number of leading prompt tokens (e.g. the wiki context) where the personal delta is
        switched off; the base Engram still applies there."""
        tok = self.tokenizer
        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left")
        ids, mask = enc["input_ids"], enc["attention_mask"]
        prompt_tokens = mask.sum(1).tolist()
        position_ids = (mask.cumsum(-1) - 1).clamp_min(0)
        dmask = None
        if delta_start is not None and use_delta:
            pad = ids.shape[1] - mask.sum(1)
            start = pad + torch.as_tensor(delta_start, dtype=torch.long)
            dmask = torch.arange(ids.shape[1]).unsqueeze(0) >= start.unsqueeze(1)
        eos = tok.eos_token_id
        newline_ids = self._newline_ids() if stop_at_newline else set()
        start = time.perf_counter()
        first_token = None
        done = torch.zeros(ids.shape[0], dtype=torch.bool)
        generated = [[] for _ in prompts]
        past = None
        cur_ids, cur_pos = ids, position_ids
        prev_enabled = self.engram_enabled
        self.engram_enabled = use_engram
        try:
            for _ in range(max_new_tokens):
                batch = None
                if use_engram:
                    n = cur_ids.shape[1]
                    batch = self.build_batch(ids, mask, hash_slice=slice(ids.shape[1] - n, None), use_delta=use_delta,
                                             delta_mask=dmask)
                with self.engram_inputs(batch):
                    out = self.model(
                        input_ids=cur_ids.to(self.device),
                        attention_mask=mask.to(self.device),
                        position_ids=cur_pos.to(self.device),
                        past_key_values=past,
                        use_cache=True,
                    )
                past = out.past_key_values
                logits = out.logits[:, -1].float()
                if logits.shape[-1] > len(tok):  # padded output vocab (e.g. Qwen3): never emit ids with no token
                    logits[:, len(tok):] = float("-inf")
                nxt = logits.argmax(-1).cpu()
                if first_token is None:
                    if self.device.type == "mps":
                        torch.mps.synchronize()
                    first_token = time.perf_counter() - start
                for b, t in enumerate(nxt.tolist()):
                    if done[b]:
                        continue
                    if t == eos or t in newline_ids:
                        done[b] = True
                    else:
                        generated[b].append(t)
                if bool(done.all()):
                    break
                ids = torch.cat([ids, nxt[:, None]], dim=1)
                mask = torch.cat([mask, torch.ones(mask.shape[0], 1, dtype=mask.dtype)], dim=1)
                if dmask is not None:
                    dmask = torch.cat([dmask, torch.ones(dmask.shape[0], 1, dtype=torch.bool)], dim=1)
                cur_ids = nxt[:, None]
                cur_pos = cur_pos[:, -1:] + 1
        finally:
            self.engram_enabled = prev_enabled
        seconds = time.perf_counter() - start
        texts = [tok.decode(g, skip_special_tokens=True).strip() for g in generated]
        return GenerationResult(texts, prompt_tokens, [len(g) for g in generated], seconds, first_token or seconds)
