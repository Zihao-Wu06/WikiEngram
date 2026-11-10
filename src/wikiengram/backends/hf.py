"""A frozen Hugging Face decoder with an Engram variant attached ("retrofit" backend).

The MoE models that ship Engram natively are hundreds of GB. For laptops, CI and single-GPU work, WikiEngram
attaches an Engram variant (by default the DeepSeek algorithm, ported bit-exactly) to a frozen decoder (Qwen3,
SmolLM2, Llama-style): the base Engram is trained once with the backbone frozen (``wikiengram adapt``), then
frozen too, and personal knowledge goes into a sparse delta. This mirrors the native setting (frozen MoE
backbone + frozen pretrained Engram + personal delta) at a size that runs anywhere.

Engram is applied to the residual stream right before the chosen decoder blocks via forward pre-hooks, so the
backbone's own code and weights are untouched.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from ..engram.deepseek import DeepSeekEngram
from ..engram.delta import PersonalDelta
from ..engram.layout import EngramSpec
from ..engram.memory import EngramBatch, EngramMemory
from ..engram.tables import BaseTable, DenseTable, ZeroTable
from ..engram.variant import EngramVariant, variant_from_dict
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
    """On-disk base Engram for a backbone: variant description, fusion params and dense tables."""

    path: Path
    variant: dict
    backbone: str
    info: dict = field(default_factory=dict)

    @classmethod
    def read(cls, path: str | Path) -> EngramBaseCheckpoint:
        path = Path(path)
        meta = json.loads((path / "engram.json").read_text())
        return cls(path, meta["spec"], meta["backbone"], meta.get("info", {}))


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    base = getattr(model, getattr(model, "base_model_prefix", "model"), None)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise TypeError(f"{type(model).__name__}: cannot find decoder layers (expected model.<prefix>.layers)")
    return layers


SAMPLE_ABOVE = 1 << 30  # tensors with more elements (only in huge native models) are checksummed on a fixed strided sample
LN2 = 0.6931471805599453


def count_confident(token_losses: torch.Tensor, stats: dict | None) -> None:
    """Add to ``stats`` how many answer tokens the model already predicts with probability >= 0.5 (per-token loss
    <= ln 2). When all of them are, greedy decoding reproduces every trained answer with a margin."""
    if stats is not None:
        stats["confident"] = stats.get("confident", 0) + int((token_losses.detach() <= LN2).sum())
        stats["total"] = stats.get("total", 0) + int(token_losses.numel())


def offloaded_weights(model: nn.Module) -> dict[int, tuple]:
    """``id(param) -> (weights_map, name)`` for parameters accelerate keeps offloaded (CPU / disk ``device_map``).

    Such parameters sit on the ``meta`` device between forward calls; their values live in the hook's weights
    map, keyed by the name relative to the hooked module."""
    out: dict[int, tuple] = {}
    for module in model.modules():
        hook = getattr(module, "_hf_hook", None)
        for h in getattr(hook, "hooks", None) or ([hook] if hook is not None else []):
            if getattr(h, "offload", False) and getattr(h, "weights_map", None) is not None:
                for name, p in module.named_parameters(recurse=h.place_submodules):
                    out.setdefault(id(p), (h.weights_map, name))
    return out


def param_checksum(params, sample_above: int | None = SAMPLE_ABOVE, offloaded: dict | None = None) -> str:
    """Hash of the raw bytes of every parameter (bit-level frozen-ness check, ~1 s per GB).

    Tensors larger than ``sample_above`` elements (MoE experts, native n-gram tables of hundreds of GB) are hashed
    on a fixed strided sample of about 4M elements plus their shape and dtype: training touches every element
    of a tensor it updates, so a sample detects it, while hashing a 180B-parameter backbone twice per run would
    take minutes. ``sample_above=None`` hashes everything. Offloaded parameters (``meta`` placeholders) are read
    from ``offloaded`` (see :func:`offloaded_weights`)."""
    h = hashlib.blake2b(digest_size=12)
    for p in params:
        t = p.detach()
        if t.device.type == "meta":
            if not offloaded or id(p) not in offloaded:
                raise RuntimeError("a parameter is on the meta device but not offloaded by accelerate: the model "
                                   "was not fully loaded")
            weights_map, name = offloaded[id(p)]
            t = weights_map[name].detach()
        h.update(str((tuple(t.shape), str(t.dtype))).encode())
        if not t.numel():
            continue
        flat = t.reshape(-1)
        if sample_above is not None and flat.numel() > sample_above:
            step = flat.numel() // (1 << 22)
            flat = flat[::step]
        h.update(flat.to("cpu").contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class EngramCausalLM:
    """Wraps a frozen ``AutoModelForCausalLM`` and injects an Engram variant before its ``layer_ids`` blocks.

    ``engram`` is an :class:`~wikiengram.engram.variant.EngramVariant`, or an ``EngramSpec`` (DeepSeek variant).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        engram: EngramVariant | EngramSpec,
        base_tables: dict[int, BaseTable] | None = None,
        fusion_state: dict[str, torch.Tensor] | None = None,
        delta: PersonalDelta | None = None,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        name: str = "custom",
    ):
        self.device = resolve_device(device)
        self.name = name
        # a model dispatched with device_map (multi-GPU / CPU / disk offload) is placed by accelerate's hooks
        self.model = (model if getattr(model, "hf_device_map", None) else model.to(self.device)).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.streamer = None
        self.tokenizer = tokenizer
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.variant = engram if isinstance(engram, EngramVariant) else DeepSeekEngram.build(engram, tokenizer, cache_dir)
        v = self.variant
        if v.hidden_size != model.config.hidden_size:
            raise ValueError(f"Engram hidden_size {v.hidden_size} != model {model.config.hidden_size}")
        n_layers = len(_decoder_layers(self.model))
        if max(v.layer_ids) >= n_layers:
            raise ValueError(f"Engram layer ids {v.layer_ids} out of range for {n_layers} layers")
        self.engrams = nn.ModuleDict({str(lid): v.make_module(lid) for lid in v.layer_ids})
        if fusion_state is not None:
            self.engrams.load_state_dict(fusion_state)
        self.engrams.to(self.device)
        for p in self.engrams.parameters():
            p.requires_grad_(False)
        if base_tables is None:
            base_tables = {lid: ZeroTable(n, v.row_dim) for lid, n in zip(v.layer_ids, v.num_rows)}
        self.memory = EngramMemory(v, base_tables, delta, self.device)
        self.engram_enabled = True
        self._batch: EngramBatch | None = None
        layers = _decoder_layers(self.model)
        self._hooks = [
            layers[lid].register_forward_pre_hook(self._make_hook(lid), with_kwargs=True)
            for lid in v.layer_ids
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
        variant: EngramVariant | None = None,
        **load_kwargs,
    ) -> EngramCausalLM:
        """Load a backbone; with ``base_path`` also load its trained base Engram (variant, tables, fusion).
        Otherwise attach a fresh ``variant`` (or a DeepSeek-style ``spec``). ``load_kwargs`` go to transformers
        (``device_map``, ``max_memory``, ``offload_folder``, quantization, ...)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = resolve_device(device)
        torch_dtype = resolve_dtype(dtype, dev)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch_dtype, **load_kwargs)
        except TypeError:  # transformers < 4.56 names it torch_dtype
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, **load_kwargs)
        base_tables, fusion = None, None
        engram: EngramVariant | EngramSpec | None = variant or spec
        if base_path is not None:
            ckpt = EngramBaseCheckpoint.read(base_path)
            engram = variant_from_dict(ckpt.variant, tokenizer, cache_dir)
            fusion = load_file(str(Path(base_path) / "fusion.safetensors"))
            base_tables = {
                lid: DenseTable.load(Path(base_path) / f"table_{lid}.safetensors", device="cpu")
                for lid in engram.layer_ids
            }
        if engram is None:
            raise ValueError("pass an EngramVariant, an EngramSpec or a base_path")
        return cls(model, tokenizer, engram, base_tables, fusion, delta, dev, cache_dir, name=model_name_or_path)

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
            "spec": self.variant.to_dict(),
            "variant": self.variant.name,
            "layout_fingerprint": self.variant.fingerprint(),
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

    def enable_streaming(self, store_on_cpu: bool = True) -> None:
        """Train through the model block by block (for models dispatched with offloading)."""
        from .streaming import StreamingTrainer

        self.streamer = StreamingTrainer(self, store_on_cpu=store_on_cpu)

    def attach_delta(self, delta: PersonalDelta | None) -> None:
        if delta is not None and delta.layout_fingerprint != self.variant.fingerprint():
            raise ValueError("delta was trained for a different Engram layout")
        if delta is not None:
            delta.to(self.device)
        self.memory.delta = delta

    def new_delta(self) -> PersonalDelta:
        return PersonalDelta(self.variant.layer_ids, self.variant.row_dim, self.variant.fingerprint(), self.device,
                             meta={"backbone": self.name, "variant": self.variant.name})

    def backbone_params(self):
        return list(self.model.parameters())

    def fusion_params(self):
        return list(self.engrams.parameters())

    def backbone_checksum(self) -> str:
        return param_checksum(self.backbone_params(), offloaded=offloaded_weights(self.model))

    def fusion_checksum(self) -> str:
        return param_checksum(self.fusion_params(), offloaded=offloaded_weights(self.model))

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
        hashes = self.variant.hash(input_ids, attention_mask)
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
        stats: dict | None = None,
    ) -> torch.Tensor:
        """Cross-entropy on positions where ``labels != -100`` (next-token shifted), lm_head only there.
        ``stats`` (optional) receives ``confident`` / ``total``: answer tokens predicted with p >= 0.5."""
        hidden = self.hidden_states(input_ids, attention_mask, batch)
        shift_labels = labels[:, 1:].to(self.device)
        sel = shift_labels != -100
        h = hidden[:, :-1][sel]
        logits = self.logits(h).float()
        loss = nn.functional.cross_entropy(logits, shift_labels[sel], reduction="none")
        count_confident(loss, stats)
        if reduction == "none":
            # per-sample mean
            counts = sel.sum(1)
            sample = torch.zeros(sel.shape[0], device=self.device).index_add_(
                0, sel.nonzero()[:, 0], loss)
            return sample / counts.clamp_min(1)
        return loss.mean()

    # ---------------------------------------------------------------------------------- prompt format
    prompt_format: str = "completion"  # or "chat": questions become the user turn of the chat template

    def format_prompt(self, text: str) -> str:
        """The prompt as this model is asked (see :mod:`wikiengram.prompts`)."""
        if self.prompt_format != "chat":
            return text
        from ..prompts import chat_wrap

        return chat_wrap(self.tokenizer, text)

    def format_sample(self, sample) -> tuple[str, str, bool]:
        """``(prompt, target, add_special_tokens)`` of a training sample. Question-answer samples follow the
        prompt format. With the chat format the answer starts on a fresh line (its leading space goes) and ends
        the assistant turn with the end-of-turn token, so the model answers and stops, as an API client expects."""
        if self.prompt_format == "chat" and getattr(sample, "qa", False):
            end = self.tokenizer.eos_token or "\n"
            return self.format_prompt(sample.prompt), sample.target.strip() + end, False
        return sample.prompt, sample.target, True

    def prompt_wrapper(self) -> tuple[str, str]:
        """Text the format adds before and after a question (empty for plain completion)."""
        if self.prompt_format != "chat":
            return "", ""
        marker = "⁣WIKIENGRAM⁣"
        before, _, after = self.format_prompt(marker).partition(marker)
        return before, after

    def context_tokens(self, context: str, question: str) -> int:
        """Leading tokens of the formatted prompt ``context + question`` that belong to the wiki context (with
        any special or template tokens before it): the positions where the personal delta stays off."""
        if not context:
            return 0
        if self.prompt_format != "chat":
            return len(self.tokenizer(context)["input_ids"])
        full = self.format_prompt(context + question)
        cut = full.index(context) + len(context)
        return len(self.tokenizer(full[:cut], add_special_tokens=False)["input_ids"])

    # ---------------------------------------------------------------------------------- generation
    def stop_token_ids(self) -> set[int]:
        """End-of-turn tokens: the tokenizer's EOS plus the model's generation-config EOS ids."""
        ids = {self.tokenizer.eos_token_id} if self.tokenizer.eos_token_id is not None else set()
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        ids |= set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())
        return ids

    @torch.no_grad()
    def complete(self, input_ids: list[int], max_new_tokens: int = 256, temperature: float = 0.0,
                 top_p: float = 1.0, stop_ids: set[int] | None = None, use_delta: bool = True,
                 delta_start: int = 0, seed: int | None = None, on_token=None) -> list[int]:
        """Free-form decoding of one sequence (the chat API): greedy at temperature 0, else nucleus sampling.
        Engram rows are hashed from the full history at every step, as in :meth:`generate`. ``delta_start``:
        leading positions where the personal delta stays off (wiki context injected into the prompt).
        ``on_token(token_id)`` is called for every new token and may return True to stop."""
        ids = torch.tensor([list(input_ids)], dtype=torch.long)
        mask = torch.ones_like(ids)
        stop = set(stop_ids) if stop_ids is not None else self.stop_token_ids()
        dmask = (torch.arange(ids.shape[1]) >= delta_start).unsqueeze(0) if delta_start and use_delta else None
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        cur, pos, past, out = ids, torch.arange(ids.shape[1]).unsqueeze(0), None, []
        streaming = [m for m in self.engrams.values() if hasattr(m, "begin_stream")]
        for m in streaming:
            m.begin_stream()
        try:
            for _ in range(max_new_tokens):
                n = cur.shape[1]
                batch = self.build_batch(ids, mask, hash_slice=slice(ids.shape[1] - n, None), use_delta=use_delta,
                                         delta_mask=dmask)
                with self.engram_inputs(batch):
                    o = self.model(input_ids=cur.to(self.device), attention_mask=mask.to(self.device),
                                   position_ids=pos.to(self.device), past_key_values=past, use_cache=True,
                                   **self._last_logits_kw())
                past = o.past_key_values
                logits = o.logits[0, -1].float().cpu()
                if logits.shape[-1] > len(self.tokenizer):  # padded output vocab: never emit ids with no token
                    logits[len(self.tokenizer):] = float("-inf")
                if temperature <= 0:
                    t = int(logits.argmax())
                else:
                    probs = torch.softmax(logits / temperature, -1)
                    if top_p < 1.0:
                        sp, order = probs.sort(descending=True)
                        sp[(sp.cumsum(0) - sp) >= top_p] = 0  # keep the smallest set reaching top_p
                        probs = torch.zeros_like(probs).scatter(0, order, sp)
                    t = int(torch.multinomial(probs / probs.sum(), 1, generator=gen))
                if t in stop:
                    break
                out.append(t)
                if on_token is not None and on_token(t):
                    break
                nxt = torch.tensor([[t]])
                ids = torch.cat([ids, nxt], dim=1)
                mask = torch.cat([mask, torch.ones(1, 1, dtype=mask.dtype)], dim=1)
                if dmask is not None:
                    dmask = torch.cat([dmask, torch.ones(1, 1, dtype=torch.bool)], dim=1)
                cur, pos = nxt, pos[:, -1:] + 1
        finally:
            for m in streaming:
                m.end_stream()
        return out

    def _last_logits_kw(self) -> dict:
        """Ask the model for the last position's logits only when it supports that (prefill of a big batch would
        otherwise hold batch x prompt x vocabulary logits)."""
        if getattr(self, "_last_logits_cache", None) is None:
            try:
                params = inspect.signature(self.model.forward).parameters
            except (TypeError, ValueError):
                params = {}
            self._last_logits_cache = ({"logits_to_keep": 1} if "logits_to_keep" in params
                                       else {"num_logits_to_keep": 1} if "num_logits_to_keep" in params else {})
        return self._last_logits_cache

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
        raw: bool = False,
    ) -> GenerationResult:
        """Greedy decoding with a KV cache. Engram hashes are recomputed from the full token history at every
        step, so n-grams that straddle the prompt/answer boundary are hashed exactly as in training.

        ``delta_start[b]``: number of leading prompt tokens (e.g. the wiki context) where the personal delta is
        switched off; the base Engram still applies there. ``raw``: continue the text as given, outside the prompt
        format (how statement samples are trained, also for chat models)."""
        tok = self.tokenizer
        texts = list(prompts) if raw else [self.format_prompt(p) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left",
                  add_special_tokens=raw or self.prompt_format != "chat")
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
        streaming = [m for m in self.engrams.values() if hasattr(m, "begin_stream")]
        for m in streaming:  # e.g. Qwen PLE's dilated conv looks back over earlier positions
            m.begin_stream()
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
                        **self._last_logits_kw(),
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
            for m in streaming:
                m.end_stream()
        seconds = time.perf_counter() - start
        texts = [tok.decode(g, skip_special_tokens=True).strip() for g in generated]
        return GenerationResult(texts, prompt_tokens, [len(g) for g in generated], seconds, first_token or seconds)
