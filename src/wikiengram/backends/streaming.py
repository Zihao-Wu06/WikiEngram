"""Train personal memory through a frozen model that does not fit in accelerator memory.

The frozen model can be placed anywhere by Hugging Face / accelerate ``device_map``: on one or several GPUs, in
CPU RAM, or offloaded to disk. Accelerate's hooks then bring a layer's weights in for each forward call. That
already makes *inference* work on any hardware; this module adds *training* of the personal delta:

1. one forward pass without gradients records, for every block from the first Engram block on, exactly what
   the block receives (hidden state, masks, position embeddings, ... - whatever the architecture passes);
2. the loss is computed on the final hidden state;
3. the backward pass walks the blocks from the last to the first Engram block, *recomputing* each block from its
   recorded input with gradients on, and pushes the gradient through it. The block's weights are brought in
   again by the same hooks and stay alive only while autograd needs them.

Peak accelerator memory is about one block's weights plus one block's activations, whatever the model size; the
result is identical to ordinary back-propagation (``tests/test_streaming.py``). Nothing here is specific to one
architecture: it works for the retrofit backend's decoders and for native Engram MoE models alike.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn


def _post_module(base: nn.Module) -> nn.Module:
    """The module a decoder applies after its last block (final norm / hyper-connection mixer)."""
    for name in ("norm", "final_layernorm", "ln_f", "hyper_connection_mixer"):
        m = getattr(base, name, None)
        if isinstance(m, nn.Module):
            return m
    raise TypeError(f"{type(base).__name__}: cannot find the final module after the decoder blocks")


@dataclass
class _Recorded:
    hidden: torch.Tensor
    in_kwargs: bool
    args: tuple
    kwargs: dict = field(default_factory=dict)


class StreamingTrainer:
    """Recompute-based backward pass for :class:`~wikiengram.backends.hf.EngramCausalLM` (and its native
    subclass) whose model may be dispatched with ``device_map`` / offloading."""

    def __init__(self, lm, store_on_cpu: bool = True):
        from .native import _decoder_layers_any

        self.lm = lm
        self.base = getattr(lm.model, lm.model.base_model_prefix)
        self.layers = _decoder_layers_any(lm.model)
        self.post = _post_module(self.base)
        self.first = min(lm.variant.layer_ids)
        self.store_on_cpu = store_on_cpu
        self._checked_post = False
        self._host: dict[int, torch.Tensor] = {}  # per block: CPU buffer for its recorded input, kept across steps

    def _host_copy(self, index: int, h: torch.Tensor) -> torch.Tensor:
        """``h`` copied into this block's CPU buffer. The buffers are kept between steps: a fresh allocation of
        every block's input (gigabytes per step) is paid in page faults, and under a container memory limit each
        fault must first reclaim page cache, which made allocation the slowest part of a step."""
        buf = self._host.get(index)
        if buf is None or buf.dtype != h.dtype or buf.numel() < h.numel():
            self._host[index] = buf = torch.empty(h.numel(), dtype=h.dtype)
        out = buf[: h.numel()].view(h.shape)
        out.copy_(h)
        return out

    def _record_hook(self, index: int, store: dict):
        def hook(_module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            h = hidden.detach()
            kept = self._host_copy(index, h) if self.store_on_cpu and h.device.type != "cpu" else h
            store[index] = _Recorded(kept, not args, tuple(args[1:]) if args else (), dict(kwargs))
            return None

        return hook

    def loss_and_backward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                          labels: torch.Tensor, batch, stats: dict | None = None) -> torch.Tensor:
        """Mean cross-entropy on labelled positions; gradients are accumulated into the batch's delta leaves.
        ``stats`` as in :meth:`EngramCausalLM.answer_loss`."""
        from .hf import count_confident

        lm = self.lm
        device = lm.device
        recorded: dict[int, _Recorded] = {}
        last: dict[str, torch.Tensor] = {}
        handles = [self.layers[i].register_forward_pre_hook(self._record_hook(i, recorded), with_kwargs=True,
                                                             prepend=True)
                   for i in range(self.first, len(self.layers))]

        def grab_last(_m, _a, _kw, out):
            last["h"] = (out[0] if isinstance(out, tuple) else out).detach()

        handles.append(self.layers[-1].register_forward_hook(grab_last, with_kwargs=True))
        try:
            with torch.no_grad(), lm.engram_inputs(batch):
                final = self.base(input_ids=input_ids.to(device),
                                  attention_mask=None if attention_mask is None else attention_mask.to(device),
                                  use_cache=False).last_hidden_state
        finally:
            for h in handles:
                h.remove()

        h_last = last["h"].to(device).requires_grad_(True)
        y = self.post(h_last)
        if not self._checked_post:  # make sure the "after the blocks" step is really just this module
            if not torch.equal(y.detach().to(final.device), final):
                raise RuntimeError(f"{type(self.base).__name__}: streaming cannot reproduce the final hidden state")
            self._checked_post = True
        shift = labels[:, 1:].to(device)
        sel = shift != -100
        logits = lm.logits(y[:, :-1][sel].to(lm.model.get_output_embeddings().weight.device)).float()
        token_losses = nn.functional.cross_entropy(logits, shift[sel].to(logits.device), reduction="none")
        count_confident(token_losses, stats)
        loss = token_losses.mean()
        loss.backward()
        grad = h_last.grad

        for i in range(len(self.layers) - 1, self.first - 1, -1):
            rec = recorded.pop(i)
            x = rec.hidden.to(device).requires_grad_(True)
            args, kwargs = ((), {**rec.kwargs, "hidden_states": x}) if rec.in_kwargs else ((x, *rec.args), rec.kwargs)
            with torch.enable_grad(), lm.engram_inputs(batch):
                out = self.layers[i](*args, **kwargs)
            out_h = out[0] if isinstance(out, tuple) else out
            torch.autograd.backward(out_h, grad.to(out_h.device))
            grad = x.grad
            del out, out_h, x, rec
        return loss.detach()


# ------------------------------------------------------------------------------------------- placement
_UNITS = {"": 1, "b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "tb": 10**12,
          "kib": 2**10, "mib": 2**20, "gib": 2**30, "tib": 2**40}


def parse_size(text: str) -> int:
    """``"20GiB"`` / ``"500MB"`` / ``"123"`` -> bytes."""
    s = text.strip().lower().replace(" ", "")
    num = s.rstrip("abcdefghijklmnopqrstuvwxyz")
    return int(float(num) * _UNITS[s[len(num):]])


def checkpoint_bytes(model_name_or_path: str) -> int | None:
    """Bytes the weights take once loaded in a 16-bit dtype (FP8 releases double), or None if unknown. Reads only
    local index / config files, or the Hub file listing for a model id."""
    path = Path(model_name_or_path)
    factor = 1
    if path.is_dir():
        config = path / "config.json"
        if config.exists():
            q = json.loads(config.read_text()).get("quantization_config") or {}
            factor = 2 if str(q.get("quant_method", "")).lower() == "fp8" else 1
        index = path / "model.safetensors.index.json"
        total = json.loads(index.read_text()).get("metadata", {}).get("total_size") if index.exists() else None
        total = total or sum(f.stat().st_size for f in path.glob("*.safetensors"))
        return int(total) * factor or None
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_name_or_path, files_metadata=True)
        total = sum(s.size or 0 for s in info.siblings or [] if s.rfilename.endswith(".safetensors"))
        if "fp8" in model_name_or_path.lower():
            factor = 2
        return total * factor or None
    except Exception:  # offline or unknown: treat as fitting
        return None


def max_memory(max_gpu_memory: str = "", max_cpu_memory: str = "") -> dict:
    """accelerate ``max_memory``: every visible accelerator plus host RAM (default: 85% / 80% of what is free)."""
    from ..utils.device import accelerators, device_memory, system_memory

    out: dict = {}
    for i, dev in enumerate(accelerators()):
        free, _ = device_memory(dev)
        out["mps" if dev.type == "mps" else i] = max_gpu_memory or f"{int(free * 0.85) // 2**20}MiB"
    avail, _ = system_memory()
    out["cpu"] = max_cpu_memory or f"{int(avail * 0.8) // 2**20}MiB"
    return out


def should_stream(model_name_or_path: str, placement: str = "auto", device: str | None = None,
                  max_gpu_memory: str = "") -> bool:
    """Whether to stream the frozen model (see ``BackendConfig.placement``)."""
    if placement in ("stream", "resident"):
        return placement == "stream"
    if placement != "auto":
        raise ValueError(f"placement must be auto, resident or stream, not {placement!r}")
    from ..utils.device import accelerators, device_memory, resolve_device, system_memory

    size = checkpoint_bytes(model_name_or_path)
    if size is None:
        return False
    if max_gpu_memory:
        budget = parse_size(max_gpu_memory)
    elif resolve_device(device).type == "cpu" or not accelerators():
        budget = int(system_memory()[0] * 0.85)
    else:
        budget = int(sum(device_memory(d)[0] for d in accelerators()) * 0.85)
    return size > budget


def local_checkpoint(model_name_or_path: str, cache_dir: str | Path | None = None) -> str:
    """A local directory holding the checkpoint (downloads the safetensors, configs and tokenizer if needed)."""
    if Path(model_name_or_path).is_dir():
        return str(model_name_or_path)
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name_or_path, cache_dir=str(cache_dir) if cache_dir else None,
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken",
                                             "*.jinja"])  # chat templates may ship as their own file


def plan_placement(model_name_or_path: str, max_gpu_memory: str | None = None, max_cpu_memory: str | None = None,
                   dtype: str | None = None) -> dict:
    """Where Hugging Face / accelerate would put each block on this machine, without loading any weights."""
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    from ..utils.device import resolve_device, resolve_dtype

    config = AutoConfig.from_pretrained(model_name_or_path)
    config = getattr(config, "text_config", None) or config
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    torch_dtype = resolve_dtype(dtype, resolve_device())
    max_mem = max_memory(max_gpu_memory or "", max_cpu_memory or "")
    no_split = getattr(model, "_no_split_modules", None) or []
    device_map = infer_auto_device_map(model, max_memory=max_mem, no_split_module_classes=no_split,
                                       dtype=torch_dtype)
    n_params = sum(p.numel() for p in model.parameters())
    counts: dict[str, int] = {}
    for _name, where in device_map.items():
        counts[str(where)] = counts.get(str(where), 0) + 1
    return {"model": model_name_or_path, "parameters_billion": round(n_params / 1e9, 2), "dtype": str(torch_dtype),
            "max_memory": {str(k): v for k, v in max_mem.items()}, "modules_per_place": counts,
            "offloaded": any(str(w) in ("cpu", "disk") for w in device_map.values()),
            "device_map": {k: str(v) for k, v in device_map.items()}}
