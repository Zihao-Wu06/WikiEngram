"""Run a native Engram MoE model straight from its safetensors checkpoint, on any hardware.

Qwen3.8-Flash-Next has about 180B parameters: a 125B backbone (512 routed experts per layer), a 51B n-gram table
and a 4B MTP head. Loading it with transformers needs all of that in memory; even the 128 table shards are first
concatenated into one ~95 GiB tensor on the CPU. This loader instead:

* keeps the non-expert weights (embeddings, attention, shared experts, routers, norms, hyper-connections and the
  PLE fusion layer; about 2.2B parameters for Qwen3.8) resident on the compute device, dequantized from FP8;
* reads routed experts from the checkpoint on demand: a forward pass loads only the experts its tokens are routed
  to (10 of 512 per token for Qwen3.8), dequantizes them on the device and drops them afterwards (an optional
  LRU cache keeps the most used ones);
* never materializes the n-gram table: WikiEngram reads only the rows it needs from the shards
  (:class:`~wikiengram.engram.tables.ShardedTable`).

The experts' arithmetic is the same as transformers' own ``*Experts`` modules (same ops, same order), so with the
same weights the results are identical (``tests/test_checkpoint_streaming.py``). Combined with the streaming
backward (:mod:`.streaming`), training needs the resident weights plus one layer's experts and activations.

FP8 checkpoints are dequantized to the compute dtype (bf16, fp16 or fp32), so no FP8 kernels or recent GPUs are
needed: any CUDA / ROCm / Intel / Apple GPU or a CPU can run it.
"""

from __future__ import annotations

import json
import mmap
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..engram.tables import read_safetensors_header
from ..utils.device import resolve_device, resolve_dtype

_FP8 = tuple(getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, n))
_TORCH_DTYPES = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
                 "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8}
_CHUNK = 32  # experts dequantized per kernel batch (bounds the float32 temporaries)
_ALIGN = 4096  # O_DIRECT alignment of offsets, sizes and buffers
_DIRECT_IO = hasattr(os, "O_DIRECT") and os.environ.get("WIKIENGRAM_DIRECT_IO", "1") != "0"
_SCRATCH = threading.local()


def _scratch(size: int) -> mmap.mmap:
    """A page-aligned buffer of at least ``size`` bytes, one per thread (anonymous memory maps are page-aligned)."""
    buf = getattr(_SCRATCH, "buf", None)
    if buf is None or len(buf) < size:
        _SCRATCH.buf = buf = mmap.mmap(-1, max(size, 4 << 20))
    return buf


def _fp8_device(device: torch.device) -> torch.device:
    """Where FP8 -> float conversion runs (Apple MPS has no float8 dtypes)."""
    return device if device.type in ("cuda", "xpu", "cpu") else torch.device("cpu")


def dequant_block_fp8(weight: torch.Tensor, scale_inv: torch.Tensor, device: torch.device,
                      block: tuple[int, int] = (128, 128)) -> torch.Tensor:
    """``weight [out, in]`` FP8 times ``scale_inv [ceil(out/bo), ceil(in/bi)]`` per block -> float32 on device."""
    dev = _fp8_device(device)
    w = weight.to(dev)
    s = scale_inv.to(dev).float()
    out_f, in_f = w.shape
    s = s.repeat_interleave(block[0], 0)[:out_f].repeat_interleave(block[1], 1)[:, :in_f]
    return w.float() * s


class SafetensorsCheckpoint:
    """Lazy access to the tensors of a (sharded) safetensors checkpoint directory."""

    def __init__(self, path: str | Path, block: tuple[int, int] = (128, 128)):
        self.root = Path(path)
        index = self.root / "model.safetensors.index.json"
        if index.exists():
            self.weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
        else:
            self.weight_map = {k: f.name for f in sorted(self.root.glob("*.safetensors"))
                               for k in read_safetensors_header(f)[0] if k != "__metadata__"}
        if not self.weight_map:
            raise FileNotFoundError(f"no safetensors weights in {self.root}")
        self.block = tuple(block)
        self._handles: dict[str, object] = {}
        self._headers: dict[str, tuple[dict, int]] = {}
        self._maps: dict[str, np.memmap] = {}
        self._fds: dict[str, int] = {}
        self._direct_fds: dict[str, int] = {}
        self._info: dict[str, tuple] = {}  # tensor_info results (asked for every expert tensor read)
        self._direct: bool | None = None  # None: not tried yet; False: the file system refused O_DIRECT
        self._lock = threading.Lock()

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def tensor_info(self, key: str) -> tuple[str, tuple[int, ...], int, int]:
        """``(dtype, shape, start, end)`` of a tensor; ``start``/``end`` are absolute byte offsets in its file."""
        info = self._info.get(key)
        if info is None:
            info = self._info[key] = self._tensor_info(key)
        return info

    def _tensor_info(self, key: str) -> tuple[str, tuple[int, ...], int, int]:
        fname = self.weight_map[key]
        if fname not in self._headers:
            with self._lock:
                if fname not in self._headers:
                    self._headers[fname] = read_safetensors_header(self.root / fname)
        header, data_start = self._headers[fname]
        meta = header[key]
        return (meta["dtype"], tuple(int(d) for d in meta["shape"]), data_start + int(meta["data_offsets"][0]),
                data_start + int(meta["data_offsets"][1]))

    def bytes_view(self, key: str) -> np.ndarray:
        """The raw bytes of a tensor as a read-only memory map (no copy, no RAM beyond the page cache)."""
        fname = self.weight_map[key]
        if fname not in self._maps:
            with self._lock:
                if fname not in self._maps:
                    self._maps[fname] = np.memmap(self.root / fname, dtype=np.uint8, mode="r")
        _, _, start, end = self.tensor_info(key)
        return self._maps[fname][start:end]

    def small(self, key: str) -> torch.Tensor:
        """A small tensor (e.g. a block scale) read through the memory map."""
        dtype, shape, _, _ = self.tensor_info(key)
        raw = torch.from_numpy(np.array(self.bytes_view(key)))
        return raw.view(_TORCH_DTYPES[dtype]).reshape(shape)

    def read_into(self, key: str, out: np.ndarray) -> None:
        """Copy a tensor's raw bytes into the contiguous uint8 buffer ``out``. Uses positioned reads where the OS
        has them: one system call per tensor instead of a page fault per 4 KB of a memory map (about 2x faster
        from the page cache), and it releases the GIL so threads read in parallel.

        On Linux the read bypasses the page cache (``O_DIRECT``; ``WIKIENGRAM_DIRECT_IO=0`` turns it off). Experts
        are read in full sweeps far larger than RAM, so caching them only evicts other pages; under a container
        memory limit every cached read also has to reclaim memory first. Measured on an RTX 5090 host (NVMe
        RAID, 92 GB container limit): about 1 GB/s through the page cache, 13-18 GB/s direct."""
        _, _, start, end = self.tensor_info(key)
        n = end - start
        if out.size != n:
            raise ValueError(f"{key}: buffer of {out.size} bytes for a {n}-byte tensor")
        if not hasattr(os, "preadv"):
            np.copyto(out, self.bytes_view(key))
            return
        fname = self.weight_map[key]
        if self._direct is not False and _DIRECT_IO:
            try:
                self._read_direct(fname, start, n, out)
                return
            except OSError:  # a file system without O_DIRECT (e.g. tmpfs): read through the page cache
                self._direct = False
        fd = self._fd(fname, os.O_RDONLY, self._fds)
        view, done = memoryview(out), 0
        while done < n:
            got = os.preadv(fd, [view[done:]], start + done)
            if got <= 0:
                raise OSError(f"{key}: short read from {fname}")
            done += got

    def _fd(self, fname: str, flags: int, cache: dict[str, int]) -> int:
        fd = cache.get(fname)
        if fd is None:
            with self._lock:
                fd = cache.get(fname)
                if fd is None:
                    fd = cache[fname] = os.open(self.root / fname, flags)
        return fd

    def _read_direct(self, fname: str, start: int, n: int, out: np.ndarray) -> None:
        """O_DIRECT needs aligned offsets, sizes and buffers: read the aligned range around the tensor into a
        page-aligned per-thread buffer, then copy the tensor's bytes out."""
        fd = self._fd(fname, os.O_RDONLY | os.O_DIRECT, self._direct_fds)
        a0 = start - start % _ALIGN
        size = -(-(start + n - a0) // _ALIGN) * _ALIGN
        buf = _scratch(size)
        view, done = memoryview(buf)[:size], 0
        while done < start + n - a0:
            got = os.preadv(fd, [view[done:]], a0 + done)
            if got <= 0 or (got % _ALIGN and done + got < start + n - a0):
                raise OSError(f"{fname}: short direct read")
            done += got
        np.copyto(out, np.frombuffer(buf, dtype=np.uint8, count=n, offset=start - a0))

    def raw(self, key: str) -> torch.Tensor:
        """The tensor as stored (FP8 stays FP8), on the CPU."""
        from safetensors import safe_open

        fname = self.weight_map[key]
        handle = self._handles.get(fname)
        if handle is None:
            with self._lock:
                handle = self._handles.get(fname)
                if handle is None:
                    handle = self._handles[fname] = safe_open(str(self.root / fname), framework="pt", device="cpu")
        return handle.get_tensor(key)

    def load(self, key: str, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
        """The tensor on ``device``; FP8 weights with block scales (``<key>_scale_inv``) are dequantized there."""
        t = self.raw(key)
        if t.dtype in _FP8:
            if key + "_scale_inv" in self:
                t = dequant_block_fp8(t, self.raw(key + "_scale_inv"), device, self.block)
            elif key.endswith(".weight") and key[:-len(".weight")] + ".weight_scale" in self:
                t = t.to(_fp8_device(device)).float() * self.raw(key[:-len(".weight")] + ".weight_scale").float().to(
                    _fp8_device(device))
            else:
                t = t.to(_fp8_device(device)).float()
        return t.to(device=device, dtype=dtype if dtype is not None and t.is_floating_point() else t.dtype)


class ExpertCache:
    """Experts kept on the accelerator as stored (FP8 bytes plus block scales: half the size of bf16).

    Entries are admitted until the budget is full and never evicted. Streamed layers are visited in the same
    order on every pass, and LRU would evict each expert just before its reuse (measured: zero hits). A fixed
    resident set saves the reads of the same experts on every pass instead."""

    def __init__(self, budget_bytes: int):
        self.budget = int(budget_bytes)
        self.used = 0
        self.items: dict = {}
        self.hits = self.misses = 0

    def get(self, key):
        value = self.items.get(key)
        if value is not None:
            self.hits += 1
        else:
            self.misses += 1
        return value

    def offer(self, key, raw: torch.Tensor, scales: list) -> None:
        size = raw.numel() * raw.element_size() + sum(s.numel() * s.element_size() for s in scales if s is not None)
        if key not in self.items and self.used + size <= self.budget:
            self.items[key] = (raw.clone(), scales)
            self.used += size


def _io_threads() -> int:
    """Threads reading experts (``WIKIENGRAM_IO_THREADS`` overrides; default: 48 with direct reads, else up to 16
    of the available CPUs)."""
    from ..utils.device import available_cpus

    env = os.environ.get("WIKIENGRAM_IO_THREADS")
    if env:
        return max(1, int(env))
    # direct reads wait on the drive, not the CPU: NVMe needs a deep queue (measured on an RTX 5090 host: 7 GB/s
    # with 32 readers, 18 GB/s with 48)
    return 48 if _DIRECT_IO else max(4, min(16, available_cpus()))


def _is_fused_experts(module: nn.Module) -> bool:
    """transformers' MoE expert collections: ``gate_up_proj [E, 2I, H]`` and ``down_proj [E, H, I]``."""
    gu, down = getattr(module, "gate_up_proj", None), getattr(module, "down_proj", None)
    return (isinstance(gu, nn.Parameter) and isinstance(down, nn.Parameter) and gu.dim() == 3 and down.dim() == 3
            and hasattr(module, "num_experts"))


def _eager_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """transformers' reference (eager) experts loop."""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
    return final_hidden_states


class _CompactExperts(nn.Module):
    """The experts one call routes to, re-indexed ``0..n-1``, laid out like the original module so that
    transformers' expert kernels (eager / batched_mm / grouped_mm) run on it unchanged."""

    def __init__(self, owner: StreamedExperts, gate_up: torch.Tensor, down: torch.Tensor):
        super().__init__()
        self.num_experts = gate_up.shape[0]
        self.gate_up_proj, self.down_proj = gate_up, down
        for attr, value in owner.layout.items():
            setattr(self, attr, value)
        self.act_fn = owner.act_fn
        self._is_expert_parallel = False
        object.__setattr__(self, "_gate_fn", owner.gate_fn)

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        if self._gate_fn is not None:
            return self._gate_fn(self, gate_up_out)
        gate, up = gate_up_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up


class StreamedExperts(nn.Module):
    """Drop-in for a transformers ``*Experts`` module whose weights stay in the checkpoint.

    ``forward(hidden_states, top_k_index, top_k_weights)`` loads only the experts that ``top_k_index`` selects and
    runs the original module's own expert implementation on them (same kernel, same accumulation order per token),
    so the result is what the original module computes. ``prefix`` is the checkpoint key of the expert list, e.g.
    ``model.language_model.layers.7.mlp.experts`` with tensors ``.{e}.gate_proj.weight`` etc."""

    def __init__(self, original: nn.Module, ckpt: SafetensorsCheckpoint, prefix: str, device: torch.device,
                 dtype: torch.dtype, cache: ExpertCache | None = None):
        super().__init__()
        self.num_experts = int(original.num_experts)
        self.act_fn = original.act_fn
        self.ckpt, self.prefix, self.device, self.dtype, self.cache = ckpt, prefix, device, dtype, cache
        self._scales: dict[tuple[int, int], torch.Tensor] = {}  # (expert, projection) -> block scales
        if getattr(original, "is_transposed", False) or not getattr(original, "is_concatenated", True):
            raise NotImplementedError(f"{type(original).__name__}: only [gate; up] row layouts are supported")
        self.layout = {a: getattr(original, a) for a in ("has_gate", "has_bias", "is_transposed", "is_concatenated",
                                                          "has_post_expert_norm", "config") if hasattr(original, a)}
        self.gate_fn = getattr(type(original), "_apply_gate", None)
        self.impl = getattr(getattr(original, "config", None), "_experts_implementation", None) or "eager"
        self.gate_up_shape = tuple(original.gate_up_proj.shape[1:])  # [2I, H] (meta shapes, nothing allocated)
        self.down_shape = tuple(original.down_proj.shape[1:])  # [H, I]
        self.fused = f"{prefix}.gate_up_proj" in ckpt
        if not self.fused and f"{prefix}.0.gate_proj.weight" not in ckpt:
            raise KeyError(f"no expert weights under {prefix} in the checkpoint")

    _staging: torch.Tensor | None = None  # shared host buffer (pinned when a CUDA device is present)
    _pool_instance: ThreadPoolExecutor | None = None

    @classmethod
    def _staging_buffer(cls, nbytes: int, pin: bool) -> torch.Tensor:
        buf = cls._staging
        if buf is None or buf.numel() < nbytes or buf.is_pinned() != pin:
            cls._staging = buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
        return buf[:nbytes]

    @classmethod
    def _io_pool(cls) -> ThreadPoolExecutor:
        if cls._pool_instance is None:
            cls._pool_instance = ThreadPoolExecutor(max_workers=_io_threads())
        return cls._pool_instance

    def _load_batch(self, items: list[tuple[int, int]], gate_up: torch.Tensor, down: torch.Tensor) -> None:
        """Experts ``items = [(slot, expert)]``: cached ones come from the accelerator cache; the others are read
        from the files into one host buffer (parallel positioned reads) and moved to the device in one transfer.
        All are then dequantized in chunks into ``gate_up[slot]`` / ``down[slot]``, with the same arithmetic as
        :func:`dequant_block_fp8`, expert by expert."""
        ck, inter, hidden = self.ckpt, self.gate_up_shape[0] // 2, self.gate_up_shape[1]
        shapes = ((inter, hidden), (inter, hidden), (hidden, inter))
        keys = [[f"{self.prefix}.{e}.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")] for _, e in items]
        dtype = ck.tensor_info(keys[0][0])[0]
        if dtype not in ("F8_E4M3", "BF16", "F16", "F32"):
            raise TypeError(f"{keys[0][0]}: unsupported expert dtype {dtype}")
        per = inter * hidden * torch.empty((), dtype=_TORCH_DTYPES[dtype]).element_size()
        n = len(items)
        work = self.device if dtype != "F8_E4M3" else _fp8_device(self.device)
        scaled = keys[0][0] + "_scale_inv" in ck
        missing, hits, cached = [], [], {}
        for k, (_, e) in enumerate(items):
            hit = self.cache.get((self.prefix, e)) if self.cache is not None else None
            if hit is None:
                missing.append(k)
            else:
                hits.append(k)
                cached[k] = hit
        order = missing + hits  # position p holds items[order[p]]: fresh reads first, then cache hits
        nm = len(missing)
        scales: list[list[torch.Tensor | None]] = [[None] * 3 for _ in range(n)]
        if missing:
            host = self._staging_buffer(nm * 3 * per, self.device.type == "cuda")
            view = host.numpy().reshape(nm, 3, per)

            def fill(q: int) -> None:
                k = missing[q]
                for m, key in enumerate(keys[k]):
                    if ck.tensor_info(key)[0] != dtype:
                        raise ValueError(f"{key}: expected {dtype}")
                    ck.read_into(key, view[q, m])
                    if scaled:
                        scales[k][m] = self._scale(items[k][1], m, key, work)

            list(self._io_pool().map(fill, range(nm)))
            fresh = host.view(nm, 3, per)
        for k in hits:
            scales[k] = list(cached[k][1])
        # the raw bytes go to the device one chunk at a time: a layer's experts never sit there twice (raw and
        # dequantized), which is what bounds the batch size on a 32 GB card
        raw = torch.empty((min(n, _CHUNK), 3, per), dtype=torch.uint8, device=work)
        bo, bi = ck.block
        for c0 in range(0, n, _CHUNK):
            c1 = min(n, c0 + _CHUNK)
            ks = order[c0:c1]
            chunk = raw[: c1 - c0]
            if c0 < nm:
                chunk[: min(c1, nm) - c0].copy_(fresh[c0 : min(c1, nm)])
            for p in range(max(c0, nm), c1):
                chunk[p - c0].copy_(cached[order[p]][0])
            if self.cache is not None:
                for p in range(c0, min(c1, nm)):
                    k = order[p]
                    self.cache.offer((self.prefix, items[k][1]), chunk[p - c0], scales[k])
            slots = torch.tensor([items[k][0] for k in ks], device=self.device)
            for m, (rows, cols) in enumerate(shapes):
                w = chunk[:, m].view(_TORCH_DTYPES[dtype]).view(c1 - c0, rows, cols)
                if dtype == "F8_E4M3":
                    w = w.float()
                    if scaled:
                        s = torch.stack([scales[k][m] for k in ks]).to(work).float()
                        w = w * s.repeat_interleave(bo, 1)[:, :rows].repeat_interleave(bi, 2)[:, :, :cols]
                w = w.to(device=self.device, dtype=self.dtype)
                if m == 0:
                    gate_up[slots, :inter] = w
                elif m == 1:
                    gate_up[slots, inter:] = w
                else:
                    down[slots] = w

    def _scale(self, expert: int, m: int, key: str, work: torch.device) -> torch.Tensor:
        """An expert tensor's block scales on the compute device, read once (about 200 bytes each; reading them
        for every expert of every layer at every step took about 17 s per training step)."""
        s = self._scales.get((expert, m))
        if s is None:
            s = self._scales[(expert, m)] = self.ckpt.small(key + "_scale_inv").to(work)
        return s

    def experts(self, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """``gate_up [n, 2I, H]`` and ``down [n, H, I]`` of experts ``ids`` on the compute device, written straight
        into preallocated stacks (no second copy of a layer's experts)."""
        gate_up = torch.empty((len(ids), *self.gate_up_shape), device=self.device, dtype=self.dtype)
        down = torch.empty((len(ids), *self.down_shape), device=self.device, dtype=self.dtype)
        if self.fused:  # checkpoint already stores [E, ...] tensors (rare for releases)
            gu_all = self.ckpt.load(f"{self.prefix}.gate_up_proj", self.device, self.dtype)
            down_all = self.ckpt.load(f"{self.prefix}.down_proj", self.device, self.dtype)
            for j, e in enumerate(ids):
                gate_up[j].copy_(gu_all[e])
                down[j].copy_(down_all[e])
            return gate_up, down
        self._load_batch(list(enumerate(ids)), gate_up, down)  # the fused layout: gate rows, then up rows
        return gate_up, down

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        hit = [e for e in torch.unique(top_k_index).tolist() if 0 <= e < self.num_experts]
        if not hit:
            return torch.zeros_like(hidden_states)
        gate_up, down = self.experts(hit)
        # order-preserving re-indexing: each token's experts keep their relative order, so per-token sums match
        remap = torch.full((self.num_experts + 1,), len(hit), dtype=torch.long, device=top_k_index.device)
        remap[torch.tensor(hit, device=top_k_index.device)] = torch.arange(len(hit), device=top_k_index.device)
        compact = _CompactExperts(self, gate_up, down)
        fn = _eager_experts_forward
        if self.impl != "eager":
            try:
                from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

                fn = ALL_EXPERTS_FUNCTIONS.get_interface(self.impl, _eager_experts_forward)
            except ImportError:  # older transformers: eager only
                pass
        return fn(compact, hidden_states, remap[top_k_index], top_k_weights)


_DTYPE_BYTES = {"F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I32": 4}


def checkpoint_summary(path: str | Path, compute_bytes: int = 2) -> dict:
    """What streaming a native checkpoint needs, from the safetensors headers only (no weights are read):
    parameters by kind and the bytes resident on the accelerator once dequantized to a 16-bit dtype."""
    ckpt = SafetensorsCheckpoint(path)
    counts = {"resident": 0, "experts": 0, "ngram_table": 0, "skipped": 0}
    stored = dict.fromkeys(counts, 0)
    per_layer_experts: dict[str, int] = {}
    for fname in sorted(set(ckpt.weight_map.values())):
        header, _ = read_safetensors_header(ckpt.root / fname)
        for key, meta in header.items():
            if key == "__metadata__" or key.endswith(("_scale_inv", ".weight_scale")):
                continue
            n = 1
            for d in meta["shape"]:
                n *= int(d)
            if key.startswith(("mtp.", "model.visual.")) or ".visual." in key:
                kind = "skipped"  # not part of the text model
            elif ".ngram_embedding." in key:
                kind = "ngram_table"
            elif ".mlp.experts." in key:
                kind = "experts"
                layer = key.split(".mlp.experts.")[0]
                per_layer_experts[layer] = per_layer_experts.get(layer, 0) + n * _DTYPE_BYTES.get(meta["dtype"], 2)
            else:
                kind = "resident"
            counts[kind] += n
            stored[kind] += n * _DTYPE_BYTES.get(meta["dtype"], 2)
    layer_bytes = max(per_layer_experts.values(), default=0)
    return {"parameters_billion": {k: round(v / 1e9, 2) for k, v in counts.items()},
            "stored_gib": {k: round(v / 2**30, 1) for k, v in stored.items()},
            "resident_gib": round(counts["resident"] * compute_bytes / 2**30, 1),
            "largest_layer_experts_stored_gib": round(layer_bytes / 2**30, 2),
            "largest_layer_experts_gib": round(layer_bytes / max(1, stored["experts"]) * counts["experts"]
                                               * compute_bytes / 2**30, 2) if counts["experts"] else 0.0}


def _quant_block(config) -> tuple[int, int]:
    q = getattr(config, "quantization_config", None) or {}
    if not isinstance(q, dict):
        q = q.to_dict()
    return tuple(q.get("weight_block_size") or (128, 128))


def load_streamed_model(path: str | Path, device=None, dtype=None, expert_cache_gb: float = 0.0):
    """Build the text model of a native Engram MoE checkpoint with resident non-expert weights and streamed
    experts. Returns ``(model, checkpoint, prefix)``; the n-gram table parameter stays on the meta device."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, AutoModelForCausalLM

    dev = resolve_device(device)
    dt = resolve_dtype(dtype, dev)
    config = AutoConfig.from_pretrained(path)
    ckpt = SafetensorsCheckpoint(path, block=_quant_block(config))
    text = getattr(config, "text_config", None) or config
    with init_empty_weights(include_buffers=False):
        try:
            model = AutoModelForCausalLM.from_config(text, dtype=dt)
        except TypeError:
            model = AutoModelForCausalLM.from_config(text, torch_dtype=dt)
    # VLM releases store the text model as model.language_model.*; text-only ones as model.*
    prefix = "model.language_model." if "model.language_model.embed_tokens.weight" in ckpt else "model."

    def key(name: str) -> str:
        return prefix + name[len("model."):] if name.startswith("model.") else name

    experts = {n: m for n, m in model.named_modules() if _is_fused_experts(m)}
    skip = {f"{n}.{p}" for n in experts for p in ("gate_up_proj", "down_proj")}
    table_params = {n for n, _ in model.named_parameters() if n.endswith("ngram_embedding.weight")}
    tie = bool(getattr(text, "tie_word_embeddings", False))
    for name, p in model.named_parameters():
        if name in skip or name in table_params:
            continue
        k = key(name)
        if k not in ckpt:
            if tie and name.endswith("lm_head.weight"):
                continue
            raise KeyError(f"checkpoint has no tensor {k} for parameter {name}")
        t = ckpt.load(k, dev, p.dtype)
        if tuple(t.shape) != tuple(p.shape):
            raise ValueError(f"{k}: checkpoint shape {tuple(t.shape)} != model shape {tuple(p.shape)}")
        set_module_tensor_to_device(model, name, dev, value=t)
    for name, b in model.named_buffers():
        k = key(name)
        if k in ckpt and b.device.type != "meta":  # computed at init; must agree with the checkpoint
            stored = ckpt.raw(k)
            if stored.shape == b.shape and not torch.equal(stored.to(b.dtype), b.cpu()):
                raise ValueError(f"buffer {name} differs from the checkpoint")
        set_module_tensor_to_device(model, name, dev, value=b)
    if tie:
        model.tie_weights()
    if expert_cache_gb < 0:  # auto: what the accelerator has left after the resident weights, minus working room
        from ..utils.device import device_memory

        free, total = device_memory(dev)
        expert_cache_gb = max(0.0, (free - max(8 * 2**30, 0.3 * total)) / 2**30)
    cache = ExpertCache(int(expert_cache_gb * 2**30)) if expert_cache_gb else None
    for name, module in experts.items():
        parent_name, _, attr = name.rpartition(".")
        setattr(model.get_submodule(parent_name), attr, StreamedExperts(module, ckpt, key(name), dev, dt, cache))
    model.eval()
    model.streamed_expert_cache = cache
    return model, ckpt, prefix
