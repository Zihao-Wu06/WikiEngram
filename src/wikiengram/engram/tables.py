"""Frozen base Engram tables.

* :class:`ZeroTable`  - an empty table (no storage); useful for tests and for a fresh attachment.
* :class:`DenseTable` - an in-memory table, e.g. the base Engram trained once by ``wikiengram adapt``.
* :class:`FP8RowTable` - DeepSeek-V4.1-Flash's native table read straight from its safetensors shard:
  ``engram.embed.weight`` (F8_E4M3, ``[rows, 256]``) with ``engram.embed.scale`` (F8_E8M0, one scale per 32
  values). Only the requested rows are read (memory-mapped), so a 98 GB table costs no RAM.
* :class:`ShardedTable` - a table split into row shards across the files of a checkpoint (Qwen3.8-Flash-Next:
  128 shards ``ngram_embedding.shard_{i}.weight``), BF16 or F8_E4M3 with one global scale. Also read lazily.

All tables return float32 rows on the CPU from ``gather``; callers move them to the compute device.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file


class BaseTable:
    num_rows: int
    dim: int

    def gather(self, rows: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class ZeroTable(BaseTable):
    def __init__(self, num_rows: int, dim: int):
        self.num_rows, self.dim = int(num_rows), int(dim)

    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.zeros(rows.numel(), self.dim)


class DenseTable(BaseTable):
    """A dense ``[num_rows, dim]`` table. ``weight`` may live on an accelerator during base adaptation."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight
        self.num_rows, self.dim = weight.shape

    @classmethod
    def random(cls, num_rows: int, dim: int, std: float = 1.0, seed: int = 0, device="cpu") -> DenseTable:
        g = torch.Generator().manual_seed(seed)
        return cls((torch.randn(num_rows, dim, generator=g) * std).to(device))

    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        return self.weight[rows.to(self.weight.device)].float().cpu()

    def save(self, path: str | Path, key: str = "weight") -> None:
        save_file({key: self.weight.detach().cpu().contiguous()}, str(path))

    @classmethod
    def load(cls, path: str | Path, key: str = "weight", device="cpu") -> DenseTable:
        return cls(load_file(str(path), device=str(device))[key])


# ---------------------------------------------------------------------------------------------- fp8


def _e4m3fn_lut() -> np.ndarray:
    b = np.arange(256, dtype=np.int64)
    sign = np.where(b >> 7, -1.0, 1.0)
    exp = (b >> 3) & 0xF
    man = b & 0x7
    normal = (1.0 + man / 8.0) * np.exp2(exp.astype(np.float64) - 7.0)
    subnormal = (man / 8.0) * 2.0**-6
    val = sign * np.where(exp == 0, subnormal, normal)
    val[(exp == 15) & (man == 7)] = np.nan  # e4m3fn has no infinities, only NaN
    return val.astype(np.float32)


def _e8m0_lut() -> np.ndarray:
    b = np.arange(256, dtype=np.float64)
    val = np.exp2(b - 127.0)
    val[255] = np.nan
    return val.astype(np.float32)


E4M3FN_LUT = _e4m3fn_lut()
E8M0_LUT = _e8m0_lut()


def dequant_fp8_rows(codes: np.ndarray, scales: np.ndarray, block: int = 32) -> np.ndarray:
    """``codes [n, dim]`` uint8 (e4m3fn) and ``scales [n, dim // block]`` uint8 (e8m0) -> float32."""
    values = E4M3FN_LUT[codes].reshape(codes.shape[0], -1, block)
    return (values * E8M0_LUT[scales][..., None]).reshape(codes.shape[0], -1)


def read_safetensors_header(path: str | Path) -> tuple[dict, int]:
    """Return the header dict and the absolute byte offset where tensor data starts."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


class FP8RowTable(BaseTable):
    """Rows of a DeepSeek-V4.1 Engram table, read lazily from a safetensors file."""

    def __init__(self, path: str | Path, weight_key: str, scale_key: str, block: int = 32):
        header, data_start = read_safetensors_header(path)
        w, s = header[weight_key], header[scale_key]
        if w["dtype"] != "F8_E4M3" or s["dtype"] != "F8_E8M0":
            raise ValueError(f"expected F8_E4M3/F8_E8M0, got {w['dtype']}/{s['dtype']}")
        self.num_rows, self.dim = (int(v) for v in w["shape"])
        self.block = block
        if int(s["shape"][1]) * block != self.dim:
            raise ValueError("scale shape does not match the weight shape and block size")
        self._w = np.memmap(path, dtype=np.uint8, mode="r", offset=data_start + w["data_offsets"][0],
                            shape=(self.num_rows, self.dim))
        self._s = np.memmap(path, dtype=np.uint8, mode="r", offset=data_start + s["data_offsets"][0],
                            shape=(self.num_rows, self.dim // block))

    @classmethod
    def from_deepseek_v41(cls, shard_path: str | Path, layer_id: int) -> FP8RowTable:
        prefix = f"layers.{layer_id}.engram.embed"
        return cls(shard_path, f"{prefix}.weight", f"{prefix}.scale")

    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        idx = rows.detach().cpu().numpy().astype(np.int64)
        order = np.argsort(idx)  # sorted reads are far friendlier to the page cache
        codes = np.empty((idx.size, self.dim), dtype=np.uint8)
        scales = np.empty((idx.size, self.dim // self.block), dtype=np.uint8)
        codes[order] = self._w[idx[order]]
        scales[order] = self._s[idx[order]]
        return torch.from_numpy(dequant_fp8_rows(codes, scales, self.block))


_RAW_DTYPES = {"F8_E4M3": np.uint8, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}


def _decode_rows(raw: np.ndarray, dtype: str, scale: float) -> np.ndarray:
    if dtype == "F8_E4M3":
        return E4M3FN_LUT[raw] * np.float32(scale)  # exact in float32 (4-bit x 8-bit mantissas)
    if dtype == "BF16":
        return (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32)


class ShardedTable(BaseTable):
    """Rows of a table stored as equal row shards in (possibly several) safetensors files.

    ``shards`` lists ``(file, tensor_key)`` in row order. The values are BF16/F16/F32, or F8_E4M3 multiplied by
    one global ``scale`` (Qwen3.8-Flash-Next-FP8 stores ``ngram_embedding.weight_scale``). Only the requested
    rows are read through memory maps, so the 51-102 GB table never needs RAM."""

    def __init__(self, shards: list[tuple[str | Path, str]], scale: float = 1.0):
        headers: dict[str, tuple[dict, int]] = {}
        self.shards = []
        for path, key in shards:
            path = str(path)
            if path not in headers:
                headers[path] = read_safetensors_header(path)
            header, start = headers[path]
            meta = header[key]
            if meta["dtype"] not in _RAW_DTYPES or len(meta["shape"]) != 2:
                raise ValueError(f"{key}: unsupported table shard {meta['dtype']} {meta['shape']}")
            rows, dim = (int(v) for v in meta["shape"])
            self.shards.append((path, start + int(meta["data_offsets"][0]), rows, dim, meta["dtype"]))
        if not self.shards or len({s[3] for s in self.shards}) != 1:
            raise ValueError("table shards must exist and share one row width")
        self.dim = self.shards[0][3]
        self.rows_per_shard = self.shards[0][2]
        if any(s[2] != self.rows_per_shard for s in self.shards[:-1]) or self.shards[-1][2] > self.rows_per_shard:
            raise ValueError("table shards must have equal row counts (the last one may be shorter)")
        self.num_rows = sum(s[2] for s in self.shards)
        self.scale = float(scale)
        self._maps: list[np.memmap | None] = [None] * len(self.shards)

    @classmethod
    def from_checkpoint(cls, root: str | Path, weight_map: dict[str, str], prefix: str) -> ShardedTable:
        """``prefix`` = the table's key without suffix, e.g. ``model.language_model.layers.1.ple.ple_embedding
        .ngram_embedding``. Accepts ``prefix.shard_{i}.weight`` shards or a single ``prefix.weight``."""
        root = Path(root)
        pat = prefix + ".shard_"
        keys = sorted((k for k in weight_map if k.startswith(pat) and k.endswith(".weight")),
                      key=lambda k: int(k[len(pat):-len(".weight")]))
        if not keys:
            keys = [prefix + ".weight"]
        scale = 1.0
        if prefix + ".weight_scale" in weight_map:
            path = root / weight_map[prefix + ".weight_scale"]
            header, start = read_safetensors_header(path)
            meta = header[prefix + ".weight_scale"]
            raw = np.fromfile(path, dtype=np.uint8, count=meta["data_offsets"][1] - meta["data_offsets"][0],
                              offset=start + meta["data_offsets"][0])
            scale = float(_decode_rows(raw.view(_RAW_DTYPES[meta["dtype"]])[:1].reshape(1, 1), meta["dtype"], 1.0)[0, 0])
        return cls([(root / weight_map[k], k) for k in keys], scale)

    def _map(self, i: int) -> np.memmap:
        if self._maps[i] is None:
            path, offset, rows, dim, dtype = self.shards[i]
            self._maps[i] = np.memmap(path, dtype=_RAW_DTYPES[dtype], mode="r", offset=offset, shape=(rows, dim))
        return self._maps[i]

    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        idx = rows.detach().cpu().numpy().astype(np.int64).reshape(-1)
        if idx.size and (idx.min() < 0 or idx.max() >= self.num_rows):
            raise IndexError(f"row out of range for a table of {self.num_rows} rows")
        out = np.empty((idx.size, self.dim), dtype=np.float32)
        shard = idx // self.rows_per_shard
        local = idx - shard * self.rows_per_shard
        for s in np.unique(shard):
            sel = np.nonzero(shard == s)[0]
            sel = sel[np.argsort(local[sel])]  # sorted reads are friendlier to the page cache
            out[sel] = _decode_rows(self._map(int(s))[local[sel]], self.shards[s][4], self.scale)
        return torch.from_numpy(out)

    def checksum(self) -> str:
        """Identity of the table: shard layout, scale and a fixed sample of rows from every shard."""
        h = hashlib.blake2b(digest_size=12)
        h.update(repr([(s[2], s[3], s[4]) for s in self.shards] + [self.scale]).encode())
        for i, (_, _, rows, _, _) in enumerate(self.shards):
            sample = np.unique(np.linspace(0, rows - 1, num=min(rows, 64)).astype(np.int64))
            h.update(np.ascontiguousarray(self._map(i)[sample]).tobytes())
        return h.hexdigest()
