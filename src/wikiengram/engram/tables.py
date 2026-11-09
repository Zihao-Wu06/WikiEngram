"""Frozen base Engram tables.

* :class:`ZeroTable`  - an empty table (no storage); useful for tests and for a fresh attachment.
* :class:`DenseTable` - an in-memory table, e.g. the base Engram trained once by ``wikiengram adapt``.
* :class:`FP8RowTable` - DeepSeek-V4.1-Flash's native table read straight from its safetensors shard:
  ``engram.embed.weight`` (F8_E4M3, ``[rows, 256]``) with ``engram.embed.scale`` (F8_E8M0, one scale per 32
  values). Only the requested rows are read (memory-mapped), so a 98 GB table costs no RAM.

All tables return float32 rows on the CPU from ``gather``; callers move them to the compute device.
"""

from __future__ import annotations

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
