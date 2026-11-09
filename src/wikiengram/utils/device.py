"""Device, dtype and memory helpers (CUDA, Apple MPS, CPU)."""

from __future__ import annotations

import os

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None and str(device) != "auto":
        return torch.device(device)
    env = os.environ.get("WIKIENGRAM_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in (None, "auto"):
        return torch.float32 if device.type == "cpu" else torch.bfloat16
    return {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}[dtype]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes(device: torch.device) -> int:
    """Peak allocated accelerator memory. On MPS PyTorch exposes no peak counter, so this reports the
    driver's current allocation (an upper bound of live tensors, sampled by the caller after each step)."""
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if os.uname().sysname == "Darwin" else 1024)
    except Exception:  # pragma: no cover
        return 0


class PeakMemoryMeter:
    """Tracks peak accelerator memory across a block (polls on MPS, uses the native counter on CUDA)."""

    def __init__(self, device: torch.device):
        self.device = device
        self.peak = 0

    def __enter__(self):
        reset_peak_memory(self.device)
        self.peak = peak_memory_bytes(self.device)
        return self

    def poll(self) -> None:
        self.peak = max(self.peak, peak_memory_bytes(self.device))

    def __exit__(self, *exc):
        self.poll()
        return False
