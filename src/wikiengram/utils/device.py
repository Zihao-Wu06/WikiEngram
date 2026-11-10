"""Device, dtype and memory helpers for every backend PyTorch supports.

* NVIDIA CUDA and AMD ROCm (PyTorch exposes ROCm through the ``torch.cuda`` API), including multi-GPU;
* Intel GPUs (``xpu``), Apple silicon (``mps``) and plain CPU.

Nothing here needs vendor-specific kernels. Precision is chosen per device: bf16 where the hardware supports it,
otherwise fp16 (older NVIDIA GPUs), and fp32 on CPU.
"""

from __future__ import annotations

import os

import torch


def _xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def available_cpus() -> int:
    """CPUs this process may really use: the affinity mask capped by a cgroup CPU quota (containers and cloud
    instances often show every host core but allow only a few)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:  # cgroup v2
        quota, period = _read("/sys/fs/cgroup/cpu.max").split()
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        quota = int(_read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
        period = int(_read("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
        if quota > 0:
            n = min(n, max(1, quota // period))
    except (OSError, ValueError):
        pass
    return n


def configure_threads() -> int:
    """Cap PyTorch's CPU threads at the CPUs actually available (``WIKIENGRAM_THREADS`` overrides). With 104
    threads on a 25-CPU quota every CPU-side step is throttled and the accelerator waits."""
    env = os.environ.get("WIKIENGRAM_THREADS")
    n = int(env) if env else available_cpus()
    if env or torch.get_num_threads() > n:
        torch.set_num_threads(max(1, n))
    return torch.get_num_threads()


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None and str(device) != "auto":
        return torch.device(device)
    env = os.environ.get("WIKIENGRAM_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():  # NVIDIA CUDA or AMD ROCm
        return torch.device("cuda")
    if _xpu_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accelerators() -> list[torch.device]:
    """Every compute device visible to this process (all GPUs for multi-GPU placement)."""
    if torch.cuda.is_available():
        return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    if _xpu_available():
        return [torch.device("xpu", i) for i in range(torch.xpu.device_count())]
    if torch.backends.mps.is_available():
        return [torch.device("mps")]
    return []


def supports_bf16(device: torch.device) -> bool:
    if device.type == "cuda":
        return bool(torch.cuda.is_bf16_supported())
    if device.type == "xpu":
        f = getattr(torch.xpu, "is_bf16_supported", None)
        return bool(f()) if f else True
    if device.type == "mps":
        try:
            torch.ones(1, dtype=torch.bfloat16, device=device)
            return True
        except Exception:  # pragma: no cover - old macOS
            return False
    return True


def resolve_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in (None, "auto"):
        if device.type == "cpu":
            return torch.float32
        return torch.bfloat16 if supports_bf16(device) else torch.float16
    return {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}[dtype]


def system_memory() -> tuple[int, int]:
    """``(available, total)`` bytes of host RAM."""
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.available), int(vm.total)
    except ImportError:  # pragma: no cover
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return total // 2, total


def device_memory(device: torch.device) -> tuple[int, int]:
    """``(free, total)`` bytes the device can use (Apple: the recommended working set of unified memory)."""
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        return int(free), int(total)
    if device.type == "xpu":
        f = getattr(torch.xpu, "mem_get_info", None)
        if f is not None:
            free, total = f(device)
            return int(free), int(total)
        total = int(torch.xpu.get_device_properties(device).total_memory)
        return int(total - torch.xpu.memory_allocated(device)), total
    if device.type == "mps":
        total = int(torch.mps.recommended_max_memory())
        return max(0, total - int(torch.mps.driver_allocated_memory())), total
    return system_memory()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "xpu" and hasattr(torch.xpu, "reset_peak_memory_stats"):
        torch.xpu.reset_peak_memory_stats(device)


def peak_memory_bytes(device: torch.device) -> int:
    """Peak allocated accelerator memory. On MPS PyTorch exposes no peak counter, so this reports the
    driver's current allocation (an upper bound of live tensors, sampled by the caller after each step)."""
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    if device.type == "xpu" and hasattr(torch.xpu, "max_memory_allocated"):
        return int(torch.xpu.max_memory_allocated(device))
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if os.uname().sysname == "Darwin" else 1024)
    except Exception:  # pragma: no cover
        return 0


class PeakMemoryMeter:
    """Tracks peak accelerator memory across a block (polls on MPS, uses the native counter elsewhere)."""

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


def hardware_summary() -> dict:
    """What this machine offers, for ``wikiengram hardware`` and placement planning."""
    avail, total = system_memory()
    out = {"torch": torch.__version__, "cpus_available": available_cpus(), "torch_threads": torch.get_num_threads(),
           "ram_available_gib": round(avail / 2**30, 1), "ram_total_gib": round(total / 2**30, 1), "devices": []}
    for dev in accelerators():
        free, tot = device_memory(dev)
        name = dev.type
        if dev.type == "cuda":
            name = torch.cuda.get_device_name(dev) + (" (ROCm)" if getattr(torch.version, "hip", None) else "")
        elif dev.type == "xpu":
            name = torch.xpu.get_device_name(dev)
        elif dev.type == "mps":
            name = "Apple GPU (unified memory)"
        out["devices"].append({"device": str(dev), "name": name, "free_gib": round(free / 2**30, 1),
                               "total_gib": round(tot / 2**30, 1), "bf16": supports_bf16(dev)})
    return out
