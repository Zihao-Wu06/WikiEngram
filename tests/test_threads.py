"""CPU thread caps for containers whose CPU quota is smaller than the visible core count."""

from __future__ import annotations

import torch

from wikiengram.utils import device


def test_available_cpus_respects_a_cgroup_quota(monkeypatch):
    real_open = open

    def fake_open(path, *a, **kw):
        if str(path) == "/sys/fs/cgroup/cpu.max":
            import io

            return io.StringIO("2500000 100000")
        if str(path).startswith("/sys/fs/cgroup/cpu/"):
            raise OSError
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(device.os, "sched_getaffinity", lambda _pid: set(range(208)), raising=False)
    assert device.available_cpus() == 25


def test_configure_threads_caps_and_honours_the_override(monkeypatch):
    before = torch.get_num_threads()
    try:
        monkeypatch.setattr(device, "available_cpus", lambda: 1)
        monkeypatch.delenv("WIKIENGRAM_THREADS", raising=False)
        torch.set_num_threads(2)
        assert device.configure_threads() == 1
        monkeypatch.setenv("WIKIENGRAM_THREADS", "2")
        assert device.configure_threads() == 2
    finally:
        torch.set_num_threads(before)
