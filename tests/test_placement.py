"""Placement decisions for models larger than the accelerator, and the `wikiengram hardware` report."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from wikiengram.backends.streaming import checkpoint_bytes, max_memory, parse_size, should_stream
from wikiengram.cli import app


def test_parse_size_units():
    assert parse_size("20GiB") == 20 * 2**30
    assert parse_size("500 MB") == 500 * 10**6
    assert parse_size("1.5gb") == 1_500_000_000
    assert parse_size("123") == 123


def _fake_checkpoint(path, nbytes, fp8=False):
    path.mkdir()
    cfg = {"model_type": "x"}
    if fp8:
        cfg["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": nbytes},
                                                                   "weight_map": {}}))
    return path


def test_checkpoint_bytes_counts_fp8_releases_at_16_bit(tmp_path):
    assert checkpoint_bytes(str(_fake_checkpoint(tmp_path / "bf16", 10**9))) == 10**9
    assert checkpoint_bytes(str(_fake_checkpoint(tmp_path / "fp8", 10**9, fp8=True))) == 2 * 10**9


def test_should_stream_follows_the_placement_setting(tmp_path):
    small = str(_fake_checkpoint(tmp_path / "small", 10**6))
    huge = str(_fake_checkpoint(tmp_path / "huge", 10**15))
    assert should_stream(small, "stream") and not should_stream(huge, "resident")
    assert not should_stream(small, "auto") and should_stream(huge, "auto")
    assert should_stream(small, "auto", max_gpu_memory="100KB")  # an explicit cap smaller than the model
    with pytest.raises(ValueError):
        should_stream(small, "sometimes")


def test_max_memory_has_host_ram_and_respects_caps():
    mm = max_memory("3GiB", "5GiB")
    assert mm["cpu"] == "5GiB" and all(v == "3GiB" for k, v in mm.items() if k != "cpu")


def test_hardware_report_plans_a_native_release(tmp_path, tiny_tokenizer):
    pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    from tests.test_checkpoint_streaming import write_release_checkpoint
    from tests.test_native_qwen import make_tiny_qwen4exp

    path = tmp_path / "release"
    write_release_checkpoint(make_tiny_qwen4exp(len(tiny_tokenizer) + 8, tiny_tokenizer.eos_token_id),
                             tiny_tokenizer, path)
    res = CliRunner().invoke(app, ["hardware", "--model", str(path), "--json"])
    assert res.exit_code == 0, res.output
    info = json.loads(res.output)
    assert "devices" in info and info["plan"]["placement"] in ("resident", "stream")
    native = info["plan"]["native_streaming"]
    assert native["parameters_billion"].keys() == {"resident", "experts", "ngram_table", "skipped"}
