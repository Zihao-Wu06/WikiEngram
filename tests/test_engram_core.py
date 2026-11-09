from __future__ import annotations

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from wikiengram.engram.delta import PersonalDelta
from wikiengram.engram.optim import RowAdamConfig, row_adam_
from wikiengram.engram.tables import E4M3FN_LUT, E8M0_LUT, FP8RowTable, dequant_fp8_rows


def test_fp8_luts_match_torch():
    codes = torch.arange(256, dtype=torch.uint8)
    e4m3 = codes.view(torch.float8_e4m3fn).float().numpy()
    finite = ~np.isnan(e4m3)
    np.testing.assert_array_equal(E4M3FN_LUT[finite], e4m3[finite])
    assert np.isnan(E4M3FN_LUT[~finite]).all()
    e8m0 = codes.view(torch.float8_e8m0fnu).float().numpy()
    finite = ~np.isnan(e8m0)
    np.testing.assert_array_equal(E8M0_LUT[finite], e8m0[finite])


def test_fp8_row_table_reads_deepseek_layout(tmp_path):
    """A synthetic shard with DeepSeek-V4.1's engram tensor names, dtypes and block scales."""
    rows, dim = 37, 64
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(rows, dim, generator=g) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(120, 130, (rows, dim // 32), generator=g, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    path = tmp_path / "model-00047-of-00096.safetensors"
    save_file({"layers.14.engram.embed.weight": w, "layers.14.engram.embed.scale": s,
               "layers.14.engram.q_weight": torch.ones(4, 8, dtype=torch.bfloat16)}, str(path))
    table = FP8RowTable.from_deepseek_v41(path, 14)
    assert (table.num_rows, table.dim) == (rows, dim)
    idx = torch.tensor([36, 0, 5, 5, 17])
    got = table.gather(idx)
    # reference: DeepSeek's ParallelEngramEmbedding dequantization
    expected = (w[idx].float().unflatten(-1, (-1, 32)) * s[idx].float().unsqueeze(-1)).flatten(-2)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(
        dequant_fp8_rows(w[idx].view(torch.uint8).numpy(), s[idx].view(torch.uint8).numpy()), expected.numpy()
    )


def test_delta_insert_lookup_remove_roundtrip(tmp_path):
    d = PersonalDelta((1, 2), dim=4, layout_fingerprint="abc")
    layer = d.layers[1]
    pos = layer.ensure(torch.tensor([10, 3, 7, 3]))
    assert layer.rows.tolist() == [3, 7, 10]
    layer.values[pos] = torch.tensor([[1.0] * 4, [2.0] * 4, [3.0] * 4])
    got = d.lookup(1, torch.tensor([7, 99, 3]))
    assert got[:, 0].tolist() == [2.0, 0.0, 1.0]
    layer.set_active(torch.tensor([7]), False)
    assert d.lookup(1, torch.tensor([7]))[0, 0] == 0.0  # masked rows read as base
    layer.set_active(torch.tensor([7]), True)
    d.scale = 0.5
    assert d.lookup(1, torch.tensor([10]))[0, 0] == 1.5
    d.scale = 1.0
    path = tmp_path / "delta.safetensors"
    d.meta["note"] = "hello"
    d.save(path)
    d2 = PersonalDelta.load(path, expect_fingerprint="abc")
    assert d2.meta["note"] == "hello"
    torch.testing.assert_close(d2.layers[1].values, layer.values)
    assert d2.layers[1].rows.tolist() == [3, 7, 10]
    with pytest.raises(ValueError):
        PersonalDelta.load(path, expect_fingerprint="other")
    assert layer.remove(torch.tensor([7, 1000])) == 1
    assert layer.rows.tolist() == [3, 10]
    assert d.lookup(1, torch.tensor([7]))[0].abs().sum() == 0


def test_row_adam_matches_torch_adam_when_every_row_is_touched():
    torch.manual_seed(0)
    values = torch.randn(6, 3)
    ref = values.clone().requires_grad_(True)
    opt = torch.optim.Adam([ref], lr=1e-2)
    m, v, st = torch.zeros_like(values), torch.zeros_like(values), torch.zeros(6, dtype=torch.int32)
    pos = torch.arange(6)
    for _ in range(5):
        grad = torch.randn(6, 3)
        ref.grad = grad.clone()
        opt.step()
        row_adam_(values, m, v, st, pos, grad, RowAdamConfig(lr=1e-2))
    torch.testing.assert_close(values, ref.detach(), rtol=1e-5, atol=1e-6)


def test_row_adam_is_lazy():
    values = torch.zeros(4, 2)
    m, v, st = torch.zeros_like(values), torch.zeros_like(values), torch.zeros(4, dtype=torch.int32)
    row_adam_(values, m, v, st, torch.tensor([1, 3]), torch.ones(2, 2), RowAdamConfig(lr=0.1))
    assert values[0].abs().sum() == 0 and values[2].abs().sum() == 0
    assert st.tolist() == [0, 1, 0, 1]
    torch.testing.assert_close(values[1], torch.tensor([-0.1, -0.1]))
