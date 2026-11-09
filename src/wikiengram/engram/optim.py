"""Row-sparse optimizers for Engram tables.

Only the rows a batch touches receive a gradient, so optimizer state is allocated and updated per row
("lazy" Adam, as in TorchRec / DeepSeek's sparse table training). Built-in ``torch.optim.SparseAdam`` works on
MPS but its cost grows with the table size; this implementation is flat in table size and runs on CPU, CUDA
and MPS with dense ops only (indexing and ``index_put``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RowAdamConfig:
    lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0  # decoupled; for a personal delta this pulls rows back towards the base table


@torch.no_grad()
def row_adam_(
    values: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    steps: torch.Tensor,
    pos: torch.Tensor,
    grad: torch.Tensor,
    cfg: RowAdamConfig,
) -> None:
    """In-place Adam on ``values[pos]``. ``pos`` must be unique; ``grad`` is ``[len(pos), dim]``.

    Each row keeps its own step count, so bias correction is exact for rows touched at different rates.
    """
    if pos.numel() == 0:
        return
    grad = grad.to(values.dtype)
    st = steps[pos] + 1
    steps[pos] = st
    m = exp_avg[pos].mul_(cfg.beta1).add_(grad, alpha=1 - cfg.beta1)
    v = exp_avg_sq[pos].mul_(cfg.beta2).addcmul_(grad, grad, value=1 - cfg.beta2)
    exp_avg[pos] = m
    exp_avg_sq[pos] = v
    t = st.to(values.dtype).unsqueeze(-1)
    m_hat = m / (1 - torch.pow(cfg.beta1, t))
    v_hat = v / (1 - torch.pow(cfg.beta2, t))
    update = m_hat / (v_hat.sqrt() + cfg.eps)
    current = values[pos]
    if cfg.weight_decay:
        current = current * (1 - cfg.lr * cfg.weight_decay)
    values[pos] = current - cfg.lr * update


@torch.no_grad()
def row_sgd_(values: torch.Tensor, pos: torch.Tensor, grad: torch.Tensor, lr: float) -> None:
    if pos.numel():
        values[pos] = values[pos] - lr * grad.to(values.dtype)
