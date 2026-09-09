"""统一训练循环（CPU 优化）。

CPU 训练要点：
    - torch.set_num_threads() 吃满物理核心
    - DataLoader 用 num_workers=0：数据已在内存，多进程反而徒增开销，
      且 Windows 下多进程 DataLoader 需要 spawn 保护，容易踩坑
    - 梯度裁剪必备：序列模型容易梯度爆炸
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.logging_utils import get_logger

from src.training.common import train_robust

logger = get_logger("P1.models.trainer")


def set_cpu_threads(n: int = 8) -> None:
    torch.set_num_threads(n)


def train_model(model: nn.Module, train_ds, valid_ds=None, epochs: int = 20,
                batch_size: int = 1024, lr: float = 1e-3, weight_decay: float = 1e-5,
                patience: int = 5, grad_clip: float = 1.0, num_threads: int = 8,
                seed: int = 42, loss_type: str = "huber", verbose: bool = True,
                max_batches_per_epoch: int | None = None,
                checkpoint_path: str | None = None, resume: bool = True):
    """训练序列模型。

    已委托给 `src.training.common.train_robust`（R7/R8 单一真相源：
    稳健初始化 + 梯度裁剪 + ReduceLROnPlateau(min_lr 地板) + 早停）。
    本函数保留原签名，供 04/10/14/18/20a 无缝调用；新增 `checkpoint_path`/
    `resume` 透传（断点续训，默认 None 等价旧行为）。

    Returns:
        (model, history) —— history 含每轮的 train_loss / valid_loss / 耗时
    """
    return train_robust(
        model, train_ds, valid_ds=valid_ds, epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay, patience=patience, grad_clip=grad_clip,
        num_threads=num_threads, seed=seed, loss_type=loss_type, verbose=verbose,
        max_batches_per_epoch=max_batches_per_epoch,
        checkpoint_path=checkpoint_path, resume=resume,
    )


@torch.no_grad()
def predict(model: nn.Module, dataset, batch_size: int = 4096,
            num_threads: int = 8) -> np.ndarray:
    """批量推理，返回预测数组。"""
    set_cpu_threads(num_threads)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)
    outs = []
    for xb, _ in loader:
        outs.append(model(xb).numpy())
    return np.concatenate(outs) if outs else np.array([])
