"""共享稳健训练辅助函数（R7 稳健 LR + R8 抽取共享训练）。

此前 `src/models/trainer.py` 的 `train_model` 与 `scripts/20a` 的
`_train_cost_sensitive` 各自手抄了一份几乎相同的训练循环（梯度裁剪、ReduceLROnPlateau、
早停、种子），属于「脚本堆 + 训练不稳」的锐评项。本模块把它们收敛为单一真相源：

- seed_everything  : 固定 CPU / numpy / torch 随机，保证可复现
- robust_init      : GRU/Linear 用 Xavier/Glorot 稳健初始化，避免初始梯度爆炸
- build_scheduler  : AdamW + ReduceLROnPlateau（带 min_lr 地板，LR 不会塌到 0）
- train_robust     : 统一训练循环（梯度裁剪 + 调度 + 早停 + 最佳态恢复）
                     支持可选的 reg_term（如 M10 成本敏感正则 cost_lam·‖p‖²）

trainer.train_model 与 20a 的 cost 循环均委托到这里，避免重复实现漂移。
"""
from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.logging_utils import get_logger

logger = get_logger("P1.training.common")


def seed_everything(seed: int = 42) -> None:
    """固定全部随机源，保证训练可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CPU 序列模型不强制确定性算法（保速度），仅固定种子即可复现


def robust_init(module: nn.Module) -> None:
    """GRU/Linear 稳健初始化：Xavier/Glorot 权重 + 零偏置。

    默认 nn.Module 的 GRU 用 uniform(-1/√H, 1/√H)，在隐藏较大时
    初始 hidden 易溢出；显式 Xavier 更稳，是「训练不稳」的兜底。
    """
    for m in module.modules():
        if isinstance(m, nn.GRU):
            for name, p in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def build_scheduler(optimizer, min_lr: float = 1e-6):
    """ReduceLROnPlateau + min_lr 地板：验证损失不降时减半 LR，
    但绝不会塌到 0（稳态下仍能继续微调）。"""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=min_lr,
    )


def train_robust(model: nn.Module, train_ds, valid_ds=None, epochs: int = 20,
                 batch_size: int = 1024, lr: float = 1e-3,
                 weight_decay: float = 1e-5, patience: int = 5,
                 grad_clip: float = 1.0, num_threads: int = 8, seed: int = 42,
                 loss_type: str = "huber", reg_term=None, verbose: bool = True,
                 max_batches_per_epoch: int | None = None,
                 checkpoint_path: str | None = None, resume: bool = True):
    """统一稳健训练循环（R7 + R8）。

    参数
    ----
    reg_term : callable(predictions) -> 标量正则项，可选。
               例：M10 成本敏感用 `lambda p: cost_lam * p.pow(2).mean()`。
    checkpoint_path : 检查点路径（可选）。每轮结束落盘（模型/优化器/调度器/
               最佳态/early-stop 状态），使长训练可在被外部 SIGKILL 后**断点续训**；
               `resume=True` 时自动从该路径恢复。默认 None = 不落盘、等价旧行为。
    Returns:
        (model, history) —— history 含每轮 train_loss/valid_loss/sec
    """
    seed_everything(seed)
    torch.set_num_threads(num_threads)

    robust_init(model)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True, generator=g)
    valid_loader = (DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False,
                               num_workers=0) if valid_ds is not None else None)

    criterion = (nn.HuberLoss(delta=0.1) if loss_type == "huber"
                 else nn.MSELoss())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer)
    logger.info("稳健训练：seed=%s lr=%.1e clip=%.1f 调度=ReduceLROnPlateau(min_lr=1e-6)",
                seed, lr, grad_clip)

    history: list[dict] = []
    best_loss, best_state, wait = float("inf"), None, 0
    start_epoch = 1
    stopped = False

    # —— 断点续训：恢复模型/优化器/调度器/最佳态/early-stop 计数 ——
    if checkpoint_path and resume and os.path.exists(checkpoint_path):
        try:
            ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["model_state"])
            optimizer.load_state_dict(ck["optimizer_state"])
            scheduler.load_state_dict(ck["scheduler_state"])
            start_epoch = int(ck["epoch"]) + 1
            best_loss = float(ck["best_loss"])
            best_state = ck["best_state"]
            wait = int(ck["wait"])
            history = list(ck.get("history", []))
            stopped = bool(ck.get("stopped", False))
            logger.info("恢复检查点：从第 %s 轮继续（最佳 %.6f，wait=%s）",
                        start_epoch, best_loss, wait)
        except Exception as e:
            logger.warning("检查点加载失败，从头训练: %s", e)
            start_epoch, stopped = 1, False

    if stopped:  # 上一 chunk 已早停，直接收敛到最佳态返回
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history

    def _save_ckpt(epoch: int, stopped_now: bool = False) -> None:
        if not checkpoint_path:
            return
        try:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch, "best_loss": best_loss,
                "best_state": best_state, "wait": wait,
                "stopped": stopped_now, "history": history,
            }, checkpoint_path)
        except Exception as e:
            logger.warning("检查点落盘失败（不影响训练）: %s", e)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        model.train()
        total, seen = 0.0, 0

        for b, (xb, yb) in enumerate(train_loader):
            if max_batches_per_epoch and b >= max_batches_per_epoch:
                break
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            if reg_term is not None:
                loss = loss + reg_term(pred)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += float(loss.item()) * len(yb)
            seen += len(yb)

        train_loss = total / max(1, seen)

        valid_loss = None
        if valid_loader is not None:
            model.eval()
            vt, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb in valid_loader:
                    p = model(xb)
                    vl = criterion(p, yb)
                    if reg_term is not None:
                        vl = vl + reg_term(p)
                    vt += float(vl.item()) * len(yb)
                    vn += len(yb)
            valid_loss = vt / max(1, vn)
            scheduler.step(valid_loss)

        score = valid_loss if valid_loss is not None else train_loss
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "valid_loss": valid_loss, "sec": round(elapsed, 1)})

        if verbose:
            if valid_loss is not None:
                logger.info("  epoch %s/%s | train %.6f | valid %.6f | %.1fs",
                            epoch, epochs, train_loss, valid_loss, elapsed)
            else:
                logger.info("  epoch %s/%s | train %.6f | %.1fs",
                            epoch, epochs, train_loss, elapsed)

        if score < best_loss - 1e-8:
            best_loss, wait = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                logger.info("早停于第 %s 轮（最佳 %.6f）", epoch, best_loss)
                _save_ckpt(epoch, stopped_now=True)
                break

        _save_ckpt(epoch, stopped_now=False)  # 每轮落盘，抗外部 SIGKILL

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
