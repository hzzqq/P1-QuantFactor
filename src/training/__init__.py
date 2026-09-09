"""P1 训练公共模块（R7 稳健训练 + R8 共享训练辅助函数入口）。"""
from src.training.common import (
    seed_everything,
    robust_init,
    build_scheduler,
    train_robust,
)

__all__ = ["seed_everything", "robust_init", "build_scheduler", "train_robust"]
