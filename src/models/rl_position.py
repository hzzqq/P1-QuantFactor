"""仓位决策策略网络（开题清单 #3 的强化学习部分）。

PolicyNet：小 MLP，state -> action ∈ [-1, 1]（tanh 限幅）。
状态 = [信号z, mom_z, vol_z, pos_z, bias_z]（决策日截面 z-score）。
由 38_rl_position.py 用 REINFORCE（高斯策略 + 批均值基线）训练与评估。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """确定性均值 μ=tanh(head)，训练时由 38_rl_position 加高斯噪声采样。"""

    def __init__(self, state_dim: int = 5, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, max(8, hidden // 2)), nn.ReLU(),
            nn.Linear(max(8, hidden // 2), 1), nn.Tanh(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)   # (B,) ∈ [-1, 1]


class ValueNet(nn.Module):
    """价值基线网络：回归到确定性动作的奖励，作为 REINFORCE 的 low-variance 基线。"""

    def __init__(self, state_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, max(8, hidden // 2)), nn.ReLU(),
            nn.Linear(max(8, hidden // 2), 1))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)   # (B,) 基线奖励估计
