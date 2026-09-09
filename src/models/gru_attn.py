"""GRU + Attention 序列模型。

设计考量（CPU 铁律下的取舍）：
    - hidden=64 / n_layers=2 / seq_len=20：把单次训练压在 30 分钟内
    - Attention 不只为效果，也为**可解释性**：
      能看出模型决策时关注了回看窗口里的哪几天，这是树模型给不了的东西
    - HuberLoss 而非 MSELoss：超额收益是厚尾分布，Huber 对极端值更稳健

与 LightGBM 的关系：
    这个模型的对手就是 LightGBM。它的输入是「过去 N 天的因子路径」，
    而 LightGBM 只有「当日快照」。如果打不过，说明时序路径没有增量信息。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUAttention(nn.Module):
    """GRU 编码 + 加性注意力池化 + 回归头。"""

    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2,
                 dropout: float = 0.2, attn_dim: int = 32):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        # N13：GRU 输入层归一化。因子经 pipeline 已截面 zscore，但逐样本/逐 batch 仍可能
        # 因缺失填充引入偏移；输入 LayerNorm 进一步拉平量纲、稳住梯度、加速收敛。
        self.input_norm = nn.LayerNorm(n_features)
        self.attn = nn.Sequential(
            nn.Linear(hidden, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1, bias=False),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        Args:
            x: (B, T, F)
        Returns:
            y: (B,)，若 return_attn=True 则额外返回注意力权重 (B, T)
        """
        x = self.input_norm(x)                            # (B, T, F) 输入归一
        out, _ = self.gru(x)                              # (B, T, H)
        w = torch.softmax(self.attn(out), dim=1)          # (B, T, 1)
        ctx = (out * w).sum(dim=1)                        # (B, H)
        y = self.head(ctx).squeeze(-1)                    # (B,)
        if return_attn:
            return y, w.squeeze(-1)
        return y


class TCNBlock(nn.Module):
    """时序卷积块，作为 GRU 的对照模型。

    因果卷积（N6 修复）：时序模型绝不可窥见未来。原实现对 Conv1d 用对称 padding，
    使每个时间步都看到了未来 (k-1)*dilation 步，等于把未来收益泄漏进特征，
    回测看似有效、实为空头支票。改为仅左填充，输出 t 只依赖 <= t 的输入。
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=0, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1)
                           if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) —— 仅左填充，因果卷积，输出长度仍为 T
        x_pad = F.pad(x, (self.pad, 0))     # 左填充，长度 T+pad
        out = self.net(x_pad)               # 因果卷积，长度 T
        return out + self.downsample(x)     # 残差用原始对齐输入（长度 T）


class TCN(nn.Module):
    """多层 TCN，取最后一个时间步做回归。"""

    def __init__(self, n_features: int, channels=(64, 64, 32),
                 kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        layers, in_ch = [], n_features
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, out_ch, kernel,
                                   dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x.transpose(1, 2))       # (B, C, T)
        return self.head(out[:, :, -1]).squeeze(-1)
