"""纯 Transformer 序列模型 —— 作为 GRUAttention 的对照（开题清单 #1 的 Transformer 部分）。

设计：用 nn.TransformerEncoder 替代 GRU 编码器，保持与 GRUAttention 一致的接口
(n_features, hidden) 与 (B,) / (B, T) 返回，便于直接接入现有训练 / 导出 pipeline。

现状（2026-09-25，毕设算法改进）：
    P1 已有 gru_attn.GRUAttention（GRU + 加性注意力，注意力本身即"可解释性"卖点）
    与 gru_attn.TCN（时序卷积对照）。但**纯 Transformer 变体此前未实现**——
    本文件补齐该空白，作为 #1「Attention/Transformer 增强 GRU」的 Transformer 对照基线。

    本文件已接入 training 入口（scripts/04_train_nn.py `--model transformer`）与
    scripts/36_transformer_compare.py 对照脚本。训练使用 dataset_h10，与 GRUAttention 同口径
    (hidden=64 / n_layers=2) 做 head-to-head，并用 StockSignal
    modules.experiment_improvements.strict_holdout_eval 的方向命中率与 GRUAttention / 49.1% 论文锚对比。
    公平性说明：已加可学习位置编码，否则注意力对时间步置换不变，对比无意义。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerSignal(nn.Module):
    """Transformer 编码器 + 注意力池化 + 回归头（对标 GRUAttention 接口）。

    Args（与 GRUAttention 对齐，便于同 pipeline 实例化）：
        n_features: 输入因子维度（来自 dataset_h10 特征列）
        hidden: d_model / 隐层维度（默认 64，与 GRUAttention 一致，公平对比）
        n_layers: encoder 层数（默认 2）
        dropout: dropout 率（默认 0.2）
        n_heads: 多头注意力头数（默认 4）
        dim_feedforward: FFN 隐层（默认 256）
    """

    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2,
                 dropout: float = 0.2, n_heads: int = 4, dim_feedforward: int = 256,
                 max_len: int = 256):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        # 输入层归一（与 GRUAttention 同口径：因子已截面 zscore，LayerNorm 进一步拉平）
        self.input_norm = nn.LayerNorm(n_features)
        self.proj = nn.Linear(n_features, hidden)  # 特征 → d_model
        # 可学习位置编码：GRU 天然感知时序，纯 Transformer 不加位置编码会对时间步置换不变，
        # 导致对比不公平。加 PE 是标准做法，使 Transformer 也具备时序感知。
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # 加性注意力池化（与 GRUAttention.attn 同结构），保留"决策关注了哪几天"的可解释性
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        Args:
            x: (B, T, F)
        Returns:
            y: (B,) 预测；return_attn=True 时额外返回注意力权重 (B, T)
        """
        x = self.input_norm(x)                 # (B, T, F)
        h = self.proj(x)                       # (B, T, hidden)
        T = h.shape[1]
        h = h + self.pos_embed[:, :T, :]       # 位置编码（对齐 GRU 的时序感知）
        h = self.encoder(h)                    # (B, T, hidden)
        w = torch.softmax(self.attn_pool(h), dim=1)   # (B, T, 1)
        ctx = (h * w).sum(dim=1)              # (B, hidden) 注意力池化
        y = self.head(ctx).squeeze(-1)        # (B,)
        if return_attn:
            return y, w.squeeze(-1)
        return y
