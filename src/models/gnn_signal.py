"""因子图卷积网络（Factor-Graph GCN）—— 开题清单 #2 的图神经网络部分。

设计（与 GRUAttention / TransformerSignal 同接口，便于 head-to-head）：
    GRUAttention 基线把 F 个因子经 Linear 朴素投影到 hidden，
    隐式假设因子间相互独立。本模型显式引入「因子相关性图」，
    用 GCN 在因子图上做消息传递，建模因子间的非线性交互，
    再接时序 GRU + 注意力池化 + 回归头，预测 horizon 日超额收益。

    图结构：节点 = 因子（F 个），边 = 训练期因子截面相关系数
    （绝对值阈值裁剪 + 对称化 + 自环），对称归一化邻接
    A_hat = D^{-1/2} A D^{-1/2}。图由训练数据确定性计算后传入，
    不随 batch 变化（因子间的结构关系近似时不变）。

公平性：hidden=64 / n_layers=2 / seq_len=20 与基线同口径；
时序感知完全由 GRU 提供，GCN 只在因子维做空间聚合，对比公平。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_factor_graph(X3d: np.ndarray, date_mask: np.ndarray,
                       n_f: int, threshold: float = 0.3) -> np.ndarray:
    """由训练期因子矩阵构建对称归一化邻接矩阵 (F, F)。

    Args:
        X3d: (n_symbols, n_dates, n_f) 三维因子数组
        date_mask: 训练期日期布尔掩码 (n_dates,)
        n_f: 因子数
        threshold: 相关系数绝对值阈值（边裁剪）
    Returns:
        A_hat: (n_f, n_f) 对称归一化邻接（含自环）
    """
    Xtr = X3d[:, date_mask, :].reshape(-1, n_f)
    Xtr = Xtr[np.isfinite(Xtr).all(axis=1)]
    if Xtr.shape[0] < n_f * 2:
        # 样本不足，退化为单位图（仅自环，等价于无图卷积）
        return np.eye(n_f, dtype=np.float32)
    corr = np.corrcoef(Xtr, rowvar=False)         # (n_f, n_f)
    corr = np.nan_to_num(corr, nan=0.0)
    A = (np.abs(corr) >= threshold).astype(np.float32)
    A = np.maximum(A, A.T)                          # 对称化
    np.fill_diagonal(A, 1.0)                        # 自环
    D = A.sum(axis=1)
    Dinv = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    A_hat = (Dinv[:, None] * A) * Dinv[None, :]
    return A_hat.astype(np.float32)


class FactorGraphGCN(nn.Module):
    """因子图卷积 + 时序 GRU 的序列模型。

    Args（与 GRUAttention 对齐）:
        n_features: 输入因子维度
        hidden: GRU 隐层（默认 64，公平对比）
        n_layers: GRU 层数（默认 2）
        gnn_layers: 因子图卷积层数（默认 2）
        gnn_hidden: 因子节点特征维度（默认 64）
        dropout: dropout 率
        adj: 因子图对称归一化邻接 (n_f, n_f)，None 时须先 set_graph
    """

    def __init__(self, n_features: int, hidden: int = 64, n_layers: int = 2,
                 gnn_layers: int = 2, gnn_hidden: int = 64, dropout: float = 0.2,
                 adj: np.ndarray | None = None):
        super().__init__()
        self.n_features = n_features
        self.hidden = hidden
        self.input_norm = nn.LayerNorm(n_features)
        # 因子投影：把每个因子节点(标量)映射为 gnn_hidden 维节点特征
        self.factor_proj = nn.Linear(1, gnn_hidden)
        # 因子图卷积层（消息传递）
        self.gnn_layers = nn.ModuleList(
            [nn.Linear(gnn_hidden, gnn_hidden) for _ in range(gnn_layers)])
        self.gnn_act = nn.ReLU()
        self.gnn_drop = nn.Dropout(dropout)
        # 时序编码器（与 GRUAttention 一致接口）
        self.gru = nn.GRU(gnn_hidden, hidden, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0)
        self.attn = nn.Sequential(
            nn.Linear(hidden, 32), nn.Tanh(), nn.Linear(32, 1, bias=False))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        if adj is not None:
            self.register_buffer("adj", torch.as_tensor(np.asarray(adj, np.float32)))
        else:
            self.adj = None

    def set_graph(self, adj: np.ndarray) -> None:
        self.register_buffer("adj", torch.as_tensor(np.asarray(adj, np.float32)))

    def _gcn(self, x: torch.Tensor) -> torch.Tensor:
        """在因子图（节点=因子，最后一维=节点特征）上做消息传递。x: (B, T, F)。"""
        b, t, f = x.shape
        h = x.unsqueeze(-1)                         # (B, T, F, 1)
        h = self.factor_proj(h)                     # (B, T, F, gnn_hidden)
        h = h.reshape(b * t, f, -1)                 # (B*T, F 节点, gnn_hidden 特征)
        for lin in self.gnn_layers:
            msg = torch.einsum("ij,njd->nid", self.adj, h)   # A @ H
            h = self.gnn_act(lin(msg))
            h = self.gnn_drop(h)
        h = h.reshape(b, t, f, -1).mean(dim=2)      # 因子节点均值池化 → (B, T, gnn_hidden)
        return h

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        """
        Args:
            x: (B, T, F)
        Returns:
            y: (B,)，若 return_attn=True 则额外返回注意力权重 (B, T)
        """
        if self.adj is None:
            raise RuntimeError("FactorGraphGCN: 必须先 set_graph(adj) 或构造时传入 adj")
        x = self.input_norm(x)                      # (B, T, F)
        h = self._gcn(x)                            # (B, T, gnn_hidden)
        out, _ = self.gru(h)                        # (B, T, hidden)
        w = torch.softmax(self.attn(out), dim=1)    # (B, T, 1)
        ctx = (out * w).sum(dim=1)                  # (B, hidden)
        y = self.head(ctx).squeeze(-1)              # (B,)
        if return_attn:
            return y, w.squeeze(-1)
        return y
