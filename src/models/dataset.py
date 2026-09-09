"""序列数据集构建。

为什么要三维数据：
    LightGBM 看的是「当日截面快照」—— 一只股票某天的 38 个因子值。
    神经网络的差异化价值在于「过去 N 天因子的演化路径」，
    所以要把数据组织成 (样本, 时间步, 特征) 的三维结构。

内存策略：
    先组织成 (n_symbols, n_dates, n_features) 的 float32 数组，
    约 5000 × 2800 × 38 × 4B ≈ 2.1GB，可接受。
    样本只保存 (symbol_idx, date_idx) 索引，取用时再切片，
    绝不预先展开成 (样本 × 时间步 × 特征) 的四维数组（那会是 TB 级）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from shared.logging_utils import get_logger

logger = get_logger("P1.models.dataset")


def build_3d(data: pd.DataFrame, factor_names: list[str],
             y_col: str = "y_excess") -> tuple[np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
    """把 long 数据转成三维数组。

    Returns:
        X3d : (n_symbols, n_dates, n_features) float32
        y3d : (n_symbols, n_dates) float32
        dates, symbols : 对应坐标轴
    """
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])

    dates = np.sort(df["date"].unique())
    symbols = np.sort(df["symbol"].unique())
    date_idx = pd.Index(dates)
    sym_idx = pd.Index(symbols)
    n_s, n_d, n_f = len(symbols), len(dates), len(factor_names)

    logger.info("构建三维数组: %s 只 × %s 天 × %s 因子", n_s, n_d, n_f)
    X3d = np.full((n_s, n_d, n_f), np.nan, dtype=np.float32)

    for k, feat in enumerate(factor_names):
        mat = df.pivot(index="date", columns="symbol", values=feat)
        mat = mat.reindex(index=date_idx, columns=sym_idx)
        # R10：pivot 默认 float64（单因子约 112MB），转 float32 后再取数，
        # 把构建期内存尖峰直接减半（build_3d 本就按 float32 落盘）。
        mat = mat.astype("float32")
        X3d[:, :, k] = mat.to_numpy(dtype="float32", na_value=np.nan).T
        if (k + 1) % 10 == 0:
            logger.debug("  已处理 %s/%s 个因子", k + 1, n_f)

    ymat = df.pivot(index="date", columns="symbol", values=y_col)
    ymat = ymat.reindex(index=date_idx, columns=sym_idx)
    y3d = ymat.to_numpy(dtype="float32", na_value=np.nan).T

    valid = int(np.isfinite(y3d).sum())
    logger.info("三维数组完成，标签有效样本 %s（%.1f%%）",
                f"{valid:,}", valid / max(1, y3d.size) * 100)
    return X3d, y3d, dates, symbols


class SequenceDataset(Dataset):
    """按需切片生成序列样本。

    Args:
        X3d, y3d    : build_3d 的输出
        seq_len     : 回看窗口长度（交易日）
        symbol_mask : 布尔数组，哪些股票参与（用于训练/测试切分）
        date_mask   : 布尔数组，哪些日期参与
        date_stride : 每隔几天取一个样本，用来控制训练规模
        max_nan_ratio: 窗口内允许的最大缺失比例
        fill        : 缺失值填充方式，"zero"（标准化后 0 即均值）或 "ffill"
    """

    def __init__(self, X3d: np.ndarray, y3d: np.ndarray, seq_len: int = 20,
                 symbol_mask: np.ndarray | None = None,
                 date_mask: np.ndarray | None = None,
                 date_stride: int = 1, max_nan_ratio: float = 0.2,
                 fill: str = "zero"):
        self.X3d = X3d
        self.y3d = y3d
        self.seq_len = seq_len
        self.fill = fill

        n_s, n_d, n_f = X3d.shape
        if symbol_mask is None:
            symbol_mask = np.ones(n_s, dtype=bool)
        if date_mask is None:
            date_mask = np.ones(n_d, dtype=bool)

        self.indices = self._collect(symbol_mask, date_mask, date_stride,
                                     max_nan_ratio)
        logger.info("序列数据集: %s 个样本（seq_len=%s, stride=%s）",
                    f"{len(self.indices):,}", seq_len, date_stride)

    def _collect(self, symbol_mask, date_mask, stride, max_nan) -> np.ndarray:
        n_s, n_d, n_f = self.X3d.shape
        min_valid = 1.0 - max_nan
        out: list[tuple[int, int]] = []

        for s in range(n_s):
            if not symbol_mask[s]:
                continue
            notnan = np.isfinite(self.X3d[s]).astype(np.float32)
            csum = np.vstack([
                np.zeros((1, n_f), dtype=np.float32),
                np.cumsum(notnan, axis=0),
            ])
            win_sum = csum[self.seq_len:] - csum[:-self.seq_len]
            ratio = win_sum.sum(axis=1) / float(self.seq_len * n_f)

            y_valid = np.isfinite(self.y3d[s, self.seq_len - 1:])
            d_valid = date_mask[self.seq_len - 1:]
            ok = np.where((ratio >= min_valid) & y_valid & d_valid)[0]
            if stride > 1:
                ok = ok[::stride]
            out.extend((s, int(d + self.seq_len - 1)) for d in ok)

        return np.array(out, dtype=np.int64).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        s, d = self.indices[i]
        x = self.X3d[s, d - self.seq_len + 1: d + 1, :].copy()   # (T, F)

        if self.fill == "zero":
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            x = _ffill_2d(x)

        # 必须显式给 float32，用 Python float() 会被 collate 成 double，
        # 与模型输出的 float32 在做 loss 时报 dtype 不匹配
        return torch.from_numpy(x), torch.tensor(self.y3d[s, d],
                                                 dtype=torch.float32)


def _ffill_2d(arr: np.ndarray) -> np.ndarray:
    """沿时间维前向填充，再补 0。"""
    df = pd.DataFrame(arr).ffill().bfill()
    return np.nan_to_num(df.to_numpy(dtype="float32"), nan=0.0)


def make_masks(dates: np.ndarray, train_end, valid_end,
               symbols: np.ndarray | None = None,
               symbol_subset: np.ndarray | None = None,
               label_horizon: int = 0):
    """生成训练/验证/测试的日期掩码。

    Args:
        train_end / valid_end: 日期分界（含）
        label_horizon: 标签窗口长度（交易日）。>0 时把训练/验证样本截止日
            在交易日轴上**回退 label_horizon**，使样本标签窗口 d..d+horizon
            不越过分界（避免训练标签偷看 validation / validation 标签偷看 test 期价格，
            标准 walk-forward hygiene）。默认 0 = 旧行为（不钳制，兼容历史脚本）。
    """
    d = pd.to_datetime(pd.Series(dates))
    di = pd.Index(d.values)
    train_cut = pd.Timestamp(train_end)
    valid_cut = pd.Timestamp(valid_end)
    if label_horizon and label_horizon > 0:
        # 在交易日轴上回退，保证标签窗口 [d, d+label_horizon] 完全落在各自区间内
        pos = di.get_indexer([pd.Timestamp(train_end)])
        if int(pos[0]) >= 0:
            k = max(0, int(pos[0]) - label_horizon)
            train_cut = di[k]
        posv = di.get_indexer([pd.Timestamp(valid_end)])
        if int(posv[0]) >= 0:
            kv = max(0, int(posv[0]) - label_horizon)
            valid_cut = di[kv]
    train = (d <= train_cut).to_numpy()
    valid = ((d > train_cut) & (d <= valid_cut)).to_numpy()
    test = (d > valid_cut).to_numpy()

    if symbol_subset is None:
        # 始终返回布尔数组（不再因 symbols 缺失而返回 None，避免下游索引崩溃）
        sym = np.ones(len(symbols), dtype=bool) if symbols is not None else np.ones(0, dtype=bool)
    else:
        sym = np.isin(symbols, symbol_subset)
    return train, valid, test, sym
