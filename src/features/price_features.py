"""量价因子（wide 格式：index=date, columns=symbol）。

为什么用 wide 二维矩阵而不是 long + groupby：
    全市场约 5000 只 × 2800 个交易日 ≈ 1400 万行，groupby().rolling() 是慢路径。
    转成 (date × symbol) 矩阵后，所有因子都是矩阵级向量化运算，快一个数量级。

铁律：
    所有因子只能使用 T 日及之前的信息，严禁未来函数。
    缺失值保留 NaN，由 pipeline 在横截面上统一处理。
"""
from __future__ import annotations

import pandas as pd

from shared.logging_utils import get_logger

logger = get_logger("P1.features.price")

DEFAULT_WINDOWS = [5, 10, 20, 60, 120]


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """避免除零产生 inf。"""
    return a / b.where(b.abs() > 1e-12)


def _min_periods(n: int, ratio: float = 0.8) -> int:
    return max(2, int(n * ratio))


def to_wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    """long -> wide（index=date，columns=symbol）"""
    wide = panel.pivot(index="date", columns="symbol", values=field)
    return wide.sort_index()


def momentum(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 日动量收益率"""
    return _safe_div(close, close.shift(n)) - 1.0


def reversal(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 日反转（动量的相反数；单独保留便于模型区分两种效应）"""
    return -momentum(close, n)


def volatility(ret_1: pd.DataFrame, n: int) -> pd.DataFrame:
    """N 日收益波动率"""
    return ret_1.rolling(n, min_periods=_min_periods(n)).std()


def volume_ratio(volume: pd.DataFrame, n: int) -> pd.DataFrame:
    """量比：当日成交量 / N 日均量，衡量放量程度"""
    mean_vol = volume.rolling(n, min_periods=_min_periods(n)).mean()
    return _safe_div(volume, mean_vol)


def volume_std(volume: pd.DataFrame, n: int) -> pd.DataFrame:
    """成交量波动：放量往往伴随信息冲击"""
    mean_vol = volume.rolling(n, min_periods=_min_periods(n)).mean()
    std_vol = volume.rolling(n, min_periods=_min_periods(n)).std()
    return _safe_div(std_vol, mean_vol)


def price_position(high: pd.DataFrame, low: pd.DataFrame,
                   close: pd.DataFrame, n: int) -> pd.DataFrame:
    """收盘价在 N 日最高/最低区间中的相对位置（0~1）"""
    hh = high.rolling(n, min_periods=_min_periods(n)).max()
    ll = low.rolling(n, min_periods=_min_periods(n)).min()
    return _safe_div(close - ll, hh - ll)


def ma_bias(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """收盘价相对 N 日均线的偏离度"""
    ma = close.rolling(n, min_periods=_min_periods(n)).mean()
    return _safe_div(close, ma) - 1.0


def amplitude(high: pd.DataFrame, low: pd.DataFrame,
              close: pd.DataFrame) -> pd.DataFrame:
    """当日振幅"""
    return _safe_div(high - low, close)


def gap(open_: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """跳空幅度：开盘相对前收的跳空"""
    return _safe_div(open_, close.shift(1)) - 1.0


def build_price_features(panel: pd.DataFrame,
                         windows: list[int] | None = None) -> dict[str, pd.DataFrame]:
    """从面板数据构建全部量价因子。

    Args:
        panel: long 格式，至少含 date / symbol / open / high / low / close / volume

    Returns:
        {因子名: wide DataFrame(index=date, columns=symbol)}
    """
    windows = windows or DEFAULT_WINDOWS
    logger.info("构建量价因子，窗口=%s，输入 %s 行", windows, f"{len(panel):,}")

    open_ = to_wide(panel, "open")
    high = to_wide(panel, "high")
    low = to_wide(panel, "low")
    close = to_wide(panel, "close")
    volume = to_wide(panel, "volume")

    ret_1 = close.pct_change(1)
    feats: dict[str, pd.DataFrame] = {}

    for n in windows:
        feats[f"mom_{n}"] = momentum(close, n)
        feats[f"rev_{n}"] = reversal(close, n)
        feats[f"vol_{n}"] = volatility(ret_1, n)
        feats[f"vr_{n}"] = volume_ratio(volume, n)
        feats[f"vstd_{n}"] = volume_std(volume, n)
        feats[f"pos_{n}"] = price_position(high, low, close, n)
        feats[f"bias_{n}"] = ma_bias(close, n)

    feats["ampl_1"] = amplitude(high, low, close)
    feats["gap_1"] = gap(open_, close)
    feats["ret_1"] = ret_1

    logger.info("共生成 %s 个因子，矩阵形状 %s",
                len(feats), next(iter(feats.values())).shape)
    return feats
