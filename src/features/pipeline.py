"""特征流水线：因子计算 → 去极值 → 截面标准化 → 对齐标签 → 落盘。

内存策略（重要）：
    全市场约 5000 只 × 2800 个交易日，一个 wide 矩阵 float32 约 56MB。
    若 43 个因子同时驻留并转 float64 会逼近 5GB，故采用
    「算一个 → 处理一个 → 立刻展平成 long 并释放」的流式处理。

防未来函数：
    因子只用 T 日及之前数据；标签是 T+1 ~ T+N 的收益，两者在最后一列对齐。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.logging_utils import get_logger

from . import labels as label_mod
from . import price_features as pf

logger = get_logger("P1.features.pipeline")


def _stack(mat: pd.DataFrame) -> pd.Series:
    """wide -> long，保留 NaN。

    手写实现而非用 DataFrame.stack()，是为了绕开不同 pandas 版本
    在 dropna / future_stack 上的行为差异（pandas 3.x 已改默认值）。
    """
    idx = pd.MultiIndex.from_product(
        [mat.index, mat.columns], names=["date", "symbol"]
    )
    values = mat.to_numpy(dtype="float32", na_value=np.nan).ravel()
    return pd.Series(values, index=idx)


def winsorize_mad(df: pd.DataFrame, n: float = 5.0) -> pd.DataFrame:
    """横截面 MAD 去极值：按当日截面中位数 ± n×1.4826×MAD 截断。"""
    med = df.median(axis=1)
    mad = df.sub(med, axis=0).abs().median(axis=1)
    span = n * 1.4826 * mad
    return df.clip(lower=med.sub(span, axis=0), upper=med.add(span, axis=0), axis=0)


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """每日横截面 z-score 标准化。量化因子必须做，否则量纲不可比。"""
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    std = std.where(std.abs() > 1e-12)
    return df.sub(mean, axis=0).div(std, axis=0)


def iter_factors(panel: pd.DataFrame, windows: list[int] | None = None):
    """逐个产出 (因子名, wide 矩阵)。"""
    windows = windows or pf.DEFAULT_WINDOWS

    open_ = pf.to_wide(panel, "open")
    high = pf.to_wide(panel, "high")
    low = pf.to_wide(panel, "low")
    close = pf.to_wide(panel, "close")
    volume = pf.to_wide(panel, "volume")
    ret_1 = close.pct_change(1)

    for n in windows:
        yield f"mom_{n}", pf.momentum(close, n)
        yield f"rev_{n}", pf.reversal(close, n)
        yield f"vol_{n}", pf.volatility(ret_1, n)
        yield f"vr_{n}", pf.volume_ratio(volume, n)
        yield f"vstd_{n}", pf.volume_std(volume, n)
        yield f"pos_{n}", pf.price_position(high, low, close, n)
        yield f"bias_{n}", pf.ma_bias(close, n)

    yield "ampl_1", pf.amplitude(high, low, close)
    yield "gap_1", pf.gap(open_, close)
    yield "ret_1", ret_1


def build_dataset(panel: pd.DataFrame, bench_df: pd.DataFrame, horizon: int = 10,
                  windows: list[int] | None = None, winsor: float = 5.0,
                  min_feature_ratio: float = 0.6) -> pd.DataFrame:
    """构建建模数据集（long 格式）。

    Returns:
        列：date / symbol / <各因子> / y_excess / y_cls / y_fwd
    """
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    # N12：面板若含重复 (date,symbol) 行，后续 inner join 会样本翻倍且标签错乱。
    # 先去重（保留最后一条），避免静默污染训练集。
    dup = int(panel.duplicated(subset=["date", "symbol"]).sum())
    if dup:
        logger.warning("面板含 %s 个重复 (date,symbol) 行，已去重（保留末条）", dup)
        panel = panel.drop_duplicates(subset=["date", "symbol"], keep="last")
    logger.info("构建数据集：%s 行面板，horizon=%s", f"{len(panel):,}", horizon)

    factor_cols: dict[str, pd.Series] = {}
    names: list[str] = []
    for name, mat in iter_factors(panel, windows):
        processed = cross_sectional_zscore(winsorize_mad(mat, winsor))
        factor_cols[name] = _stack(processed)
        names.append(name)
        del processed, mat

    X = pd.DataFrame(factor_cols)
    logger.info("因子数 %s，原始样本 %s 行", len(names), f"{len(X):,}")

    labs = label_mod.build_labels(panel, bench_df, horizon)
    y = pd.DataFrame({
        "y_excess": _stack(labs["excess"]),
        "y_cls": _stack(labs["cls"]),
        "y_fwd": _stack(labs["fwd_ret"]),
    })

    data = X.join(y, how="inner")

    # 除零会产生 inf（如停牌日价格为 0），必须清理。
    # 只要有一个 inf，mean() 之类的统计结果就会被整体污染成 NaN。
    data = data.replace([np.inf, -np.inf], np.nan)

    # 因子缺失过多的样本直接丢弃（多见于上市初期）
    thresh = max(1, int(len(names) * min_feature_ratio))
    valid = data[names].notna().sum(axis=1) >= thresh
    data = data[valid]

    # 标签缺失（未来数据不足）的样本必须丢弃
    before = len(data)
    data = data.dropna(subset=["y_excess"])
    logger.info("过滤后样本 %s 行（丢弃 %s 行：因子不足或标签缺失）",
                f"{len(data):,}", f"{before - len(data):,}")

    data = data.reset_index()
    # R9：特征/标签默认 float64，全市场宽表驻留近 GB。落盘前降为 float32
    #（build_3d 本就按 float32 读取，精度足够），磁盘与回读内存各减半。
    float_cols = data.select_dtypes(include=["float64", "float32"]).columns
    if len(float_cols):
        data[float_cols] = data[float_cols].astype("float32")
    data.attrs["factor_names"] = names
    return data
