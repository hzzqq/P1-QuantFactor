"""量化评估指标。

指标口径说明（写报告时直接用）：
    IC   —— 每日横截面上，预测值与未来超额收益的 Spearman 秩相关。
            经验阈值：> 0.02 有信息，> 0.05 不错，> 0.08 已经很强。
    ICIR —— IC 均值 / IC 标准差，衡量信号稳定性。
            未年化 > 0.3 可用，年化（×√252）后 > 2 算优秀。
    分组单调性 —— 按预测值分 N 组，各组平均超额收益应单调递增。
            这是比 IC 更直观、更难造假的证据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.logging_utils import get_logger

logger = get_logger("P1.eval.metrics")


def daily_ic(df: pd.DataFrame, pred_col: str = "pred",
             y_col: str = "y_excess", min_n: int = 5) -> pd.Series:
    """每日横截面 IC（Spearman 秩相关），全向量化实现。"""
    sub = df[["date", pred_col, y_col]].dropna().copy()
    if sub.empty:
        return pd.Series(dtype="float64")

    grp = sub.groupby("date", sort=True)
    sub["p_rank"] = grp[pred_col].rank()
    sub["y_rank"] = grp[y_col].rank()

    mean_p = grp["p_rank"].transform("mean")
    mean_y = grp["y_rank"].transform("mean")
    dp = sub["p_rank"] - mean_p
    dy = sub["y_rank"] - mean_y

    cov = (dp * dy).groupby(sub["date"]).sum()
    var_p = (dp**2).groupby(sub["date"]).sum()
    var_y = (dy**2).groupby(sub["date"]).sum()
    n = grp.size()

    # 某一日预测或标签在截面上为常数 → 方差为 0，分母 0 会产生 inf/nan。
    # 显式把「分母 <= 0」的日子置 nan（0/0 也归入 nan），下游 ic_summary 会 dropna。
    denom = np.sqrt(var_p * var_y)
    denom = denom.where(denom > 0)
    ic = cov / denom
    return ic.where(n >= min_n).sort_index()


def ic_summary(ic: pd.Series) -> dict:
    """IC 序列的汇总统计。"""
    s = ic.dropna()
    if len(s) < 2:
        return {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                "ic_positive_rate": np.nan, "t_stat": np.nan, "n_days": len(s)}
    mean, std = float(s.mean()), float(s.std())
    ir = mean / std if std > 0 else np.nan
    # 单样本 t 检验：判断 IC 是否显著不为 0
    t_stat = mean / std * np.sqrt(len(s)) if std > 0 else np.nan
    return {
        "ic_mean": mean,
        "ic": mean,  # 别名：下游脚本曾误用 .get("ic") 取到 None，保留别名避免静默缺失
        "ic_std": std,
        "icir": ir,
        "icir_annual": ir * np.sqrt(252) if ir == ir else np.nan,
        "ic_positive_rate": float((s > 0).mean()),
        "t_stat": float(t_stat) if t_stat == t_stat else np.nan,
        "n_days": int(len(s)),
    }


def quantile_returns(df: pd.DataFrame, pred_col: str = "pred",
                     y_col: str = "y_excess", n_groups: int = 5) -> pd.DataFrame:
    """按预测值横截面分 N 组，观察各组平均超额收益。

    组号越大表示预测越强。理想结果是第 1 组到第 N 组单调递增。
    """
    sub = df[["date", pred_col, y_col]].dropna().copy()
    if sub.empty:
        return pd.DataFrame()

    def _cut(s: pd.Series) -> pd.Series:
        if s.notna().sum() < n_groups:
            return pd.Series(np.nan, index=s.index)
        return pd.qcut(s.rank(method="first"), n_groups, labels=False)

    sub["group"] = sub.groupby("date")[pred_col].transform(_cut)
    out = sub.dropna(subset=["group"]).groupby("group")[y_col].agg(
        mean="mean", std="std", count="count"
    )
    out.index = out.index.astype(int) + 1
    return out


def top_bottom_spread(df: pd.DataFrame, pred_col: str = "pred",
                      y_col: str = "y_excess", top_pct: float = 0.2,
                      min_n: int = 10) -> pd.Series:
    """每日做多预测最高的一批、做空最低的一批，返回多空收益差序列。"""
    sub = df[["date", pred_col, y_col]].dropna().copy()

    def _pick(g: pd.DataFrame) -> float:
        if len(g) < min_n:
            return np.nan
        k = max(1, int(len(g) * top_pct))
        top = g.nlargest(k, pred_col)[y_col].mean()
        bottom = g.nsmallest(k, pred_col)[y_col].mean()
        return top - bottom

    return sub.groupby("date", sort=True).apply(_pick, include_groups=False).dropna()


def decile_monotonicity(df: pd.DataFrame, pred_col: str = "pred",
                        y_col: str = "y_excess", n_groups: int = 5) -> dict:
    """分组单调性检验——比 IC 更直觉、更难造假的信号证据。

    按预测值横截面分 N 组，检验各组平均超额收益是否随预测强度单调递增：
        - spread   : 第 N 组 - 第 1 组 的平均超额（多空分组收益差）
        - slope    : 组均值对组号的线性回归斜率（>0 即单调向上）
        - monotonic: 各组均值是否严格单调递增
    """
    q = quantile_returns(df, pred_col, y_col, n_groups)
    if q.empty or len(q) < 2:
        return {"spread": np.nan, "slope": np.nan, "monotonic": False}
    means = q["mean"].values.astype(float)
    groups = np.arange(1, len(means) + 1)
    slope = float(np.polyfit(groups, means, 1)[0]) if len(means) > 1 else np.nan
    spread = float(means[-1] - means[0])
    mono = bool(np.all(np.diff(means) > 0))
    return {"spread": spread, "slope": slope, "monotonic": mono}


def summarize(pred_df: pd.DataFrame, pred_col: str = "pred",
              y_col: str = "y_excess", horizon: int = 10) -> dict:
    """一次性产出全部核心指标。

    Args:
        horizon: 持有/调仓周期（交易日），用于把多空价差序列年化。
            默认 10 与标签窗口一致；horizon≠10 时必须传入正确值，否则年化口径失真。
    """
    ic = daily_ic(pred_df, pred_col, y_col)
    summary = ic_summary(ic)
    spread = top_bottom_spread(pred_df, pred_col, y_col)
    if len(spread) > 0:
        summary["spread_mean"] = float(spread.mean())
        summary["spread_std"] = float(spread.std())
        summary["spread_sharpe"] = (
            float(spread.mean() / spread.std() * np.sqrt(252 / horizon))
            if spread.std() > 0 else np.nan
        )
        summary["spread_win_rate"] = float((spread > 0).mean())
    return summary


def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 1000,
                 ci: float = 0.95, seed: int = 0) -> tuple:
    """对样本做非参数 bootstrap，返回 (lo, hi) 置信区间。

    用于给回测的「累计收益 / 夏普」这类点估计附上不确定性区间，
    避免把单次回测的幸运值当成稳健结论。
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]          # 丢弃 NaN/inf（如除零产生的坏样本），避免污染区间
    if len(arr) < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = len(arr)
    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = stat_fn(arr[idx])
    lo = float(np.percentile(draws, 50.0 * (1 - ci)))
    hi = float(np.percentile(draws, 50.0 * (1 + ci)))
    return (lo, hi)


def backtest_bootstrap(rets: list[float], rebalance_freq: int = 10,
                       n_boot: int = 1000, ci: float = 0.95,
                       seed: int = 0, years: float | None = None) -> dict:
    """对回测的逐桶收益序列，bootstrap 出累计收益与夏普的置信区间。

    Args:
        years: 保留形参以兼容调用方（run_backtest 传入年化年限），
            本函数内部不依赖它（置信区间基于逐桶收益经验分布），仅占位避免 TypeError。

    返回 {total_return_ci: (lo, hi), sharpe_ci: (lo, hi)}；
    样本不足时返回空 dict（调用方原样合并即可）。
    """
    a = np.asarray(rets, dtype=float)
    a = a[np.isfinite(a)]                # 丢弃 NaN/inf 收益，避免累计/夏普统计量被污染
    if len(a) < 5:
        return {}

    def _total(x: np.ndarray) -> float:
        return float((1.0 + x).prod() - 1.0)

    def _sharpe(x: np.ndarray) -> float:
        if x.std() == 0:
            return np.nan
        return float(x.mean() / x.std() * np.sqrt(252 / rebalance_freq))

    return {
        "total_return_ci": bootstrap_ci(a, _total, n_boot, ci, seed),
        "sharpe_ci": bootstrap_ci(a, _sharpe, n_boot, ci, seed),
    }
