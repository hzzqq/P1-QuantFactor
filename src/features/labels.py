"""标签定义。

铁律（金融项目措辞纪律）：
    这里输出的不是「涨跌预测」，而是「未来 N 日相对基准的超额收益」。
    绝对收益里混杂着大盘 beta，用它做标签训练出来的模型只是在赌市场方向。
"""
from __future__ import annotations

import pandas as pd

from shared.logging_utils import get_logger

logger = get_logger("P1.features.labels")


def forward_return(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """未来 N 日收益：T 日收盘买入，T+N 日收盘卖出。

    注意 shift(-n) 是向前看，这是标签，严禁用于特征。
    """
    return close.shift(-n) / close - 1.0


def benchmark_forward_return(bench_close: pd.Series, n: int) -> pd.Series:
    """基准的未来 N 日收益"""
    return bench_close.shift(-n) / bench_close - 1.0


def excess_return(close: pd.DataFrame, bench_close: pd.Series,
                  n: int) -> pd.DataFrame:
    """超额收益 = 个股未来 N 日收益 − 基准未来 N 日收益"""
    stock_fwd = forward_return(close, n)
    bench_fwd = benchmark_forward_return(bench_close, n)
    return stock_fwd.sub(bench_fwd, axis=0)


def to_classification(excess: pd.DataFrame) -> pd.DataFrame:
    """把超额收益转成二分类标签：跑赢基准=1，跑输=0，NaN 保持 NaN。"""
    label = (excess > 0).astype("float")
    label[excess.isna()] = pd.NA
    return label


def build_labels(panel: pd.DataFrame, bench_df: pd.DataFrame,
                 horizon: int = 10) -> dict[str, pd.DataFrame]:
    """构建标签集。

    Returns:
        {"fwd_ret": 未来N日绝对收益, "excess": 超额收益, "cls": 二分类标签}
    """
    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    bench_close = (
        bench_df.set_index("date")["close"].sort_index()
        if "date" in bench_df.columns
        else bench_df["close"]
    )
    bench_close = bench_close.reindex(close.index).ffill()

    fwd = forward_return(close, horizon)
    exc = excess_return(close, bench_close, horizon)
    cls = to_classification(exc)

    logger.info("标签构建完成 horizon=%s，超额收益有效样本 %s",
                horizon, f"{int(exc.notna().sum().sum()):,}")
    return {"fwd_ret": fwd, "excess": exc, "cls": cls}
