"""事件因子适配器（P4 TextPulse → P1 因子 #39，数据就绪时启用）。

数据契约（P4 财务情感输出应满足）：
    parquet 列：[date, symbol, sentiment]
    - date: 交易日（与本系统 panel 的 date 对齐）
    - symbol: 股票代码（与抓取列表一致，如 sh600000）
    - sentiment: 连续值 ∈ [-1, 1]，+1 强看多 / -1 强看空 / 0 中性
      （也可为分类标签 positive/neutral/negative，见 `label_to_score`）

本模块把上述情感信号转成 P1 可消费的横截面因子：
    1. 按日横截面 z-score（与本系统其它 38 因子口径一致）
    2. 对齐到 panel 的 (date×symbol) 索引，缺失日/股填 0（中性）
    3. 输出单列 wide DataFrame，列名 `event_sentiment`，可直接拼为第 39 维

⚠️ 当前阻塞：P4 TextPulse 目前只是「合成中文情感分类器 demo」（模板+线索词造句子训练），
   没有「逐股财经新闻/公告 → 情感分」的真实数据管线，也没有财务域标注数据。
   因此本适配器**代码已就绪但无真实输入**——真实文本源 + 财务域训练到位后即可一键并入，
   不要拿合成 demo 的输出当真实因子喂进来（会污染信号）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def label_to_score(series: pd.Series) -> pd.Series:
    """分类标签 / 连续值 → 连续分 ∈ [-1, 1]。"""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).clip(-1.0, 1.0)
    return series.astype(str).str.lower().map(LABEL_MAP).fillna(0.0)


def build_event_factor(sentiment_df: pd.DataFrame,
                       panel: pd.DataFrame,
                       value_col: str = "sentiment") -> pd.DataFrame:
    """把逐股情感分转成对齐 panel 的横截面 z 因子（单列 `event_sentiment`）。"""
    df = sentiment_df[["date", "symbol", value_col]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df[value_col] = label_to_score(df[value_col])

    # 1) 日横截面 z-score（与现有 38 因子口径一致）
    g = df.groupby("date")[value_col]
    z = (df[value_col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    df["event_sentiment"] = z.fillna(0.0)

    # 2) 对齐 panel 索引，缺失填 0
    idx = panel[["date", "symbol"]].drop_duplicates()
    out = idx.merge(df[["date", "symbol", "event_sentiment"]],
                    on=["date", "symbol"], how="left")
    wide = out.pivot(index="date", columns="symbol",
                     values="event_sentiment").fillna(0.0)
    return wide


def load_and_build(sentiment_path: str,
                   panel_path: Optional[str] = None) -> pd.DataFrame:
    sent = pd.read_parquet(sentiment_path)
    if panel_path is None:
        from shared import paths
        panel_path = paths.DATA / "P1" / "processed" / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    return build_event_factor(sent, panel)


if __name__ == "__main__":
    # 冒烟测试：用 3 行玩具数据验证合并逻辑可运行（非真实数据）
    toy = pd.DataFrame({
        "date": ["2026-01-05", "2026-01-05", "2026-01-06"],
        "symbol": ["sh600000", "sh600001", "sh600000"],
        "sentiment": [0.8, -0.5, 0.3],
    })
    panel = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-05",
                                "2026-01-06", "2026-01-06"]),
        "symbol": ["sh600000", "sh600001", "sh600000", "sh600001"],
        "close": [1, 2, 1, 2],
    })
    wide = build_event_factor(toy, panel)
    print("[冒烟测试] event_sentiment wide 形状:", wide.shape)
    print(wide.round(3))
    print("⚠️ 注意：这是代码冒烟测试，使用玩具数据；"
          "P4 真实财务情感数据就绪前不可用为因子。")
