"""P1 → StockSignal 信号导出。

把 P1 产出的横截面预测分数（pred）整理成 StockSignal 可 ingestion 的信号文件：
- 每个调仓日，按 pred 排序取多/空候选，输出结构化 JSON；
- 映射为 StockSignal 事件驱动体系能识别的离散信号（看多 / 看空）；
- 仅输出有效信号（剔除中性），符合「事件标注仅限利好/利空」的展示约定。

输出 schema（与 StockSignal 约定的最小接口）：
    {
      "generated_at": "ISO 时间",
      "horizon": 10,
      "model": "baseline_lgb",
      "latest_date": "2026-08-14",
      "top_long":  [{"symbol","score","rank"}...],
      "top_short": [{"symbol","score","rank"}...],
      "daily": [{"date","symbol","score","signal"}...]   # 全样本、可回放
    }
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SIGNAL_LONG = "看多"
SIGNAL_SHORT = "看空"
SIGNAL_NONE = "中性"


def build_signals(preds: pd.DataFrame,
                  top_n: int = 20,
                  long_quantile: float = 0.9,
                  short_quantile: float = 0.1,
                  keep_days: int = 60) -> dict:
    """从预测表生成信号字典。

    keep_days：仅在输出的 daily 中保留最近 N 个交易日（用于回放/对接），
    避免单次导出超大 JSON；Top/Bottom 只看最新交易日。
    """
    df = preds[["date", "symbol", "pred"]].dropna().copy()
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=keep_days * 1.5)  # 交易日约 1.5 天/个

    daily = []
    top_long_all, top_short_all = [], []
    for dt, g in df.groupby("date"):
        g = g.copy()
        g["rank"] = g["pred"].rank(pct=True)
        lo_thr = g["pred"].quantile(long_quantile)
        sh_thr = g["pred"].quantile(short_quantile)
        is_latest = pd.Timestamp(dt) >= pd.Timestamp(max_date)
        for _, r in g.iterrows():
            if r["pred"] >= lo_thr:
                sig = SIGNAL_LONG
            elif r["pred"] <= sh_thr:
                sig = SIGNAL_SHORT
            else:
                sig = SIGNAL_NONE
            # 仅保留近期逐日信号，控制文件体积
            if pd.Timestamp(dt) >= cutoff:
                daily.append({
                    "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "symbol": r["symbol"],
                    "score": round(float(r["pred"]), 6),
                    "signal": sig,
                })
        if is_latest:
            top_long_all = (g[g["pred"] >= lo_thr]
                            .sort_values("pred", ascending=False)
                            .head(top_n)
                            [["symbol", "pred", "rank"]]
                            .to_dict("records"))
            top_short_all = (g[g["pred"] <= sh_thr]
                             .sort_values("pred")
                             .head(top_n)
                             [["symbol", "pred", "rank"]]
                             .to_dict("records"))

    latest = pd.Timestamp(max_date).strftime("%Y-%m-%d")
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_date": latest,
        "top_long": top_long_all,
        "top_short": top_short_all,
        "daily": daily,
    }


def export_to_file(preds: pd.DataFrame, out_path: Path,
                   top_n: int = 20, model: str = "baseline_lgb",
                   horizon: int = 10) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sig = build_signals(preds, top_n=top_n)
    sig["model"] = model
    sig["horizon"] = horizon
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sig, f, ensure_ascii=False, indent=2)
    return sig
