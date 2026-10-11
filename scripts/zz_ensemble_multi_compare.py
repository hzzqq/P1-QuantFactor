"""T-255 三路融合对照：v39+GRU（现役） vs v39+Transformer vs v39+GRU+Transformer。

同参生产口径（37/35/36 实测）：rf=15 / top_pct=0.1 / buffer=0.1 / 长空组合。
输入均为刷新后数据（K线至 2026-10-09，标签至 09-17）：
    pred_baseline_h10_v39.parquet / pred_gru_h10.parquet / pred_transformer_h10.parquet
输出 report_ensemble_multi.json + 控制台对照表。诚实红线：三路无优势则维持现役。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shared import paths  # noqa: E402
from src.backtest import run_backtest_continuous, DEFAULT_COST  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
W_GRU = 0.25
RF, TOP_PCT, BUFFER = 15, 0.1, 0.1
ZERO = {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}


def _z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
            ).fillna(0.0).clip(-3, 3)


def main() -> int:
    v39 = pd.read_parquet(PROCESSED / "pred_baseline_h10_v39.parquet")
    gru = pd.read_parquet(PROCESSED / "pred_gru_h10.parquet")
    tfm = pd.read_parquet(PROCESSED / "pred_transformer_h10.parquet")
    for d in (v39, gru, tfm):
        d["date"] = pd.to_datetime(d["date"])
        d.drop_duplicates(subset=["date", "symbol"], keep="last", inplace=True)
    v39["z_v39"] = _z(v39, "pred")
    gru["z_gru"] = _z(gru, "pred")
    tfm["z_tfm"] = _z(tfm, "pred")

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    kw = dict(horizon=10, top_pct=TOP_PCT, rebalance_freq=RF, mode="long_short")

    def build(w_g, w_t):
        base = v39[["date", "symbol", "z_v39"]]
        m = base.merge(gru[["date", "symbol", "z_gru"]], on=["date", "symbol"], how="left")
        m = m.merge(tfm[["date", "symbol", "z_tfm"]], on=["date", "symbol"], how="left")
        m["pred"] = (1 - w_g - w_t) * m["z_v39"] \
            + w_g * m["z_gru"].fillna(0.0) + w_t * m["z_tfm"].fillna(0.0)
        return m[["date", "symbol", "pred"]]

    configs = {
        "v39+GRU(w.25)现役": build(W_GRU, 0.0),
        "v39+Transformer(w.25)": build(0.0, W_GRU),
        "三路(0.6/0.2/0.2)": build(0.2, 0.2),
    }
    out = {"config": {"w_gru_current": W_GRU, "rf": RF, "top_pct": TOP_PCT, "buffer": BUFFER},
           "compare": []}
    for name, pred in configs.items():
        net = run_backtest_continuous(pred, panel, **kw, cost=DEFAULT_COST, buffer=BUFFER)
        n = net.ls_stats or {}
        row = {"name": name, "net_total_return": round(float(n.get("total_return", float("nan"))), 4),
               "net_sharpe": round(float(n.get("sharpe", float("nan"))), 3),
               "net_max_dd": round(float(n.get("max_drawdown", float("nan"))), 4),
               "net_win_rate": round(float(n.get("win_rate", float("nan"))), 4)}
        out["compare"].append(row)
        print(f"{name:<28} 净累计 {row['net_total_return']:+.2%} | 夏普 {row['net_sharpe']:+.3f} "
              f"| 回撤 {row['net_max_dd']:.2%} | 胜率 {row['net_win_rate']:.3f}", flush=True)

    out["conclusion"] = "三路/Transformer 若未稳定优于现役（净夏普与累计收益同向占优），维持现役并留痕"
    with open(PROCESSED / "report_ensemble_multi.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("报告已写: report_ensemble_multi.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
