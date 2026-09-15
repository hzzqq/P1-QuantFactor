"""阶段 36：对 35 选出的最优配置做稳健性验证（防网格过拟合）。

35 的结论：rf=15 / top_pct=0.1 / buffer=0.1，ensemble w_gru=0.25 → 净夏普 0.718。
但 w=0.25 是在同一份数据上挑出来的，且峰值偏尖（0.25→0.5 掉 0.14），需验证：
  1) 细化权重网格（0.10~0.40），看峰值是否平滑、是否稳定在某一带；
  2) buffer=0 / 0.1 两种配置各扫一遍（GRU 单独跑时明显偏好 buf=0）；
  3) 子区间复核（2024-2026 最近段），确认不是靠早年 easy years 撑起来的。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths  # noqa: E402
from src.backtest import run_backtest, run_backtest_continuous, DEFAULT_COST  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
OUT = PROCESSED / "report_validate_ensemble.json"
HORIZON, MODE = 10, "long_short"
ZERO = {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}


def _z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
            ).fillna(0.0).clip(-3, 3)


def _run(preds, panel, rf, top_pct, buffer):
    kw = dict(horizon=HORIZON, top_pct=top_pct, rebalance_freq=rf, mode=MODE)
    if buffer > 0:
        net = run_backtest_continuous(preds, panel, **kw, cost=DEFAULT_COST, buffer=buffer)
        gross = run_backtest_continuous(preds, panel, **kw, cost=ZERO, buffer=buffer)
    else:
        net = run_backtest(preds, panel, **kw, cost=DEFAULT_COST)
        gross = run_backtest(preds, panel, **kw, cost=ZERO)
    n, g = net.ls_stats or {}, gross.ls_stats or {}
    return {
        "rf": rf, "top_pct": top_pct, "buffer": buffer,
        "net_sharpe": round(float(n.get("sharpe", float("nan"))), 4),
        "net_total_return": round(float(n.get("total_return", float("nan"))), 4),
        "net_mdd": round(float(n.get("max_drawdown", float("nan"))), 4),
        "gross_sharpe": round(float(g.get("sharpe", float("nan"))), 4),
    }


def main() -> int:
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    v39 = pd.read_parquet(PROCESSED / "pred_baseline_h10_v39.parquet")
    gru = pd.read_parquet(PROCESSED / "pred_gru_h10.parquet")
    for d in (v39, gru):
        d["date"] = pd.to_datetime(d["date"])
        d.drop_duplicates(subset=["date", "symbol"], keep="last", inplace=True)

    common = sorted(set(v39["date"].unique()) & set(gru["date"].unique()))
    v39c = v39[v39["date"].isin(common)].copy()
    gruc = gru[gru["date"].isin(common)].copy()
    v39c["z"] = _z(v39c, "pred")
    gruc["z"] = _z(gruc, "pred")
    base = (v39c[["date", "symbol", "z"]].rename(columns={"z": "z_v39"})
            .merge(gruc[["date", "symbol", "z"]].rename(columns={"z": "z_gru"}),
                   on=["date", "symbol"], how="inner")
            .drop_duplicates(subset=["date", "symbol"], keep="last"))
    print(f"[36] 对齐 {len(base):,} 行 | {common[0].date()} ~ {common[-1].date()}", flush=True)

    results = []
    weights = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.5, 1.0]

    for buffer in (0.0, 0.1):
        print("\n" + "=" * 68, flush=True)
        print(f"权重细扫  buffer={buffer}  (rf=15, top_pct=0.1)", flush=True)
        print("=" * 68, flush=True)
        for w in weights:
            m = base[["date", "symbol"]].copy()
            m["pred"] = ((1 - w) * base["z_v39"] + w * base["z_gru"]).to_numpy()
            r = _run(m, panel, 15, 0.1, buffer)
            r["w_gru"] = w
            r["model"] = "v39" if w == 0 else ("gru" if w == 1.0 else f"ens{w:.2f}")
            results.append(r)
            print(f"  w={w:<5} → 净夏普 {r['net_sharpe']:+.3f} | 累计 "
                  f"{r['net_total_return']:+.2%} | 回撤 {r['net_mdd']:.2%} | "
                  f"毛 {r['gross_sharpe']:+.3f}", flush=True)

    # 子区间复核：最近段 2024-01-01 起（防靠早年撑收益）
    print("\n" + "=" * 68, flush=True)
    print("子区间复核：2024-01-01 起（只留最近 ~2.6 年）", flush=True)
    print("=" * 68, flush=True)
    cut = pd.Timestamp("2024-01-01")
    sub = base[base["date"] >= cut]
    sub_res = []
    for w in (0.0, 0.15, 0.20, 0.25, 0.30):
        m = sub[["date", "symbol"]].copy()
        m["pred"] = ((1 - w) * sub["z_v39"] + w * sub["z_gru"]).to_numpy()
        for buffer in (0.0, 0.1):
            r = _run(m, panel, 15, 0.1, buffer)
            r["w_gru"] = w
            r["subperiod"] = "2024+"
            r["model"] = "v39" if w == 0 else f"ens{w:.2f}"
            sub_res.append(r)
            print(f"  w={w:<5} buf={buffer} → 净夏普 {r['net_sharpe']:+.3f} | "
                  f"累计 {r['net_total_return']:+.2%} | 回撤 {r['net_mdd']:.2%}", flush=True)

    OUT.write_text(json.dumps({
        "note": "阶段36 稳健性验证：细化权重网格 + 双 buffer 配置 + 2024+ 子区间复核",
        "window": [str(common[0].date()), str(common[-1].date())],
        "weight_sweep": results,
        "subperiod_2024plus": sub_res,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[36] 已保存 {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
