"""阶段 27 - ②：合并 v39 baseline 8 年预测，与原始 baseline(38 因子) 对比 IC / 回测。

合并 pred_baseline_h10_v39_part1/part2a/2025/2026 → pred_baseline_h10_v39.parquet，
计算逐年 + 整体 IC、ICIR、IC 正占比，以及 long-short 回测，对比原始 baseline。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths
from src.eval import metrics
from src.backtest import run_backtest, DEFAULT_COST

PROCESSED = paths.DATA / "P1" / "processed"


def main() -> int:
    t0 = time.time()
    parts = [
        "pred_baseline_h10_v39_part1.parquet",
        "pred_baseline_h10_v39_part2a.parquet",
        "pred_baseline_h10_v39_2025.parquet",
        "pred_baseline_h10_v39_2026.parquet",
    ]
    dfs = [pd.read_parquet(PROCESSED / p) for p in parts]
    v39 = pd.concat(dfs, ignore_index=True)
    v39.to_parquet(PROCESSED / "pred_baseline_h10_v39.parquet", index=False)
    print(f"[29] 合并 v39 baseline: {len(v39):,} 行, 年份 {sorted(v39['year'].unique())}", flush=True)

    orig = pd.read_parquet(PROCESSED / "pred_baseline_h10.parquet")
    print(f"[29] 原始 baseline: {len(orig):,} 行, 年份 {sorted(orig['year'].unique())}", flush=True)

    # 整体指标
    s_v39 = metrics.summarize(v39, "pred", "y_excess")
    s_orig = metrics.summarize(orig, "pred", "y_excess")

    # 逐年 IC
    from scipy.stats import spearmanr
    def per_year_ic(d):
        out = {}
        for y, sub in d.groupby("year"):
            out[int(y)] = float(spearmanr(sub["y_excess"], sub["pred"]).correlation)
        return out
    ic_v39 = per_year_ic(v39)
    ic_orig = per_year_ic(orig)

    # long-short 回测
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    bt_v39 = run_backtest(v39[["date", "symbol", "pred"]], panel, horizon=10,
                          top_pct=0.1, rebalance_freq=15, cost=DEFAULT_COST, mode="long_short").ls_stats
    bt_orig = run_backtest(orig[["date", "symbol", "pred"]], panel, horizon=10,
                           top_pct=0.1, rebalance_freq=15, cost=DEFAULT_COST, mode="long_short").ls_stats

    report = {
        "model": "lightgbm_v39(38+log_amount)",
        "overall": {
            "ic_mean": float(s_v39["ic_mean"]), "icir": float(s_v39["icir"]),
            "ic_positive_rate": float(s_v39["ic_positive_rate"]),
            "orig_ic_mean": float(s_orig["ic_mean"]), "orig_icir": float(s_orig["icir"]),
            "orig_ic_positive_rate": float(s_orig["ic_positive_rate"]),
            "ic_delta": float(s_v39["ic_mean"] - s_orig["ic_mean"]),
        },
        "yearly_ic": {str(y): {"v39": ic_v39.get(y), "orig": ic_orig.get(y),
                               "delta": (ic_v39.get(y) - ic_orig.get(y)) if (ic_v39.get(y) is not None and ic_orig.get(y) is not None) else None}
                       for y in sorted(set(ic_v39) | set(ic_orig))},
        "long_short": {
            "v39": {k: float(bt_v39.get(k)) for k in ("total_return", "sharpe", "max_drawdown", "win_rate")},
            "orig": {k: float(bt_orig.get(k)) for k in ("total_return", "sharpe", "max_drawdown", "win_rate")},
        },
        "verdict": None,
    }
    # 判定：v39 整体 IC 是否高于 orig 且逐年多胜
    n_win = sum(1 for y in report["yearly_ic"] if (report["yearly_ic"][y]["delta"] or 0) > 0)
    n_years = len(report["yearly_ic"])
    report["verdict"] = (
        f"v39 整体 IC {s_v39['ic_mean']:.4f} vs orig {s_orig['ic_mean']:.4f} "
        f"(Δ {s_v39['ic_mean']-s_orig['ic_mean']:+.4f})；逐年 {n_win}/{n_years} 胜；"
        + ("log_amount 提 IC → 建议并入生产" if (s_v39["ic_mean"] > s_orig["ic_mean"]) else "log_amount 未提 IC")
    )
    json.dump(report, open(PROCESSED / "report_baseline_v39.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print(f"[29] IC  v39={s_v39['ic_mean']:.4f} orig={s_orig['ic_mean']:.4f} Δ={s_v39['ic_mean']-s_orig['ic_mean']:+.4f}")
    print(f"[29] ICIR v39={s_v39['icir']:.4f} orig={s_orig['icir']:.4f}")
    print(f"[29] LS净收益 v39={bt_v39.get('total_return'):.4f} orig={bt_orig.get('total_return'):.4f}")
    print(f"[29] verdict: {report['verdict']}")
    print(f"[29] 完成 {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
