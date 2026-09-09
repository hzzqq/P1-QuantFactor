"""阶段 21b：EV t10 输入归一(N13) 重训权重 vs 旧生产权重(M1 raw EV) · 同口径对比。

目的：判定是否用新权重替换生产信号。
隔离变量 —— 两者同为 800 子集 / 15 epoch / seq40·h128·lr1e-3·layers=2 / seed=42 /
同测试集（train/valid/test 切分一致），差异**仅来自 input_norm**（N13）。故 IC 与
long-short 净收益/夏普的任何变化都可归因于 input_norm。

对比口径（与 14_event_factor_iterate.py run_one 完全一致）：
    run_backtest(pred[["date","symbol","pred"]], panel,
                 horizon=10, top_pct=0.1, rebalance_freq=15,
                 cost=DEFAULT_COST, mode="long_short").ls_stats

用法：
    python scripts/21b_retrain_compare.py        # 需先跑完 21（生成 pred_gru_ev_t10_inputnorm_h10.parquet）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]      # E:/project/sj
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths
from shared.logging_utils import get_logger
from src.eval import metrics
from src.backtest import run_backtest, DEFAULT_COST

logger = get_logger("P1.ev21b_compare")
PROCESSED = paths.DATA / "P1" / "processed"
OLD = PROCESSED / "pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet"   # M1 旧生产信号
NEW = PROCESSED / "pred_gru_ev_t10_inputnorm_h10.parquet"           # 21 重训（input_norm）
HORIZON, TOP_PCT, REBAL, COST = 10, 0.1, 15, DEFAULT_COST


def evaluate(name: str, pred_path: Path) -> dict:
    if not pred_path.exists():
        raise FileNotFoundError(f"{name} 缺失：{pred_path}（请先跑完 21_retrain_ev_inputnorm.py）")
    pred = pd.read_parquet(pred_path)
    s = metrics.summarize(pred, "pred", "y_excess")
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    bt = run_backtest(pred[["date", "symbol", "pred"]], panel,
                      horizon=HORIZON, top_pct=TOP_PCT,
                      rebalance_freq=REBAL, cost=COST, mode="long_short").ls_stats
    by_year = {int(y): metrics.summarize(g, "pred", "y_excess")
               for y, g in pred.groupby("year")}
    return {
        "name": name, "pred_path": str(pred_path), "n": len(pred),
        "ic_mean": s.get("ic_mean"), "icir": s.get("icir"),
        "icir_annual": s.get("icir_annual"),
        "ic_positive_rate": s.get("ic_positive_rate"),
        "t_stat": s.get("t_stat"), "n_days": s.get("n_days"),
        "net": bt.get("total_return"), "sharpe": bt.get("sharpe"),
        "mdd": bt.get("max_drawdown"), "win_rate": bt.get("win_rate"),
        "n_buckets": bt.get("n_buckets"),
        "by_year_ic": {str(y): (v.get("ic_mean") if v.get("ic_mean") == v.get("ic_mean") else None)
                       for y, v in by_year.items()},
    }


def _fmt(v, d=4):
    return "-" if v is None or (isinstance(v, float) and v != v) else f"{v:.{d}f}"


def main() -> int:
    old = evaluate("M1_rawEV_old(14生产权重,无input_norm)", OLD)
    new = evaluate("M2_inputnorm_new(21重训,input_norm ON)", NEW)

    def d(k):
        return (new.get(k) or 0) - (old.get(k) or 0)

    better = ((new["ic_mean"] or 0) >= (old["ic_mean"] or 0) and
              (new["net"] or 0) >= (old["net"] or 0))
    report = {
        "config": {"horizon": HORIZON, "top_pct": TOP_PCT,
                   "rebalance_freq": REBAL, "cost": COST,
                   "apples_to_apples": "两者同 800子集/15ep/seq40_h128_lr1e-3_L2/seed42/同测试集，仅差 input_norm"},
        "old": old, "new": new,
        "delta": {"ic_mean": d("ic_mean"), "icir": d("icir"),
                  "icir_annual": d("icir_annual"), "net": d("net"),
                  "sharpe": d("sharpe"), "mdd": d("mdd"),
                  "win_rate": d("win_rate")},
        "verdict": "REPLACE" if better else "KEEP_OLD",
    }
    out = PROCESSED / "report_ev21b_compare.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")

    print("=" * 72)
    print("  EV t10 input_norm 重训 vs 旧生产权重(M1 raw EV) — 同口径对比")
    print("=" * 72)
    print(f"{'指标':<14}{'OLD(14)':>16}{'NEW(21)':>16}{'Δ':>14}")
    for k, label in [("ic_mean", "IC均值"), ("icir", "ICIR"),
                     ("icir_annual", "ICIR年化"), ("ic_positive_rate", "IC正占比"),
                     ("net", "净收益"), ("sharpe", "夏普"),
                     ("mdd", "最大回撤"), ("win_rate", "胜率")]:
        ov, nv = old.get(k), new.get(k)
        print(f"{label:<14}{_fmt(ov):>16}{_fmt(nv):>16}{_fmt((nv or 0) - (ov or 0)):>14}")
    print("-" * 72)
    print("  分年度 IC（OLD vs NEW）：")
    yrs = sorted(set(list(old["by_year_ic"]) + list(new["by_year_ic"])),
                 key=lambda x: int(x))
    for y in yrs:
        print(f"   {y}  OLD={_fmt(old['by_year_ic'].get(y))}  "
              f"NEW={_fmt(new['by_year_ic'].get(y))}")
    print("=" * 72)
    print(f"  结论：{report['verdict']}  "
          f"（IC↑ 且 净收益↑ → 替换生产信号；否则保留旧权重）")
    print(f"  报告已保存：{out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("21b 对比异常退出")
        raise
