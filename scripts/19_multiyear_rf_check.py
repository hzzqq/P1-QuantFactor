"""阶段 19：多年度 rf=10 vs rf=15 复核（用现有 walk-forward 全量预测，零重训）。

目的：阶段 18 发现 rebalance_freq=15 与标签 horizon=10 错配，导致信号被压成负；
对齐 rf=10 后全量 EV 2026 净由 -1.87% → +10.68%。但 +10.68% 仅验证了 2026 单年，
需确认「rf=10 优于 rf=15」这一效应在 2022-2025 多年度同样成立（否则只是单年偶然）。

做法：用 08-31 的 walk-forward 全量预测（pred_gru_h10 / pred_fusion_h10，覆盖
2022-2026 每年样本外），逐年（按信号生成日所在年）跑 run_backtest，rf 分别取 10 与 15，
对比 net / sharpe / mdd / 桶数。零重训、秒级。

用法：
    python scripts/19_multiyear_rf_check.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]      # E:/project/sj
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths
from src.backtest import engine as BE

PROCESSED = paths.DATA / "P1" / "processed"
YEARS = [2022, 2023, 2024, 2025, 2026]
RFS = [10, 15]


def main() -> int:
    ap = argparse.ArgumentParser(description="多年度 rf=10 vs rf=15 复核")
    ap.add_argument("--preds", nargs="+",
                    default=["pred_gru_h10", "pred_fusion_h10"],
                    help="预测文件名（不含 .parquet）")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--top-pct", type=float, default=0.1)
    args = ap.parse_args()

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    print(f"面板 {len(panel):,} 行 | {panel['symbol'].nunique()} 只")

    rows = []
    for name in args.preds:
        p = PROCESSED / f"{name}.parquet"
        if not p.exists():
            print(f"skip {name} (not found)"); continue
        preds = pd.read_parquet(p)
        preds["date"] = pd.to_datetime(preds["date"])
        print(f"\n=== {name}: {len(preds):,} 行, 年份覆盖 "
              f"{preds['date'].dt.year.min()}~{preds['date'].dt.year.max()} ===")
        for y in YEARS:
            lo, hi = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
            sub = preds[(preds["date"] >= lo) & (preds["date"] <= hi)]
            if sub.empty:
                print(f"  {y}: 无预测, skip"); continue
            for rf in RFS:
                res = BE.run_backtest(sub, panel, horizon=args.horizon,
                                      top_pct=args.top_pct, rebalance_freq=rf,
                                      cost=BE.DEFAULT_COST)
                st = res.ls_stats
                rows.append(dict(model=name, year=y, rf=rf,
                                 net=st["total_return"], sharpe=st["sharpe"],
                                 mdd=st["max_drawdown"], buckets=st["n_buckets"]))
                print(f"  {y} rf{rf:2d}: net {st['total_return']:+.2%} | "
                      f"sharpe {st['sharpe']:+.2f} | mdd {st['max_drawdown']:.2%} | "
                      f"buckets {st['n_buckets']}")

    df = pd.DataFrame(rows)
    out = PROCESSED / "report_multiyear_rf_check.csv"
    df.to_csv(out, index=False)
    print(f"\n=> 明细: {out}")

    print("\n=== 判读：rf=10 在哪些年份优于 rf=15（按 net）===")
    print(f"{'model':18s} | {'rf10胜年':>10s} | {'逐年净(rf10 / rf15)':>}")
    for name in args.preds:
        sub = df[df.model == name]
        if sub.empty:
            continue
        wins = 0
        line = f"{name:18s} | "
        for y in YEARS:
            a = sub[(sub.year == y) & (sub.rf == 10)]["net"].values
            b = sub[(sub.year == y) & (sub.rf == 15)]["net"].values
            if len(a) and len(b):
                ok = a[0] > b[0]
                wins += int(ok)
                line += f"  {y}:{a[0]:+.1%}/{b[0]:+.1%}{'*' if ok else ''}"
        line = line + f" | {wins}/{len(sub[sub.rf==10])} 年胜"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
