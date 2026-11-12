"""校正基准相对（2026-11-11）：修复 excess 起点除零 bug。

原脚本把 excess = le/be - 1（起点 0）又除以 excess[0] 导致除零。
正确做法：用比值序列 le/be（起点 1.0），窗口内超额 = ratio_end/ratio_start - 1。

只重算 6 次回测（3 个 rf 的基准 + 3 个 rf 的多头 gross），合并进已落盘的 JSON。
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
from src.backtest import run_backtest_continuous  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
HORIZON = 10
TOP_PCT = 0.1
ZERO_COST = {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}
RFS = [10, 20, 40]

panel = pd.read_parquet(PROCESSED / "panel.parquet")
all_dates = pd.to_datetime(sorted(panel["date"].unique()))
recent_start = str(all_dates[-59].date())
windows = {
    "FULL": (None, None),
    "2024+": ("2024-01-01", None),
    "2026": ("2026-01-01", None),
    "recent59td": (recent_start, None),
}
ens = pd.read_parquet(PROCESSED / "pred_ens_v39gru_w025_h10.parquet")
ens["date"] = pd.to_datetime(ens["date"])

rel_rows = []
for rf in RFS:
    bench = run_backtest_continuous(ens, panel, horizon=HORIZON, top_pct=1.0,
                                   rebalance_freq=rf, mode="long_only",
                                   buffer=0.0, cost=ZERO_COST)
    lo_g = run_backtest_continuous(ens, panel, horizon=HORIZON, top_pct=TOP_PCT,
                                  rebalance_freq=rf, mode="long_only",
                                  buffer=0.0, cost=ZERO_COST)
    be = bench.long_equity.copy()
    be.index = pd.to_datetime(bench.equity_dates)
    le = lo_g.long_equity.copy()
    le.index = pd.to_datetime(lo_g.equity_dates)
    common = be.index.intersection(le.index)
    be = be.reindex(common)
    le = le.reindex(common)
    ratio = le / be  # 比值序列，起点≈1.0
    for wname, (lo, hi) in windows.items():
        m = pd.Series(True, index=ratio.index)
        if lo:
            m &= (ratio.index >= pd.Timestamp(lo))
        if hi:
            m &= (ratio.index <= pd.Timestamp(hi))
        sub = ratio.reindex(ratio.index[m]).dropna()
        if len(sub) > 1:
            excess_tr = float(sub.iloc[-1] / sub.iloc[0] - 1.0)
            excess_end = float(sub.iloc[-1])
        else:
            excess_tr, excess_end = float("nan"), float("nan")
        rel_rows.append({
            "rf": rf, "window": wname,
            "excess_total_return": round(excess_tr, 4),
            "excess_end": round(excess_end, 4),
        })
    rr = [r for r in rel_rows if r["rf"] == rf and r["window"] == "FULL"][0]
    print(f"rf={rf} 基准相对(FULL 累计超额) = {rr['excess_total_return']:+.2%}", flush=True)

outp = PROCESSED / "reports" / "portfolio_construction_diag_20261111.json"
d = json.loads(outp.read_text(encoding="utf-8"))
d["benchmark_relative"] = rel_rows
d["note"] += " （基准相对已校正：用 le/be 比值序列，窗口内 excess=ratio_end/ratio_start-1）"
outp.write_text(json.dumps(d, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(f"已合并校正结果: {outp}")
