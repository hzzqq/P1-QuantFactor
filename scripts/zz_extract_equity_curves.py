"""抽取 P1 组合构建诊断的权益曲线（2026-11-11）。

为前端看板提供可直接画折线的累计净值序列：
- long_only_net  : 纯多头（net，真实成本）
- long_short_net : 市场中性（net，真实成本）
- benchmark_gross: 等权全市场（gross，零成本，作市场参照）
均 rf=10, buffer=0, top_pct=0.1(基准=1.0)。equity_dates 三者在同 rf 下对齐。
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
from src.backtest import (run_backtest_continuous, DEFAULT_COST)  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
HORIZON = 10
TOP_PCT = 0.1
ZERO_COST = {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}
RF = 10

panel = pd.read_parquet(PROCESSED / "panel.parquet")
ens = pd.read_parquet(PROCESSED / "pred_ens_v39gru_w025_h10.parquet")
ens["date"] = pd.to_datetime(ens["date"])

out = {}

lo = run_backtest_continuous(ens, panel, horizon=HORIZON, top_pct=TOP_PCT,
                            rebalance_freq=RF, mode="long_only", buffer=0.0, cost=DEFAULT_COST)
ls = run_backtest_continuous(ens, panel, horizon=HORIZON, top_pct=TOP_PCT,
                            rebalance_freq=RF, mode="long_short", buffer=0.0, cost=DEFAULT_COST)
bm = run_backtest_continuous(ens, panel, horizon=HORIZON, top_pct=1.0,
                            rebalance_freq=RF, mode="long_only", buffer=0.0, cost=ZERO_COST)

for name, r in (("long_only_net", lo), ("long_short_net", ls), ("benchmark_gross", bm)):
    eq = r.equity
    dates = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(eq.index)]
    out[name] = {"dates": dates, "values": [round(float(v), 4) for v in eq.values]}

# 2026 起始索引，供前端标记"近期弱势"段
cut = pd.Timestamp("2026-01-01")
idx2026 = next(i for i, d in enumerate(out["long_only_net"]["dates"]) if d >= "2026-01-01")
out["idx_2026"] = idx2026

outp = PROCESSED / "reports" / "portfolio_equity_curves_20261111.json"
outp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"已保存权益曲线: {outp}")
print(f"  长期多头净值终点={out['long_only_net']['values'][-1]:.3f} | "
      f"市场中性终点={out['long_short_net']['values'][-1]:.3f} | "
      f"基准终点={out['benchmark_gross']['values'][-1]:.3f}")
print(f"  2026 起始索引={idx2026} / 日期={out['long_only_net']['dates'][idx2026]}")
