"""阶段 6：降换手率实验（修订版）。

前一轮尝试的「连续持仓 + 缓冲区调仓」对**日度重估的 GRU 信号无效**：
GRU 预测日度波动大，上一期持仓下一期几乎全部跌出缓冲区（top 15% 都留不住），
每期照样全换（换手 ~100%），且因持仓窗口略长反而加大逐桶方差、波动拖累把复利从
16% 拖到 0.35%。故该杠杆废弃。

真正能线性压低交易成本的杠杆是**调仓频率**：非重叠桶每期 100% 换手，
年换手 ≈ 252/horizon 次；把 rebalance_freq 从 10 提到 20/30，交易次数与成本近似减半。
本脚本用正确的桶引擎扫描 rebalance_freq，量化「降频 → 降成本 → 提净收益」的权衡，
并定位净收益最大化的较优频率。

口径：GRU 预测（同区间 2022-01-01 起），多空 top10%，DEFAULT_COST。
"""
from __future__ import annotations

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
from shared.logging_utils import get_logger

from src.backtest import run_backtest, DEFAULT_COST

logger = get_logger("P1.turnover_sweep")
PROCESSED = paths.DATA / "P1" / "processed"

START = pd.Timestamp("2022-01-01")
HORIZON = 10
TOP_PCT = 0.1
MODE = "long_short"
FREQS = [5, 10, 15, 20, 30]


def _gross_cost() -> dict:
    return {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}


def main() -> int:
    t0 = time.time()
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    preds = pd.read_parquet(PROCESSED / "pred_gru_h10.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[preds["date"] >= START]
    logger.info("GRU 预测（同区间 %s~）：%s 行 | %s 只",
                START.date(), f"{len(preds):,}", preds["symbol"].nunique())

    rows = []
    for rf in FREQS:
        net = run_backtest(preds, panel, horizon=HORIZON, top_pct=TOP_PCT,
                           rebalance_freq=rf, cost=DEFAULT_COST, mode=MODE)
        gross = run_backtest(preds, panel, horizon=HORIZON, top_pct=TOP_PCT,
                             rebalance_freq=rf, cost=_gross_cost(), mode=MODE)
        s, g = net.ls_stats, gross.ls_stats
        n_b = s.get("n_buckets", 0)
        annual_turn = (n_b * 2 * TOP_PCT) / (n_b * rf / 252.0) if n_b else float("nan")
        rows.append({
            "rebalance_freq": rf,
            "n_buckets": n_b,
            "net_total_return": s.get("total_return"),
            "net_annual": s.get("annual_return"),
            "net_sharpe": s.get("sharpe"),
            "net_mdd": s.get("max_drawdown"),
            "win_rate": s.get("win_rate"),
            "gross_total_return": g.get("total_return"),
            "cost_drag_pp": (g.get("total_return", 0) - s.get("total_return", 0)) * 100,
            # 近似年换手（单边）：每桶交易 2*top_pct 仓位，年桶数 = 252/rf
            "approx_annual_turnover": 2 * TOP_PCT * (252.0 / rf),
        })
        logger.info("[rf=%2d] 净 %.2f%% 夏普 %.2f | 毛 %.2f%% | 成本吞噬 %.1fpp | 近似年换手 %.1f",
                    rf, s.get("total_return", 0) * 100, s.get("sharpe", 0),
                    g.get("total_return", 0) * 100, rows[-1]["cost_drag_pp"],
                    rows[-1]["approx_annual_turnover"])

    df = pd.DataFrame(rows)
    out = PROCESSED / "report_turnover_sweep_h10.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    logger.info("扫描结果已保存: %s", out)

    print("\n## 降换手率扫描（GRU，同区间 2022-01-01 起，多空 top10% h10，桶引擎）\n")
    print("杠杆：调仓频率（频率↓ → 交易次数↓ → 成本↓）。缓冲区连续持仓已证对日度信号无效。\n")
    hdr = ("| rebal_freq | 桶数 | 净收益 | 年化 | 夏普 | 最大回撤 | 胜率 | "
           "毛收益 | 成本吞噬(pp) | 近似年换手 |")
    sep = "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    print(hdr); print(sep)
    for _, r in df.iterrows():
        print("| {rf:>2d} | {nb} | {nt:.2%} | {na:.2%} | {sh:.2f} | {md:.2%} | "
              "{wr:.1%} | {gt:.2%} | {cd:.1f} | {at:.1f} |".format(
                  rf=int(r["rebalance_freq"]), nb=r["n_buckets"], nt=r["net_total_return"],
                  na=r["net_annual"], sh=r["net_sharpe"], md=r["net_mdd"],
                  wr=r["win_rate"], gt=r["gross_total_return"], cd=r["cost_drag_pp"],
                  at=r["approx_annual_turnover"]))
    best = df.loc[df["net_total_return"].idxmax()]
    print(f"\n净收益最大化：rebalance_freq={best['rebalance_freq']} → 净 {best['net_total_return']:.2%} / 夏普 {best['net_sharpe']:.2f}")
    print(f"耗时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
