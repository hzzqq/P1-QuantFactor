"""阶段 20b：M10 报告恢复 / 快速回测（不重训）。

读 `scripts/20_m10_full_confirm.py` 已落盘的预测文件，跑 M10 × M1 在 rf=10/15 的回测，
写 `report_m10_full_confirm.csv`。用于 20 在长训练后被外部信号/OOM 杀掉、但模型+预测已
落盘时，秒级补出报告（不碰训练、不加载大三维数组）。

前置：20 已成功落盘
  - data/P1/processed/pred_gru_m10_cost_full_h10.parquet
  - data/P1/processed/pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet（M1 对照，阶段14产物）
用法：
    python scripts/20b_m10_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths
from shared.logging_utils import get_logger
from src.backtest import run_backtest, DEFAULT_COST
from src.eval import metrics

PROCESSED = paths.DATA / "P1" / "processed"
HORIZON = 10
TOP_PCT = 0.1
logger = get_logger("P1.m10_report")


def bt_bucket(pred_df, panel, rf=10):
    return run_backtest(pred_df[["date", "symbol", "pred"]], panel, horizon=HORIZON,
                        top_pct=TOP_PCT, rebalance_freq=rf, cost=DEFAULT_COST,
                        mode="long_short", bootstrap=True).ls_stats


def main() -> int:
    m10_path = PROCESSED / "pred_gru_m10_cost_full_h10.parquet"
    if not m10_path.exists():
        raise SystemExit(f"找不到 {m10_path} —— 请先跑 scripts/20_m10_full_confirm.py 落盘预测")
    m10_pdf = pd.read_parquet(m10_path)
    m10_ic = metrics.summarize(m10_pdf, "pred", "y_excess")
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    rows = []
    for rf in (10, 15):
        st = bt_bucket(m10_pdf, panel, rf=rf)
        rows.append(dict(iter="M10", universe="full(1427)", variant="cost_sensitive", rf=rf,
                         ic=m10_ic.get("ic"), icir=m10_ic.get("icir"),
                         net=st["total_return"], sharpe=st["sharpe"],
                         mdd=st["max_drawdown"], buckets=st["n_buckets"]))
        logger.info("M10 rf=%d: IC %.4f ICIR %.4f net %.2f%% sharpe %.2f mdd %.2f%% buckets %d",
                    rf, m10_ic.get("ic_mean"), m10_ic.get("icir"), st["total_return"]*100,
                    st["sharpe"], st["max_drawdown"]*100, st["n_buckets"])

    FULL_EV = PROCESSED / "pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet"
    if FULL_EV.exists():
        m1_pdf = pd.read_parquet(FULL_EV)
        m1_ic = metrics.summarize(m1_pdf, "pred", "y_excess")
        for rf in (10, 15):
            st = bt_bucket(m1_pdf, panel, rf=rf)
            rows.append(dict(iter="M1", universe="full(1427)", variant="raw", rf=rf,
                             ic=m1_ic.get("ic"), icir=m1_ic.get("icir"),
                             net=st["total_return"], sharpe=st["sharpe"],
                             mdd=st["max_drawdown"], buckets=st["n_buckets"]))
            logger.info("M1(raw) rf=%d: IC %.4f ICIR %.4f net %.2f%% sharpe %.2f mdd %.2f%% buckets %d",
                        rf, m1_ic.get("ic_mean"), m1_ic.get("icir"), st["total_return"]*100,
                        st["sharpe"], st["max_drawdown"]*100, st["n_buckets"])
    else:
        logger.warning("未找到阶段14全量 EV 预测 %s，跳过 M1 对照", FULL_EV.name)

    out = PROCESSED / "report_m10_full_confirm.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("=> %s", out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("M10 报告恢复异常退出")
        raise
