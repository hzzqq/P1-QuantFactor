"""阶段 5：成本敏感回测。

对 03（LightGBM 基线）与 04（GRU）产出的预测表，跑含涨跌停 + 交易成本的回测，
对比「毛收益（无成本）」与「净收益（扣成本）」，验证信号在真实约束下是否成立。

用法：
    python scripts/05_backtest.py
    python scripts/05_backtest.py --top-pct 0.1 --rebalance-freq 10
    python scripts/05_backtest.py --pred data/P1/processed/pred_gru_h10.parquet

    # 跨模型**同区间**公平对比（基线覆盖 2019-，GRU 仅 2022-，必须对齐）
    python scripts/05_backtest.py --pred data/P1/processed/pred_baseline_h10.parquet \
        --start-date 2022-01-01
"""
from __future__ import annotations

import argparse
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared import paths
from shared.logging_utils import get_logger

from src.backtest import (run_backtest, run_backtest_continuous,
                          DEFAULT_COST, BacktestResult)

logger = get_logger("P1.backtest")
PROCESSED = paths.DATA / "P1" / "processed"


def _gross_cost() -> dict:
    return {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 成本敏感回测")
    ap.add_argument("--pred", type=str, default=None,
                    help="预测表路径（默认跑 baseline + gru 两份）")
    ap.add_argument("--panel", type=str, default=None)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--top-pct", type=float, default=0.1)
    ap.add_argument("--rebalance-freq", type=int, default=10,
                    help="调仓频率（交易日）；默认 = horizon（非重叠桶，与标签窗口对齐）。"
                         "注意：曾误设为 15 造成与 horizon=10 错配，把信号压成负；"
                         "阶段 18/19 已证 rf=10 在多年度 4/5 占优，须与 horizon 对齐")
    ap.add_argument("--mode", choices=["long_short", "long_only"], default="long_short")
    ap.add_argument("--start-date", type=str, default=None,
                    help="回测起始日（含），如 2022-01-01；用于跨模型**同区间**公平对比")
    ap.add_argument("--end-date", type=str, default=None,
                    help="回测结束日（含），如 2026-08-14")
    ap.add_argument("--buffer", type=float, default=0.0,
                    help="缓冲区宽度（排名分位）；>0 启用连续持仓+缓冲区调仓（降换手）")
    args = ap.parse_args()

    panel_path = args.panel or (PROCESSED / "panel.parquet")
    panel = pd.read_parquet(panel_path)
    logger.info("行情面板 %s 行 | %s 只", f"{len(panel):,}", panel["symbol"].nunique())

    preds_to_run = []
    if args.pred:
        # 从文件名推导 tag（如 pred_gru_h10.parquet -> gru），
        # 否则多次 --pred 调用会互相覆盖同名权益曲线图。
        tag = Path(args.pred).stem.replace("pred_", "").replace(f"_h{args.horizon}", "")
        preds_to_run.append((tag or "custom", Path(args.pred)))
    else:
        for name in ("pred_baseline_h10", "pred_gru_h10"):
            p = PROCESSED / f"{name}.parquet"
            if p.exists():
                preds_to_run.append((name.replace("pred_", "").replace("_h10", ""), p))

    if not preds_to_run:
        logger.error("没有可用的预测表，请先跑 03/04")
        return 1

    reports = {}
    t0 = time.time()
    for tag, p in preds_to_run:
        preds = pd.read_parquet(p)
        # 同区间对齐：跨模型对比必须限定同一时间窗，否则覆盖期长的模型
        # 会用「easy years」虚增收益，这是最常见的不公平对比陷阱。
        if args.start_date or args.end_date:
            preds["date"] = pd.to_datetime(preds["date"])
            n_before = len(preds)
            if args.start_date:
                preds = preds[preds["date"] >= pd.Timestamp(args.start_date)]
            if args.end_date:
                preds = preds[preds["date"] <= pd.Timestamp(args.end_date)]
            logger.info("区间过滤 %s~%s：%s -> %s 行",
                        args.start_date or "-", args.end_date or "-",
                        f"{n_before:,}", f"{len(preds):,}")
        logger.info("=== 回测 %s（%s 行）===", tag, f"{len(preds):,}")

        common = dict(horizon=args.horizon, top_pct=args.top_pct,
                      rebalance_freq=args.rebalance_freq, mode=args.mode)
        if args.buffer > 0:
            # 连续持仓 + 缓冲区调仓：只对变动部分计成本，换手率大幅下降
            logger.info("降换手模式：连续持仓 + 缓冲区 buffer=%.3f", args.buffer)
            net = run_backtest_continuous(preds, panel, **common,
                                          cost=DEFAULT_COST, buffer=args.buffer)
            gross = run_backtest_continuous(preds, panel, **common,
                                            cost=_gross_cost(), buffer=args.buffer)
        else:
            net = run_backtest(preds, panel, **common,
                               cost=DEFAULT_COST)
            gross = run_backtest(preds, panel, **common,
                                 cost=_gross_cost())

        print(net.summary_text())
        print(f"  （对比）毛收益无成本累计 : {gross.ls_stats.get('total_return', float('nan')):.2%}"
              f"  夏普 {gross.ls_stats.get('sharpe', float('nan')):.2f}")

        reports[tag] = {
            "net": net.to_dict(),
            "gross": gross.to_dict(),
        }

        # 权益曲线图（英文标签，避免 CJK 字体缺失告警）
        suffix = f"_b{args.buffer:g}" if args.buffer > 0 else ""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(net.equity.index, net.equity.values, label=f"{tag} net (cost)", lw=1.5)
        ax.plot(gross.equity.index, gross.equity.values, label=f"{tag} gross (no cost)",
                lw=1.2, alpha=0.6, ls="--")
        ax.set_title(f"P1 Backtest · {tag} [{net.mode}] (LS, h={args.horizon})")
        ax.set_xlabel("Rebalance bucket")
        ax.set_ylabel("Net value (start=1)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = paths.MODELS / "P1" / f"backtest_{tag}_h{args.horizon}{suffix}.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        logger.info("权益曲线已保存: %s", out_png)

    rp = PROCESSED / f"report_backtest_h{args.horizon}.json"
    with rp.open("w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2, default=str)
    logger.info("回测报告已保存: %s", rp)
    logger.info("总耗时 %.1f 秒", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
