"""阶段 17：锐评后的 10 次迭代完善（严格 2026 hold-out）。

前置结论（锐评）：
  20 组 GRU 网格 IC 真实（0.05–0.10，ICIR≤0.98），但多空净收益薄（最优 t16 +1.93%）。
  根因 **不是模型容量**，而是 **执行层**：
    ① scripts 10/12 用 run_backtest（非重叠桶，每 15 日全量清仓重建 → ~200% 换手，
       年成本拖累 ~5–6%）把 alpha 吃光；仓库里已有 run_backtest_continuous（缓冲带，
       只交易变动部分）却没接进评估 → 近乎零成本的头号改进。
    ② regime 门控在「同窗」上选阈值，严格 OOS 不稳健。
    ③ 顶层 10% 多空是「高换手+噪声放大」结构；等权无仓位加权；无预测收缩/中性化。

本脚本在 **同一份 t16 预测（train≤2024，测仅 2026，模型从未见 2026）** 上做 10 次
执行/组合层杠杆实验，全部严格 2026 hold-out，逐轮记录 IC/ICIR/净收益/夏普/回撤/换手，
定位真正能改善 P&L 的杠杆。无重训，秒级跑完。

用法：
    python scripts/17_improve_iterate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths
from shared.config import load_config
from src.eval import metrics
from src.backtest import engine as BE

PROCESSED = paths.DATA / "P1" / "processed"
HORIZON = 10
TOP_PCT = 0.1
BEST_TAG = "t16_seq30_h128_lr1e-3"
COST = BE.DEFAULT_COST


def load():
    pred = pd.read_parquet(
        PROCESSED / f"pred_gru_tune_{BEST_TAG}_h{HORIZON}.parquet")
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred[pred["date"].dt.year == 2026].copy().reset_index(drop=True)
    base = pd.read_parquet(PROCESSED / "pred_baseline_h10.parquet")
    base["date"] = pd.to_datetime(base["date"])
    base = base[base["date"].dt.year == 2026].copy()
    base = base[["date", "symbol", "pred"]].rename(columns={"pred": "pred_base"})
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    return pred, base, panel


def ic_block(pred_df):
    s = metrics.summarize(pred_df, "pred", "y_excess")
    return s


def bt_bucket(pred_df, top_pct=TOP_PCT, rf=15, mode="long_short"):
    bt = BE.run_backtest(pred_df[["date", "symbol", "pred"]],
                         PANEL, horizon=HORIZON, top_pct=top_pct,
                         rebalance_freq=rf, cost=COST, mode=mode).ls_stats
    return bt


def bt_cont(pred_df, top_pct=TOP_PCT, rf=15, buffer=0.02, mode="long_short"):
    bt = BE.run_backtest_continuous(pred_df[["date", "symbol", "pred"]],
                                    PANEL, horizon=HORIZON, top_pct=top_pct,
                                    rebalance_freq=rf, cost=COST, mode=mode,
                                    buffer=buffer).ls_stats
    return bt


def row(name, pred_df, bt, note=""):
    s = ic_block(pred_df)
    return {
        "iter": name, "ic": s.get("ic_mean"), "icir": s.get("icir"),
        "pos": s.get("ic_positive_rate"),
        "net": bt.get("total_return"), "sharpe": bt.get("sharpe"),
        "mdd": bt.get("max_drawdown"), "win": bt.get("win_rate"),
        "turnover": bt.get("avg_turnover"), "buckets": bt.get("n_buckets"),
        "note": note,
    }


def main() -> int:
    global PANEL
    import argparse
    ap = argparse.ArgumentParser(description="P1 执行/组合层 10 次迭代（rf 可调）")
    ap.add_argument("--rf", type=int, default=15,
                    help="rebalance_freq（默认 15=原错配口径；阶段19后改 10 做正确口径重跑）")
    args = ap.parse_args()
    pred, base, PANEL = load()
    print(f"[load] t16 2026 预测 {len(pred):,} 行 | {pred['symbol'].nunique()} 只")
    print(f"[rf] rebalance_freq = {args.rf}")

    rows = []

    # I1 基线（非重叠桶，top10%，多空）
    rows.append(row(f"I1 基线(非重叠桶 rf{args.rf} top10%)", pred,
                    bt_bucket(pred, rf=args.rf), "脚本10/12 原口径复现"))

    # I2 连续缓冲（buffer=0.02）→ 砍换手
    rows.append(row("I2 连续缓冲 buffer=0.02", pred,
                    bt_cont(pred, rf=args.rf, buffer=0.02), "头号近乎零成本改进"))

    # I3 连续缓冲更大（buffer=0.05）
    rows.append(row("I3 连续缓冲 buffer=0.05", pred,
                    bt_cont(pred, rf=args.rf, buffer=0.05), "更低换手"))

    # I4 连续 + 调仓频率 30（持有更久）
    rows.append(row("I4 连续 rf=30 buffer=0.02", pred,
                    bt_cont(pred, rf=30, buffer=0.02), "更少换仓（低频对照）"))

    # I5 连续 + top5% 集中
    rows.append(row("I5 连续 top5% buffer=0.02", pred,
                    bt_cont(pred, rf=args.rf, top_pct=0.05, buffer=0.02), "高置信集中"))

    # I6 连续 + top20%
    rows.append(row("I6 连续 top20% buffer=0.02", pred,
                    bt_cont(pred, rf=args.rf, top_pct=0.20, buffer=0.02), "更宽"))

    # I7 集成混合（0.7 GRU + 0.3 baseline）→ 改变排序，更稳
    m = pred.merge(base, on=["date", "symbol"], how="inner")
    m = m.dropna(subset=["pred", "pred_base", "y_excess"]).reset_index(drop=True)
    blend = m.copy()
    blend["pred"] = 0.7 * m["pred"] + 0.3 * m["pred_base"]
    rows.append(row("I7 集成(0.7GRU+0.3base)", blend,
                    bt_cont(blend, rf=args.rf, buffer=0.02), "稳健排序"))

    # I8 置信过滤（去掉 |pred| 弱半，降噪交易）
    med = m["pred"].abs().median()
    strong = m[m["pred"].abs() >= med].copy()
    rows.append(row("I8 置信过滤(去弱半)", strong,
                    bt_cont(strong, rf=args.rf, buffer=0.02), "降噪"))

    # I9 仅做多（连续）
    rows.append(row("I9 仅做多(连续 top10%)", pred,
                    bt_cont(pred, rf=args.rf, mode="long_only"), "免空方成本/跌停风险"))

    # I10 组合最优：集成 + 置信过滤 + 连续 buffer=0.03 + rf=30 + top5%
    combo = strong.copy()
    combo["pred"] = 0.7 * strong["pred"] + 0.3 * strong["pred_base"]
    rows.append(row("I10 组合(集成+置信+连续rf30+top5%)", combo,
                    bt_cont(combo, top_pct=0.05, rf=30, buffer=0.03),
                    "累积最优配置（低频对照）"))

    df = pd.DataFrame(rows)
    out = PROCESSED / f"report_improve_10x_rf{args.rf}.csv"
    df.to_csv(out, index=False, float_format="%.4f")

    print("\n## P1 GRU 锐评后 10 次迭代（严格 2026 hold-out，t16 预测）\n")
    print("| 迭代 | IC | ICIR | 正占比 | 净收益 | 夏普 | 最大回撤 | 胜率 | 换手/期 | 桶数 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        print("| {n} | {ic:.4f} | {icir:.4f} | {pos:.3f} | {net:.2%} | {sh:.2f} | "
              "{mdd:.2%} | {win:.2%} | {to} | {nb} |".format(
                  n=r["iter"], ic=r["ic"], icir=r["icir"], pos=r["pos"],
                  net=r["net"], sh=r["sharpe"], mdd=r["mdd"], win=r["win"],
                  to=(f"{r['turnover']:.1%}" if r["turnover"] is not None else "-"),
                  nb=r["buckets"]))
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
