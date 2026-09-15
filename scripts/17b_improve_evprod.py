"""阶段 17 衍生：在【生产 EV 信号】上重跑 10 个执行/组合层杠杆（rf=10 正确口径）。

背景：
  09-04 的 `17_improve_iterate.py` 已在 t16 预测（train≤2024、测仅 2026）上把 10 个杠杆跑过，
  结论：I1（非重叠桶 rf=10）是唯一正 P&L 配置，连续/集成/置信/仅做多（I2–I9）全负。
  但那份用的是 **t16 调参 GRU**；现在生产信号已升级为 **input_norm EV 权重**
  （`pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet`，IC=0.0941，M1 全量 +16.83%）。

本脚本在【生产 EV 预测】上复刻同一组 10 个杠杆，回答：「在真正部署的信号上，
执行层配置结论是否稳健？I1 是否仍最优？」—— 这是部署前的执行层就绪性检查。

⚠️ 口径声明：生产 EV 预测本身**仅覆盖 2026**（模型 proper test 窗口），故本报告的
**绝对净收益/夏普是「单年子集」数字，偏高**（见 2026-09-15 立的「子集乐观」红线）。
**决策依据是「杠杆相对排序」而非绝对数值**：所有杠杆看到同一 2026 窗口，排序可比。

用法：
    python scripts/17b_improve_evprod.py            # rf=10（默认，正确口径）
    python scripts/17b_improve_evprod.py --rf 15    # 错配口径对照（应更差，证方法学）
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
from src.eval import metrics
from src.backtest import engine as BE

PROCESSED = paths.DATA / "P1" / "processed"
HORIZON = 10
TOP_PCT = 0.1
COST = BE.DEFAULT_COST
# 生产 EV 预测文件名（input_norm 权重，IC=0.0941）
EV_PRED_FILE = "pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet"


def load():
    pred = pd.read_parquet(PROCESSED / EV_PRED_FILE)
    pred["date"] = pd.to_datetime(pred["date"])
    # 该文件本身仅含 2026，这里显式过滤以自文档化口径
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
    ap = argparse.ArgumentParser(description="P1 执行/组合层 10 次迭代（生产 EV 信号，rf 可调）")
    ap.add_argument("--rf", type=int, default=10,
                    help="rebalance_freq（默认 10=正确口径，与 horizon 对齐）")
    args = ap.parse_args()
    pred, base, PANEL = load()
    print(f"[load] 生产EV 2026 预测 {len(pred):,} 行 | {pred['symbol'].nunique()} 只")
    print(f"[rf] rebalance_freq = {args.rf}")

    rows = []
    rows.append(row(f"I1 基线(非重叠桶 rf{args.rf} top10%)", pred,
                    bt_bucket(pred, rf=args.rf), "脚本10/12 原口径复现"))
    rows.append(row("I2 连续缓冲 buffer=0.02", pred,
                    bt_cont(pred, rf=args.rf, buffer=0.02), "头号近乎零成本改进"))
    rows.append(row("I3 连续缓冲 buffer=0.05", pred,
                    bt_cont(pred, rf=args.rf, buffer=0.05), "更低换手"))
    rows.append(row("I4 连续 rf=30 buffer=0.02", pred,
                    bt_cont(pred, rf=30, buffer=0.02), "更少换仓（低频对照）"))
    rows.append(row("I5 连续 top5% buffer=0.02", pred,
                    bt_cont(pred, rf=args.rf, top_pct=0.05, buffer=0.02), "高置信集中"))
    rows.append(row("I6 连续 top20% buffer=0.02", pred,
                    bt_cont(pred, rf=args.rf, top_pct=0.20, buffer=0.02), "更宽"))
    m = pred.merge(base, on=["date", "symbol"], how="inner")
    m = m.dropna(subset=["pred", "pred_base", "y_excess"]).reset_index(drop=True)
    blend = m.copy()
    blend["pred"] = 0.7 * m["pred"] + 0.3 * m["pred_base"]
    rows.append(row("I7 集成(0.7EV+0.3base)", blend,
                    bt_cont(blend, rf=args.rf, buffer=0.02), "稳健排序"))
    med = m["pred"].abs().median()
    strong = m[m["pred"].abs() >= med].copy()
    rows.append(row("I8 置信过滤(去弱半)", strong,
                    bt_cont(strong, rf=args.rf, buffer=0.02), "降噪"))
    rows.append(row("I9 仅做多(连续 top10%)", pred,
                    bt_cont(pred, rf=args.rf, mode="long_only"), "免空方成本/跌停风险"))
    combo = strong.copy()
    combo["pred"] = 0.7 * strong["pred"] + 0.3 * strong["pred_base"]
    rows.append(row("I10 组合(集成+置信+连续rf30+top5%)", combo,
                    bt_cont(combo, top_pct=0.05, rf=30, buffer=0.03),
                    "累积最优配置（低频对照）"))

    df = pd.DataFrame(rows)
    out = PROCESSED / f"report_improve_10x_rf{args.rf}_evprod.csv"
    df.to_csv(out, index=False, float_format="%.4f")

    print("\n## P1 生产 EV 信号 · 10 次执行/组合层迭代（严格 2026 hold-out，⚠️单年子集口径）\n")
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
    print("\n⚠️ 注意：上表净收益/夏普为 2026 单年子集数字（信号仅覆盖 2026），")
    print("   决策请看「杠杆相对排序」—— I1 是否仍唯一正 P&L 配置。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
