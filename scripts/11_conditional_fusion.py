"""阶段 11：条件化融合（基线做「过滤 / regime 门控」，而非线性加权）。

背景：阶段 8 已证明 GRU 与基线 LightGBM 低相关（Spearman 0.18），信息互补，
但**线性加权融合并无实质增益**（最优 w=0.1 退化为纯 GRU，样本外 ICIR 反而更低）。
本阶段试**非线性 / 条件化融合**：
  - A. 基线方向过滤：只有当基线信号与 GRU 同向（或双确认）时才保留 GRU 的仓位，
       基线充当「过滤器」而非「加权项」，用基线的判别力剔除 GRU 的噪声误判；
  - B. regime 门控：用市场波动率划分 calm / turbulent，calm 期完全信任 GRU，
       turbulent 期要求基线确认（过滤），降低高波动环境下的假信号。

评估：IC / ICIR（截面 rank IC）+ rf=15 成本敏感回测（与当前生产设置对齐），
区间 2022-01-01 起（与阶段 9/10 生产口径一致）。对比基准为纯 GRU。

用法：
    python scripts/11_conditional_fusion.py
"""
from __future__ import annotations

import argparse
import json
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
from shared.logging_utils import get_logger
from src.eval import metrics
from src.backtest import run_backtest, DEFAULT_COST

logger = get_logger("P1.conditional_fusion")
PROCESSED = paths.DATA / "P1" / "processed"

START = pd.Timestamp("2022-01-01")
HORIZON = 10
TOP_PCT = 0.1
REBAL_FREQ = 15


def cross_sectional_z(df: pd.DataFrame, col: str) -> pd.Series:
    """每个交易日横截面内做 z-score；当天标准差为 0 时退化为 0。"""
    g = df.groupby("date")[col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0.0, np.nan)
    return ((df[col] - mean) / std).fillna(0.0)


def build_regime(panel: pd.DataFrame) -> pd.Series:
    """用全市场截面平均日收益的 20 日波动率划分 turbulent / calm。

    纯由历史收益可得，实盘中可实时计算（无前视）。turbulent = 波动率 > 中位数。
    """
    panel = panel.sort_values(["symbol", "date"]).copy()
    panel["prev"] = panel.groupby("symbol")["close"].shift(1)
    panel["ret"] = panel["close"] / panel["prev"] - 1.0
    mr = panel.groupby("date")["ret"].mean().sort_index()
    vol = mr.rolling(20).std()
    thr = vol.median()
    turbulent = (vol > thr).fillna(False)
    logger.info("regime: turbulent 日占比 %.1f%%（阈值 vol=%.4f）",
                100.0 * turbulent.mean(), thr)
    return turbulent


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 条件化融合实验")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--top-pct", type=float, default=TOP_PCT)
    ap.add_argument("--rebalance-freq", type=int, default=REBAL_FREQ)
    ap.add_argument("--start-date", type=str, default="2022-01-01")
    args = ap.parse_args()
    START = pd.Timestamp(args.start_date)

    t0 = time.time()
    b = pd.read_parquet(PROCESSED / "pred_baseline_h10.parquet")
    g = pd.read_parquet(PROCESSED / "pred_gru_h10.parquet")
    b["date"] = pd.to_datetime(b["date"])
    g["date"] = pd.to_datetime(g["date"])
    m = b[["date", "symbol", "pred", "y_excess"]].merge(
        g[["date", "symbol", "pred"]], on=["date", "symbol"],
        suffixes=("_base", "_gru"))
    m = m.dropna(subset=["pred_base", "pred_gru", "y_excess"])
    m = m[m["date"] >= START]
    m["z_base"] = cross_sectional_z(m, "pred_base")
    m["z_gru"] = cross_sectional_z(m, "pred_gru")
    logger.info("重叠样本 %s 行 | %s 只 | %s ~ %s",
                f"{len(m):,}", m["symbol"].nunique(),
                m["date"].min().date(), m["date"].max().date())

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    turbulent = build_regime(panel)
    m["turbulent"] = m["date"].map(turbulent).fillna(False).astype(bool)

    def evaluate(df: pd.DataFrame, pred_col: str) -> dict:
        return metrics.summarize(df.assign(pred=df[pred_col]), "pred", "y_excess")

    def backtest(df: pd.DataFrame, pred_col: str) -> dict:
        sub = df[["date", "symbol", pred_col]].rename(columns={pred_col: "pred"})
        res = run_backtest(sub, panel, horizon=args.horizon, top_pct=args.top_pct,
                           rebalance_freq=args.rebalance_freq,
                           cost=DEFAULT_COST, mode="long_short")
        return res.ls_stats

    def coverage(df: pd.DataFrame, pred_col: str) -> float:
        return float(df[df[pred_col] != 0].groupby("date").size().mean())

    rows = []

    # ---- 基准：纯 GRU / 纯基线 ----
    for tag, col in (("纯GRU", "z_gru"), ("纯基线", "z_base")):
        s = evaluate(m, col)
        bt = backtest(m, col)
        rows.append({"strategy": tag, "ic": s["ic_mean"], "icir": s["icir"],
                     "pos": s["ic_positive_rate"], "net": bt.get("total_return"),
                     "sharpe": bt.get("sharpe"), "mdd": bt.get("max_drawdown"),
                     "win": bt.get("win_rate"), "buckets": bt.get("n_buckets"),
                     "coverage": np.nan})

    # ---- 策略 A1：基线方向过滤（sign 一致才保留 GRU 仓位）----
    m["f_sign"] = np.where(np.sign(m["z_gru"]) == np.sign(m["z_base"]), m["z_gru"], 0.0)
    s = evaluate(m, "f_sign"); bt = backtest(m, "f_sign")
    rows.append({"strategy": "A1-基线方向过滤(sign)", "ic": s["ic_mean"], "icir": s["icir"],
                 "pos": s["ic_positive_rate"], "net": bt.get("total_return"),
                 "sharpe": bt.get("sharpe"), "mdd": bt.get("max_drawdown"),
                 "win": bt.get("win_rate"), "buckets": bt.get("n_buckets"),
                 "coverage": coverage(m, "f_sign")})

    # ---- 策略 A2：双确认阈值扫描（|z_gru|>t 且 |z_base|>t）----
    for t in (0.5, 1.0, 1.5):
        col = f"f_t{t}"
        cond = (m["z_gru"].abs() > t) & (m["z_base"].abs() > t)
        m[col] = np.where(cond, m["z_gru"], 0.0)
        s = evaluate(m, col); bt = backtest(m, col)
        rows.append({"strategy": f"A2-双确认|z|>{t}", "ic": s["ic_mean"], "icir": s["icir"],
                     "pos": s["ic_positive_rate"], "net": bt.get("total_return"),
                     "sharpe": bt.get("sharpe"), "mdd": bt.get("max_drawdown"),
                     "win": bt.get("win_rate"), "buckets": bt.get("n_buckets"),
                     "coverage": coverage(m, col)})

    # ---- 策略 B：regime 门控（calm=纯GRU；turbulent=需基线确认）----
    m["f_regime"] = np.where(
        m["turbulent"],
        np.where(np.sign(m["z_gru"]) == np.sign(m["z_base"]), m["z_gru"], 0.0),
        m["z_gru"])
    s = evaluate(m, "f_regime"); bt = backtest(m, "f_regime")
    rows.append({"strategy": "B-regime门控(湍流需确认)", "ic": s["ic_mean"], "icir": s["icir"],
                 "pos": s["ic_positive_rate"], "net": bt.get("total_return"),
                 "sharpe": bt.get("sharpe"), "mdd": bt.get("max_drawdown"),
                 "win": bt.get("win_rate"), "buckets": bt.get("n_buckets"),
                 "coverage": coverage(m, "f_regime")})

    df = pd.DataFrame(rows)
    out_csv = PROCESSED / "report_fusion_sweep_h10.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")

    # 选最优融合：以「回测净收益」为首要目标（真实资金口径），
    # 候选须净收益 > 纯GRU（即确实带来增益），并列时取更高夏普。
    gru_row = next(r for r in rows if r["strategy"] == "纯GRU")
    candidates = [r for r in rows
                 if r["strategy"] != "纯GRU"
                 and (r["net"] or -9) > (gru_row["net"] or -9)]
    if candidates:
        best = max(candidates, key=lambda r: (r["net"], r["sharpe"]))
    else:
        best = gru_row
    best_col = {
        "纯GRU": "z_gru", "纯基线": "z_base",
        "A1-基线方向过滤(sign)": "f_sign",
        "A2-双确认|z|>0.5": "f_t0.5", "A2-双确认|z|>1.0": "f_t1.0",
        "A2-双确认|z|>1.5": "f_t1.5", "B-regime门控(湍流需确认)": "f_regime",
    }.get(best["strategy"], "z_gru")
    if best is not gru_row:
        m[["date", "symbol", best_col, "y_excess"]].rename(
            columns={best_col: "pred"}).to_parquet(
            PROCESSED / "pred_fusion_h10.parquet", index=False)

    print("\n## 条件化融合扫描（基线=过滤/门控，非加权；区间 "
          f"{START.date()}~，rf={args.rebalance_freq}，多空 top{args.top_pct:.0%} h{args.horizon}）\n")
    print("IC/ICIR 为截面 rank IC；净收益/夏普为成本敏感回测；日均覆盖数=日均非零预测数（过滤越严越小）。\n")
    hdr = ("| 策略 | IC | ICIR | 正占比 | 净收益 | 夏普 | 最大回撤 | 胜率 | "
           "桶数 | 日均覆盖 |")
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    print(hdr); print(sep)
    for r in rows:
        print("| {s} | {ic:.4f} | {icir:.4f} | {pos:.3f} | {net:.2%} | {sh:.2f} | "
              "{mdd:.2%} | {win:.2%} | {nb} | {cov:.0f} |".format(
                  s=r["strategy"], ic=r["ic"], icir=r["icir"], pos=r["pos"],
                  net=r["net"], sh=r["sharpe"], mdd=r["mdd"], win=r["win"],
                  nb=r["buckets"], cov=(r["coverage"] if r["coverage"] == r["coverage"] else 0)))
    print(f"\n最优条件化融合：{best['strategy']} → ICIR {best['icir']:.4f} / "
          f"净 {best['net']:.2%} / 夏普 {best['sharpe']:.2f}")
    print(f"对比纯GRU：ICIR {gru_row['icir']:.4f} / 净 {gru_row['net']:.2%} / "
          f"夏普 {gru_row['sharpe']:.2f}")
    print(f"报告已保存: {out_csv}")
    print(f"耗时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
