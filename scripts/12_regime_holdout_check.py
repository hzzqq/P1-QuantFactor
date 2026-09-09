"""阶段 12：regime 门控融合的【严格 2026 hold-out】验证。

目的：阶段 11 的 regime 门控增益（+25.52% vs 纯 GRU +21.94%）是在 2022–2026 全窗口
测的，而波动率阈值也是在该窗口上选的，存在轻微「同窗优化」偏差。本脚本用
**严格 2026 样本外**验证它是否真的带来增益：

  - GRU 预测 = t02 网格的 2026 预测（训≤2024、测仅 2026，模型从未见过 2026）
  - 波动率阈值 = 仅用 ≤2024-12-31 的历史波动率中位数（训练期数据，无前视）
  - 门控：calm（低波动）→ 纯 GRU；turbulent（高波动）→ 需基线 sign 确认
  - 对比：纯 GRU(z_gru) vs regime 门控，在 2026 上算 IC/ICIR 与 rf=15 成本回测

用法：
    python scripts/12_regime_holdout_check.py
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
from src.backtest import run_backtest, DEFAULT_COST

PROCESSED = paths.DATA / "P1" / "processed"
HORIZON = 10
TOP_PCT = 0.1
REBAL_FREQ = 15
TRAIN_END = pd.Timestamp("2024-12-31")
TEST_YEAR = 2026


def cross_sectional_z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0.0, np.nan)
    return ((df[col] - mean) / std).fillna(0.0)


def main() -> int:
    # 1) t02 网格的 2026 GRU 预测（严格 hold-out）
    gru = pd.read_parquet(
        PROCESSED / "pred_gru_tune_t02_seq20_h64_lr3e-3_h10.parquet")
    gru["date"] = pd.to_datetime(gru["date"])
    gru = gru[gru["date"].dt.year == TEST_YEAR].copy()
    print(f"[GRU t02 2026] {len(gru):,} 行 | {gru['symbol'].nunique()} 只 | "
          f"{gru['date'].min().date()}~{gru['date'].max().date()}")

    # 2) 基线 2026 预测（作确认信号）
    base = pd.read_parquet(PROCESSED / "pred_baseline_h10.parquet")
    base["date"] = pd.to_datetime(base["date"])
    base = base[base["date"].dt.year == TEST_YEAR].copy()

    m = gru[["date", "symbol", "pred", "y_excess"]].merge(
        base[["date", "symbol", "pred"]], on=["date", "symbol"],
        suffixes=("_gru", "_base"))
    m = m.dropna(subset=["pred_base", "pred_gru", "y_excess"]).reset_index(drop=True)
    m["z_gru"] = cross_sectional_z(m, "pred_gru")
    m["z_base"] = cross_sectional_z(m, "pred_base")
    print(f"[重叠 2026] {len(m):,} 行 | {m['symbol'].nunique()} 只")

    # 3) regime 阈值：严格只用 ≤2024-12-31 的历史波动率中位数
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel = panel.sort_values(["symbol", "date"]).copy()
    panel["prev"] = panel.groupby("symbol")["close"].shift(1)
    panel["ret"] = panel["close"] / panel["prev"] - 1.0
    mr = panel.groupby("date")["ret"].mean().sort_index()
    vol_all = mr.rolling(20).std()
    vol_train = vol_all[vol_all.index <= TRAIN_END]
    thr = vol_train.median()
    print(f"[regime 阈值] 训练期 vol 中位数 = {thr:.4f}（仅≤2024数据）")

    vol_2026 = vol_all[vol_all.index.year == TEST_YEAR]
    m["turbulent"] = m["date"].map(vol_2026 > thr).fillna(False).astype(bool)
    print(f"[2026 regime] turbulent 日占比 = {100.0*m['turbulent'].mean():.1f}%")

    # 4) B 门控：calm=纯GRU；turbulent=需基线 sign 确认
    m["f_regime"] = np.where(
        m["turbulent"],
        np.where(np.sign(m["z_gru"]) == np.sign(m["z_base"]), m["z_gru"], 0.0),
        m["z_gru"])

    # 5) 评估：纯 GRU vs regime 门控（2026 严格 hold-out）
    panel_bt = panel
    rows = []
    for tag, col in (("纯GRU(2026 hold-out)", "z_gru"),
                     ("B-regime门控(2026 hold-out)", "f_regime")):
        s = metrics.summarize(m.assign(pred=m[col]), "pred", "y_excess")
        sub = m[["date", "symbol", col]].rename(columns={col: "pred"})
        bt = run_backtest(sub, panel_bt, horizon=HORIZON, top_pct=TOP_PCT,
                          rebalance_freq=REBAL_FREQ, cost=DEFAULT_COST,
                          mode="long_short").ls_stats
        coverage = float(sub[sub["pred"] != 0].groupby("date").size().mean())
        rows.append({"strategy": tag, "ic": s["ic_mean"], "icir": s["icir"],
                     "pos": s["ic_positive_rate"], "net": bt.get("total_return"),
                     "sharpe": bt.get("sharpe"), "mdd": bt.get("max_drawdown"),
                     "win": bt.get("win_rate"), "buckets": bt.get("n_buckets"),
                     "coverage": coverage})

    df = pd.DataFrame(rows)
    print("\n## 严格 2026 hold-out：regime 门控验证（rf=15，多空 top10% h10）\n")
    print("| 策略 | IC | ICIR | 正占比 | 净收益 | 夏普 | 最大回撤 | 胜率 | 桶数 | 日均覆盖 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        print("| {s} | {ic:.4f} | {icir:.4f} | {pos:.3f} | {net:.2%} | {sh:.2f} | "
              "{mdd:.2%} | {win:.2%} | {nb} | {cov:.0f} |".format(
                  s=r["strategy"], ic=r["ic"], icir=r["icir"], pos=r["pos"],
                  net=r["net"], sh=r["sharpe"], mdd=r["mdd"], win=r["win"],
                  nb=r["buckets"], cov=r["coverage"]))

    # 落盘融合预测（若门控在严格 hold-out 上净收益更高）
    gru_net = next(r["net"] for r in rows if r["strategy"].startswith("纯GRU"))
    reg_net = next(r["net"] for r in rows if r["strategy"].startswith("B"))
    verdict = "REGIME 门控在严格 2026 hold-out 上净收益更高 → 增益真实" if (
        reg_net or -9) > (gru_net or -9) else \
        "REGIME 门控在严格 2026 hold-out 上净收益未超纯 GRU → 增益可能部分来自同窗优化"
    print(f"\n结论：{verdict}")

    out = PROCESSED / "report_regime_holdout_2026.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
