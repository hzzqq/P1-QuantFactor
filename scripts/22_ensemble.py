"""阶段 24 - 信号融合 Ensemble：EV GRU(input_norm) × LightGBM baseline。

背景：
    阶段 23 已落地两个稳定信号：
      - EV GRU(input_norm) 生产信号：pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet
        （仅覆盖 2026，模型训练数据截至 2025-12，只能合法预测 2026）
      - LightGBM baseline（IC 早停 + 2 年验证修复后）：pred_baseline_h10.parquet
        （覆盖 2019–2026）
    两者唯一共同年份 = 2026。故集成评估在 2026 交集上做。

方法：
    - 对 (date,symbol) 内连接，取两模型预测。
    - 逐日横截面 z-score（消除量纲/分布差异）后等权平均 → ensemble 分数。
    - 评估：IC（全样本 Spearman）、月度 IC→ICIR、long-short 回测（top10%/rf=15/成本敏感）
      对 gru_only / baseline_only / ensemble 三路对比；并算两模型预测相关性（互补性）。

产物：
    data/P1/processed/pred_ensemble_h10.parquet
    data/P1/processed/report_ensemble_h10.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths
from src.eval import metrics
from src.backtest import run_backtest, DEFAULT_COST

logger = None
PROCESSED = paths.DATA / "P1" / "processed"
GRU = PROCESSED / "pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet"
BASE = PROCESSED / "pred_baseline_h10.parquet"


def zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    """逐日横截面 z-score（按 date 分组标准化 pred）。"""
    return df.groupby("date")[col].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12))


def main() -> int:
    t0 = time.time()
    gru = pd.read_parquet(GRU)
    base = pd.read_parquet(BASE)
    print(f"GRU:  {len(gru):,} 行 | {gru.symbol.nunique()} 只 | {gru.date.min().date()}~{gru.date.max().date()}")
    print(f"BASE: {len(base):,} 行 | {base.symbol.nunique()} 只 | {base.date.min().date()}~{base.date.max().date()}")

    # 仅 2026 交集
    g = gru[gru.year == 2026][["date", "symbol", "pred", "y_excess"]].rename(
        columns={"pred": "gru", "y_excess": "y"})
    b = base[base.year == 2026][["date", "symbol", "pred", "y_excess"]].rename(
        columns={"pred": "base", "y_excess": "y"})
    m = g.merge(b, on=["date", "symbol"], how="inner")
    print(f"2026 交集：{len(m):,} 行 | {m.symbol.nunique()} 只 | {m.date.min().date()}~{m.date.max().date()}")
    # 标签一致性校验
    ydiff = (m["y_x"] - m["y_y"]).abs().max()
    print(f"标签 y_excess 两源最大差：{ydiff:.2e}（应≈0）")
    m = m.rename(columns={"y_x": "y"})[["date", "symbol", "gru", "base", "y"]]

    # 逐日横截面 z-score 后等权平均
    m["gru_z"] = zscore_by_date(m, "gru")
    m["base_z"] = zscore_by_date(m, "base")
    m["ens"] = (m["gru_z"] + m["base_z"]) / 2.0

    # 互补性：两模型预测相关性
    corr = float(np.corrcoef(m["gru_z"], m["base_z"])[0, 1])
    print(f"\n两模型(标准化后)预测相关性 corr = {corr:.4f}（越低越互补）")

    # 月度 IC → ICIR
    def monthly_ic(s: pd.Series):
        m2 = m.assign(s=s)
        g2 = m2.groupby(m2["date"].dt.to_period("M"))
        ics = [spearmanr(grp["y"], grp["s"]).correlation for _, grp in g2 if len(grp) > 5]
        ics = pd.Series([x for x in ics if x == x])
        return ics.mean(), (ics.std(ddof=0) and ics.mean() / ics.std(ddof=0) or float("nan")), ics

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    rows = []
    for name, col in [("gru_only", "gru"), ("baseline_only", "base"), ("ensemble", "ens")]:
        s = metrics.summarize(m[["date", "symbol", col, "y"]].rename(columns={col: "pred"}), "pred", "y")
        mic, micir, _ = monthly_ic(m[col])
        bt = run_backtest(m[["date", "symbol", col]].rename(columns={col: "pred"}), panel,
                          horizon=10, top_pct=0.1, rebalance_freq=15, cost=DEFAULT_COST,
                          mode="long_short").ls_stats
        rows.append({
            "model": name, "ic_mean": round(s["ic_mean"], 4),
            "ic_positive_rate": round(s["ic_positive_rate"], 4),
            "icir_monthly": round(micir, 4), "ic_monthly_mean": round(mic, 4),
            "ls_net": round(bt.get("total_return") or 0, 4),
            "ls_sharpe": round(bt.get("sharpe") or 0, 4),
            "ls_mdd": round(bt.get("max_drawdown") or 0, 4),
            "ls_win": round(bt.get("win_rate") or 0, 4),
        })
        print(f"[{name:13s}] IC={s['ic_mean']:.4f} ICIRm={micir:.4f} net={bt.get('total_return'):.4f} "
              f"sharpe={bt.get('sharpe'):.4f} mdd={bt.get('max_drawdown'):.4f} win={bt.get('win_rate'):.4f}")

    # 落盘 ensemble 信号
    out = m[["date", "symbol", "ens", "y", "gru", "base"]].rename(columns={"ens": "pred", "y": "y_excess"})
    out["year"] = 2026
    out.to_parquet(PROCESSED / "pred_ensemble_h10.parquet", index=False)
    rep = {"coverage": "2026 intersection", "n_rows": len(m), "n_symbols": int(m.symbol.nunique()),
           "pred_corr_gru_base": round(corr, 4),
           "models": rows, "elapsed_sec": round(time.time() - t0, 1)}
    json.dump(rep, open(PROCESSED / "report_ensemble_h10.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print(f"\n已落盘 pred_ensemble_h10.parquet + report_ensemble_h10.json（耗时 {time.time()-t0:.1f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
