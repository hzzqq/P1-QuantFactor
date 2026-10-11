"""T-265 B 路：二路融合权重网格对照（生产口径回测）。

背景（zz_ensemble_multi_compare.py）：三路对照显示 Transformer 替换路（w=0.25）
净收益劣于现役 v39+GRU(w_gru=0.25)。本脚本不换模型，做**权重维度**的诚实扫描：
  1) v39+GRU      ：w_gru ∈ {0.05..0.40}（现役 0.25 在网格内，作锚点）
  2) v39+Transformer：w_tfm ∈ {0.05..0.40}
  3) 纯二路 GRU+Transformer 秩平均：w_gru ∈ {0.3, 0.5, 0.7}（无 v39 基座）
同参生产口径（rf=15 / top0.1 / buffer0.1 / 长空 / DEFAULT_COST）。
诚实红线：若最优网格点对现役的提升属噪声级（夏普差 <0.05 或收益差 <5%），如实标注
「无稳定优势」，不构成切换生产依据；结论由老板拍板。
产出：report_ensemble_pairwise.json。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shared import paths  # noqa: E402
from src.backtest import run_backtest_continuous, DEFAULT_COST  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
RF, TOP_PCT, BUFFER = 15, 0.1, 0.1
GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
PAIR_GRID = (0.3, 0.5, 0.7)


def _z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
            ).fillna(0.0).clip(-3, 3)


def main() -> int:
    t0 = time.time()
    v39 = pd.read_parquet(PROCESSED / "pred_baseline_h10_v39.parquet")
    gru = pd.read_parquet(PROCESSED / "pred_gru_h10.parquet")
    tfm = pd.read_parquet(PROCESSED / "pred_transformer_h10.parquet")
    for d in (v39, gru, tfm):
        d["date"] = pd.to_datetime(d["date"])
        d.drop_duplicates(subset=["date", "symbol"], keep="last", inplace=True)
    v39["z_v39"] = _z(v39, "pred")
    gru["z_gru"] = _z(gru, "pred")
    tfm["z_tfm"] = _z(tfm, "pred")

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    kw = dict(horizon=10, top_pct=TOP_PCT, rebalance_freq=RF, mode="long_short")

    def build(w_g: float, w_t: float, use_v39: bool = True) -> pd.DataFrame:
        if use_v39:
            m = v39[["date", "symbol", "z_v39"]]
            m = m.merge(gru[["date", "symbol", "z_gru"]], on=["date", "symbol"], how="left")
            m = m.merge(tfm[["date", "symbol", "z_tfm"]], on=["date", "symbol"], how="left")
            m["pred"] = ((1 - w_g - w_t) * m["z_v39"]
                         + w_g * m["z_gru"].fillna(0.0) + w_t * m["z_tfm"].fillna(0.0))
        else:
            m = gru[["date", "symbol", "z_gru"]].merge(
                tfm[["date", "symbol", "z_tfm"]], on=["date", "symbol"], how="inner")
            m["pred"] = w_g * m["z_gru"] + (1 - w_g) * m["z_tfm"]
        return m[["date", "symbol", "pred"]]

    configs = []
    for w in GRID:
        configs.append((f"v39+GRU(w={w:.2f})", build(w, 0.0)))
    for w in GRID:
        configs.append((f"v39+Transformer(w={w:.2f})", build(0.0, w)))
    for w in PAIR_GRID:
        configs.append((f"GRU+TF二路(gru={w:.1f})", build(w, 0.0, use_v39=False)))

    out = {"config": {"rf": RF, "top_pct": TOP_PCT, "buffer": BUFFER,
                      "grid_v39": list(GRID), "grid_pair": list(PAIR_GRID)},
           "compare": []}
    print(f"共 {len(configs)} 组配置，回测中…", flush=True)
    for name, pred in configs:
        net = run_backtest_continuous(pred, panel, **kw, cost=DEFAULT_COST, buffer=BUFFER)
        n = net.ls_stats or {}
        row = {"name": name,
               "net_total_return": round(float(n.get("total_return", float("nan"))), 4),
               "net_sharpe": round(float(n.get("sharpe", float("nan"))), 3),
               "net_max_dd": round(float(n.get("max_drawdown", float("nan"))), 4),
               "net_win_rate": round(float(n.get("win_rate", float("nan"))), 4)}
        out["compare"].append(row)
        print(f"{name:<26} 净累计 {row['net_total_return']:+.2%} | 夏普 {row['net_sharpe']:+.3f} "
              f"| 回撤 {row['net_max_dd']:.2%} | 胜率 {row['net_win_rate']:.3f}", flush=True)

    base = next(r for r in out["compare"] if r["name"] == "v39+GRU(w=0.25)")
    best = max(out["compare"], key=lambda r: r["net_sharpe"])
    out["baseline_current"] = base
    out["best_grid"] = best
    d_sharpe = round(best["net_sharpe"] - base["net_sharpe"], 3)
    d_ret = round(best["net_total_return"] - base["net_total_return"], 4)
    stable = (d_sharpe >= 0.05) and (d_ret >= 0.05)
    out["delta_vs_current"] = {"sharpe": d_sharpe, "total_return": d_ret}
    out["conclusion"] = (
        f"网格最优={best['name']}（夏普Δ={d_sharpe:+.3f}, 收益Δ={d_ret:+.2%}）；"
        + ("达到稳定优势阈值（Δ夏普≥0.05 且 Δ收益≥5%），可提请老板拍板切换"
           if stable else "未达稳定优势阈值（噪声级/更差），维持现役 v39+GRU(w=0.25) 并留痕"))
    with open(PROCESSED / "report_ensemble_pairwise.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结论:", out["conclusion"])
    print("报告: report_ensemble_pairwise.json 耗时", round(time.time() - t0, 1), "秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
