"""阶段 50 - alpha 再生：校验 v41 baseline 预测是否兑现估值因子增量（隔离、非破坏）。

比对 pred_baseline_h10_v41（41维，含 ep/btop）vs 现役 pred_baseline_h10_v39（39维）
在 FULL/2024+/2026/recent59td 的 IC（daily_ic vs y_excess）。若 v41 在 2024+/2026
窗口 IC 显著高于 v39 → 估值因子增量兑现，可进入"取代生产"评估；否则如实记录。

仅读两个 pred 文件做对比，不写生产文件、不碰 v39。
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

from src.eval.metrics import daily_ic, ic_summary  # noqa: E402

PROC = ROOT / "data" / "P1" / "processed"
RECENT_TD = 59


def _window_mask(dates: pd.Series, kind: str) -> pd.Series:
    if kind == "FULL":
        return pd.Series(True, index=dates.index)
    if kind == "2024+":
        return dates >= pd.Timestamp("2024-01-01")
    if kind == "2026":
        return dates >= pd.Timestamp("2026-01-01")
    if kind == "recent59td":
        uniq = np.sort(pd.unique(dates))
        cutoff = uniq[-RECENT_TD] if len(uniq) > RECENT_TD else uniq[0]
        return dates >= cutoff
    raise ValueError(kind)


def _ic_by_window(df: pd.DataFrame, col: str = "pred") -> dict:
    dates = pd.to_datetime(df["date"])
    out = {}
    for w in ["FULL", "2024+", "2026", "recent59td"]:
        m = _window_mask(dates, w)
        ic = daily_ic(df[m], pred_col=col, y_col="y_excess")
        s = ic_summary(ic)
        out[w] = {"ic_mean": s["ic_mean"], "icir": s["icir"],
                  "ic_positive_rate": s["ic_positive_rate"], "t_stat": s["t_stat"],
                  "n_days": s["n_days"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v41-path", type=str, default="pred_baseline_h10_v41.parquet")
    ap.add_argument("--v39-path", type=str,
                   default="pred_baseline_h10_v39_hgb.parquet",
                   help="v39 同模型对照（HGB 重训控制，非现役 lightgbm 生产 pred）")
    args = ap.parse_args()
    t0 = time.time()
    v41 = pd.read_parquet(PROC / args.v41_path)
    v39 = pd.read_parquet(PROC / args.v39_path)
    print(f"[39] v41({args.v41_path}) {len(v41):,} 行 年份 {sorted(v41['year'].unique())} | "
          f"v39({args.v39_path}) {len(v39):,} 行 年份 {sorted(v39['year'].unique())}", flush=True)
    ic41 = _ic_by_window(v41)
    ic39 = _ic_by_window(v39)
    print("\n=== IC 对比（v41 vs v39，按窗口）===")
    print(f"  {'窗口':10s} {'v41_IC':>10s} {'v39_IC':>10s} {'ΔIC':>10s} "
          f"| v41_t      v39_t")
    report = {"v41": ic41, "v39": ic39, "delta": {}, "verdict": None}
    for w in ["FULL", "2024+", "2026", "recent59td"]:
        d = ic41[w]["ic_mean"] - ic39[w]["ic_mean"]
        report["delta"][w] = d
        print(f"  {w:10s} {ic41[w]['ic_mean']:+10.4f} {ic39[w]['ic_mean']:+10.4f} "
              f"{d:+10.4f} | t{ic41[w]['t_stat']:+8.2f}  t{ic39[w]['t_stat']:+8.2f}")
    # 判定：2024+/2026 任一段 ΔIC 显著正则估值增量兑现
    sig = (ic41["2024+"]["ic_mean"] > ic39["2024+"]["ic_mean"] + 0.003 or
           ic41["2026"]["ic_mean"] > ic39["2026"]["ic_mean"] + 0.003)
    report["verdict"] = (
        "v41(v41 含估值因子) 在 2024+/2026 窗口 IC 高于 v39 → 估值增量兑现，"
        "建议进入取代生产评估" if sig else
        "v41 在关键窗口未显著优于 v39 → 估值增量未兑现，维持 v39 生产")
    json.dump(report, open(PROC / "report_v41_validation.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print(f"\n[39] verdict: {report['verdict']}")
    print(f"[39] 完成 {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
