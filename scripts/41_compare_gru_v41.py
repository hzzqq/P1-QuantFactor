"""阶段 50 - B 生产化：GRU 生产模型 vs v41（含估值因子）IC 对照。

生产 GRU 信号来自 `pred_gru_h10.parquet`（38 维原始特征，与生产线上 ks 消费的 GRU 组件同源）。
v41 GRU 来自 `pred_gru_h10_v41.parquet`（同超参 seq_len=20/hidden=64/layers=2/epochs=15/stride=5，仅多 ep/btop 两维）。
两者架构/超参完全一致，仅特征集不同 → 公平隔离「估值因子对 GRU 的贡献」。

比 FULL/2024+/2026/recent59td 的 IC（daily_ic vs y_excess）。判定：v41 在 2024+/2026 显著更优
→ 估值增量在生产模型级也成立；否则如实报告。仅读不写、不动生产 pred/signal。
"""
from __future__ import annotations

import argparse
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
        if m.sum() == 0:
            out[w] = None
            continue
        ic = daily_ic(df[m], pred_col=col, y_col="y_excess")
        s = ic_summary(ic)
        out[w] = {"ic_mean": s["ic_mean"], "icir": s["icir"],
                  "ic_positive_rate": s["ic_positive_rate"], "t_stat": s["t_stat"],
                  "n_days": s["n_days"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-path", type=str, default="pred_gru_h10.parquet")
    ap.add_argument("--v41-path", type=str, default="pred_gru_h10_v41.parquet")
    args = ap.parse_args()
    t0 = time.time()
    prod = pd.read_parquet(PROC / args.prod_path)
    v41 = pd.read_parquet(PROC / args.v41_path)
    print(f"[41] 生产GRU {len(prod):,} 行 年份 {sorted(prod['year'].dropna().unique().astype(int))} | "
          f"v41GRU {len(v41):,} 行 年份 {sorted(v41['year'].dropna().unique().astype(int))}", flush=True)
    icp = _ic_by_window(prod)
    icv = _ic_by_window(v41)
    print("\n=== GRU IC 对比（生产 vs v41，按窗口）===")
    print(f"  {'窗口':10s} {'prod_IC':>10s} {'v41_IC':>10s} {'ΔIC':>10s} | prod_t    v41_t")
    report = {"prod": icp, "v41": icv, "delta": {}, "verdict": None}
    for w in ["FULL", "2024+", "2026", "recent59td"]:
        p, v = icp.get(w), icv.get(w)
        if p is None or v is None:
            print(f"  {w:10s}  {'—':>10s}  {'—':>10s}  {'—':>10s} | 窗口无数据")
            continue
        d = v["ic_mean"] - p["ic_mean"]
        report["delta"][w] = d
        print(f"  {w:10s} {p['ic_mean']:+10.4f} {v['ic_mean']:+10.4f} {d:+10.4f} | "
              f"t{p['t_stat']:+8.2f}  t{v['t_stat']:+8.2f}")
    sig = ((icv.get("2024+") and icp.get("2024+") and icv["2024+"]["ic_mean"] > icp["2024+"]["ic_mean"] + 0.003)
           or (icv.get("2026") and icp.get("2026") and icv["2026"]["ic_mean"] > icp["2026"]["ic_mean"] + 0.003))
    report["verdict"] = (
        "v41 GRU（含估值因子）在 2024+/2026 窗口 IC 显著高于生产 GRU → 估值增量在生产模型级成立，"
        "建议进入生产替换评估" if sig else
        "v41 GRU 在关键窗口未显著优于生产 GRU → 估值增量未在 GRU 级兑现，维持生产")
    print(f"\n[41] verdict: {report['verdict']}")
    print(f"[41] 完成 {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
