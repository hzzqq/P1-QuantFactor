"""阶段 42 - B 生产化：构造 v41 生产候选信号并做取代评估（不覆盖现役）。

背景（来自阶段 37 / 2026-11-01 拍板）：
  - 现役 ensemble `pred_ens_v39gru_w025_h10.parquet` 实际 W_GRU=0（纯 v39 序），
    因为当时发现生产 GRU（38 维、无估值因子）在近期有害被 fusion 排除。
  - 阶段 41/本窗口结论：v41 GRU（同架构 + ep/btop 估值因子）IC 在生产模型级显著优于
    生产 GRU，且在 2026 / recent59td 生产 GRU 已转负时仍强正 → 估值增量在 GRU 级成立。

本脚本只换 GRU 半边（baseline 固定为现役同款 real-LGB v39，保证只隔离 GRU 效应）：
  pred = (1 - W_GRU) * z_v39 + W_GRU * z_gru_v41      （z 为截面 z-score，clip ±3）
  - 候选写出到 **新文件名** `pred_ens_v41gru_w025_h10.parquet`（不覆盖现役 v39 版）。
  - 现役 v39 版 `pred_ens_v39gru_w025_h10.parquet` 与 ks 消费链保持不动（守红线）。
  - 扫 W_GRU ∈ {0.0, 0.15, 0.25}，比对 FULL/2024+/2026/recent59td 的 IC 与
    RF=15/TOP=0.1/BUFFER=0.1 连续回测（净/毛），输出取代评估结论。

结论判据：当候选（W_GRU>0）在 2024+/2026 回测净夏普与近期 IC 显著优于现役纯 v39
→ 建议 boss 决策将生产切换到 v41 候选（替换/重指向），本脚本只产出候选与证据。
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

from shared import paths  # noqa: E402
from src.backtest import run_backtest, run_backtest_continuous, DEFAULT_COST  # noqa: E402
from src.eval.metrics import daily_ic, ic_summary  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
RECENT_TD = 59
RF, TOP_PCT, BUFFER = 15, 0.1, 0.1
ZERO = {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}
V39_FILE = "pred_baseline_h10_v39.parquet"      # 现役同款 real-LGB v39（baseline 不变）
GRU_V41_FILE = "pred_gru_h10_v41.parquet"        # 已修正跨年重复的 v41 GRU
INCUMBENT = "pred_ens_v39gru_w025_h10.parquet"   # 现役（纯 v39, W=0）
CANDIDATE = "pred_ens_v41gru_w025_h10.parquet"   # 候选（v39 + v41 GRU）


def _z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
            ).fillna(0.0).clip(-3, 3)


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
                  "ic_positive_rate": s["ic_positive_rate"], "t_stat": s["t_stat"]}
    return out


def _backtest(out: pd.DataFrame, panel: pd.DataFrame) -> dict:
    kw = dict(horizon=10, top_pct=TOP_PCT, rebalance_freq=RF, mode="long_short")
    net = run_backtest_continuous(out, panel, **kw, cost=DEFAULT_COST, buffer=BUFFER)
    gross = run_backtest_continuous(out, panel, **kw, cost=ZERO, buffer=BUFFER)
    n, g = net.ls_stats or {}, gross.ls_stats or {}
    return {
        "net_total_return": n.get("total_return"), "net_annual": n.get("annual_return"),
        "net_sharpe": n.get("sharpe"), "net_maxdd": n.get("max_drawdown"),
        "net_win_rate": n.get("win_rate"), "gross_sharpe": g.get("sharpe"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-gru", type=float, default=0.25,
                    help="候选 GRU 权重（候选默认 0.25，复现历史配置）")
    ap.add_argument("--out", default=None, help="候选输出名（默认 CANDIDATE）")
    args = ap.parse_args()
    t0 = time.time()

    v39 = pd.read_parquet(PROCESSED / V39_FILE)
    gru = pd.read_parquet(PROCESSED / GRU_V41_FILE)
    for d in (v39, gru):
        d["date"] = pd.to_datetime(d["date"])
        d.drop_duplicates(subset=["date", "symbol"], keep="last", inplace=True)
    v39["z_v39"] = _z(v39, "pred")
    gru["z_gru"] = _z(gru, "pred")

    panel = pd.read_parquet(PROCESSED / "panel.parquet")

    # 基准：现役纯 v39（W=0），直接回测现役 on-disk 文件做精确对照
    inc = pd.read_parquet(PROCESSED / INCUMBENT)
    inc["date"] = pd.to_datetime(inc["date"])
    inc.drop_duplicates(subset=["date", "symbol"], keep="last", inplace=True)
    inc_j = inc.merge(v39[["date", "symbol", "y_excess"]], on=["date", "symbol"], how="left")
    inc_ic = _ic_by_window(inc_j)
    inc_bt = _backtest(inc, panel)
    print("[42] 现役(纯v39,W=0) IC:", {k: (f"{v['ic_mean']:+.4f}" if v else "—")
          for k, v in inc_ic.items()})
    print(f"[42] 现役(纯v39,W=0) 回测: 净夏普 {inc_bt['net_sharpe']:+.3f} | "
          f"净累计 {inc_bt['net_total_return']:+.2%} | 回撤 {inc_bt['net_maxdd']:.2%} | "
          f"毛夏普 {inc_bt['gross_sharpe']:+.3f}", flush=True)

    # 候选扫 W_GRU
    sweep = [0.0, 0.15, 0.25]
    results = {}
    for W in sweep:
        m = v39[["date", "symbol", "z_v39", "pred", "y_excess"]].merge(
            gru[["date", "symbol", "z_gru"]], on=["date", "symbol"], how="left")
        has_gru = m["z_gru"].notna()
        m["pred"] = (1 - W) * m["z_v39"] + W * m["z_gru"].fillna(0.0)
        ic = _ic_by_window(m)
        bt = _backtest(m[["date", "symbol", "pred"]], panel)
        results[W] = {"ic": ic, "bt": bt,
                      "gru_coverage": f"{gru['date'].min().date()}~{gru['date'].max().date()}"}
        print(f"\n[42] 候选 W_GRU={W:.2f} GRU覆盖 {results[W]['gru_coverage']}")
        print("     IC:", {k: (f"{v['ic_mean']:+.4f}" if v else "—") for k, v in ic.items()})
        print(f"     回测: 净夏普 {bt['net_sharpe']:+.3f} | 净累计 {bt['net_total_return']:+.2%} | "
              f"回撤 {bt['net_maxdd']:.2%} | 毛夏普 {bt['gross_sharpe']:+.3f}", flush=True)

    # 选最优 W>0（以净夏普为准，近期 IC 为辅）
    best_w = max((w for w in sweep if w > 0),
                 key=lambda w: (results[w]["bt"]["net_sharpe"] or -9e9))
    print(f"\n[42] 候选最优 W_GRU={best_w:.2f}（净夏普最高）")

    # 写出候选（新文件名，不覆盖现役）
    W = args.w_gru
    m = v39[["date", "symbol", "z_v39"]].merge(
        gru[["date", "symbol", "z_gru"]], on=["date", "symbol"], how="left")
    m["pred"] = (1 - W) * m["z_v39"] + W * m["z_gru"].fillna(0.0)
    out_path = PROCESSED / (args.out or CANDIDATE)
    m[["date", "symbol", "pred"]].to_parquet(out_path, index=False)
    print(f"[42] 候选已写出: {out_path}  ({len(m):,} 行, W_GRU={W})", flush=True)

    # 取代评估判据
    c = results[best_w]
    inc_sharpe = inc_bt["net_sharpe"] or -9e9
    c_sharpe = c["bt"]["net_sharpe"] or -9e9
    inc_26 = (inc_ic.get("2026") or {}).get("ic_mean", -9e9)
    c_26 = (c["ic"].get("2026") or {}).get("ic_mean", -9e9)
    verdict = (
        "v41 候选（v39 + v41 GRU@%.2f）在净夏普(%.3f vs %.3f)与 2026 IC(%.4f vs %.4f) "
        "均显著优于现役纯 v39 → 建议 boss 决策切换到 v41 候选（替换/重指向 ks 消费文件）"
        % (best_w, c_sharpe, inc_sharpe, c_26, inc_26)
        if (c_sharpe > inc_sharpe + 0.05 and c_26 > inc_26 + 0.003) else
        "v41 候选未在净夏普与 2026 IC 上同时显著优于现役纯 v39 → 维持现役，不替换")
    print(f"\n[42] verdict: {verdict}")
    report = {"best_w": best_w, "args_w": W, "incumbent": {"ic": inc_ic, "bt": inc_bt},
              "candidates": {str(w): {"ic": r["ic"], "bt": r["bt"],
                                     "gru_coverage": r["gru_coverage"]} for w, r in results.items()},
              "verdict": verdict, "elapsed_sec": round(time.time() - t0, 1)}
    (PROCESSED / "report_v41_ensemble.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[42] 报告已写 report_v41_ensemble.json  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
