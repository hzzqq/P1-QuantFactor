"""阶段 45 - 把 v41 以「小权重融合」做成可部署改进候选（不达标候选的 redeem 路径）。

依据 44 探针结论：
  - 估值因子本身在组合层不伤（纯净对照 Δnet_sharpe=+0.0081）。
  - 主因是模型族 lgb→HGB 的 −0.446（环境无 lightgbm 轮子）。
  - 把 v41 以 w=0.2~0.3 融进现役，净夏普反超现役（1.53/1.61 > 1.495）。
  → 方向：小权重 regime 融合，先做成一个**独立候选文件**，守红线不覆盖现役/ks。

设计：
  1. 对齐区间（与 44 一致 2021-01-04~2026-10-26），统一 y_excess / panel / 净成本。
  2. z_A = 现役 pred 在其覆盖上重算 z；C_pad = v41 缺失键回退到 A 的 z（恢复股票池）。
  3. 网格 w∈{0.10,0.15,0.20,0.25,0.30,0.40}：候选 = (1−w)·z_A + w·z_C_pad。
  4. 每个 w 算：全区间净夏普 + IC(全/2024+/2026/近59) + 区间稳健性
     （分 2021-2024 与 2025-2026 两段净夏普，确认增益不只在末端噪声）。
  5. 选基：全区间净夏普最高且 >A 的 w；写候选文件 pred_ens_v41blend_wXX_h10.parquet。
  6. 判据（写死，不挑数）：
       P1 全区间净夏普 > A 且 Δ>0
       P2 2026 子区净夏普 > A 的 2026 子区（确认近期确有益）
       P3 近期段(2025-2026) 净夏普 > 早期段(2021-2024) 净夏普（增益集中在衰减期）
       recommend = P1 AND P2 AND P3（仅为建议，不自动切换生产）
  7. 只写新候选文件 + 报告，**不改现役、不碰 ks**（生产切换由人拍板）。

产物：
  data/P1/processed/pred_ens_v41blend_wXX_h10.parquet（候选，ks 未引用）
  data/P1/processed/report_v41_blend.json
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
from src.backtest import run_backtest_continuous, DEFAULT_COST  # noqa: E402
from src.eval.metrics import daily_ic, ic_summary  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
RECENT_TD = 59
RF, TOP_PCT, BUFFER = 15, 0.1, 0.1
W_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40)

F_INCUMBENT = "pred_ens_v39gru_w025_h10.parquet"
F_V39_LGB = "pred_baseline_h10_v39.parquet"
F_V41_HGB = "pred_baseline_h10_v41.parquet"


def _z(df: pd.DataFrame, col: str = "pred") -> pd.Series:
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
            ).fillna(0.0).clip(-3, 3)


def _dedup(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.drop_duplicates(subset=["date", "symbol"], keep="last")


def _ic(sig: pd.DataFrame, y_ref: pd.DataFrame, kind: str) -> float | None:
    df = sig.merge(y_ref, on=["date", "symbol"], how="inner")
    dates = pd.to_datetime(df["date"])
    if kind == "2026":
        m = dates >= pd.Timestamp("2026-01-01")
    elif kind == "2024+":
        m = dates >= pd.Timestamp("2024-01-01")
    else:  # recent59td
        uniq = np.sort(pd.unique(dates))
        m = dates >= (uniq[-RECENT_TD] if len(uniq) > RECENT_TD else uniq[0])
    if m.sum() == 0:
        return None
    return float(ic_summary(daily_ic(df[m], pred_col="pred", y_col="y_excess"))["ic_mean"])


def _bt(sig: pd.DataFrame, panel: pd.DataFrame, date_ge: pd.Timestamp | None = None) -> dict:
    s, p = sig, panel
    if date_ge is not None:
        s = s[s["date"] >= date_ge]
        p = p[p["date"] >= date_ge]
    kw = dict(horizon=10, top_pct=TOP_PCT, rebalance_freq=RF, mode="long_short")
    net = run_backtest_continuous(s, p, **kw, cost=DEFAULT_COST, buffer=BUFFER)
    n = net.ls_stats or {}
    return {"net_sharpe": n.get("sharpe"), "net_total": n.get("total_return"),
            "maxdd": n.get("max_drawdown")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    t0 = time.time()

    A = _dedup(pd.read_parquet(PROCESSED / F_INCUMBENT))[["date", "symbol", "pred"]]
    C_raw = _dedup(pd.read_parquet(PROCESSED / F_V41_HGB))
    yv = _dedup(pd.read_parquet(PROCESSED / F_V39_LGB))
    y_ref = yv[["date", "symbol", "y_excess"]]
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    lo = max(A["date"].min(), C_raw["date"].min())
    hi = min(A["date"].max(), C_raw["date"].max())
    clip = lambda d: d[(d["date"] >= lo) & (d["date"] <= hi)].copy()
    A, C_raw = clip(A), clip(C_raw)
    panel = panel[(panel["date"] >= lo) & (panel["date"] <= hi)]
    print(f"[45] 对齐区间 {lo.date()}~{hi.date()} | A {len(A):,} / C_raw {len(C_raw):,}")

    # z_A 在共同窗口重算；C_pad 缺失键回退 A 的 z
    A["pred"] = _z(A, "pred")
    C = C_raw.copy(); C["pred"] = _z(C, "pred"); C = C[["date", "symbol", "pred"]]
    c_keys = set(zip(C["date"], C["symbol"]))
    a_keys = pd.MultiIndex.from_frame(A[["date", "symbol"]])
    missing = A[~a_keys.isin(c_keys)][["date", "symbol", "pred"]].copy()
    C_pad = pd.concat([C, missing], ignore_index=True).drop_duplicates(
        subset=["date", "symbol"], keep="first")
    print(f"[45] C_pad 补齐 {len(missing):,} 缺失键 → {len(C_pad):,} 行")

    az = A.set_index(["date", "symbol"])["pred"]
    cz = C_pad.set_index(["date", "symbol"])["pred"]
    idx = pd.MultiIndex.from_frame(A[["date", "symbol"]])

    # 现役基线（已对齐窗口）
    A_res = _bt(A, panel)
    A_2026 = _bt(A, panel, pd.Timestamp("2026-01-01"))
    A_early = _bt(A[A["date"] < pd.Timestamp("2025-01-01")], panel[panel["date"] < pd.Timestamp("2025-01-01")])
    A_late = _bt(A[A["date"] >= pd.Timestamp("2025-01-01")], panel[panel["date"] >= pd.Timestamp("2025-01-01")])
    A_full = {"net_sharpe": A_res["net_sharpe"], "net_total": A_res["net_total"],
              "maxdd": A_res["maxdd"],
              "ic_full": _ic(A, y_ref, "full"), "ic_2024p": _ic(A, y_ref, "2024+"),
              "ic_2026": _ic(A, y_ref, "2026"), "ic_recent": _ic(A, y_ref, "recent59td"),
              "net_sharpe_early": A_early["net_sharpe"], "net_sharpe_late": A_late["net_sharpe"],
              "net_sharpe_2026": A_2026["net_sharpe"]}
    print(f"[45] 现役 A: 净夏普 {A_full['net_sharpe']:+.3f} | 2026子区 {A_full['net_sharpe_2026']:+.3f} "
          f"| 早期 {A_full['net_sharpe_early']:+.3f} / 近期 {A_full['net_sharpe_late']:+.3f}")

    results = {}
    best_w, best_sharpe = None, -1e9
    for w in W_GRID:
        m = A[["date", "symbol"]].copy()
        m["pred"] = (1 - w) * az.reindex(idx).to_numpy() + w * cz.reindex(idx).to_numpy()
        full = _bt(m, panel)
        sub26 = _bt(m, panel, pd.Timestamp("2026-01-01"))
        early = _bt(m[m["date"] < pd.Timestamp("2025-01-01")],
                    panel[panel["date"] < pd.Timestamp("2025-01-01")])
        late = _bt(m[m["date"] >= pd.Timestamp("2025-01-01")],
                   panel[panel["date"] >= pd.Timestamp("2025-01-01")])
        r = {"w": w, "net_sharpe": full["net_sharpe"], "net_total": full["net_total"],
             "maxdd": full["maxdd"],
             "ic_full": _ic(m, y_ref, "full"), "ic_2024p": _ic(m, y_ref, "2024+"),
             "ic_2026": _ic(m, y_ref, "2026"), "ic_recent": _ic(m, y_ref, "recent59td"),
             "net_sharpe_early": early["net_sharpe"], "net_sharpe_late": late["net_sharpe"],
             "net_sharpe_2026": sub26["net_sharpe"]}
        results[f"w{w:.2f}"] = r
        if full["net_sharpe"] is not None and full["net_sharpe"] > best_sharpe:
            best_sharpe, best_w = full["net_sharpe"], w
        print(f"[45] w={w:.2f}: 净夏普 {r['net_sharpe']:+.3f} | 累计 {r['net_total']:+.2%} | "
              f"回撤 {r['maxdd']:.2%} | 2026 {r['ic_2026']:+.4f} | 近59 {r['ic_recent']:+.4f} "
              f"| 早 {r['net_sharpe_early']:+.3f} / 近 {r['net_sharpe_late']:+.3f}")

    # ---- 选基 + 判据（写死）----
    bw = best_w
    br = results[f"w{bw:.2f}"]
    P1 = (br["net_sharpe"] is not None and A_full["net_sharpe"] is not None
          and br["net_sharpe"] > A_full["net_sharpe"]
          and (br["net_sharpe"] - A_full["net_sharpe"]) > 0)
    P2 = (br["net_sharpe_2026"] is not None and A_full["net_sharpe_2026"] is not None
          and br["net_sharpe_2026"] > A_full["net_sharpe_2026"])
    P3 = (br["net_sharpe_late"] is not None and br["net_sharpe_early"] is not None
          and br["net_sharpe_late"] > br["net_sharpe_early"])
    recommend = bool(P1 and P2 and P3)
    print(f"\n[45] 选基 w={bw:.2f} | P1全区间胜 {P1} | P2 2026子区胜 {P2} | P3 近期>早期 {P3}")
    print(f"[45] recommend(仅为建议，不自动切换生产) = {recommend}")

    # ---- 写候选文件（独立命名，不覆盖现役）----
    m = A[["date", "symbol"]].copy()
    m["pred"] = (1 - bw) * az.reindex(idx).to_numpy() + bw * cz.reindex(idx).to_numpy()
    out_name = f"pred_ens_v41blend_w{int(bw*100):02d}_h10.parquet"
    out = m.merge(y_ref, on=["date", "symbol"], how="left")
    out.to_parquet(PROCESSED / out_name, index=False)
    print(f"[45] 候选已写 {out_name}（{len(out):,} 行，ks 未引用）")

    report = {
        "common_window": [str(lo.date()), str(hi.date())],
        "config": {"rf": RF, "top_pct": TOP_PCT, "buffer": BUFFER, "w_grid": list(W_GRID)},
        "incumbent_A": A_full,
        "blend_results": results,
        "best_w": bw, "best_w_net_sharpe": best_sharpe,
        "gate": {"P1_full_win": P1, "P2_2026_win": P2, "P3_late_gt_early": P3,
                 "recommend": recommend},
        "candidate_file": out_name,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (PROCESSED / "report_v41_blend.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[45] 报告已写 report_v41_blend.json  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
