"""阶段 46 - 前向验证（rolling-origin / walk-forward）blend 候选，排除 in-sample 嫌疑。

问题：45 的 w=0.30 是在**含 2026 的全窗口(2021-2026)** 上按净夏普选出来的，有 in-sample
嫌疑。严谨做法：
  1. 训练窗 TRAIN = 2021-01-04 ~ 2025-12-31，在该窗重选最优 w（净夏普）。
  2. 测试窗（留出，绝对不参与选 w）：
     - TEST_B = 2026 全段 2026-01-01 ~ 2026-10-26 (~10 个月，最贴近当前部署)
     - TEST_A = 最后 6 月 2026-05-01 ~ 2026-10-26
  3. 在测试窗比较：现役 A vs (训练窗选出的 blend) vs (45 原 in-sample 选的 w=0.30)。
     指标：净夏普 / 净累计 / 回撤 / IC(全/2026/近59) / 2026 子区净夏普。
  4. 判据（写死，不挑数）：
     F1 训练窗选出的 w 在 TEST_B 净夏普 > A 在 TEST_B 净夏普
     F2 训练窗选出的 w 在 TEST_B 的 2026 IC > A 在 TEST_B 的 2026 IC
     F3 训练窗选出的 w 在 TEST_A(最后6月) 净夏普 > A 在 TEST_A 净夏普
     promote = F1 AND F2 AND F3（**仅为建议，不自动切换生产**）

注：z 是逐日横截面标准化，与所选日期区间无关（无时间泄漏）；唯一超参 w 在 TRAIN 选，
TEST 仅用于评估。只读 + 产出报告，不改现役/ks。

产物：data/P1/processed/report_v41_forward.json
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
TRAIN_END = pd.Timestamp("2025-12-31")
TEST_B_LO = pd.Timestamp("2026-01-01")
TEST_A_LO = pd.Timestamp("2026-05-01")

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
    elif kind == "full":
        m = pd.Series(True, index=df.index)
    else:  # recent59td（在传入 sig 已是测试窗的前提下，取该窗末 59 交易日）
        uniq = np.sort(pd.unique(dates))
        m = dates >= (uniq[-RECENT_TD] if len(uniq) > RECENT_TD else uniq[0])
    if m.sum() == 0:
        return None
    return float(ic_summary(daily_ic(df[m], pred_col="pred", y_col="y_excess"))["ic_mean"])


def _bt(sig: pd.DataFrame, panel: pd.DataFrame) -> dict:
    kw = dict(horizon=10, top_pct=TOP_PCT, rebalance_freq=RF, mode="long_short")
    net = run_backtest_continuous(sig, panel, **kw, cost=DEFAULT_COST, buffer=BUFFER)
    n = net.ls_stats or {}
    return {"net_sharpe": n.get("sharpe"), "net_total": n.get("total_return"),
            "maxdd": n.get("max_drawdown")}


def _eval(sig: pd.DataFrame, panel: pd.DataFrame, y_ref: pd.DataFrame) -> dict:
    full = _bt(sig, panel)
    return {"net_sharpe": full["net_sharpe"], "net_total": full["net_total"],
            "maxdd": full["maxdd"],
            "ic_full": _ic(sig, y_ref, "full"),
            "ic_2026": _ic(sig, y_ref, "2026"),
            "ic_recent": _ic(sig, y_ref, "recent59td")}


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
    print(f"[46] 对齐区间 {lo.date()}~{hi.date()} | A {len(A):,} / C {len(C_raw):,}")

    # 逐日横截面 z（与所选区间无关，无泄漏）
    A_z = A.copy(); A_z["pred"] = _z(A, "pred")
    C = C_raw.copy(); C["pred"] = _z(C, "pred"); C = C[["date", "symbol", "pred"]]
    c_keys = set(zip(C["date"], C["symbol"]))
    a_keys = pd.MultiIndex.from_frame(A_z[["date", "symbol"]])
    missing = A_z[~a_keys.isin(c_keys)][["date", "symbol", "pred"]].copy()
    C_pad = pd.concat([C, missing], ignore_index=True).drop_duplicates(
        subset=["date", "symbol"], keep="first")
    az = A_z.set_index(["date", "symbol"])["pred"]
    cz = C_pad.set_index(["date", "symbol"])["pred"]
    idx = pd.MultiIndex.from_frame(A_z[["date", "symbol"]])

    def blend(w):
        m = A_z[["date", "symbol"]].copy()
        m["pred"] = (1 - w) * az.reindex(idx).to_numpy() + w * cz.reindex(idx).to_numpy()
        return m

    # ---- 训练窗选 w（仅 2021~2025）----
    A_tr = A_z[A_z["date"] <= TRAIN_END]
    panel_tr = panel[panel["date"] <= TRAIN_END]
    best_w, best_sh = None, -1e9
    for w in W_GRID:
        r = _bt(blend(w)[blend(w)["date"] <= TRAIN_END], panel_tr)
        if r["net_sharpe"] is not None and r["net_sharpe"] > best_sh:
            best_sh, best_w = r["net_sharpe"], w
    A_tr_bt = _bt(A_tr, panel_tr)
    print(f"[46] 训练窗(≤2025)选 w={best_w:.2f} 净夏普 {best_sh:+.3f} | "
          f"现役A训练窗 {A_tr_bt['net_sharpe']:+.3f}")

    # ---- 测试窗评估 ----
    def eval_on(lo_t: pd.Timestamp, hi_t: pd.Timestamp, tag: str):
        A_t = A_z[(A_z["date"] >= lo_t) & (A_z["date"] <= hi_t)]
        p_t = panel[(panel["date"] >= lo_t) & (panel["date"] <= hi_t)]
        y_t = y_ref[(y_ref["date"] >= lo_t) & (y_ref["date"] <= hi_t)]
        res = {"A_现役": _eval(A_t, p_t, y_t)}
        for w in (best_w, 0.30):
            b = blend(w)
            b_t = b[(b["date"] >= lo_t) & (b["date"] <= hi_t)]
            res[f"blend_w{w:.2f}"] = _eval(b_t, p_t, y_t)
        print(f"\n[46] === {tag} ({lo_t.date()}~{hi_t.date()}) ===")
        for k, v in res.items():
            print(f"  {k:14s} 净夏普 {v['net_sharpe']:+.3f} | 累计 {v['net_total']:+.2%} | "
                  f"回撤 {v['maxdd']:.2%} | IC全 {v['ic_full']:+.4f} | IC2026 {v['ic_2026']:+.4f} | "
                  f"近59 {v['ic_recent']:+.4f}")
        return res

    testB = eval_on(TEST_B_LO, hi, "TEST_B 2026全段")
    testA = eval_on(TEST_A_LO, hi, "TEST_A 最后6月")

    # ---- 判据（写死）----
    aB = testB["A_现役"]; bB = testB[f"blend_w{best_w:.2f}"]
    aA = testA["A_现役"]; bA = testA[f"blend_w{best_w:.2f}"]
    F1 = (bB["net_sharpe"] is not None and aB["net_sharpe"] is not None
          and bB["net_sharpe"] > aB["net_sharpe"])
    F2 = (bB["ic_2026"] is not None and aB["ic_2026"] is not None
          and bB["ic_2026"] > aB["ic_2026"])
    F3 = (bA["net_sharpe"] is not None and aA["net_sharpe"] is not None
          and bA["net_sharpe"] > aA["net_sharpe"])
    promote = bool(F1 and F2 and F3)
    print(f"\n[46] 判据：F1(TEST_B净夏普胜) {F1} | F2(TEST_B 2026IC胜) {F2} | "
          f"F3(TEST_A最后6月净夏普胜) {F3}")
    print(f"[46] promote(仅为建议，不自动切换生产) = {promote}")

    report = {
        "aligned_window": [str(lo.date()), str(hi.date())],
        "train_window": ["2021-01-04", "2025-12-31"],
        "config": {"rf": RF, "top_pct": TOP_PCT, "buffer": BUFFER, "w_grid": list(W_GRID)},
        "train_selected_w": best_w, "train_selected_w_net_sharpe": best_sh,
        "incumbent_train_net_sharpe": A_tr_bt["net_sharpe"],
        "testB_2026": testB, "testA_last6m": testA,
        "gate": {"F1_testB_net_win": F1, "F2_testB_2026ic_win": F2,
                 "F3_testA_net_win": F3, "promote": promote},
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (PROCESSED / "report_v41_forward.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[46] 报告已写 report_v41_forward.json  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
