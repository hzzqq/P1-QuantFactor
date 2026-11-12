"""阶段 43 - L3（老板拍板「按建议做」）：升级生产 baseline 到 v41（含估值因子）的取代评估。

为什么不是直接替换
------------------
现役生产信号 `pred_ens_v39gru_w025_h10.parquet` 实际 W_GRU=0（纯 v39 序），
即其 100% 来自 baseline（real-LightGBM v39）。阶段 42 已证「在 GRU 半边复产」
不达净夏普门槛；但 v41 **baseline**（HGB, 含 ep/btop）近59日 IC +0.1104
远强于现役 baseline +0.0632 → 真正的估值增量在 baseline 半边。本脚本做最终判据。

公平性三道处理（缺一不可，否则结论无效）
----------------------------------------
1. **对齐时间区间**：现役 v39 覆盖 2019~2026-11-09，HGB 两版仅 2021-01-04~2026-10-26
   → 全部裁到共同区间 [max(min_date), min(max_date)] 后再回测/算 IC。
   （记忆红线：跨模型对比必须对齐时间区间；子集乐观红线。）
2. **去重**：HGB 两版各带 8,936 条重复 (date,symbol)（数据源遗留）→ drop_duplicates keep last。
3. **统一标签**：所有组的 IC 统一 merge 现役 real-LGB v39 的 y_excess（同源同口径），
   回测统一用 panel.parquet，构造统一用 _z()（含 clip±3，与阶段 37 完全一致）。

两层对照（诚实拆混杂，不包装成「纯估值因子升级」）
--------------------------------------------------
- A 现役 ens（real-LGB v39, z）      = 生产实际基线
- B v39 HGB 控制（同族同 39 维, z）   A→B = 模型族 lgb→HGB 的混杂
- C v41 HGB（含 ep/btop, 41 维, z）  = 候选；B→C = 估值因子 + 股票池变化
  生产判据 = A vs C（实际替换效果）；归因 = A→B→C 拆解。
  注：环境无 lightgbm 轮子（py3.13 / ks py3.12 均无），候选只能用 HGB；
  v41 数据集永久排除 PE≤0 亏损股（少 ~8.4% 行），股票池与现役不同 —— 两者都会在结论中标注。

判据（与阶段 42 同口径，避免标准漂移）
--------------------------------------
候选 C 净夏普 > 现役 A + 0.05 **且** C 的 2026 IC > A + 0.003 → 达标；
否则如实判定「不达标，维持现役」。

守红线：本脚本**只读评估 + 产出独立候选文件**，绝不覆盖现役
`pred_ens_v39gru_w025_h10.parquet`，也不触碰 ks 仓任何 signal/代码。
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
ZERO = {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}

F_INCUMBENT = "pred_ens_v39gru_w025_h10.parquet"   # 现役（纯 v39, 已 z 过）
F_V39_LGB = "pred_baseline_h10_v39.parquet"        # 现役 baseline 原件（提供统一 y_excess）
F_V39_HGB = "pred_baseline_h10_v39_hgb.parquet"    # 同族控制
F_V41_HGB = "pred_baseline_h10_v41.parquet"        # 候选（含估值）
F_CANDIDATE = "pred_ens_v41base_h10.parquet"       # 候选输出（新文件名，不覆盖现役）


def _z(df: pd.DataFrame, col: str = "pred") -> pd.Series:
    """逐日截面 z-score + clip±3，与阶段 37 _z 完全一致。"""
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
            ).fillna(0.0).clip(-3, 3)


def _load(path: Path, already_z: bool = False) -> pd.DataFrame:
    d = pd.read_parquet(path)
    d["date"] = pd.to_datetime(d["date"])
    n0 = len(d)
    d.drop_duplicates(subset=["date", "symbol"], keep="last", inplace=True)
    n_dup = n0 - len(d)
    if not already_z:
        d["pred"] = _z(d, "pred")
    print(f"    {path.name:42s} {n0:>9,} 行 → {len(d):>9,} 唯一键"
          f"（去重 {n_dup:,}）  {d['date'].min().date()}~{d['date'].max().date()}")
    return d[["date", "symbol", "pred"]]


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


def _ic_by_window(sig: pd.DataFrame, y_ref: pd.DataFrame) -> dict:
    df = sig.merge(y_ref, on=["date", "symbol"], how="inner")
    dates = pd.to_datetime(df["date"])
    out = {}
    for w in ["FULL", "2024+", "2026", "recent59td"]:
        m = _window_mask(dates, w)
        if m.sum() == 0:
            out[w] = None
            continue
        s = ic_summary(daily_ic(df[m], pred_col="pred", y_col="y_excess"))
        out[w] = {"ic_mean": s["ic_mean"], "icir": s["icir"],
                  "t_stat": s["t_stat"], "n_days": s["n_days"]}
    return out


def _backtest(sig: pd.DataFrame, panel: pd.DataFrame) -> dict:
    kw = dict(horizon=10, top_pct=TOP_PCT, rebalance_freq=RF, mode="long_short")
    net = run_backtest_continuous(sig, panel, **kw, cost=DEFAULT_COST, buffer=BUFFER)
    gross = run_backtest_continuous(sig, panel, **kw, cost=ZERO, buffer=BUFFER)
    n, g = net.ls_stats or {}, gross.ls_stats or {}
    return {"net_total_return": n.get("total_return"), "net_annual": n.get("annual_return"),
            "net_sharpe": n.get("sharpe"), "net_maxdd": n.get("max_drawdown"),
            "net_win_rate": n.get("win_rate"), "gross_sharpe": g.get("sharpe")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-candidate", action="store_true",
                    help="仅当判据达标时写出候选文件（默认仍写，供审阅；不覆盖现役）")
    args = ap.parse_args()
    t0 = time.time()

    print("[43] 载入四组信号（统一去重 + z 构造）")
    inc = _load(PROCESSED / F_INCUMBENT, already_z=True)
    b = _load(PROCESSED / F_V39_HGB)
    c = _load(PROCESSED / F_V41_HGB)

    # 统一标签（来自现役 real-LGB v39，同源同口径）
    yv = pd.read_parquet(PROCESSED / F_V39_LGB)
    yv["date"] = pd.to_datetime(yv["date"])
    y_ref = (yv[["date", "symbol", "y_excess"]]
             .drop_duplicates(subset=["date", "symbol"], keep="last"))

    # 公平性处理 1：裁到共同区间（三组取交集）
    lo = max(inc["date"].min(), b["date"].min(), c["date"].min())
    hi = min(inc["date"].max(), b["date"].max(), c["date"].max())
    print(f"\n[43] 共同区间裁剪: {lo.date()} ~ {hi.date()}"
          f"（现役 {inc['date'].min().date()}~{inc['date'].max().date()}，"
          f"HGB 两版 {b['date'].min().date()}~{b['date'].max().date()}）")
    inc, b, c = [d[(d["date"] >= lo) & (d["date"] <= hi)].copy() for d in (inc, b, c)]
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[(panel["date"] >= lo) & (panel["date"] <= hi)]

    groups = {"A_现役(real-LGB_v39)": inc, "B_v39_HGB控制": b, "C_v41_HGB候选": c}
    results = {}
    for name, sig in groups.items():
        results[name] = {"ic": _ic_by_window(sig, y_ref), "bt": _backtest(sig, panel),
                         "n_rows": len(sig),
                         "n_dates": int(sig["date"].nunique()),
                         "n_symbols": int(sig["symbol"].nunique())}

    print("\n[43] === 对齐区间回测（RF=15, top=0.1, buffer=0.1, long_short, 净成本）===")
    print(f"  {'组':24s} {'行数':>10s} {'交易日':>7s} {'股票':>6s} "
          f"{'净夏普':>9s} {'净累计':>10s} {'回撤':>9s} {'毛夏普':>9s}")
    for name, r in results.items():
        bt = r["bt"]
        print(f"  {name:24s} {r['n_rows']:>10,} {r['n_dates']:>7} {r['n_symbols']:>6} "
              f"{bt['net_sharpe']:>+9.3f} {bt['net_total_return']:>+10.2%} "
              f"{bt['net_maxdd']:>9.2%} {bt['gross_sharpe']:>+9.3f}")

    print("\n[43] === 对齐区间 IC（统一 y_excess）===")
    print(f"  {'组':24s} {'FULL':>11s} {'2024+':>11s} {'2026':>11s} {'recent59td':>11s}")
    for name, r in results.items():
        cells = []
        for w in ["FULL", "2024+", "2026", "recent59td"]:
            v = r["ic"].get(w)
            cells.append(f"{v['ic_mean']:+.4f}(t{v['t_stat']:+.1f})" if v else "—")
        print(f"  {name:24s} " + " ".join(f"{x:>11s}" for x in cells))

    # 归因拆解
    a, bb, cc = (results[k] for k in ["A_现役(real-LGB_v39)", "B_v39_HGB控制", "C_v41_HGB候选"])
    attrib = {
        "模型族混杂_lgb→HGB(A→B)_净夏普Δ": round(bb["bt"]["net_sharpe"] - a["bt"]["net_sharpe"], 4),
        "估值因子+股票池(B→C)_净夏普Δ": round(cc["bt"]["net_sharpe"] - bb["bt"]["net_sharpe"], 4),
        "生产实际替换(A→C)_净夏普Δ": round(cc["bt"]["net_sharpe"] - a["bt"]["net_sharpe"], 4),
        "生产实际替换(A→C)_2026_ICΔ": round(cc["ic"]["2026"]["ic_mean"] - a["ic"]["2026"]["ic_mean"], 4),
        "股票池覆盖差异_行数": cc["n_rows"] - a["n_rows"],
    }
    print("\n[43] === 归因拆解（诚实标注混杂）===")
    for k, v in attrib.items():
        print(f"  {k:38s} {v}")

    # 判据（与 42 同口径）
    pass_sharpe = cc["bt"]["net_sharpe"] > a["bt"]["net_sharpe"] + 0.05
    pass_ic = cc["ic"]["2026"]["ic_mean"] > a["ic"]["2026"]["ic_mean"] + 0.003
    verdict = (
        "达标：v41 baseline 候选在对齐区间净夏普与 2026 IC 均显著优于现役 → 建议进入生产替换"
        if (pass_sharpe and pass_ic) else
        "不达标：v41 baseline 候选未在净夏普与 2026 IC 上同时显著优于现役 → 维持现役，不替换")
    print(f"\n[43] 判据: 净夏普Δ>{0.05} → {pass_sharpe} | 2026 ICΔ>{0.003} → {pass_ic}")
    print(f"[43] verdict: {verdict}")

    # 候选写出（新文件名；无论达标与否都写，供审阅；绝不覆盖现役 / 不触碰 ks）
    out_path = PROCESSED / F_CANDIDATE
    c[["date", "symbol", "pred"]].to_parquet(out_path, index=False)
    print(f"[43] 候选已写出（独立文件，ks 未引用）: {out_path}  ({len(c):,} 行)")

    report = {
        "common_window": [str(lo.date()), str(hi.date())],
        "config": {"rf": RF, "top_pct": TOP_PCT, "buffer": BUFFER, "cost": "DEFAULT_COST"},
        "groups": results, "attribution": attrib,
        "gate": {"pass_sharpe": bool(pass_sharpe), "pass_ic": bool(pass_ic)},
        "verdict": verdict, "candidate_file": F_CANDIDATE,
        "honest_boundary": [
            "现役 baseline 是 real-LightGBM；候选是 HistGradientBoosting（环境无 lgb 轮子）→ 模型族混杂",
            "v41 数据集永久排除 PE<=0 亏损股，股票池比现役少约 8.4% 行 → 覆盖差异",
            "全部裁到共同区间 2021-01-04~2026-10-26，recent59td 为该区间末 59 个交易日（高方差）",
            "IC 标签统一取自现役 real-LGB v39 的 y_excess，构造统一用 _z() clip±3",
        ],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (PROCESSED / "report_v41_baseline_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[43] 报告已写 report_v41_baseline_eval.json  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
