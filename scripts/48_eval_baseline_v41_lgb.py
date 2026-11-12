"""阶段 48 - 真·同族修复：用 LightGBM 同族复跑 v39/v41 baseline，做净值层终判。

背景
----
长期阻塞项「环境无 lightgbm 轮子」已于 2026-11-12 解除（从 aliyun 镜像装 lightgbm 4.7.0
到默认 env，cp313 兼容；baseline_lgb.py 本就用现代 `lgb.train`+`callbacks` API，4.7.0 直接可用）。
本脚本是阶段 43 的同族修正版：43 被迫用 HistGradientBoosting 作同族控制，引入
「lgb→HGB 模型族退化（−0.446）」这一混杂，导致无法判定估值因子本身是否带来净收益。
现在用**真正的 lgb** 复跑 v39（控制）与 v41（候选，含 ep/btop 估值因子），模型族一致，
A'→C' 差异**仅**来自估值因子 + 股票池变化（无模型族混杂）→ 终判估值因子在净值层是否成立。

公平性三道处理（与 43 同口径）
----------------------------
1. 对齐时间区间：所有组裁到共同区间 [max(min), min(max)]。
2. 去重：各 pred 按 (date,symbol) drop_duplicates keep last（数据源遗留重复，43 同处理）。
3. 统一标签：IC 的 y_excess 统一取自 lgb v39 控制（同源同口径）；构造统一用 _z() clip±3。

两层对照
--------
- A 现役 ens（real-LGB v39，生产，已 z）          = 生产实际基线（faithfulness 基准）
- B v39 LGB 控制（本次复跑，real-LGB，39 维, z）   = 复跑的生产基线；B vs A 验「复跑是否忠实」
- C v41 LGB 候选（本次复跑，real-LGB，41 维, z）  = 候选；B→C = 估值因子 + 股票池（同族，无家族混杂）

判据（与 43/42 同口径）
------------------------
候选 C 净夏普 > 控制 B + 0.05 **且** C 的 2026 IC > B + 0.003 → 达标（估值因子在同族下兑现净收益）；
否则如实判「不达标」。

主对比用 B 与 C 的**键交集**（消除 v41 剔除 24 只 PE≤0 亏损股带来的股票池覆盖差异），
并附全池（C 行数更少）数字供透明。

守红线：只读评估 + 产出独立报告；不覆盖现役 `pred_ens_v39gru_w025_h10.parquet`、
不触碰 ks 仓任何 signal/代码；候选 pred 由 47/28 训练脚本独立写出。
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

F_INCUMBENT = "pred_ens_v39gru_w025_h10.parquet"   # 现役（纯 v39, 已 z）
F_V39_LGB = "pred_baseline_h10_v39.parquet"        # 现役 baseline 原件（raw, z 后=生产真实 v39；faithful 基准）
F_V41_LGB = "pred_baseline_h10_v41_lgb.parquet"    # 本次复跑 lgb v41 候选


def _z(df: pd.DataFrame, col: str = "pred") -> pd.Series:
    """逐日截面 z-score + clip±3，与阶段 37/43 _z 完全一致。"""
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
    print(f"    {path.name:40s} {n0:>9,} 行 → {len(d):>9,} 唯一键"
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


def _fmt_bt(bt: dict) -> str:
    return (f"净夏普 {bt['net_sharpe']:>+8.3f} | 净累计 {bt['net_total_return']:>+9.2%} "
            f"| 回撤 {bt['net_maxdd']:>8.2%} | 毛夏普 {bt['gross_sharpe']:>+8.3f}")


def _fmt_ic(ic: dict) -> str:
    cells = []
    for w in ["FULL", "2024+", "2026", "recent59td"]:
        v = ic.get(w)
        cells.append(f"{v['ic_mean']:+.4f}(t{v['t_stat']:+.1f})" if v else "—")
    return " ".join(f"{x:>13s}" for x in cells)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-incumbent", action="store_true",
                    help="跳过现役对比（faithfulness 检查），仅做同族 B vs C")
    args = ap.parse_args()
    t0 = time.time()

    print("[48] 载入信号（统一去重 + z 构造）")
    b = _load(PROCESSED / F_V39_LGB)
    c = _load(PROCESSED / F_V41_LGB)
    a_inc = _load(PROCESSED / F_INCUMBENT, already_z=True) if not args.no_incumbent else None

    # 统一标签（来自 lgb v39 控制，同源同口径；含 y_excess）
    yv = pd.read_parquet(PROCESSED / F_V39_LGB)
    yv["date"] = pd.to_datetime(yv["date"])
    y_ref = (yv[["date", "symbol", "y_excess"]]
             .drop_duplicates(subset=["date", "symbol"], keep="last"))

    # 共同区间（三/两组交集）
    all_min = [b["date"].min(), c["date"].min()] + ([a_inc["date"].min()] if a_inc is not None else [])
    all_max = [b["date"].max(), c["date"].max()] + ([a_inc["date"].max()] if a_inc is not None else [])
    lo, hi = max(all_min), min(all_max)
    print(f"\n[48] 共同区间: {lo.date()} ~ {hi.date()}")
    b = b[(b["date"] >= lo) & (b["date"] <= hi)].copy()
    c = c[(c["date"] >= lo) & (c["date"] <= hi)].copy()
    if a_inc is not None:
        a_inc = a_inc[(a_inc["date"] >= lo) & (a_inc["date"] <= hi)].copy()

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel[(panel["date"] >= lo) & (panel["date"] <= hi)]

    # ---- Faithfulness：复跑 lgb v39 控制 vs 现役 ----
    faith = {}
    if a_inc is not None:
        print("\n[48] === Faithfulness：复跑 lgb v39 控制(B) vs 现役 real-LGB v39(A) ===")
        print(f"  {'组':28s} {_fmt_bt.__name__ and '净值指标':28s}")
        ba = _backtest(a_inc, panel)
        bb = _backtest(b, panel)
        print(f"  {'A_现役(real-LGB_v39)':28s} {_fmt_bt(ba)}")
        print(f"  {'B_v39_LGB控制(复跑)':28s} {_fmt_bt(bb)}")
        gap = round(bb["net_sharpe"] - ba["net_sharpe"], 4)
        faith = {"A_net_sharpe": ba["net_sharpe"], "B_net_sharpe": bb["net_sharpe"],
                 "gap_B_minus_A": gap,
                 "ic_A": _ic_by_window(a_inc, y_ref), "ic_B": _ic_by_window(b, y_ref)}
        print(f"  >> 复跑偏差 Δ净夏普(B−A) = {gap:+.4f}  "
              f"（|Δ|<0.05 视为忠实复现；≥0.05 需排查 lightgbm 版本/配置漂移）")

    # ---- 主对比：同族 B vs C，键交集（消除股票池差异）----
    print("\n[48] === 主对比（同族 lgb）：键交集 B∩C（消除 v41 股票池剔除差异）===")
    keys_b = set(zip(b["date"], b["symbol"]))
    keys_c = set(zip(c["date"], c["symbol"]))
    inter = keys_b & keys_c
    print(f"  B 键数 {len(keys_b):,} | C 键数 {len(keys_c):,} | 交集 {len(inter):,}"
          f"（C 少 {len(keys_b)-len(keys_c):,}，= v41 剔除 PE<=0 亏损股）")
    bi = b.merge(c, on=["date", "symbol"], suffixes=("_b", "_c"))[["date", "symbol", "pred_b", "pred_c"]]
    sig_bi = bi[["date", "symbol", "pred_b"]].rename(columns={"pred_b": "pred"})
    sig_ci = bi[["date", "symbol", "pred_c"]].rename(columns={"pred_c": "pred"})

    bt_bi = _backtest(sig_bi, panel)
    bt_ci = _backtest(sig_ci, panel)
    ic_bi = _ic_by_window(sig_bi, y_ref)
    ic_ci = _ic_by_window(sig_ci, y_ref)
    print(f"  {'B_v39_LGB控制(交集)':28s} {_fmt_bt(bt_bi)}")
    print(f"  {'C_v41_LGB候选(交集)':28s} {_fmt_bt(bt_ci)}")
    print(f"  {'':28s} {'FULL':>13s}{'2024+':>13s}{'2026':>13s}{'recent59td':>13s}")
    print(f"  {'B IC':28s} {_fmt_ic(ic_bi)}")
    print(f"  {'C IC':28s} {_fmt_ic(ic_ci)}")

    d_sharpe = round(bt_ci["net_sharpe"] - bt_bi["net_sharpe"], 4)
    d_ic2026 = round(ic_ci["2026"]["ic_mean"] - ic_bi["2026"]["ic_mean"], 4)
    d_icfull = round(ic_ci["FULL"]["ic_mean"] - ic_bi["FULL"]["ic_mean"], 4)

    # ---- 透明：全池 B vs C（C 行数更少）----
    print("\n[48] === 透明：全池（不交集，C 覆盖更少）===")
    bt_bf = _backtest(b, panel)
    bt_cf = _backtest(c, panel)
    ic_bf = _ic_by_window(b, y_ref)
    ic_cf = _ic_by_window(c, y_ref)
    print(f"  {'B_v39_LGB控制(全池)':28s} {_fmt_bt(bt_bf)}")
    print(f"  {'C_v41_LGB候选(全池)':28s} {_fmt_bt(bt_cf)}")
    print(f"  {'B IC':28s} {_fmt_ic(ic_bf)}")
    print(f"  {'C IC':28s} {_fmt_ic(ic_cf)}")

    # ---- 判据 ----
    pass_sharpe = bt_ci["net_sharpe"] > bt_bi["net_sharpe"] + 0.05
    pass_ic = ic_ci["2026"]["ic_mean"] > ic_bi["2026"]["ic_mean"] + 0.003
    verdict = (
        "达标：v41(含估值) 在同族 lgb 下，键交集净夏普与 2026 IC 均显著优于 v39 → "
        "估值因子真实兑现净收益，建议进入生产替换评估"
        if (pass_sharpe and pass_ic) else
        "不达标：即便同族 lgb，v41 未在键交集净夏普与 2026 IC 上同时显著优于 v39 → "
        "维持现役；估值因子的 IC 增量仍未能兑现为组合层净收益")
    print("\n[48] === 判据（与 43/42 同口径）===")
    print(f"  键交集 净夏普Δ(C−B) > +0.05 → {pass_sharpe}  "
          f"({bt_ci['net_sharpe']:+.3f} − {bt_bi['net_sharpe']:+.3f} = {d_sharpe:+.4f})")
    print(f"  键交集 2026 ICΔ(C−B) > +0.003 → {pass_ic}  ({d_ic2026:+.4f})")
    print(f"  键交集 FULL ICΔ = {d_icfull:+.4f}")
    print(f"[48] verdict: {verdict}")

    report = {
        "common_window": [str(lo.date()), str(hi.date())],
        "config": {"rf": RF, "top_pct": TOP_PCT, "buffer": BUFFER, "cost": "DEFAULT_COST",
                   "family": "real-LightGBM 4.7.0 (同族)"},
        "faithfulness": faith,
        "same_family_intersection": {
            "n_keys_B": len(keys_b), "n_keys_C": len(keys_c), "n_intersection": len(inter),
            "B": {"bt": bt_bi, "ic": ic_bi}, "C": {"bt": bt_ci, "ic": ic_ci},
            "delta_net_sharpe": d_sharpe, "delta_ic_2026": d_ic2026, "delta_ic_full": d_icfull,
        },
        "same_family_fullpool": {
            "B": {"bt": bt_bf, "ic": ic_bf}, "C": {"bt": bt_cf, "ic": ic_cf},
        },
        "gate": {"pass_sharpe": bool(pass_sharpe), "pass_ic": bool(pass_ic)},
        "verdict": verdict,
        "honest_boundary": [
            "模型族已统一为 real-LightGBM（环境已恢复 lightgbm 4.7.0），A'→C' 无家族混杂",
            "主对比用键交集消除 v41 永久剔除 24 只 PE<=0 亏损股带来的股票池覆盖差异",
            "recent59td 为共同区间末 59 交易日（高方差，谨慎解读）",
            "IC 标签统一取自 lgb v39 控制 y_excess，构造统一用 _z() clip±3",
            "faithfulness: 复跑 lgb v39 控制应≈现役 real-LGB v39（Δ净夏普<0.05 视为忠实）；"
            "若偏差大，需排查 lightgbm 版本/num_threads 漂移",
        ],
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (PROCESSED / "report_v41_lgb_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[48] 报告已写 report_v41_lgb_eval.json  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
