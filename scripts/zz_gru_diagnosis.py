"""阶段 ZZ：gru 单路负超额诊断（2026-11-01）。

针对敏感性矩阵中 gru 全线唯一负超额的发现，做三件可核验的诊断：
  1. 信号重叠度：gru 与 baseline/ens/ev/fusion 每日 top-10 的重叠率（Jaccard/交集均值）；
  2. 独有 vs 共享归因：gru top-10 中「仅 gru 有」与「至少一路共享」的票，各自 h10 收益/胜率；
  3. 时段分解：gru / baseline / fusion 组合净值按月的收益对比，定位负超额发生在哪些时段。

口径与 zz_paper_portfolio 完全一致（T+1 开盘建仓、持有 horizon=10、T+11 开盘平仓、无成本无过滤），
仅做描述性统计，不造因果。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import zz_paper_portfolio as zz  # noqa: E402

MODELS = zz.DEFAULT_MODELS
OTHERS = MODELS[1:]  # baseline / ens_v39gru_w025 / ev / fusion
K = 10
HORIZON = 10


def load_sigs() -> dict[str, dict]:
    sigs = {}
    for m in MODELS:
        f = zz.SIGNAL_DIR / zz.SIGNAL_FILES[m]
        sigs[m] = json.loads(f.read_text(encoding="utf-8"))
    return sigs


def topk_by_date(sig: dict, date: str, k: int = K) -> set[str]:
    rows = [d for d in sig["daily"] if d["date"] == date]
    rows.sort(key=lambda x: -x["score"])
    return {r["symbol"] for r in rows[:k]}


def main() -> int:
    sigs = load_sigs()
    frames, calendar = zz.load_prices()

    dates_by_model = {m: {x["date"] for x in sigs[m]["daily"]} for m in MODELS}
    common = sorted(set.intersection(*dates_by_model.values()))
    print(f"信号日交集: {len(common)} 日（{common[0]} ~ {common[-1]}）")

    # ---- 1) 重叠率 ----
    overlap: dict[str, list[float]] = {m: [] for m in OTHERS}
    for d in common:
        g = topk_by_date(sigs["gru"], d)
        for m in OTHERS:
            o = topk_by_date(sigs[m], d)
            overlap[m].append(len(g & o) / K)
    overlap_rows = {m: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                        "min": float(np.min(v)), "max": float(np.max(v))}
                    for m, v in overlap.items()}

    # ---- 2) 独有 vs 共享：gru top-10 按日分类 ----
    # 对每个共同信号日，gru top10 中：
    #   shared  = 至少被 baseline/ens/ev/fusion 一路的当日 top10 覆盖
    #   unique  = 仅 gru 有
    h_ret_of = {}  # (date, sym) -> h_ret（复用 build_trades 无成本逻辑）
    gru_daily = sigs["gru"]["daily"]
    trades = zz.build_trades(frames, gru_daily, K, HORIZON)
    for t in trades:
        if not t["incomplete"]:
            h_ret_of[(t["signal_date"], t["symbol"])] = t["h_ret"]

    other_topk: dict[str, set[str]] = {}
    stats = {"shared": [], "unique": []}
    for d in common:
        g = topk_by_date(sigs["gru"], d)
        union_other = set()
        for m in OTHERS:
            union_other |= topk_by_date(sigs[m], d)
        for sym in g:
            key = (d, sym)
            if key not in h_ret_of:
                continue
            bucket = "shared" if sym in union_other else "unique"
            stats[bucket].append(h_ret_of[key])
    bucket_stats = {}
    for bucket, vals in stats.items():
        bucket_stats[bucket] = {
            "n": len(vals),
            "mean": round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "win": round(float(np.mean([v > 0 for v in vals])), 4),
            "cum": round(float(np.prod([1.0 + v for v in vals]) - 1.0), 4),
        }

    # ---- 3) 时段分解：gru/baseline/fusion 全区间逐月收益 ----
    month_stats: dict[str, dict] = {}
    for m in ["gru", "baseline", "fusion"]:
        res = zz.run_model(m, frames, calendar, K, 0)
        nav = res["nav"]  # [[date, nav, bench], ...]
        nav_df = pd.DataFrame(nav, columns=["date", "nav", "bench"])
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        nav_df["ym"] = nav_df["date"].dt.strftime("%Y-%m")
        monthly = {}
        for ym, grp in nav_df.groupby("ym"):
            grp = grp.sort_values("date")
            monthly[ym] = round(float(grp["nav"].iloc[-1] / grp["nav"].iloc[0] - 1.0), 4)
        month_stats[m] = monthly
    months = sorted(set.union(*[set(v) for v in month_stats.values()]))

    # ---- 4) gru 信号面特征（每日信号行数 / score 极值，与其他路对比） ----
    sig_shape = {}
    for m in MODELS:
        daily = sigs[m]["daily"]
        per_day = {}
        for x in daily:
            per_day.setdefault(x["date"], []).append(x["score"])
        dates = sorted(per_day)
        n_per_day = [len(per_day[d]) for d in dates]
        scores = [s for v in per_day.values() for s in v]
        sig_shape[m] = {
            "signal_days": len(dates),
            "rows_per_day_avg": round(float(np.mean(n_per_day)), 1),
            "score_mean": round(float(np.mean(scores)), 4),
            "score_min": round(float(np.min(scores)), 4),
            "score_max": round(float(np.max(scores)), 4),
        }

    out = {
        "generated_at": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "common_signal_days": common[0] + "~" + common[-1],
        "overlap_with_gru_top10": overlap_rows,
        "gru_top10_h10_by_bucket": bucket_stats,
        "monthly_returns": month_stats,
        "signal_shape": sig_shape,
    }
    op = zz.REPORT_DIR / "gru_diagnosis_20261101.json"
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入: {op}")

    # ---- 终端摘要 ----
    print("\n[1] gru 与各路 top10 平均重叠率（交集/10）:")
    for m, v in overlap_rows.items():
        print(f"    gru ∩ {m:<14} 均值 {v['mean']:.2%} 中位 {v['median']:.2%} 范围 {v['min']:.0%}~{v['max']:.0%}")
    print("\n[2] gru top10 分桶 h10（全部共同信号日）:")
    for b in ["shared", "unique"]:
        s = bucket_stats[b]
        print(f"    {b:<8} n={s['n']:<4} 均值 {s['mean']:+.2%} 中位 {s['median']:+.2%} 胜率 {s['win']:.2%} 累计 {s['cum']:+.2%}")
    print("\n[3] 月度组合收益（gru / baseline / fusion）:")
    print("    月份        gru      baseline  fusion")
    for ym in months:
        g = month_stats["gru"].get(ym); b = month_stats["baseline"].get(ym); f = month_stats["fusion"].get(ym)
        print(f"    {ym}  {g:+.2%}   {b:+.2%}   {f:+.2%}" if g is not None and b is not None and f is not None
              else f"    {ym}  部分缺失")
    print("\n[4] 信号面特征:")
    for m, s in sig_shape.items():
        print(f"    {m:<14} 信号日 {s['signal_days']} 日均行数 {s['rows_per_day_avg']} "
              f"score[{s['score_min']:.2f}, {s['score_max']:.2f}] 均值 {s['score_mean']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
