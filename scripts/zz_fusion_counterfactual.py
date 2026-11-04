"""阶段 ZZ：gru 权重下调 / 独有票剔除 反事实回测（2026-11-01）。

背景：gru 单路诊断显示 2026 窗口 gru 全线唯一负超额、与主流路几乎正交。
生产融合（11_conditional_fusion）最优策略已退化为纯基线（GRU 被事实排除）；
gru 的真实"权重"落点在 ensemble：ens_v39gru_w025 = 0.75*z_v39 + 0.25*z_gru（37_build_ensemble_pred.py）。

本脚本在模拟盘框架（zz_paper_portfolio 口径，T+1 开盘/持有10日/等权净值）下做离线反事实：
  A. w_gru 扫描：ens(w) = (1-w)*z_v39 + w*z_gru，w ∈ {0.25(生产), 0.15, 0.10, 0.05, 0.00}，
     看 2026 窗口下最优 w 是否 < 0.25（验证"应下调 gru 权重"）；
  B. 独有票剔除：现版 ens 每日 top-10 中剔除"gru 独有票"（gru 当日 top10 且 baseline/ev/fusion 均未选），
     验证剔除规则有无增量。

窗口：与敏感性矩阵一致，信号日 2026-07-14 ~ 2026-10-12（60 信号日，end=各路 latest 最小值，可传参）。
纯离线、不写生产文件；结论供老板拍板是否改 37 脚本 W_GRU。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import zz_paper_portfolio as zz  # noqa: E402

W_LIST = [0.25, 0.15, 0.10, 0.05, 0.00]
K = 10
HORIZON = 10


def _z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
            ).fillna(0.0).clip(-3, 3)


def pred_to_daily(preds: pd.DataFrame, score_col: str) -> list[dict]:
    """pred 层 → 与 signal json 同构的 daily（date/symbol/score/signal）。"""
    df = preds[preds[score_col] != 0.0]
    return [{"date": d.strftime("%Y-%m-%d"), "symbol": s, "score": float(v), "signal": 1}
            for d, s, v in zip(df["date"], df["symbol"], df[score_col])]


def topk_set(daily: list[dict], date: str, k: int = K) -> set[str]:
    rows = [d for d in daily if d["date"] == date]
    rows.sort(key=lambda x: -x["score"])
    return {r["symbol"] for r in rows[:k]}


def summarize(frames, calendar, daily, start, end, label) -> dict:
    trades = zz.build_trades(frames, daily, K, HORIZON, start, end)
    done = [t for t in trades if not t["incomplete"]]
    h = [t["h_ret"] for t in done]
    nav = zz.portfolio_curve(frames, done, calendar, HORIZON)
    bench = zz.bench_curve(frames, calendar)
    start_i = 0
    if done:
        sd = pd.Timestamp(min(t["entry_date"] for t in done))
        start_i = calendar.index(sd) if sd in calendar else 0
    nav_a = nav.iloc[start_i:] / nav.iloc[start_i]
    bench_a = bench.iloc[start_i:] / bench.iloc[start_i]
    m = zz.metrics(nav_a, bench_a)
    out = {"label": label, "trades": len(done),
           "h10_win": round(float(np.mean([r > 0 for r in h])), 4) if h else None,
           "h10_avg": round(float(np.mean(h)), 4) if h else None,
           "cum": m["cum_ret"], "excess": m["excess"], "maxdd": m["max_dd"],
           "n_days": m["n_days"]}
    print(f"  {label:<34} 笔 {len(done):<4} 累计 {m['cum_ret']:+.2%} 超额 {m['excess']:+.2%} "
          f"回撤 {m['max_dd']:.2%} h10胜率 {out['h10_win'] if out['h10_win'] is not None else float('nan'):.2%}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-07-28")
    ap.add_argument("--end", default="2026-10-12")
    ap.add_argument("--out", default="fusion_counterfactual_20261101.json")
    args = ap.parse_args()

    v39 = pd.read_parquet(zz.ROOT / "data" / "P1" / "processed" / "pred_baseline_h10_v39.parquet",
                          columns=["date", "symbol", "pred"])
    gru = pd.read_parquet(zz.ROOT / "data" / "P1" / "processed" / "pred_gru_h10.parquet",
                          columns=["date", "symbol", "pred"])
    for d in (v39, gru):
        d["date"] = pd.to_datetime(d["date"]).dt.normalize()
        d.drop_duplicates(subset=["date", "symbol"], keep="last", inplace=True)
    v39["z_v39"] = _z(v39, "pred")
    gru["z_gru"] = _z(gru, "pred")
    m = v39[["date", "symbol", "z_v39"]].merge(gru[["date", "symbol", "z_gru"]],
                                               on=["date", "symbol"], how="left")
    print(f"pred 合并: {len(m):,} 行 | gru 覆盖 {m['z_gru'].notna().mean():.1%}")

    frames, calendar = zz.load_prices()
    print(f"kline: {len(frames)} 股 / 交易日历 {len(calendar)} 日")
    results = []

    # ---- A. w_gru 扫描（含 w=0.25 重建，与现版 ens 对照一致性） ----
    print(f"\n[A] w_gru 扫描（窗口 {args.start} ~ {args.end}，k={K}）")
    for w in W_LIST:
        ens_z = (1 - w) * m["z_v39"] + w * m["z_gru"].fillna(0.0)
        daily = pred_to_daily(m.assign(score=ens_z), "score")
        results.append(summarize(frames, calendar, daily, args.start, args.end,
                                 f"ens w_gru={w:.2f}"))

    # ---- 一致性校验：现版 ens 信号 json vs w=0.25 重建 ----
    prod = json.loads((zz.SIGNAL_DIR / zz.SIGNAL_FILES["ens_v39gru_w025"]).read_text(encoding="utf-8"))
    prod_daily = prod["daily"]
    w25_daily = [d for d in pred_to_daily(m.assign(score=(1 - 0.25) * m["z_v39"] + 0.25 * m["z_gru"].fillna(0.0)), "score")
                 if args.start <= d["date"] <= args.end]
    dates = sorted({d["date"] for d in prod_daily if args.start <= d["date"] <= args.end})
    inter = []
    for dt in dates:
        p = topk_set(prod_daily, dt)
        r = topk_set(w25_daily, dt)
        inter.append(len(p & r) / K)
    print(f"\n[校验] 现版 ens json vs 重建 w=0.25 top10 平均交集: {np.mean(inter):.1%}（{len(dates)} 信号日）")
    results.append({"label": "consistency_prod_vs_rebuild_w025",
                    "top10_overlap": round(float(np.mean(inter)), 4), "n_days": len(dates)})

    # ---- 校验 2：baseline json vs 重建 w=0.00（纯 v39） ----
    base_prod = json.loads((zz.SIGNAL_DIR / zz.SIGNAL_FILES["baseline"]).read_text(encoding="utf-8"))["daily"]
    w00_daily = [d for d in pred_to_daily(m.assign(score=m["z_v39"]), "score")
                 if args.start <= d["date"] <= args.end]
    dates_b = sorted({d["date"] for d in base_prod if args.start <= d["date"] <= args.end})
    inter_b = [len(topk_set(base_prod, dt) & topk_set(w00_daily, dt)) / K for dt in dates_b]
    print(f"[校验] baseline json vs 重建 w=0.00(纯v39) top10 平均交集: {np.mean(inter_b):.1%}（{len(dates_b)} 信号日）")
    results.append({"label": "consistency_baseline_vs_rebuild_w000",
                    "top10_overlap": round(float(np.mean(inter_b)), 4), "n_days": len(dates_b)})

    # ---- B. 独有票剔除（现版 ens top10 剔除 gru 独有票） ----
    print(f"\n[B] 独有票剔除（现版 ens 每日 top-10 中剔除 gru 独有票）")
    gru_daily = json.loads((zz.SIGNAL_DIR / zz.SIGNAL_FILES["gru"]).read_text(encoding="utf-8"))["daily"]
    baseline_daily = json.loads((zz.SIGNAL_DIR / zz.SIGNAL_FILES["baseline"]).read_text(encoding="utf-8"))["daily"]
    ev_daily = json.loads((zz.SIGNAL_DIR / zz.SIGNAL_FILES["ev"]).read_text(encoding="utf-8"))["daily"]
    fusion_daily = json.loads((zz.SIGNAL_DIR / zz.SIGNAL_FILES["fusion"]).read_text(encoding="utf-8"))["daily"]

    results.append(summarize(frames, calendar, prod_daily, args.start, args.end, "prod ens（对照）"))
    n_excl_total, n_eff_days = 0, 0
    filtered: list[dict] = []
    for dt in dates:
        g = topk_set(gru_daily, dt)
        others = set()
        for od in (baseline_daily, ev_daily, fusion_daily):
            others |= topk_set(od, dt)
        gru_unique = g - others
        n_excl = 0
        new_daily: list[dict] = []
        for row in prod_daily:
            if row["date"] != dt:
                continue
            if row["symbol"] in gru_unique and row["symbol"] in topk_set(prod_daily, dt):
                n_excl += 1
                continue  # 剔除 gru 独有票（仅当它进了 ens top10）
            new_daily.append(row)
        filtered.extend(new_daily)
        n_excl_total += n_excl
        n_eff_days += (1 if n_excl else 0)
    print(f"  触发剔除的信号日 {n_eff_days}/{len(dates)}，共剔除 {n_excl_total} 笔 top10 成员")
    results.append(summarize(frames, calendar, filtered, args.start, args.end,
                             "prod ens 剔除 gru 独有票"))

    # ---- 落盘 ----
    out = {"generated_at": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
           "window": {"start": args.start, "end": args.end}, "k": K,
           "results": results}
    op = zz.REPORT_DIR / args.out
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {op}")

    # ---- 结论判定 ----
    w_rows = {r["label"]: r for r in results if r["label"].startswith("ens w_gru")}
    best_w = max(w_rows, key=lambda k: w_rows[k]["cum"])
    prod_row = next(r for r in results if r["label"] == "prod ens（对照）")
    print(f"\n[w 扫描最优] {best_w} 累计 {w_rows[best_w]['cum']:+.2%} vs 生产 w=0.25 {w_rows['ens w_gru=0.25']['cum']:+.2%} "
          f"vs 现版 ens {prod_row['cum']:+.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
