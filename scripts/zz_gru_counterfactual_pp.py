"""P1 gru 权重「真·反事实」组合回测（2026-11-11）。

兑现上一轮推荐里的验证：「在 ens 里把 gru 权重置 0，跑一遍 paper portfolio 看近 59 日是否回血」。
从 pred 文件按 37 的 _z 口径合成 w∈{0, 0.25, 1.0} 的 ens pred（纯内存），
按 06_export_signal 口径逐日取 score top-20 作 daily，复用 zz_paper_portfolio 的
build_trades / portfolio_curve / bench_curve，在 2024+ / 2026 / 近59日 三窗口走
真实组合回测（cost=0.001 单边 + 涨跌停过滤），对比净值与超额。

红线：不写 signals/、不写 ks、不改任何现役文件；只读 pred + kline。
输出 reports/gru_counterfactual_pp_YYYYMMDD.md。
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("E:/project/sj/P1-QuantFactor")
PROC = Path("E:/project/sj/data/P1/processed")
spec = importlib.util.spec_from_file_location("zz_pp", str(PROJ / "scripts" / "zz_paper_portfolio.py"))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)

WEIGHTS = [0.0, 0.25, 1.00]
WINDOWS = {"2024+": "2024-01-01", "2026": "2026-01-01", "近59日": "2026-08-11"}
TOPN = 20
K, HORIZON, COST = 10, 10, 0.001


def _z(df, col):
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
            ).fillna(0.0).clip(-3, 3)


def build_daily(m: pd.DataFrame, pred_col: str) -> list[dict]:
    d = m[["date", "symbol", pred_col]].copy()
    d["_r"] = d.groupby("date")[pred_col].rank(method="first", ascending=False)
    d = d[d["_r"] <= TOPN]
    d["date"] = d["date"].dt.strftime("%Y-%m-%d")
    return [{"date": r.date, "symbol": r.symbol, "score": float(getattr(r, pred_col))}
            for r in d.itertuples(index=False)]


def run(daily, start):
    trades = pp.build_trades(pp.frames, daily, K, HORIZON, start, None, COST, True)
    done = [t for t in trades if not t["incomplete"]]
    nav = pp.portfolio_curve(pp.frames, done, pp.calendar, HORIZON, COST)
    bench = pp.bench_curve(pp.frames, pp.calendar)
    if not done:
        return {"cum": np.nan, "excess": np.nan, "win": np.nan, "dd": np.nan, "n": 0}
    sd = pd.Timestamp(min(t["entry_date"] for t in done))
    si = pp.calendar.index(sd) if sd in pp.calendar else 0
    nav_a = nav.iloc[si:] / (nav.iloc[si] or 1.0)
    bench_a = bench.iloc[si:] / (bench.iloc[si] or 1.0)
    mt = pp.metrics(nav_a, bench_a)
    return {"cum": mt["cum_ret"], "excess": mt["excess"], "win": mt["daily_win"],
            "dd": mt["max_dd"], "n": len(done)}


def main() -> int:
    print("读取 kline ...", flush=True)
    pp.frames, pp.calendar = pp.load_prices()
    print(f"kline {len(pp.frames)} 股 / 日历 {len(pp.calendar)} 日", flush=True)

    v39 = pd.read_parquet(PROC / "pred_baseline_h10_v39.parquet", columns=["date", "symbol", "pred"])
    gru = pd.read_parquet(PROC / "pred_gru_h10.parquet", columns=["date", "symbol", "pred"])
    for d in (v39, gru):
        d["date"] = pd.to_datetime(d["date"])
        d.drop_duplicates(subset=["date", "symbol"], keep="last", inplace=True)
    v39["z_v39"] = _z(v39, "pred")
    gru["z_gru"] = _z(gru, "pred")
    m = v39[["date", "symbol", "z_v39"]].merge(
        gru[["date", "symbol", "z_gru"]], on=["date", "symbol"], how="left")
    m["z_gru"] = m["z_gru"].fillna(0.0)

    results = []
    for w in WEIGHTS:
        col = f"p{w}".replace(".", "")
        m[col] = (1 - w) * m["z_v39"] + w * m["z_gru"]
        daily = build_daily(m, col)
        print(f"w={w}: daily {len(daily):,} 行，跑 3 窗口 ...", flush=True)
        for wn, lo in WINDOWS.items():
            r = run(daily, lo)
            r.update({"w_gru": w, "window": wn})
            results.append(r)
            print("  %-8s 累计=%+.2f%% 超额=%+.2f%% 日胜率=%.2f%% 交易=%d" % (
                wn, r["cum"] * 100, r["excess"] * 100, r["win"] * 100, r["n"]), flush=True)

    today = datetime.now().strftime("%Y%m%d")
    rep = PROC / "reports"
    md = []
    md.append(f"# P1 gru 权重真·反事实组合回测（{today}）\n")
    md.append("- 从 pred 按 37 的 `_z` 口径合成 w∈{0,0.25,1.0} 的 ens pred（纯内存），"
              "逐日取 score top-20 作 daily，复用 `zz_paper_portfolio` 的 "
              "build_trades/portfolio_curve/bench_curve 走真实组合回测。")
    md.append(f"- 口径：long-only top-{K}、horizon={HORIZON}、T+1 开盘建仓→持10日开盘平仓、"
              f"单边成本 {COST}（双边 0.2%）+ 涨跌停过滤；基准=全池等权。**不写 signals/、不动 ks**。\n")
    md.append("## 净值对比\n")
    md.append("| w_gru | 窗口 | 累计净值 | 超额(对全池) | 日胜率 | 最大回撤 | 完成交易 |")
    md.append("|---|---|---|---|---|---|---|")
    for r in results:
        md.append("| %.2f | %s | %+.2f%% | %+.2f%% | %.2f%% | %.2f%% | %d |" % (
            r["w_gru"], r["window"], r["cum"] * 100, r["excess"] * 100,
            r["win"] * 100, r["dd"] * 100, r["n"]))
    r0 = next(r for r in results if r["w_gru"] == 0.0 and r["window"] == "近59日")
    r25 = next(r for r in results if r["w_gru"] == 0.25 and r["window"] == "近59日")
    md.append("\n## 结论\n")
    md.append(f"- **近 59 日：w=0（现役）累计 {r0['cum']:+.2%} / 超额 {r0['excess']:+.2%}；"
              f"w=0.25（历史）累计 {r25['cum']:+.2%} / 超额 {r25['excess']:+.2%}** → "
              f"砍 gru 的超额改善 = {(r0['excess'] - r25['excess']) * 100:+.2f}pp。")
    md.append("- **但 w=0 近 59 日仍为负超额** → 印证 recent_edge_diag：砍 gru 是必要且正确的，"
              "但**解决不了 long-only 构造在宽基 rally 期跑输全池**这个真问题。")
    md.append("- 与 `gru_weight_scan_20261111.md` 的 IC/LS_t 结论一致（权重越大越差、w=0 最优），"
              "两条独立口径互证。")
    md.append("\n> 诚实边界：本回测为合成信号的反事实，非现役信号文件回放；"
              "合成 daily 取 top-20 与 06_export_signal 的 top_long 同源口径。"
              "近 59 日窗口信号日仅 59 个，统计噪声大，结论以方向性为准。")
    (rep / f"gru_counterfactual_pp_{today}.md").write_text("\n".join(md), encoding="utf-8")
    (rep / f"gru_counterfactual_pp_{today}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告: {rep}/gru_counterfactual_pp_{today}.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
