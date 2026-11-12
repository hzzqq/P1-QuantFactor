"""P1 变现层诊断（2026-11-11）：为什么 IC≈0.06 但净值近期为负。

复用 zz_paper_portfolio 已验证的回测函数（load_prices/build_trades/portfolio_curve/bench_curve），
一次性读 kline 后，对生产信号在多个成本档位与涨跌停过滤下重算净值，量化：
  1. 成本拖累：cost=0(毛) vs cost=0.0005/0.001/0.0015/0.002 + 涨跌停 的累计/超额；
  2. 换手率：top-k=10、horizon=10、信号每日生成 → 单边日换手≈1/horizon≈10%/日；
  3. gru 腿拖累：gru / baseline / ev / ens / fusion 各自的净值横比（2026 与 2024+）。

输出 reports/monetization_diag_YYYYMMDD.md。
"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("E:/project/sj/P1-QuantFactor")
spec = importlib.util.spec_from_file_location(
    "zz_pp", str(PROJ / "scripts" / "zz_paper_portfolio.py"))
pp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pp)

TOPK = 10
COST_LEVELS = [0.0, 0.0005, 0.001, 0.0015, 0.002]
WINDOWS = {"2024+": ("2024-01-01", None), "2026": ("2026-01-01", None)}
MODELS = ["gru", "baseline", "ev", "ens_v39gru_w025", "fusion"]


def run_model(model: str, start, end, cost: float, limit: bool) -> dict:
    sig = json.loads((pp.SIGNAL_DIR / pp.SIGNAL_FILES[model]).read_text(encoding="utf-8"))
    daily = sig["daily"]
    horizon = int(sig.get("horizon", 10))
    trades = pp.build_trades(pp.frames, daily, TOPK, horizon, start, end, cost, limit)
    done = [t for t in trades if not t["incomplete"]]
    nav = pp.portfolio_curve(pp.frames, done, pp.calendar, horizon, cost)
    bench = pp.bench_curve(pp.frames, pp.calendar)
    if done:
        sd = pd.Timestamp(min(t["entry_date"] for t in done))
        si = pp.calendar.index(sd) if sd in pp.calendar else 0
    else:
        si = 0
    nav_a = nav.iloc[si:] / (nav.iloc[si] or 1.0)
    bench_a = bench.iloc[si:] / (bench.iloc[si] or 1.0)
    m = pp.metrics(nav_a, bench_a)
    # 换手率（分析式）：单边日换手≈1/horizon；年化换手≈244/horizon
    one_way_daily = 1.0 / horizon
    return {"model": model, "cost": cost, "limit": limit, "horizon": horizon,
            "n_trades": len(done), "metrics": m,
            "one_way_daily_turnover": one_way_daily,
            "annual_turnover": 244.0 / horizon}


def main() -> int:
    print("读取 kline ...")
    pp.frames, pp.calendar = pp.load_prices()
    print(f"kline {len(pp.frames)} 股 / 日历 {len(pp.calendar)} 日")

    rows = []
    for wname, (lo, hi) in WINDOWS.items():
        for cost in COST_LEVELS:
            r = run_model("ens_v39gru_w025", lo, hi, cost, limit=cost > 0)
            m = r["metrics"]
            rows.append({
                "window": wname, "cost": cost, "limit": cost > 0,
                "cum_net": m["cum_ret"], "annual_net": m["annual"],
                "excess_net": m["excess"], "dd": m["max_dd"],
                "win": m["daily_win"], "n_trades": r["n_trades"],
            })

    # 各模型横比（cost=0.001 + 涨跌停，现实口径；两窗口）
    per_model = {}
    for wname, (lo, hi) in WINDOWS.items():
        per_model[wname] = {}
        for mod in MODELS:
            r = run_model(mod, lo, hi, 0.001, True)
            m = r["metrics"]
            per_model[wname][mod] = {"cum": m["cum_ret"], "annual": m["annual"],
                                     "excess": m["excess"], "dd": m["max_dd"],
                                     "win": m["daily_win"], "n": r["n_trades"]}

    # 报告
    today = datetime.now().strftime("%Y%m%d")
    rep = Path("E:/project/sj/data/P1/processed/reports")
    rep.mkdir(parents=True, exist_ok=True)

    md = []
    md.append(f"# P1 变现层诊断（{today}）\n")
    md.append("- 策略口径：long-only top-{0}，horizon=10，信号日 T→T+1 开盘建仓→持 10 日开盘平仓（无前视）；"
              "净值=持仓逐日等权收益累乘；基准=全池等权。".format(TOPK))
    md.append("- **默认 S12 是毛口径（cost=0、无涨跌停）**；本诊断补上现实成本与涨跌停，量化拖累。\n")

    md.append("## 一、ens 信号：成本档位扫描（毛 vs 现实）\n")
    md.append("| 窗口 | 单边成本 | 涨跌停 | 累计净值 | 年化 | 超额(对基准) | 最大回撤 | 日胜率 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for row in rows:
        md.append("| {window} | {cost:.4f} | {lim} | {cum:+.2%} | {ann:+.2%} | {exc:+.2%} | {dd:.2%} | {win:.2%} |".format(
            window=row["window"], cost=row["cost"], lim="是" if row["limit"] else "否",
            cum=row["cum_net"], ann=row["annual_net"], exc=row["excess_net"],
            dd=row["dd"], win=row["win"]))
    md.append("\n> 单边成本 0.001≈0.1%（佣+印花+滑点约 0.2% 双边）；0.0015≈0.3% 双边。")

    md.append("\n## 二、换手率（分析式，与 top-k/horizon 绑定）\n")
    md.append("- 单边日换手 ≈ 1/horizon = **1/10 = 10%/日**；年化单边换手 ≈ 244/10 ≈ **24.4 次/年**；"
              "双边成本拖累 ≈ 年化换手 × 双边成本 = 24.4 × 2×cost。")
    md.append("- cost=0.001（双边0.2%）：年化成本拖累 ≈ **4.9%**；cost=0.0015（双边0.3%）：≈ **7.3%**。"
              "弱信号（年化 alpha 个位数%）被此吞噬。")

    md.append("\n## 三、各模型腿横比（cost=0.001 + 涨跌停，现实口径）\n")
    md.append("| 模型 | 2024+ 累计 | 2024+ 超额 | 2026 累计 | 2026 超额 | 2026 日胜率 |")
    md.append("|---|---|---|---|---|---|")
    for mod in MODELS:
        a = per_model["2024+"][mod]; b = per_model["2026"][mod]
        md.append("| {mod} | {ac:+.2%} | {ae:+.2%} | {bc:+.2%} | {be:+.2%} | {bw:.2%} |".format(
            mod=mod, ac=a["cum"], ae=a["excess"], bc=b["cum"], be=b["excess"], bw=b["win"]))
    md.append("\n> gru 腿在 2026 若显著差于 ens，即说明 gru 拖累集成；baseline/ev 若优于 ens，说明应降低 gru 权重。")

    md.append("\n## 四、结论要点\n")
    ens_a_net = per_model["2026"]["ens_v39gru_w025"]["excess"]
    gru_26 = per_model["2026"]["gru"]["excess"]
    base_26 = per_model["2026"]["baseline"]["excess"]
    ev_26 = per_model["2026"]["ev"]["excess"]
    gross_2026 = next(r["excess_net"] for r in rows if r["window"] == "2026" and r["cost"] == 0.0)
    md.append(f"- **成本拖累是硬伤**：ens 在 2026 窗口，毛口径（cost=0）超额={gross_2026:+.2%}；"
              "现实成本(0.001+涨跌停)后进一步下滑；叠加 ~4.9%–7.3% 年化成本拖累，弱信号被吞噬。")
    md.append(f"- **gru 腿拖累**：2026 超额 gru={gru_26:+.2%} / baseline={base_26:+.2%} / ev={ev_26:+.2%} / "
              f"ens={ens_a_net:+.2%} —— gru 若最差则确为拖累腿。")
    md.append("- **IC→P&L 断层根因**：IC(对基准的秩相关)为正 ≠ 净值对全池基准超额为正；本策略 long-only top-k，"
              "若宽基/全池在样本期跑赢被选中的反转/动量股，则超额为负（即便 IC 正）。叠加高换手成本 → 净值必负。")
    md.append("\n## 五、修正后的可执行项\n")
    md.append("1. **降换手**：把每日全轮换改为更低频（如每 5/10 日调仓）或降 top-k/延长 horizon，直接砍成本拖累。")
    md.append("2. **gru 腿降权/替换**：若 gru 2026 最差，从 ens 权重中移除或降权。")
    md.append("3. **组合构建改 full-long-short 或 低换手等权**：匹配 IC 测的是相对排序，long-only-top-k 未必是最优变现载体。")

    (rep / f"monetization_diag_{today}.md").write_text("\n".join(md), encoding="utf-8")
    (rep / f"monetization_diag_{today}.json").write_text(
        json.dumps({"rows": rows, "per_model": per_model}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    # 控制台
    print("\n=== ens 成本扫描 ===")
    for row in rows:
        print("  %s cost=%.4f%s 累计=%+.2f%% 超额=%+.2f%% dd=%+.2f%%" % (
            row["window"], row["cost"], " L" if row["limit"] else "  ",
            row["cum_net"] * 100, row["excess_net"] * 100, row["dd"] * 100))
    print("\n=== 各模型 2026 (cost=0.001+L) ===")
    for mod in MODELS:
        b = per_model["2026"][mod]
        print("  %-18s 累计=%+.2f%% 超额=%+.2f%% 日胜率=%.2f%%" % (
            mod, b["cum"] * 100, b["excess"] * 100, b["win"] * 100))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
