"""阶段 ZZ：P1 信号回溯模拟盘绩效实证（paper portfolio backtest）。

把 P1 产出的五路 signal_*.json（gru / baseline / ens_v39gru_w025 / ev / fusion）
变成可核验的"信号 → 收益"实证：
  - 逐日按 score 降序取 top-k（与 06_export_signal 的 top_long 同源口径，已实测一致）；
  - 信号日 T 收盘生成 → T+1 交易日开盘价建仓（无前视）；
  - 持有 horizon 个交易日（json.horizon，目前均为 10）→ 第 horizon+1 个交易日开盘价平仓；
  - 逐日等权组合净值 + 全池等权基准，输出 胜率 / 年化 / 最大回撤 / 超额 等指标。

数据源：
  - 信号：E:/project/sj/data/P1/processed/signals/signal_*.json
  - 行情：E:/project/sj/data/P1/raw/kline/*.parquet（1492 只，baostock 日线，2018 起）

诚实标注（默认版本不计入，勿误读为可交易净值）：
  - 未计佣金 / 印花税 / 滑点（可用 --cost 按单边比例计入，线性近似）；
  - 未模拟涨跌停无法成交（可用 --limit-filter 过滤涨停开盘买不进 / 跌停开盘卖不出；
    沪深主板 10%、创业板 300/301 与科创板 688/689 20%，北交所 30% 未处理）；
  - 停牌按该股无记录自然顺延；
  - 复权口径以 kline 现状为准（open/close 为 baostock 原始字段）。

用法：
  python scripts/zz_paper_portfolio.py                          # 五路全跑，top-k=10
  python scripts/zz_paper_portfolio.py --top-k 5
  python scripts/zz_paper_portfolio.py --cost 0.0015 --limit-filter   # 现实口径：单边0.15%+涨跌停
  python scripts/zz_paper_portfolio.py --models gru fusion
  python scripts/zz_paper_portfolio.py --start-date 2026-07-28 --end-date 2026-10-12
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"E:/project/sj")
PROJ = ROOT / "P1-QuantFactor"
DATA = ROOT / "data" / "P1"
KLINE_DIR = DATA / "raw" / "kline"
SIGNAL_DIR = DATA / "processed" / "signals"
REPORT_DIR = DATA / "processed" / "reports"

DEFAULT_MODELS = ["gru", "baseline", "ens_v39gru_w025", "ev", "fusion"]
SIGNAL_FILES = {
    "gru": "signal_gru_h10.json",
    "baseline": "signal_baseline_h10.json",
    "ens_v39gru_w025": "signal_ens_v39gru_w025_h10.json",
    "ev": "signal_ev_h10.json",
    "fusion": "signal_fusion_h10.json",
}
TRADING_DAYS = 244  # 年化口径（A 股近似）


def load_prices() -> tuple[dict[str, pd.DataFrame], list[pd.Timestamp]]:
    """读取全部 kline 并构建 {symbol: df} 与全局交易日历。"""
    frames: dict[str, pd.DataFrame] = {}
    for f in sorted(KLINE_DIR.glob("*.parquet")):
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        # N16：kline 经桥接 append 可能含重复 date 行，去重(保留末条)避免
        # set_index("date") 出现非唯一索引 → bench_curve 内 .get(prev) 返回 Series 崩。
        df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        frames[str(df["symbol"].iloc[0])] = df[["date", "open", "close"]].copy()
    # 全局日历：所有股票日期并集（任一 A 股交易日至少有一只股票有记录）
    all_days = sorted({d for df in frames.values() for d in df["date"].tolist()})
    # 过滤单日覆盖过少的日期（隔离异常稀疏的日子，避免伪交易日）
    day_counts: dict[pd.Timestamp, int] = {}
    for df in frames.values():
        for d in df["date"].tolist():
            day_counts[d] = day_counts.get(d, 0) + 1
    calendar = [d for d in all_days if day_counts[d] >= 50]
    return frames, calendar


def pick_topk(daily: list[dict], date: str, k: int) -> list[str]:
    """某信号日的 top-k（score 降序，与 top_long 同源口径）。"""
    rows = [d for d in daily if d["date"] == date]
    rows.sort(key=lambda x: -x["score"])
    return [r["symbol"] for r in rows[:k]]


def limit_pct(sym: str) -> float:
    """涨跌停幅度：创业板(300/301)/科创板(688/689) 20%，其余沪深主板 10%。北交所未处理。"""
    return 0.20 if sym.startswith(("300", "301", "688", "689")) else 0.10


def build_trades(frames: dict, daily: list[dict], k: int, horizon: int,
                 start_date: str | None = None, end_date: str | None = None,
                 cost: float = 0.0, limit_filter: bool = False) -> list[dict]:
    """构造全部交易：信号日 T → T+1 交易日开盘建仓 → 持有 horizon 交易日 → 开盘平仓。

    start_date / end_date（'YYYY-MM-DD'，含）用于把多路信号对齐到同一信号日区间后横比。
    cost：单边交易成本比例（线性近似，建仓/平仓各扣一次，h_ret 计 2*cost）。
    limit_filter：涨停开盘买不进（跳过建仓）/ 跌停开盘卖不出（跳过平仓）。
    """
    trades: list[dict] = []
    dates = sorted({d["date"] for d in daily})
    if start_date:
        dates = [d for d in dates if d >= start_date]
    if end_date:
        dates = [d for d in dates if d <= end_date]
    for t in dates:
        for sym in pick_topk(daily, t, k):
            df = frames.get(sym)
            if df is None:
                continue
            t_dt = pd.Timestamp(t)
            idx = df["date"].searchsorted(t_dt, side="right")  # 首个 date > T
            if idx >= len(df):
                trades.append({"signal_date": t, "symbol": sym, "incomplete": True,
                               "reason": "无后续行情"})
                continue
            entry = df.iloc[idx]
            # 涨跌停过滤：涨停开盘买不进
            if limit_filter and idx > 0:
                prev_c = df["close"].iloc[idx - 1]
                if prev_c > 0 and entry["open"] / prev_c - 1.0 >= limit_pct(sym) - 1e-6:
                    trades.append({"signal_date": t, "symbol": sym, "incomplete": True,
                                   "reason": "涨停开盘买不进"})
                    continue
            exit_pos = idx + horizon  # 持有 horizon 个交易日 → 下一个交易日开盘卖
            if exit_pos >= len(df):
                trades.append({"signal_date": t, "symbol": sym, "incomplete": True,
                               "reason": "后续交易日不足"})
                continue
            exit_row = df.iloc[exit_pos]
            # 涨跌停过滤：跌停开盘卖不出
            if limit_filter and exit_pos > 0:
                prev_c = df["close"].iloc[exit_pos - 1]
                if prev_c > 0 and exit_row["open"] / prev_c - 1.0 <= -limit_pct(sym) + 1e-6:
                    trades.append({"signal_date": t, "symbol": sym, "incomplete": True,
                                   "reason": "跌停开盘卖不出"})
                    continue
            h_ret = exit_row["open"] / entry["open"] - 1.0 - 2.0 * cost
            trades.append({
                "signal_date": t, "symbol": sym,
                "entry_date": str(entry["date"].date()), "entry_open": float(entry["open"]),
                "exit_date": str(exit_row["date"].date()), "exit_open": float(exit_row["open"]),
                "h_ret": float(h_ret), "cost": cost, "incomplete": False, "reason": "",
            })
    return trades


def portfolio_curve(frames: dict, trades: list[dict], calendar: list[pd.Timestamp],
                    horizon: int, cost: float = 0.0) -> pd.Series:
    """逐日等权组合净值：持仓区间内逐日收益（建仓日 open→close、中间 close→close、平仓日 prev_close→open）。

    cost：单边成本比例，建仓日与平仓日各线性扣一次。
    """
    cal_idx = {d: i for i, d in enumerate(calendar)}
    daily_ret: dict[int, list[float]] = {}
    for tr in trades:
        if tr["incomplete"]:
            continue
        df = frames.get(tr["symbol"])
        if df is None:
            continue
        b = cal_idx.get(pd.Timestamp(tr["entry_date"]))
        if b is None:
            continue
        loc = df["date"].searchsorted(pd.Timestamp(tr["entry_date"]), side="left")
        s = cal_idx.get(pd.Timestamp(tr["exit_date"]))
        if s is None or s > len(calendar) - 1:
            continue
        # 建仓日（open→close，扣买入成本）
        daily_ret.setdefault(b, []).append(df["close"].iloc[loc] / df["open"].iloc[loc] - 1.0 - cost)
        # 中间持有日
        for i in range(b + 1, s):
            r = None
            for j in range(loc + 1, len(df)):
                d_j = cal_idx.get(df["date"].iloc[j])
                if d_j is None or d_j < i:
                    continue
                if d_j == i:
                    r = df["close"].iloc[j] / df["close"].iloc[j - 1] - 1.0
                    break
                break
            if r is not None:
                daily_ret.setdefault(i, []).append(r)
        # 平仓日（prev_close → open，扣卖出成本）
        # 找到平仓日前后两行
        rows = df[(df["date"] >= pd.Timestamp(tr["entry_date"]))].reset_index(drop=True)
        if len(rows) > horizon:
            prev_close = rows["close"].iloc[horizon - 1]
            exit_open = rows["open"].iloc[horizon]
            daily_ret.setdefault(s, []).append(exit_open / prev_close - 1.0 - cost)
    n = len(calendar)
    ret = pd.Series(0.0, index=pd.RangeIndex(n))
    for i, vals in daily_ret.items():
        if 0 <= i < n and vals:
            ret.iloc[i] = float(np.mean(vals))
    return (1.0 + ret).cumprod()


def bench_curve(frames: dict, calendar: list[pd.Timestamp]) -> pd.Series:
    """全池等权基准净值（close 日收益）。"""
    closes = {sym: df.set_index("date")["close"] for sym, df in frames.items()}
    rets: list[float] = []
    prev = None
    for d in calendar:
        if prev is not None:
            cur = {s: ser.get(d) for s, ser in closes.items()}
            vals = []
            for s, c in cur.items():
                p = closes[s].get(prev)
                if c is not None and p is not None and p > 0:
                    vals.append(c / p - 1.0)
            rets.append(float(np.mean(vals)) if vals else 0.0)
        else:
            rets.append(0.0)
        prev = d
    r = pd.Series(rets)
    return (1.0 + r).cumprod()


def metrics(nav: pd.Series, bench: pd.Series) -> dict:
    n = len(nav)
    if n < 2:
        return {"n_days": int(n), "cum_ret": 0.0, "annual": 0.0, "max_dd": 0.0,
                "daily_win": 0.0, "excess": 0.0}
    cum = float(nav.iloc[-1] - 1.0)
    annual = (1.0 + cum) ** (TRADING_DAYS / (n - 1)) - 1.0 if cum > -1 else -1.0
    dd = float((nav / nav.cummax() - 1.0).min())
    daily_r = nav.pct_change().dropna()
    win = float((daily_r > 0).mean())
    excess = float(nav.iloc[-1] - bench.iloc[-1]) if len(bench) == n else float("nan")
    return {"n_days": int(n), "cum_ret": round(cum, 4), "annual": round(annual, 4),
            "max_dd": round(dd, 4), "daily_win": round(win, 4), "excess": round(excess, 4)}


def run_model(model: str, frames: dict, calendar: list[pd.Timestamp],
              k: int, min_history: int, start_date: str | None = None,
              end_date: str | None = None, cost: float = 0.0,
              limit_filter: bool = False) -> dict:
    f = SIGNAL_DIR / SIGNAL_FILES[model]
    sig = json.loads(f.read_text(encoding="utf-8"))
    daily = sig["daily"]
    horizon = int(sig.get("horizon", 10))
    trades = build_trades(frames, daily, k, horizon, start_date, end_date, cost, limit_filter)
    done = [t for t in trades if not t["incomplete"]]
    h_rets = [t["h_ret"] for t in done]
    nav = portfolio_curve(frames, done, calendar, horizon, cost)
    bench = bench_curve(frames, calendar)
    # 区间对齐：从首笔建仓日开始
    start_i = 0
    if done:
        start_d = pd.Timestamp(min(t["entry_date"] for t in done))
        start_i = calendar.index(start_d) if start_d in calendar else 0
    nav_a = nav.iloc[start_i:] / (nav.iloc[start_i] if nav.iloc[start_i] != 0 else 1.0)
    bench_a = bench.iloc[start_i:] / (bench.iloc[start_i] if bench.iloc[start_i] != 0 else 1.0)
    m = metrics(nav_a, bench_a)
    sig_dates = sorted({d["date"] for d in daily})
    if start_date:
        sig_dates = [d for d in sig_dates if d >= start_date]
    if end_date:
        sig_dates = [d for d in sig_dates if d <= end_date]
    res = {
        "model": model, "horizon": horizon, "top_k": k,
        "cost": cost, "limit_filter": limit_filter,
        "signal_days": len(sig_dates),
        "latest_signal": sig_dates[-1] if sig_dates else None,
        "start_date": start_date, "end_date": end_date,
        "trades_total": len(trades), "trades_done": len(done),
        "trades_incomplete": len(trades) - len(done),
        "skip_counts": {r: sum(1 for t in trades if t["incomplete"] and t["reason"] == r)
                        for r in sorted({t["reason"] for t in trades if t["incomplete"]})},
        "h10_win_rate": round(float(np.mean([r > 0 for r in h_rets])), 4) if h_rets else None,
        "h10_avg": round(float(np.mean(h_rets)), 4) if h_rets else None,
        "h10_median": round(float(np.median(h_rets)), 4) if h_rets else None,
        "metrics": m,
        "min_history": min_history,
        "nav": [[str(calendar[start_i + i].date()), round(float(nav_a.iloc[i]), 4),
                 round(float(bench_a.iloc[i]), 4)] for i in range(len(nav_a))],
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 信号回溯模拟盘绩效实证")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--min-history", type=int, default=0,
                    help="信号日之后至少需要的交易日数（0 表示不限制，不足者计 incomplete）")
    ap.add_argument("--start-date", default=None, help="信号日区间起点 YYYY-MM-DD（含）")
    ap.add_argument("--end-date", default=None, help="信号日区间终点 YYYY-MM-DD（含）")
    ap.add_argument("--cost", type=float, default=0.0,
                    help="单边交易成本比例（如 0.0015=0.15%，双边计 2*cost，线性近似）")
    ap.add_argument("--limit-filter", action="store_true",
                    help="涨跌停过滤：涨停开盘买不进跳过建仓、跌停开盘卖不出跳过平仓（主板10%/双创20%）")
    args = ap.parse_args()

    frames, calendar = load_prices()
    print(f"kline: {len(frames)} 股 / 交易日历 {len(calendar)} 日 "
          f"({calendar[0].date()} ~ {calendar[-1].date()})"
          + (f" / 信号区间 {args.start_date}~{args.end_date}" if args.start_date or args.end_date else ""))

    results = []
    for m in args.models:
        if m not in SIGNAL_FILES:
            print(f"[skip] 未知模型 {m}")
            continue
        r = run_model(m, frames, calendar, args.top_k, args.min_history,
                      args.start_date, args.end_date, args.cost, args.limit_filter)
        results.append(r)
        mt = r["metrics"]
        skips = r["skip_counts"]
        skip_buy = skips.get("涨停开盘买不进", 0)
        print(f"[{r['model']:<14}] 信号 {r['latest_signal']} | 交易 {r['trades_done']}/{r['trades_total']} "
              f"| 累计 {mt['cum_ret']:+.2%} | 年化 {mt['annual']:+.2%} | 回撤 {mt['max_dd']:.2%} "
              f"| 日胜率 {mt['daily_win']:.2%} | 超额 {mt['excess']:+.2%} "
              f"| h10胜率 {r['h10_win_rate'] if r['h10_win_rate'] is not None else float('nan'):.2%}"
              + (f" | 涨停跳过 {skip_buy}" if args.limit_filter else ""))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = {"generated_at": stamp, "top_k": args.top_k,
           "n_kline": len(frames), "calendar_days": len(calendar),
           "models": results}
    op = REPORT_DIR / f"paper_portfolio_{stamp}.json"
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
