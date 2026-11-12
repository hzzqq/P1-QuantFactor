"""组合构建改造诊断（2026-11-11）。

背景：因子/IC 复盘 + 变现层 + gru 腿诊断已定位——近期弱势的真问题是
「IC 不兑现成 P&L」：long-only top-k 在 2026 因 LongXS_t 转负（全池普涨、
模型多头跑输全市场均值）而亏损；但多空边缘 LS_t 仍 +2.07（相对排序仍有效）。

本脚本回答用户批准的改造方向「**基准相对 / 市场中性 + 降频低换手**」：

1. 市场中性（long_short = 0.5*多 + 0.5*空，beta 中性）能否绕过 long-only 的
   全市场 beta 拖累，在 2026 / 近59交易日 把 LS 边缘兑现成收益？
2. 降频低换手（提高 rebalance_freq + 加大 buffer 连续持有）能否在 2026 降低
   成本拖累、改善净收益，且不在 2024+ 把有效信号磨掉（36 脚本警告的过拟合陷阱）？
3. 基准相对（long-only gross 减等权全市场 gross）量化 2026 的 alpha 是否仍为负。

严格遵循 36 脚本教训：不在同一份数据上挑峰值参数，而是**分时段**报告，
看每个构造在 FULL / 2024+ / 2026 / 近59交易日 是否一致有效。

用法：
    python scripts/zz_portfolio_construction_diag.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths  # noqa: E402
from src.backtest import (run_backtest_continuous, DEFAULT_COST)  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
HORIZON = 10
TOP_PCT = 0.1

ZERO_COST = {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}

MODES = ["long_only", "long_short"]
RFS = [10, 20, 40]
BUFFERS = [0.0, 0.1]


def make_windows(recent_start: str) -> dict:
    """固定四个可比时段（均按日历日切片，跨 rf 可比）。"""
    return {
        "FULL": (None, None),
        "2024+": ("2024-01-01", None),
        "2026": ("2026-01-01", None),
        "recent59td": (recent_start, None),  # 最近 59 交易日（真·近期弱势窗）
    }


def segment_stats(equity: pd.Series, dates, lo=None, hi=None,
                  last_n: int | None = None, rf: int = 10) -> dict | None:
    """从 cumprod equity 序列切片统计一段窗口的累计/年化/夏普/回撤。"""
    d = pd.to_datetime(dates)
    eq = equity.copy()
    eq.index = d
    if last_n is not None:
        idx = eq.index[-last_n:]
    else:
        mask = pd.Series(True, index=eq.index)
        if lo is not None:
            mask &= (eq.index >= pd.Timestamp(lo))
        if hi is not None:
            mask &= (eq.index <= pd.Timestamp(hi))
        idx = eq.index[mask]
    sub = eq.reindex(idx)
    sub = sub[sub.notna()]
    if len(sub) < 2:
        return None
    rets = sub.pct_change().dropna().values.astype(float)
    if len(rets) < 2 or rets.std() == 0:
        return None
    total = float((1 + rets).prod() - 1)
    n = len(rets)
    years = n * rf / 252.0
    ann = float((1 + total) ** (1 / max(years, 1e-9)) - 1) if years > 0 else float("nan")
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252 / rf))
    peak = np.cumprod(1 + rets)
    mdd = float((peak / np.maximum.accumulate(peak) - 1).min())
    return {
        "total_return": round(total, 4),
        "annual_return": round(ann, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(mdd, 4),
        "n_buckets": int(n),
    }


def seg_all(equity, dates, rf, windows):
    out = {}
    for name, (lo, hi) in windows.items():
        out[name] = segment_stats(equity, dates, lo=lo, hi=hi, rf=rf)
    return out


def run_once(preds, panel, mode, rf, buffer, top_pct, cost):
    return run_backtest_continuous(
        preds, panel, horizon=HORIZON, top_pct=top_pct,
        rebalance_freq=rf, mode=mode, buffer=buffer, cost=cost)


def _s(rec, side, wname):
    d = rec.get(side, {}).get(wname)
    if not d:
        return "n/a"
    return f"{d.get('total_return', float('nan')):+.1%}/夏普{d.get('sharpe', float('nan')):+.2f}"


def main() -> int:
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    print(f"[diag] panel {len(panel):,} 行 / {panel['symbol'].nunique()} 只", flush=True)
    all_dates = pd.to_datetime(sorted(panel["date"].unique()))
    recent_start = str(all_dates[-59].date())  # 最近 59 交易日起点（日历日）
    windows = make_windows(recent_start)
    print(f"[diag] 近期弱势窗 recent59td = {recent_start} ~ {all_dates[-1].date()}", flush=True)

    ens = pd.read_parquet(PROCESSED / "pred_ens_v39gru_w025_h10.parquet")
    ens["date"] = pd.to_datetime(ens["date"])
    print(f"[diag] ENS {len(ens):,} 行 / {ens['symbol'].nunique()} 只 | "
          f"{ens['date'].min().date()}~{ens['date'].max().date()}", flush=True)

    results = []

    # ---- 等权全市场基准（gross，top_pct=1.0，零成本）---- 作为「市场收益」参照
    bench = {}
    for rf in RFS:
        b_full = run_backtest_continuous(
            ens, panel, horizon=HORIZON, top_pct=1.0,
            rebalance_freq=rf, mode="long_only", buffer=0.0, cost=ZERO_COST)
        # 给基准 long_equity 补上日期索引（引擎未带）
        b_full.long_equity = b_full.long_equity.copy()
        b_full.long_equity.index = pd.to_datetime(b_full.equity_dates)
        bench[rf] = b_full
        print(f"[diag] 基准(等权全市场,gross) rf={rf} 累计="
              f"{b_full.long_stats.get('total_return', float('nan')):+.2%} | "
              f"夏普={b_full.long_stats.get('sharpe', float('nan')):+.2f}", flush=True)

    # ---- 主网格：mode × rf × buffer，net + gross ----
    grid = [(m, rf, bf) for m in MODES for rf in RFS for bf in BUFFERS]
    total = len(grid) * 2
    done = 0
    for mode in MODES:
        for rf in RFS:
            for bf in BUFFERS:
                for tag, cost in (("net", DEFAULT_COST), ("gross", ZERO_COST)):
                    done += 1
                    try:
                        r = run_once(ens, panel, mode, rf, bf, TOP_PCT, cost)
                        ls = r.ls_stats or {}
                        rec = {
                            "mode": mode, "rf": rf, "buffer": bf, "cost": tag,
                            "top_pct": TOP_PCT,
                            "avg_turnover": round(float(ls.get("avg_turnover", 0.0)), 4),
                            "n_buckets_full": int(ls.get("n_buckets", 0) or 0),
                            "ls": seg_all(r.equity, r.equity_dates, rf, windows),
                            "long": seg_all(r.long_equity, r.equity_dates, rf, windows),
                            "short": seg_all(r.short_equity, r.equity_dates, rf, windows),
                        }
                        results.append(rec)
                        lab = f"{mode:>10} rf={rf:<3} buf={bf:<4} {tag:<5}"
                        print(f"  [{done:2d}/{total}] {lab} | 换手 {rec['avg_turnover']:.0%} | "
                              f"FULL {_s(rec,'ls','FULL')} | 2026 {_s(rec,'ls','2026')} | "
                              f"近59td {_s(rec,'ls','recent59td')}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [{done:2d}/{total}] {mode} rf={rf} buf={bf} {tag} 失败: {e!r}",
                              flush=True)

    # ---- 先落盘网格（防止后续段崩溃丢数据）----
    grid_out = PROCESSED / "reports" / "portfolio_construction_diag_20261111.json"
    grid_out.write_text(json.dumps({
        "horizon": HORIZON, "top_pct": TOP_PCT,
        "pred": "pred_ens_v39gru_w025_h10.parquet (w_gru=0, == z_v39)",
        "date_range": [str(ens["date"].min().date()), str(ens["date"].max().date())],
        "windows": list(windows.keys()), "window_defs": windows,
        "recent_start": recent_start,
        "grid_results": results,
        "benchmark_relative": "pending",
        "note": "long_short = 市场中性(0.5多+0.5空)；long_only = 纯多头；"
                "基准相对 = 纯多头gross − 等权全市场gross(同rf)；"
                "net 含 A 股真实成本，gross 零成本。",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[diag] 网格已落盘: {grid_out}", flush=True)

    # ---- 基准相对：long-only gross 减 等权全市场 gross（同 rf）----
    print("[diag] 计算基准相对（long-only gross − 等权全市场 gross）...", flush=True)
    rel_rows = []
    for rf in RFS:
        lo_g = run_once(ens, panel, "long_only", rf, 0.0, TOP_PCT, ZERO_COST)
        le = lo_g.long_equity.copy()
        le.index = pd.to_datetime(lo_g.equity_dates)
        be = bench[rf].long_equity         # 已带日期索引
        common = le.index.intersection(be.index)
        le = le.reindex(common)
        be = be.reindex(common)
        excess = (le / be) - 1.0           # 累计超额净值（起点≈0）
        for wname, (lo, hi) in windows.items():
            m = pd.Series(True, index=excess.index)
            if lo:
                m &= (excess.index >= pd.Timestamp(lo))
            if hi:
                m &= (excess.index <= pd.Timestamp(hi))
            sub = excess.reindex(excess.index[m]).dropna()
            rel_rows.append({
                "rf": rf, "window": wname,
                "excess_total_return": round(float(sub.iloc[-1] / sub.iloc[0] - 1.0)
                                            if len(sub) > 1 else float("nan"), 4),
                "excess_end": round(float(sub.iloc[-1]) if len(sub) else float("nan"), 4),
            })
        rr = [r for r in rel_rows if r["rf"] == rf and r["window"] == "FULL"][0]
        print(f"  rf={rf} 基准相对(FULL 累计超额) = {rr['excess_total_return']:+.2%}", flush=True)

    # ---- 最终落盘 ----
    out = {
        "horizon": HORIZON, "top_pct": TOP_PCT,
        "pred": "pred_ens_v39gru_w025_h10.parquet (w_gru=0, == z_v39)",
        "date_range": [str(ens["date"].min().date()), str(ens["date"].max().date())],
        "windows": list(windows.keys()), "window_defs": windows,
        "recent_start": recent_start,
        "grid_results": results,
        "benchmark_relative": rel_rows,
        "note": "long_short = 市场中性(0.5多+0.5空)；long_only = 纯多头；"
                "基准相对 = 纯多头gross − 等权全市场gross(同rf)；"
                "net 含 A 股真实成本，gross 零成本。",
    }
    grid_out.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")
    print(f"[diag] 最终结果已保存: {grid_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
