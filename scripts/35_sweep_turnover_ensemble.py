"""阶段 35：A(换手/成本优化) + B(GRU×v39 低相关 ensemble) 联合网格扫描。

背景与动机（2026-09-14）：
- v39 生产基线用 --rebalance-freq 15 回测得净夏普 0.78；但 05_backtest.py 明确记载
  rf=15 与 horizon=10 错配是历史误设（"把信号压成负"），rf=10 才是正确对齐。
  → A 的核心：把 rf / top_pct / buffer（连续持仓降换手）扫一遍，看能捞回多少。
- GRU 与 LightGBM 相关性仅 0.204（2026），两者净夏普都在 0.78 量级 → 低相关叠加有空间。
  → B 的核心：截面 z-score 后按权重 w 混合，扫 w 找最优。
- 公平性：GRU 覆盖期短于 v39，故统一裁到两者日期交集，避免"easy years"虚增。

用法：
    python scripts/35_sweep_turnover_ensemble.py
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
from src.backtest import (run_backtest, run_backtest_continuous,  # noqa: E402
                          DEFAULT_COST)

PROCESSED = paths.DATA / "P1" / "processed"
OUT_JSON = PROCESSED / "report_sweep_turnover_ensemble.json"

HORIZON = 10
MODE = "long_short"


def _pred_col(df: pd.DataFrame) -> str:
    """自动识别预测列（排除 date/symbol 等键列）。"""
    for c in ("pred", "y_pred", "score", "yhat"):
        if c in df.columns:
            return c
    keys = {"date", "symbol", "code", "y", "label", "ret", "target"}
    for c in df.columns:
        if c.lower() not in keys:
            return c
    raise RuntimeError(f"无法识别预测列: {list(df.columns)}")


def _zscore_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    """按日期截面 z-score（同一天横截面标准化），并裁剪极端值。"""
    g = df.groupby("date")[col]
    z = (df[col] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
    return z.fillna(0.0).clip(-3, 3)


def _run(preds: pd.DataFrame, panel: pd.DataFrame, rf: int, top_pct: float,
         buffer: float) -> dict:
    """跑一次回测（net + gross），返回关键指标。"""
    common = dict(horizon=HORIZON, top_pct=top_pct, rebalance_freq=rf, mode=MODE)
    if buffer > 0:
        net = run_backtest_continuous(preds, panel, **common, cost=DEFAULT_COST,
                                      buffer=buffer)
        gross = run_backtest_continuous(preds, panel, **common,
                                        cost={"commission": 0.0, "slippage": 0.0,
                                              "stamp": 0.0}, buffer=buffer)
    else:
        net = run_backtest(preds, panel, **common, cost=DEFAULT_COST)
        gross = run_backtest(preds, panel, **common,
                             cost={"commission": 0.0, "slippage": 0.0, "stamp": 0.0})
    n = net.ls_stats or {}
    g = gross.ls_stats or {}
    return {
        "rf": rf,
        "top_pct": top_pct,
        "buffer": buffer,
        "net_sharpe": round(float(n.get("sharpe", float("nan"))), 4),
        "net_total_return": round(float(n.get("total_return", float("nan"))), 4),
        "net_annual": round(float(n.get("annual_return", float("nan"))), 4),
        "net_mdd": round(float(n.get("max_drawdown", float("nan"))), 4),
        "win_rate": round(float(n.get("win_rate", float("nan"))), 4),
        "gross_sharpe": round(float(g.get("sharpe", float("nan"))), 4),
        "n_trades": int(getattr(net, "n_trades", 0) or 0),
        "n_buckets": int(n.get("n_buckets", 0) or 0),
    }


def main() -> int:
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    print(f"[35] panel {len(panel):,} 行 | {panel['symbol'].nunique()} 只", flush=True)

    v39 = pd.read_parquet(PROCESSED / "pred_baseline_h10_v39.parquet")
    gru = pd.read_parquet(PROCESSED / "pred_gru_h10.parquet")
    c39, cgru = _pred_col(v39), _pred_col(gru)
    print(f"[35] v39 预测列={c39}  行={len(v39):,}；GRU 预测列={cgru}  行={len(gru):,}",
          flush=True)

    for d in (v39, gru):
        d["date"] = pd.to_datetime(d["date"])

    # 关键：pred_gru_h10 是多个 walk-forward 分期预测拼接的，(date,symbol) 会重叠，
    # 重复标签会让回测内的 rank/reindex 直接崩（实测 435 万行 vs 上限 ~131 万行）。
    def _dedup(df: pd.DataFrame, name: str) -> pd.DataFrame:
        n0 = len(df)
        out = df.drop_duplicates(subset=["date", "symbol"], keep="last")
        if len(out) != n0:
            print(f"[35]   {name} 去重(date,symbol)：{n0:,} -> {len(out):,}", flush=True)
        return out

    v39 = _dedup(v39, "v39")
    gru = _dedup(gru, "GRU")

    # 公平区间：两者日期交集（GRU 覆盖期短，避免 v39 用 easy years 虚增）
    common_dates = sorted(set(v39["date"].unique()) & set(gru["date"].unique()))
    print(f"[35] 日期交集 {len(common_dates)} 天："
          f"{common_dates[0].date()} ~ {common_dates[-1].date()}", flush=True)
    v39c = _dedup(v39[v39["date"].isin(common_dates)].copy(), "v39(裁剪后)")
    gruc = _dedup(gru[gru["date"].isin(common_dates)].copy(), "GRU(裁剪后)")
    print(f"[35] 区间内：v39 {len(v39c):,} 行/{v39c['symbol'].nunique()} 只；"
          f"GRU {len(gruc):,} 行/{gruc['symbol'].nunique()} 只", flush=True)

    # 截面 z-score（混合前各自的量纲归一化）
    v39c["z"] = _zscore_by_date(v39c, c39)
    gruc["z"] = _zscore_by_date(gruc, cgru)
    base = v39c[["date", "symbol"]].merge(
        gruc[["date", "symbol", "z"]].rename(columns={"z": "z_gru"}),
        on=["date", "symbol"], how="inner")
    base = base.merge(v39c[["date", "symbol", "z"]].rename(columns={"z": "z_v39"}),
                      on=["date", "symbol"], how="inner")
    base = base.drop_duplicates(subset=["date", "symbol"], keep="last")
    print(f"[35] v39∩GRU 对齐后 {len(base):,} 行 / {base['symbol'].nunique()} 只", flush=True)
    corr = base["z_v39"].corr(base["z_gru"])
    print(f"[35] v39 与 GRU 预测相关性 = {corr:.4f}（越低越有叠加价值）", flush=True)

    results: list[dict] = []

    # ---------- A：换手/配置网格（先在 v39 上扫） ----------
    print("\n" + "=" * 70, flush=True)
    print("A) 换手/成本配置扫描（v39，日期已对齐 GRU 区间）", flush=True)
    print("=" * 70, flush=True)
    grid = []
    for rf in (10, 15, 20, 30):
        for top_pct in (0.1, 0.2):
            for buffer in (0.0, 0.1):
                grid.append((rf, top_pct, buffer))
    for i, (rf, tp, bf) in enumerate(grid, 1):
        # 注意：v39c 本来就有一列 'pred'，不能 rename(z->pred)（会产生两个同名列，
        # 导致 groupby('date')['pred'] 返回 DataFrame 而报错）。必须新建干净列。
        p = v39c[["date", "symbol"]].copy()
        p["pred"] = v39c["z"].to_numpy()
        try:
            r = _run(p, panel, rf, tp, bf)
            r["model"] = "v39"
            results.append(r)
            print(f"  [{i:2d}/{len(grid)}] rf={rf:<3} top={tp:<5} buf={bf:<4} → "
                  f"净夏普 {r['net_sharpe']:+.3f} | 净累计 {r['net_total_return']:+.2%} | "
                  f"回撤 {r['net_mdd']:.2%} | 毛夏普 {r['gross_sharpe']:+.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:2d}/{len(grid)}] rf={rf} top={tp} buf={bf} 失败: {e!r}", flush=True)

    # ---------- B：GRU × v39 权重扫描（用 A 的最优配置） ----------
    v39_runs = [r for r in results if r["model"] == "v39"]
    best = max(v39_runs, key=lambda r: r["net_sharpe"]) if v39_runs else None
    if best is None:
        print("A 阶段无有效结果，跳过 B", flush=True)
        return 1
    print(f"\n[A 最优] rf={best['rf']} top_pct={best['top_pct']} buffer={best['buffer']} "
          f"→ 净夏普 {best['net_sharpe']:+.3f}（对照组 rf=15/0.1/0 需另看）", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"B) GRU × v39 权重扫描（固定 rf={best['rf']}, top={best['top_pct']}, "
          f"buf={best['buffer']}）", flush=True)
    print("=" * 70, flush=True)
    rf_b, tp_b, bf_b = best["rf"], best["top_pct"], best["buffer"]
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        m = base.copy()
        m["pred"] = (1.0 - w) * m["z_v39"] + w * m["z_gru"]
        tag = "v39" if w == 0 else ("gru" if w == 1.0 else f"ens(w_gru={w})")
        try:
            r = _run(m[["date", "symbol", "pred"]], panel, rf_b, tp_b, bf_b)
            r["model"] = tag
            r["w_gru"] = w
            results.append(r)
            print(f"  w_gru={w:<5} ({tag:<14}) → 净夏普 {r['net_sharpe']:+.3f} | "
                  f"净累计 {r['net_total_return']:+.2%} | 回撤 {r['net_mdd']:.2%} | "
                  f"毛夏普 {r['gross_sharpe']:+.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  w_gru={w} 失败: {e!r}", flush=True)

    # ---------- 汇总 ----------
    print("\n" + "=" * 70, flush=True)
    print("汇总（按净夏普降序，前 10）", flush=True)
    print("=" * 70, flush=True)
    ranked = sorted(results, key=lambda r: r["net_sharpe"], reverse=True)
    for r in ranked[:10]:
        print(f"  {r.get('model','?'):<16} rf={r['rf']:<3} top={r['top_pct']:<5} "
              f"buf={r['buffer']:<4} → 净夏普 {r['net_sharpe']:+.3f} | "
              f"累计 {r['net_total_return']:+.2%} | 回撤 {r['net_mdd']:.2%}", flush=True)

    OUT_JSON.write_text(json.dumps({
        "horizon": HORIZON,
        "mode": MODE,
        "date_range": [str(common_dates[0].date()), str(common_dates[-1].date())],
        "n_common_days": len(common_dates),
        "v39_gru_corr": float(corr),
        "best": ranked[0] if ranked else None,
        "best_v39_only": best,
        "all": results,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[35] 结果已保存: {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
