"""阶段 38：把阶段 35/36 的「换手/成本优化」原样套到 h20。

前提核查：GRU 目前只有 h10 的权重与预测（gru_best_h10_* / pred_gru_h10 / pred_gru_wf_*），
若不存在 h20 的 GRU 预测，则 B（ensemble）在 h20 上不可做，只做 A（换手/成本优化）。

A 段：rf × top_pct × buffer 网格（h20 的 horizon=20，rf 需含 20 即与 horizon 对齐一档）
复核：最优配置 vs 当前配置(rf=15/0.1/buf=0) 在 **2024+ 子区间** 的表现
      —— h10 上正是这一项证明 buffer 是真收益（现配置 2024+ 为负）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths  # noqa: E402
from src.backtest import run_backtest, run_backtest_continuous, DEFAULT_COST  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
OUT = PROCESSED / "report_sweep_h20_turnover.json"
HORIZON, MODE = 20, "long_short"
ZERO = {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}


def _run(preds, panel, rf, top_pct, buffer):
    kw = dict(horizon=HORIZON, top_pct=top_pct, rebalance_freq=rf, mode=MODE)
    if buffer > 0:
        net = run_backtest_continuous(preds, panel, **kw, cost=DEFAULT_COST, buffer=buffer)
        gross = run_backtest_continuous(preds, panel, **kw, cost=ZERO, buffer=buffer)
    else:
        net = run_backtest(preds, panel, **kw, cost=DEFAULT_COST)
        gross = run_backtest(preds, panel, **kw, cost=ZERO)
    n, g = net.ls_stats or {}, gross.ls_stats or {}
    return {
        "rf": rf, "top_pct": top_pct, "buffer": buffer,
        "net_sharpe": round(float(n.get("sharpe", float("nan"))), 4),
        "net_total_return": round(float(n.get("total_return", float("nan"))), 4),
        "net_annual": round(float(n.get("annual_return", float("nan"))), 4),
        "net_mdd": round(float(n.get("max_drawdown", float("nan"))), 4),
        "win_rate": round(float(n.get("win_rate", float("nan"))), 4),
        "gross_sharpe": round(float(g.get("sharpe", float("nan"))), 4),
    }


def main() -> int:
    # --- 前提核查：h20 有没有 GRU 预测 ---
    gru_h20 = sorted(PROCESSED.glob("pred_gru*h20*.parquet"))
    mdl = paths.MODELS / "P1"
    gru_h20_pt = sorted(mdl.glob("*h20*.pt")) if mdl.exists() else []
    print("[38] 前提核查：h20 的 GRU 资产", flush=True)
    print(f"  pred_gru*h20*.parquet : {[p.name for p in gru_h20] or '无'}", flush=True)
    print(f"  models/P1/*h20*.pt    : {[p.name for p in gru_h20_pt] or '无'}", flush=True)
    can_ensemble = bool(gru_h20)
    print(f"  → h20 能否做 ensemble: {'能' if can_ensemble else '不能（GRU 只有 h10）'}",
          flush=True)

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    p = pd.read_parquet(PROCESSED / "pred_baseline_h20_v39.parquet")
    p["date"] = pd.to_datetime(p["date"])
    n0 = len(p)
    p = p.drop_duplicates(subset=["date", "symbol"], keep="last")
    print(f"\n[38] pred_baseline_h20_v39: {n0:,} → 去重后 {len(p):,} 行 | "
          f"{p['date'].min().date()} ~ {p['date'].max().date()}", flush=True)
    preds = p[["date", "symbol", "pred"]].copy()

    results = []
    grid = [(rf, tp, bf) for rf in (10, 15, 20, 30)
            for tp in (0.1, 0.2) for bf in (0.0, 0.1)]
    print("\n" + "=" * 70, flush=True)
    print(f"A) h20 换手/成本配置扫描（{len(grid)} 组，全量区间）", flush=True)
    print("=" * 70, flush=True)
    for i, (rf, tp, bf) in enumerate(grid, 1):
        try:
            r = _run(preds, panel, rf, tp, bf)
            results.append(r)
            print(f"  [{i:2d}/{len(grid)}] rf={rf:<3} top={tp:<5} buf={bf:<4} → "
                  f"净夏普 {r['net_sharpe']:+.3f} | 累计 {r['net_total_return']:+.2%} | "
                  f"回撤 {r['net_mdd']:.2%} | 毛 {r['gross_sharpe']:+.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i:2d}/{len(grid)}] rf={rf} top={tp} buf={bf} 失败: {e!r}", flush=True)

    if not results:
        print("无有效结果", flush=True)
        return 1

    best = max(results, key=lambda r: r["net_sharpe"])
    print(f"\n[A 最优] rf={best['rf']} top={best['top_pct']} buf={best['buffer']} → "
          f"净夏普 {best['net_sharpe']:+.3f} | 累计 {best['net_total_return']:+.2%}", flush=True)

    # --- 2024+ 子区间复核（h10 上这是决定性验证） ---
    print("\n" + "=" * 70, flush=True)
    print("复核：2024+ 子区间（最优配置 vs 当前配置 rf=15/0.1/buf=0）", flush=True)
    print("=" * 70, flush=True)
    cut = pd.Timestamp("2024-01-01")
    sub = preds[preds["date"] >= cut]
    sub_res = []
    for rf, tp, bf, tag in [(best["rf"], best["top_pct"], best["buffer"], "最优"),
                            (15, 0.1, 0.0, "当前"),
                            (10, 0.1, 0.0, "rf10默认"),
                            (20, 0.1, 0.0, "rf20=horizon")]:
        try:
            r = _run(sub, panel, rf, tp, bf)
            r["tag"] = tag
            sub_res.append(r)
            print(f"  {tag:<12} rf={rf:<3} top={tp} buf={bf} → 净夏普 {r['net_sharpe']:+.3f} | "
                  f"累计 {r['net_total_return']:+.2%} | 回撤 {r['net_mdd']:.2%}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {tag} 失败: {e!r}", flush=True)

    OUT.write_text(json.dumps({
        "horizon": HORIZON,
        "can_ensemble_h20": can_ensemble,
        "note": "h20 换手/成本优化扫描；GRU 仅 h10，故 h20 无 ensemble。",
        "best": best,
        "subperiod_2024plus": sub_res,
        "all": results,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[38] 已保存 {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
