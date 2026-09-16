"""阶段 49：A/B 深化的成本/风控改造验证与稳健性扫描。

对比（均扣成本）：
  h10 ensemble（生产 rf=15/top=0.1/buffer=0.1）：
    1) fixed  + equal        (生产基线，应≈1.170 净夏普)
    2) liquidity + equal     (A 任务：流动性分档滑点，更真)
    3) fixed + inv_vol       (B 任务：应==equal，因已退化为 equal)
  h20 baseline（生产 rf=10/top=0.1/buffer=0）：
    4) fixed + equal         (生产基线，应≈1.445 净夏普 / -21% 回撤)
    5) fixed + equal + volt扫描(0.08/0.10/0.12/0.15)  [max_leverage=1.0 只去杠杆]

校验：默认档(equal/fixed/无volt) 数值必须与改造前一致（不改默认行为）。
记录：liquidity 是否显著改变结论；vol_target 在 target<自然波动时是否真压回撤。
"""
from __future__ import annotations
import sys, time, json, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"E:/project/sj"); PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.backtest import run_backtest, run_backtest_continuous, DEFAULT_COST

PROC = ROOT / "data" / "P1" / "processed"
panel = pd.read_parquet(PROC / "panel.parquet")
print(f"panel {len(panel):,} rows | {panel['symbol'].nunique()} symbols")

H10 = PROC / "pred_ens_v39gru_w025_h10.parquet"
H20 = PROC / "pred_baseline_h20_v39.parquet"
h10_pred = pd.read_parquet(H10)
h20_pred = pd.read_parquet(H20)


def run(tag, pred, *, horizon, rf, top, buffer, cost_model, sizer, vol_target=None, max_lev=1.0):
    t0 = time.time()
    kw = dict(horizon=horizon, top_pct=top, rebalance_freq=rf, cost=DEFAULT_COST,
              cost_model=cost_model, sizer=sizer, vol_target=vol_target, max_leverage=max_lev)
    if buffer > 0:
        r = run_backtest_continuous(pred, panel, **kw, buffer=buffer)
    else:
        r = run_backtest(pred, panel, **kw)
    s = r.ls_stats
    print(f"[{tag}] cost={cost_model} sizer={sizer} volt={vol_target} | "
          f"netSharpe={s.get('sharpe',float('nan')):.3f} "
          f"cumRet={s.get('total_return',float('nan')):.2%} "
          f"mdd={s.get('max_drawdown',float('nan')):.2%} "
          f"ann={s.get('annual_return',float('nan')):.2%} "
          f"turn={s.get('avg_turnover', float('nan')):.1%} ({time.time()-t0:.1f}s)")
    return {"tag": tag, "cost_model": cost_model, "sizer": sizer,
            "vol_target": vol_target, "sharpe": s.get("sharpe"),
            "total_return": s.get("total_return"), "max_drawdown": s.get("max_drawdown"),
            "annual_return": s.get("annual_return"), "n_buckets": s.get("n_buckets"),
            "avg_turnover": s.get("avg_turnover")}


rows = []
# ---- h10 ensemble ----
rows.append(run("h10 baseline", h10_pred, horizon=10, rf=15, top=0.1, buffer=0.1,
                cost_model="fixed", sizer="equal"))
rows.append(run("h10 A-liquidity", h10_pred, horizon=10, rf=15, top=0.1, buffer=0.1,
                cost_model="liquidity", sizer="equal"))
rows.append(run("h10 B-invvol", h10_pred, horizon=10, rf=15, top=0.1, buffer=0.1,
                cost_model="fixed", sizer="inv_vol"))   # 应 == baseline（退化）

# ---- h20 baseline ----
rows.append(run("h20 baseline", h20_pred, horizon=20, rf=10, top=0.1, buffer=0.0,
                cost_model="fixed", sizer="equal"))
for vt in (0.08, 0.10, 0.12, 0.15):
    rows.append(run(f"h20 B-volt{vt}", h20_pred, horizon=20, rf=10, top=0.1, buffer=0.0,
                    cost_model="fixed", sizer="equal", vol_target=vt, max_lev=1.0))

# h20 自然年化波动（无缩放基准）用于判断 target 是否低于自然波动
base = [r for r in rows if r["tag"] == "h20 baseline"][0]
print(f"\n[h20 自然] sharpe={base['sharpe']:.3f} cumRet={base['total_return']:.2%} "
      f"mdd={base['max_drawdown']:.2%}")

out = PROC / "report_deepen_ab.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(f"\n对比表已存 {out}")
print("\n=== 汇总（全量 2019-2026）===")
print(f"{'tag':16} {'cost':9} {'sizer':8} {'volt':5} {'Sharpe':>7} {'cumRet':>9} {'MDD':>8}")
for r in rows:
    vt = "-" if r["vol_target"] is None else f"{r['vol_target']:.2f}"
    tn = "-" if r["avg_turnover"] is None else f"{r['avg_turnover']:.1%}"
    print(f"{r['tag']:16} {r['cost_model']:9} {r['sizer']:8} {vt:5} "
          f"{r['sharpe']:7.3f} {r['total_return']:9.2%} {r['max_drawdown']:8.2%} {tn:>6}")

# ---- 2024+ 子区间稳健性（h20 baseline vs volt0.08）----
print("\n=== 2024+ 子区间稳健性（h20）===")
sub = h20_pred.copy()
sub["date"] = pd.to_datetime(sub["date"])
sub = sub[sub["date"] >= pd.Timestamp("2024-01-01")]
print(f"  2024+ 样本 {len(sub):,} 行")
for cfg in (("baseline", None), ("volt0.08", 0.08)):
    tag, vt = cfg
    r = run(f"h20-2024 {tag}", sub, horizon=20, rf=10, top=0.1, buffer=0.0,
            cost_model="fixed", sizer="equal", vol_target=vt, max_lev=1.0)
    rows.append(r)

