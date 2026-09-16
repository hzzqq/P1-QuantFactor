"""阶段 52：h10-ensemble × h20 组合（主副信号选择，数据化决策）。

之前两路已各自最优：
  - h10 ensemble（v39+GRU×0.25）：rf=15/top=0.1/buffer=0.1，净夏普 1.170 / MDD -10.04%
  - h20 baseline：rf=10/top=0.1/buffer=0，净夏普 1.445 / MDD -21.00%
  - h20 + vol_target=0.08（阶段49 B）：净夏普 1.500 / MDD -15.95%（已去杠杆）

本脚本把「h20(volt0.08)」与「h10-ensemble」作为两个 sleeve，按时对齐做
**多策略组合**（月频收益加权），对比：
  100% h20(volt0.08) / 100% h10 / 50-50 / 波动率平价
看组合是否因低相关进一步降回撤、提夏普，从而给出「主/副」结论。

注：组合用月频收益近似（把每策略权益曲线 resample 到月末再合并），
相对比较口径一致，足以支撑主副决策。
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
h10 = pd.read_parquet(PROC / "pred_ens_v39gru_w025_h10.parquet")
h20 = pd.read_parquet(PROC / "pred_baseline_h20_v39.parquet")


def monthly_returns(r: "object") -> pd.Series:
    """把回测权益曲线 resample 到月末，返回月频收益序列（datetime 索引）。"""
    eq = r.equity.copy()
    eq.index = pd.to_datetime(eq.index)
    me = eq.resample("ME").last().dropna()
    return me.pct_change().dropna()


def combo_stats(w1: float, r1: "object", r2: "object") -> dict:
    m1 = monthly_returns(r1)
    m2 = monthly_returns(r2)
    idx = m1.index.intersection(m2.index)
    m1, m2 = m1.reindex(idx), m2.reindex(idx)
    port = w1 * m1 + (1 - w1) * m2
    eq = (1 + port).cumprod()
    n = len(port)
    years = n / 12.0
    total = float(eq.iloc[-1] - 1)
    ann = float((1 + total) ** (1 / max(years, 1e-9)) - 1)
    sharpe = float(port.mean() / port.std() * np.sqrt(12)) if port.std() > 0 else np.nan
    mdd = float((eq / eq.cummax() - 1).min())
    # 两 sleeve 月收益相关性（低相关 → 组合分散化收益大）
    corr = float(m1.corr(m2)) if n > 2 else np.nan
    return {"w_h20": round(w1, 3), "w_h10": round(1 - w1, 3),
            "sharpe": sharpe, "total_return": total, "annual_return": ann,
            "max_drawdown": mdd, "n_months": n, "sleeve_corr": corr}


def run_sleeves():
    t0 = time.time()
    # h20 + vol_target=0.08（阶段49 B 最优）
    r_h20 = run_backtest(h20, panel, horizon=20, top_pct=0.1, rebalance_freq=10,
                         cost=DEFAULT_COST, vol_target=0.08, max_leverage=1.0)
    # h10 ensemble 生产配置（buffer=0.1 降换手）
    r_h10 = run_backtest_continuous(h10, panel, horizon=10, top_pct=0.1,
                                    rebalance_freq=15, cost=DEFAULT_COST, buffer=0.1)
    print(f" sleeve h20(volt0.08): netSharpe={r_h20.ls_stats['sharpe']:.3f} "
          f"mdd={r_h20.ls_stats['max_drawdown']:.2%}")
    print(f" sleeve h10(ens)     : netSharpe={r_h10.ls_stats['sharpe']:.3f} "
          f"mdd={r_h10.ls_stats['max_drawdown']:.2%}")
    return r_h20, r_h10


def main() -> int:
    logger_info = None
    r_h20, r_h10 = run_sleeves()

    rows = []
    rows.append(combo_stats(1.0, r_h20, r_h10))   # 100% h20
    rows.append(combo_stats(0.0, r_h20, r_h10))   # 100% h10
    rows.append(combo_stats(0.5, r_h20, r_h10))   # 50-50
    # 波动率平价
    v1 = monthly_returns(r_h20).std()
    v2 = monthly_returns(r_h10).std()
    wp = (1 / v1) / (1 / v1 + 1 / v2)
    rows.append(combo_stats(wp, r_h20, r_h10))

    out = PROC / "report_combine.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n=== 组合对比（月频收益加权）===")
    print(f"{'w_h20':>6} {'w_h10':>6} {'Sharpe':>7} {'cumRet':>9} {'MDD':>8} {'corr':>6}")
    for r in rows:
        print(f"{r['w_h20']:6.2f} {r['w_h10']:6.2f} {r['sharpe']:7.3f} "
              f"{r['total_return']:9.2%} {r['max_drawdown']:8.2%} {r['sleeve_corr']:6.2f}")
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
