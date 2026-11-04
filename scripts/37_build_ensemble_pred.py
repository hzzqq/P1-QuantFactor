"""阶段 37：生成 v39 × GRU 的 ensemble 生产预测文件。

历史依据（35/36 实测，2022-2026 同窗口 + 2024+ 子区间双重验证，w_gru=0.25）：
  - 配置 rf=15 / top_pct=0.1 / **buffer=0.1**（连续持仓+缓冲区降换手）
  - 权重 w_gru=0.25（全窗口净夏普 0.718 vs v39 单独 0.606；2024+ 为 0.923 vs 0.840）
  - 关键：当前生产配置 buffer=0 在 2024+ 子区间净夏普 -0.188（亏损），
    buffer=0.1 同期 +0.840 → 换手优化是本次收益的主要来源。

2026-11-01 生产变更（老板拍板，依据 gru反事实与漂移检查_2026-11-01.md）：
  - 2026 窗口（49 信号日 07-28~10-12）反事实：w_gru 0.25→0 单调改善
    （累计 +9.94%→+12.02%、回撤 -17.02%→-15.23%、胜率 46.0%→48.0%），w=0（纯 v39）最优；
  - 训练侧漂移检查：38 特征 PSI 仅 2 轻度/0 剧变（特征边际分布稳定），但 IC 衰减/反转
    （rev_5 衰减 78%、bias_5 反转）→ 训练侧重训/增补特征证据弱；
  - 生产 fusion 已退化为纯基线（GRU 被 fusion 扫描排除），ens 侧同步下调。
  - **默认 --w-gru 0.00**（ens 退化为纯 v39 序）；历史 0.25 可传参复现。
  - 兼容性说明：输出文件名保留 pred_ens_v39gru_w025_h10.parquet（ks 侧
    gen_morning_event_sentiment.py 等硬编码引用），文件名中的 w025 不再反映实际权重，
    实际权重以 --w-gru 为准（2026-11-01 起为 0.00）。
  - GRU 覆盖 2022 起；2022 之前 GRU 缺失，按 w_gru=0 处理（退化为 v39 单独，
    因 pred=(1-w)*z_v39+w*0 与 z_v39 同序，排名不变）。

产出：pred_ens_v39gru_w025_h10.parquet（供 06_export_signal.py / zz_export_ens.py 使用）
"""
from __future__ import annotations

import argparse
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
from src.backtest import run_backtest, run_backtest_continuous, DEFAULT_COST  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
# 文件名保留 w025 兼容下游（ks gen_morning_event_sentiment 等硬编码引用）；实际权重见 --w-gru
OUT_PRED = PROCESSED / "pred_ens_v39gru_w025_h10.parquet"
W_GRU_DEFAULT = 0.00  # 2026-11-01 拍板：ens 内 gru 权重下调至 0（纯 v39 序）
RF, TOP_PCT, BUFFER = 15, 0.1, 0.1
ZERO = {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}


def _z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
            ).fillna(0.0).clip(-3, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-gru", type=float, default=W_GRU_DEFAULT,
                    help="ens 内 gru 权重；2026-11-01 拍板默认 0.00（历史 0.25 可传参复现）")
    ap.add_argument("--out", default=None, help="输出 parquet 路径（默认保留 w025 兼容名）")
    args = ap.parse_args()
    W_GRU = args.w_gru
    out_path = Path(args.out) if args.out else OUT_PRED

    v39 = pd.read_parquet(PROCESSED / "pred_baseline_h10_v39.parquet")
    gru = pd.read_parquet(PROCESSED / "pred_gru_h10.parquet")
    for d in (v39, gru):
        d["date"] = pd.to_datetime(d["date"])
        d.drop_duplicates(subset=["date", "symbol"], keep="last", inplace=True)

    v39["z_v39"] = _z(v39, "pred")
    gru["z_gru"] = _z(gru, "pred")

    m = v39[["date", "symbol", "z_v39", "pred"]].merge(
        gru[["date", "symbol", "z_gru"]], on=["date", "symbol"], how="left")
    has_gru = m["z_gru"].notna()
    # GRU 缺失段退化为 v39 单独（填 0 = 中性 z，与 z_v39 同序，排名不变）
    m["pred"] = (1 - W_GRU) * m["z_v39"] + W_GRU * m["z_gru"].fillna(0.0)
    print(f"[37] w_gru={W_GRU} 合并后 {len(m):,} 行；其中含 GRU 的行 {int(has_gru.sum()):,} "
          f"({has_gru.mean():.1%})，GRU 覆盖 "
          f"{m.loc[has_gru,'date'].min().date()} ~ {m.loc[has_gru,'date'].max().date()}",
          flush=True)
    print(f"[37] 全量区间 {m['date'].min().date()} ~ {m['date'].max().date()}", flush=True)

    out = m[["date", "symbol", "pred"]].copy()
    out.to_parquet(out_path, index=False)
    print(f"[37] 已写出 ensemble 预测: {out_path}  ({len(out):,} 行, w_gru={W_GRU})", flush=True)

    # 复核：用目标配置回测，确认与 35/36 结论一致
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    kw = dict(horizon=10, top_pct=TOP_PCT, rebalance_freq=RF, mode="long_short")
    net = run_backtest_continuous(out, panel, **kw, cost=DEFAULT_COST, buffer=BUFFER)
    gross = run_backtest_continuous(out, panel, **kw, cost=ZERO, buffer=BUFFER)
    n, g = net.ls_stats or {}, gross.ls_stats or {}
    print("\n[37] 复核回测（rf=15, top=0.1, buffer=0.1，全量区间）", flush=True)
    print(f"  净: 累计 {n.get('total_return', float('nan')):+.2%} | "
          f"年化 {n.get('annual_return', float('nan')):+.2%} | "
          f"夏普 {n.get('sharpe', float('nan')):+.3f} | "
          f"回撤 {n.get('max_drawdown', float('nan')):.2%}", flush=True)
    print(f"  毛: 夏普 {g.get('sharpe', float('nan')):+.3f}", flush=True)

    # 对照组：v39 单独（raw pred，buffer=0），同区间
    v39p = v39[["date", "symbol", "pred"]].copy()
    base_net = run_backtest(v39p, panel, **kw, cost=DEFAULT_COST)
    bn = base_net.ls_stats or {}
    print("\n[37] 对照：v39 单独（raw pred, rf=15, top=0.1, buffer=0）", flush=True)
    print(f"  净: 累计 {bn.get('total_return', float('nan')):+.2%} | "
          f"夏普 {bn.get('sharpe', float('nan')):+.3f} | "
          f"回撤 {bn.get('max_drawdown', float('nan')):.2%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
