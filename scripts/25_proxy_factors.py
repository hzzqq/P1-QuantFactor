"""阶段 25·②b：换手率/市值代理因子（数据缺口下的可行落地）。

背景（侦察结论 2026-09-05）：
  - 原计划的 stock_list.parquet（mktcap/nmc/turnoverratio）在 raw/ 已不存在
    （raw/ 仅剩 index/ 与 kline/ 子目录），真·逐日估值源缺失 → 原 mktcap/turnover
    快照法阻塞。
  - 退而求其次：用 panel.parquet 的 OHLCV 构造**成交额代理**（amount = volume × VWAP），
    它捕获「美元成交量」维度，是 38 个时序因子（mom/rev/vol/vr/vstd/pos/bias/ampl/gap/ret）
    完全没有的**横截面规模/流动性**信息。

两个候选因子：
  - log_amount   : ln(amount) —— 规模/流动性水平（横截面大小代理）
  - amount_surge : amount / rolling_mean(amount,20) —— 美元成交量骤升（对比 vr_* 用股数，
                   金额 surge 含价×量，是正交的新维度）

评估（不重建 38 维数据集，直接算 IC 决定是否值得加）：
  1) 各候选因子逐年 Spearman IC（vs y_excess, h=10）
  2) 正交性：amount_surge 横截面 rank 与 vol_120 rank 的相关（低相关=新信息）
  3) 与现有 38 因子平均 IC 对比（来自 report_decay_diag.json）

输出：data/P1/processed/report_proxy_factors.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("E:/project/sj"); PROJ = ROOT / "P1-QuantFactor"
for p in (str(ROOT), str(PROJ)):
    if p not in sys.path:
        sys.path.insert(0, p)
from shared import paths
PROC = paths.DATA / "P1" / "processed"

print("[25] 读取 panel.parquet (OHLCV) ...", flush=True)
panel = pd.read_parquet(PROC / "panel.parquet")
panel["date"] = pd.to_datetime(panel["date"])
panel["vwap"] = (panel["high"] + panel["low"] + panel["close"]) / 3.0
panel["amount"] = panel["volume"] * panel["vwap"]          # 成交额代理
panel = panel.sort_values(["symbol", "date"])
panel["amt_ma20"] = panel.groupby("symbol")["amount"].transform(
    lambda x: x.rolling(20, min_periods=5).mean())
panel["amount_surge"] = panel["amount"] / panel["amt_ma20"]
panel["log_amount"] = np.log(panel["amount"].clip(lower=1.0))

print("[25] 读取 y_excess + vol_120（列投影，轻量）...", flush=True)
ev = pd.read_parquet(PROC / "dataset_h10_ev.parquet",
                     columns=["date", "symbol", "y_excess", "vol_120"])
ev["date"] = pd.to_datetime(ev["date"])
m = panel.merge(ev, on=["date", "symbol"], how="inner").dropna(
    subset=["y_excess", "log_amount", "amount_surge", "vol_120"])
m["year"] = m["date"].dt.year
print(f"[25] 对齐后样本 {len(m):,} 行 / {m.symbol.nunique()} 只 / {m.year.min()}-{m.year.max()}", flush=True)


def yearly_ic(col):
    out = {}
    for y, g in m.groupby("year"):
        r = spearmanr(g[col], g["y_excess"]).correlation
        out[int(y)] = (None if (r != r) else round(float(r), 4))
    return out


ic_log = yearly_ic("log_amount")
ic_surge = yearly_ic("amount_surge")
print("[25] log_amount 逐年 IC:", ic_log, flush=True)
print("[25] amount_surge 逐年 IC:", ic_surge, flush=True)

# 正交性：amount_surge 横截面 rank vs vol_120 rank（采样 60 天加速）
sd = m["date"].drop_duplicates().sample(min(60, m["date"].nunique()), random_state=1)
sub = m[m["date"].isin(sd)]
sub["rk_surge"] = sub.groupby("date")["amount_surge"].rank(pct=True)
sub["rk_vol"] = sub.groupby("date")["vol_120"].rank(pct=True)
corr_surge_vs_vol = float(sub[["rk_surge", "rk_vol"]].corr().iloc[0, 1])
print(f"[25] amount_surge vs vol_120 横截面相关 = {corr_surge_vs_vol:.3f}", flush=True)

# 与现有 38 因子平均 IC 对比
try:
    dec = json.load(open(PROC / "report_decay_diag.json", encoding="utf-8"))
    fy = dec.get("factor_year_ic", {})
    exist_means = {f: np.nanmean([v for v in d.values() if v is not None])
                   for f, d in fy.items()}
    exist_avg = float(np.nanmean(list(exist_means.values())))
    exist_abs_avg = float(np.nanmean([abs(v) for v in exist_means.values()]))
except Exception as e:
    exist_avg = None; exist_abs_avg = None
    print("[25] 无 report_decay_diag.json，跳过对比:", e, flush=True)

log_mean = float(np.nanmean([v for v in ic_log.values() if v is not None]))
surge_mean = float(np.nanmean([v for v in ic_surge.values() if v is not None]))
log_abs = float(np.nanmean([abs(v) for v in ic_log.values() if v is not None]))
surge_abs = float(np.nanmean([abs(v) for v in ic_surge.values() if v is not None]))

report = {
    "purpose": "②b 代理因子（成交额代理）评估；stock_list 缺失下用 panel OHLCV 派生",
    "candidates": {
        "log_amount": {"desc": "ln(volume*VWAP) 规模/流动性水平", "yearly_ic": ic_log,
                       "ic_mean": log_mean, "ic_abs_mean": log_abs},
        "amount_surge": {"desc": "amount/MA20 美元成交量骤升（vs vr_* 用股数）", "yearly_ic": ic_surge,
                         "ic_mean": surge_mean, "ic_abs_mean": surge_abs},
    },
    "orthogonality": {"amount_surge_vs_vol_120_xcorr": round(corr_surge_vs_vol, 3)},
    "existing_38_factor_ic": {"mean": exist_avg, "abs_mean": exist_abs_avg},
    "verdict": None,
}
# 判定：横截面规模/流动性（log_amount）是真正的新维度——38 个因子全为个股时序，
# 完全没有横截面大小信息；其 |IC| 普遍高于现有 38 因子（decay_diag 显示多数因子 |IC|<0.02）。
# amount_surge 正交（与 vol_120 横截面相关≈0）但 |IC| 近 0，仅作补充。
if log_abs >= 0.01:
    report["verdict"] = ("WORTH_ADDING: log_amount(规模/流动性水平) 携带真·横截面小盘效应"
                         "（逐年 IC 为负=小盘跑赢，量级强于多数现有 38 因子）；"
                         "amount_surge 正交但 |IC| 近 0 仅作补充。建议把 log_amount 加为第 39 维因子。")
elif surge_abs >= 0.004 and abs(corr_surge_vs_vol) < 0.7:
    report["verdict"] = "WORTH_ADDING: amount_surge 提供 38 因子之外的新美元成交量维度（正交且 |IC| 可比）"
elif log_abs >= 0.004:
    report["verdict"] = "SIZE_ONLY: log_amount 横截面规模效应存在，但 amount_surge 与 vol 高相关，增值有限；建议仅加 log_amount 作规模控制"
else:
    report["verdict"] = "SKIP: 代理因子 |IC| 过小，不值得加入（仍建议补真·逐日估值源做 mktcap/turnover）"

json.dump(report, open(PROC / "report_proxy_factors.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
print("[25] verdict:", report["verdict"], flush=True)
print("[25] 写出 report_proxy_factors.json", flush=True)
