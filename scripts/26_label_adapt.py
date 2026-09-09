"""阶段 26·③：标签/horizon 低波动 regime 适配评估。

侦察结论（2026-09-05 decay_diag）：y_excess(h=10) 的 std 从 2015 的 2.48 崩到
2022+ 的 0.07–0.09 —— 后 2021 是**低波动、低信噪比**结构性环境。这很可能压低了
所有信号的逐年 IC，而非因子真的失效。

本脚本评估两种适配思路（均不重建数据集，纯 IC 诊断）：
  A) 风险调整标签：y_adj = y_excess / rolling_std(y_excess, 60)（个股已实现波动归一）。
     若 baseline 信号在 2021+ 低波动年的 IC 经 y_adj 后更稳定/更高，说明风险调整标签有用。
  B) 更长 horizon（h=20）：dataset_h20 **不存在**（raw 仅 index/ + kline/），重建需重跑
     02_build_features.py。本脚本只读该脚本估算重建成本，并给出建议，不实际重建。

输出：data/P1/processed/report_label_adapt.json
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

print("[26] 读取 baseline 信号（已含 y_excess）...", flush=True)
base = pd.read_parquet(PROC / "pred_baseline_h10.parquet")
# pred_baseline_h10.parquet 已含 y_excess / year，无需再 merge dataset
m = base.dropna(subset=["y_excess", "pred"]).copy()
m["date"] = pd.to_datetime(m["date"])
m["year"] = m["date"].dt.year

# A) 风险调整标签（个股 60 日已实现波动归一）
m = m.sort_values(["symbol", "date"])
m["y_vol60"] = m.groupby("symbol")["y_excess"].transform(
    lambda x: x.rolling(60, min_periods=20).std())
m["y_adj"] = m["y_excess"] / m["y_vol60"].replace(0, np.nan)
m = m.dropna(subset=["y_adj"])

print("[26] 逐年对比 baseline IC：原始标签 vs 风险调整标签 ...", flush=True)
ic_raw, ic_adj = {}, {}
for y, g in m.groupby("year"):
    r = spearmanr(g["pred"], g["y_excess"]).correlation
    a = spearmanr(g["pred"], g["y_adj"]).correlation
    ic_raw[int(y)] = (None if (r != r) else round(float(r), 4))
    ic_adj[int(y)] = (None if (a != a) else round(float(a), 4))
print("[26] IC_raw :", ic_raw, flush=True)
print("[26] IC_adj :", ic_adj, flush=True)

# 低波动年（2021+）平均 IC 对比
low = [y for y in ic_raw if y >= 2021]
hi = [y for y in ic_raw if y < 2021]
def mean(xs): return float(np.nanmean([x for x in xs if x is not None]))
raw_low, adj_low = mean([ic_raw[y] for y in low]), mean([ic_adj[y] for y in low])
raw_hi, adj_hi = mean([ic_raw[y] for y in hi]), mean([ic_adj[y] for y in hi])
print(f"[26] 低波动年(2021+) IC_raw={raw_low:.4f} IC_adj={adj_low:.4f}", flush=True)
print(f"[26] 高波动年(<2021) IC_raw={raw_hi:.4f} IC_adj={adj_hi:.4f}", flush=True)

# B) h=20 重建成本估算
build = PROJ / "scripts" / "02_build_features.py"
h20_exists = (PROC / "dataset_h20_ev.parquet").exists()
rebuild_lines = None
if build.exists():
    import re
    txt = build.read_text(encoding="utf-8", errors="ignore")
    rebuild_lines = len(txt.splitlines())
rebuild_cost = ("需重跑 02_build_features.py（%d 行）并新增 horizon=20 分支；"
                "预计与 h=10 同量级（608MB×2 特征+标签重建，约 10–20 分钟）"
                % (rebuild_lines or 0)) if not h20_exists else "dataset_h20 已存在"
print("[26] h20 重建:", rebuild_cost, flush=True)

verdict = ("RISK_ADJ_HELPS" if (adj_low - raw_low) > 0.005
           else "RISK_ADJ_MARGINAL")
report = {
    "regime": "y_excess std 从 2015 的 2.48 崩到 2022+ 的 0.07–0.09（低波动/低信噪比）",
    "A_risk_adjusted_label": {
        "ic_raw_by_year": ic_raw, "ic_adj_by_year": ic_adj,
        "lowvol_years_ic_raw": round(raw_low, 4), "lowvol_years_ic_adj": round(adj_low, 4),
        "highvol_years_ic_raw": round(raw_hi, 4), "highvol_years_ic_adj": round(adj_hi, 4),
    },
    "B_longer_horizon_h20": {
        "dataset_h20_present": h20_exists,
        "rebuild_estimate": rebuild_cost,
    },
    "verdict": verdict,
    "recommendation": (
        "低波动年风险调整标签 IC 提升显著 → 建议重训 baseline/GRU 用 y_adj 标签；"
        if verdict == "RISK_ADJ_HELPS" else
        "风险调整标签增益有限 → 优先做 B(h=20 重建) 或换更长 horizon 适配低波动 regime；"
    ) + ("h=20 重建成本约 10–20 分钟，列为后续任务。" if not h20_exists else ""),
}
json.dump(report, open(PROC / "report_label_adapt.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=str)
print("[26] verdict:", verdict, flush=True)
print("[26] 写出 report_label_adapt.json", flush=True)
