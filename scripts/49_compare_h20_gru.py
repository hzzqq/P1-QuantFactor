"""阶段 49：h20 信号强度 × GRU 多年的并排对比（结论产出）。

把四条信号路的逐年 IC 拉到同一张表，回答老板的两个问题：
  1) 信号强度：h20(20 日持有) 是否比 h10 更强更稳？
     → baseline_h10 (pred_baseline_h10_v39) vs baseline_h20 (pred_baseline_h20_v39) 逐年 IC。
  2) GRU 多年：GRU 在多年 walk-forward 下是否稳健？h20 上是否仍成立？
     → gru_h10 (report_gru_wf.json) vs gru_h20 (report_gru_wf_h20.json) 逐年 IC。

baseline 逐年 IC 由预测 parquet 现算（spearman(y_excess, pred)）；GRU 直接读已有 report
（walk_forward 已算好 IC）。

用法：
    python scripts/49_compare_h20_gru.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
PROCESSED = ROOT / "data" / "P1" / "processed"


def baseline_yearly_ic(pred_path: Path) -> dict:
    df = pd.read_parquet(pred_path, columns=["date", "symbol", "pred", "y_excess", "year"])
    df = df.dropna(subset=["y_excess", "pred"])
    out = {}
    for y, sub in df.groupby("year"):
        if len(sub) < 50:
            continue
        r = spearmanr(sub["y_excess"], sub["pred"]).correlation
        out[int(y)] = round(float(r), 4)
    return out


def main() -> int:
    # baseline 逐年 IC（现算）
    base_h10 = baseline_yearly_ic(PROCESSED / "pred_baseline_h10_v39.parquet")
    base_h20 = baseline_yearly_ic(PROCESSED / "pred_baseline_h20_v39.parquet")
    # GRU 逐年 IC（读 report）
    gru_h10 = {int(k): v["ic_mean"] for k, v in
               json.load(open(PROCESSED / "report_gru_wf.json", encoding="utf-8")).items()}
    gru_h20 = {int(k): v["ic_mean"] for k, v in
               json.load(open(PROCESSED / "report_gru_wf_h20.json", encoding="utf-8")).items()} \
        if (PROCESSED / "report_gru_wf_h20.json").exists() else {}

    years = sorted(set(base_h10) | set(base_h20) | set(gru_h10) | set(gru_h20))
    rows = []
    for y in years:
        rows.append({
            "year": y,
            "base_h10": base_h10.get(y),
            "base_h20": base_h20.get(y),
            "gru_h10": gru_h10.get(y),
            "gru_h20": gru_h20.get(y),
        })
    tbl = pd.DataFrame(rows)

    def mean(d):
        vals = [v for v in d.values() if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {
        "base_h10_mean": mean(base_h10),
        "base_h20_mean": mean(base_h20),
        "gru_h10_mean": mean(gru_h10),
        "gru_h20_mean": mean(gru_h20) if gru_h20 else None,
    }

    print("=== 逐年 IC 对比（spearman, 截面内） ===")
    print(tbl.to_string(index=False))
    print("\n=== 均值 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 信号强度判据（h20 vs h10，同源 baseline）：均值差 > 0 即 h20 更强
    if summary["base_h20_mean"] is not None and summary["base_h10_mean"] is not None:
        delta = summary["base_h20_mean"] - summary["base_h10_mean"]
        verdict_strength = ("h20 更强更稳" if delta > 0.002 else
                           "h20 持平/略弱" if delta < -0.002 else "h20 与 h10 持平")
        print(f"\n[信号强度] base_h20 − base_h10 = {delta:+.4f} → {verdict_strength}")

    # GRU 多年判据：早期(2019-2021) vs 后期(2022-2026) 均值
    def submean(d, lo, hi):
        vals = [d[y] for y in d if lo <= y <= hi and d.get(y) is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    if gru_h10:
        early = submean(gru_h10, 2019, 2021); late = submean(gru_h10, 2022, 2026)
        print(f"[GRU多年·h10] 早期(19-21) IC={early} 后期(22-26) IC={late} "
              f"→ {'逐年走强成立' if (late and early and late > early + 0.01) else '趋势不明显'}")

    # GRU h20 是否仍成立（逐年走强 + 同窗口 h10 vs h20）
    if gru_h20:
        early_h = submean(gru_h20, 2021, 2023); late_h = submean(gru_h20, 2024, 2026)
        vals_h = [v for v in gru_h20.values() if v is not None]
        pos_h = sum(1 for v in vals_h if v > 0)
        verdict_h = ("逐年走强成立" if (late_h and early_h and late_h > early_h + 0.01)
                     else "趋势不明显/受异常年拖累")
        print(f"[GRU多年·h20] 早期(21-23) IC={early_h} 后期(24-26) IC={late_h} "
              f"正IC年={pos_h}/{len(vals_h)} → {verdict_h}")
        common = sorted(set(gru_h10) & set(gru_h20))
        if common:
            m_h10c = submean(gru_h10, common[0], common[-1])
            m_h20c = submean(gru_h20, common[0], common[-1])
            cmp = ("h20 不弱于 h10" if m_h20c >= m_h10c - 0.005
                   else "h20 略弱于 h10（2023 异常年拖累）")
            print(f"[GRU信号·同窗口] h10均值={m_h10c} h20均值={m_h20c} → {cmp}")
            summary["gru_h20_early"] = early_h
            summary["gru_h20_late"] = late_h
            summary["gru_h20_positive_years"] = f"{pos_h}/{len(vals_h)}"
            summary["gru_same_window_h10"] = round(float(m_h10c), 4)
            summary["gru_same_window_h20"] = round(float(m_h20c), 4)
            summary["gru_h20_verdict"] = verdict_h

    out = {"table": tbl.to_dict(orient="records"), "summary": summary}
    json.dump(out, open(PROCESSED / "report_compare_h20_gru.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print("\n→ report_compare_h20_gru.json 已写出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
