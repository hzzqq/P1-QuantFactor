"""阶段 30 - ②b：合并 v40 baseline 重训分片，计算整体 IC，与 v39 生产基线对比，出 verdict。

对比对象：
  - h10: pred_baseline_h10_v40.parquet  vs 生产 pred_baseline_h10.parquet (v39, IC 0.0603)
  - h20: pred_baseline_h20_v40.parquet  vs pred_baseline_h20_v39.parquet     (IC 0.0640)

若 v40 在对应 horizon 上 IC 更高 → 打印 06_export_signal.py 落地命令（手动确认后执行，
避免 model 键撞车）。不自动覆盖生产信号。

用法：
    python scripts/34_consolidate_real.py --horizon 10
    python scripts/34_consolidate_real.py --horizon 20
"""
from __future__ import annotations

import argparse
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

from shared import paths
from src.eval import metrics

PROCESSED = paths.DATA / "P1" / "processed"

PARTS = {
    10: ["pred_baseline_h10_v40_part1.parquet", "pred_baseline_h10_v40_part2a.parquet",
         "pred_baseline_h10_v40_part2b.parquet"],
    20: ["pred_baseline_h20_v40_part1.parquet", "pred_baseline_h20_v40_part2a.parquet",
         "pred_baseline_h20_v40_part2b.parquet"],
}
PROD = {10: "pred_baseline_h10.parquet", 20: "pred_baseline_h20_v39.parquet"}
OUT = {10: "pred_baseline_h10_v40.parquet", 20: "pred_baseline_h20_v40.parquet"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[10, 20])
    args = ap.parse_args()
    h = args.horizon

    parts = [PROCESSED / p for p in PARTS[h]]
    present = [p for p in parts if p.exists()]
    if len(present) < len(parts):
        missing = [str(p) for p in parts if not p.exists()]
        print(f"[34] 缺失分片，无法合并：{missing}")
        return 1
    dfs = [pd.read_parquet(p) for p in present]
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(PROCESSED / OUT[h], index=False)
    print(f"[34] 合并 {len(present)} 片 → {OUT[h]} ({len(combined):,} 行)")

    s = metrics.summarize(combined, "pred", "y_excess")
    print(f"[34] v40 h{h} 整体 IC={s['ic_mean']:.4f} ICIR={s['icir']:.4f} "
          f"IC正占比={s['ic_positive_rate']:.4f}")

    prod = pd.read_parquet(PROCESSED / PROD[h])
    sp = metrics.summarize(prod, "pred", "y_excess")
    print(f"[34] 生产(v39) h{h} 整体 IC={sp['ic_mean']:.4f} ICIR={sp['icir']:.4f} "
          f"IC正占比={sp['ic_positive_rate']:.4f}")

    d_ic = s["ic_mean"] - sp["ic_mean"]
    print(f"[34] ΔIC(v40 - v39) = {d_ic:+.4f}")
    verdict = "REPLACE" if d_ic > 0 else "KEEP_V39"
    print(f"[34] verdict = {verdict}")

    # 逐年 IC 对比
    print("[34] 逐年 IC: v40 vs v39")
    years = sorted(combined["year"].unique())
    for y in years:
        a = combined[combined["year"] == y]
        b = prod[prod["year"] == y]
        if len(a) >= 100 and len(b) >= 100:
            ica = spearmanr(a["y_excess"], a["pred"]).correlation
            icb = spearmanr(b["y_excess"], b["pred"]).correlation
            print(f"  {y}: v40={ica:+.4f}  v39={icb:+.4f}  d={ica-icb:+.4f}")

    report = {
        "horizon": h,
        "v40_ic_mean": s["ic_mean"], "v40_icir": s["icir"],
        "v39_ic_mean": sp["ic_mean"], "v39_icir": sp["icir"],
        "delta_ic": d_ic, "verdict": verdict,
    }
    json.dump(report, open(PROCESSED / f"report_v40_h{h}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print(f"[34] 写出 report_v40_h{h}.json")
    if verdict == "REPLACE":
        print(f"[34] 落地命令（确认后手跑）：")
        print(f"  python scripts/06_export_signal.py --model baseline --horizon {h} "
              f"--pred {PROCESSED/OUT[h]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
