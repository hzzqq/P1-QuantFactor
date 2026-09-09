"""阶段 30 - ②b：baseline(LightGBM) 在 v40 数据集（39 + 真·log_mktcap_real + turnover_daily）上
复跑 walk-forward，验证真因子是否比 v39 代理因子进一步提 IC。

完全复用 28 的分块+内存优化逻辑，仅改默认数据集指向 v40。逐 test_years 落盘预测分片，
便于分块前台执行（避开 660s 墙钟）。

用法（与 28 一致，仅 --ds/--meta 默认 v40）：
    python scripts/33_retrain_real_baseline.py --test-years 2019,2020,2021,2022 \
        --out-part pred_baseline_h10_v40_part1.parquet --yearly-part yearly_v40_part1.json
    python scripts/33_retrain_real_baseline.py --test-years 2023,2024 \
        --out-part pred_baseline_h10_v40_part2a.parquet --yearly-part yearly_v40_part2a.json
    python scripts/33_retrain_real_baseline.py --test-years 2025,2026 \
        --out-part pred_baseline_h10_v40_part2b.parquet --yearly-part yearly_v40_part2b.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths
from shared.logging_utils import get_logger
from src.models import baseline_lgb

logger = get_logger("P1.v40_baseline")
PROCESSED = paths.DATA / "P1" / "processed"
DS = "dataset_h10_v40.parquet"
META = "dataset_h10_v40_meta.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--ds", type=str, default=DS)
    ap.add_argument("--meta", type=str, default=META)
    ap.add_argument("--test-years", type=str, required=True)
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--valid-years", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--label-clip", type=float, default=0.5)
    ap.add_argument("--eval-metric", type=str, default="ic")
    ap.add_argument("--out-part", type=str, required=True)
    ap.add_argument("--yearly-part", type=str, required=True)
    args = ap.parse_args()

    meta = json.load(open(PROCESSED / args.meta, encoding="utf-8"))
    factor_names = meta["factor_names"]
    test_years = [int(y) for y in args.test_years.split(",") if y.strip()]
    lo_year = min(test_years) - (args.train_years + 2)
    hi_year = max(test_years)
    lo = pd.Timestamp(year=lo_year, month=1, day=1)
    hi = pd.Timestamp(year=hi_year, month=12, day=31)
    data = pd.read_parquet(PROCESSED / args.ds, filters=[("date", ">=", lo), ("date", "<=", hi)]) \
        .replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    fcols = [c for c in data.columns if c not in ("date", "symbol", "y_cls", "y_fwd")]
    data[fcols] = data[fcols].astype(np.float32)
    del fcols
    if args.label_clip > 0:
        data = data[data["y_excess"].abs() <= args.label_clip]
    logger.info("v40 数据集 %s 行 | %s 因子", f"{len(data):,}", len(factor_names))

    t0 = time.time()
    preds, yearly = baseline_lgb.walk_forward(
        data, factor_names, y_col="y_excess",
        train_years=args.train_years, valid_years=args.valid_years,
        num_boost_round=args.rounds, eval_metric=args.eval_metric,
        test_years=test_years,
    )
    if preds.empty:
        logger.error("无预测产出，年份 %s 样本不足？", test_years)
        return 1
    preds.to_parquet(PROCESSED / args.out_part, index=False)
    with open(PROCESSED / args.yearly_part, "w", encoding="utf-8") as f:
        json.dump(yearly, f, ensure_ascii=False, indent=2, default=str)
    bi = [d["best_iteration"] for d in yearly]
    logger.info("v40 分片 %s 完成：%s 年 | best_iteration %s | %.1fs",
                test_years, len(yearly), bi, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
