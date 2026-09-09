"""阶段 23 - ②a 复跑（分块）：按 test_years 分片跑 baseline walk-forward。

用途：03_train_baseline.py 一次性跑全 16 年 walk-forward 会越过前台墙钟上限被 SIGTERM。
本驱动加载一次数据，仅对给定年份列表跑 walk_forward（valid_years=2 / eval_metric=ic 已是
baseline_lgb.walk_forward 默认），落盘该分片的预测与逐年 best_iteration，便于分块前台执行。

用法：
    python scripts/03b_baseline_chunk.py --test-years 2011,2012,2013,2014,2015 \
        --out-part pred_baseline_h10_part1.parquet --yearly-part yearly_part1.json
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
from shared.config import load_config
from shared.logging_utils import get_logger
from src.models import baseline_lgb
from src.eval import metrics

logger = get_logger("P1.baseline_chunk")
PROCESSED = paths.DATA / "P1" / "processed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--test-years", type=str, required=True,
                    help="逗号分隔的测试年份，如 2011,2012,2013")
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--label-clip", type=float, default=0.5)
    ap.add_argument("--out-part", type=str, required=True)
    ap.add_argument("--yearly-part", type=str, required=True)
    args = ap.parse_args()

    horizon = args.horizon
    meta = json.load(open(PROCESSED / f"dataset_h{horizon}_meta.json", encoding="utf-8"))
    factor_names = meta["factor_names"]
    data = pd.read_parquet(PROCESSED / f"dataset_h{horizon}.parquet") \
        .replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    if args.label_clip > 0:
        data = data[data["y_excess"].abs() <= args.label_clip]
    logger.info("数据集 %s 行 | %s 因子", f"{len(data):,}", len(factor_names))

    test_years = [int(y) for y in args.test_years.split(",") if y.strip()]
    t0 = time.time()
    preds, yearly = baseline_lgb.walk_forward(
        data, factor_names, y_col="y_excess",
        train_years=args.train_years, num_boost_round=args.rounds,
        test_years=test_years,
    )
    if preds.empty:
        logger.error("无预测产出，年份 %s 样本不足？", test_years)
        return 1
    preds.to_parquet(PROCESSED / args.out_part, index=False)
    with open(PROCESSED / args.yearly_part, "w", encoding="utf-8") as f:
        json.dump(yearly, f, ensure_ascii=False, indent=2, default=str)
    bi = [d["best_iteration"] for d in yearly]
    logger.info("分片 %s 完成：%s 年 | best_iteration 分布 %s | 耗时 %.1fs",
                test_years, len(yearly), bi, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
