"""阶段 50 - B：用 HistGradientBoosting(同族 GBDT) 重训 baseline，对 v39 与 v41 做**同模型**公平对照。

背景
----
环境无 lightgbm 轮子，生产 baseline(`baseline_lgb`) 无法复跑。Histogram GBDT 与 LightGBM
同族，`HistGradientBoostingRegressor` 可用，作为公平对照：对 **v39(39维)** 和 **v41(41维,含
ep/btop 估值因子)** 用同一模型分别 walk-forward，隔离出估值因子的真实增量。

本脚本是 `28_retrain_v39_baseline.py` 的镜像，仅把 `baseline_lgb` 换成 `baseline_hgb`，
并保留 `--ds/--meta` 覆盖（决定跑 v39 还是 v41）。逐 test_years 落盘分片，便于分块前台执行
（避开墙钟上限），后续内联合并。

用法（示例）：
    # v41 候选（2019-2022 分片）
    python scripts/40_retrain_v41_baseline.py --ds dataset_h10_v41.parquet \
        --meta dataset_h10_v41_meta.json --test-years 2019,2020,2021,2022 \
        --out-part pred_baseline_h10_v41_part1.parquet --yearly-part yearly_v41_part1.json
    # v39 同模型对照（不覆盖生产 lightgbm pred，落 *_hgb 控制名）
    python scripts/40_retrain_v41_baseline.py --ds dataset_h10_v39.parquet \
        --meta dataset_h10_v39_meta.json --test-years 2019,2020,2021,2022 \
        --out-part pred_baseline_h10_v39_hgb_part1.parquet --yearly-part yearly_v39_hgb_part1.json
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
from src.models import baseline_hgb

logger = get_logger("P1.v41_baseline_hgb")
PROCESSED = paths.DATA / "P1" / "processed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--ds", type=str, required=True, help="数据集 parquet 文件名")
    ap.add_argument("--meta", type=str, required=True, help="数据集 meta 文件名")
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
        .replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["y_excess"])
    if args.label_clip > 0:
        data = data[data["y_excess"].abs() <= args.label_clip]
    fcols = [c for c in data.columns if c not in ("date", "symbol", "y_cls", "y_fwd")]
    data[fcols] = data[fcols].astype(np.float32)
    del fcols
    logger.info("%s 数据集 %s 行 | %s 因子 | 分片年份 %s",
                args.ds, f"{len(data):,}", len(factor_names), test_years)

    t0 = time.time()
    preds, yearly = baseline_hgb.walk_forward(
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
    logger.info("%s 分片 %s 完成：%s 年 | best_iteration %s | %.1fs",
                args.ds, test_years, len(yearly), bi, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
