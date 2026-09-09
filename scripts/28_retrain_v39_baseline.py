"""阶段 27 - ②：baseline(LightGBM) 在 v39 数据集（38+log_amount）上复跑 walk-forward。

复用 03b 的分块逻辑，但数据指向 dataset_h10_v39.parquet（39 因子）。
逐 test_years 落盘预测 + yearly best_iteration，便于分块前台执行（避开 660s 墙钟）。

用法：
    python scripts/28_retrain_v39_baseline.py --test-years 2019,2020,2021,2022 \
        --out-part pred_baseline_h10_v39_part1.parquet --yearly-part yearly_v39_part1.json
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

logger = get_logger("P1.v39_baseline")
PROCESSED = paths.DATA / "P1" / "processed"
DS = "dataset_h10_v39.parquet"
META = "dataset_h10_v39_meta.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--ds", type=str, default=DS, help="数据集 parquet 文件名")
    ap.add_argument("--meta", type=str, default=META, help="数据集 meta 文件名")
    ap.add_argument("--test-years", type=str, required=True)
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--valid-years", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--label-clip", type=float, default=0.5)
    ap.add_argument("--label-horizon", type=int, default=10,
                    help="make_masks 回滚天数；h20 须传 20")
    ap.add_argument("--eval-metric", type=str, default="ic")
    ap.add_argument("--out-part", type=str, required=True)
    ap.add_argument("--yearly-part", type=str, required=True)
    args = ap.parse_args()

    meta = json.load(open(PROCESSED / args.meta, encoding="utf-8"))
    factor_names = meta["factor_names"]
    test_years = [int(y) for y in args.test_years.split(",") if y.strip()]
    # 内存控制：用 pyarrow 过滤器下推，只读取 test_years 所需历史窗口的行组
    # （dataset 按 date 排序分多行组，pushdown 跳过不匹配行组）。walk_forward 对测试年 Y
    # 仅用 [Y-3-2, Y]（train_years=3 + valid_years=2），故下界取 Y-5，避免读多余年份撑爆
    # 环境 per-process commit 上限 OOM。
    lo_year = min(test_years) - (args.train_years + 2)
    hi_year = max(test_years)
    lo = pd.Timestamp(year=lo_year, month=1, day=1)
    hi = pd.Timestamp(year=hi_year, month=12, day=31)
    data = pd.read_parquet(PROCESSED / args.ds, filters=[("date", ">=", lo), ("date", "<=", hi)]) \
        .replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    # 因子列 downcast 到 float32，减半内存
    fcols = [c for c in data.columns if c not in ("date", "symbol", "y_cls", "y_fwd")]
    data[fcols] = data[fcols].astype(np.float32)
    del fcols
    if args.label_clip > 0:
        data = data[data["y_excess"].abs() <= args.label_clip]
    logger.info("v39 数据集 %s 行 | %s 因子", f"{len(data):,}", len(factor_names))

    test_years = [int(y) for y in args.test_years.split(",") if y.strip()]
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
    logger.info("v39 分片 %s 完成：%s 年 | best_iteration %s | %.1fs",
                test_years, len(yearly), bi, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
