"""阶段 3：训练 LightGBM 基线并评估（含滚动验证）。

用法：
    python scripts/03_train_baseline.py
    python scripts/03_train_baseline.py --rounds 200 --max-train-rows 2000000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # E:/project/sj
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                               # noqa: E402
import pandas as pd                              # noqa: E402

from shared import paths                         # noqa: E402
from shared.config import load_config            # noqa: E402
from shared.logging_utils import get_logger      # noqa: E402

from src.eval import metrics                     # noqa: E402
from src.models import baseline_lgb              # noqa: E402

logger = get_logger("P1.train_baseline")
CONFIG_PATH = PROJ / "config" / "default.yaml"
PROCESSED = paths.DATA / "P1" / "processed"
MODELS_DIR = paths.MODELS / "P1"


def _fmt(v, digits=4):
    if v is None or v != v:      # NaN
        return "     -"
    return f"{v: .{digits}f}"


def print_report(overall: dict, by_year: dict, qr: pd.DataFrame) -> None:
    print("\n" + "=" * 66)
    print("  LightGBM 基线 · 滚动验证结果")
    print("=" * 66)
    print(f"  IC 均值      : {_fmt(overall.get('ic_mean'))}"
          f"   （>0.02 有信息，>0.05 不错，>0.08 很强）")
    print(f"  IC 标准差    : {_fmt(overall.get('ic_std'))}")
    print(f"  ICIR         : {_fmt(overall.get('icir'))}"
          f"   （未年化 >0.3 可用）")
    print(f"  ICIR(年化)   : {_fmt(overall.get('icir_annual'), 2)}")
    print(f"  IC 为正占比  : {_fmt(overall.get('ic_positive_rate'), 3)}")
    print(f"  t 统计量     : {_fmt(overall.get('t_stat'), 2)}"
          f"   （|t|>2 表示显著不为 0）")
    print(f"  有效交易日   : {overall.get('n_days')}")
    if "spread_mean" in overall:
        print(f"  多空收益差   : {_fmt(overall.get('spread_mean'))} / 期")
        print(f"  多空胜率     : {_fmt(overall.get('spread_win_rate'), 3)}")
        print(f"  多空夏普     : {_fmt(overall.get('spread_sharpe'), 2)}")

    print("\n  分年度 IC：")
    print("   年份      IC均值     ICIR    正占比    样本数")
    for y in sorted(by_year):
        s = by_year[y]
        print(f"   {y}   {_fmt(s.get('ic_mean'), 4)}  "
              f"{_fmt(s.get('icir'), 4)}  {_fmt(s.get('ic_positive_rate'), 3)}"
              f"   {s.get('n_days')}")

    if not qr.empty:
        print("\n  分组超额收益（第1组=预测最弱，第5组=预测最强）：")
        print("   组号    平均超额收益      标准差      样本数")
        for g, row in qr.iterrows():
            print(f"    {g}      {_fmt(row['mean'], 5)}     "
                  f"{_fmt(row['std'], 5)}    {int(row['count']):>8,}")
    print("=" * 66 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 训练 LightGBM 基线")
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--max-train-rows", type=int, default=0,
                        help="训练集最大行数，0 表示不限（控制训练耗时）")
    parser.add_argument("--label-clip", type=float, default=0.5,
                        help="剔除 |超额收益| 超过该值的样本，须与 04_train_nn.py 保持一致")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    horizon = args.horizon or cfg.get_path("label.horizon", 10)

    meta_path = PROCESSED / f"dataset_h{horizon}_meta.json"
    if not meta_path.exists():
        logger.error("找不到数据集元信息 %s，请先运行 02_build_features.py", meta_path)
        return 1
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    factor_names = meta["factor_names"]

    ds_path = PROCESSED / f"dataset_h{horizon}.parquet"
    logger.info("加载数据集 %s", ds_path)
    data = pd.read_parquet(ds_path).replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["y_excess"])
    if args.label_clip > 0:
        before = len(data)
        data = data[data["y_excess"].abs() <= args.label_clip]
        logger.info("标签截断 |y|<=%s：%s -> %s 行（剔除 %s）",
                    args.label_clip, f"{before:,}", f"{len(data):,}",
                    f"{before - len(data):,}")
    logger.info("数据集 %s 行 | %s 只股票 | %s 个因子",
                f"{len(data):,}", data["symbol"].nunique(), len(factor_names))

    t0 = time.time()
    preds, yearly = baseline_lgb.walk_forward(
        data, factor_names, y_col="y_excess",
        train_years=args.train_years,
        num_boost_round=args.rounds,
        max_train_rows=args.max_train_rows or None,
    )
    if preds.empty:
        logger.error("没有产出任何预测，请检查数据量是否足够")
        return 1

    overall = metrics.summarize(preds, "pred", "y_excess")
    by_year = {int(y): metrics.summarize(g, "pred", "y_excess")
               for y, g in preds.groupby("year")}
    qr = metrics.quantile_returns(preds, "pred", "y_excess", n_groups=5)
    print_report(overall, by_year, qr)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_name = args.out or f"pred_baseline_h{horizon}.parquet"
    preds.to_parquet(PROCESSED / out_name, index=False)

    report = {
        "model": "lightgbm",
        "horizon": horizon,
        "train_years": args.train_years,
        "num_boost_round": args.rounds,
        "overall": overall,
        "by_year": {str(k): v for k, v in by_year.items()},
        "quantile_returns": qr.reset_index().to_dict(orient="records"),
        "yearly_training": yearly,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    rp = PROCESSED / f"report_baseline_h{horizon}.json"
    with rp.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    logger.info("预测已保存: %s", PROCESSED / out_name)
    logger.info("报告已保存: %s", rp)
    logger.info("总耗时 %.1f 秒", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
