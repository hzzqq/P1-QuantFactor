"""Predict-only Transformer 推理（不重训）：用 transformer_best_h10_{y}.pt 对刷新后的
dataset_h10 延伸 2022~2026 预测，供多模型融合升级（T-255）做同窗对照。

设计：完全复用 04_train_nn.py --model transformer 的口径（seq20 / hidden=64 /
layers=2 / heads=4 / ff=256 / label_clip=0.5 / masks train_end=y-2 / valid_end=y-1），
只推理不训练——权重与 09-25 对照实验（IC 0.0465 vs GRU 0.0374）完全一致，无重训漂移。

产物：data/P1/processed/pred_transformer_h10.parquet（调用方须先备份）。

用法：
    python scripts/zz_predict_only_transformer.py --threads 8
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths  # noqa: E402
from shared.config import load_config  # noqa: E402
from shared.logging_utils import get_logger  # noqa: E402
from src.models import dataset as ds_mod  # noqa: E402
from src.models import transformer_signal  # noqa: E402
from src.models import trainer  # noqa: E402

logger = get_logger("P1.predict_only_transformer")
CONFIG_PATH = PROJ / "config" / "default.yaml"
PROCESSED = paths.DATA / "P1" / "processed"
MODELS_DIR = paths.MODELS / "P1"
OUT = PROCESSED / "pred_transformer_h10.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label-clip", type=float, default=0.5)
    ap.add_argument("--train-years", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(CONFIG_PATH)
    horizon = args.horizon
    meta_path = PROCESSED / f"dataset_h{horizon}_meta.json"
    if not meta_path.exists():
        logger.error("找不到 %s，请先 02_build_features.py", meta_path)
        return 1
    factor_names = json.load(open(meta_path, encoding="utf-8"))["factor_names"]

    t0 = time.time()
    data = pd.read_parquet(PROCESSED / f"dataset_h{horizon}.parquet")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    if args.label_clip > 0:
        before = len(data)
        data = data[data["y_excess"].abs() <= args.label_clip]
        logger.info("标签截断: %s -> %s", f"{before:,}", f"{len(data):,}")
    logger.info("数据集 %s 行 | %s 只 | %s 因子",
                f"{len(data):,}", data["symbol"].nunique(), len(factor_names))

    X3d, y3d, dates, symbols = ds_mod.build_3d(data, factor_names)
    del data
    gc.collect()
    n_f = X3d.shape[2]

    years = sorted(pd.to_datetime(pd.Series(dates)).dt.year.unique())
    test_years = [y for y in years if y >= years[0] + args.train_years + 1]
    logger.info("test_years=%s", test_years)

    all_preds = []
    for y in test_years:
        train_end = f"{y - 2}-12-31"
        valid_end = f"{y - 1}-12-31"
        tr_m, va_m, te_m, sym_m = ds_mod.make_masks(
            dates, train_end, valid_end, symbols, None)
        if te_m.sum() == 0:
            logger.warning("跳过 %s：测试区间为空", y)
            continue
        test_ds = ds_mod.SequenceDataset(X3d, y3d, args.seq_len, sym_m, te_m,
                                         date_stride=1)
        ckpt = MODELS_DIR / f"transformer_best_h{horizon}_{y}.pt"
        if not ckpt.exists():
            logger.warning("跳过 %s：无权重 %s", y, ckpt.name)
            continue
        sd = torch.load(ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # 架构自权重推断（与训练口径强校验）
        hidden = int(sd["proj.weight"].shape[0])
        n_layers = len({k.split(".")[2] for k in sd if k.startswith("encoder.layers.")})
        if int(sd["proj.weight"].shape[1]) != n_f:
            logger.error("%s 权重 in_dim=%s != 数据集 %s，拒绝推理",
                         y, sd["proj.weight"].shape[1], n_f)
            return 1
        model = transformer_signal.TransformerSignal(
            n_f, hidden=hidden, n_layers=n_layers,
            n_heads=4, dim_feedforward=256, dropout=0.2)
        model.load_state_dict(sd)
        model.eval()
        logger.info("%s 推理：test=%s 样本 | arch h=%s l=%s", y,
                    f"{len(test_ds):,}", hidden, n_layers)
        pred = trainer.predict(model, test_ds, num_threads=args.threads)
        idx = test_ds.indices
        all_preds.append(pd.DataFrame({
            "date": pd.to_datetime(dates[idx[:, 1]]),
            "symbol": np.asarray(symbols)[idx[:, 0]],
            "pred": pred,
            "y_excess": y3d[idx[:, 0], idx[:, 1]],
            "year": y,
        }))

    if not all_preds:
        logger.error("无预测产出")
        return 1
    preds = pd.concat(all_preds, ignore_index=True)
    preds = preds.drop_duplicates(subset=["date", "symbol"], keep="last")
    preds.to_parquet(OUT, index=False)
    logger.info("已写出 %s (%s 行, date %s~%s, %.1fs)",
                OUT.name, f"{len(preds):,}",
                preds["date"].min().date(), preds["date"].max().date(),
                time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
