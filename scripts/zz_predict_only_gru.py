"""Predict-only GRU 推理（不重训）：用已保存的 gru_best_h10_{y}.pt 对新区划跑推理。

设计：完全复用 04_train_nn.py 的 mask / build_3d / predict 逻辑，但跳过训练，
直接加载各 test_year 的静态权重，预测对应 test 窗口。这样 pred_gru_h10.parquet
的结构（date/symbol/pred/y_excess/year）与 04 完全一致，仅 2026 窗口随新数据延伸。

用法：
    python scripts/zz_predict_only_gru.py --horizon 10 --seq-len 20 --threads 8
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
from src.models import gru_attn  # noqa: E402
from src.models import trainer  # noqa: E402

logger = get_logger("P1.predict_only_gru")
CONFIG_PATH = PROJ / "config" / "default.yaml"
PROCESSED = paths.DATA / "P1" / "processed"
MODELS_DIR = paths.MODELS / "P1"


def _infer_arch(sd: dict) -> tuple[int, int]:
    """从 state_dict 推断 hidden / n_layers。"""
    hid = int(sd["gru.weight_ih_l0"].shape[0] // 3)
    nl = sum(1 for k in sd if k.startswith("gru.weight_ih_l"))
    return hid, nl


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
        ckpt = MODELS_DIR / f"gru_best_h{horizon}_{y}.pt"
        if not ckpt.exists():
            logger.warning("跳过 %s：无权重 %s", y, ckpt.name)
            continue
        sd = torch.load(ckpt, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        hid, nl = _infer_arch(sd)
        model = gru_attn.GRUAttention(n_f, hidden=hid, n_layers=nl)
        model.load_state_dict(sd)
        model.eval()
        logger.info("%s 推理：test=%s 样本 | arch h=%s l=%s", y,
                    f"{len(test_ds):,}", hid, nl)
        pred = trainer.predict(model, test_ds, num_threads=args.threads)
        idx = test_ds.indices
        all_preds.append(pd.DataFrame({
            "date": pd.to_datetime(dates[idx[:, 1]]),
            "symbol": np.array(symbols)[idx[:, 0]],
            "pred": pred,
            "y_excess": y3d[idx[:, 0], idx[:, 1]],
            "year": y,
        }))

    if not all_preds:
        logger.error("无预测产出")
        return 1
    preds = pd.concat(all_preds, ignore_index=True)
    preds = preds.drop_duplicates(subset=["date", "symbol"], keep="last")
    out = PROCESSED / f"pred_gru_h10.parquet"
    preds.to_parquet(out, index=False)
    logger.info("已写出 %s (%s 行, date %s~%s, %.1fs)",
                out.name, f"{len(preds):,}",
                preds["date"].min().date(), preds["date"].max().date(),
                time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
