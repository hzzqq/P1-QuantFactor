"""Predict-only EV 推理（不重训）：用生产 EV 权重对刷新后的 dataset_h10_ev 延伸 2026 预测。

设计：完全复用 21c_land_production.py 的掩码口径（全宇宙 symbol_subset=None /
train_end=2024-12-31 / valid_end=2025-12-31 / label_horizon=0 / label_clip=0.5），
加载生产权重 gru_ev_t10_seq40_h128_lr1e-3_h10.pt，对 2026 严格 hold-out 窗口
重新预测 → 覆盖写 pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet（调用方须先备份）。

口径铁律：
    - 只推理不训练：模型权重与已部署信号完全一致，无重训漂移；
    - in_dim 守卫：数据集因子数必须等于权重 input_norm 维度（47），不符立即失败；
    - 2026 窗口为样本外（训练数据截至 ~2025-12），无泄漏。

用法：
    python scripts/zz_predict_only_ev.py --threads 8
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
from shared.logging_utils import get_logger  # noqa: E402
from src.models import dataset as ds_mod  # noqa: E402
from src.models import gru_attn  # noqa: E402
from src.models import trainer  # noqa: E402

logger = get_logger("P1.predict_only_ev")
PROCESSED = paths.DATA / "P1" / "processed"
MODELS_DIR = paths.MODELS / "P1"

HORIZON = 10
SEQ_LEN = 40
HIDDEN = 128
LAYERS = 2
TRAIN_END = "2024-12-31"
VALID_END = "2025-12-31"
LABEL_CLIP = 0.5
DEFAULT_WEIGHT = "gru_ev_t10_seq40_h128_lr1e-3_h10.pt"
OUT = PROCESSED / f"pred_gru_ev_t10_seq40_h128_lr1e-3_h{HORIZON}.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--weight", type=str, default=DEFAULT_WEIGHT)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = ap.parse_args()
    t0 = time.time()

    meta_path = PROCESSED / f"dataset_h{HORIZON}_ev_meta.json"
    if not meta_path.exists():
        logger.error("找不到 %s，请先 14_event_factor_iterate.py --build-only", meta_path)
        return 1
    factor_names = json.load(open(meta_path, encoding="utf-8"))["factor_names"]

    data = pd.read_parquet(PROCESSED / f"dataset_h{HORIZON}_ev.parquet")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    if LABEL_CLIP > 0:
        data = data[data["y_excess"].abs() <= LABEL_CLIP]
    logger.info("EV 数据集 %s 行 | %s 只 | %s 因子",
                f"{len(data):,}", data["symbol"].nunique(), len(factor_names))

    X3d, y3d, dates, symbols = ds_mod.build_3d(data, factor_names)
    del data
    gc.collect()
    n_f = X3d.shape[2]

    sd = torch.load(MODELS_DIR / args.weight, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    w_dim = int(sd["input_norm.weight"].shape[0])
    hid = int(sd["gru.weight_ih_l0"].shape[0] // 3)
    nl = sum(1 for k in sd if k.startswith("gru.weight_ih_l"))
    if w_dim != n_f or hid != HIDDEN or nl != LAYERS:
        logger.error("权重口径不符: in_dim %s!=%s / hidden %s!=%s / layers %s!=%s —— 拒绝推理",
                     w_dim, n_f, hid, HIDDEN, nl, LAYERS)
        return 1

    tr_m, va_m, te_m, sym_m = ds_mod.make_masks(
        dates, TRAIN_END, VALID_END, symbols, None, label_horizon=0)
    logger.info("掩码: train=%s valid=%s test=%s | 全宇宙 sym=%s",
                int(tr_m.sum()), int(va_m.sum()), int(te_m.sum()), int(sym_m.sum()))
    if te_m.sum() == 0:
        logger.error("2026 测试窗口为空")
        return 1
    test_ds = ds_mod.SequenceDataset(X3d, y3d, args.seq_len, sym_m, te_m, date_stride=1)
    logger.info("test 样本数: %s", f"{len(test_ds):,}")

    model = gru_attn.GRUAttention(n_f, hidden=hid, n_layers=nl)
    model.load_state_dict(sd)
    model.eval()
    logger.info("已加载生产权重 %s (in_dim=%s h=%s l=%s)，开始推理...",
                args.weight, n_f, hid, nl)
    pred = trainer.predict(model, test_ds, num_threads=args.threads)
    idx = test_ds.indices
    pred_df = pd.DataFrame({
        "date": pd.to_datetime(dates[idx[:, 1]]),
        "symbol": np.asarray(symbols)[idx[:, 0]],
        "pred": np.asarray(pred, dtype=np.float32),
        "y_excess": y3d[idx[:, 0], idx[:, 1]].astype(np.float32),
        "year": pd.to_datetime(dates[idx[:, 1]]).year.astype(np.int32),
    })
    pred_df.to_parquet(OUT, index=False)
    logger.info("已写出 %s (%s 行, %s 只, date %s~%s, %.1fs)",
                OUT.name, f"{len(pred_df):,}", pred_df["symbol"].nunique(),
                pred_df["date"].min().date(), pred_df["date"].max().date(),
                time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
