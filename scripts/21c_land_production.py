"""阶段 23 - ①落地：把 input_norm 重训权重落到生产信号（全宇宙覆盖）。

背景：
    21_retrain_ev_inputnorm.py 以 max_symbols=800 子集重训，产出的
    pred_gru_ev_t10_inputnorm_h10.parquet 仅覆盖 789 只（OLD 生产信号覆盖 1411 只）。
    直接覆盖会缩小可交易宇宙 → 不是干净的「落地」。

    本脚本只做预测（不训练），用与 14_event_factor_iterate.py 产出 OLD 生产信号
    **完全一致**的掩码（全宇宙 symbol_subset=None / train_end=2024-12-31 /
    valid_end=2025-12-31 / label_horizon=10），加载 21 的 input_norm 模型权重，
    对 2026 严格 hold-out 窗口重新预测 → 干净的单模型、全宇宙覆盖替换。

    GRUAttention 本身符号无关（对 47 维序列建模，不用股票身份），故 800 子集训练的
    权重可安全预测全 1411 只的序列。模型训练数据截至 ~2025-12-08，2026 窗口为
    样本外，无泄漏。

产物（临时，校验后交换）：
    data/P1/processed/pred_gru_ev_t10_seq40_h128_lr1e-3_h10_inputnorm_fullcov.parquet

用法：
    python scripts/21c_land_production.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]      # E:/project/sj
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths
from shared.logging_utils import get_logger
from src.models import dataset as ds_mod
from src.models import gru_attn
from src.models import trainer

logger = get_logger("P1.land_production")
PROCESSED = paths.DATA / "P1" / "processed"

HORIZON = 10
SEQ_LEN = 40
HIDDEN = 128
LAYERS = 2
TRAIN_END = "2024-12-31"
VALID_END = "2025-12-31"
LABEL_CLIP = 0.5
NEW_PT = paths.MODELS / "P1" / "gru_ev_t10_inputnorm_full.pt"
OUT = PROCESSED / "pred_gru_ev_t10_seq40_h128_lr1e-3_h10_inputnorm_fullcov.parquet"


def main() -> int:
    t0 = time.time()
    # 1) 加载 EV 47 维数据集（与 14 同口径）
    meta = json.load(open(PROCESSED / f"dataset_h{HORIZON}_ev_meta.json", encoding="utf-8"))
    factor_names = meta["factor_names"]
    data = pd.read_parquet(PROCESSED / f"dataset_h{HORIZON}_ev.parquet")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    if LABEL_CLIP > 0:
        data = data[data["y_excess"].abs() <= LABEL_CLIP]
    X3d, y3d, dates, symbols = ds_mod.build_3d(data, factor_names)
    logger.info("EV 数据集 %s 行 | %s 只 | 输入维度 %s", f"{len(data):,}", len(symbols), X3d.shape[2])

    # 2) 与 OLD 生产信号完全一致的掩码（全宇宙）。
    #    注意：OLD 由旧版 make_masks 生成（label_horizon=0，不回退），
    #    故 valid_cut=2025-12-31、test 自 2026-01-05 起。这里显式 label_horizon=0
    #    以 1:1 复刻 OLD 的测试窗口（1411 只 / 149 日 / 2026 窗口）。
    tr_m, va_m, te_m, sym_m = ds_mod.make_masks(
        dates, TRAIN_END, VALID_END, symbols, symbol_subset=None, label_horizon=0)
    logger.info("掩码: train=%s valid=%s test=%s | 全宇宙 sym=%s",
                int(tr_m.sum()), int(va_m.sum()), int(te_m.sum()), int(sym_m.sum()))

    test_ds = ds_mod.SequenceDataset(X3d, y3d, SEQ_LEN, sym_m, te_m, date_stride=1)
    logger.info("test 样本数: %s", f"{len(test_ds):,}")

    # 3) 加载 input_norm 模型权重（不训练）
    model = gru_attn.GRUAttention(X3d.shape[2], hidden=HIDDEN, n_layers=LAYERS)
    sd = torch.load(NEW_PT, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)
    model.eval()
    logger.info("已加载 input_norm 权重: %s", NEW_PT.name)

    # 4) 预测
    pred = trainer.predict(model, test_ds, num_threads=8)
    idx = test_ds.indices
    pred_df = pd.DataFrame({
        "date": pd.to_datetime(dates[idx[:, 1]]),
        "symbol": symbols[idx[:, 0]],
        "pred": np.asarray(pred, dtype=np.float32),
        "y_excess": y3d[idx[:, 0], idx[:, 1]].astype(np.float32),
        "year": pd.to_datetime(dates[idx[:, 1]]).year.astype(np.int32),
    })
    pred_df.to_parquet(OUT, index=False)
    logger.info("已落盘(临时): %s | %s 行 / %s 只 / %s~%s | 耗时 %.1fs",
                OUT.name, f"{len(pred_df):,}", pred_df.symbol.nunique(),
                pred_df.date.min().date(), pred_df.date.max().date(), time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
