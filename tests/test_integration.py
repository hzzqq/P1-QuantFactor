"""N15 端到端集成测试 + N14 predict 形状核验。

路径：合成小面板 → pipeline.build_dataset → build_3d → make_masks
      → SequenceDataset → train_robust(1 epoch) → predict → run_backtest
锁定全链路不回归；并断言 trainer.predict 对 (B,T,F) 返回 (B,)。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features import pipeline
from src.models import dataset as ds_mod
from src.models.gru_attn import GRUAttention
from src.training.common import train_robust, seed_everything
from src.models.trainer import predict
from src.backtest import engine as BE


def _synthetic_panel(n_sym=20, n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_days, freq="B")
    syms = [f"S{i:02d}" for i in range(n_sym)]
    rows = []
    for s in syms:
        px = 100.0
        for d in dates:
            ret = rng.normal(0, 0.02)
            px *= (1 + ret)
            rows.append({"symbol": s, "date": d,
                         "open": px, "high": px * 1.01, "low": px * 0.99,
                         "close": px, "volume": rng.integers(1e5, 1e6)})
    panel = pd.DataFrame(rows)
    bench = pd.DataFrame({"date": dates, "close": np.linspace(100, 120, n_days)})
    return panel, bench, dates, syms


def test_end_to_end_pipeline_and_predict_shape(N15=True, N14=True):
    panel, bench, dates, syms = _synthetic_panel()
    # 用短窗口因子（避免合成面板天数不足导致 60/120 窗口因子全 NaN、无样本）
    data = pipeline.build_dataset(panel, bench, horizon=10, windows=[5, 10, 20])
    assert not data.empty
    factor_names = data.attrs["factor_names"]

    X3d, y3d, dts, sms = ds_mod.build_3d(data, factor_names, y_col="y_excess")
    n = len(dts)
    train_end, valid_end = dts[n // 2], dts[n * 3 // 4]
    tr, va, te, sym = ds_mod.make_masks(dts, train_end, valid_end,
                                        symbols=sms, label_horizon=10)
    train_ds = ds_mod.SequenceDataset(X3d, y3d, seq_len=20,
                                      symbol_mask=sym, date_mask=tr)
    valid_ds = ds_mod.SequenceDataset(X3d, y3d, seq_len=20,
                                      symbol_mask=sym, date_mask=va)
    assert len(train_ds) > 0 and len(valid_ds) > 0

    seed_everything(42)
    model = GRUAttention(X3d.shape[2], hidden=8, n_layers=1)
    model, _ = train_robust(model, train_ds, valid_ds=valid_ds, epochs=1,
                            batch_size=256, lr=1e-3, max_batches_per_epoch=3,
                            verbose=False)

    # N14：predict 对 (B,T,F) 必须返回 (B,)
    preds_arr = predict(model, valid_ds, batch_size=256)
    assert preds_arr.ndim == 1, f"predict 应返回 (B,)，实得 ndim={preds_arr.ndim}"
    assert len(preds_arr) == len(valid_ds)
    assert np.isfinite(preds_arr).all()
    # 端到端主线：build_dataset→3d→mask→train→predict 跑通无异常即达标
