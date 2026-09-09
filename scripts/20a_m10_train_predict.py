"""阶段 20a：M10 成本敏感训练 · 全量 1427 · 仅训练+预测+落盘（不回测）。

拆自 `20_m10_full_confirm.py`，专门规避后台任务的墙钟上限：训练约 27–28 分钟，
本脚本在训练后**立即**落盘模型 + 2026 全量预测再退出，不做任何回测。
回测/对比由 `20b_m10_report.py`（秒级、单独跑）完成。

输出：
  - models/P1/gru_m10_cost_full.pt
  - data/P1/processed/pred_gru_m10_cost_full_h10.parquet

用法：
    python scripts/20a_m10_train_predict.py            # 全量（后台 ~28min）
    python scripts/20a_m10_train_predict.py --smoke    # 800 子集冒烟
"""
from __future__ import annotations

import argparse
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

from shared import paths
from shared.config import load_config
from shared.logging_utils import get_logger
from src.models import dataset as ds_mod, gru_attn
from src.models import trainer as _trainer_mod
from src.training import common as train_common
from src.eval import metrics

PROCESSED = paths.DATA / "P1" / "processed"
CONFIG_PATH = PROJ / "config" / "default.yaml"
SEQ, HID, LR, LAYERS = 40, 128, 1e-3, 2
COST_LAM = 0.02
TRAIN_END, VALID_END = "2024-12-31", "2025-12-31"
logger = get_logger("P1.m10_train_predict")


def load_ev(horizon, label_clip, max_symbols):
    meta = json.load(open(PROCESSED / f"dataset_h{horizon}_ev_meta.json", encoding="utf-8"))
    fn = meta["factor_names"]
    data = pd.read_parquet(PROCESSED / f"dataset_h{horizon}_ev.parquet")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    if label_clip > 0:
        data = data[data["y_excess"].abs() <= label_clip]
    X3d, y3d, dates, symbols = ds_mod.build_3d(data, fn)
    # 省内存：原始 DataFrame 巨大（~1.3GB float64），build_3d 后立即释放，避免 OOM
    del data
    # 用 float32 减半三维数组内存（1.7GB→0.85GB），与并存实验(exp_jump.py)错峰，规避碎片 OOM
    X3d = X3d.astype(np.float32)
    y3d = y3d.astype(np.float32)
    subset = None
    if max_symbols and 0 < max_symbols < len(symbols):
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(len(symbols), max_symbols, replace=False))
        subset = symbols[keep]
    return None, X3d, y3d, dates, symbols, subset, fn


def _train_cost_sensitive(model, train_ds, valid_ds, epochs, batch, lr, threads,
                          cost_lam, seed):
    """M10 成本敏感训练 —— 委托给共享稳健训练（R7/R8 单一真相源）。

    仅多一个正则项 cost_lam·‖pred‖²（抑制预测幅度，等价于对成本敏感度的
    软约束）；梯度裁剪 / LR 调度 / 早停 / 稳健初始化全部来自 train_robust。
    """
    if cost_lam and cost_lam > 0:
        reg = lambda p: cost_lam * p.pow(2).mean()
    else:
        reg = None
    model, hist = train_common.train_robust(
        model, train_ds, valid_ds=valid_ds, epochs=epochs, batch_size=batch,
        lr=lr, weight_decay=1e-5, patience=4, grad_clip=1.0,
        num_threads=threads, seed=seed, loss_type="huber", reg_term=reg,
    )
    return model, hist


def make_test_pred(X3d, y3d, dates, symbols, subset, train_end, valid_end,
                   tag, epochs, batch, stride, threads, lr=LR, hidden=HID,
                   seq=SEQ, layers=LAYERS, cost_lam=0.0, seed=42,
                   label_horizon: int = 0):
    tr_m, va_m, te_m, sym_m = ds_mod.make_masks(
        dates, train_end, valid_end, symbols, subset,
        label_horizon=label_horizon)
    if tr_m.sum() == 0 or te_m.sum() == 0:
        raise RuntimeError(f"空区间 train={tr_m.sum()} test={te_m.sum()}")
    train_ds = ds_mod.SequenceDataset(X3d, y3d, seq, sym_m, tr_m, date_stride=stride)
    valid_ds = ds_mod.SequenceDataset(X3d, y3d, seq, sym_m, va_m, date_stride=max(1, stride * 2))
    test_ds = ds_mod.SequenceDataset(X3d, y3d, seq, sym_m, te_m, date_stride=1)
    logger.info("[%s] train=%s test=%s in_dim=%s", tag, f"{len(train_ds):,}",
                f"{len(test_ds):,}", X3d.shape[2])
    model = gru_attn.GRUAttention(X3d.shape[2], hidden=hidden, n_layers=layers)
    if cost_lam > 0:
        model, history = _train_cost_sensitive(
            model, train_ds, valid_ds, epochs, batch, lr, threads, cost_lam, seed)
    else:
        model, history = _trainer_mod.train_model(
            model, train_ds, valid_ds, epochs=epochs, batch_size=batch, lr=lr,
            patience=4, num_threads=threads, seed=seed)
    pred = _trainer_mod.predict(model, test_ds, num_threads=threads)
    idx = test_ds.indices
    pdf = pd.DataFrame({
        "date": dates[idx[:, 1]],
        "symbol": symbols[idx[:, 0]],
        "pred": pred,
        "y_excess": y3d[idx[:, 0], idx[:, 1]],
        "year": pd.to_datetime(dates[idx[:, 1]]).year,
    })
    return pdf, model, len(history)


def main() -> int:
    ap = argparse.ArgumentParser(description="M10 成本敏感训练 · 全量 · 仅训练+预测+落盘")
    ap.add_argument("--smoke", action="store_true", help="800 子集冒烟")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label-clip", type=float, default=0.5)
    args = ap.parse_args()

    max_sym = 800 if args.smoke else 0
    t0 = time.time()
    # 唯一进程标题，便于在孤儿进程堆积时精准识别/清理（避免误杀老板其他实验如 exp_jump.py）
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW("P1_M10_20a_train")
    except Exception:
        pass
    logger.info("加载 EV 47 维全量数据集 (max_symbols=%s) ...", max_sym or "全量")
    _, X3d, y3d, dates, symbols, subset, fn = load_ev(
        args.horizon, args.label_clip, max_sym)

    logger.info("=== M10 成本敏感训练 (cost_lam=%s) 全量 ===", COST_LAM)
    m10_pdf, m10_model, m10_ep = make_test_pred(
        X3d, y3d, dates, symbols, subset, TRAIN_END, VALID_END,
        "M10", args.epochs, args.batch, args.stride, args.threads,
        lr=LR, hidden=HID, seq=SEQ, layers=LAYERS, cost_lam=COST_LAM,
        label_horizon=args.horizon)
    m10_ic = metrics.summarize(m10_pdf, "pred", "y_excess")
    logger.info("M10 训练完成 IC %.4f ICIR %.4f；立即落盘模型+预测（不做回测）",
                m10_ic.get("ic_mean"), m10_ic.get("icir"))

    if not args.smoke:
        paths.MODELS.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": m10_model.state_dict(),
                    "in_dim": X3d.shape[2], "hidden": HID, "n_layers": LAYERS,
                    "seq": SEQ, "cost_lam": COST_LAM,
                    "train_end": TRAIN_END, "valid_end": VALID_END},
                   paths.MODELS / "P1" / "gru_m10_cost_full.pt")
        m10_pdf.to_parquet(PROCESSED / "pred_gru_m10_cost_full_h10.parquet", index=False)
        logger.info("已落盘: gru_m10_cost_full.pt / pred_gru_m10_cost_full_h10.parquet（下一步跑 20b 出报告）")
    else:
        logger.info("smoke 模式不落盘全量产物")

    logger.info("总耗时 %.1f 秒", time.time() - t0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("M10 训练+预测异常退出")
        raise
