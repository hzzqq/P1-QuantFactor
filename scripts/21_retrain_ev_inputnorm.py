"""阶段 21：EV 47 维生产模型 · 输入归一(N13) 重训 · 单锚点 t10 全量。

目的：把阶段 22 的 N13（GRUAttention.input_norm）落到**生产权重**。
隔离变量 —— 除 input_norm 外，超参与训练例程与 `14_event_factor_iterate.py` 产出
`pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet` 的 t10 完全一致（seq40/h128/lr1e-3/
layers=2，普通 train_model，非成本敏感），故新权重 vs 旧 M1(raw EV) 的差异**仅来自
input_norm**。

安全范式（继承 20a）：仅训练+预测+**立即落盘**，不做回测；float32；唯一进程标题
便于孤儿清理；build_3d 后早释 data。约 25–28 分钟，后台跑。

输出：
  - models/P1/gru_ev_t10_inputnorm_full.pt
  - data/P1/processed/pred_gru_ev_t10_inputnorm_h10.parquet

用法：
    python scripts/21_retrain_ev_inputnorm.py --smoke     # 800 子集冒烟
    python scripts/21_retrain_ev_inputnorm.py             # 全量（后台 ~28min）
"""
from __future__ import annotations

import argparse
import json
import os
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
# t10 = 与旧生产权重完全一致的超参（仅差 input_norm）
SEQ, HID, LR, LAYERS = 40, 128, 1e-3, 2
COST_LAM = 0.0  # 普通 GRU 训练（与 14 的 t10 路径一致，隔离 N13）
TRAIN_END, VALID_END = "2024-12-31", "2025-12-31"
logger = get_logger("P1.ev21_inputnorm")


def load_ev(horizon, label_clip, max_symbols):
    meta = json.load(open(PROCESSED / f"dataset_h{horizon}_ev_meta.json", encoding="utf-8"))
    fn = meta["factor_names"]
    data = pd.read_parquet(PROCESSED / f"dataset_h{horizon}_ev.parquet")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    if label_clip > 0:
        data = data[data["y_excess"].abs() <= label_clip]
    X3d, y3d, dates, symbols = ds_mod.build_3d(data, fn)
    del data  # 省内存：build_3d 后立即释放原始 DataFrame
    X3d = X3d.astype(np.float32)
    y3d = y3d.astype(np.float32)
    subset = None
    if max_symbols and 0 < max_symbols < len(symbols):
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(len(symbols), max_symbols, replace=False))
        subset = symbols[keep]
    return None, X3d, y3d, dates, symbols, subset, fn


def make_test_pred(X3d, y3d, dates, symbols, subset, train_end, valid_end,
                   tag, epochs, batch, stride, threads, lr=LR, hidden=HID,
                   seq=SEQ, layers=LAYERS, cost_lam=0.0, seed=42,
                   label_horizon: int = 0, checkpoint_path: str | None = None):
    tr_m, va_m, te_m, sym_m = ds_mod.make_masks(
        dates, train_end, valid_end, symbols, subset,
        label_horizon=label_horizon)
    if tr_m.sum() == 0 or te_m.sum() == 0:
        raise RuntimeError(f"空区间 train={tr_m.sum()} test={te_m.sum()}")
    train_ds = ds_mod.SequenceDataset(X3d, y3d, seq, sym_m, tr_m, date_stride=stride)
    valid_ds = ds_mod.SequenceDataset(X3d, y3d, seq, sym_m, va_m, date_stride=max(1, stride * 2))
    test_ds = ds_mod.SequenceDataset(X3d, y3d, seq, sym_m, te_m, date_stride=1)
    logger.info("[%s] train=%s test=%s in_dim=%s (input_norm ON)", tag,
                f"{len(train_ds):,}", f"{len(test_ds):,}", X3d.shape[2])
    model = gru_attn.GRUAttention(X3d.shape[2], hidden=hidden, n_layers=layers)
    if cost_lam > 0:
        reg = lambda p: cost_lam * p.pow(2).mean()
        model, history = train_common.train_robust(
            model, train_ds, valid_ds=valid_ds, epochs=epochs, batch_size=batch,
            lr=lr, weight_decay=1e-5, patience=4, grad_clip=1.0,
            num_threads=threads, seed=seed, loss_type="huber", reg_term=reg,
            checkpoint_path=checkpoint_path)
    else:
        # 与 14 的 t10 普通训练路径一致（隔离 N13）
        model, history = _trainer_mod.train_model(
            model, train_ds, valid_ds, epochs=epochs, batch_size=batch, lr=lr,
            patience=4, num_threads=threads, seed=seed,
            checkpoint_path=checkpoint_path)
    # 早停判定（仅 early-stop 触发的完成；是否达总轮数由 main 对照 args.epochs 判定）
    stopped = False
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            stopped = bool(ck.get("stopped"))
        except Exception:
            stopped = False
    return model, len(history), stopped, test_ds


def main() -> int:
    ap = argparse.ArgumentParser(description="EV t10 输入归一重训 · 全量 · 仅训练+预测+落盘（断点续训）")
    ap.add_argument("--smoke", action="store_true", help="800 子集冒烟（epochs=12 快速验证管线）")
    ap.add_argument("--horizon", type=int, default=10)
    # 与 14_event_factor_iterate.py 的 t10 生产配置严格对齐：epochs=15 / batch=1024 / stride=5
    # / threads=8 / label_clip=0.5 / seed=42 / 800 子集，从而「仅差 input_norm」做 apples-to-apples
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label-clip", type=float, default=0.5)
    ap.add_argument("--max-symbols", type=int, default=800,
                    help="训练子集大小；800 与 14 生产一致（seed=42 选同一批）；0=全量 1427")
    ap.add_argument("--chunk", type=int, default=4,
                    help="每轮调用最多训练的轮数（断点续训用，避免被墙钟 SIGKILL）")
    args = ap.parse_args()

    CKPT = paths.MODELS / "P1" / "ev21_inputnorm_ckpt.pt"
    # 从检查点推断已完成轮数，本 chunk 仅再跑 --chunk 轮（保证单次调用在超时内结束）
    start = 0
    if os.path.exists(CKPT):
        try:
            ck = torch.load(CKPT, map_location="cpu", weights_only=False)
            start = int(ck.get("epoch", 0))
        except Exception:
            start = 0
    chunk_epochs = min(args.epochs, start + args.chunk)

    max_sym = args.max_symbols
    t0 = time.time()
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW("P1_EV21_inputnorm_train")
    except Exception:
        pass
    logger.info("加载 EV 47 维全量数据集 (max_symbols=%s) ...", max_sym or "全量")
    _, X3d, y3d, dates, symbols, subset, fn = load_ev(
        args.horizon, args.label_clip, max_sym)

    logger.info("=== EV t10 输入归一重训 (input_norm ON, cost_lam=%s) 全量 ===", COST_LAM)
    model, ep, stopped, test_ds = make_test_pred(
        X3d, y3d, dates, symbols, subset, TRAIN_END, VALID_END,
        "EV_t10_inputnorm", chunk_epochs, args.batch, args.stride, args.threads,
        lr=LR, hidden=HID, seq=SEQ, layers=LAYERS, cost_lam=COST_LAM,
        label_horizon=args.horizon, checkpoint_path=str(CKPT))

    # 完成判定：early-stop 触发，或已训练到总目标轮数 args.epochs
    complete = stopped
    if not complete and os.path.exists(CKPT):
        try:
            ck = torch.load(CKPT, map_location="cpu", weights_only=False)
            complete = int(ck.get("epoch", 0)) >= args.epochs
        except Exception:
            complete = False

    if not complete:
        logger.info("本 chunk 训练至 ~%s/%s 轮（检查点已落盘 %s），下次调用自动续训；本段耗时 %.1fs",
                    chunk_epochs, args.epochs, CKPT.name, time.time() - t0)
        return 0

    # 完成：预测 + 立即落盘（不做回测）
    pred = _trainer_mod.predict(model, test_ds, num_threads=args.threads)
    idx = test_ds.indices
    pdf = pd.DataFrame({
        "date": dates[idx[:, 1]],
        "symbol": symbols[idx[:, 0]],
        "pred": pred,
        "y_excess": y3d[idx[:, 0], idx[:, 1]],
        "year": pd.to_datetime(dates[idx[:, 1]]).year,
    })
    ic = metrics.summarize(pdf, "pred", "y_excess")
    logger.info("训练完成 IC %.4f ICIR %.4f；立即落盘模型+预测（不做回测）",
                ic.get("ic_mean"), ic.get("icir"))

    if not args.smoke:
        paths.MODELS.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "in_dim": X3d.shape[2], "hidden": HID, "n_layers": LAYERS,
                    "seq": SEQ, "cost_lam": COST_LAM, "input_norm": True,
                    "train_end": TRAIN_END, "valid_end": VALID_END,
                    "note": "N13 input_norm retrain of t10, apples-to-apples vs stage14/15"},
                   paths.MODELS / "P1" / "gru_ev_t10_inputnorm_full.pt")
        pdf.to_parquet(PROCESSED / "pred_gru_ev_t10_inputnorm_h10.parquet", index=False)
        # 训练完成，清理检查点避免误用
        try:
            os.remove(CKPT)
        except Exception:
            pass
        logger.info("已落盘: gru_ev_t10_inputnorm_full.pt / pred_gru_ev_t10_inputnorm_h10.parquet（下一步跑 21b 出报告）")
    else:
        logger.info("smoke 模式不落盘全量产物")

    logger.info("总耗时 %.1f 秒", time.time() - t0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("EV t10 输入归一重训异常退出")
        raise
