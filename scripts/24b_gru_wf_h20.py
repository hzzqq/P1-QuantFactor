"""阶段 24·②：GRU(EV 47 维) 多年级 walk-forward，但 horizon=20（更长持有期）。

与 24_gru_walkforward.py 同构，关键差异：
  1) HORIZON=20（读 dataset_h20_ev.parquet，由 27c 产出）；
  2) **修复 GRU 去重红线**：逐年测试掩码原 `(d>valid_end)` 无上界 → 溢出到后续年份，
     同一 (date,symbol) 被多个年份模型重复预测（v41 GRU 曾 1.86× 冗余）。
     这里 `te_m &= (dates.year == year)` 钳制，每年模型只预测自己这一年；
  3) 产物独立命名 pred_gru_wf_h20_{year}.parquet / report_gru_wf_h20.json，不与 h10 混淆。

用法（年份 2019–2026 各跑一次，断点续训）：
    python scripts/24b_gru_wf_h20.py --year 2019
    ...
"""
from __future__ import annotations
import argparse, json, os, sys, time
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
from shared.logging_utils import get_logger
from src.models import dataset as ds_mod, gru_attn
from src.models import trainer as _trainer_mod
from src.training import common as train_common
from src.eval import metrics

PROCESSED = paths.DATA / "P1" / "processed"
SEQ, HID, LR, LAYERS = 40, 128, 1e-3, 2
VALID_YEARS = 2
HORIZON = 20
TAG = "h20"
logger = get_logger("P1.gru_wf_h20")


def get_X3d_cache(horizon, label_clip, max_symbols, ds_suffix=""):
    x3d_path = PROCESSED / f"_gru_wf_X3d_h{horizon}_ms{max_symbols}{ds_suffix}.npy"
    y3d_path = PROCESSED / f"_gru_wf_y3d_h{horizon}_ms{max_symbols}{ds_suffix}.npy"
    meta_path = PROCESSED / f"_gru_wf_meta_h{horizon}_ms{max_symbols}{ds_suffix}.json"
    if x3d_path.exists() and y3d_path.exists() and meta_path.exists():
        t0 = time.time()
        X3d = np.load(x3d_path)
        y3d = np.load(y3d_path)
        meta = json.load(open(meta_path, encoding="utf-8"))
        dates = np.array(meta["dates"], dtype="datetime64[ns]")
        symbols = np.array(meta["symbols"], dtype=object)
        logger.info("加载 X3d 缓存 %.1fs (shape %s)", time.time() - t0, X3d.shape)
        return X3d, y3d, dates, symbols
    meta = json.load(open(PROCESSED / f"dataset_h{horizon}_ev{ds_suffix}_meta.json", encoding="utf-8"))
    fn = meta["factor_names"]
    data = pd.read_parquet(PROCESSED / f"dataset_h{horizon}_ev{ds_suffix}.parquet")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    if label_clip > 0:
        data = data[data["y_excess"].abs() <= label_clip]
    X3d, y3d, dates, symbols = ds_mod.build_3d(data, fn)
    del data
    X3d = X3d.astype(np.float32); y3d = y3d.astype(np.float32)
    if max_symbols and 0 < max_symbols < len(symbols):
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(len(symbols), max_symbols, replace=False))
        symbols = symbols[keep]
        X3d = X3d[keep]; y3d = y3d[keep]
    np.save(x3d_path, X3d)
    np.save(y3d_path, y3d)
    json.dump({"dates": [str(d) for d in dates], "symbols": [str(s) for s in symbols]},
              open(meta_path, "w", encoding="utf-8"))
    logger.info("构建并缓存 X3d (shape %s) → %s", X3d.shape, x3d_path.name)
    return X3d, y3d, dates, symbols


def train_year(year, epochs, chunk, batch, stride, threads, label_clip, max_symbols, lr, hidden, seq, layers, ds_suffix="", tag=""):
    valid_end = f"{year-1}-12-31"
    train_end = f"{year-1-VALID_YEARS}-12-31"
    CKPT = paths.MODELS / "P1" / f"gru_wf{tag}_{year}_ckpt.pt"
    start = 0
    if os.path.exists(CKPT):
        try:
            ck = torch.load(CKPT, map_location="cpu", weights_only=False)
            start = int(ck.get("epoch", 0))
        except Exception:
            start = 0
    chunk_epochs = min(epochs, start + chunk)

    t0 = time.time()
    X3d, y3d, dates, symbols = get_X3d_cache(HORIZON, label_clip, max_symbols, ds_suffix)
    tr_m, va_m, te_m, sym_m = ds_mod.make_masks(dates, train_end, valid_end, symbols, None, label_horizon=HORIZON)
    # ⚠️ 去重红线：钳制测试掩码到「仅本年」，避免溢出到后续年份重复预测
    # （make_masks 返回只读数组，不能用 &= 原地操作，改用 & 生成新数组）
    te_m = te_m & (pd.to_datetime(dates).year == year)
    if te_m.sum() == 0:
        logger.error("年 %s 测试集为空（数据不足），跳过", year); return 1
    logger.info("[年 %s] train=%s valid=%s test=%s（train_end=%s valid_end=%s）",
                year, f"{tr_m.sum():,}", f"{va_m.sum():,}", f"{te_m.sum():,}", train_end, valid_end)
    train_ds = ds_mod.SequenceDataset(X3d, y3d, seq, sym_m, tr_m, date_stride=stride)
    valid_ds = ds_mod.SequenceDataset(X3d, y3d, seq, sym_m, va_m, date_stride=max(1, stride * 2))
    test_ds = ds_mod.SequenceDataset(X3d, y3d, seq, sym_m, te_m, date_stride=1)
    model = gru_attn.GRUAttention(X3d.shape[2], hidden=hidden, n_layers=layers)
    model, history = _trainer_mod.train_model(model, train_ds, valid_ds, epochs=chunk_epochs,
                                               batch_size=batch, lr=lr, patience=4, num_threads=threads,
                                               seed=42, checkpoint_path=str(CKPT))
    stopped = False
    if os.path.exists(CKPT):
        try:
            ck = torch.load(CKPT, map_location="cpu", weights_only=False); stopped = bool(ck.get("stopped"))
        except Exception:
            stopped = False
    complete = stopped
    if not complete and os.path.exists(CKPT):
        try:
            ck = torch.load(CKPT, map_location="cpu", weights_only=False)
            complete = int(ck.get("epoch", 0)) >= epochs
        except Exception:
            complete = False
    if not complete:
        logger.info("[年 %s] 训练至 ~%s/%s 轮（检查点 %s），下次续训；%.1fs",
                    year, chunk_epochs, epochs, CKPT.name, time.time() - t0)
        return 0
    pred = _trainer_mod.predict(model, test_ds, num_threads=threads)
    idx = test_ds.indices
    pdf = pd.DataFrame({
        "date": pd.to_datetime(dates[idx[:, 1]]),
        "symbol": symbols[idx[:, 0]],
        "pred": pred,
        "y_excess": y3d[idx[:, 0], idx[:, 1]],
        "year": year,
    })
    ic = metrics.summarize(pdf, "pred", "y_excess")
    out = PROCESSED / f"pred_gru_wf_{tag}_{year}.parquet"
    pdf.to_parquet(out, index=False)
    rep = {}
    if (PROCESSED / f"report_gru_wf_{tag}.json").exists():
        rep = json.load(open(PROCESSED / f"report_gru_wf_{tag}.json", encoding="utf-8"))
    rep[str(year)] = {"ic_mean": round(float(ic["ic_mean"]), 4),
                      "icir": round(float(ic.get("icir", float("nan"))), 4),
                      "ic_positive_rate": round(float(ic.get("ic_positive_rate", float("nan"))), 4),
                      "train_end": train_end, "valid_end": valid_end,
                      "n_train": int(tr_m.sum()), "n_test": int(te_m.sum()),
                      "stopped_epoch": int(ck.get("epoch", 0)) if os.path.exists(CKPT) else None}
    json.dump(rep, open(PROCESSED / f"report_gru_wf_{tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    logger.info("[年 %s] 完成 IC=%.4f ICIR=%.4f → %s；report 已更新；%.1fs",
                year, ic["ic_mean"], ic.get("icir"), out.name, time.time() - t0)
    try:
        os.remove(CKPT)
    except Exception:
        pass
    return 0


def main():
    ap = argparse.ArgumentParser(description="GRU EV walk-forward 单年训练（h20, 断点续训）")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label-clip", type=float, default=0.5)
    ap.add_argument("--max-symbols", type=int, default=800)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--hidden", type=int, default=HID)
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--ds-suffix", type=str, default="")
    ap.add_argument("--tag", type=str, default=TAG)
    args = ap.parse_args()
    train_year(args.year, args.epochs, args.chunk, args.batch, args.stride, args.threads,
               args.label_clip, args.max_symbols, args.lr, args.hidden, args.seq, args.layers,
               ds_suffix=args.ds_suffix, tag=args.tag)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("GRU walk-forward(h20) 异常")
        raise
