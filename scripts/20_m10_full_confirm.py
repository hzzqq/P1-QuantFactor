"""阶段 20：M10 成本敏感训练 · 全量 1427 确认（接阶段 19 主线 A）。

复用阶段 18 的 _train_cost_sensitive（Huber + λ·E[pred²] 收缩，cost_lam=0.02），在
EV 47 维全量数据集上用 t10 锚定窗口（train≤2024-12-31 / valid 2025 / test 2026）重训，
回测 rf=10（与阶段 18 M1 全量 EV raw rf=10 净 +10.68% 对齐口径），并跑 rf=15 验证
方法学。同时复用阶段 14 全量 EV-t10 预测作 M1 对照。

输出：
  - data/P1/processed/report_m10_full_confirm.csv（M1 vs M10 × rf=10/15）
  - models/P1/gru_m10_cost_full.pt（成本敏感全量模型，供 StockSignal 接入）
  - data/P1/processed/pred_gru_m10_cost_full_h10.parquet（全量 2026 预测，供导出信号）

用法：
    python scripts/20_m10_full_confirm.py            # 全量（后台 ~30min）
    python scripts/20_m10_full_confirm.py --smoke    # 800 子集冒烟
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

ROOT = Path(__file__).resolve().parents[2]      # E:/project/sj
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths
from shared.config import load_config
from shared.logging_utils import get_logger
from src.models import dataset as ds_mod, gru_attn
from src.models import trainer as _trainer_mod
from src.backtest import run_backtest, DEFAULT_COST
from src.eval import metrics

PROCESSED = paths.DATA / "P1" / "processed"
CONFIG_PATH = PROJ / "config" / "default.yaml"
SEQ, HID, LR, LAYERS = 40, 128, 1e-3, 2       # t10 配置（与阶段 18 一致）
COST_LAM = 0.02                                 # M10 成本敏感收缩系数
TRAIN_END, VALID_END = "2024-12-31", "2025-12-31"
logger = get_logger("P1.m10_full_confirm")


def load_ev(horizon, label_clip, max_symbols):
    meta = json.load(open(PROCESSED / f"dataset_h{horizon}_ev_meta.json", encoding="utf-8"))
    fn = meta["factor_names"]
    data = pd.read_parquet(PROCESSED / f"dataset_h{horizon}_ev.parquet")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    if label_clip > 0:
        data = data[data["y_excess"].abs() <= label_clip]
    X3d, y3d, dates, symbols = ds_mod.build_3d(data, fn)
    subset = None
    if max_symbols and 0 < max_symbols < len(symbols):
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(len(symbols), max_symbols, replace=False))
        subset = symbols[keep]
    return data, X3d, y3d, dates, symbols, subset, fn


def make_test_pred(X3d, y3d, dates, symbols, subset, train_end, valid_end,
                   tag, epochs, batch, stride, threads, lr=LR, hidden=HID,
                   seq=SEQ, layers=LAYERS, cost_lam=0.0, seed=42):
    tr_m, va_m, te_m, sym_m = ds_mod.make_masks(
        dates, train_end, valid_end, symbols, subset)
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


def _train_cost_sensitive(model, train_ds, valid_ds, epochs, batch, lr, threads,
                          cost_lam, seed):
    """成本敏感训练：Huber + λ·E[pred²]（训练时收缩，抑制极端低确信预测→降换手）。"""
    torch.manual_seed(seed); np.random.seed(seed)
    _trainer_mod.set_cpu_threads(threads)
    g = torch.Generator(); g.manual_seed(seed)
    tl = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True,
                                     num_workers=0, drop_last=True, generator=g)
    vl = torch.utils.data.DataLoader(valid_ds, batch_size=batch * 2, shuffle=False,
                                     num_workers=0) if valid_ds is not None else None
    crit = torch.nn.HuberLoss(delta=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)
    hist = []; best, best_st, wait = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); tot, seen = 0.0, 0
        for xb, yb in tl:
            opt.zero_grad(set_to_none=True)
            p = model(xb)
            loss = crit(p, yb) + cost_lam * p.pow(2).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += float(loss.item()) * len(yb); seen += len(yb)
        tl_ = tot / max(1, seen)
        vl_ = None
        if vl is not None:
            model.eval(); vt, vn = 0.0, 0
            with torch.no_grad():
                for xb, yb in vl:
                    vt += float((crit(model(xb), yb)).item()) * len(yb); vn += len(yb)
            vl_ = vt / max(1, vn); sched.step(vl_)
        score = vl_ if vl_ is not None else tl_
        hist.append({"epoch": ep, "train_loss": tl_, "valid_loss": vl_, "sec": round(time.time() - t0, 1)})
        logger.info("  [cost] epoch %s | train %.6f | valid %s", ep, tl_, vl_)
        if score < best - 1e-8:
            best, wait = score, 0
            best_st = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 4:
                logger.info("  [cost] 早停于第 %s 轮", ep); break
    if best_st is not None:
        model.load_state_dict(best_st)
    return model, hist


def bt_bucket(pred_df, panel, horizon, top_pct=0.1, rf=10):
    return run_backtest(pred_df[["date", "symbol", "pred"]], panel, horizon=horizon,
                        top_pct=top_pct, rebalance_freq=rf, cost=DEFAULT_COST,
                        mode="long_short", bootstrap=True).ls_stats


def main() -> int:
    ap = argparse.ArgumentParser(description="M10 成本敏感训练 · 全量确认")
    ap.add_argument("--smoke", action="store_true", help="800 子集冒烟")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label-clip", type=float, default=0.5)
    ap.add_argument("--top-pct", type=float, default=0.1)
    args = ap.parse_args()

    max_sym = 800 if args.smoke else 0
    t0 = time.time()
    logger.info("加载 EV 47 维全量数据集 (max_symbols=%s) ...", max_sym or "全量")
    data, X3d, y3d, dates, symbols, subset, fn = load_ev(
        args.horizon, args.label_clip, max_sym)
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    rows = []

    # ── M10：成本敏感训练全量 ──
    logger.info("=== M10 成本敏感训练 (cost_lam=%s) 全量 ===", COST_LAM)
    m10_pdf, m10_model, m10_ep = make_test_pred(
        X3d, y3d, dates, symbols, subset, TRAIN_END, VALID_END,
        "M10", args.epochs, args.batch, args.stride, args.threads,
        lr=LR, hidden=HID, seq=SEQ, layers=LAYERS, cost_lam=COST_LAM)
    m10_ic = metrics.summarize(m10_pdf, "pred", "y_excess")

    # 立刻落盘模型 + 预测（防长训练后回测阶段被外部信号/OOM 杀掉丢产物）
    if not args.smoke:
        torch.save({"state_dict": m10_model.state_dict(),
                    "in_dim": X3d.shape[2], "hidden": HID, "n_layers": LAYERS,
                    "seq": SEQ, "cost_lam": COST_LAM,
                    "train_end": TRAIN_END, "valid_end": VALID_END},
                   paths.MODELS / "P1" / "gru_m10_cost_full.pt")
        m10_pdf.to_parquet(PROCESSED / "pred_gru_m10_cost_full_h10.parquet", index=False)
        logger.info("模型/预测已保存（回测前先落盘）: gru_m10_cost_full.pt / pred_gru_m10_cost_full_h10.parquet")

    # 释放大三维数组，降低回测阶段内存压力（防 OOM 被杀）
    del X3d, y3d, dates, symbols, data, subset
    import gc; gc.collect()

    for rf in (10, 15):
        st = bt_bucket(m10_pdf, panel, args.horizon, args.top_pct, rf=rf)
        rows.append(dict(iter="M10", universe="full(1427)" if not max_sym else "subset(800)",
                         variant="cost_sensitive", rf=rf,
                         ic=m10_ic.get("ic"), icir=m10_ic.get("icir"),
                         net=st["total_return"], sharpe=st["sharpe"],
                         mdd=st["max_drawdown"], buckets=st["n_buckets"]))
        logger.info("M10 rf=%d: IC %.4f ICIR %.4f net %.2f%% sharpe %.2f mdd %.2f%% buckets %d",
                    rf, m10_ic.get("ic"), m10_ic.get("icir"), st["total_return"]*100,
                    st["sharpe"], st["max_drawdown"]*100, st["n_buckets"])

    # ── M1 对照：复用阶段 14 全量 EV-t10 预测（raw rf=10 → +10.68%）──
    FULL_EV = PROCESSED / "pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet"
    if FULL_EV.exists():
        m1_pdf = pd.read_parquet(FULL_EV)
        m1_ic = metrics.summarize(m1_pdf, "pred", "y_excess")
        for rf in (10, 15):
            st = bt_bucket(m1_pdf, panel, args.horizon, args.top_pct, rf=rf)
            rows.append(dict(iter="M1", universe="full(1427)", variant="raw", rf=rf,
                             ic=m1_ic.get("ic_mean"), icir=m1_ic.get("icir"),
                             net=st["total_return"], sharpe=st["sharpe"],
                             mdd=st["max_drawdown"], buckets=st["n_buckets"]))
            logger.info("M1(raw) rf=%d: IC %.4f ICIR %.4f net %.2f%% sharpe %.2f mdd %.2f%% buckets %d",
                        rf, m1_ic.get("ic"), m1_ic.get("icir"), st["total_return"]*100,
                        st["sharpe"], st["max_drawdown"]*100, st["n_buckets"])
    else:
        logger.warning("未找到阶段14全量 EV 预测 %s，跳过 M1 对照", FULL_EV.name)

    out = PROCESSED / "report_m10_full_confirm.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("=> %s", out)
    logger.info("总耗时 %.1f 秒", time.time() - t0)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("M10 全量确认异常退出（若回测阶段被杀，模型/预测应已在回测前落盘，可跑 scripts/20b_m10_report.py 恢复报告）")
        raise
