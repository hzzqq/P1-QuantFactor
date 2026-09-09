"""阶段 18：模型层迭代 × 10（严格 2026 全量 hold-out）。

承接阶段 14 结论：事件因子(牧羊人市场广度, 9 维) 已确认是**真实增益**
（全量 IC 0.074→0.084 +13.5%，净 -4.11%→-1.87% 亏损减半）。本阶段补齐其余
模型层杠杆，并纠正一个关键评价方法学错误。

⚠️ 核心纠正：前序脚本(12/14/17)回测用 rebalance_freq=15，与**标签 horizon=10 错配**——
信号 10 日窗口内的边缘在持有 15 日时衰减殆尽，把本可盈利的信号压成负。对齐到 rf=10 后，
全量 EV 信号由 -1.87% → +10.68%。这不是信号变强，是评价方法学纠错。

10 次迭代映射：
  M1  EV 全量 raw            bucket rf=10   (主结论)
  M2  EV 全量 raw            bucket rf=15   (错配对照, 复现旧 -1.87%)
  M3  EV + 横截面中性化       bucket rf=10   (对多空排名=no-op)
  M4  EV + 收缩 λ=0.7        bucket rf=10   (对多空排名=no-op)
  M5  EV + 中性化+收缩0.7      bucket rf=10   (no-op)
  M6  EV + 置信过滤(|pred|前60%) bucket rf=10 (更严选→真增益)
  M7  EV 全量 + 连续缓冲回测    continuous rf=10 (低换手)
  M8  EV 置信过滤 + 连续缓冲    continuous rf=10
  M9  TS-CV 集成 (2折 expanding, subset 表征 IC/ICIR/净)
  M10 成本敏感(训练时 L2-on-pred 收缩, subset 表征)
  REF-a/REF-b 基线 38维 pred_gru_tune_t16(subset) bucket rf=10/15 (证 rf 效应是方法学)

说明：M1–M8 全部源自阶段14已存全量 EV-t10 预测(206475行)，零重训 → 全量口径可比。
      M9/M10 为训练层杠杆，subset 快速表征信号质量，全量确认留作后续。

产物：report_model_layer_full.csv + model_layer_report.html

用法：
  python scripts/18_model_layer_iterate.py --smoke        # subset 冒烟(约8min)
  python scripts/18_model_layer_iterate.py                # 全量+subset（后台）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
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
from src.eval import metrics
from src.models import dataset as ds_mod
from src.models import gru_attn
from src.models import trainer
from src.backtest import run_backtest, run_backtest_continuous, DEFAULT_COST

logger = get_logger("P1.model_layer_iterate")
CONFIG_PATH = PROJ / "config" / "default.yaml"
PROCESSED = paths.DATA / "P1" / "processed"
MODELS_DIR = paths.MODELS / "P1"

SEQ, HID, LR, LAYERS = 40, 128, 1e-3, 2   # t10 配置


# ───────────────────────────── 数据 / 训练 ─────────────────────────────
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
        model, history = trainer.train_model(
            model, train_ds, valid_ds, epochs=epochs, batch_size=batch, lr=lr,
            patience=4, num_threads=threads, seed=seed)
    pred = trainer.predict(model, test_ds, num_threads=threads)
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
    trainer.set_cpu_threads(threads)
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


# ───────────────────────────── 后处理 / 回测 ─────────────────────────────
def post_variants(pred_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    out["raw"] = pred_df.copy()
    g = pred_df.groupby("date")["pred"]
    neu = (pred_df["pred"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    neu = neu.fillna(0.0)
    out["neu"] = pred_df.assign(pred=neu)
    out["shr"] = pred_df.assign(pred=pred_df["pred"] * 0.7)
    out["neu_shr"] = pred_df.assign(pred=neu * 0.7)
    mag_pct = g.transform(lambda s: s.abs().rank(pct=True))
    keep = mag_pct >= 0.4
    cf = pred_df.copy(); cf.loc[~keep, "pred"] = np.nan
    out["conf"] = cf
    return out


def bt_bucket(pred_df, panel, horizon, top_pct=0.1, rf=10):
    return run_backtest(pred_df[["date", "symbol", "pred"]], panel, horizon=horizon,
                        top_pct=top_pct, rebalance_freq=rf, cost=DEFAULT_COST,
                        mode="long_short").ls_stats


def bt_cont(pred_df, panel, horizon, top_pct=0.1, buffer=0.02, rf=10):
    return run_backtest_continuous(pred_df[["date", "symbol", "pred"]], panel,
                                   horizon=horizon, top_pct=top_pct,
                                   rebalance_freq=rf, cost=DEFAULT_COST,
                                   mode="long_short", buffer=buffer).ls_stats


def ic_of(pred_df, col="pred"):
    return metrics.summarize(pred_df, col, "y_excess")


# ───────────────────────────── 主流程 ─────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="P1 模型层迭代 ×10")
    ap.add_argument("--smoke", action="store_true", help="subset 冒烟(约8min)")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label-clip", type=float, default=0.5)
    ap.add_argument("--top-pct", type=float, default=0.1)
    ap.add_argument("--rf", type=int, default=10, help="rebalance_freq，默认=标签horizon(10)")
    args = ap.parse_args()

    cfg = load_config(CONFIG_PATH)
    horizon = args.horizon or cfg.get_path("label.horizon", 10)
    seed = cfg.get_path("seed", 42)
    rf = args.rf

    max_sym = 800 if args.smoke else args.max_symbols
    t0 = time.time()
    logger.info("加载 EV 数据集 h=%s (max_symbols=%s) ...", horizon, max_sym or "全量")
    data, X3d, y3d, dates, symbols, subset, fn = load_ev(
        horizon, args.label_clip, max_sym)

    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    rows = []

    # ── 全量块：复用阶段14已存的全量 EV-t10 预测（206475 行 = 全量）──
    FULL_EV = PROCESSED / "pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet"
    if FULL_EV.exists():
        logger.info("复用阶段14全量 EV 预测: %s", FULL_EV.name)
        full_pdf = pd.read_parquet(FULL_EV)
    else:
        logger.info("=== 全量重训 EV-t10 (anchor) ===")
        full_pdf, model, _ = make_test_pred(
            X3d, y3d, dates, symbols, subset, "2024-12-31", "2025-12-31",
            "EV-full-t10", args.epochs, args.batch, args.stride, args.threads, seed=seed)
        full_pdf.to_parquet(FULL_EV, index=False)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(),
                   MODELS_DIR / f"gru_ev_full_t10_seq{SEQ}_h{HID}_lr{LR}_h{horizon}.pt")

    logger.info("全量预测 %s 行；开始 8 个后处理/引擎变体 (rf=%s) ...",
                f"{len(full_pdf):,}", rf)
    variants = post_variants(full_pdf)
    for name, tag in [("raw", "M1"), ("neu", "M3"), ("shr", "M4"),
                      ("neu_shr", "M5"), ("conf", "M6")]:
        s = ic_of(variants[name]); bt = bt_bucket(variants[name], panel, horizon,
                                                  top_pct=args.top_pct, rf=rf)
        rows.append(dict(iter=tag, universe="full", variant=name, engine="bucket", rf=rf,
                        ic=s["ic_mean"], icir=s["icir"], pos=s["ic_positive_rate"],
                        net=bt.get("total_return"), sharpe=bt.get("sharpe"),
                        mdd=bt.get("max_drawdown"), win=bt.get("win_rate"),
                        turnover=bt.get("avg_turnover")))
        logger.info("[%s] %s bucket rf=%s → IC=%.4f net=%.2f%% 夏普=%.2f",
                    tag, name, rf, s["ic_mean"], (bt.get("total_return") or 0) * 100,
                    bt.get("sharpe"))
    # M2：raw 在 rf=15（错配，复现阶段14的 -1.87%）作为方法学对照
    s = ic_of(variants["raw"]); bt15 = bt_bucket(variants["raw"], panel, horizon,
                                                top_pct=args.top_pct, rf=15)
    rows.append(dict(iter="M2", universe="full", variant="raw", engine="bucket", rf=15,
                    ic=s["ic_mean"], icir=s["icir"], pos=s["ic_positive_rate"],
                    net=bt15.get("total_return"), sharpe=bt15.get("sharpe"),
                    mdd=bt15.get("max_drawdown"), win=bt15.get("win_rate"),
                    turnover=bt15.get("avg_turnover")))
    logger.info("[M2] raw bucket rf=15(错配) → net=%.2f%% (复现阶段14)",
                (bt15.get("total_return") or 0) * 100)
    for name, tag in [("raw", "M7"), ("conf", "M8")]:
        s = ic_of(variants[name]); bt = bt_cont(variants[name], panel, horizon,
                                                top_pct=args.top_pct)
        rows.append(dict(iter=tag, universe="full", variant=name, engine="continuous", rf=rf,
                        ic=s["ic_mean"], icir=s["icir"], pos=s["ic_positive_rate"],
                        net=bt.get("total_return"), sharpe=bt.get("sharpe"),
                        mdd=bt.get("max_drawdown"), win=bt.get("win_rate"),
                        turnover=bt.get("avg_turnover")))
        logger.info("[%s] %s continuous rf=%s → net=%.2f%% 夏普=%.2f 换手=%.1f%%",
                    tag, name, rf, (bt.get("total_return") or 0) * 100,
                    bt.get("sharpe"), (bt.get("avg_turnover") or 0) * 100)

    # ── 基线参照（subset 38维 pred_gru_tune_t16）：证明 rf 效应是方法学而非 EV 专属 ──
    BASE = PROCESSED / "pred_gru_tune_t16_seq30_h128_lr1e-3_h10.parquet"
    if BASE.exists():
        base = pd.read_parquet(BASE)[["date", "symbol", "pred"]]
        for brf, btag in [(rf, "REF-a"), (15, "REF-b")]:
            # 基线 pred 文件无 y_excess 列 → 仅回测净/夏普，IC 留空
            bbt = bt_bucket(base, panel, horizon, top_pct=args.top_pct, rf=brf)
            rows.append(dict(iter=btag, universe="subset", variant="baseline_t16(38维)",
                            engine="bucket", rf=brf, ic=float("nan"), icir=float("nan"),
                            pos=float("nan"), net=bbt.get("total_return"),
                            sharpe=bbt.get("sharpe"), mdd=bbt.get("max_drawdown"),
                            win=bbt.get("win_rate"), turnover=bbt.get("avg_turnover")))
            logger.info("[%s] 基线 subset bucket rf=%s → net=%.2f%%",
                        btag, brf, (bbt.get("total_return") or 0) * 100)

    # ── subset 训练层表征：TS-CV 集成 (M9) / 成本敏感 (M10) ──
    # 训练层杠杆在固定 800 子集快速表征（全量确认留作后续），与全量 EV 复用解耦
    tscv_subset = None
    if len(symbols) > 800:
        rng = np.random.default_rng(42)
        tscv_subset = symbols[np.sort(rng.choice(len(symbols), 800, replace=False))]
    if tscv_subset is not None:
        logger.info("=== subset 训练层表征 (TS-CV / 成本敏感) ===")
        pA, _, _ = make_test_pred(X3d, y3d, dates, symbols, tscv_subset, "2023-12-31",
                                  "2024-12-31", "tscv-A", args.epochs, args.batch,
                                  args.stride, args.threads, seed=seed)
        pB, _, _ = make_test_pred(X3d, y3d, dates, symbols, tscv_subset, "2024-12-31",
                                  "2025-12-31", "tscv-B", args.epochs, args.batch,
                                  args.stride, args.threads, seed=seed)
        m = pA.merge(pB, on=["date", "symbol", "y_excess", "year"], suffixes=("_a", "_b"))
        tscv = m.assign(pred=(m["pred_a"] + m["pred_b"]) / 2.0)[
            ["date", "symbol", "pred", "y_excess", "year"]]
        s = ic_of(tscv); bt = bt_bucket(tscv, panel, horizon, top_pct=args.top_pct, rf=rf)
        rows.append(dict(iter="M9", universe="subset", variant="tscv_ensemble",
                        engine="bucket", rf=rf, ic=s["ic_mean"], icir=s["icir"],
                        pos=s["ic_positive_rate"], net=bt.get("total_return"),
                        sharpe=bt.get("sharpe"), mdd=bt.get("max_drawdown"),
                        win=bt.get("win_rate"), turnover=bt.get("avg_turnover")))
        logger.info("[M9] TS-CV 集成 subset → IC=%.4f ICIR=%.4f net=%.2f%%",
                    s["ic_mean"], s["icir"], (bt.get("total_return") or 0) * 100)
        pC, _, _ = make_test_pred(X3d, y3d, dates, symbols, tscv_subset, "2024-12-31",
                                  "2025-12-31", "cost", args.epochs, args.batch,
                                  args.stride, args.threads, cost_lam=0.02, seed=seed)
        s = ic_of(pC); bt = bt_bucket(pC, panel, horizon, top_pct=args.top_pct, rf=rf)
        rows.append(dict(iter="M10", universe="subset", variant="cost_sensitive",
                        engine="bucket", rf=rf, ic=s["ic_mean"], icir=s["icir"],
                        pos=s["ic_positive_rate"], net=bt.get("total_return"),
                        sharpe=bt.get("sharpe"), mdd=bt.get("max_drawdown"),
                        win=bt.get("win_rate"), turnover=bt.get("avg_turnover")))
        logger.info("[M10] 成本敏感 subset → IC=%.4f ICIR=%.4f net=%.2f%%",
                    s["ic_mean"], s["icir"], (bt.get("total_return") or 0) * 100)

    df = pd.DataFrame(rows)
    out_csv = PROCESSED / "report_model_layer_full.csv"
    df.to_csv(out_csv, index=False, float_format="%.6f")
    _write_html(df, rf)
    print("\n" + "=" * 86)
    print(f"  模型层迭代 ×10（严格 2026 hold-out, rf={rf}）")
    print("=" * 86)
    print(df.to_string(index=False,
          formatters={c: (lambda v: f"{v:+.4f}" if isinstance(v, float) else v)
                      for c in ["ic", "icir", "pos", "net", "sharpe", "mdd", "win", "turnover"]}))
    print("=" * 86)
    print(f"  总耗时 {time.time()-t0:.1f}s | {out_csv}")
    return 0


def _read_csv(name):
    p = PROCESSED / name
    return pd.read_csv(p) if p.exists() else None


def _write_html(df: pd.DataFrame, rf: int) -> None:
    evfull = _read_csv("report_tune_ev_full.csv")
    reg = _read_csv("report_regime_holdout_2026.csv")

    def cls(v):
        return "pos" if (v or 0) > 0 else "neg"

    truth = ""
    if evfull is not None and reg is not None:
        e = evfull.iloc[0]
        b = reg[reg.strategy.str.contains("纯GRU")].iloc[0]
        m1 = df[df.iter == "M1"]
        m1net = m1["net"].iloc[0] if not m1.empty else e["net_2026"]
        m1ic = m1["ic"].iloc[0] if not m1.empty else e["ic_2026"]
        truth = (f"<tr><td>基线 38维 GRU (rf=15,全量)</td>"
                 f"<td class='{cls(b['net'])}'>{b['net']*100:+.2f}%</td>"
                 f"<td>{b['ic']:.4f}</td></tr>"
                 f"<tr><td>EV 47维(牧羊人) (rf=15,全量)</td>"
                 f"<td class='{cls(e['net_2026'])}'>{e['net_2026']*100:+.2f}%</td>"
                 f"<td>{e['ic_2026']:.4f}</td></tr>"
                 f"<tr style='background:#1e3a2f'><td><b>EV 47维 (rf={rf},全量)</b></td>"
                 f"<td><b class='{cls(m1net)}'>{m1net*100:+.2f}%</b></td>"
                 f"<td><b>{m1ic:.4f}</b></td></tr>")

    res_rows = ""
    for r in df.itertuples():
        tv = r.turnover if pd.notna(r.turnover) else 0.0
        res_rows += (f"<tr><td>{r.iter}</td><td>{r.universe}</td><td>{r.variant}</td>"
                     f"<td>{r.engine}</td><td>rf={r.rf}</td>"
                     f"<td class='{cls(r.ic)}'>{r.ic:+.4f}</td>"
                     f"<td class='{cls(r.net)}'>{r.net*100:+.2f}%</td>"
                     f"<td class='{cls(r.sharpe)}'>{r.sharpe:+.3f}</td>"
                     f"<td>{r.mdd*100:.2f}%</td>"
                     f"<td>{tv*100:.1f}%</td></tr>")

    html = f"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<title>P1 模型层迭代 ×10 报告</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0f1419;color:#e6e6e6;margin:0;padding:32px}}
h1{{color:#7fd1a0}} h2{{color:#9ec5fe;border-bottom:1px solid #2a3441;padding-bottom:8px}}
table{{border-collapse:collapse;width:100%;margin:16px 0;font-size:13px}}
th,td{{border:1px solid #2a3441;padding:7px 9px;text-align:right}}
th{{background:#1a2430;color:#9ec5fe}} td:first-child,th:first-child{{text-align:left}}
.pos{{color:#ff6b6b}} .neg{{color:#51cf66}}
.card{{background:#161c24;border:1px solid #2a3441;border-radius:10px;padding:18px;margin:14px 0}}
.note{{color:#a0aec0;font-size:13px;line-height:1.7}}</style></head>
<body><h1>模型层迭代 ×10 · 严格 2026 全量 hold-out</h1>
<div class=card><div class=note><b>核心纠正：</b>前序脚本（12/14/17）回测用
<code>rebalance_freq=15</code>，与<b>标签 horizon=10 错配</b>——信号 10 日窗口内的边缘在持有 15 日时
衰减殆尽，把本可盈利的信号压成负。对齐到 <code>rf={rf}</code> 后，全量 EV 信号由
<b>-1.87% → +{df[df.iter=='M1']['net'].iloc[0]*100:+.2f}%</b>。这不是信号变强，是<b>评价方法学纠错</b>。</div></div>
<h2>① 对账真值表（事件因子增益 + 频率对齐）</h2>
<table><tr><th>信号</th><th>2026净</th><th>IC</th></tr>{truth}</table>
<h2>② 模型层 10 迭代（rf={rf}）</h2>
<table><tr><th>迭代</th><th>宇宙</th><th>变体</th><th>引擎</th><th>频率</th>
<th>IC</th><th>净</th><th>夏普</th><th>回撤</th><th>换手</th></tr>{res_rows}</table>
<div class=card><div class=note><b>读法：</b>
M1(raw,rf={rf}) 是主结论；M2(rf=15) 复现旧负值证明频率效应。
<b>中性化(M3)/收缩(M4/M5)对多空排名回测是 no-op</b>（单调变换不改变排名，IC 与净不变），
其价值在多头或置信过滤的选择层。真正再增益的廉价杠杆是 <b>置信过滤(M6)</b>
（仅留 |pred| 前 60% → 等价于更严选 top/bottom 6%，降噪降成本）与
<b>连续缓冲(M7/M8)</b>（更低换手、更贴近实盘）。M9/M10 为训练层表征（subset），全量确认留作后续。</div></div>
</body></html>"""
    out = PROCESSED / "model_layer_report.html"
    out.write_text(html, encoding="utf-8")
    logger.info("HTML 报告: %s (%s bytes)", out, out.stat().st_size)


if __name__ == "__main__":
    raise SystemExit(main())
