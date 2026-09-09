"""阶段 14：接入 StockSignal 真实事件/情绪因子（牧羊人市场广度）并重跑 20 迭代。

    20 次迭代 = t01~t20 超参网格：t01~t10 与 10_iterate_tune.py 一致（对基线可比），
    t11~t20 围绕最优 t10(seq40/h128) 扩张 lr/seq/hidden 并引入 layers 深度维度。

背景纠正：
    上一轮评估认为「P4-TextPulse 合成 demo 无真实文本管线，事件因子不可接入」。
    但老板指出事件因子在 StockSignal 里真实存在。核查结果：
      - 逐股事件数据 events.csv / news.db 实际为空（4 行 stub / 0 行）→ 不能用；
      - 真实可用的是 `E:/project/ks/StockSignal/data/shepherd_history.csv`
        （牧羊人 17/9 项市场广度情绪，2007-01-05 ~ 2026-08-21，4771 行，
        与 P1 面板日期重叠 99.8%）。

设计：
    牧羊人指标是「市场级」（每日一行，非逐股）。正规做法不是做逐股横截面 z
    （那样所有股票同一值、z 后全 0），而是把它作为「市场 regime / 上下文」特征，
    按日广播到全部股票 → 成为 GRU 的额外输入通道。每个指标先全局 z-score
    （与 P1 既有的 38 个标准化因子同量级 mean≈0/std≈1），前向填充缺失日，再合并。

产物：
    - dataset_h10_ev.parquet / _ev_meta.json ：47 维（38 + 9 牧羊人）
    - pred_gru_ev_{tag}_h10.parquet                ：每组的 2026 严格 hold-out 预测
    - report_tune_grid_ev.csv                      ：10 组 2026 IC/ICIR/净/夏普

与基线（report_tune_grid.csv，38 维）同口径严格对比，判定事件因子是否增益。

用法：
    python scripts/14_event_factor_iterate.py --build-only        # 仅构建 EV 数据集
    python scripts/14_event_factor_iterate.py                      # 构建 + 跑全网格(后台)
    python scripts/14_event_factor_iterate.py --smoke             # 仅 t02 冒烟验证
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
from src.backtest import run_backtest, DEFAULT_COST

logger = get_logger("P1.event_factor_iterate")
CONFIG_PATH = PROJ / "config" / "default.yaml"
PROCESSED = paths.DATA / "P1" / "processed"
MODELS_DIR = paths.MODELS / "P1"

SHEPHERD_CSV = Path(r"E:/project/ks/StockSignal/data/shepherd_history.csv")
# 9 个牧羊人市场广度指标（其余列如 flat_count 已含，connect_hl/zt_* 近期才有真实值）
SHEEP_COLS = ["up_count", "down_count", "flat_count", "limit_up", "limit_down",
              "red_ratio", "connect_hl", "zt_fail_ratio", "zt_prev_ret"]

# t01~t10 与 10_iterate_tune.py 完全一致（保证与 38 维基线横向可比）；
# t11~t20 为扩张组：围绕最优 t10(seq40/h128) 加密 lr、加长 seq(50)、加大 hidden(160)，
# 并首次引入 layers 深度维度(1/2/3)。共 20 组 = 「迭代 20 次」。
GRID = [
    dict(tag="t01_seq20_h64_lr1e-3", seq_len=20, hidden=64, lr=1e-3, layers=2),
    dict(tag="t02_seq20_h64_lr3e-3", seq_len=20, hidden=64, lr=3e-3, layers=2),
    dict(tag="t03_seq20_h64_lr5e-4", seq_len=20, hidden=64, lr=5e-4, layers=2),
    dict(tag="t04_seq30_h64_lr1e-3", seq_len=30, hidden=64, lr=1e-3, layers=2),
    dict(tag="t05_seq40_h64_lr1e-3", seq_len=40, hidden=64, lr=1e-3, layers=2),
    dict(tag="t06_seq20_h96_lr1e-3", seq_len=20, hidden=96, lr=1e-3, layers=2),
    dict(tag="t07_seq20_h128_lr1e-3", seq_len=20, hidden=128, lr=1e-3, layers=2),
    dict(tag="t08_seq30_h96_lr1e-3", seq_len=30, hidden=96, lr=1e-3, layers=2),
    dict(tag="t09_seq30_h96_lr3e-3", seq_len=30, hidden=96, lr=3e-3, layers=2),
    dict(tag="t10_seq40_h128_lr1e-3", seq_len=40, hidden=128, lr=1e-3, layers=2),
    # ── t11~t20：围绕最优 t10(seq40/h128) 与 t01/t02 扩张 + 引入 layers 深度维度 ──
    dict(tag="t11_seq40_h128_lr3e-3", seq_len=40, hidden=128, lr=3e-3, layers=2),
    dict(tag="t12_seq40_h128_lr5e-4", seq_len=40, hidden=128, lr=5e-4, layers=2),
    dict(tag="t13_seq50_h128_lr1e-3", seq_len=50, hidden=128, lr=1e-3, layers=2),
    dict(tag="t14_seq40_h160_lr1e-3", seq_len=40, hidden=160, lr=1e-3, layers=2),
    dict(tag="t15_seq30_h128_lr1e-3", seq_len=30, hidden=128, lr=1e-3, layers=2),
    dict(tag="t16_seq40_h96_lr1e-3", seq_len=40, hidden=96, lr=1e-3, layers=2),
    dict(tag="t17_seq20_h64_lr2e-3", seq_len=20, hidden=64, lr=2e-3, layers=2),
    dict(tag="t18_seq40_h128_lr1e-3_L1", seq_len=40, hidden=128, lr=1e-3, layers=1),
    dict(tag="t19_seq40_h128_lr1e-3_L3", seq_len=40, hidden=128, lr=1e-3, layers=3),
    dict(tag="t20_seq50_h160_lr1e-3", seq_len=50, hidden=160, lr=1e-3, layers=2),
]


# ───────────────────────────── 阶段 A：构建 EV 数据集 ─────────────────────────────
def build_event_dataset(horizon: int = 10) -> list[str]:
    """把牧羊人市场情绪并入 P1 数据集，产出 47 维 EV 数据集。"""
    base = pd.read_parquet(PROCESSED / f"dataset_h{horizon}.parquet")
    meta = json.load(open(PROCESSED / f"dataset_h{horizon}_meta.json",
                          encoding="utf-8"))
    fn = meta["factor_names"]
    logger.info("基线数据集 %s 行 | %s 因子", f"{len(base):,}", len(fn))

    sheep = pd.read_csv(SHEPHERD_CSV)
    sheep["date"] = pd.to_datetime(sheep["date"])
    sheep = sheep[["date"] + SHEEP_COLS].sort_values("date").drop_duplicates("date")
    # 1) 时间维前向填充（早期 connect_hl/zt_* 多为 NaN）
    sheep[SHEEP_COLS] = sheep[SHEEP_COLS].ffill()
    # 2) 全局 z-score（与既有 38 因子同量级 mean≈0/std≈1）
    for c in SHEEP_COLS:
        m, s = sheep[c].mean(), sheep[c].std()
        sheep[c] = (sheep[c] - m) / s if (s and s == s and s > 0) else 0.0
    sheep[SHEEP_COLS] = sheep[SHEEP_COLS].fillna(0.0)

    # 3) 按日广播合并（base 一行 = 一只股票某天；sheep 一天一行 → 自动广播）
    out = base.merge(sheep, on="date", how="left")
    for c in SHEEP_COLS:
        out[c] = out[c].fillna(0.0)

    new_fn = fn + SHEEP_COLS
    out.to_parquet(PROCESSED / f"dataset_h{horizon}_ev.parquet", index=False)
    json.dump(
        {"factor_names": new_fn, "horizon": horizon, "n_base": len(fn),
         "added": SHEEP_COLS,
         "source": "E:/project/ks/StockSignal/data/shepherd_history.csv",
         "note": "事件/情绪因子：牧羊人市场广度(9项)，全局z-score+ffill+按日广播；"
                 "逐股 events.csv/news.db 为空故用市场级regime特征"},
        open(PROCESSED / f"dataset_h{horizon}_ev_meta.json", "w",
             encoding="utf-8"), ensure_ascii=False, indent=2)

    nan_fraction = float(out[SHEEP_COLS].isna().mean().mean())
    logger.info("EV 数据集 -> %s 行 | %s 维(38+9) | 新增列 NaN 占比 %.4f",
                f"{len(out):,}", len(new_fn), nan_fraction)
    return new_fn


# ───────────────────────────── 阶段 B：10 迭代网格 ─────────────────────────────
def load_ev_dataset(horizon: int, label_clip: float):
    meta = json.load(open(PROCESSED / f"dataset_h{horizon}_ev_meta.json",
                          encoding="utf-8"))
    factor_names = meta["factor_names"]
    data = pd.read_parquet(PROCESSED / f"dataset_h{horizon}_ev.parquet")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    if label_clip > 0:
        data = data[data["y_excess"].abs() <= label_clip]
    X3d, y3d, dates, symbols = ds_mod.build_3d(data, factor_names)
    return data, X3d, y3d, dates, symbols, factor_names


def pick_subset(symbols, max_symbols, seed):
    if max_symbols <= 0 or len(symbols) <= max_symbols:
        return None
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(symbols), max_symbols, replace=False))
    return symbols[keep]


def run_one(cfg, X3d, y3d, dates, symbols, symbol_subset, args, seed):
    train_end, valid_end = args.train_end, args.valid_end
    tr_m, va_m, te_m, sym_m = ds_mod.make_masks(
        dates, train_end, valid_end, symbols, symbol_subset,
        label_horizon=args.horizon)
    if tr_m.sum() == 0 or te_m.sum() == 0:
        raise RuntimeError(f"空区间: train={tr_m.sum()} test={te_m.sum()}")

    train_ds = ds_mod.SequenceDataset(X3d, y3d, cfg["seq_len"], sym_m, tr_m,
                                      date_stride=args.stride)
    valid_ds = ds_mod.SequenceDataset(X3d, y3d, cfg["seq_len"], sym_m, va_m,
                                      date_stride=max(1, args.stride * 2))
    test_ds = ds_mod.SequenceDataset(X3d, y3d, cfg["seq_len"], sym_m, te_m,
                                     date_stride=1)
    if len(train_ds) < 1000 or len(test_ds) < 50:
        raise RuntimeError(f"样本不足 train={len(train_ds)} test={len(test_ds)}")

    logger.info("[%s] train=%s test=%s | seq=%s h=%s lr=%s | in_dim=%s",
                cfg["tag"], f"{len(train_ds):,}", f"{len(test_ds):,}",
                cfg["seq_len"], cfg["hidden"], cfg["lr"], X3d.shape[2])

    model = gru_attn.GRUAttention(X3d.shape[2], hidden=cfg["hidden"],
                                  n_layers=cfg["layers"])
    model, history = trainer.train_model(
        model, train_ds, valid_ds, epochs=args.epochs,
        batch_size=args.batch_size, lr=cfg["lr"], patience=4,
        num_threads=args.threads, seed=seed)

    ckpt = MODELS_DIR / f"gru_ev_{cfg['tag']}_h{args.horizon}.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt)

    pred = trainer.predict(model, test_ds, num_threads=args.threads)
    idx = test_ds.indices
    pred_df = pd.DataFrame({
        "date": dates[idx[:, 1]],
        "symbol": symbols[idx[:, 0]],
        "pred": pred,
        "y_excess": y3d[idx[:, 0], idx[:, 1]],
        "year": pd.to_datetime(dates[idx[:, 1]]).year,
    })
    # EV 专属命名，绝不覆盖基线 pred_gru_tune_*
    pred_df.to_parquet(PROCESSED / f"pred_gru_ev_{cfg['tag']}_h{args.horizon}.parquet",
                       index=False)

    s = metrics.summarize(pred_df, "pred", "y_excess")
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    bt = run_backtest(pred_df[["date", "symbol", "pred"]], panel,
                      horizon=args.horizon, top_pct=args.top_pct,
                      rebalance_freq=args.rebalance_freq,
                      cost=DEFAULT_COST, mode="long_short").ls_stats

    return {
        "tag": cfg["tag"], "seq_len": cfg["seq_len"], "hidden": cfg["hidden"],
        "lr": cfg["lr"], "layers": cfg["layers"], "in_dim": X3d.shape[2],
        "n_train": len(train_ds), "n_test": len(test_ds), "epochs": len(history),
        "ic_2026": s["ic_mean"], "icir_2026": s["icir"],
        "ic_pos_2026": s["ic_positive_rate"],
        "net_2026": bt.get("total_return"), "sharpe_2026": bt.get("sharpe"),
        "mdd_2026": bt.get("max_drawdown"), "win_2026": bt.get("win_rate"),
        "buckets_2026": bt.get("n_buckets"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 事件因子(牧羊人) + 10 迭代")
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="仅跑 t02 冒烟验证（约 3min）")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-symbols", type=int, default=800)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label-clip", type=float, default=0.5)
    ap.add_argument("--train-end", type=str, default="2024-12-31")
    ap.add_argument("--valid-end", type=str, default="2025-12-31")
    ap.add_argument("--top-pct", type=float, default=0.1)
    ap.add_argument("--rebalance-freq", type=int, default=15)
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--tags", type=str, default=None,
                    help="逗号分隔的多个 tag（分片并行用）：一次加载数据集跑完这批 tag；"
                         "各分片必须配不同的 --out-csv，否则全量重写会互相覆盖丢行")
    ap.add_argument("--out-csv", type=str, default="report_tune_grid_ev.csv")
    args = ap.parse_args()

    cfg = load_config(CONFIG_PATH)
    horizon = args.horizon or cfg.get_path("label.horizon", 10)
    seed = cfg.get_path("seed", 42)
    args.horizon = horizon

    # 阶段 A：构建 EV 数据集（不存在或 --build-only 时重建）
    ev_ds = PROCESSED / f"dataset_h{horizon}_ev.parquet"
    if args.build_only or not ev_ds.exists():
        build_event_dataset(horizon)
        if args.build_only:
            print("EV 数据集已构建。")
            return 0

    out_csv = PROCESSED / args.out_csv
    done = set()
    if out_csv.exists():
        try:
            done = set(pd.read_csv(out_csv)["tag"].tolist())
            logger.info("续跑：已完成 %s 个配置", len(done))
        except Exception:
            done = set()

    wanted = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    if args.config:
        wanted = [args.config] + wanted
    if wanted:
        known = {c["tag"] for c in GRID}
        bad = [t for t in wanted if t not in known]
        if bad:
            logger.error("未找到配置 %s", ", ".join(bad))
            return 1
        configs = [c for c in GRID if c["tag"] in set(wanted)]
    else:
        configs = GRID
    if args.smoke:
        configs = [c for c in GRID if c["tag"] == "t02_seq20_h64_lr3e-3"] or [GRID[1]]

    t0 = time.time()
    logger.info("加载 EV 数据集 h=%s ...", horizon)
    data, X3d, y3d, dates, symbols, _ = load_ev_dataset(horizon, args.label_clip)
    logger.info("EV 数据集 %s 行 | %s 只 | 输入维度 %s",
                f"{len(data):,}", len(symbols), X3d.shape[2])

    symbol_subset = pick_subset(symbols, args.max_symbols, seed)
    logger.info("股票子集: %s 只（None=全量）",
                len(symbol_subset) if symbol_subset is not None else "全量")

    rows = []
    if out_csv.exists():
        try:
            rows = pd.read_csv(out_csv).to_dict(orient="records")
        except Exception:
            rows = []

    for cfg in configs:
        if cfg["tag"] in done:
            logger.info("跳过已完成: %s", cfg["tag"])
            continue
        logger.info("=== 迭代 %s ===", cfg["tag"])
        try:
            r = run_one(cfg, X3d, y3d, dates, symbols, symbol_subset, args, seed)
            rows.append(r)
            pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6f")
            logger.info("[%s] 完成 → IC=%.4f ICIR=%.4f | 净 %.2f%% 夏普 %.2f",
                        cfg["tag"], r["ic_2026"], r["icir_2026"],
                        (r["net_2026"] or 0) * 100, r["sharpe_2026"])
        except Exception as e:
            logger.error("[%s] 失败: %s", cfg["tag"], e)
            traceback.print_exc()
            rows.append({**cfg, "error": str(e)})
            pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6f")

    # 汇总
    print("\n" + "=" * 82)
    print("  事件因子(牧羊人) + GRU 调参 × 2026 严格 hold-out（EV 47 维, rf=15）")
    print("=" * 82)
    print(f"  {'tag':<22} {'seq':>4} {'hid':>4} {'lr':>7} {'IC':>7} "
          f"{'ICIR':>7} {'净%':>8} {'夏普':>6} {'MDD%':>7} {'桶':>4}")
    for r in rows:
        if "error" in r:
            print(f"  {r['tag']:<22} ERROR: {r['error'][:60]}")
            continue
        print(f"  {r['tag']:<22} {r['seq_len']:>4} {r['hidden']:>4} "
              f"{r['lr']:>7.1e} {r['ic_2026']:>7.4f} {r['icir_2026']:>7.4f} "
              f"{(r['net_2026'] or 0)*100:>8.2f} {r['sharpe_2026']:>6.2f} "
              f"{(r['mdd_2026'] or 0)*100:>7.2f} {r['buckets_2026']:>4}")
    print("=" * 82)
    ok = [r for r in rows if "error" not in r]
    if ok:
        best = max(ok, key=lambda r: r["net_2026"] or -9)
        print(f"  最优(按2026净): {best['tag']} → 净 "
              f"{(best['net_2026'] or 0)*100:.2f}% / 夏普 {best['sharpe_2026']:.2f}")
    print(f"  总耗时 {time.time()-t0:.1f}s | 结果: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
