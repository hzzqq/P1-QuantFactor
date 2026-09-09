"""阶段 10：GRU 调参 × 2026 严格 hold-out（10 次迭代）。

把「GRU 调参」与「2026 全年严格 hold-out」合并进同一个可复现单元：
每个迭代 = 一组超参 (seq_len, hidden, lr)，训练**单一模型**：
    - 训练：所有数据 ≤ train_end（默认 2024-12-31）
    - 验证：valid_end 当年（默认 2025）
    - 测试：严格 hold-out，仅 2026（模型训练时从未见过）
在 2026 上计算 IC / ICIR 并跑 rf=15 成本敏感回测（与生产设置对齐）。

10 组配置覆盖 seq_len ∈ {20,30,40} × hidden ∈ {64,96,128} × lr 搜索，
固定同一只股票子集（seeded，默认 800 只）以保证横向可比。增量写 CSV，
中断可续跑（已完成的 tag 跳过）。

用法：
    # 调参网格（子集 800 只，后台跑 10 组）
    python scripts/10_iterate_tune.py --max-symbols 800 --out-csv report_tune_grid.csv

    # 对最优配置做全量(1427只)确认
    python scripts/10_iterate_tune.py --max-symbols 0 --config tag_01 --out-csv report_tune_full.csv
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

logger = get_logger("P1.iterate_tune")
CONFIG_PATH = PROJ / "config" / "default.yaml"
PROCESSED = paths.DATA / "P1" / "processed"
MODELS_DIR = paths.MODELS / "P1"

# 10 次迭代的调参网格
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
    # 扩展至 20 组（更密 lr 搜索 + 补充 hidden/seq 组合）
    dict(tag="t11_seq20_h64_lr2e-3", seq_len=20, hidden=64, lr=2e-3, layers=2),
    dict(tag="t12_seq20_h96_lr3e-3", seq_len=20, hidden=96, lr=3e-3, layers=2),
    dict(tag="t13_seq20_h128_lr3e-3", seq_len=20, hidden=128, lr=3e-3, layers=2),
    dict(tag="t14_seq30_h64_lr5e-4", seq_len=30, hidden=64, lr=5e-4, layers=2),
    dict(tag="t15_seq30_h96_lr5e-4", seq_len=30, hidden=96, lr=5e-4, layers=2),
    dict(tag="t16_seq30_h128_lr1e-3", seq_len=30, hidden=128, lr=1e-3, layers=2),
    dict(tag="t17_seq40_h64_lr3e-3", seq_len=40, hidden=64, lr=3e-3, layers=2),
    dict(tag="t18_seq40_h96_lr1e-3", seq_len=40, hidden=96, lr=1e-3, layers=2),
    dict(tag="t19_seq40_h128_lr3e-3", seq_len=40, hidden=128, lr=3e-3, layers=2),
    dict(tag="t20_seq20_h64_lr1e-4", seq_len=20, hidden=64, lr=1e-4, layers=2),
]


def load_dataset(horizon: int, label_clip: float):
    meta_path = PROCESSED / f"dataset_h{horizon}_meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        factor_names = json.load(f)["factor_names"]
    data = pd.read_parquet(PROCESSED / f"dataset_h{horizon}.parquet")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    if label_clip > 0:
        data = data[data["y_excess"].abs() <= label_clip]
    X3d, y3d, dates, symbols = ds_mod.build_3d(data, factor_names)
    return data, X3d, y3d, dates, symbols, factor_names


def pick_subset(symbols: np.ndarray, max_symbols: int, seed: int) -> np.ndarray | None:
    if max_symbols <= 0 or len(symbols) <= max_symbols:
        return None
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(symbols), max_symbols, replace=False))
    return symbols[keep]


def run_one(cfg, X3d, y3d, dates, symbols, symbol_subset, args, seed):
    """训练单一模型（训≤train_end，测=test_year），返回结果 dict。"""
    train_end, valid_end = args.train_end, args.valid_end
    tr_m, va_m, te_m, sym_m = ds_mod.make_masks(
        dates, train_end, valid_end, symbols, symbol_subset)
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

    logger.info("[%s] train=%s test=%s | seq=%s h=%s lr=%s",
                cfg["tag"], f"{len(train_ds):,}", f"{len(test_ds):,}",
                cfg["seq_len"], cfg["hidden"], cfg["lr"])

    model = gru_attn.GRUAttention(X3d.shape[2], hidden=cfg["hidden"],
                                  n_layers=cfg["layers"])
    model, history = trainer.train_model(
        model, train_ds, valid_ds, epochs=args.epochs,
        batch_size=args.batch_size, lr=cfg["lr"], patience=4,
        num_threads=args.threads, seed=seed)

    ckpt = MODELS_DIR / f"gru_tune_{cfg['tag']}_h{args.horizon}.pt"
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
    pred_df.to_parquet(PROCESSED / f"pred_gru_tune_{cfg['tag']}_h{args.horizon}.parquet",
                       index=False)

    # 2026 严格 hold-out 评估
    s = metrics.summarize(pred_df, "pred", "y_excess")

    # 2026 回测（rf=15，与生产对齐）
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    bt = run_backtest(pred_df[["date", "symbol", "pred"]], panel,
                      horizon=args.horizon, top_pct=args.top_pct,
                      rebalance_freq=args.rebalance_freq,
                      cost=DEFAULT_COST, mode="long_short").ls_stats

    return {
        "tag": cfg["tag"], "seq_len": cfg["seq_len"], "hidden": cfg["hidden"],
        "lr": cfg["lr"], "layers": cfg["layers"],
        "n_train": len(train_ds), "n_test": len(test_ds), "epochs": len(history),
        "ic_2026": s["ic_mean"], "icir_2026": s["icir"],
        "ic_pos_2026": s["ic_positive_rate"],
        "net_2026": bt.get("total_return"), "sharpe_2026": bt.get("sharpe"),
        "mdd_2026": bt.get("max_drawdown"), "win_2026": bt.get("win_rate"),
        "buckets_2026": bt.get("n_buckets"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 GRU 调参 × 2026 严格 hold-out")
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=2)
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
    ap.add_argument("--out-csv", type=str, default="report_tune_grid.csv")
    ap.add_argument("--config", type=str, default=None,
                    help="只跑该 tag（如 t01_seq20_h64_lr1e-3）；不指定则跑全网格")
    args = ap.parse_args()

    cfg = load_config(CONFIG_PATH)
    horizon = args.horizon or cfg.get_path("label.horizon", 10)
    seed = cfg.get_path("seed", 42)
    args.horizon = horizon

    out_csv = PROCESSED / args.out_csv
    done = set()
    if out_csv.exists():
        try:
            done = set(pd.read_csv(out_csv)["tag"].tolist())
            logger.info("续跑：已完成 %s 个配置，跳过", len(done))
        except Exception:
            done = set()

    configs = [c for c in GRID if c["tag"] == args.config] if args.config else GRID
    if args.config and not configs:
        logger.error("未找到配置 %s", args.config)
        return 1

    t0 = time.time()
    logger.info("加载数据集 h=%s ...", horizon)
    data, X3d, y3d, dates, symbols, _ = load_dataset(horizon, args.label_clip)
    logger.info("数据集 %s 行 | %s 只", f"{len(data):,}", len(symbols))

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

    # 汇总表
    print("\n" + "=" * 78)
    print("  GRU 调参 × 2026 严格 hold-out 汇总（rf=15，多空 top10% h10）")
    print("=" * 78)
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
    print("=" * 78)
    ok = [r for r in rows if "error" not in r]
    if ok:
        best = max(ok, key=lambda r: r["net_2026"] or -9)
        print(f"  最优（按 2026 净收益）: {best['tag']} → 净 "
              f"{(best['net_2026'] or 0)*100:.2f}% / 夏普 {best['sharpe_2026']:.2f}")
    print(f"  总耗时 {time.time()-t0:.1f}s | 结果: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
