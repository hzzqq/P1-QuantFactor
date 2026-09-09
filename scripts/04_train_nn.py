"""阶段 4：训练 GRU+Attention 序列模型，并与 LightGBM 基线对比。

用法：
    python scripts/04_train_nn.py                       # 默认参数
    python scripts/04_train_nn.py --seq-len 30 --hidden 96
    python scripts/04_train_nn.py --max-symbols 500 --stride 10 --epochs 10   # 快速试跑

CPU 节流参数说明：
    --max-symbols  参与训练的股票数上限（越少越快）
    --stride       每隔几个交易日取一个训练样本（越大越快）
    --epochs       训练轮数
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # E:/project/sj
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                               # noqa: E402
import pandas as pd                              # noqa: E402
import torch                                     # noqa: E402

from shared import paths                         # noqa: E402
from shared.config import load_config            # noqa: E402
from shared.logging_utils import get_logger      # noqa: E402

from src.eval import metrics                     # noqa: E402
from src.models import dataset as ds_mod         # noqa: E402
from src.models import gru_attn                  # noqa: E402
from src.models import trainer                   # noqa: E402

logger = get_logger("P1.train_nn")
CONFIG_PATH = PROJ / "config" / "default.yaml"
PROCESSED = paths.DATA / "P1" / "processed"
MODELS_DIR = paths.MODELS / "P1"


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 训练神经网络")
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=20, help="回看窗口（交易日）")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stride", type=int, default=5,
                        help="训练样本采样步长（交易日）")
    parser.add_argument("--max-symbols", type=int, default=0,
                        help="参与训练的股票数上限，0=全部")
    parser.add_argument("--train-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, nargs="*", default=None)
    parser.add_argument("--model", default="gru", choices=["gru", "tcn"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--label-clip", type=float, default=0.5,
                        help="剔除 |超额收益| 超过该值的样本（异常复权/停牌复牌）")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    horizon = args.horizon or cfg.get_path("label.horizon", 10)
    seed = cfg.get_path("seed", 42)

    meta_path = PROCESSED / f"dataset_h{horizon}_meta.json"
    if not meta_path.exists():
        logger.error("找不到 %s，请先运行 02_build_features.py", meta_path)
        return 1
    with meta_path.open("r", encoding="utf-8") as f:
        factor_names = json.load(f)["factor_names"]

    t0 = time.time()
    data = pd.read_parquet(PROCESSED / f"dataset_h{horizon}.parquet")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    if args.label_clip > 0:
        before = len(data)
        data = data[data["y_excess"].abs() <= args.label_clip]
        logger.info("标签截断 |y|<=%s：%s -> %s 行（剔除 %s）",
                    args.label_clip, f"{before:,}", f"{len(data):,}",
                    f"{before - len(data):,}")
    logger.info("数据集 %s 行 | %s 只股票 | %s 因子",
                f"{len(data):,}", data["symbol"].nunique(), len(factor_names))

    X3d, y3d, dates, symbols = ds_mod.build_3d(data, factor_names)
    # R9：data 在 build_3d 后不再使用（date/symbol 已随返回值传出），
    # 立即释放大 DataFrame，避免与 X3d/y3d 同时驻留撑爆内存。
    del data
    import gc; gc.collect()
    n_s, n_d, n_f = X3d.shape

    # 控制训练规模（CPU 铁律：单次训练 ≤ 30 分钟）
    rng = np.random.default_rng(seed)
    symbol_subset = None
    if args.max_symbols and len(symbols) > args.max_symbols:
        keep = np.sort(rng.choice(len(symbols), args.max_symbols, replace=False))
        symbol_subset = symbols[keep]
        logger.info("限制股票数：%s -> %s", len(symbols), args.max_symbols)

    years = sorted(pd.to_datetime(pd.Series(dates)).dt.year.unique())
    if args.test_years:
        test_years = [y for y in args.test_years if y in years]
    else:
        test_years = [y for y in years if y >= years[0] + args.train_years + 1]

    all_preds, yearly = [], []
    for y in test_years:
        train_end = f"{y - 2}-12-31"
        valid_end = f"{y - 1}-12-31"
        tr_m, va_m, te_m, sym_m = ds_mod.make_masks(
            dates, train_end, valid_end, symbols, symbol_subset
        )
        if tr_m.sum() == 0 or te_m.sum() == 0:
            logger.warning("跳过 %s：训练或测试区间为空", y)
            continue

        train_ds = ds_mod.SequenceDataset(X3d, y3d, args.seq_len, sym_m, tr_m,
                                          date_stride=args.stride)
        valid_ds = ds_mod.SequenceDataset(X3d, y3d, args.seq_len, sym_m, va_m,
                                          date_stride=max(1, args.stride * 2))
        test_ds = ds_mod.SequenceDataset(X3d, y3d, args.seq_len, sym_m, te_m,
                                         date_stride=1)

        if len(train_ds) < 1000 or len(test_ds) < 50:
            logger.warning("跳过 %s：样本不足 (train=%s, test=%s)",
                           y, len(train_ds), len(test_ds))
            continue

        logger.info("=== %s | 训练样本 %s | 测试样本 %s ===",
                    y, f"{len(train_ds):,}", f"{len(test_ds):,}")

        if args.model == "gru":
            model = gru_attn.GRUAttention(n_f, hidden=args.hidden,
                                          n_layers=args.layers)
        else:
            model = gru_attn.TCN(n_f)

        model, history = trainer.train_model(
            model, train_ds, valid_ds, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, patience=4,
            num_threads=args.threads, seed=seed,
        )

        # 保存该年最佳模型权重，供 P5 导出 ONNX / 部署（待办：对接 P5）
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ckpt = MODELS_DIR / f"gru_best_h{horizon}_{y}.pt"
        torch.save(model.state_dict(), ckpt)
        logger.info("模型权重已保存: %s", ckpt)

        pred = trainer.predict(model, test_ds, num_threads=args.threads)
        idx = test_ds.indices
        pred_df = pd.DataFrame({
            "date": dates[idx[:, 1]],
            "symbol": symbols[idx[:, 0]],
            "pred": pred,
            "y_excess": y3d[idx[:, 0], idx[:, 1]],
            "year": y,
        })
        all_preds.append(pred_df)
        s = metrics.summarize(pred_df, "pred", "y_excess")
        yearly.append({"year": y, **{k: v for k, v in s.items() if k != "n_days"},
                       "n_train": len(train_ds), "n_test": len(test_ds),
                       "epochs": len(history)})
        logger.info("%s 结果: IC=%.4f ICIR=%.4f 正占比=%.3f",
                    y, s["ic_mean"], s["icir"], s["ic_positive_rate"])

    if not all_preds:
        logger.error("没有产出预测")
        return 1

    preds = pd.concat(all_preds, ignore_index=True)
    overall = metrics.summarize(preds, "pred", "y_excess")
    qr = metrics.quantile_returns(preds, "pred", "y_excess", 5)

    print("\n" + "=" * 62)
    print(f"  {args.model.upper()} 序列模型 · 滚动验证结果")
    print("=" * 62)
    print(f"  IC 均值     : {overall['ic_mean']: .4f}")
    print(f"  ICIR        : {overall['icir']: .4f}  (年化 {overall['icir_annual']: .2f})")
    print(f"  IC 正占比   : {overall['ic_positive_rate']: .3f}")
    print(f"  t 统计量    : {overall['t_stat']: .2f}")
    if "spread_mean" in overall:
        print(f"  多空收益差  : {overall['spread_mean']: .5f} / 期"
              f"   胜率 {overall['spread_win_rate']:.3f}")
    print("\n  分组超额收益（1=最弱 → 5=最强）：")
    for g, row in qr.iterrows():
        print(f"    第{g}组   {row['mean']: .5f}")
    print("=" * 62 + "\n")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_name = args.out or f"pred_{args.model}_h{horizon}.parquet"
    preds.to_parquet(PROCESSED / out_name, index=False)

    report = {
        "model": args.model, "horizon": horizon, "seq_len": args.seq_len,
        "hidden": args.hidden, "epochs": args.epochs, "stride": args.stride,
        "max_symbols": args.max_symbols, "overall": overall,
        "by_year": yearly,
        "quantile_returns": qr.reset_index().to_dict(orient="records"),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    rp = PROCESSED / f"report_{args.model}_h{horizon}.json"
    with rp.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    logger.info("预测已保存: %s", PROCESSED / out_name)
    logger.info("报告已保存: %s", rp)
    logger.info("总耗时 %.1f 秒", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
