"""阶段 27 - ②：把 log_amount（规模/流动性水平因子）并入数据集，作为第 39/48 维。

背景：
  - 38 个现有因子全是个股时序变换（动量/反转/波动/量比/振幅/乖离），完全不含横截面
    规模/流动性维度；阶段 26 的代理因子诊断显示 log_amount 自带显著小盘效应（逐年 IC 为负，
    量级强于多数现有因子），且与 vol_120 横截面正交 → 是 38 因子之外真正的新信息。
  - 数据缺口：stock_list.parquet 已不存在（raw/ 只剩 index/ + kline/），无逐日 mktcap/turnover；
    当前 panel.parquet 仅 2.69M 行，而数据集 3.32M 行（689K 行不在 panel 中）→ 若从 panel 派生
    会静默缩宇宙。故从**上游 raw/kline**（1429 只，覆盖数据集 100%）派生 amount=volume*VWAP。

做法（不破坏生产数据集）：
  - 从 raw/kline 计算 amount = volume*(high+low+2*close)/4，log_amount = log(amount)（>0 才取，否则 NaN）。
  - 每 symbol 内 forward/back-fill，覆盖停牌日的规模值（规模不随停牌突变）。
  - 落盘 _log_amount.parquet（symbol,date,log_amount）便于复用。
  - 分别 merge 进 dataset_h10 → dataset_h10_v39（38→39）、dataset_h10_ev → dataset_h10_ev_v48（47→48）。
  - 仅新增列，**不删不改**原生产数据集；覆盖缺失率会打印出来（应为 0）。

用法：
    python scripts/27_add_logamount_factor.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
RAW_KLINE = paths.DATA / "P1" / "raw" / "kline"
LA_PATH = PROCESSED / "_log_amount.parquet"


def build_log_amount() -> pd.DataFrame:
    """从 raw/kline 派生 (symbol, date, log_amount)。"""
    if LA_PATH.exists():
        t0 = time.time()
        la = pd.read_parquet(LA_PATH)
        print(f"[27] 复用缓存 _log_amount.parquet ({len(la):,} 行) {time.time()-t0:.1f}s")
        return la
    files = sorted(glob.glob(str(RAW_KLINE / "*.parquet")))
    print(f"[27] 读取 {len(files)} 个 raw kline 文件...", flush=True)
    t0 = time.time()
    parts = []
    for f in files:
        df = pd.read_parquet(
            f, columns=["symbol", "date", "open", "close", "high", "low", "volume"]
        )
        vwap = (df["high"] + df["low"] + 2 * df["close"]) / 4.0
        amount = df["volume"] * vwap
        la = pd.DataFrame({
            "symbol": df["symbol"],
            "date": pd.to_datetime(df["date"]),
            "log_amount": np.where(amount > 0, np.log(amount), np.nan),
        })
        parts.append(la)
    big = pd.concat(parts, ignore_index=True)
    del parts
    # 每 symbol 内 forward/back-fill，覆盖停牌日
    big = big.sort_values(["symbol", "date"])
    big["log_amount"] = big.groupby("symbol")["log_amount"].ffill().bfill()
    big = big.dropna(subset=["log_amount"])
    big.to_parquet(LA_PATH, index=False)
    print(f"[27] 构建 log_amount 完成 {len(big):,} 行 {time.time()-t0:.1f}s → {LA_PATH.name}")
    return big


def augment(src_name: str, dst_name: str, n_old: int) -> None:
    t0 = time.time()
    df = pd.read_parquet(PROCESSED / src_name)
    la = build_log_amount()
    merged = df.merge(la, on=["date", "symbol"], how="left")
    n_missing = int(merged["log_amount"].isna().sum())
    print(f"[27] {src_name}: 合并 log_amount 后缺失 {n_missing} 行 "
          f"({n_missing/len(merged)*100:.3f}%)", flush=True)
    if n_missing:
        # 仍有缺失（raw 也缺的极端情况）→ 用全局中位数填充，保证不缩宇宙
        med = merged["log_amount"].median()
        merged["log_amount"] = merged["log_amount"].fillna(med)
        print(f"[27]   用全局中位数 {med:.4f} 填充剩余缺失")
    # 列顺序：原因子 + log_amount + 标签
    factor_cols = [c for c in df.columns if c not in ("date", "symbol")]
    label_cols = [c for c in factor_cols if c.startswith("y_")]
    feat_cols = [c for c in factor_cols if not c.startswith("y_")]
    new_feats = feat_cols + ["log_amount"]
    new_order = ["date", "symbol"] + new_feats + label_cols
    merged = merged[new_order]
    merged.to_parquet(PROCESSED / dst_name, index=False)
    # meta
    meta = json.load(open(PROCESSED / src_name.replace(".parquet", "_meta.json"), encoding="utf-8"))
    meta["factor_names"] = meta["factor_names"] + ["log_amount"]
    meta["rows"] = int(len(merged))
    meta["symbols"] = int(merged["symbol"].nunique())
    meta["added_factor"] = "log_amount"
    meta["added_source"] = "raw/kline amount=volume*(high+low+2*close)/4, log, per-symbol ffill/bfill"
    meta["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(PROCESSED / dst_name.replace(".parquet", "_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"[27] 写出 {dst_name}（{len(merged):,} 行, {len(new_feats)} 因子）"
          f" + meta，{time.time()-t0:.1f}s")


def main() -> int:
    print("[27] 开始：log_amount 因子并入数据集", flush=True)
    # 38 维 → 39 维
    augment("dataset_h10.parquet", "dataset_h10_v39.parquet", 38)
    # 47 维(EV) → 48 维
    augment("dataset_h10_ev.parquet", "dataset_h10_ev_v48.parquet", 47)
    print("[27] 完成。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
