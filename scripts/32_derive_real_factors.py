"""阶段 30 - ②b：从 baostock_daily 派生真·规模/换手因子，并入 v39 数据集 → v40。

派生逻辑（基于 baostock 真实字段）：
  - float_shares ≈ volume(手) * 100 / turn(%)   （turn 为换手率百分比，volume 为手=100股）
  - log_mktcap_real = log(close * float_shares) = log(close * volume * 100 / turn)
      这是**真实流通市值**的对数值，优于阶段 27 的纯 amount 代理（amount 缺少流通股本维度）。
  - turnover_daily = turn / 100   （真实逐日换手率，小数）

并入：
  - dataset_h10_v39 (39维) → dataset_h10_v40 (41维：39 + log_mktcap_real + turnover_daily)
  - dataset_h20_v39 (39维) → dataset_h20_v40 (41维)
  - 同时保留原 log_amount（阶段27代理）以便对比，不删除。

用法：
    python scripts/32_derive_real_factors.py
"""
from __future__ import annotations

import json
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
SRC = paths.DATA / "P1" / "raw" / "baostock_daily" / "baostock_daily.parquet"


def derive_real() -> pd.DataFrame:
    if not SRC.exists():
        raise FileNotFoundError(f"未找到 {SRC}；请先跑 31_baostock_fetch.py --combine")
    t0 = time.time()
    df = pd.read_parquet(SRC)
    print(f"[32] 读 baostock_daily {len(df):,} 行", flush=True)
    # 计算真实因子
    turn = df["turn"].astype(float)
    vol = df["volume"].astype(float)
    close = df["close"].astype(float)
    # 防 0/负
    turn_safe = turn.clip(lower=1e-4)
    float_shares = vol * 100.0 / turn_safe          # 流通股本（股）
    log_mktcap_real = np.log((close * float_shares).clip(lower=1e-6))
    turnover_daily = (turn / 100.0).clip(lower=0.0)
    out = pd.DataFrame({
        "symbol": df["symbol"].astype(str),
        "date": pd.to_datetime(df["date"]),
        "log_mktcap_real": log_mktcap_real.astype(np.float32),
        "turnover_daily": turnover_daily.astype(np.float32),
    })
    out = out.replace([np.inf, -np.inf], np.nan)
    print(f"[32] 派生完成 {len(out):,} 行 {time.time()-t0:.1f}s", flush=True)
    return out


def augment(src_name: str, dst_name: str, new_factors: list[str]) -> None:
    rf = derive_real()
    t0 = time.time()
    df = pd.read_parquet(PROCESSED / src_name)
    merged = df.merge(rf, on=["date", "symbol"], how="left")
    n_missing = int(merged[new_factors].isna().all(axis=1).sum())
    print(f"[32] {src_name}: 合并真实因子后整行缺失 {n_missing} "
          f"({n_missing/len(merged)*100:.3f}%)", flush=True)
    for c in new_factors:
        if merged[c].isna().any():
            med = merged[c].median()
            merged[c] = merged[c].fillna(med)
            print(f"[32]   {c} 用中位数 {med:.4f} 填充剩余缺失")
    # 列序：原因子 + 新因子 + 标签
    factor_cols = [c for c in df.columns if c not in ("date", "symbol")]
    label_cols = [c for c in factor_cols if c.startswith("y_")]
    feat_cols = [c for c in factor_cols if not c.startswith("y_")]
    new_feats = feat_cols + new_factors
    new_order = ["date", "symbol"] + new_feats + label_cols
    merged = merged[new_order]
    merged.to_parquet(PROCESSED / dst_name, index=False)
    meta = json.load(open(PROCESSED / src_name.replace(".parquet", "_meta.json"), encoding="utf-8"))
    meta["factor_names"] = meta["factor_names"] + new_factors
    meta["rows"] = int(len(merged))
    meta["symbols"] = int(merged["symbol"].nunique())
    meta["added_factors"] = new_factors
    meta["added_source"] = "baostock_daily: log_mktcap_real=log(close*vol*100/turn), turnover_daily=turn/100"
    meta["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(PROCESSED / dst_name.replace(".parquet", "_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"[32] 写出 {dst_name}（{len(merged):,} 行, {len(new_feats)} 因子）"
          f" + meta，{time.time()-t0:.1f}s", flush=True)


def main() -> int:
    print("[32] 开始：真·规模/换手因子并入数据集 → v40", flush=True)
    augment("dataset_h10_v39.parquet", "dataset_h10_v40.parquet",
            ["log_mktcap_real", "turnover_daily"])
    augment("dataset_h20_v39.parquet", "dataset_h20_v40.parquet",
            ["log_mktcap_real", "turnover_daily"])
    print("[32] 完成。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
