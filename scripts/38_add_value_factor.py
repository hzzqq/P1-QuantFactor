"""阶段 50 - alpha 再生：把估值因子(ep/btop)并入 v39 → v41 数据集。

背景
----
37_value_factor_ic_raw.py 已证：估值因子(EP/BTOP)对现有 39 因子的**残差 IC 在 2024+/2026
窗口显著为正**（ep 2024+ +0.0122 t+5.0 / 2026 +0.0271 t+6.65；btop 2026 +0.0145 t+3.03），
方向与换手率/log_amount(负)相反、**正交增量**——是②b 之后真正的新 alpha 源。
本脚本把 ep/btop 从干净 raw 派生并 merge 进 dataset_h10_v39 → dataset_h10_v41（41 维）。

做法（守 32 的"不伪造"纪律）
----
- ep = 1/peTTM（仅 peTTM>0）；btop = 1/pbMRQ（仅 pbMRQ>0）。
- ep 跨日缺失（亏损股当日无 PE）用**同 symbol 内 ffill/bfill** 覆盖（估值不逐日突变），
  仍缺失（长期亏损）的行在并入时 dropna（绝不中位数伪造）。
- btop null 仅 0.13%，直接 dropna。
- 覆盖门禁：symbol 覆盖率 <0.95 拒绝（防部分成功伪造）；达标才构建。
- 仅新增列，不删不改原 39 维；meta 记录 added_factors / source / coverage / built_at。

用法：
    python scripts/38_add_value_factor.py
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

RAW = paths.DATA / "P1" / "raw" / "baostock_daily" / "baostock_daily.parquet"
PROC = paths.DATA / "P1" / "processed"
SRC = PROC / "dataset_h10_v39.parquet"
DST = PROC / "dataset_h10_v41.parquet"
MIN_COVERAGE = 0.95


def derive_value() -> pd.DataFrame:
    t0 = time.time()
    df = pd.read_parquet(RAW, columns=["symbol", "date", "peTTM", "pbMRQ",
                                       "tradestatus", "isST"])
    df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce")
    df["isST"] = pd.to_numeric(df["isST"], errors="coerce")
    df = df[df["isST"].fillna(0) != 1]
    df = df[df["tradestatus"].fillna(1) == 1]
    pe = df["peTTM"].astype(float)
    pb = df["pbMRQ"].astype(float)
    out = pd.DataFrame({
        "symbol": df["symbol"].astype(str),
        "date": pd.to_datetime(df["date"]),
        "ep": np.where(pe > 0, 1.0 / pe, np.nan).astype(np.float32),
        "btop": np.where(pb > 0, 1.0 / pb, np.nan).astype(np.float32),
    })
    # ep 同 symbol 内 ffill/bfill 覆盖亏损股当日缺口
    out = out.sort_values(["symbol", "date"])
    out["ep"] = out.groupby("symbol")["ep"].ffill().bfill()
    out = out.replace([np.inf, -np.inf], np.nan)
    print(f"[38] 派生 ep/btop 完成 {len(out):,} 行 {time.time()-t0:.1f}s", flush=True)
    return out


def augment() -> None:
    vf = derive_value()
    t0 = time.time()
    df = pd.read_parquet(SRC)
    ds_syms = set(df["symbol"].astype(str).unique())
    vf_syms = set(vf["symbol"].astype(str).unique())
    covered = ds_syms & vf_syms
    coverage = len(covered) / len(ds_syms) if ds_syms else 0.0
    print(f"[38] 数据集 {len(ds_syms)} 只，估值源覆盖 {len(covered)} 只，"
          f"覆盖率 {coverage:.1%}", flush=True)
    if coverage < MIN_COVERAGE:
        if os.environ.get("P1_ALLOW_PARTIAL"):
            print(f"[38] ⚠️ 覆盖率 {coverage:.1%} < {MIN_COVERAGE:.0%}，"
                  f"P1_ALLOW_PARTIAL=1 → 仍构建（抽样验证用，非全量生产）", flush=True)
        else:
            raise RuntimeError(
                f"[38] 覆盖率 {coverage:.1%} < {MIN_COVERAGE:.0%}，拒绝伪造。")
    merged = df.merge(vf, on=["date", "symbol"], how="left")
    n_before = len(merged)
    new_factors = ["ep", "btop"]
    merged = merged.dropna(subset=new_factors)
    n_dropped = n_before - len(merged)
    print(f"[38] 丢弃新因子缺失行 {n_dropped}（{n_dropped/n_before:.3%}），不填充", flush=True)
    # 列序：原因子 + 新因子 + 标签
    factor_cols = [c for c in df.columns if c not in ("date", "symbol")]
    label_cols = [c for c in factor_cols if c.startswith("y_")]
    feat_cols = [c for c in factor_cols if not c.startswith("y_")]
    new_feats = feat_cols + new_factors
    new_order = ["date", "symbol"] + new_feats + label_cols
    merged = merged[new_order]
    merged.to_parquet(DST, index=False)
    meta = json.load(open(SRC.with_name("dataset_h10_v39_meta.json"), encoding="utf-8"))
    meta["factor_names"] = meta["factor_names"] + new_factors
    meta["rows"] = int(len(merged))
    meta["symbols"] = int(merged["symbol"].nunique())
    meta["added_factors"] = new_factors
    meta["added_source"] = "baostock_daily: ep=1/peTTM(>0), btop=1/pbMRQ(>0); ep 同symbol ffill/bfill"
    meta["coverage_symbols"] = round(coverage, 4)
    meta["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(DST.with_name("dataset_h10_v41_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"[38] 写出 {DST.name}（{len(merged):,} 行, {len(new_feats)} 因子, "
          f"覆盖率 {coverage:.1%}）+ meta，{time.time()-t0:.1f}s", flush=True)


def main() -> int:
    print("[38] alpha 再生：估值因子并入数据集 → v41", flush=True)
    augment()
    print("[38] 完成。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
