"""阶段 28③：复用 h10_v39 因子，仅重建 horizon=20 标签 → dataset_h20_v39.parquet。

h10 与 h20 的唯一差异是标签 y_excess 的前向窗口（10→20 日）；38 个 trailing 窗口因子
完全 horizon 无关。故直接复用 label_mod.build_labels 算 h20 超额收益，按 20 万行分批
merge 进现有 dataset_h10_v39，避免一次性载 3.32M 行撞沙箱每进程 commit 上限（SIGSEGV）。

用法：
    python scripts/27b_build_h20_labels.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from shared import paths  # noqa: E402
from shared.config import load_config  # noqa: E402
from shared.logging_utils import get_logger  # noqa: E402
from src.features import labels as label_mod  # noqa: E402

logger = get_logger("P1.build_h20_labels")
CONFIG_PATH = PROJ / "config" / "default.yaml"
PROCESSED = paths.DATA / "P1" / "processed"
HORIZON = 20
SRC = PROCESSED / "dataset_h10_v39.parquet"
DST = PROCESSED / f"dataset_h20_v39.parquet"
SRC_META = PROCESSED / "dataset_h10_v39_meta.json"
DST_META = PROCESSED / f"dataset_h20_v39_meta.json"
BENCH = "sh000300"
BATCH = 200_000


def main() -> int:
    t0 = time.time()
    cfg = load_config(CONFIG_PATH)
    benchmark = cfg.get_path("data.benchmark", BENCH)

    # 1) 轻量加载 panel 仅 close 列（标签计算只需 close + bench）
    logger.info("加载 panel(仅 close) ...")
    panel = pd.read_parquet(
        PROCESSED / "panel.parquet", columns=["date", "symbol", "close"]
    )
    panel["date"] = pd.to_datetime(panel["date"])
    bench_path = paths.DATA / "P1" / "raw" / "index" / f"{benchmark}.parquet"
    bench_df = pd.read_parquet(bench_path)
    bench_df["date"] = pd.to_datetime(bench_df["date"])

    # 2) 复用管线算 horizon=20 超额收益 → stack 成小表 (date,symbol,y_excess_h20)
    labs = label_mod.build_labels(panel, bench_df, horizon=HORIZON)
    exc = labs["excess"].stack().reset_index()
    exc.columns = ["date", "symbol", "y_excess_h20"]
    exc["date"] = pd.to_datetime(exc["date"])
    exc = exc[["date", "symbol", "y_excess_h20"]]
    logger.info("h20 标签 %s 行（有效 %s）", f"{len(exc):,}",
                f"{int(exc['y_excess_h20'].notna().sum()):,}")

    # 3) 分批读取 v39，逐批 merge + 换标签 + dropna + 写出
    if DST.exists():
        DST.unlink()
    writer = None
    total = 0
    pf = pq.ParquetFile(SRC)
    fac_cols = None
    for batch in pf.iter_batches(batch_size=BATCH):
        df = batch.to_pandas()
        df = df.merge(exc, on=["date", "symbol"], how="left")
        df = df.rename(columns={"y_excess": "y_excess_h10"})
        df["y_excess"] = df["y_excess_h20"]
        df = df.drop(columns=["y_excess_h20"])
        df = df.dropna(subset=["y_excess"])
        if len(df) == 0:
            continue
        # 因子 downcast 省内存/磁盘
        if fac_cols is None:
            fac_cols = [c for c in df.columns if c not in
                        ("date", "symbol", "y_excess", "y_excess_h10", "y_cls", "y_fwd")]
        for c in fac_cols:
            df[c] = df[c].astype(np.float32)
        # y_excess 也 float32
        df["y_excess"] = df["y_excess"].astype(np.float32)
        if "y_excess_h10" in df.columns:
            df["y_excess_h10"] = df["y_excess_h10"].astype(np.float32)
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(DST, tbl.schema)
        writer.write_table(tbl)
        total += len(df)
    writer.close()
    logger.info("写出 %s：%s 行（尾部无未来 20 日已 dropna）", DST.name, f"{total:,}")

    # 4) meta
    if SRC_META.exists():
        meta = json.load(open(SRC_META, encoding="utf-8"))
    else:
        meta = {"factor_names": fac_cols}
    meta["horizon"] = HORIZON
    meta["n_rows"] = int(total)
    meta["label"] = "y_excess (forward excess return, horizon=20)"
    meta["note"] = "因子同 dataset_h10_v39；仅标签改为 horizon=20"
    json.dump(meta, open(DST_META, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    logger.info("完成 %.1fs → %s", time.time() - t0, DST.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
