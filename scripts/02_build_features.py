"""阶段 2：构建特征与标签数据集。

用法：
    python scripts/02_build_features.py                 # 用默认参数
    python scripts/02_build_features.py --horizon 20    # 预测未来 20 日
    python scripts/02_build_features.py --rebuild-panel # 强制从 raw 重建面板
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

import pandas as pd                              # noqa: E402

from shared import paths                         # noqa: E402
from shared.config import load_config            # noqa: E402
from shared.logging_utils import get_logger      # noqa: E402

from src.data import storage                     # noqa: E402
from src.features import pipeline                # noqa: E402

logger = get_logger("P1.build_features")
CONFIG_PATH = PROJ / "config" / "default.yaml"
PROCESSED_DIR = paths.DATA / "P1" / "processed"
INDEX_DIR = paths.DATA / "P1" / "raw" / "index"


def load_benchmark(symbol: str) -> pd.DataFrame:
    path = INDEX_DIR / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"基准数据不存在: {path}，请先运行 scripts/01_fetch_data.py"
        )
    return pd.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 特征工程")
    parser.add_argument("--horizon", type=int, default=0,
                        help="预测未来 N 日，默认取配置值")
    parser.add_argument("--rebuild-panel", action="store_true",
                        help="强制从 raw 文件重建面板")
    parser.add_argument("--out", default=None, help="输出文件名")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    horizon = args.horizon or cfg.get_path("label.horizon", 10)
    windows = cfg.get_path("features.windows", [5, 10, 20, 60, 120])
    benchmark = cfg.get_path("data.benchmark", "sh000300")

    t0 = time.time()
    logger.info("=== 特征构建开始 horizon=%s ===", horizon)

    panel = None if args.rebuild_panel else storage.load_panel()
    if panel is None or panel.empty:
        logger.info("面板缺失或要求重建，从 raw 文件合并...")
        panel = storage.build_panel()
        if panel.empty:
            logger.error("没有可用的日线数据，请先运行 01_fetch_data.py")
            return 1
        storage.save_panel(panel)

    logger.info("面板: %s 行 | %s 只股票 | %s ~ %s",
                f"{len(panel):,}", panel["symbol"].nunique(),
                panel["date"].min().date(), panel["date"].max().date())

    bench = load_benchmark(benchmark)
    logger.info("基准 %s: %s 行", benchmark, f"{len(bench):,}")

    data = pipeline.build_dataset(panel, bench, horizon=horizon, windows=windows)

    names = data.attrs.get("factor_names", [])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / (args.out or f"dataset_h{horizon}.parquet")
    data.to_parquet(out_path, index=False)

    meta_path = PROCESSED_DIR / f"dataset_h{horizon}_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({
            "horizon": horizon,
            "windows": windows,
            "benchmark": benchmark,
            "factor_names": names,
            "rows": int(len(data)),
            "symbols": int(data["symbol"].nunique()),
            "date_range": [str(data["date"].min().date()),
                           str(data["date"].max().date())],
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, ensure_ascii=False, indent=2)

    logger.info("数据集已保存: %s", out_path)
    logger.info("  %s 行 | %s 只股票 | %s 个因子 | %s ~ %s",
                f"{len(data):,}", data["symbol"].nunique(), len(names),
                data["date"].min().date(), data["date"].max().date())
    logger.info("  标签分布: 超额收益均值 %.4f，中位数 %.4f，正样本占比 %.2f%%",
                data["y_excess"].mean(), data["y_excess"].median(),
                (data["y_excess"] > 0).mean() * 100)
    logger.info("=== 完成，耗时 %.1f 秒 ===", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
