"""阶段 1：抓取日线数据（腾讯前复权）+ A股列表（新浪）。

用法：
    # 先跑小样本验证链路（强烈建议）
    python scripts/01_fetch_data.py --limit 20

    # 全市场抓取
    python scripts/01_fetch_data.py --workers 12

    # 忽略已有数据，全量重抓
    python scripts/01_fetch_data.py --fresh
"""
from __future__ import annotations

import argparse
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

from src.data import fetcher, sources, storage   # noqa: E402

logger = get_logger("P1.fetch_data")
CONFIG_PATH = ROOT / "P1-QuantFactor" / "config" / "default.yaml"
INDEX_DIR = paths.DATA / "P1" / "raw" / "index"


def fetch_benchmark(symbol: str, start: str, end: str, req_cfg: dict) -> pd.DataFrame:
    """抓取基准指数，用于后续计算超额收益。"""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    path = INDEX_DIR / f"{symbol}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        logger.info("基准已存在 %s，行数 %s", symbol, len(df))
        return df

    df = sources.fetch_index(
        symbol, start, end,
        timeout=req_cfg.get("timeout", 15),
        retries=req_cfg.get("retries", 3),
        retry_sleep=req_cfg.get("retry_sleep", 1.0),
        sleep=req_cfg.get("sleep", 0.12),
    )
    if df.empty:
        logger.error("基准指数抓取失败: %s", symbol)
        return df
    df.to_parquet(path, index=False)
    logger.info("基准 %s 已保存，%s 行，%s ~ %s", symbol, len(df),
                df["date"].min().date(), df["date"].max().date())
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 数据抓取")
    parser.add_argument("--limit", type=int, default=0,
                        help="只抓前 N 只，0 表示全部")
    parser.add_argument("--workers", type=int, default=0,
                        help="并发线程数，默认取配置值")
    parser.add_argument("--fresh", action="store_true",
                        help="忽略已有数据，全量重抓")
    parser.add_argument("--start", default=None, help="覆盖配置中的起始日期")
    parser.add_argument("--end", default=None, help="覆盖配置中的结束日期")
    parser.add_argument("--no-panel", action="store_true",
                        help="跳过面板合并（大批量抓取时可省时间）")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    data_cfg = cfg.get_path("data", {})
    req_cfg = data_cfg.get("request", {})
    start = args.start or data_cfg.get("start_date", "2018-01-01")
    end = args.end or data_cfg.get("end_date", "2026-08-30")
    adjust = data_cfg.get("adjust", "qfq")
    segment_years = data_cfg.get("segment_years", 2)
    workers = args.workers or req_cfg.get("max_workers", 8)

    t0 = time.time()
    logger.info("=== P1 数据抓取开始 %s ~ %s (adjust=%s) ===", start, end, adjust)

    # 1. 股票列表
    list_df = storage.load_stock_list()
    if list_df is None or list_df.empty:
        logger.info("抓取 A 股列表（新浪）...")
        list_df = sources.fetch_stock_list(
            timeout=req_cfg.get("timeout", 15),
            retries=req_cfg.get("retries", 3),
            sleep=req_cfg.get("sleep", 0.12),
        )
        if list_df.empty:
            logger.error("股票列表抓取失败，终止")
            return 1
        storage.save_stock_list(list_df)
    logger.info("A股列表 %s 条", f"{len(list_df):,}")

    # 2. 股票池过滤
    symbols = fetcher.build_universe(list_df, cfg.get_path("universe", {}))
    logger.info("过滤后股票池 %s 只", f"{len(symbols):,}")
    if args.limit > 0:
        symbols = symbols[: args.limit]
        logger.info("小样本模式：只抓前 %s 只", args.limit)

    # 3. 基准指数
    benchmark = data_cfg.get("benchmark", "sh000300")
    fetch_benchmark(benchmark, start, end, req_cfg)

    # 4. 并发抓取日线
    stats = fetcher.fetch_many(
        symbols, start, end, adjust=adjust,
        incremental=not args.fresh,
        max_workers=workers,
        req_cfg=req_cfg,
        segment_years=segment_years,
    )

    # 5. 合并面板
    if not args.no_panel:
        panel = storage.build_panel(symbols)
        if not panel.empty:
            storage.save_panel(panel)
            logger.info("面板: %s 行 | %s 只股票 | %s ~ %s",
                        f"{len(panel):,}", panel["symbol"].nunique(),
                        panel["date"].min().date(), panel["date"].max().date())

    logger.info("=== 完成，耗时 %.1f 秒 ===", time.time() - t0)
    logger.info("统计: %s", {k: v for k, v in stats.items() if k != "failures"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
