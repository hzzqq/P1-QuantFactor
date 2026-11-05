"""并发抓取与增量更新。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import pandas as pd
from tqdm import tqdm

from shared.logging_utils import get_logger

from . import sources, storage

logger = get_logger("P1.data.fetcher")


def _fetch_symbol(symbol: str, start: str, end: str, adjust: str,
                  incremental: bool, req_cfg: dict, segment_years: int) -> dict:
    """抓一只股票，返回统计信息。失败不抛异常，交由调用方汇总。"""
    origin_start = start
    old_df: pd.DataFrame | None = None

    if incremental:
        state = storage.load_fetch_state().get(symbol)
        if state and state.get("last_date"):
            next_day = pd.Timestamp(state["last_date"]) + timedelta(days=1)
            if next_day > pd.Timestamp(end):
                return {"symbol": symbol, "status": "skipped", "rows": 0}
            start = next_day.strftime("%Y-%m-%d")
        old_df = storage.load_kline(symbol)

    try:
        df = sources.fetch_kline(
            symbol, start, end, adjust=adjust,
            timeout=req_cfg.get("timeout", 15),
            retries=req_cfg.get("retries", 3),
            retry_sleep=req_cfg.get("retry_sleep", 1.0),
            sleep=req_cfg.get("sleep", 0.12),
            segment_years=segment_years,
        )
    except Exception as err:  # noqa: BLE001 - 单只失败不影响整体
        return {"symbol": symbol, "status": "error", "rows": 0, "msg": str(err)[:120]}

    if df.empty:
        return {"symbol": symbol, "status": "empty", "rows": 0}

    if old_df is not None and not old_df.empty:
        df = pd.concat([old_df, df], ignore_index=True)
        df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    storage.save_kline(df, symbol)
    storage.update_fetch_state(symbol, df["date"].max(), len(df))
    return {"symbol": symbol, "status": "ok", "rows": len(df),
            "start": str(df["date"].min().date()), "end": str(df["date"].max().date())}


def fetch_many(symbols: list[str], start: str, end: str, adjust: str = "qfq",
               incremental: bool = True, max_workers: int = 8,
               req_cfg: dict | None = None, segment_years: int = 2) -> dict:
    """并发抓取多只股票。

    Returns:
        统计字典：ok / empty / error / skipped 计数，以及失败清单
    """
    req_cfg = req_cfg or {}
    stats = {"ok": 0, "empty": 0, "error": 0, "skipped": 0, "total_rows": 0}
    failures: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_symbol, sym, start, end, adjust, incremental,
                        req_cfg, segment_years): sym
            for sym in symbols
        }
        with tqdm(total=len(futures), desc="抓取日线", unit="只") as bar:
            for future in as_completed(futures):
                res = future.result()
                stats[res["status"]] = stats.get(res["status"], 0) + 1
                if res["status"] == "ok":
                    stats["total_rows"] += res["rows"]
                elif res["status"] == "error":
                    failures.append(res)
                bar.update(1)
                bar.set_postfix(ok=stats["ok"], err=stats["error"],
                                empty=stats["empty"])

    stats["failures"] = failures[:20]
    logger.info(
        "抓取完成 成功=%s 空=%s 失败=%s 跳过=%s 总行数=%s",
        stats["ok"], stats["empty"], stats["error"], stats["skipped"],
        f"{stats['total_rows']:,}",
    )
    if failures:
        logger.warning("失败样例（最多 5 条）: %s", failures[:5])
    return stats


def build_universe(list_df: pd.DataFrame, cfg: dict) -> list[str]:
    """按配置过滤股票池。"""
    df = list_df.copy()

    if cfg.get("exclude_bj", True) and "symbol" in df.columns:
        before = len(df)
        df = df[~df["symbol"].str.startswith("bj", na=False)]
        logger.info("剔除北交所: %s -> %s", before, len(df))

    if cfg.get("exclude_st", True) and "name" in df.columns:
        before = len(df)
        mask = df["name"].astype(str).str.upper().str.contains("ST", na=False)
        df = df[~mask]
        logger.info("剔除 ST: %s -> %s", before, len(df))

    # 剔除 ETF / 基金等非股票标的
    if "symbol" in df.columns:
        df = df[~df["symbol"].str.startswith(("sh51", "sh58", "sz15", "sz16",
                                              "sh11", "sz12"), na=False)]

    # 按流通市值取前 N 只。
    # 小盘股流动性差、冲击成本高、噪声大，对"要能真用"的产品是负担而非资产。
    max_n = cfg.get("max_by_market_cap", 0)
    if max_n and "nmc" in df.columns and len(df) > max_n:
        before = len(df)
        df = df.sort_values("nmc", ascending=False).head(max_n)
        logger.info("按流通市值取前 %s 只: %s -> %s", max_n, before, len(df))

    return df["symbol"].dropna().unique().tolist()
