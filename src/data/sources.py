"""行情数据源封装。

数据源选择说明（重要）：
    本机网络环境下东方财富接口（push2his.eastmoney.com）连接被拒，
    akshare 大部分接口建立在东财之上，故整体弃用 akshare。
    实测可用且稳定的两个源：
      - 腾讯 web.ifzq.gtimg.cn：前复权日线，支持日期区间（格式须为 YYYY-MM-DD）
      - 新浪 vip.stock.finance.sina.com.cn：A股列表，含市值 / 换手率

腾讯接口要点（踩过的坑）：
  1. 日期必须是 "YYYY-MM-DD"，写成 "20250101" 会返回 {"msg": "param error"}
  2. 单次最多约 640 行，超过返回异常，故按年分段抓取
  3. 前复权数据在 "qfqday" 字段，不复权在 "day" 字段
  4. 返回行顺序为 [日期, 开, 收, 高, 低, 成交量(手)] —— 收盘在第 3 位，不是第 2 位
"""
from __future__ import annotations

import json
import random
import time
from datetime import timedelta

import pandas as pd
import requests

from shared.logging_utils import get_logger

logger = get_logger("P1.data.sources")

TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
SINA_LIST_URL = (
    "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "Market_Center.getHQNodeData"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://gu.qq.com/",
}

MAX_ROWS_PER_REQUEST = 640
KLINE_COLUMNS = ["date", "open", "close", "high", "low", "volume"]


def normalize_symbol(code: str) -> str:
    """把 6 位代码补成带市场前缀的形式：600519 -> sh600519"""
    c = str(code).strip().lower()
    if c.startswith(("sh", "sz", "bj")):
        return c
    if c.startswith(("60", "68", "51", "58", "11")):
        return "sh" + c
    if c.startswith(("00", "30", "12", "15", "16")):
        return "sz" + c
    if c.startswith(("4", "8", "92")):
        return "bj" + c
    return "sh" + c


def _request(url: str, params: dict, timeout: int, retries: int,
             retry_sleep: float) -> requests.Response:
    """带退避重试的 GET。

    腾讯对高频请求会返回 501（Not Implemented），这不是真的未实现，
    而是限流。对 5xx / 429 采用更长退避并叠加随机抖动 —— 抖动很关键，
    否则多线程会整齐地同时重试，再次撞上限流。
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if resp.status_code >= 500 or resp.status_code == 429:
                raise requests.HTTPError(
                    f"{resp.status_code} {resp.reason}（疑似限流）", response=resp
                )
            resp.raise_for_status()
            return resp
        except Exception as err:  # noqa: BLE001 - 网络异常统一兜底重试
            last_err = err
            if attempt >= retries - 1:
                break
            status = getattr(getattr(err, "response", None), "status_code", 0) or 0
            if status >= 500 or status == 429:
                wait = retry_sleep * (4**attempt) + random.uniform(0.5, 2.5)
            else:
                wait = retry_sleep * (2**attempt)
            logger.debug("重试 %s/%s：%s，等待 %.1fs",
                         attempt + 1, retries, str(err)[:60], wait)
            time.sleep(wait)
    raise RuntimeError(f"请求失败 {url} params={params}: {last_err}")


def _date_segments(start: str, end: str, years: int) -> list[tuple[str, str]]:
    """按 N 年切分区间，保证每段不超过 640 个交易日。"""
    starts = pd.date_range(start=start, end=end, freq=f"{years}YS")
    segments: list[tuple[str, str]] = []
    for seg_start in starts:
        seg_end = min(seg_start + timedelta(days=365 * years) - timedelta(days=1),
                      pd.Timestamp(end))
        if seg_start > pd.Timestamp(end):
            break
        segments.append((seg_start.strftime("%Y-%m-%d"), seg_end.strftime("%Y-%m-%d")))

    if not segments:
        segments = [(start, end)]

    # 覆盖切分后遗漏的尾部
    last_end = segments[-1][1]
    if last_end < end:
        segments.append(
            (pd.Timestamp(last_end).strftime("%Y-%m-%d"), end)
        )
    return segments


def _parse_kline_rows(rows: list) -> pd.DataFrame:
    """解析日线行。

    腾讯返回的日线行宽并不稳定：多数是 6 列，偶发多带一列（成交额等）。
    这里统一截取前 6 列，避免硬性假设列数导致整批抓取失败。
    """
    cleaned: list[list] = []
    widths: set[int] = set()
    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        widths.add(len(row))
        cleaned.append(list(row[:6]))

    if not cleaned:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    if widths != {6}:
        logger.debug("日线行宽异常 %s，已统一截取前 6 列", sorted(widths))

    df = pd.DataFrame(cleaned, columns=KLINE_COLUMNS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in KLINE_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df.drop_duplicates(subset="date").sort_values("date")
    return df.reset_index(drop=True)


def fetch_kline(symbol: str, start: str, end: str, adjust: str = "qfq",
                timeout: int = 15, retries: int = 3, retry_sleep: float = 1.0,
                sleep: float = 0.12, segment_years: int = 2) -> pd.DataFrame:
    """抓取单只标的的前复权日线。

    Args:
        symbol: 带前缀代码，如 "sz000001"；也接受 6 位裸代码
        start / end: "YYYY-MM-DD"
        adjust: qfq 前复权 / hfq 后复权 / "" 不复权
    """
    symbol = normalize_symbol(symbol)
    frames: list[pd.DataFrame] = []

    for seg_start, seg_end in _date_segments(start, end, segment_years):
        params = {
            "param": f"{symbol},day,{seg_start},{seg_end},{MAX_ROWS_PER_REQUEST},{adjust}"
        }
        try:
            resp = _request(TENCENT_KLINE_URL, params, timeout, retries, retry_sleep)
        except RuntimeError as err:
            logger.warning("跳过分段 %s~%s: %s", seg_start, seg_end, err)
            continue

        payload = resp.json() or {}
        node = payload.get("data") or {}
        if not isinstance(node, dict):
            continue
        symbol_node = node.get(symbol) or {}
        rows = symbol_node.get("qfqday") or symbol_node.get("day") or []
        if not rows:
            continue

        frames.append(_parse_kline_rows(rows))
        time.sleep(sleep)

    if not frames:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    df.insert(0, "symbol", symbol)

    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df.loc[mask].reset_index(drop=True)


def fetch_index(symbol: str, start: str, end: str, **kwargs) -> pd.DataFrame:
    """抓取指数日线（指数无复权概念，强制不复权）。"""
    kwargs.pop("adjust", None)
    return fetch_kline(symbol, start, end, adjust="", **kwargs)


def fetch_stock_list(node: str = "hs_a", page_size: int = 100,
                     max_pages: int = 80, timeout: int = 15, retries: int = 3,
                     retry_sleep: float = 1.0, sleep: float = 0.12) -> pd.DataFrame:
    """抓取 A 股列表（新浪），含名称、市值、换手率。"""
    rows: list[dict] = []
    for page in range(1, max_pages + 1):
        params = {
            "page": page,
            "num": page_size,
            "sort": "symbol",
            "asc": 1,
            "node": node,
        }
        try:
            resp = _request(SINA_LIST_URL, params, timeout, retries, retry_sleep)
        except RuntimeError as err:
            logger.warning("列表抓取失败 page=%s: %s", page, err)
            break

        text = (resp.text or "").strip()
        if not text or text in ("null", "[]"):
            break
        try:
            batch = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("列表第 %s 页解析失败，停止翻页", page)
            break

        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        time.sleep(sleep)

    if not rows:
        return pd.DataFrame(columns=["symbol", "code", "name"])

    df = pd.DataFrame(rows)
    keep = [c for c in ["symbol", "code", "name", "trade", "mktcap", "nmc",
                        "turnoverratio", "changepercent"] if c in df.columns]
    df = df[keep].copy()

    # 新浪返回的数值字段是字符串（如 "0.000"），不转会在落 Parquet 时炸
    for col in ["trade", "mktcap", "nmc", "turnoverratio", "changepercent"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)
