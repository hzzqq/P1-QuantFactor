"""数据落盘层。

目录约定：
    data/P1/raw/kline/{symbol}.parquet      每只股票一个文件，便于增量更新
    data/P1/processed/panel.parquet         合并后的面板，特征计算输入
    data/P1/meta/fetch_state.json           抓取进度，支持断点续传
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd

from shared import paths
from shared.logging_utils import get_logger

logger = get_logger("P1.data.storage")

RAW_DIR = paths.DATA / "P1" / "raw" / "kline"
PROCESSED_DIR = paths.DATA / "P1" / "processed"
META_DIR = paths.DATA / "P1" / "meta"

STATE_FILE = META_DIR / "fetch_state.json"
PANEL_FILE = PROCESSED_DIR / "panel.parquet"
LIST_FILE = META_DIR / "stock_list.parquet"

# 并发抓取时会多线程同时读写 fetch_state.json，必须用锁保证 load→modify→save 原子，
# 否则互相截断会让 JSON 损坏、下一次 load 直接崩溃（已踩坑）。
_state_lock = threading.Lock()

for _d in (RAW_DIR, PROCESSED_DIR, META_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def kline_path(symbol: str) -> Path:
    return RAW_DIR / f"{symbol}.parquet"


def save_kline(df: pd.DataFrame, symbol: str) -> Path:
    """保存单只股票日线（覆盖写）。"""
    if df.empty:
        raise ValueError(f"空数据，拒绝写入: {symbol}")
    path = kline_path(symbol)
    df.to_parquet(path, index=False)
    return path


def load_kline(symbol: str) -> pd.DataFrame | None:
    """读取单只股票日线，不存在返回 None。"""
    path = kline_path(symbol)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def list_saved_symbols() -> list[str]:
    return sorted(p.stem for p in RAW_DIR.glob("*.parquet"))


def load_fetch_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as err:
        # 文件损坏（并发写被截断）时不要崩溃，降级为无进度从头续传
        logger.warning("fetch_state.json 解析失败，已忽略旧进度: %s", err)
        return {}


def save_fetch_state(state: dict) -> None:
    # 先写临时文件再原子替换，避免写到一半被另一线程读到半截 JSON
    tmp = STATE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


def update_fetch_state(symbol: str, last_date, rows: int) -> None:
    with _state_lock:
        state = load_fetch_state()
        state[symbol] = {
            "last_date": str(pd.Timestamp(last_date).date()),
            "rows": int(rows),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_fetch_state(state)


def save_panel(df: pd.DataFrame) -> Path:
    df.to_parquet(PANEL_FILE, index=False)
    logger.info("面板已保存: %s (%s 行)", PANEL_FILE, f"{len(df):,}")
    return PANEL_FILE


def load_panel() -> pd.DataFrame | None:
    if not PANEL_FILE.exists():
        return None
    return pd.read_parquet(PANEL_FILE)


def build_panel(symbols: list[str] | None = None) -> pd.DataFrame:
    """把所有 raw 文件合并成面板。"""
    symbols = symbols or list_saved_symbols()
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        df = load_kline(sym)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    # N16：raw kline 经桥接 append 可能含重复 (date,symbol) 行，写入前清掉，
    # 与 build_dataset 口径 (keep='last') 一致，避免下游 pivot / 回测崩溃、标签错乱。
    dup = int(panel.duplicated(subset=["date", "symbol"]).sum())
    if dup:
        logger.warning("build_panel 发现 %s 个重复 (date,symbol) 行，已去重(保留末条)", dup)
        panel = panel.drop_duplicates(subset=["date", "symbol"], keep="last")
    return panel


def save_stock_list(df: pd.DataFrame) -> Path:
    df.to_parquet(LIST_FILE, index=False)
    return LIST_FILE


def load_stock_list() -> pd.DataFrame | None:
    if not LIST_FILE.exists():
        return None
    return pd.read_parquet(LIST_FILE)
