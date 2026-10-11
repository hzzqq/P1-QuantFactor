"""一次性桥接（健壮版）：用 baostock 把 kline + 基准推进到 2026-09-15。

相对初版的加固（遵守项目「长任务禁令」四件套）：
- ① 硬超时：socket.setdefaulttimeout(30)
- ② 检查点落盘：跳过已到 09-15 的标的（幂等 + 可断点续跑）
- ③ 高频进度日志：每 50 只打印 + 末尾汇总
- ④ 优先拆短任务：支持 --limit/--offset 分批；baostock 会话掉线自动重登
- 源格式：fetcher 返 sh600000（无点），baostock 需 sh.600000（有点）→ to_bs()
"""
from __future__ import annotations

import argparse
import re
import socket
import sys
import time
from pathlib import Path

ROOT = Path(r"E:/project/sj")
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import baostock as bs  # noqa: E402
import pandas as pd  # noqa: E402
from src.data import storage  # noqa: E402

# 动态指向"最新交易日"：END 取运行当日，baostock 只会返回到实际可拿到的最后交易日，
# 断点续跑据此判断 symbol 是否已到最新（修复原硬编码 2026-09-15 导致无法增量到更新日期的问题）。
END = time.strftime("%Y-%m-%d")
START = "2018-01-01"
TARGET = pd.Timestamp(END)
FIELDS = "date,open,high,low,close,volume"
RELOGIN_FAILS = 20  # 连续失败达到此数 → 重登


def to_bs(code: str) -> str:
    if "." in code:
        return code
    m = re.match(r"^([a-z]{2})(\d+)$", code)
    return f"{m.group(1)}.{m.group(2)}" if m else code


def fetch_one(code: str):
    """拉取单只。baostock 限流响应（error_msg 含"请稍后再试"/"接收数据异常"）
    不算异常、原实现会直接判 FAIL 并密集重试加剧限流——这里改为指数退避重试，
    禁止密集重试。"""
    bs_code = to_bs(code)
    backoff = 5
    for _ in range(8):
        try:
            rs = bs.query_history_k_data_plus(bs_code, FIELDS, START, END, "d", "2")
            if rs.error_code == "0":
                rows = []
                while rs.error_code == "0" and rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
                    for c in ["open", "high", "low", "close", "volume"]:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    df["date"] = pd.to_datetime(df["date"])
                    df["symbol"] = code.replace(".", "")
                    return df[["symbol", "date", "open", "close", "high", "low", "volume"]]
                return None  # 空数据（退市/无交易），非限流
            msg = rs.error_msg or ""
            if "请稍后再试" in msg or "接收数据异常" in msg or "网络" in msg:
                # 限流：降速退避，避免密集重试
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            return None
        except Exception:
            time.sleep(3)
    return None


CACHE_FILE = PROJ / "logs" / ".last_latest_available.txt"


def _read_cached_latest():
    try:
        if CACHE_FILE.exists():
            return pd.to_datetime(CACHE_FILE.read_text().strip())
    except Exception:
        pass
    return None


def probe_latest_available(fallback: "pd.Timestamp | None" = None):
    """探明 baostock 实际可服务的最新交易日（可能早于"今天"：盘中/节假日/服务端延迟）。
    用流动性最好的沪深300做探针；**失败时禁止回退到 today**（节假日/盘中 today 是非交易日，
    会导致 TARGET 失真、所有 symbol 被误判为落后而全量重刷，违反断点续跑），而是回退到
    fallback（已落盘行情的真实最大交易日）或上次成功探测的缓存值。"""
    backoff = 4
    for _ in range(6):
        try:
            rs = bs.query_history_k_data_plus("sh.000300", FIELDS, "2026-01-01", END, "d", "2")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                dt = pd.to_datetime([r[0] for r in rows]).max()
                try:
                    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                    CACHE_FILE.write_text(str(dt.date()))
                except Exception:
                    pass
                return dt
            # 0 行（节假日/网络错误）→ 不在此处回退到 today，交给下方 fallback
        except Exception:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)
    cached = _read_cached_latest()
    if cached is not None:
        print(f"probe 失败，回退到缓存的最新交易日 {cached.date()}", flush=True)
        return cached
    if fallback is not None:
        return fallback
    return pd.Timestamp(END)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    socket.setdefaulttimeout(30)
    symbols = storage.list_saved_symbols()

    bs.login()

    # 先扫描已落盘行情的真实最大交易日，作为 probe 失败时的 fallback（保证断点续跑正确，
    # 避免 baostock 宕机/节假日时把 TARGET 误设为非交易日的 today 而触发全量重刷）。
    # 同时构建 symbol→maxdate 映射，供下方断点续跑复用，省去二次 1492 次加载。
    saved_max = None
    sym_max = {}
    for s in symbols:
        df = storage.load_kline(s)
        if df is not None and "date" in df.columns and len(df):
            dmax = df["date"].max()
            sym_max[s] = dmax
            if saved_max is None or dmax > saved_max:
                saved_max = dmax
    if saved_max is not None:
        print(f"saved_global_max={saved_max.date()} (probe 失败时的回退阈值)", flush=True)

    global TARGET
    TARGET = probe_latest_available(fallback=saved_max)

    # 断点续跑：跳过已到最新交易日的标的（复用已扫描的 sym_max，避免二次加载）
    todo = []
    skipped = 0
    for s in symbols:
        if s in sym_max and sym_max[s] >= TARGET:
            skipped += 1
        else:
            todo.append(s)
    print(f"latest_available={TARGET.date()} total={len(symbols)} skipped(已到{TARGET.date()})={skipped} todo={len(todo)}", flush=True)

    if args.offset:
        todo = todo[args.offset:]
    if args.limit:
        todo = todo[: args.limit]
    print(f"will refresh this run: {len(todo)}", flush=True)

    try:
        ok = fail = 0
        consec_fail = 0
        t0 = time.time()
        for i, code in enumerate(todo, 1):
            df = fetch_one(code)
            if df is not None and len(df):
                storage.save_kline(df, code.replace(".", ""))
                ok += 1
                consec_fail = 0
            else:
                fail += 1
                consec_fail += 1
                print(f"FAIL {code}", flush=True)
                if consec_fail >= RELOGIN_FAILS:
                    print("连续失败过多，重登 baostock...", flush=True)
                    try:
                        bs.logout()
                    except Exception:
                        pass
                    bs.login()
                    consec_fail = 0
            if i % 50 == 0:
                print(f"progress {i}/{len(todo)} ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s", flush=True)
            time.sleep(0.4)  # 限速：请求间留间隙，避免触发限流墙
        print(f"KLINE DONE ok={ok} fail={fail}", flush=True)

        # 基准沪深300（超额收益标签所必需）
        try:
            rs = bs.query_history_k_data_plus("sh.000300", FIELDS, START, END, "d", "2")
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rows:
                b = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
                for c in ["open", "high", "low", "close", "volume"]:
                    b[c] = pd.to_numeric(b[c], errors="coerce")
                b["date"] = pd.to_datetime(b["date"])
                idx = ROOT / "data" / "P1" / "raw" / "index"
                idx.mkdir(parents=True, exist_ok=True)
                b.to_parquet(idx / "sh000300.parquet", index=False)
                print(f"BENCH rows {len(b)} max {b['date'].max()}", flush=True)
        except Exception as e:
            print(f"BENCH ERR {e!r}", flush=True)
        print("BRIDGE ALL DONE", flush=True)
        return 0
    finally:
        try:
            bs.logout()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
