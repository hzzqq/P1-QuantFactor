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

END = "2026-09-15"
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
    bs_code = to_bs(code)
    for _ in range(4):
        try:
            rs = bs.query_history_k_data_plus(bs_code, FIELDS, START, END, "d", "2")
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
            return None
        except Exception:
            time.sleep(3)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    args = ap.parse_args()

    socket.setdefaulttimeout(30)
    symbols = storage.list_saved_symbols()
    # 断点续跑：跳过已到目标日的标的
    todo = []
    skipped = 0
    for s in symbols:
        df = storage.load_kline(s)
        if df is not None and "date" in df.columns and df["date"].max() >= TARGET:
            skipped += 1
        else:
            todo.append(s)
    print(f"total={len(symbols)} skipped(已到{TARGET.date()})={skipped} todo={len(todo)}", flush=True)

    if args.offset:
        todo = todo[args.offset:]
    if args.limit:
        todo = todo[: args.limit]
    print(f"will refresh this run: {len(todo)}", flush=True)

    bs.login()
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
