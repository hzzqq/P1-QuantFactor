"""阶段 31b - 用 tushare 拉真实逐日换手率/流通股本，替代 baostock（后者服务端故障+挂死）。

产出与 31_baostock_fetch 同 schema 的 baostock_daily.parquet，使下游 32_derive_real_factors.py
**零改动复用**：
    symbol(str, sh600000) / date / close / volume(手) / turn(换手率%) / float_share(流通股本,股)
    / peTTM / pbMRQ
数据来自 tushare daily_basic（每交易日一次调用返回全市场）：
    - turnover_rate → turn（换手率%）
    - vol           → volume（手）
    - close
    - float_share   → 流通股本（股）
    - pe / pb       → peTTM / pbMRQ

token 从环境变量 TUSHARE_TOKEN 读取（**绝不落盘 / 不提交**）。

健壮性（吸取 baostock 教训）：
    - 每交易日一次调用，主循环 sleep 限流（默认 0.3s，约 200 调用/分，远低于 tushare 上限）。
    - 单次调用套**线程级超时**（默认 30s）+ 重试 3 次，避免任何后端挂死拖垮整轮。
    - **断点续跑**：每交易日结果落盘为 tmp/{trade_date}.parquet，重启自动跳过已完成的日期。
    - --dry：只拉 1 个交易日验证 token + 列结构，不拉全量。

用法：
    export TUSHARE_TOKEN=你的token
    python scripts/31b_tushare_fetch.py --mode stock --start 2015-01-27 --end 2026-08-14
    python scripts/31b_tushare_fetch.py --dry                 # 连通性 + 列校验
    python scripts/31b_tushare_fetch.py --mode stock --start 2015-01-27 --end 2026-08-14 --max-symbols 50  # 抽样验证用
    python scripts/31b_tushare_fetch.py --mode date --start 2023-01-01 --end 2026-08-14 --max-dates 800    # 仅高积分账号
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import threading
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

OUT = paths.DATA / "P1" / "raw" / "baostock_daily" / "baostock_daily.parquet"
TMP = paths.DATA / "P1" / "raw" / "baostock_daily" / "_tushare_tmp"
TMP_STOCK = paths.DATA / "P1" / "raw" / "baostock_daily" / "_tushare_tmp_stock"

FIELDS = "ts_code,trade_date,close,turnover_rate,vol,amount,float_share,pe,pb"
CALL_TIMEOUT = 30.0
RETRY = 3


def _load_pro(token: str):
    import tushare as ts
    ts.set_token(token)
    return ts.pro_api(token)


def _call_with_timeout(pro, trade_date: str, timeout: float):
    """线程级超时保护（tushare 底层 requests 偶发挂死时也能回收）。返回 DataFrame 或 None。"""
    box: dict = {}

    def _w():
        try:
            box["df"] = pro.daily_basic(trade_date=trade_date, fields=FIELDS)
        except Exception as e:  # noqa: BLE001
            box["exc"] = repr(e)[:160]

    th = threading.Thread(target=_w, daemon=True)
    th.start()
    th.join(timeout=timeout)
    if th.is_alive():
        return None
    if "exc" in box:
        return ("EXC", box["exc"])
    return box.get("df")


def _to_symbol(ts_code: str) -> str:
    code, ex = ts_code.split(".")
    return ex.lower() + code


def _map_chunk(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "symbol": df["ts_code"].map(_to_symbol).astype(str),
        "date": pd.to_datetime(df["trade_date"], format="%Y%m%d"),
        "close": pd.to_numeric(df["close"], errors="coerce").astype(np.float32),
        "volume": pd.to_numeric(df["vol"], errors="coerce").astype(np.float32),
        "turn": pd.to_numeric(df["turnover_rate"], errors="coerce").astype(np.float32),
        "float_share": pd.to_numeric(df["float_share"], errors="coerce").astype(np.float64),
        "peTTM": pd.to_numeric(df["pe"], errors="coerce").astype(np.float32),
        "pbMRQ": pd.to_numeric(df["pb"], errors="coerce").astype(np.float32),
    })
    return out


def _parse_rate_gap(msg: str) -> float | None:
    """从 tushare 频率限制消息解析需要的冷却秒数；无法解析返回 None。

    实测消息两种形态：
        "...频率超限(1次/分钟)..."   -> 60s
        "...频率超限(1次/小时)..."   -> 3600s
    """
    import re
    m = re.search(r"(\d+)\s*次/小时", msg)
    if m:
        return 3600.0
    m = re.search(r"(\d+)\s*次/分钟", msg)
    if m:
        return 60.0
    if "每小时" in msg:
        return 3600.0
    if "每分钟" in msg:
        return 60.0
    return None


def _run_units(pro, units, tmp: Path, call_fn, sleep: float, label: str):
    """自适应限速调度器：按 units 逐单元调用 call_fn，成功率限制自动退避到对应周期。

    call_fn(pro, u) 返回：DataFrame / ("EXC", msg) / None（与 _call_with_timeout 同约）。
    断点续跑：已落盘 {u}.parquet 的单元自动跳过。
    """
    done = skip = fail = 0
    tmp.mkdir(parents=True, exist_ok=True)
    gap = float(sleep)          # 当前认定的最小调用间隔（自适应更新）
    next_call = time.time() + 5  # 起手先清 5s 余量
    for i, u in enumerate(units):
        part = tmp / f"{u}.parquet"
        if part.exists():
            skip += 1
            continue
        ok = False
        for _ in range(RETRY):
            wait = max(0.0, next_call - time.time())
            if wait > 0:
                time.sleep(wait)
            res = call_fn(pro, u)
            now = time.time()
            if isinstance(res, tuple) and res[0] == "EXC":
                msg = res[1]
                g = _parse_rate_gap(msg)
                if g:
                    gap = g + 20.0  # 留 20s 余量
                    print(f"    [RATELIMIT {u}] {msg[:70]} → 退避 {gap:.0f}s", flush=True)
                    next_call = now + gap
                    time.sleep(gap)
                    continue
                print(f"    [WARN {u}] exc {msg}", flush=True)
                time.sleep(3)
                continue
            if res is None or (isinstance(res, pd.DataFrame) and res.empty):
                time.sleep(3)
                continue
            try:
                _map_chunk(res).to_parquet(part, index=False)
            except Exception as e:  # noqa: BLE001
                print(f"    [WARN {u}] map err {repr(e)[:120]}", flush=True)
                time.sleep(3)
                continue
            ok = True
            next_call = now + gap  # 成功后，下一单元至少间隔 gap
            break
        if ok:
            done += 1
        else:
            fail += 1
            print(f"    [FAIL] {u} 重试 {RETRY} 次仍失败", flush=True)
        if (i + 1) % 10 == 0:
            print(f"[{label}] 进度 {i+1}/{len(units)} 完成 {done} 跳过 {skip} 失败 {fail}", flush=True)
    return done, skip, fail


def _fetch_range(pro, start: str, end: str, sleep: float, max_dates: int | None, tmp: Path = TMP):
    cal = pro.trade_cal(exchange="SSE", start_date=start.replace("-", ""),
                        end_date=end.replace("-", ""), is_open="1")
    dates = sorted(cal["cal_date"].tolist())
    if max_dates:
        dates = dates[:max_dates]
    print(f"[31b] 区间 {start}~{end} 共 {len(dates)} 个交易日", flush=True)
    return _run_units(pro, dates, tmp,
                      lambda p, u: _call_with_timeout(p, u, CALL_TIMEOUT),
                      sleep, "31b-date")


def _load_symbols_from_v39() -> list[str]:
    """从本地 v39 数据集读取股票列表（零 API 消耗），转成 tushare ts_code 格式。"""
    p = paths.DATA / "P1" / "processed" / "dataset_h10_v39.parquet"
    df = pd.read_parquet(p, columns=["symbol"])
    syms = sorted(df["symbol"].astype(str).unique().tolist())
    out = []
    for s in syms:  # sh600000 -> 600000.SH
        ex = s[:2].upper()
        code = s[2:]
        out.append(f"{code}.{ex}")
    return out


def _call_one_symbol(pro, ts_code: str, sd: str, ed: str, timeout: float):
    """按股票拉全历史（绕开 trade_cal 的 1次/小时 限制）。返回 DataFrame / ("EXC",msg) / None。"""
    box: dict = {}

    def _w():
        try:
            box["df"] = pro.daily_basic(ts_code=ts_code, start_date=sd, end_date=ed, fields=FIELDS)
        except Exception as e:  # noqa: BLE001
            box["exc"] = repr(e)[:160]

    th = threading.Thread(target=_w, daemon=True)
    th.start()
    th.join(timeout=timeout)
    if th.is_alive():
        return None
    if "exc" in box:
        return ("EXC", box["exc"])
    return box.get("df")


def _fetch_by_ts_code(pro, start: str, end: str, sleep: float, max_symbols: int | None, tmp: Path = TMP_STOCK):
    syms = _load_symbols_from_v39()
    if max_symbols:
        syms = syms[:max_symbols]
    print(f"[31b] 按股票模式：{len(syms)} 只（列表来自本地 v39，不调用 trade_cal）", flush=True)
    sd, ed = start.replace("-", ""), end.replace("-", "")
    return _run_units(pro, syms, tmp,
                      lambda p, u: _call_one_symbol(p, u, sd, ed, CALL_TIMEOUT),
                      sleep, "31b-stock")


def _combine(tmp: Path = TMP):
    parts = sorted(tmp.glob("*.parquet"))
    if not parts:
        print("[31b] 无分片可合并")
        return 1
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df = df.dropna(subset=["symbol", "date"]).drop_duplicates(subset=["date", "symbol"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"[31b] 合并 {len(parts)} 片 → {OUT.name} ({len(df):,} 行, "
          f"{df['symbol'].nunique()} 只, {df['date'].min().date()}~{df['date'].max().date()})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-27")
    ap.add_argument("--end", default="2026-08-14")
    ap.add_argument("--sleep", type=float, default=65.0,
                     help="调用间隔秒数；免费/低积分账号 daily_basic 限 1次/分钟，默认 65s 以留余量")
    ap.add_argument("--max-dates", type=int, default=None)
    ap.add_argument("--max-symbols", type=int, default=None,
                     help="按股票模式下的抽样只数；不传则全量 1427 只")
    ap.add_argument("--mode", choices=["date", "stock"], default="stock",
                     help="date=按交易日拉全市场(依赖 trade_cal，免费账号 1次/小时 易限)；"
                          "stock=按股票拉全历史(绕开 trade_cal，推荐)")
    ap.add_argument("--token-env", default="TUSHARE_TOKEN")
    ap.add_argument("--dry", action="store_true", help="只拉 1 个交易日验证 token+列")
    args = ap.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        print(f"[31b] 未找到环境变量 {args.token_env}。请先 export TUSHARE_TOKEN=你的token")
        return 1
    pro = _load_pro(token)

    if args.dry:
        print("[31b] --dry：拉 1 个交易日校验 token + 列结构")
        res = _call_with_timeout(pro, "20240102", CALL_TIMEOUT)
        if res is None:
            print("[31b] DRY FAIL：调用超时（token 可能有效但网络/服务端异常）")
            return 1
        if isinstance(res, tuple):
            print(f"[31b] DRY FAIL：{res[1]}")
            return 1
        out = _map_chunk(res)
        print(f"[31b] DRY OK：{len(out)} 行；列={out.columns.tolist()}")
        print(out.head(3).to_string())
        return 0

    t0 = time.time()
    if args.mode == "stock":
        tmp = TMP_STOCK
        done, skip, fail = _fetch_by_ts_code(
            pro, args.start, args.end, args.sleep, args.max_symbols, tmp)
    else:
        tmp = TMP
        done, skip, fail = _fetch_range(
            pro, args.start, args.end, args.sleep, args.max_dates, tmp)
    print(f"[31b] 拉取完成 完成 {done} 跳过 {skip} 失败 {fail} "
          f"{time.time()-t0:.0f}s", flush=True)
    if fail:
        print(f"[31b] ⚠️ {fail} 个单元失败，合并时将缺失（下游 32 覆盖率护栏会拦截不全）")
    rc = _combine(tmp)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
