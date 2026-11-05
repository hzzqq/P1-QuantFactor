"""一次性桥接（健壮版）：用 baostock 把 kline + 基准推进到 2026-09-15。

相对初版的加固（遵守项目「长任务禁令」四件套）：
- ① 硬超时：socket.setdefaulttimeout(30)
- ② 检查点落盘：跳过已到 09-15 的标的（幂等 + 可断点续跑）
- ③ 高频进度日志：每 50 只打印 + 末尾汇总
- ④ 优先拆短任务：支持 --limit/--offset 分批；baostock 会话掉线自动重登
- 源格式：fetcher 返 sh600000（无点），baostock 需 sh.600000（有点）→ to_bs()

2026-11-05 加固（baostock 账号被拉黑 10001011 后的兜底）：
- 登录结果必须校验：`bs.login()` 返回 error_code!=0 时**立即 gg**，不再把 1492 次 FAIL 打进日志、
  也不再密集重试（那正是把账号打进黑名单的诱因）。
- 新增 `--source {auto,baostock,sina}`：auto 下 baostock 登录失败自动切新浪源
  （akshare `stock_zh_a_daily`，前复权），**只补本地尾部**、不动已有历史；
  追加前用重叠日收盘做口径校验，相对差 >1% 视为复权基准不同（除权/送股）→ 该只拒写并报 BASIS_MISMATCH。
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


def _ak():
    """延迟导入 akshare（仅 sina 兜底路径需要，baostock 正常时不影响启动）。"""
    import akshare as ak
    return ak


def fetch_sina_tail(code: str, start_ts: "pd.Timestamp", local: "pd.DataFrame | None"):
    """新浪前复权取 code 自 start_ts 起的日线尾巴，并做重叠口径校验。

    返回 (df, reason)：reason ∈ {"ok", "empty", "basis_mismatch", "err:<msg>"}。
    校验用本地最后一根与新浪同一交易日的收盘做比较：新浪 qfq 是按**今天**复权的，
    若中间发生除权，同一历史日两套口径会差一个因子，直接拼接会在接缝处制造假跳空。
    """
    try:
        ak = _ak()
    except Exception as e:  # noqa: BLE001 - akshare 未装/导入失败要如实报，不静默
        return None, f"err:akshare 不可用 {e!r}"
    sd = (start_ts - pd.Timedelta(days=10)).strftime("%Y%m%d")
    ed = pd.Timestamp(END).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_daily(symbol=code, start_date=sd, end_date=ed, adjust="qfq")
    except Exception as e:  # noqa: BLE001 - 单只失败进 FAIL，不中断整轮
        return None, f"err:{e!r}"
    if df is None or not len(df):
        return None, "empty"
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if local is not None and len(local):
        last = local.iloc[-1]
        hit = df[df["date"] == last["date"]]
        if len(hit):
            ak_close = float(hit.iloc[0]["close"])
            bs_close = float(last["close"])
            if bs_close and abs(ak_close - bs_close) / abs(bs_close) > 0.01:
                return None, f"basis_mismatch(bs={bs_close:.3f},ak={ak_close:.3f})"
    df["symbol"] = code.replace(".", "")
    out = df[["symbol", "date", "open", "close", "high", "low", "volume"]].copy()
    return out, "ok"


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


def probe_latest_available_sina(fallback: "pd.Timestamp | None" = None):
    """新浪源同款探针（sh000300 指数），口径与 baostock 版一致：
    失败不回落 today，回落缓存/已落盘真实最大交易日。"""
    try:
        ak = _ak()
        idx = ak.stock_zh_index_daily(symbol="sh000300")
        if idx is not None and len(idx):
            dt = pd.to_datetime(idx["date"]).max()
            try:
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                CACHE_FILE.write_text(str(dt.date()))
            except Exception:  # noqa: BLE001 - 缓存写失败不影响主流程
                pass
            return dt
    except Exception as e:  # noqa: BLE001 - 探针失败走回退，但要留痕
        print(f"probe(sina) 失败: {e!r}", flush=True)
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
    ap.add_argument("--source", choices=["auto", "baostock", "sina"], default="auto",
                    help="auto=优先 baostock，登录失败切 sina；baostock 失败即退出不写数据")
    args = ap.parse_args()

    socket.setdefaulttimeout(30)
    symbols = storage.list_saved_symbols()

    bs_ok = False
    if args.source in ("auto", "baostock"):
        lg = bs.login()
        bs_ok = lg.error_code == "0"
        if not bs_ok:
            print(f"BAOSTOCK LOGIN FAIL rc={lg.error_code} msg={lg.error_msg}", flush=True)
            if args.source == "baostock":
                print("BRIDGE ABORT: baostock 不可用，本轮不写任何数据（避免再打黑名单）", flush=True)
                return 2
            print("切换新浪源（--source auto）：仅补本地尾部，不改历史", flush=True)

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
    TARGET = (probe_latest_available(fallback=saved_max) if bs_ok
              else probe_latest_available_sina(fallback=saved_max))

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
            if bs_ok:
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
                        except Exception:  # noqa: BLE001 - 登出失败不影响重登
                            pass
                        bs.login()
                        consec_fail = 0
            else:
                local = storage.load_kline(code)
                if local is not None and len(local):
                    start = local["date"].max() + pd.Timedelta(days=1)
                else:
                    start = pd.Timestamp(START)
                tail, reason = fetch_sina_tail(code, start, local)
                if tail is not None and len(tail):
                    # 日期统一成同一 dtype 再拼接：本地 parquet 是 datetime64[us]、
                    # akshare 返回 datetime.date 对象，混用会让 drop_duplicates 认不出「同一天」→ 写出重复行
                    merged = pd.concat([local, tail])
                    merged["date"] = pd.to_datetime(merged["date"])
                    merged = (merged.drop_duplicates(subset=["date"], keep="first")
                              .sort_values("date"))
                    storage.save_kline(merged, code.replace(".", ""))
                    ok += 1
                else:
                    fail += 1
                    print(f"FAIL {code} {reason}", flush=True)
            if i % 50 == 0:
                print(f"progress {i}/{len(todo)} ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s", flush=True)
            time.sleep(0.4)  # 限速：请求间留间隙，避免触发限流墙
        print(f"KLINE DONE ok={ok} fail={fail}", flush=True)

        # 基准沪深300（超额收益标签所必需）
        if bs_ok:
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
        else:
            # 新浪源基准：与个股同口径只补尾部，已有历史原样保留（避免整段改写基准）
            try:
                ak = _ak()
                b = ak.stock_zh_index_daily(symbol="sh000300")
                b = b[["date", "open", "high", "low", "close", "volume"]].copy()
                b["date"] = pd.to_datetime(b["date"])
                idx = ROOT / "data" / "P1" / "raw" / "index"
                idx.mkdir(parents=True, exist_ok=True)
                fp = idx / "sh000300.parquet"
                if fp.exists():
                    old = pd.read_parquet(fp)
                    old["date"] = pd.to_datetime(old["date"])
                    b = (pd.concat([old, b]).drop_duplicates(subset=["date"], keep="first")
                         .sort_values("date"))
                b.to_parquet(fp, index=False)
                print(f"BENCH rows {len(b)} max {b['date'].max()}", flush=True)
            except Exception as e:
                print(f"BENCH ERR {e!r}", flush=True)
        print("BRIDGE ALL DONE", flush=True)
        return 0
    finally:
        if bs_ok:
            try:
                bs.logout()
            except Exception:  # noqa: BLE001 - 收尾登出失败不影响结果
                pass


if __name__ == "__main__":
    raise SystemExit(main())
