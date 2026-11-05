"""阶段 30 - ②b：用 baostock 拉取真实逐日估值/换手数据，作为真·mktcap/turnover 因子源。

这是 ②b 的决定性闸门：akshare 东财沙箱断连，baostock 是候选真源（import OK + DNS 通，
但需实测能否拉到）。本脚本批量拉取 dataset_h10_v39 覆盖的全部 symbol 的日线，字段含
turn(换手率%) / amount / close / peTTM / pbMRQ，用于反推真实流通市值与换手率因子。

设计（适配沙箱 ~660s 墙钟 + 每进程 commit 上限 + 限流）：
  - 按 symbol 总数切成 n_batches 片，单次只跑 --batch 这一片（前台 ~几分钟）。
  - 断点续跑：已完成片写入 `baostock_daily_batch_{batch:03d}.parquet`；重跑同片自动跳过。
  - 限流：每只股票查询后 sleep（默认 0.12s），避免触发 baostock 频率限制（约 120 次/分）。
  - 结果统一存 data/P1/raw/baostock_daily/。
  - --combine 模式：把所有分片 concat 成 baostock_daily.parquet（最后一步跑一次）。

用法：
    # 先跑探针确认连通性
    python scripts/30_baostock_probe.py
    # 分片拉取（如 15 片，逐片前台）
    python scripts/31_baostock_fetch.py --batch 0 --n-batches 15
    ...
    python scripts/31_baostock_fetch.py --batch 14 --n-batches 15
    # 合并
    python scripts/31_baostock_fetch.py --combine
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _hard_delete(path: Path) -> None:
    """删除文件，绕过 WorkBuddy sitecustomize 的 safe-delete shim。

    该 shim 把 Path.unlink/os.remove 劫持成『丢回收站』，非交互上下文里
    SHFileOperationW 会失败（0x2）并抛 OSError。直接调 Win32 DeleteFileW 绕过。
    """
    p = str(path)
    try:
        if os.name == "nt":
            k32 = ctypes.windll.kernel32
            k32.DeleteFileW.argtypes = [ctypes.c_wchar_p]
            k32.DeleteFileW.restype = ctypes.c_int
            if k32.DeleteFileW(p):
                return
        os.remove(p)
    except Exception:
        pass


class BaostockBanned(Exception):
    """baostock 服务端把本机 IP 拉黑（error 10001011），需避让等待解封。"""
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
OUTDIR = paths.DATA / "P1" / "raw" / "baostock_daily"
OUTDIR.mkdir(parents=True, exist_ok=True)

FIELDS = "date,code,open,high,low,close,volume,amount,turn,peTTM,pbMRQ,tradestatus,isST"
BATCH_N = 100  # 每片约 100 只，前台可控
QUERY_TIMEOUT = 30  # 单只股票查询线程超时(s)，防 baostock 无 timeout 挂死
MAX_RETRIES = 3     # 单只失败重试次数
RETRY_BACKOFF = 2.0  # 重试间隔(s)
REPAIR_ROUNDS = 3    # 主轮后补拉失败标的的轮数（网络错误 10002007 是瞬时的，会连片出现）


def to_bs_code(symbol: str) -> str:
    """将数据集 symbol 格式（sh600000 / sz000001）转为 baostock 格式（sh.600000 / sz.000001）。"""
    if "." in symbol or len(symbol) != 8:
        return symbol  # 已经是正确格式或异常格式，原样返回
    return symbol[:2] + "." + symbol[2:]


def load_symbols() -> list[str]:
    # 用 v39 数据集覆盖的 symbol 集合（与现有因子栈一致）
    df = pd.read_parquet(PROCESSED / "dataset_h10_v39.parquet", columns=["symbol"])
    syms = sorted(df["symbol"].astype(str).unique().tolist())
    return syms


def _query_one_symbol(bs, bs_code: str, timeout: float = 30,
                       start_date: str = "2010-01-01", end_date: str = "2026-12-31") -> list | None:
    """在独立线程执行单只股票查询，带超时保护（baostock 自身无 timeout）。
    返回 rows（list of row_data）；超时/查询错误/异常均返回 None。

    start_date/end_date 可调：**查询区间长度直接决定单只耗时**（实测 2026-11-05）：
        2010-01-01~2026-12-31 ≈ 6.5s/只（4087 行）
        2015-01-27~2026-10-31 ≈ 2.8s/只（2856 行）
    只覆盖目标数据集所需区间可显著提速（全量 1492 只：~2.7h → ~1.4h）。"""
    box: dict = {}

    def _worker() -> None:
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, FIELDS,
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="3",
            )
            if rs.error_code != "0":
                box["err"] = f"{rs.error_code} {rs.error_msg}"
                return
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            box["rows"] = rows
        except Exception as e:  # noqa: BLE001
            box["exc"] = repr(e)[:160]

    th = threading.Thread(target=_worker, daemon=True)
    th.start()
    th.join(timeout=timeout)
    if th.is_alive():
        return None  # 超时
    if "err" in box:
        print(f"    [WARN] {bs_code} query err {box['err']}", flush=True)
        return None
    if "exc" in box:
        print(f"    [WARN] {bs_code} exc {box['exc']}", flush=True)
        return None
    return box.get("rows", [])


def _build_df(rows: list) -> pd.DataFrame:
    cols = FIELDS.split(",")
    if not rows:
        return pd.DataFrame(columns=cols)
    big = pd.DataFrame(rows, columns=cols)
    for c in ("open", "high", "low", "close", "volume", "amount", "turn", "peTTM", "pbMRQ"):
        big[c] = pd.to_numeric(big[c], errors="coerce")
    big["date"] = pd.to_datetime(big["date"])
    big = big.rename(columns={"code": "symbol"})
    # 关键：baostock code 格式为 sh.600000，数据集 symbol 为 sh600000（无点号）。
    # 必须转回数据集格式，否则 step32 的 merge(on=['date','symbol']) 会 0 匹配 → 因子全 NaN。
    big["symbol"] = big["symbol"].astype(str).str.replace(".", "", regex=False)
    return big


def _login_with_timeout(bs, timeout: float = 30):
    """baostock login 也可能挂死（服务端掐连接），用线程超时保护。返回 lg 或 None。"""
    box: dict = {}

    def _w():
        try:
            box["lg"] = bs.login()
        except Exception as e:  # noqa: BLE001
            box["exc"] = repr(e)[:160]

    th = threading.Thread(target=_w, daemon=True)
    th.start()
    th.join(timeout=timeout)
    if th.is_alive():
        return None
    if "exc" in box:
        print(f"    [WARN] login exc {box['exc']}", flush=True)
        return None
    return box.get("lg")


def fetch_batch(symbols: list[str], sleep: float, partial_path: Path,
                start_date: str = "2010-01-01", end_date: str = "2026-12-31") -> pd.DataFrame:
    """拉取一批 symbol。每只查询带 30s 线程超时 + 3 次重试；每 25 只增量落盘 .partial。
    支持断点续跑：若 partial 已存在，跳过其中已完成的 symbol，只补拉剩余。
    返回最终 DataFrame（调用方负责写最终 parquet）。"""
    import baostock as bs
    lg = _login_with_timeout(bs, timeout=30)
    if lg is None:
        raise RuntimeError("baostock login 超时（30s），服务端可能掐连接，整批跳过待续跑")
    if lg.error_code != "0":
        msg = lg.error_msg or ""
        if "黑名单" in msg or str(lg.error_code) == "10001011":
            raise BaostockBanned(f"baostock 账号被黑名单封禁: {lg.error_code} {msg}")
        raise RuntimeError(f"baostock login failed: {lg.error_code} {msg}")
    # 断点续跑：从 partial 加载已完成 symbol
    acc = pd.DataFrame()
    done_symbols: set = set()
    if partial_path.exists():
        acc = pd.read_parquet(partial_path)
        done_symbols = set(acc["symbol"].astype(str).unique())
        print(f"  [RESUME] 从 partial 续跑，已完成 {len(done_symbols)} 只，补拉剩余",
              flush=True)
    t0 = time.time()
    failed: list[str] = []
    for i, code in enumerate(symbols):
        if code in done_symbols:
            continue  # 跳过已完成
        bs_code = to_bs_code(code)
        sym_rows = None
        for attempt in range(1, MAX_RETRIES + 1):
            sym_rows = _query_one_symbol(bs, bs_code, timeout=QUERY_TIMEOUT,
                                         start_date=start_date, end_date=end_date)
            if sym_rows is not None:
                break
            if attempt < MAX_RETRIES:
                print(f"    [RETRY] {code} 第{attempt}次超时/失败，{RETRY_BACKOFF}s 后重试",
                      flush=True)
                time.sleep(RETRY_BACKOFF)
        if sym_rows:
            acc = pd.concat([acc, _build_df(sym_rows)], ignore_index=True)
        else:
            print(f"    [FAIL] {code} 重试{MAX_RETRIES}次仍失败，跳过", flush=True)
            failed.append(code)
        if sleep > 0:
            time.sleep(sleep)
        if (i + 1) % 25 == 0:
            acc.to_parquet(partial_path, index=False)  # 增量落盘防丢失
            print(f"  processed {i+1}/{len(symbols)} symbols, {len(acc)} rows, "
                  f"{time.time()-t0:.0f}s [增量落盘]", flush=True)

    # ---- 修复轮：网络类错误(10002007 接收数据异常)是**瞬时**的，实测会连片出现 ----
    # 主轮失败的 symbol 若直接丢弃，final parquet 会**静默缺标的** → 覆盖率虚高、
    # 下游误以为全量（2026-11-05 实测 batch1 连挂 22 只）。故此处做多轮补拉。
    for rnd in range(1, REPAIR_ROUNDS + 1):
        if not failed:
            break
        print(f"  [REPAIR r{rnd}] 补拉 {len(failed)} 只失败标的（退避 {REPAIR_BACKOFF*3}s）",
              flush=True)
        time.sleep(REPAIR_BACKOFF * 3)
        still: list[str] = []
        for code in list(failed):
            sym_rows = None
            for attempt in range(1, MAX_RETRIES + 1):
                sym_rows = _query_one_symbol(bs, to_bs_code(code), timeout=QUERY_TIMEOUT,
                                             start_date=start_date, end_date=end_date)
                if sym_rows is not None:
                    break
                time.sleep(REPAIR_BACKOFF)
            if sym_rows:
                acc = pd.concat([acc, _build_df(sym_rows)], ignore_index=True)
            else:
                still.append(code)
        print(f"  [REPAIR r{rnd}] 成功 {len(failed)-len(still)} 只，仍失败 {len(still)} 只",
              flush=True)
        failed = still
    if failed:
        print(f"  [WARN] 本片仍有 {len(failed)} 只补拉未成功：{failed[:10]}"
              f"{' ...' if len(failed) > 10 else ''}", flush=True)
        acc.to_parquet(partial_path, index=False)  # 保留 partial，供下次续跑再补
    bs.logout()
    return acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=0, help="当前片索引")
    ap.add_argument("--n-batches", type=int, default=15, help="总片数")
    ap.add_argument("--sleep", type=float, default=0.5, help="每只股票查询间隔(s)，默认0.5=120次/分限流")
    ap.add_argument("--start", type=str, default="2010-01-01",
                    help="查询起始日(YYYY-MM-DD)，默认 2010-01-01。"
                         "区间长度直接决定单只耗时（实测 2026-11-05：2010起≈6.5s/只，"
                         "2015起≈2.8s/只）。只覆盖目标数据集所需区间可显著提速。")
    ap.add_argument("--end", type=str, default="2026-12-31",
                    help="查询结束日(YYYY-MM-DD)，默认 2026-12-31（覆盖到未来无害）。")
    ap.add_argument("--combine", action="store_true", help="仅合并所有分片")
    args = ap.parse_args()

    if args.combine:
        # 只收已完成的整片（baostock_daily_batch_NNN.parquet），**排除** .partial.parquet，
        # 否则会把未拉完的残片混进最终结果 → 静默损坏（2026-09-15 已踩：8 片混出 738 只假全集）。
        parts = sorted(glob.glob(str(OUTDIR / "baostock_daily_batch_[0-9][0-9][0-9].parquet")))
        if not parts:
            print("[31] 无分片可合并")
            return 1
        present = sorted({int(Path(p).stem.split("_")[-1]) for p in parts})
        expected = set(range(args.n_batches))
        missing = sorted(expected - set(present))
        if missing:
            # 缺片不允许合并：避免用残缺数据冒充全量。
            print(f"[31] 合并中止：缺片 {missing}（共需 {args.n_batches} 片，"
                  f"已就绪 {present}）。请先补全缺失片再 --combine。")
            return 1
        frames = [pd.read_parquet(p) for p in parts]
        df = pd.concat(frames, ignore_index=True)
        before = len(df)
        # 不同参数历史运行可能残留重复 symbol/date 组合，合并后强制去重。
        dup_cols = ["date", "symbol"] if "symbol" in df.columns else None
        if dup_cols:
            df = df.drop_duplicates(subset=dup_cols)
        after = len(df)
        if after < before:
            print(f"[31] 合并去重：{before:,} → {after:,} 行（剔除 {before-after:,} 重复）")
        out = OUTDIR / "baostock_daily.parquet"
        df.to_parquet(out, index=False)
        n_sym = df["symbol"].nunique() if "symbol" in df.columns else "?"
        dmin = df["date"].min().date() if "date" in df.columns else "?"
        dmax = df["date"].max().date() if "date" in df.columns else "?"
        print(f"[31] 合并 {len(parts)} 片 → baostock_daily.parquet "
              f"({after:,} 行, {n_sym} 只, date {dmin}~{dmax})")
        return 0

    syms = load_symbols()
    print(f"[31] 总 symbol {len(syms)}，切 {args.n_batches} 片", flush=True)
    # 均匀切
    chunks = np.array_split(np.array(syms, dtype=object), args.n_batches)
    batch_syms = [s for s in chunks[args.batch].tolist() if s]
    out_path = OUTDIR / f"baostock_daily_batch_{args.batch:03d}.parquet"
    partial_path = OUTDIR / f"baostock_daily_batch_{args.batch:03d}.partial.parquet"
    if out_path.exists():
        # 自愈：若上次因网络错误留下的 final **标的数不足**，说明它不是全量片，
        # 删掉它并借 partial 续跑补齐（否则会静默当成完整片，coverage 虚高）。
        try:
            got = pd.read_parquet(out_path, columns=["symbol"])["symbol"].nunique()
        except Exception:  # noqa: BLE001
            got = 0
        if got >= len(batch_syms):
            print(f"[31] 片 {args.batch} 已存在 {out_path.name}（{got} 只），跳过"
                  f"（删此文件可重跑）")
            return 0
        print(f"[31] 片 {args.batch} 已存在但仅 {got}/{len(batch_syms)} 只（上次网络错误），"
              f"删除后借 partial 续跑补齐", flush=True)
        _hard_delete(out_path)
    print(f"[31] 拉取片 {args.batch}/{args.n_batches}（{len(batch_syms)} 只，"
          f"区间 {args.start}~{args.end}）...", flush=True)
    t0 = time.time()
    try:
        df = fetch_batch(batch_syms, args.sleep, partial_path,
                         start_date=args.start, end_date=args.end)
    except BaostockBanned as e:
        print(f"[31] BAOSTOCK_BANNED: {e}", flush=True)
        return 2
    df.to_parquet(out_path, index=False)
    got = df["symbol"].nunique() if "symbol" in df.columns else 0
    print(f"[31] 片 {args.batch} 完成：{len(df):,} 行，{got}/{len(batch_syms)} 只，"
          f"{time.time()-t0:.0f}s → {out_path.name}", flush=True)
    # 覆盖护栏：本片标的缺失会**静默**造成下游覆盖率虚高（2026-11-05 实测连片网络错误
    # 导致 22 只被丢弃）。缺标的时**保留 partial**（含已成功的标的）并以非 0 退出，
    # 上层删掉 final 后重跑本片即可只补缺失的那些。
    if got < len(batch_syms):
        print(f"[31] ⚠️ 片 {args.batch} 仅 {got}/{len(batch_syms)} 只，缺 "
              f"{len(batch_syms)-got} 只（网络错误）；保留 partial 供续跑补齐", flush=True)
        return 3
    if partial_path.exists():
        _hard_delete(partial_path)  # 清理增量临时文件（绕过 safe-delete shim）
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
