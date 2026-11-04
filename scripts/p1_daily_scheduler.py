# -*- coding: utf-8 -*-
"""p1_daily_scheduler.py —— P1 数据链每日调度决策器（配合豆包定时任务「P1数据链每日推进」使用）。

职责：回答"今天要不要跑 P1 全链"，输出结构化决策供 Agent 触发时消费。
判断依据：
  1. K 线桥接（18_bridge_kline，cf399ccd 每日 19-20 点触发）是否完成 —— 看当日 bridge_run_*.log 是否含完成标志；
  2. K 线最新交易日是否领先 dataset（dataset 最新 = K 线最新 - h10 前瞻裁剪，设计内）；
  3. 领先则 RUN（跑 02_build_features 起的全链），否则 SKIP 并给原因。

输出：单行 JSON {action, reason, bridge_log, bridge_done, kline_latest, dataset_latest, ts}
"""
import datetime as dt
import glob
import json
import os
import random

BASE = r"E:\project\sj\P1-QuantFactor"
LOGS = os.path.join(BASE, "logs")
KLINE_DIR = r"E:\project\sj\data\P1\raw\kline"
DATASET = r"E:\project\sj\data\P1\processed\dataset_h10.parquet"
DONE_MARKS = ("KLINE DONE", "BRIDGE ALL DONE", "ALL KLINE DONE", "BRIDGE_DONE")


def latest_bridge_log(today: dt.date):
    """优先今天的 bridge 日志，其次最近一天。返回 (path, done)。"""
    pat = os.path.join(LOGS, f"bridge_run_{today:%Y%m%d}*.log")
    hits = sorted(glob.glob(pat))
    if not hits:
        all_hits = sorted(glob.glob(os.path.join(LOGS, "bridge_run_*.log")))
        hits = all_hits[-1:] if all_hits else []
    if not hits:
        return None, False
    p = hits[-1]
    try:
        t = open(p, encoding="utf-8", errors="replace").read()
    except Exception:
        return p, False
    return p, any(m in t for m in DONE_MARKS)


def kline_latest_date():
    """kline 目录随机抽 30 个 parquet 的 date max（快速近似，够判断齐整性）。"""
    files = [f for f in glob.glob(os.path.join(KLINE_DIR, "*.parquet")) if "logamount" not in f]
    if not files:
        return None
    sample = random.sample(files, min(30, len(files)))
    import pandas as pd
    mx = None
    for p in sample:
        try:
            d = pd.read_parquet(p, columns=["date"])["date"].max()
            if mx is None or d > mx:
                mx = d
        except Exception:
            continue
    return str(mx.date()) if mx is not None else None


def dataset_latest_date():
    if not os.path.exists(DATASET):
        return None
    import pandas as pd
    try:
        d = pd.read_parquet(DATASET, columns=["date"])["date"].max()
        return str(d.date())
    except Exception:
        return None


def main():
    today = dt.date.today()
    log_path, done = latest_bridge_log(today)
    kl = kline_latest_date()
    dl = dataset_latest_date()

    if not done:
        action = "SKIP"
        reason = f"BRIDGE_INCOMPLETE（{os.path.basename(log_path) if log_path else '无 bridge 日志'}，无完成标志；K 线桥接未齐，等待 cf399ccd 侧完成或告警）"
    elif kl and dl and kl > dl:
        action = "RUN"
        reason = f"K 线最新 {kl} 领先 dataset {dl}，可推进全链"
    else:
        action = "SKIP"
        reason = f"DATASET_CURRENT（dataset {dl} 已对齐 K 线 {kl} 设计上限）" if kl else "KLINE_UNKNOWN"

    print(json.dumps({
        "action": action, "reason": reason,
        "bridge_log": log_path, "bridge_done": done,
        "kline_latest": kl, "dataset_latest": dl,
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
