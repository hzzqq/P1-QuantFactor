"""P1 信号恢复一次性驱动（T-244，2026-10-11）——p1_recovery_plan.md §二 A′ 落地。

前置：面板(panel.parquet→10-08)与基座数据集(dataset_h10.parquet→09-16)已由
10-08 自动化轮刷新，本驱动不再重建，只做「EV 重建→log_amount→双路推理→
baseline 2026 增补→集成→导出→复制 ks」。

步骤（每步 exit 码 + 心跳落盘 E:/tmp/p1_recovery/heartbeat.log，失败即停）：
    S1  备份将被覆盖/删除的产物（*.bak_20261011，先备份后覆盖铁律）
    S2  EV 数据集重建（14_event_factor_iterate.py --build-only）+ 因子口径守卫
    S3  log_amount 重建（删缓存 → 27_add_logamount_factor.py）+ 覆盖率守卫
    S4  EV predict-only 推理（zz_predict_only_ev.py，复用生产权重不重训）
    S5  GRU predict-only 推理（zz_predict_only_gru.py，2022~2026）
    S6  baseline v39 2026 增补 + 合并（28_retrain_v39_baseline.py）
    S7  baseline 原版 2026 增补 + 合并（28 指向 dataset_h10）
    S8  集成预测（37_build_ensemble_pred.py）
    S9  导出四路信号（gru / baseline / ens / ev）
    S10 复制到 ks data/p1_signals/ + latest_date 终检

断点续跑：python zz_recovery_push_ev_20261011.py --from S6
并发守卫：驱动锁 E:/tmp/p1_recovery/driver.lock（含 PID，存活即拒绝二启）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # E:/project/sj
PROJ = ROOT / "P1-QuantFactor"
PD = ROOT / "data" / "P1" / "processed"
KS = Path("E:/project/ks/StockSignal/data/p1_signals")
PY = "E:/project/sj/env/Scripts/python.exe"
HB = Path("E:/tmp/p1_recovery/heartbeat.log")
DRIVER_LOG = Path("E:/tmp/p1_recovery/driver_20261011.log")
LOCK = Path("E:/tmp/p1_recovery/driver.lock")
BAK = ".bak_20261011"
SIGNALS = ["signal_gru_h10.json", "signal_baseline_h10.json",
           "signal_ens_v39gru_w025_h10.json", "signal_ev_h10.json"]
EV_PRED = "pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJ))


def log(msg: str) -> None:
    line = f"[{datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    DRIVER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIVER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def hb(msg: str) -> None:
    HB.parent.mkdir(parents=True, exist_ok=True)
    with open(HB, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%F %T} {msg}\n")


def fail(step: str, msg: str) -> int:
    hb(f"FAIL {step} {msg}")
    log(f"!!! {step} FAILED: {msg}")
    return 1


def run_step(step: str, args: list[str], timeout: int = 7200) -> int:
    """以 sj python 跑一个脚本步骤（cwd=PROJ，与 zz_daily_push.sh 同环境）。"""
    hb(f"START {step} {' '.join(args)}")
    t0 = time.time()
    p = subprocess.run([PY] + args, cwd=str(PROJ), timeout=timeout)
    dt = time.time() - t0
    if p.returncode != 0:
        return fail(step, f"exit={p.returncode} ({dt:.0f}s)")
    hb(f"OK {step} ({dt:.0f}s)")
    log(f"--- {step} OK ({dt:.0f}s)")
    return 0


def run_pycode(step: str, code: str, timeout: int = 1800) -> int:
    """以 sj python 跑一段校验/合并代码（写临时文件避免 -c 引号问题）。"""
    tmp = Path("E:/tmp/p1_recovery") / f"_{step.lower()}.py"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(code, encoding="utf-8")
    return run_step(step, [str(tmp)], timeout=timeout)


# ───────────────────────────── S1 备份 ─────────────────────────────
BACKUP_FILES = [
    "dataset_h10_ev.parquet",
    "dataset_h10_ev_meta.json",
    "dataset_h10_ev_v48.parquet",
    "dataset_h10_ev_v48_meta.json",
    EV_PRED,
    "pred_gru_h10.parquet",
    "pred_baseline_h10.parquet",
    "pred_baseline_h10_v39.parquet",
    "pred_ens_v39gru_w025_h10.parquet",
    "_log_amount.parquet",
]


def s1_backup() -> int:
    log("=== S1 备份 ===")
    done = []
    for name in BACKUP_FILES:
        src, dst = PD / name, PD / f"{name}{BAK}"
        if not src.exists():
            log(f"  skip(不存在) {name}")
            continue
        if dst.exists():
            log(f"  skip(已有备份) {dst.name}")
            continue
        shutil.copy2(src, dst)
        done.append(name)
    sb = PD / f"signals{BAK[1:]}"          # signals_bak_20261011
    sb.mkdir(exist_ok=True)
    for s in SIGNALS:
        src = PD / "signals" / s
        dst = sb / s
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    log(f"  本次新备份 {len(done)} 个: {done}")
    hb(f"OK S1 backup n={len(done)}")
    return 0


# ───────────────────────────── S2 EV 重建 + 守卫 ─────────────────────────────
S2_VERIFY = """
import json
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
PD = Path('E:/project/sj/data/P1/processed')
BAK = '.bak_20261011'
new = json.load(open(PD/'dataset_h10_ev_meta.json', encoding='utf-8'))
old = json.load(open(PD/('dataset_h10_ev_meta.json'+BAK), encoding='utf-8'))
assert new['factor_names'] == old['factor_names'], (
    'EV 因子口径漂移! new=%s old=%s' % (new['factor_names'], old['factor_names']))
assert len(new['factor_names']) == 47, 'EV 应为 47 维, got %s' % len(new['factor_names'])
n_rows = pq.ParquetFile(PD/'dataset_h10_ev.parquet').metadata.num_rows
dates = pd.read_parquet(PD/'dataset_h10_ev.parquet', columns=['date'])['date']
dmax = pd.to_datetime(dates).max()
assert n_rows > 2_000_000, 'EV 行数异常: %s' % n_rows
sheep = pd.read_csv('E:/project/ks/StockSignal/data/shepherd_history.csv')
smax = pd.to_datetime(sheep['date']).max()
assert smax >= dmax, '牧羊人 CSV(%s) 覆盖不足数据集边界(%s)' % (smax, dmax)
print('[S2v] EV 47 维口径一致 | 行数 %s | 数据至 %s | 牧羊人至 %s' % (f'{n_rows:,}', dmax.date(), smax.date()))
"""

S3_VERIFY = """
import json
from pathlib import Path
import pandas as pd
PD = Path('E:/project/sj/data/P1/processed')
la = pd.read_parquet(PD/'_log_amount.parquet')
lamax = pd.to_datetime(la['date']).max()
m39 = json.load(open(PD/'dataset_h10_v39_meta.json', encoding='utf-8'))
m48 = json.load(open(PD/'dataset_h10_ev_v48_meta.json', encoding='utf-8'))
assert lamax >= pd.Timestamp('2026-09-16'), '_log_amount 覆盖不足: %s' % lamax
assert len(m39['factor_names']) == 39 and m39['factor_names'][-1] == 'log_amount'
assert len(m48['factor_names']) == 48 and m48['factor_names'][-1] == 'log_amount'
print('[S3v] log_amount 至 %s (%s 行) | v39=39 维 v48=48 维 OK' % (lamax.date(), f"{len(la):,}"))
"""

S6_MERGE = """
import pandas as pd
from pathlib import Path
PD = Path('E:/project/sj/data/P1/processed')
old = pd.read_parquet(PD/'pred_baseline_h10_v39.parquet')
new = pd.read_parquet(PD/'pred_baseline_h10_v39_2026_new.parquet')
old['date'] = pd.to_datetime(old['date']); new['date'] = pd.to_datetime(new['date'])
old_pre = old[old['date'].dt.year < 2026]
merged = pd.concat([old_pre, new], ignore_index=True).drop_duplicates(subset=['date','symbol'], keep='last')
merged.to_parquet(PD/'pred_baseline_h10_v39.parquet', index=False)
print('[S6m] baseline_v39 merged rows', len(merged), 'datemax', merged['date'].max().date())
"""

S7_MERGE = """
import pandas as pd
from pathlib import Path
PD = Path('E:/project/sj/data/P1/processed')
old = pd.read_parquet(PD/'pred_baseline_h10.parquet')
new = pd.read_parquet(PD/'pred_baseline_h10_2026_new.parquet')
old['date'] = pd.to_datetime(old['date']); new['date'] = pd.to_datetime(new['date'])
old_pre = old[old['date'].dt.year < 2026]
merged = pd.concat([old_pre, new], ignore_index=True).drop_duplicates(subset=['date','symbol'], keep='last')
merged.to_parquet(PD/'pred_baseline_h10.parquet', index=False)
print('[S7m] baseline_original merged rows', len(merged), 'datemax', merged['date'].max().date())
"""

S9_ENS = """
import sys
sys.path.insert(0, 'E:/project/sj'); sys.path.insert(0, 'E:/project/sj/P1-QuantFactor')
import pandas as pd
from pathlib import Path
from src.signal import export_to_file
PD = Path('E:/project/sj/data/P1/processed')
ens = pd.read_parquet(PD/'pred_ens_v39gru_w025_h10.parquet')
out = PD/'signals'/'signal_ens_v39gru_w025_h10.json'
sig = export_to_file(ens, out, top_n=20, model='ens_v39gru_w025', horizon=10)
print('[S9] ens signal latest_date', sig['latest_date'], 'n_long', len(sig['top_long']), 'n_short', len(sig['top_short']))
"""

S10_VERIFY = """
import json
from pathlib import Path
KS = Path('E:/project/ks/StockSignal/data/p1_signals')
files = ['signal_gru_h10.json', 'signal_baseline_h10.json',
         'signal_ens_v39gru_w025_h10.json', 'signal_ev_h10.json']
lds = {}
for f in files:
    sig = json.load(open(KS/f, encoding='utf-8'))
    lds[f] = sig['latest_date']
    gen = sig.get('generated_at', '')
    print('[S10v]', f, 'latest_date=', sig['latest_date'], '| generated_at=', gen,
          '| top_long=', len(sig.get('top_long', [])), '| top_short=', len(sig.get('top_short', [])))
    assert sig.get('top_long') and sig.get('top_short'), f + ' top 榜为空'
uniq = sorted(set(lds.values()))
assert len(uniq) == 1, '四路 latest_date 不一致: %s' % lds
assert uniq[0] >= '2026-09-10', 'latest_date 前进不足: %s' % uniq[0]
print('[S10v] 四路 latest_date 一致 =', uniq[0], '—— 信号恢复成功')
"""


def s10_copy() -> int:
    log("=== S10 复制到 ks ===")
    for s in SIGNALS:
        src, dst = PD / "signals" / s, KS / s
        if not src.exists():
            return fail("S10", f"缺少 {src}")
        shutil.copy2(src, dst)
        log(f"  copied {s}")
    return run_pycode("S10v", S10_VERIFY)


STEPS = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]


def acquire_lock() -> bool:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        old = LOCK.read_text(encoding="utf-8").strip()
        pid = int(old.split()[0]) if old.split() else 0
        alive = False
        try:
            os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
        if alive and pid != os.getpid():
            log(f"另一驱动实例仍在运行 (pid={pid})，拒绝二启")
            return False
    LOCK.write_text(f"{os.getpid()} {datetime.now():%F %T}", encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="S1", choices=STEPS)
    args = ap.parse_args()

    if not acquire_lock():
        return 1
    hb(f"=== DRIVER START from={args.start} pid={os.getpid()} ===")
    log(f"=== P1 恢复驱动 START from={args.start} pid={os.getpid()} ===")

    idx = STEPS.index(args.start)
    for i, step in enumerate(STEPS[idx:], start=idx + 1):
        log(f"===== [{i}/10] {step} =====")
        if step == "S1":
            rc = s1_backup()
        elif step == "S2":
            rc = run_step("S2", ["scripts/14_event_factor_iterate.py",
                                 "--build-only", "--horizon", "10"], timeout=3600)
            if rc == 0:
                rc = run_pycode("S2v", S2_VERIFY)
        elif step == "S3":
            la = PD / "_log_amount.parquet"
            if la.exists():
                la.unlink()
                log("  已删过期 _log_amount.parquet（强制从扩展后 kline 重建）")
            rc = run_step("S3", ["scripts/27_add_logamount_factor.py"], timeout=7200)
            if rc == 0:
                rc = run_pycode("S3v", S3_VERIFY)
        elif step == "S4":
            rc = run_step("S4", ["scripts/zz_predict_only_ev.py", "--threads", "8"],
                          timeout=7200)
        elif step == "S5":
            rc = run_step("S5", ["scripts/zz_predict_only_gru.py", "--horizon", "10",
                                 "--seq-len", "20", "--threads", "8"], timeout=10800)
        elif step == "S6":
            rc = run_step("S6", ["scripts/28_retrain_v39_baseline.py", "--test-years", "2026",
                                 "--out-part", "pred_baseline_h10_v39_2026_new.parquet",
                                 "--yearly-part", "yearly_v39_2026_new.json"], timeout=7200)
            if rc == 0:
                rc = run_pycode("S6m", S6_MERGE)
        elif step == "S7":
            rc = run_step("S7", ["scripts/28_retrain_v39_baseline.py",
                                 "--ds", "dataset_h10.parquet",
                                 "--meta", "dataset_h10_meta.json",
                                 "--test-years", "2026",
                                 "--out-part", "pred_baseline_h10_2026_new.parquet",
                                 "--yearly-part", "yearly_2026_new.json"], timeout=7200)
            if rc == 0:
                rc = run_pycode("S7m", S7_MERGE)
        elif step == "S8":
            rc = run_step("S8", ["scripts/37_build_ensemble_pred.py"], timeout=7200)
        elif step == "S9":
            rc = run_step("S9a", ["scripts/06_export_signal.py",
                                  "--model", "gru", "--horizon", "10"])
            if rc == 0:
                rc = run_step("S9b", ["scripts/06_export_signal.py",
                                      "--model", "baseline", "--horizon", "10"])
            if rc == 0:
                rc = run_pycode("S9c", S9_ENS)
            if rc == 0:
                rc = run_step("S9d", ["scripts/06_export_signal.py",
                                      "--pred", str(PD / EV_PRED),
                                      "--model", "ev", "--horizon", "10"])
        elif step == "S10":
            rc = s10_copy()
        if rc != 0:
            LOCK.unlink(missing_ok=True)
            hb(f"=== DRIVER ABORT at {step} ===")
            log(f"=== 驱动中止于 {step}，修复后可 --from {step} 续跑 ===")
            return rc

    LOCK.unlink(missing_ok=True)
    hb("=== DRIVER ALL DONE ===")
    log("=== P1 恢复驱动 ALL DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
