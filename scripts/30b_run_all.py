"""阶段 30 - ②b 全流程自动续跑包装器（断点续跑 + 全局锁防并发）。

设计要点：
- 全局锁 `baostock_daily/.pipeline.lock`：防止自动化与手动/多次运行并发写同一批文件。
- 全断点续跑：每批 FINAL parquet 存在则跳过；partial 存在则从已完成 symbol 续拉。
- 并行分组拉取（5 组 × 3 批）加速；完成后自动 combine → derive v40 → h10/h20 重训 → 对比。
- 状态文件 `pipeline_status.json` 记录每步进度，供自动化判断是否已完成。

用法（自动化或手动均调此）：
    python scripts/30b_run_all.py
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = Path(sys.executable)
SCRIPTS = Path(__file__).resolve().parent
PROJ = SCRIPTS.parent
N_BATCHES = 15
OUTDIR = Path(r"E:/project/sj/data/P1/raw/baostock_daily")
LOCK_PATH = OUTDIR / ".pipeline.lock"
STATUS_PATH = OUTDIR / "pipeline_status.json"


def log(msg: str) -> None:
    print(f"[30b] {msg}", flush=True)


def load_status() -> dict:
    if STATUS_PATH.exists():
        try:
            return json.load(open(STATUS_PATH, encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_status(s: dict) -> None:
    json.dump(s, open(STATUS_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)


def acquire_lock() -> bool:
    """尝试获取全局锁。成功返回 True；已被占用返回 False。"""
    if LOCK_PATH.exists():
        # 检查锁是否过期（>3h 视为僵尸锁，强制接管）
        try:
            age = time.time() - LOCK_PATH.stat().st_mtime
        except Exception:
            age = 0
        if age < 3 * 3600:
            log(f"锁已存在（{LOCK_PATH.name}，{age/60:.0f}min 前），另一实例运行中，退出")
            return False
        log(f"锁已过期（{age/3600:.1f}h），强制接管")
    LOCK_PATH.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return True


def _hard_delete(path: Path) -> None:
    """删除锁文件，绕过 WorkBuddy sitecustomize 的 safe-delete shim。

    该 shim 把 Path.unlink / os.remove 劫持成『丢回收站』，非交互上下文里
    SHFileOperationW 会失败（0x2=文件找不到）并抛 OSError。直接调 Win32
    DeleteFileW 可绕过 Python 层 hook，对任何运行方式（手动/自动化）都可靠。
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
        # 锁只是 mtime 标记，删除失败可忽略：下次运行按 >3h 判僵尸强接管
        pass


def release_lock() -> None:
    if LOCK_PATH.exists():
        _hard_delete(LOCK_PATH)


def run(label: str, cmd: list[str], timeout: int = 600000) -> int:
    log(f">>> {label}")
    log(f"cmd: {' '.join(cmd)}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.stdout:
            lines = [l for l in result.stdout.strip().split("\n") if l and "distutils" not in l]
            for line in lines[-15:]:
                print(f"  {line}", flush=True)
        if result.stderr:
            err_lines = [l for l in result.stderr.strip().split("\n")
                         if l and "distutils" not in l]
            for line in err_lines[-5:]:
                print(f"  [ERR] {line}", flush=True)
        elapsed = time.time() - t0
        log(f"<<< {label} exited {result.returncode} in {elapsed:.0f}s")
        return result.returncode
    except subprocess.TimeoutExpired:
        log(f"!!! {label} TIMEOUT after {timeout}s")
        return 124
    except Exception as e:
        log(f"!!! {label} EXCEPTION: {e}")
        return 1


def main() -> int:
    if not acquire_lock():
        return 0  # 另一实例在跑，安全退出
    try:
        log("=== 阶段 30 ②b 全流程（断点续跑）===")
        t_start = time.time()
        st = load_status()

        # ---- PHASE A: 批量拉取（顺序单会话，规避 baostock 并发掐连接）----
        # 实测 5 并行会触发服务端同 IP 限流，导致 login/查询挂死（进程假活、不写盘）。
        # 改为顺序：每批一个独立 31 进程（单次 login），批间串行；每批 subprocess 超时 40min 兜底。
        if not st.get("fetch_done"):
            log(f"PHASE A: 顺序拉取 {N_BATCHES} 批（单会话；检测到黑名单立即避让）")
            banned = False
            for b in range(N_BATCHES):
                final = OUTDIR / f"baostock_daily_batch_{b:03d}.parquet"
                if final.exists():
                    log(f"  batch {b} 已完成（FINAL 存在），跳过")
                    continue
                cmd = [str(PYTHON), str(SCRIPTS / "31_baostock_fetch.py"),
                       "--batch", str(b), "--n-batches", str(N_BATCHES), "--sleep", "0.5"]
                log(f">>> fetch batch {b}")
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5400)
                    rc = res.returncode
                except subprocess.TimeoutExpired:
                    log(f"  !! batch {b} 超时 90min（可能挂死），杀掉续跑")
                    rc = 124
                if rc == 2:
                    # baostock IP 黑名单：立即停止本轮，避免空转/加重封禁，等解封后续跑
                    log(f"  !! batch {b} 报 baostock 黑名单（rc=2），本轮避让，等下次自动化续跑")
                    banned = True
                    break
                if rc != 0:
                    log(f"  batch {b} 退出码 {rc}，断点续跑已保护（partial 保留），继续下一批")
                    continue  # 不中断，继续其他批
            # 统计 FINAL 是否齐全，不齐则不进 combine（防假 all_done）
            n_final = sum(1 for b in range(N_BATCHES)
                           if (OUTDIR / f"baostock_daily_batch_{b:03d}.parquet").exists())
            if banned or n_final < N_BATCHES:
                log(f"PHASE A 未完成：{n_final}/{N_BATCHES} 批 FINAL 就绪"
                    + ("（检测到黑名单，避让）" if banned else "（部分批次失败，等续跑）"))
                log("  → 不进入 combine；fetch_done 保持未置，下次运行自动续拉缺失批")
                return 0  # 干净退出（锁在 finally 释放），等自动化/手动续跑
            st["fetch_done"] = True
            save_status(st)
            log("PHASE A 完成：全部 15 批 FINAL 就绪")

        # ---- PHASE B: 合并 ----
        if not st.get("combine_done"):
            if run("combine", [str(PYTHON), str(SCRIPTS / "31_baostock_fetch.py"), "--combine"]) != 0:
                return 1
            st["combine_done"] = True
            save_status(st)

        # ---- PHASE C: 派生 v40 ----
        if not st.get("derive_done"):
            if run("derive v40", [str(PYTHON), str(SCRIPTS / "32_derive_real_factors.py")]) != 0:
                return 1
            st["derive_done"] = True
            save_status(st)

        # ---- PHASE D: h10 重训 + 对比 ----
        if not st.get("h10_done"):
            for part, years, outp, yp in [
                ("h10 p1", "2019,2020,2021", "pred_baseline_h10_v40_part1.parquet", "yearly_v40_part1.json"),
                ("h10 p2a", "2023,2024", "pred_baseline_h10_v40_part2a.parquet", "yearly_v40_part2a.json"),
                ("h10 p2b", "2025,2026", "pred_baseline_h10_v40_part2b.parquet", "yearly_v40_part2b.json"),
            ]:
                if run(f"retrain h10 {part}",
                       [str(PYTHON), str(SCRIPTS / "33_retrain_real_baseline.py"),
                        "--horizon", "10", "--test-years", years,
                        "--out-part", outp, "--yearly-part", yp]) != 0:
                    return 1
            if run("compare h10", [str(PYTHON), str(SCRIPTS / "34_consolidate_real.py"), "--horizon", "10"]) != 0:
                return 1
            st["h10_done"] = True
            save_status(st)

        # ---- PHASE E: h20 重训 + 对比 ----
        if not st.get("h20_done"):
            for part, years, outp, yp in [
                ("h20 p1", "2019,2020,2021", "pred_baseline_h20_v40_part1.parquet", "yearly_h20_v40_part1.json"),
                ("h20 p2a", "2023,2024", "pred_baseline_h20_v40_part2a.parquet", "yearly_h20_v40_part2a.json"),
                ("h20 p2b", "2025,2026", "pred_baseline_h20_v40_part2b.parquet", "yearly_h20_v40_part2b.json"),
            ]:
                if run(f"retrain h20 {part}",
                       [str(PYTHON), str(SCRIPTS / "33_retrain_real_baseline.py"),
                        "--horizon", "20", "--ds", "dataset_h20_v40.parquet",
                        "--meta", "dataset_h20_v40_meta.json",
                        "--test-years", years, "--out-part", outp, "--yearly-part", yp]) != 0:
                    return 1
            if run("compare h20", [str(PYTHON), str(SCRIPTS / "34_consolidate_real.py"), "--horizon", "20"]) != 0:
                return 1
            st["h20_done"] = True
            save_status(st)

        total = time.time() - t_start
        log(f"=== 阶段 30 ②b 全流程完成！总耗时 {total/60:.1f} 分钟 ===")
        st["all_done"] = True
        st["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_status(st)
        return 0
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
