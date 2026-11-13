"""Tier-2 编排：构建 h20 EV 数据集 → GRU h20 多年 WF(2019-2026, 断点续训) → 对比分析。

后台运行；日志见 data/P1/processed/gru_h20_wf_log.txt
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path("E:/project/sj")
PROC = ROOT / "data" / "P1" / "processed"
PROJ = ROOT / "P1-QuantFactor"
PY = os.path.expanduser("~/.workbuddy/binaries/python/envs/default/Scripts/python.exe")
LOG = PROC / "gru_h20_wf_log.txt"

EPOCHS, CHUNK = 12, 12  # 单发：每年一次性训满 12 轮（含早停），不依赖断点续训，避免续训累积脆性与完成判定歧义

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    print(msg, flush=True)

def run(cmd):
    log(" ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], capture_output=False)
    return r.returncode

def main():
    log("=== Tier-2 GRU h20 WF START ===")
    # 1) 27c: 构建 dataset_h20_ev
    log("=== 27c build dataset_h20_ev ===")
    if run([PY, PROJ / "scripts" / "27c_build_h20_ev_labels.py"]) != 0:
        log("FATAL 27c"); sys.exit(1)
    # 2) 24b: GRU h20 WF 各年（单发训练，完成即写 report）
    #    完成判定：report 中已有该年条目（24b 仅在真正完成/早停完成时写）。
    #    注：dataset_h10_ev 起点 2018-01-24，2019/2020 训练窗落在数据前→空训练，跳过。
    rep_path = PROC / "report_gru_wf_h20.json"
    for y in range(2021, 2027):
        rep = {}
        if rep_path.exists():
            try:
                rep = json.load(open(rep_path, encoding="utf-8"))
            except Exception:
                pass
        if str(y) in rep:
            log(f"年 {y} 已完成，跳过")
            continue
        rc = run([PY, PROJ / "scripts" / "24b_gru_wf_h20.py", "--year", str(y),
                  "--epochs", str(EPOCHS), "--chunk", str(CHUNK)])
        if rc == 1:
            log(f"WARN 年 {y} rc=1（测试集为空，数据不可用，跳过）")
            continue
        if rc != 0:
            log(f"WARN 年 {y} rc={rc}（非致命，继续后续年）")
    # 3) 49: 对比分析
    log("=== 49 compare ===")
    run([PY, PROJ / "scripts" / "49_compare_h20_gru.py"])
    log("=== Tier-2 GRU h20 WF DONE ===")

if __name__ == "__main__":
    main()
