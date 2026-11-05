# -*- coding: utf-8 -*-
"""p1_daily_runner.py —— P1 数据链每日自动推进 runner（Windows 计划任务入口，20:45 触发）。

逻辑：调度器决策(RUN/SKIP) -> RUN 则顺序执行全链各步(退出码校验) -> 心跳 + 体检 -> 红灯标记 -> 看板留痕。
依赖：E:\\project\\sj\\env\\Scripts\\python.exe（sj 管道环境）；E:\\project\\ks\\StockSignal（ks 信号消费侧）。
"""
import datetime as dt
import json
import os
import subprocess
import sys

SJ_PY = r"E:\project\sj\env\Scripts\python.exe"
KS_PY = r"E:\project\ks\StockSignal\.venv\Scripts\python.exe"
BASE = r"E:\project\sj\P1-QuantFactor"
SCR = os.path.join(BASE, "scripts")
PD = r"E:\project\sj\data\P1\processed"
KS_SIG = r"E:\project\ks\StockSignal\data\p1_signals"
HB = r"E:\project\ks\StockSignal\logs\p1_daily_heartbeat.log"
BOARD = r"E:\project\team\TASKS.md"
SCHED = os.path.join(SCR, "p1_daily_scheduler.py")
MERGE = r"E:\project\ks\_merge_baseline.py"
QUALITY = r"E:\project\ks\StockSignal\scripts\p1_signal_quality_report.py"
LOGD = r"E:\project\ks\StockSignal\logs"


def log(msg, hb=True):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if hb:
        with open(HB, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run(script, *args, cwd=BASE, py=SJ_PY, hb=True):
    cmd = [py, os.path.join(SCR, script)] + list(args) if os.path.sep not in script else [py, script] + list(args)
    log(f"RUN {script} {' '.join(args)}", hb=hb)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (r.stdout or "").strip().splitlines()
    tail = tail[-3:] if tail else []
    for t in tail:
        log(f"  | {t}", hb=False)
    if r.returncode != 0:
        log(f"FAIL {script} rc={r.returncode} err={ (r.stderr or '')[-300:] }")
    return r.returncode


def scheduler_decision():
    r = subprocess.run([SJ_PY, SCHED], capture_output=True, text=True, encoding="utf-8", errors="replace")
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"action": "SKIP", "reason": f"SCHED_PARSE_FAIL {r.stdout[-200:]}"}


def copy_signals():
    files = ["signal_gru_h10.json", "signal_baseline_h10.json",
             "signal_ens_v39gru_w025_h10.json", "signal_ev_h10.json", "signal_fusion_h10.json"]
    for f in files:
        src = os.path.join(PD, "signals", f)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(KS_SIG, f))
    # 验证 latest_date 一致
    import glob as g
    jf = g.glob(os.path.join(KS_SIG, "signal_*.json"))
    lds = {}
    import json as j
    for p in jf:
        try:
            s = j.load(open(p, encoding="utf-8"))
            lds[os.path.basename(p)] = s.get("latest_date")
        except Exception:
            pass
    uniq = sorted(set(lds.values()))
    ok = len(uniq) == 1
    log(f"SIGNALS_COPY {'OK' if ok else 'INCONSISTENT'} dates={uniq} files={len(lds)}")
    return ok


def quality_check():
    r = subprocess.run([KS_PY, QUALITY, "--json"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    full = (r.stdout or "").strip()
    try:
        i, k = full.find("{"), full.rfind("}")
        j = json.loads(full[i:k + 1]) if i >= 0 and k > i else {}
    except Exception:
        j = {"parse": "fail", "raw": full[:200]}
    red = r.returncode != 0
    log(f"QUALITY rc={r.returncode} red={'YES' if red else 'no'} {json.dumps(j, ensure_ascii=False)[:400]}")
    return red, j


def board_append(action, reason, dates, red):
    today = f"{dt.date.today():%Y-%m-%d}"
    text = (f"| T-{dt.date.today():%m%d} | P1 数据链每日自动推进（Windows 计划任务，{today}） | 豆包(sj)（本账号） | "
            f"{'done' if action=='RUN' else 'skipped'} | sj: P1-QuantFactor 管道；ks: data/p1_signals | "
            f"[{today} {dt.datetime.now():%H:%M}] action={action} reason={reason} dates={dates} "
            f"quality_red={'YES' if red else 'no'}（心跳/日志：StockSignal/logs/p1_daily_heartbeat.log） |\r\n")
    with open(BOARD, "a", encoding="utf-8") as f:
        f.write(text)


def main():
    today = f"{dt.date.today():%Y%m%d}"
    log(f"RUNNER_START {today}")
    dec = scheduler_decision()
    log(f"SCHED action={dec.get('action')} reason={dec.get('reason')}")
    dates = {"kline": dec.get("kline_latest"), "dataset": dec.get("dataset_latest")}

    if dec.get("action") == "RUN":
        steps = [
            ("02_build_features.py", "--rebuild-panel"),
            ("14_event_factor_iterate.py", "--build-only", "--horizon", "10"),
            ("27_add_logamount_factor.py",),
            ("zz_predict_only_gru.py", "--horizon", "10", "--seq-len", "20", "--threads", "8"),
            ("zz_predict_only_ev.py", "--threads", "8"),
            ("28_retrain_v39_baseline.py", "--test-years", "2026", "--out-part", "pred_baseline_h10_v39_2026_new.parquet", "--yearly-part", "yearly_v39_2026_new.json"),
            ("MERGE_v39",),
            ("28_retrain_v39_baseline.py", "--ds", "dataset_h10.parquet", "--meta", "dataset_h10_meta.json", "--test-years", "2026", "--out-part", "pred_baseline_h10_2026_new.parquet", "--yearly-part", "yearly_2026_new.json"),
            ("MERGE_orig",),
            ("37_build_ensemble_pred.py", "--w-gru", "0.00"),
            ("EXPORT",),
        ]
        failed = False
        for st in steps:
            if st[0] == "MERGE_v39":
                r = subprocess.run([SJ_PY, MERGE, "v39"], capture_output=True, text=True, encoding="utf-8", errors="replace")
                log(f"MERGE v39 rc={r.returncode} { (r.stdout or '').strip().splitlines()[-1:] }")
                if r.returncode != 0:
                    failed = True; break
            elif st[0] == "MERGE_orig":
                r = subprocess.run([SJ_PY, MERGE, "original"], capture_output=True, text=True, encoding="utf-8", errors="replace")
                log(f"MERGE orig rc={r.returncode} { (r.stdout or '').strip().splitlines()[-1:] }")
                if r.returncode != 0:
                    failed = True; break
            elif st[0] == "EXPORT":
                for a in [("06_export_signal.py", "--model", "gru", "--horizon", "10"),
                          ("06_export_signal.py", "--model", "baseline", "--horizon", "10"),
                          ("zz_export_ens.py",),
                          ("06_export_signal.py", "--pred", os.path.join(PD, "pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet"), "--model", "ev", "--horizon", "10")]:
                    if run(*a) != 0:
                        failed = True; break
                if failed: break
            else:
                if run(*st) != 0:
                    failed = True; break
        if not failed:
            ok = copy_signals()
            r = subprocess.run([SJ_PY, os.path.join(SCR, "11_conditional_fusion.py")], capture_output=True, text=True, encoding="utf-8", errors="replace")
            log(f"FUSION rc={r.returncode}")
            if r.returncode == 0:
                subprocess.run([SJ_PY, os.path.join(SCR, "06_export_signal.py"), "--model", "fusion"], capture_output=True, text=True)
                import shutil
                shutil.copy2(os.path.join(PD, "signals", "signal_fusion_h10.json"), os.path.join(KS_SIG, "signal_fusion_h10.json"))
                # 融合信号是最后一批落盘的，必须重跑一致性校验：
                # 之前只在融合导出前校验，会把当日的 fusion 旧日期误判成 INCONSISTENT
                ok = copy_signals()
                run("zz_paper_portfolio.py", "--top-k", "10", hb=False)
                run("zz_paper_sensitivity.py", hb=False)
            log(f"CHAIN_DONE ok={ok}")
        else:
            log("CHAIN_ABORTED")
    else:
        log(f"SKIP reason={dec.get('reason')}")

    red, qj = quality_check()
    board_append(dec.get("action"), dec.get("reason"), dates, red)
    log("RUNNER_END")


if __name__ == "__main__":
    main()
