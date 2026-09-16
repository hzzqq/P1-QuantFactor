"""baostock 当前可用性探针（不写盘、不碰分片，纯探测）。

复用 31_baostock_fetch 的真实查询逻辑（_query_one_symbol / to_bs_code / FIELDS / load_symbols），
取首批 N 只股票探测：成功/失败数、平均耗时、失败样例。

go/no-go 闸门：
  - 失败率≈0 且 ~11s/只 → 全量约 4h，可考虑续跑；
  - 仍报错（10002007/UTF-8）或整体挂死 → 放弃本次续跑。

【关键硬化】baostock 收到损坏数据时会进入**不可中断的内部阻塞**（其底层网络/解压线程不在
31 的 worker 线程内，31 的 join(timeout) 超时保护对整体挂死无效）。为避免探针像 09-16 那样
白等 10 分钟，本脚本把核心探测放进**独立子进程**，父进程以 280s 墙钟硬超时 join；超时即
terminate 并判定 NOGO（挂死）。结果经临时 JSON 回传。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 子进程墙钟硬上限：超过即判定 baostock 整体挂死
WALLCLOCK_LIMIT = 280.0


def _load_fetch_module():
    spec = importlib.util.spec_from_file_location(
        "baofetch_probe", str(PROJ / "scripts" / "31_baostock_fetch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_probe(n: int, timeout: float, sleep: float, out_path: str) -> None:
    """在子进程中执行真实探测，结果写入 out_path（JSON）。"""
    try:
        mod = _load_fetch_module()
        bs = __import__("baostock")
        lg = mod._login_with_timeout(bs, timeout=30)
        if lg is None:
            json.dump({"verdict": "NOGO", "reason": "login-timeout"}, open(out_path, "w"))
            return
        if lg.error_code != "0":
            json.dump({"verdict": "NOGO", "reason": f"login-{lg.error_code}"},
                      open(out_path, "w"))
            return

        syms = mod.load_symbols()
        probe = syms[:n]
        ok = fail = 0
        times: list[float] = []
        errs: list[str] = []
        for code in probe:
            t = time.time()
            rows = mod._query_one_symbol(bs, mod.to_bs_code(code), timeout=timeout)
            times.append(time.time() - t)
            if rows:
                ok += 1
            else:
                fail += 1
                errs.append(code)
            time.sleep(sleep)
        bs.logout()

        if fail == 0 and times and np.mean(times) < 20:
            est_h = len(syms) * np.mean(times) / 3600.0
            json.dump({"verdict": "GO", "ok": ok, "fail": fail, "n": len(probe),
                       "mean_s": float(np.mean(times)), "max_s": float(max(times)),
                       "total_s": float(sum(times)), "errs": errs, "est_h": float(est_h)},
                      open(out_path, "w"))
        else:
            json.dump({"verdict": "NOGO", "reason": "errors-or-slow", "ok": ok, "fail": fail,
                       "n": len(probe), "mean_s": float(np.mean(times)) if times else 0.0,
                       "errs": errs}, open(out_path, "w"))
    except Exception as e:  # noqa: BLE001
        json.dump({"verdict": "NOGO", "reason": f"exc:{repr(e)[:160]}"}, open(out_path, "w"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    out_path = tempfile.mktemp(suffix=".probe.json")
    p = mp.Process(target=_run_probe, args=(args.n, args.timeout, args.sleep, out_path))
    p.start()
    p.join(timeout=WALLCLOCK_LIMIT)
    if p.is_alive():
        p.terminate()
        try:
            p.join(timeout=5)
        except Exception:
            pass
        print(f"[PROBE] 整体挂死 > {WALLCLOCK_LIMIT:.0f}s（baostock 查询不可中断阻塞）→ GATE=NOGO")
        return 1

    try:
        res = json.load(open(out_path, encoding="utf-8"))
    except Exception:
        print("[PROBE] 结果文件缺失 → GATE=NOGO")
        return 1
    finally:
        try:
            Path(out_path).unlink()
        except Exception:
            pass

    v = res.get("verdict")
    if v == "GO":
        print(f"[PROBE] 成功 {res['ok']}/{res['n']}，失败 {res['fail']}/{res['n']}")
        print(f"[PROBE] 平均 {res['mean_s']:.1f}s/只，最慢 {res['max_s']:.1f}s，总 {res['total_s']:.0f}s")
        print(f"[PROBE] GATE=GO: 全量约 {res['est_h']:.1f}h 可考虑续跑")
        return 0
    print(f"[PROBE] GATE=NOGO: reason={res.get('reason')} "
          f"ok={res.get('ok')}/{res.get('n')} fail={res.get('fail')}")
    if res.get("errs"):
        print(f"[PROBE] 失败样例: {res['errs'][:10]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
