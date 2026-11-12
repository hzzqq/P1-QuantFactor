"""阶段 44 - 判定不达标候选 `pred_ens_v41base_h10.parquet` 是否还有可修复价值。

老板指令（2026-11-12）：「如果不达标候选修复没有意义就删了，还有价值可以留着接着改进」。
本脚本给出**可验证的判据**，用于决定删 / 留，并把规则固化成项目约定。

为什么值得再探一次（修正 43 的过度结论）
------------------------------------------
43 的归因 `B(v39 HGB)→C(v41 HGB) 净夏普 −0.1006` **混了两个变量**：
  (a) 估值因子 ep/btop 本身；
  (b) 股票池覆盖差异：v41 数据集永久排除 PE<=0 亏损股（少 24 只 / 75,662 行）。
43 未拆开 (a)/(b)，我在结论里写的「重启唯一前置条件 = 恢复 lightgbm」因此过绝对：
若 (b) 才是主因，则**补齐股票池即可挽回，不需要 lgb**。本探针专门拆这个。

四层实验（全部对齐区间 2021-01-04~2026-10-26，统一 y_excess / panel / 净成本）
---------------------------------------------------------------------------
1. **纯净对照（拆混杂核心）**：B_raw 与 C_raw 都裁到**共同 (date,symbol) 交集**，
   再各自在同一交集股票池上重算 _z() → 股票池与 z 基准完全一致，
   剩下的差异**纯粹是估值因子**。记为 B∩ / C∩。
   - 若 C∩ 净夏普 >= B∩ − 0.03 → 估值因子本身不伤组合层，(b) 是主因 → 有救。
   - 若 C∩ 明显差于 B∩ → 估值因子本身在组合层有害 → 补齐也没用 → 删。
2. **补齐股票池 C_pad**：C 的 z 在其覆盖键上，缺失键回退到现役 A 的 z
   （缺失回退，常见工程做法；两者同为 z 尺度，标注为近似）→ 股票池与现役完全一致。
3. **小权重融合 C_mix(w)**：(1−w)*A + w*C_pad，w∈{0.1,0.2,0.3}。
   43 只测了「整组替换」；v41 的 IC 优势集中在近期，小权重或可在几乎不伤
   全区间的同时改善近期表现。
4. **2026 子区间单独回测**：全区间会被 2021-2025 拖累，单独看 2026 是否真有增益。

预先声明的判据（写死在代码里，禁止事后挑数）
--------------------------------------------
  G1 估值因子本身不伤组合层：C∩ 净夏普 >= B∩ 净夏普 − 0.03
  G2 存在可用变体：任一 (C_pad / C_mix*) 净夏普 >= A − 0.05 且 2026 IC >= A + 0.005
  G3 近期确有增益：任一变体 2026 子区间净夏普 > A 的 2026 子区间净夏普
  redeemable = G1 AND (G2 OR G3)
  → True：候选留着接着改进；False：修复无意义 → 删。

只读评估 + 产出报告，**不删文件、不改现役/ks**（删除由人按报告执行）。
"""
from __future__ import annotations

import argparse
import json
import sys
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
from src.backtest import run_backtest_continuous, DEFAULT_COST  # noqa: E402
from src.eval.metrics import daily_ic, ic_summary  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
RECENT_TD = 59
RF, TOP_PCT, BUFFER = 15, 0.1, 0.1
ZERO = {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}

F_INCUMBENT = "pred_ens_v39gru_w025_h10.parquet"   # 现役 A（已 z）
F_V39_LGB = "pred_baseline_h10_v39.parquet"        # 统一 y_excess
F_V39_HGB = "pred_baseline_h10_v39_hgb.parquet"    # B_raw
F_V41_HGB = "pred_baseline_h10_v41.parquet"        # C_raw
F_CANDIDATE = "pred_ens_v41base_h10.parquet"       # 待判定候选


def _z(df: pd.DataFrame, col: str = "pred") -> pd.Series:
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
            ).fillna(0.0).clip(-3, 3)


def _dedup(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.drop_duplicates(subset=["date", "symbol"], keep="last")


def _ic(sig: pd.DataFrame, y_ref: pd.DataFrame, kind: str) -> float | None:
    df = sig.merge(y_ref, on=["date", "symbol"], how="inner")
    dates = pd.to_datetime(df["date"])
    if kind == "2026":
        m = dates >= pd.Timestamp("2026-01-01")
    else:  # recent59td
        uniq = np.sort(pd.unique(dates))
        m = dates >= (uniq[-RECENT_TD] if len(uniq) > RECENT_TD else uniq[0])
    if m.sum() == 0:
        return None
    return float(ic_summary(daily_ic(df[m], pred_col="pred", y_col="y_excess"))["ic_mean"])


def _bt(sig: pd.DataFrame, panel: pd.DataFrame, sub2026: bool = False) -> dict:
    s = sig
    p = panel
    if sub2026:
        s = s[s["date"] >= pd.Timestamp("2026-01-01")]
        p = p[p["date"] >= pd.Timestamp("2026-01-01")]
    kw = dict(horizon=10, top_pct=TOP_PCT, rebalance_freq=RF, mode="long_short")
    net = run_backtest_continuous(s, p, **kw, cost=DEFAULT_COST, buffer=BUFFER)
    n = net.ls_stats or {}
    return {"net_sharpe": n.get("sharpe"), "net_total": n.get("total_return"),
            "maxdd": n.get("max_drawdown")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    t0 = time.time()

    # ---- 载入 ----
    A = _dedup(pd.read_parquet(PROCESSED / F_INCUMBENT))[["date", "symbol", "pred"]]
    B_raw = _dedup(pd.read_parquet(PROCESSED / F_V39_HGB))
    C_raw = _dedup(pd.read_parquet(PROCESSED / F_V41_HGB))
    yv = _dedup(pd.read_parquet(PROCESSED / F_V39_LGB))
    y_ref = yv[["date", "symbol", "y_excess"]]
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])

    lo = max(A["date"].min(), B_raw["date"].min(), C_raw["date"].min())
    hi = min(A["date"].max(), B_raw["date"].max(), C_raw["date"].max())
    clip = lambda d: d[(d["date"] >= lo) & (d["date"] <= hi)].copy()
    A, B_raw, C_raw = clip(A), clip(B_raw), clip(C_raw)
    panel = panel[(panel["date"] >= lo) & (panel["date"] <= hi)]
    print(f"[44] 对齐区间 {lo.date()}~{hi.date()} | A {len(A):,} / B_raw {len(B_raw):,} / C_raw {len(C_raw):,}")

    # ---- 1. 纯净对照：共同交集 + 同一股票池重算 z ----
    keys_b = set(zip(B_raw["date"], B_raw["symbol"]))
    keys_c = set(zip(C_raw["date"], C_raw["symbol"]))
    inter = keys_b & keys_c
    Bcap = B_raw[B_raw.set_index(["date", "symbol"]).index.isin(inter)].copy()
    Ccap = C_raw[C_raw.set_index(["date", "symbol"]).index.isin(inter)].copy()
    Bcap["pred"] = _z(Bcap, "pred")
    Ccap["pred"] = _z(Ccap, "pred")
    Bcap, Ccap = Bcap[["date", "symbol", "pred"]], Ccap[["date", "symbol", "pred"]]
    print(f"[44] 交集键 {len(inter):,} | A 独有 {len(set(zip(A['date'],A['symbol']))-keys_c):,} "
          f"| B 独有 {len(keys_b-keys_c):,} | C 独有 {len(keys_c-keys_b):,}")

    # ---- 2. 生产口径 z（先算全量再裁，与 43 一致） ----
    B = B_raw.copy(); B["pred"] = _z(B, "pred"); B = B[["date", "symbol", "pred"]]
    C = C_raw.copy(); C["pred"] = _z(C, "pred"); C = C[["date", "symbol", "pred"]]

    # C_pad：缺失键回退到现役 A 的 z
    c_keys = set(zip(C["date"], C["symbol"]))
    a_keys = pd.MultiIndex.from_frame(A[["date", "symbol"]])
    missing = A[~a_keys.isin(c_keys)].copy()
    C_pad = pd.concat([C, missing], ignore_index=True)
    C_pad = C_pad.drop_duplicates(subset=["date", "symbol"], keep="first")
    print(f"[44] C_pad 补齐 {len(missing):,} 个缺失键 → {len(C_pad):,} 行（与 A 股票池一致）")

    variants = {
        "A_现役": A,
        "B_v39HGB": B,
        "C_候选原样": C,
        "C_pad_补齐": C_pad,
    }
    for w in (0.1, 0.2, 0.3):
        m = A[["date", "symbol"]].copy()
        cz = C_pad.set_index(["date", "symbol"])["pred"]
        az = A.set_index(["date", "symbol"])["pred"]
        idx = pd.MultiIndex.from_frame(m[["date", "symbol"]])
        m["pred"] = (1 - w) * az.reindex(idx).to_numpy() + w * cz.reindex(idx).to_numpy()
        variants[f"C_mix_w{w}"] = m

    # ---- 主表 ----
    print("\n[44] === 主表（对齐区间，净成本）===")
    print(f"  {'变体':16s} {'净夏普':>9s} {'净累计':>10s} {'回撤':>9s} "
          f"{'2026IC':>9s} {'近59IC':>9s} {'26子区夏普':>11s} {'26子区累计':>11s}")
    res = {}
    for name, sig in variants.items():
        full = _bt(sig, panel)
        sub = _bt(sig, panel, sub2026=True)
        r = {**full, "ic_2026": _ic(sig, y_ref, "2026"),
             "ic_recent": _ic(sig, y_ref, "recent59td"),
             "sub_sharpe": sub["net_sharpe"], "sub_total": sub["net_total"]}
        res[name] = r
        f = lambda x, fmt: (format(x, fmt) if x is not None else "—")
        print(f"  {name:16s} {f(r['net_sharpe'],'>+9.3f')} {f(r['net_total'],'>+10.2%')} "
              f"{f(r['maxdd'],'>9.2%')} {f(r['ic_2026'],'>+9.4f')} {f(r['ic_recent'],'>+9.4f')} "
              f"{f(r['sub_sharpe'],'>+11.3f')} {f(r['sub_total'],'>+11.2%')}")

    # ---- 纯净对照 ----
    print("\n[44] === 纯净对照（共同交集、同一股票池重算 z：只差估值因子）===")
    cap_res = {}
    for name, sig in [("B∩_v39HGB", Bcap), ("C∩_v41HGB", Ccap)]:
        full = _bt(sig, panel)
        r = {**full, "ic_2026": _ic(sig, y_ref, "2026"),
             "n_rows": len(sig)}
        cap_res[name] = r
        print(f"  {name:16s} 净夏普 {r['net_sharpe']:+.3f} | 净累计 {r['net_total']:+.2%} | "
              f"2026IC {r['ic_2026']:+.4f} | {r['n_rows']:,} 行")
    d_cap = cap_res["C∩_v41HGB"]["net_sharpe"] - cap_res["B∩_v39HGB"]["net_sharpe"]
    print(f"  → 估值因子本身净夏普Δ = {d_cap:+.4f}")

    # ---- 判据（预先声明，写死） ----
    a = res["A_现役"]
    G1 = d_cap >= -0.03
    mods = {k: v for k, v in res.items() if k.startswith("C_pad") or k.startswith("C_mix")}
    G2_list = [k for k, v in mods.items()
               if v["net_sharpe"] >= a["net_sharpe"] - 0.05
               and v["ic_2026"] is not None and v["ic_2026"] >= a["ic_2026"] + 0.005]
    G3_list = [k for k, v in mods.items()
               if v["sub_sharpe"] is not None and a["sub_sharpe"] is not None
               and v["sub_sharpe"] > a["sub_sharpe"]]
    G2, G3 = bool(G2_list), bool(G3_list)
    redeemable = bool(G1 and (G2 or G3))
    print("\n[44] === 判据（预先声明）===")
    print(f"  G1 估值因子本身不伤组合层 (C∩−B∩ >= −0.03): {G1}  (Δ={d_cap:+.4f})")
    print(f"  G2 存在可用变体 (净夏普>=A−0.05 且 2026IC>=A+0.005): {G2}  {G2_list}")
    print(f"  G3 任一变体 2026 子区间净夏普 > 现役: {G3}  {G3_list}")
    verdict = ("有可修复价值 → 候选留着接着改进（方向：补齐股票池 / 小权重分 regime 融合）"
               if redeemable else
               "修复无意义 → 删除候选（估值因子本身在组合层无增益，补齐/小权重均不达标）")
    print(f"\n[44] redeemable = {redeemable}")
    print(f"[44] verdict: {verdict}")

    report = {
        "common_window": [str(lo.date()), str(hi.date())],
        "config": {"rf": RF, "top_pct": TOP_PCT, "buffer": BUFFER},
        "intersection_keys": len(inter),
        "variants": res, "pure_comparison": cap_res,
        "pure_delta": round(d_cap, 4),
        "gate": {"G1": G1, "G2": G2, "G3": G3, "G2_variants": G2_list, "G3_variants": G3_list},
        "redeemable": redeemable, "verdict": verdict,
        "candidate_file": F_CANDIDATE,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (PROCESSED / "report_v41base_redeemability.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[44] 报告已写 report_v41base_redeemability.json  ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
