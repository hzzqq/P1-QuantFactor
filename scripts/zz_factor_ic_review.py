"""P1 因子 / IC 复盘（2026-11-10）：诊断近期模型走弱的根因。

不重训、不改信号，只读 dataset_h10.parquet + meta，逐因子重算 Rank-IC：
  - 全样本 / 2026YTD / 近 1 年(252) / 近 6 月(126) / 近 3 月(63) 五个窗口的
    mean IC、ICIR、正 IC 占比。
  - decay_pct = (1 - 近1年IC / 全样本IC) * 100   （衰减幅度，≥50% 标记）
  - reversal  = 全样本与近1年 IC 异号且 |全样本IC|>0.005  （反转）
  - PSI = 训练期(2021-2025) vs 近期(2026) 分布漂移（>0.25 标记）

输出 reports/factor_ic_review_YYYYMMDD.{md,json}。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths  # noqa: E402
from shared.logging_utils import get_logger  # noqa: E402

logger = get_logger("P1.factor_ic_review")
PROCESSED = paths.DATA / "P1" / "processed"


def _rank(v: np.ndarray) -> np.ndarray:
    """平均秩（与 scipy 默认一致），用于 Spearman。"""
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v), dtype="float64")
    ranks[order] = np.arange(1, len(v) + 1, dtype="float64")
    return ranks


def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    qs = np.quantile(ref, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    rp, _ = np.histogram(ref, qs)
    cp, _ = np.histogram(cur, qs)
    rp = np.clip(rp / max(1, len(ref)), 1e-4, None)
    cp = np.clip(cp / max(1, len(cur)), 1e-4, None)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--psi-ref-start", default="2021-01-01")
    ap.add_argument("--psi-recent-start", default="2026-01-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = json.load(open(PROCESSED / f"dataset_h{args.horizon}_meta.json",
                          encoding="utf-8"))
    factors = meta["factor_names"]
    df = pd.read_parquet(
        PROCESSED / f"dataset_h{args.horizon}.parquet",
        columns=["date", "symbol", "y_excess"] + list(factors),
    )
    df["date"] = pd.to_datetime(df["date"])
    n0 = len(df)
    df = df.dropna(subset=["y_excess"])  # IC 必须有前瞻标签；尾部(11-09)无标签行剔除
    logger.info("样本 %s → 有标签 %s（剔除尾部无标签 %s）",
                f"{n0:,}", f"{len(df):,}", f"{n0 - len(df):,}")

    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].values
    uniq = pd.to_datetime(np.unique(dates))
    n_d = len(uniq)
    logger.info("交易日 %s，因子 %s", f"{n_d:,}", len(factors))

    # ── 逐日 Rank-IC（Spearman = rank 的 Pearson）──
    logger.info("计算逐日 Rank-IC ...")
    ic = np.full((n_d, len(factors)), np.nan)
    for i, d in enumerate(uniq):
        m = dates == d
        g = df.iloc[m]
        ns = len(g)
        if ns < 50:
            continue
        ry = g["y_excess"].to_numpy(dtype="float64")
        fmat = g[factors].to_numpy(dtype="float64")           # (ns, F)
        out_row = np.full(len(factors), np.nan)
        for j in range(len(factors)):
            fv = fmat[:, j]
            ok = np.isfinite(ry) & np.isfinite(fv)
            if ok.sum() < 50:
                continue
            rf = _rank(fv[ok])
            ryv = _rank(ry[ok])
            rc = rf - rf.mean()
            yc = ryv - ryv.mean()
            den = np.sqrt((rc ** 2).sum()) * np.sqrt((yc ** 2).sum())
            if den == 0:
                continue
            out_row[j] = float((rc * yc).sum() / den)
        ic[i] = out_row

    ic_df = pd.DataFrame(ic, index=uniq, columns=factors)
    # 窗口切分
    def wmean(col, lo=None, hi=None):
        s = ic_df[col].dropna()
        if lo is not None:
            s = s[s.index >= pd.Timestamp(lo)]
        if hi is not None:
            s = s[s.index <= pd.Timestamp(hi)]
        return s

    win = {
        "FULL": None,
        "Y2026": "2026-01-01",
        "1Y": uniq[-252] if n_d >= 252 else uniq[0],
        "6M": uniq[-126] if n_d >= 126 else uniq[0],
        "3M": uniq[-63] if n_d >= 63 else uniq[0],
    }

    rows = []
    for f in factors:
        rec = {"factor": f}
        stats = {}
        for wname, lo in win.items():
            s = wmean(f, lo)
            if len(s) == 0:
                stats[wname] = (np.nan, np.nan, np.nan)
                continue
            m = s.mean()
            sd = s.std()
            icir = m / sd * np.sqrt(len(s)) if sd and sd > 0 else np.nan
            pos = (s > 0).mean()
            stats[wname] = (m, icir, pos)
        full_m = stats["FULL"][0]
        y1_m = stats["1Y"][0]
        rec["ic_full"] = full_m
        rec["ic_1y"] = y1_m
        rec["icir_1y"] = stats["1Y"][1]
        rec["ic_6m"] = stats["6M"][0]
        rec["ic_3m"] = stats["3M"][0]
        rec["ic_y2026"] = stats["Y2026"][0]
        rec["pos_1y"] = stats["1Y"][2]
        # 衰减：近1年相对全样本塌缩比例
        rec["decay_pct"] = (1 - y1_m / full_m) * 100 if (full_m and full_m != 0
                                                         and not np.isnan(y1_m)) else np.nan
        # 反转：全样本与近1年异号且全样本显著
        rec["reversal"] = bool(full_m * y1_m < 0 and abs(full_m) > 0.005)
        # PSI：训练期 vs 近期
        ref = df[(df["date"] >= args.psi_ref_start) &
                (df["date"] < args.psi_recent_start)][f].to_numpy(dtype="float64")
        cur = df[df["date"] >= args.psi_recent_start][f].to_numpy(dtype="float64")
        rec["psi"] = _psi(ref, cur) if len(ref) > 100 and len(cur) > 100 else np.nan
        rows.append(rec)

    out = pd.DataFrame(rows)
    # 排名：按近1年 IC 降序
    out = out.sort_values("ic_1y", ascending=False, na_position="last").reset_index(drop=True)

    # ── 标记问题因子（剔除近零 IC 噪声；衰减要求 |IC全|>0.01 且同号且近1年<50%）──
    def _is_decay(r):
        return (abs(r["ic_full"]) > 0.01) and (not np.isnan(r["decay_pct"])) and \
            (r["decay_pct"] >= 50) and (not r["reversal"])

    def _is_improved(r):
        return (abs(r["ic_full"]) > 0.01) and (not np.isnan(r["ic_1y"])) and \
            (r["ic_full"] != 0) and (r["ic_1y"] / r["ic_full"] > 1.5) and (not r["reversal"])

    out["flag_decay"] = out.apply(_is_decay, axis=1)
    out["flag_improved"] = out.apply(_is_improved, axis=1)
    out["flag_psi"] = out["psi"].apply(lambda x: (x is not None) and (not np.isnan(x)) and x > 0.25)

    # ── 输出 ──
    today = datetime.now().strftime("%Y%m%d")
    out_md = PROCESSED / "reports" / f"factor_ic_review_{today}.md"
    out_json = PROCESSED / "reports" / f"factor_ic_review_{today}.json"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    md = []
    md.append(f"# P1 因子/IC 复盘（{today}）\n")
    md.append(f"- 样本：有标签 {len(df):,} 行 / {n_d:,} 交易日 / {len(factors)} 因子（horizon={args.horizon}）")
    md.append(f"- 窗口：FULL(全样本) / Y2026(2026 至今有标签) / 1Y(近252) / 6M(近126) / 3M(近63)")
    md.append(f"- 衰减 decay_pct=(1-近1年IC/全样本IC)×100；反转=全样本与近1年 IC 异号且全样本|IC|>0.005；PSI>0.25 为分布漂移\n")

    md.append("## 一、Top 10 仍有效因子（按近1年 IC 降序）\n")
    md.append("| 因子 | IC全 | IC近1年 | ICIR近1年 | 正占比 | IC近3月 | 衰减% | PSI |")
    md.append("|---|---|---|---|---|---|---|---|")
    for _, r in out.head(10).iterrows():
        md.append("| %s | %.4f | %.4f | %.3f | %.1f%% | %.4f | %s | %.3f |" % (
            r["factor"], r["ic_full"], r["ic_1y"], r["icir_1y"] or 0,
            (r["pos_1y"] or 0) * 100, r["ic_3m"],
            "%.0f" % r["decay_pct"] if not np.isnan(r["decay_pct"]) else "NA",
            r["psi"] if not np.isnan(r["psi"]) else 0.0))

    md.append("\n## 二、最衰减因子（衰减≥50%，曾经有效现在塌了）\n")
    decayed = out[out["flag_decay"]].sort_values("decay_pct", ascending=False)
    md.append("| 因子 | IC全 | IC近1年 | 衰减% | 反转 | PSI |")
    md.append("|---|---|---|---|---|---|")
    if len(decayed):
        for _, r in decayed.iterrows():
            md.append("| %s | %.4f | %.4f | %.0f%% | %s | %.3f |" % (
                r["factor"], r["ic_full"], r["ic_1y"], r["decay_pct"],
                "⚠反" if r["reversal"] else "", r["psi"] if not np.isnan(r["psi"]) else 0.0))
    else:
        md.append("| （无，衰减≥50% 的因子为 0） | | | | | |")

    md.append("\n## 三、反转因子（IC 变号）\n")
    rev = out[out["reversal"]]
    md.append("| 因子 | IC全 | IC近1年 | IC近3月 | PSI |")
    md.append("|---|---|---|---|---|")
    if len(rev):
        for _, r in rev.iterrows():
            md.append("| %s | %.4f | %.4f | %.4f | %.3f |" % (
                r["factor"], r["ic_full"], r["ic_1y"], r["ic_3m"],
                r["psi"] if not np.isnan(r["psi"]) else 0.0))
    else:
        md.append("| （无反转因子） | | | | |")

    md.append("\n## 四、分布漂移 PSI>0.25 因子\n")
    psi_f = out[out["flag_psi"]].sort_values("psi", ascending=False)
    md.append("| 因子 | PSI | IC全 | IC近1年 | 衰减% |")
    md.append("|---|---|---|---|---|")
    if len(psi_f):
        for _, r in psi_f.iterrows():
            md.append("| %s | %.3f | %.4f | %.4f | %s |" % (
                r["factor"], r["psi"], r["ic_full"], r["ic_1y"],
                "%.0f" % r["decay_pct"] if not np.isnan(r["decay_pct"]) else "NA"))
    else:
        md.append("| （无 PSI>0.25 因子，分布稳定） | | | | |")

    md.append("\n## 五、结论要点\n")
    md.append(f"- 衰减≥50% 因子数：**{int(out['flag_decay'].sum())} / {len(out)}**；"
              f"反转因子数：**{int(out['reversal'].sum())}**；"
              f"PSI>0.25 因子数：**{int(out['flag_psi'].sum())}**。")
    # 与历史 claim 对账
    for key in ("rev_5", "bias_5"):
        if key in out["factor"].values:
            r = out[out["factor"] == key].iloc[0]
            md.append(f"- 历史 claim 对账 **{key}**：IC全={r['ic_full']:.4f} / "
                      f"IC近1年={r['ic_1y']:.4f} / 衰减={('%.0f%%' % r['decay_pct']) if not np.isnan(r['decay_pct']) else 'NA'} / "
                      f"反转={'是⚠' if r['reversal'] else '否'} / PSI={r['psi']:.3f}")
    md.append("\n> 口径：Rank-IC = 每日截面因子秩与前瞻{y}日超额收益秩的 Spearman 相关；"
             "ICIR = mean(IC)·√N / std(IC)。尾部无标签日(11-09)已剔除，不参与 IC。"
             .replace("{y}", str(args.horizon)))

    out_md.write_text("\n".join(md), encoding="utf-8")
    out.to_json(out_json, orient="records", force_ascii=False, indent=2)
    logger.info("报告已写：%s / %s", out_md.name, out_json.name)

    # 控制台摘要
    print(f"\n=== 因子/IC 复盘摘要（{today}）===")
    print(f"衰减≥50%: {int(out['flag_decay'].sum())} | 反转: {int(out['reversal'].sum())} | PSI>0.25: {int(out['flag_psi'].sum())}")
    print("\nTop5 仍有效（近1年 IC）:")
    for _, r in out.head(5).iterrows():
        print(f"  {r['factor']:14s} IC全={r['ic_full']:+.4f} 近1年={r['ic_1y']:+.4f} 衰减={('%.0f%%'%r['decay_pct']) if not np.isnan(r['decay_pct']) else 'NA'}")
    print("\nTop5 最衰减:")
    for _, r in decayed.head(5).iterrows():
        print(f"  {r['factor']:14s} IC全={r['ic_full']:+.4f} 近1年={r['ic_1y']:+.4f} 衰减={r['decay_pct']:.0f}% {'反转⚠' if r['reversal'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
