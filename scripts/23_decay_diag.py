"""阶段 25 - 信号衰减根因诊断。

问题：baseline 全期 IC 0.0494，但 2021 后腰斩（2020 0.091 → 2021 0.025 → 2022 0.010）。
GRU 仅 validated 在 2026（无多年级 track record）。本脚本用有 8 年 track record 的
dataset_h10（38 因子，2015-2026）做数据驱动诊断：

  1) 逐因子逐年 Spearman IC → factor×year 矩阵；识别「2021 前强、2021 后失效/翻号」的因子。
  2) 按前缀分类（mom/rev/vol/vr/vstd/pos/bias + ampl/gap/ret）→ 分类级 pre(≤2020) vs post(≥2021) 平均 IC。
  3) 标签 y_excess 漂移：逐年均值/标准差/分位 → 看 target 是否 regime 变化。
  4) baseline 信号逐年 IC（从 pred_baseline_h10.parquet）交叉比对，确认是「因子失效」还是「模型失效」。

产物：data/P1/processed/report_decay_diag.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths

PROCESSED = paths.DATA / "P1" / "processed"
CUT = 2021  # regime 分界年（2021 核心资产崩塌）


def main() -> int:
    t0 = time.time()
    meta = json.load(open(PROCESSED / "dataset_h10_meta.json", encoding="utf-8"))
    fn = meta["factor_names"]
    cols = fn + ["date", "y_excess"]
    print(f"加载 dataset_h10（仅 {len(fn)} 因子 + date/y_excess）...")
    df = pd.read_parquet(PROCESSED / "dataset_h10.parquet", columns=cols)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    df["year"] = pd.to_datetime(df["date"]).dt.year
    years = sorted(df["year"].unique())
    print(f"样本 {len(df):,} 行 | 年份 {years[0]}~{years[-1]}")

    # 1) 逐因子逐年 IC
    print("计算逐因子逐年 IC ...")
    fac_year = {}
    for f in fn:
        sub = df[["year", f, "y_excess"]].dropna(subset=[f])
        g = sub.groupby("year").apply(lambda x: spearmanr(x[f], x["y_excess"]).correlation)
        fac_year[f] = {int(y): (round(float(v), 4) if v == v else None) for y, v in g.items()}

    # 2) pre vs post 平均 IC
    def avg_ic(f, lo, hi):
        vs = [fac_year[f][y] for y in years if lo <= y <= hi and fac_year[f].get(y) is not None]
        return float(np.nanmean(vs)) if vs else None
    pre = {f: avg_ic(f, years[0], CUT - 1) for f in fn}
    post = {f: avg_ic(f, CUT, years[-1]) for f in fn}

    # 3) 分类汇总
    cats = {}
    for f in fn:
        cat = "".join(c for c in f if not c.isdigit() and c != "_") if not f.split("_")[0].isdigit() else f.split("_")[0]
        cat = f.split("_")[0]  # mom/rev/vol/vr/vstd/pos/bias/ampl/gap/ret
        cats.setdefault(cat, []).append(f)
    cat_pre = {c: np.nanmean([pre[f] for f in fs if pre[f] is not None]) for c, fs in cats.items()}
    cat_post = {c: np.nanmean([post[f] for f in fs if post[f] is not None]) for c, fs in cats.items()}

    # 4) 标签漂移
    yb = df.groupby("year")["y_excess"].agg(["mean", "std", "skew",
                                             lambda x: x.quantile(0.05),
                                             lambda x: x.quantile(0.95)])
    yb.columns = ["mean", "std", "skew", "q05", "q95"]

    # 5) baseline 信号逐年 IC（交叉比对）
    base = pd.read_parquet(PROCESSED / "pred_baseline_h10.parquet", columns=["year", "pred", "y_excess"])
    base = base.dropna(subset=["pred", "y_excess"])
    base_ic = base.groupby("year").apply(lambda x: spearmanr(x["pred"], x["y_excess"]).correlation)
    base_ic = {int(y): round(float(v), 4) for y, v in base_ic.items()}

    # 识别失效因子：pre 强(>0.02) 且 post 弱(<0 或降 >50%)
    decayed = []
    for f in fn:
        p, q = pre[f], post[f]
        if p is not None and q is not None and p > 0.02 and (q < 0 or q < p * 0.5):
            decayed.append({"factor": f, "pre_ic": round(p, 4), "post_ic": round(q, 4),
                            "drop": round(q - p, 4)})

    rep = {
        "cut_year": CUT,
        "years": [int(y) for y in years],
        "factor_year_ic": fac_year,
        "factor_pre_ic": {k: (round(v, 4) if v is not None else None) for k, v in pre.items()},
        "factor_post_ic": {k: (round(v, 4) if v is not None else None) for k, v in post.items()},
        "category_pre_ic": {k: round(v, 4) for k, v in cat_pre.items()},
        "category_post_ic": {k: round(v, 4) for k, v in cat_post.items()},
        "label_drift_by_year": {int(y): {c: round(float(v), 4) for c, v in row.items()}
                                 for y, row in yb.iterrows()},
        "baseline_signal_ic_by_year": base_ic,
        "decayed_factors": sorted(decayed, key=lambda d: d["drop"]),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    json.dump(rep, open(PROCESSED / "report_decay_diag.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)

    # 打印摘要
    print("\n=== 分类级平均 IC：pre(≤%d) vs post(≥%d) ===" % (CUT - 1, CUT))
    for c in sorted(cat_pre, key=lambda x: cat_post[x] - cat_pre[x]):
        dp = cat_pre[c]; dq = cat_post[c]
        arrow = "↓" if dq < dp else "↑"
        print(f"  {c:8s} pre={dp:+.4f}  post={dq:+.4f}  {arrow}{abs(dq-dp):.4f}")
    print("\n=== baseline 信号逐年 IC（交叉比对）===")
    for y in years:
        print(f"  {y}: {base_ic.get(int(y))}")
    print("\n=== 失效因子（pre>0.02 且 post<0 或降>50%）===")
    for d in rep["decayed_factors"]:
        print(f"  {d['factor']:10s} pre={d['pre_ic']:+.4f} post={d['post_ic']:+.4f} drop={d['drop']:+.4f}")
    print(f"\n已落盘 report_decay_diag.json（耗时 {time.time()-t0:.1f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
