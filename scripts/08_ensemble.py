"""阶段 8：基线(LightGBM) × GRU 信号融合实验。

核心问题：GRU 是否带来**增量信息**，还是只是基线的噪声副本？

判据：
  - 若两信号相关性很高（>0.7），融合无意义，GRU 可直接弃用；
  - 若相关性低（<0.4）且融合后 IC/ICIR 显著优于任一单信号，说明互补。

方法：
  1. 取两信号在 (date, symbol) 上的交集；
  2. 每个交易日横截面内，对两信号各自做 z-score 标准化（消除尺度差异）；
  3. score = w * z_base + (1 - w) * z_gru，扫描 w ∈ [0, 1]；
  4. **权重选择用 2022-2024，在 2025-2026 上做严格样本外验证**，
     避免在同一区间上选权重又报成绩（那是自欺）。

用法：
    python scripts/08_ensemble.py
    python scripts/08_ensemble.py --split-year 2025 --out pred_ens_h10.parquet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]      # E:/project/sj
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                               # noqa: E402
import pandas as pd                              # noqa: E402

from shared import paths                         # noqa: E402
from src.eval import metrics                     # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"


def _fmt(v, digits=4):
    if v is None or v != v:
        return "     -"
    return f"{v: .{digits}f}"


def cross_sectional_z(df: pd.DataFrame, col: str) -> pd.Series:
    """每个交易日横截面内做 z-score；当天标准差为 0 时退化为 0。"""
    g = df.groupby("date")[col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0.0, np.nan)
    z = (df[col] - mean) / std
    return z.fillna(0.0)


def load_pair(horizon: int) -> pd.DataFrame:
    b = pd.read_parquet(PROCESSED / f"pred_baseline_h{horizon}.parquet")
    g = pd.read_parquet(PROCESSED / f"pred_gru_h{horizon}.parquet")
    b["date"] = pd.to_datetime(b["date"])
    g["date"] = pd.to_datetime(g["date"])
    m = b[["date", "symbol", "pred", "y_excess"]].merge(
        g[["date", "symbol", "pred"]], on=["date", "symbol"],
        suffixes=("_base", "_gru"))
    m = m.dropna(subset=["pred_base", "pred_gru", "y_excess"])
    m["year"] = m["date"].dt.year
    return m


def build_score(m: pd.DataFrame, w: float) -> pd.Series:
    """w 为基线权重，(1-w) 为 GRU 权重；已各自做过横截面 z-score。"""
    return w * m["z_base"] + (1.0 - w) * m["z_gru"]


def evaluate(df: pd.DataFrame, score_col: str = "score") -> dict:
    return metrics.summarize(df, score_col, "y_excess")


def print_block(title: str, res: dict) -> None:
    print(f"  {title:<26} IC={_fmt(res.get('ic_mean'))}  "
          f"ICIR={_fmt(res.get('icir'))}  "
          f"正占比={_fmt(res.get('ic_positive_rate'), 3)}  "
          f"多空夏普={_fmt(res.get('spread_sharpe'), 2)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 基线 × GRU 融合实验")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--split-year", type=int, default=2025,
                    help=">= 该年份为样本外验证区；更早年份用于选权重")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    m = load_pair(args.horizon)
    print("\n" + "=" * 78)
    print("  P1 信号融合实验 · 基线 LightGBM × GRU+Attention")
    print("=" * 78)
    print(f"  重叠样本 : {len(m):,} 行 | {m.symbol.nunique()} 只 | "
          f"{m.date.min().date()} ~ {m.date.max().date()}")

    # ---- 1. 相关性：判断 GRU 是否只是基线的副本 ----
    pear = m[["pred_base", "pred_gru"]].corr().iloc[0, 1]
    spear = m[["pred_base", "pred_gru"]].corr(method="spearman").iloc[0, 1]
    # 逐日 rank IC 相关（更贴近因子研究惯例）
    def _daily_ic_corr(g):
        if g["pred_base"].nunique() < 5 or g["pred_gru"].nunique() < 5:
            return np.nan
        return g[["pred_base", "pred_gru"]].corr(method="spearman").iloc[0, 1]
    dic = m.groupby("date").apply(_daily_ic_corr, include_groups=False).dropna()

    print(f"\n  【信号相关性】")
    print(f"    Pearson  {pear: .4f}    Spearman  {spear: .4f}    "
          f"逐日 Spearman 均值  {dic.mean(): .4f}")
    if spear < 0.4:
        print("    → 低相关：两信号捕捉的是**互补信息**，融合有实质增益空间")
    elif spear < 0.7:
        print("    → 中等相关：融合可能有小幅增益")
    else:
        print("    → 高相关：GRU 近乎基线的副本，融合意义有限")

    # ---- 2. 横截面标准化 ----
    m["z_base"] = cross_sectional_z(m, "pred_base")
    m["z_gru"] = cross_sectional_z(m, "pred_gru")

    # ---- 3. 划分：选权重区 / 样本外验证区 ----
    tr = m[m.year < args.split_year].copy()
    te = m[m.year >= args.split_year].copy()
    print(f"\n  【区间划分】选权重区 <{args.split_year}：{len(tr):,} 行  |  "
          f"样本外 >={args.split_year}：{len(te):,} 行")

    # ---- 4. 单信号基准（各自区间） ----
    print(f"\n  【单信号基准】")
    for tag, df in (("选权重区", tr), (f"样本外 >={args.split_year}", te)):
        if df.empty:
            continue
        rb = evaluate(df.assign(score=df["z_base"]))
        rg = evaluate(df.assign(score=df["z_gru"]))
        print(f"   -- {tag} --")
        print_block("纯基线 LightGBM", rb)
        print_block("纯 GRU", rg)

    # ---- 5. 在选权重区扫描 w ----
    print(f"\n  【权重扫描（选权重区 <{args.split_year}，按 ICIR 选优）】")
    print("     w(基线)      IC        ICIR      多空夏普")
    best_w, best_icir = None, -np.inf
    scan = []
    for w in np.arange(0.0, 1.01, 0.1):
        r = evaluate(tr.assign(score=build_score(tr, w)))
        icir = r.get("icir") or -np.inf
        scan.append((round(float(w), 2), r))
        print(f"     {w: .1f}     {_fmt(r.get('ic_mean'))}   "
              f"{_fmt(r.get('icir'))}    {_fmt(r.get('spread_sharpe'), 2)}")
        if icir > best_icir:
            best_icir, best_w = icir, float(w)
    print(f"    → 选权重区最优 w = {best_w:.1f}（ICIR {best_icir:.4f}）")

    # ---- 6. 严格样本外验证（用上面选出的 w，不改） ----
    print(f"\n  【严格样本外验证（>={args.split_year}，权重 {best_w:.1f} 冻结）】")
    te = te.copy()
    te["score"] = build_score(te, best_w)
    r_ens = evaluate(te)
    r_b = evaluate(te.assign(score=te["z_base"]))
    r_g = evaluate(te.assign(score=te["z_gru"]))
    print_block(f"融合 w={best_w:.1f}", r_ens)
    print_block("纯基线 LightGBM", r_b)
    print_block("纯 GRU", r_g)

    ic_b = r_b.get("ic_mean") or 0.0
    ic_e = r_ens.get("ic_mean") or 0.0
    gain = (ic_e / ic_b - 1.0) * 100 if ic_b else float("nan")
    print(f"\n    融合相对纯基线 IC 提升：{gain:+.1f}%   "
          f"(IC {ic_b:.4f} → {ic_e:.4f})")

    # 分年度 + 分组单调性
    print(f"\n  【样本外分年度 IC】")
    print("     年份      融合IC      基线IC      GRU-IC")
    for y in sorted(te.year.unique()):
        gy = te[te.year == y]
        e = evaluate(gy).get("ic_mean")
        bb = evaluate(gy.assign(score=gy["z_base"])).get("ic_mean")
        gg = evaluate(gy.assign(score=gy["z_gru"])).get("ic_mean")
        print(f"     {y}    {_fmt(e)}    {_fmt(bb)}    {_fmt(gg)}")

    qr = metrics.quantile_returns(te, "score", "y_excess", n_groups=5)
    print(f"\n  【样本外分组超额收益（1=最弱，5=最强）】")
    print("     组号     平均超额收益      样本数")
    for g_, row in qr.iterrows():
        print(f"      {g_}      {_fmt(row['mean'], 5)}     {int(row['count']):>9,}")
    print("=" * 78 + "\n")

    # ---- 7. 落盘：融合预测 + 报告 ----
    out_name = args.out or f"pred_ens_h{args.horizon}.parquet"
    te[["date", "symbol", "score", "y_excess", "year"]].rename(
        columns={"score": "pred"}).to_parquet(PROCESSED / out_name, index=False)

    report = {
        "model": "ensemble_lgb_gru",
        "horizon": args.horizon,
        "split_year": args.split_year,
        "weight_baseline": best_w,
        "weight_gru": round(1.0 - best_w, 2),
        "correlation": {
            "pearson": float(pear), "spearman": float(spear),
            "daily_spearman_mean": float(dic.mean()),
        },
        "oos_overall": r_ens,
        "oos_baseline_only": r_b,
        "oos_gru_only": r_g,
        "oos_ic_gain_vs_baseline_pct": float(gain),
        "oos_by_year": {str(int(y)): evaluate(te[te.year == y])
                        for y in sorted(te.year.unique())},
        "quantile_returns": qr.reset_index().to_dict(orient="records"),
        "weight_scan": [{"w": w, **r} for w, r in scan],
        "n_rows_overlap": int(len(m)),
        "n_rows_oos": int(len(te)),
    }
    rp = PROCESSED / f"report_ens_h{args.horizon}.json"
    with rp.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  融合预测已保存: {PROCESSED / out_name}")
    print(f"  报告已保存:     {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
