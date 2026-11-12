"""P1 轻量重加权 quick test（2026-11-11）：因子级再加权能否让近期信号回血。

不重训生产模型、不动现役信号。只用 dataset_h10 的 38 因子 + y_excess：
  - 用「陈旧窗口(2021-22)」估计因子权重 vs「近期窗口(2023)」估计的权重，
    各组成一个线性秩集成，在样本外窗口(2024-01-01 起)测 daily rank-IC 与多空收益。
  - 变体：
      V_stale      陈旧权重（模拟冻结在旧数据的模型）
      V_recent     近期权重（重加权到当前仍强的因子）
      V_recent_nobias  近期权重但剔除 bias_5（隔离 bias_5 反转影响）
      V_voltilt    近期权重 + 仅保留波动率/成交量族（向波动率族倾斜的极端版）
  - 同时把生产信号(pred_gru/pred_baseline/pred_ens)近期 daily rank-IC 拉来当 benchmark。

输出 reports/reweight_quicktest_YYYYMMDD.{md,json}。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("E:/project/sj/data/P1/processed")
REP = PROC / "reports"

REVERSAL_FAMILY = {
    "rev_5", "rev_10", "rev_20", "rev_60", "rev_120",
    "mom_5", "mom_10", "mom_20", "mom_60", "mom_120",
    "pos_120", "bias_10", "bias_120",
}
VOL_FAMILY = {  # 波动率/成交量/振幅/价差/量比 族（稳健且近1年强）
    "vol_5", "vol_10", "vol_20", "vol_60", "vol_120",
    "vstd_5", "vstd_10", "vstd_20", "vstd_60", "vstd_120",
    "vr_5", "vr_10", "vr_20", "vr_60", "vr_120",
    "ampl_1", "gap_1", "log_amount",
}


def daily_rankic(test: pd.DataFrame, score_col: str, y_col: str = "y_excess") -> pd.Series:
    t = test.copy()
    t["_s"] = t.groupby("date")[score_col].rank(pct=True)
    t["_y"] = t.groupby("date")[y_col].rank(pct=True)
    return t.groupby("date").apply(lambda g: g["_s"].corr(g["_y"]), include_groups=False)


def long_short(test: pd.DataFrame, score_col: str, y_col: str = "y_excess",
               top: float = 0.1, bot: float = 0.1) -> dict:
    t = test.copy()
    t["_r"] = t.groupby("date")[score_col].rank(pct=True)
    g = t.groupby("date")
    out = []
    for _, d in g:
        thr_hi = d["_r"].quantile(1 - top)
        thr_lo = d["_r"].quantile(bot)
        long = d.loc[d["_r"] >= thr_hi, y_col].mean()
        short = d.loc[d["_r"] <= thr_lo, y_col].mean()
        out.append(long - short)
    s = pd.Series(out)
    n = len(s)
    mean = s.mean()
    sd = s.std()
    t_stat = mean / (sd / np.sqrt(n)) if sd and sd > 0 else np.nan
    ann = mean / sd * np.sqrt(252 / 10) if sd and sd > 0 else np.nan  # 10日 horizon 年化
    return {"ls_mean": mean, "ls_t": t_stat, "ls_sharpe_ann": ann, "n_days": n}


def weight_from_window(df: pd.DataFrame, factors, lo, hi) -> pd.Series:
    w = df[(df["date"] >= lo) & (df["date"] <= hi)].reset_index(drop=True)
    dates = w["date"].values
    uniq = pd.to_datetime(np.unique(dates))
    dmap = {d: i for i, d in enumerate(uniq)}
    didx = np.array([dmap[d] for d in dates])
    fr = w.groupby("date")[factors].rank(pct=True).to_numpy(dtype="float64")
    yr = w.groupby("date")["y_excess"].rank(pct=True).to_numpy(dtype="float64")
    n_d = len(uniq)
    ic = np.full((n_d, len(factors)), np.nan)
    for i in range(n_d):
        m = didx == i
        ns = int(m.sum())
        if ns < 50:
            continue
        f = fr[m]
        yc = yr[m]
        fc = f - f.mean(0)
        ycc = (yc - yc.mean()).reshape(-1, 1)
        den = np.sqrt((fc ** 2).sum(0)) * np.sqrt((ycc ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            c = (fc * ycc).sum(0) / np.where(den == 0, np.nan, den)
        ic[i] = c
    return pd.Series(np.nanmean(ic, axis=0), index=factors)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale-lo", default="2021-01-01")
    ap.add_argument("--stale-hi", default="2022-12-31")
    ap.add_argument("--recent-lo", default="2023-01-01")
    ap.add_argument("--recent-hi", default="2023-12-31")
    ap.add_argument("--test-lo", default="2024-01-01")
    args = ap.parse_args()

    meta = json.load(open(PROC / "dataset_h10_meta.json", encoding="utf-8"))
    factors = meta["factor_names"]
    df = pd.read_parquet(PROC / "dataset_h10.parquet",
                         columns=["date", "symbol", "y_excess"] + list(factors))
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["y_excess"]).sort_values("date").reset_index(drop=True)

    # 估计权重
    w_stale = weight_from_window(df, factors, args.stale_lo, args.stale_hi)
    w_recent = weight_from_window(df, factors, args.recent_lo, args.recent_hi)

    # 测试窗口
    test = df[df["date"] >= args.test_lo].reset_index(drop=True)
    fr_test = test.groupby("date")[factors].rank(pct=True)  # 0..1

    def score(w: pd.Series) -> np.ndarray:
        return fr_test.values @ w.reindex(factors).fillna(0.0).values

    variants = {}
    variants["V_stale"] = score(w_stale)
    variants["V_recent"] = score(w_recent)
    # 剔除 bias_5 的近期权重
    w_recent_nobias = w_recent.copy(); w_recent_nobias["bias_5"] = 0.0
    variants["V_recent_nobias"] = score(w_recent_nobias)
    # 向波动率族倾斜：仅保留 VOL_FAMILY（其余权重置 0）
    w_vol = w_recent.copy()
    for f in factors:
        if f not in VOL_FAMILY:
            w_vol[f] = 0.0
    variants["V_voltilt"] = score(w_vol)

    results = {}
    for name, sc in variants.items():
        test["_sc"] = sc
        dic = daily_rankic(test, "_sc")
        ls = long_short(test, "_sc")
        results[name] = {
            "ic_mean": float(dic.mean()), "ic_std": float(dic.std()),
            "icir": float(dic.mean() / dic.std() * np.sqrt(len(dic))) if dic.std() and dic.std() > 0 else np.nan,
            "ls_mean": float(ls["ls_mean"]), "ls_t": float(ls["ls_t"]),
            "ls_sharpe_ann": float(ls["ls_sharpe_ann"]), "n_days": int(ls["n_days"]),
        }

    # 生产信号 benchmark（从 dataset 合并 y_excess，兼容无该列的 ens 文件）
    y_lookup = df[["date", "symbol", "y_excess"]]
    bench = {}
    for f, key in [("pred_gru_h10.parquet", "gru"),
                   ("pred_baseline_h10.parquet", "baseline"),
                   ("pred_ens_v39gru_w025_h10.parquet", "ens")]:
        p = pd.read_parquet(PROC / f)
        p["date"] = pd.to_datetime(p["date"])
        p = p.drop(columns=[c for c in ("y_excess",) if c in p.columns], errors="ignore")
        m = p.merge(y_lookup, on=["date", "symbol"], how="inner").dropna(subset=["y_excess"])
        m = m[m["date"] >= args.test_lo]
        m["_y"] = m.groupby("date")["y_excess"].rank(pct=True)
        m["_p"] = m.groupby("date")["pred"].rank(pct=True)
        dic = m.groupby("date").apply(lambda g: g["_p"].corr(g["_y"]), include_groups=False)
        bench[key] = float(dic.mean())

    # bias_5 单独对账
    bias_stale = float(w_stale.get("bias_5", np.nan))
    bias_recent = float(w_recent.get("bias_5", np.nan))

    out = {
        "test_window": f"{args.test_lo} ~ {test['date'].max().date()}",
        "stale_window": f"{args.stale_lo}~{args.stale_hi}",
        "recent_window": f"{args.recent_lo}~{args.recent_hi}",
        "variants": results,
        "production_benchmark_ic": bench,
        "bias_5": {"ic_stale": bias_stale, "ic_recent": bias_recent,
                    "reversed": bool(bias_stale * bias_recent < 0)},
    }

    today = datetime.now().strftime("%Y%m%d")
    REP.mkdir(parents=True, exist_ok=True)
    (REP / f"reweight_quicktest_{today}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # markdown
    md = []
    md.append(f"# P1 重加权 quick test（{today}）\n")
    md.append(f"- 测试窗口（样本外）：**{out['test_window']}**；陈旧权重窗 {out['stale_window']}；近期权重窗 {out['recent_window']}")
    md.append(f"- 口径：线性秩集成 score=Σ w_f·rank(f)；daily rank-IC = 每日 score 秩 vs 超额收益秩的 Spearman；"
              "多空=每日 top10%−bottom10% 的超额收益均值。\n")
    md.append("## 一、各变体近期样本外表现\n")
    md.append("| 变体 | IC均值 | ICIR | 多空均值 | 多空t | 多空夏普(年化) |")
    md.append("|---|---|---|---|---|---|")
    order = ["V_stale", "V_recent", "V_recent_nobias", "V_voltilt"]
    for k in order:
        r = results[k]
        md.append("| %s | %.4f | %.3f | %.4f | %.2f | %.2f |" % (
            k, r["ic_mean"], r["icir"], r["ls_mean"], r["ls_t"], r["ls_sharpe_ann"]))
    md.append("\n## 二、生产信号 benchmark（同窗口 daily rank-IC）\n")
    md.append("| 信号 | IC均值 |")
    md.append("|---|---|")
    for k, v in bench.items():
        md.append(f"| {k} | {v:.4f} |")
    md.append("\n## 三、bias_5 反转对账\n")
    md.append(f"- IC_陈旧={bias_stale:+.4f} / IC_近期={bias_recent:+.4f} / 反转={'是⚠' if bias_stale*bias_recent<0 else '否'}")
    md.append("\n## 四、结论要点\n")
    best = max(results, key=lambda k: results[k]["ic_mean"])
    md.append(f"- **近期权重(V_recent) 相对陈旧权重(V_stale) IC：{results['V_recent']['ic_mean']:+.4f} vs {results['V_stale']['ic_mean']:+.4f}**"
              f"（{'改善' if results['V_recent']['ic_mean']>results['V_stale']['ic_mean'] else '恶化'}）。")
    md.append(f"- 剔除 bias_5(V_recent_nobias)：IC={results['V_recent_nobias']['ic_mean']:+.4f} —— "
              f"{'bias_5(近期方向)有正贡献' if results['V_recent_nobias']['ic_mean']<results['V_recent']['ic_mean'] else 'bias_5是噪声/拖累'}。")
    md.append(f"- 极端向波动率族倾斜(V_voltilt)：IC={results['V_voltilt']['ic_mean']:+.4f}。")
    md.append(f"- 与生产信号比：best 变体 **{best}** IC={results[best]['ic_mean']:+.4f}，"
              f"生产 ens/baseline/gru = {bench.get('ens',0):.4f}/{bench.get('baseline',0):.4f}/{bench.get('gru',0):.4f}。")
    md.append("\n> 说明：本测试是因子级线性秩集成（非神经网络），仅用于验证『重加权/翻号』方向是否成立；"
             "生产模型为非线性融合，本结果不能直接等同重训收益，但能指明该不该往这个方向走。")
    (REP / f"reweight_quicktest_{today}.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n=== 重加权 quick test（{today}）===")
    for k in order:
        r = results[k]
        print(f"  {k:18s} IC={r['ic_mean']:+.4f} ICIR={r['icir']:.3f} LS_t={r['ls_t']:.2f} LS_sharpe={r['ls_sharpe_ann']:.2f}")
    print("  生产 benchmark IC:", {k: round(v, 4) for k, v in bench.items()})
    print(f"  bias_5: IC_stale={bias_stale:+.4f} IC_recent={bias_recent:+.4f} reversed={bias_stale*bias_recent<0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
