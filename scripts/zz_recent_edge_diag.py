"""P1 边缘衰减诊断（2026-11-11）：近期弱势是「真边缘衰减」还是「变现/构造问题」？

用 FULL-HISTORY 的 pred_*.parquet（gru/baseline/ens，2021~2026-11-09）直接测
模型级 long-short 边缘（top10% − bottom10% 的 y_excess 价差），横跨多个时间窗，
确认边缘是否近期塌缩。同时测 long-only 超额（top10% − 全池均值）对照 paper portfolio。

口径：
  - y_excess = 前瞻 horizon 日超额收益（与 IC 同一定义），直接用，无需 kline，快且可比。
  - 每交易日按 pred 秩取 top10%(多) / bottom10%(空)；LS = mean(y|多) − mean(y|空)；
    LongXS = mean(y|多) − mean(y|全池)。
  - LS/longXS 的 NAV = ∏(1+daily)，t = mean/(std/√n)。
输出 reports/recent_edge_diag_YYYYMMDD.md。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("E:/project/sj/data/P1/processed")
PREDS = {
    "gru": "pred_gru_h10.parquet",
    "baseline": "pred_baseline_h10.parquet",
    "ens": "pred_ens_v39gru_w025_h10.parquet",
}
WINDOWS = {
    "FULL(2021+)": "2021-01-01",
    "2024+": "2024-01-01",
    "2026": "2026-01-01",
    "近59日(2026-08-11+)": "2026-08-11",
}


def analyze(pred: pd.DataFrame, lo: str) -> dict:
    d = pred[pred["date"] >= lo].copy()
    # 每交易日 pred 秩（pct）
    d["_r"] = d.groupby("date")["pred"].rank(pct=True)
    g = d.groupby("date")
    long_m = g.apply(lambda x: x.loc[x["_r"] >= 0.9, "y_excess"].mean(), include_groups=False)
    short_m = g.apply(lambda x: x.loc[x["_r"] <= 0.1, "y_excess"].mean(), include_groups=False)
    all_m = g["y_excess"].mean()
    ls = (long_m - short_m).dropna()
    longxs = (long_m - all_m).dropna()
    # IC: 每日 pred 秩 vs y_excess 秩
    d["_yr"] = d.groupby("date")["y_excess"].rank(pct=True)
    ic = d.groupby("date").apply(lambda x: x["_r"].corr(x["_yr"]), include_groups=False).dropna()
    n = len(ls)

    def nav(s: pd.Series):
        return float((1.0 + s).prod() - 1.0)

    def tstat(s: pd.Series):
        return float(s.mean() / (s.std() / np.sqrt(len(s)))) if s.std() and s.std() > 0 else float("nan")

    return {
        "ic_mean": float(ic.mean()), "icir": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if ic.std() and ic.std() > 0 else float("nan"),
        "ls_nav": nav(ls), "ls_t": tstat(ls), "ls_mean": float(ls.mean()),
        "longxs_nav": nav(longxs), "longxs_t": tstat(longxs), "longxs_mean": float(longxs.mean()),
        "n_days": int(n),
    }


def main() -> int:
    # y_excess 查找表（ens 文件无该列，从 dataset 合并）
    ds = pd.read_parquet(PROC / "dataset_h10.parquet", columns=["date", "symbol", "y_excess"])
    ds["date"] = pd.to_datetime(ds["date"])
    y_lookup = ds.dropna(subset=["y_excess"])
    rows = []
    for model, fn in PREDS.items():
        pred = pd.read_parquet(PROC / fn)
        pred["date"] = pd.to_datetime(pred["date"])
        if "y_excess" not in pred.columns:
            pred = pred.merge(y_lookup, on=["date", "symbol"], how="inner")
        pred = pred.dropna(subset=["y_excess"])
        for wname, lo in WINDOWS.items():
            r = analyze(pred, lo)
            r["model"] = model
            r["window"] = wname
            rows.append(r)

    today = datetime.now().strftime("%Y%m%d")
    rep = PROC / "reports"
    rep.mkdir(parents=True, exist_ok=True)

    # 控制台
    print("=== 模型级边缘衰减（IC / long-short 边缘t / long-only 超额t）===")
    for r in rows:
        print("  %-9s %-18s IC=%.4f LS_t=%+.2f LongXS_t=%+.2f" % (
            r["model"], r["window"], r["ic_mean"], r["ls_t"], r["longxs_t"]))

    # 报告
    md = []
    md.append(f"# P1 边缘衰减诊断（{today}）\n")
    md.append("- 用 FULL-HISTORY pred 文件（gru/baseline/ens，2021~2026-11-09）直接测模型级边缘："
              "每交易日按 pred 秩取 top10%(多)/bottom10%(空)，价差=mean(y|多)−mean(y|空)；"
              "long-only 超额=mean(y|多)−mean(y|全池)。y_excess=前瞻10日超额（与 IC 同定义）。\n")
    md.append("> 注：`LS_nav` 累计是把每日 decile 价差复利 2000+ 日的放大假象（未封顶、未计成本），"
             "**不可当真实收益**；本文以 **IC / LS_t（多空边缘显著性）/ LongXS_t（多头超额显著性）** 为准。\n")
    md.append("## 一、long-short 边缘（多−空 价差，相对排序有效性）\n")
    md.append("| 模型 | 窗口 | IC | LS_t | LS日均 |")
    md.append("|---|---|---|---|---|")
    for r in rows:
        md.append("| %s | %s | %.4f | %+.2f | %.4f |" % (
            r["model"], r["window"], r["ic_mean"], r["ls_t"], r["ls_mean"]))
    md.append("\n## 二、long-only 超额（top10% − 全池均值，对照 paper portfolio 口径）\n")
    md.append("| 模型 | 窗口 | LongXS_t | LongXS日均 |")
    md.append("|---|---|---|---|")
    for r in rows:
        md.append("| %s | %s | %+.2f | %.4f |" % (
            r["model"], r["window"], r["longxs_t"], r["longxs_mean"]))
    md.append("\n## 三、结论要点（诚实、修正之前两版临时结论）\n")
    gru_full = next(r for r in rows if r["model"] == "gru" and r["window"] == "FULL(2021+)")
    gru_recent = next(r for r in rows if r["model"] == "gru" and r["window"] == "近59日(2026-08-11+)")
    ens_recent = next(r for r in rows if r["model"] == "ens" and r["window"] == "近59日(2026-08-11+)")
    ens_full = next(r for r in rows if r["model"] == "ens" and r["window"] == "FULL(2021+)")
    md.append(f"- **近期弱势是真实边缘衰减，且分模型不同步**："
              f"gru 在最近 59 日 **IC={gru_recent['ic_mean']:+.4f}、LS_t={gru_recent['ls_t']:+.2f}（显著反号）**"
              f"——已从历史正边缘（FULL IC={gru_full['ic_mean']:+.4f}）**逆转成反向预测**，正在主动亏钱；"
              f"baseline 边缘衰减到≈0；**ens 仍保留多空边缘**（近59日 IC={ens_recent['ic_mean']:+.4f}、LS_t={ens_recent['ls_t']:+.2f}）。")
    md.append(f"- **IC→P&L 断层在 long-only 构造**：ens 多空边缘仍在（LS_t={ens_recent['ls_t']:+.2f}>0），"
              f"但 **long-only 超额 LongXS_t={ens_recent['longxs_t']:+.2f}（转负）**——相对排序仍有效，"
              f"但全池 1492 只等权近期跑赢被选中的多头，导致纯多头组合亏钱。这与 paper portfolio 近59日 −15%~−33% 超额一致。")
    md.append(f"- **成本不是主因**：monetization_diag 实测得 0→0.4% 双边仅多拖 ~1.4%（59 日窗口），远低于边缘本身损失。")
    md.append(f"- **之前临时结论订正**：①「gru 是最好的一条腿、不必砍」**错误**——那是比 59 日组合净值时 gru 看似最不差，"
              f"但看 IC 它已显著反号（−0.177），**现应从 ens 降权/移除**；②「成本硬伤/年化 4.9%~7.3%」**高估**，实证仅 ~1.4%。")
    md.append("\n## 四、修正后的可执行项\n")
    md.append("1. **立即降权/移除 gru**：它近 59 日已反号（IC −0.177），继续留在 ens 只会拖累；保留 ens/baseline 中仍有效的多空边缘。")
    md.append("2. **改 long-only 构造**：IC 测的是相对排序，纯多头 top-k 在宽基 rally 期必跑输全池；"
              "改「基准相对/市场中性」或降频低换手，匹配边缘的可变现形态（A 股做空受限，long-only 需另有对策）。")
    md.append("3. **试更长 horizon / 换收益目标**：10 日超额边缘太薄；试 20/60 日看边缘是否变厚。")
    md.append("4. **重加权/翻 bias_5 仍不做**（reweight_quicktest 已证空间极小）；真正杠杆在 ① 砍 gru ② 改组合构造。")
    (rep / f"recent_edge_diag_{today}.md").write_text("\n".join(md), encoding="utf-8")
    (rep / f"recent_edge_diag_{today}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告: {rep}/recent_edge_diag_{today}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
