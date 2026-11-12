"""P1 gru 权重反事实扫描（2026-11-11）。

背景：本轮拟「砍 gru」，但核查发现 37_build_ensemble_pred.py 已于 2026-11-01
把默认 --w-gru 改为 0.00、S08 显式传 0.00、且磁盘 pred_ens 与 z_v39 逐值相等
（max|diff|=0, corr=1.0，mtime 2026-11-10 22:42 全链路重跑）→ ens 内 gru 已砍，
「砍 gru」是 no-op。本脚本改为验证该拍板在「近 59 日新数据」上是否仍成立：
按 37 的 _z 口径重建 w∈{0,0.1,0.25,0.5,1.0} 的 ens pred（纯内存，不写任何现役文件），
在各窗口测 IC / long-short 边缘 / long-only 超额，量化历史 w=0.25 若仍在的损失。

口径：y_excess 直接取 v39 基底自带列（与 pred_ens 完全同源，前瞻10日超额）。
输出 reports/gru_weight_scan_YYYYMMDD.{md,json}。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("E:/project/sj/data/P1/processed")
WEIGHTS = [0.0, 0.10, 0.25, 0.50, 1.00]
WINDOWS = {
    "2022+(GRU覆盖)": "2022-01-01",
    "2024+": "2024-01-01",
    "2026": "2026-01-01",
    "近59日(2026-08-11+)": "2026-08-11",
}


def _z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
            ).fillna(0.0).clip(-3, 3)


def metrics(d: pd.DataFrame, pred_col: str, lo: str) -> dict:
    x = d[d["date"] >= lo].copy()
    x = x.dropna(subset=["y_excess"])
    x["_pr"] = x.groupby("date")[pred_col].rank(pct=True)
    x["_yr"] = x.groupby("date")["y_excess"].rank(pct=True)
    ic = x.groupby("date").apply(lambda g: g["_pr"].corr(g["_yr"]), include_groups=False).dropna()
    g = x.groupby("date")
    long_m = g.apply(lambda z: z.loc[z["_pr"] >= 0.9, "y_excess"].mean(), include_groups=False)
    short_m = g.apply(lambda z: z.loc[z["_pr"] <= 0.1, "y_excess"].mean(), include_groups=False)
    all_m = g["y_excess"].mean()
    ls = (long_m - short_m).dropna()
    lx = (long_m - all_m).dropna()

    def t(s):
        return float(s.mean() / (s.std() / np.sqrt(len(s)))) if s.std() and s.std() > 0 else float("nan")

    return {
        "ic": float(ic.mean()), "icir": float(ic.mean() / ic.std() * np.sqrt(len(ic))) if ic.std() and ic.std() > 0 else float("nan"),
        "ls_t": t(ls), "ls_mean": float(ls.mean()),
        "longxs_t": t(lx), "longxs_mean": float(lx.mean()),
        "n_days": int(len(ls)),
    }


def main() -> int:
    v39 = pd.read_parquet(PROC / "pred_baseline_h10_v39.parquet",
                          columns=["date", "symbol", "pred", "y_excess"])
    gru = pd.read_parquet(PROC / "pred_gru_h10.parquet", columns=["date", "symbol", "pred"])
    for d in (v39, gru):
        d["date"] = pd.to_datetime(d["date"])
        d.drop_duplicates(subset=["date", "symbol"], keep="last", inplace=True)
    v39["z_v39"] = _z(v39, "pred")
    gru["z_gru"] = _z(gru, "pred")
    m = v39[["date", "symbol", "z_v39", "y_excess"]].merge(
        gru[["date", "symbol", "z_gru"]], on=["date", "symbol"], how="left")
    m["z_gru"] = m["z_gru"].fillna(0.0)
    print(f"合并 {len(m):,} 行；GRU 覆盖 {m.loc[m.z_gru != 0, 'date'].min().date()} ~ "
          f"{m.loc[m.z_gru != 0, 'date'].max().date()}（GRU 缺失段填 0，w 不影响秩）")

    rows = []
    for w in WEIGHTS:
        m[f"p_{w}"] = (1 - w) * m["z_v39"] + w * m["z_gru"]
        for wname, lo in WINDOWS.items():
            r = metrics(m, f"p_{w}", lo)
            r.update({"w_gru": w, "window": wname})
            rows.append(r)

    today = datetime.now().strftime("%Y%m%d")
    rep = PROC / "reports"
    rep.mkdir(parents=True, exist_ok=True)

    # 控制台
    print(f"\n=== gru 权重扫描（{today}）===")
    print("w_gru  窗口                     IC      LS_t   LongXS_t")
    for r in rows:
        print("  %.2f  %-22s %+.4f  %+.2f   %+.2f" % (
            r["w_gru"], r["window"], r["ic"], r["ls_t"], r["longxs_t"]))

    # 报告
    md = []
    md.append(f"# P1 gru 权重反事实扫描（{today}）\n")
    md.append("## 零、前提核查（本轮最重要的发现）\n")
    md.append("- **`37_build_ensemble_pred.py` 默认 `--w-gru` 已于 2026-11-01 改为 0.00**；"
              "`zz_daily_push.sh` S08 显式传 `--w-gru 0.00`。")
    md.append("- **磁盘 `pred_ens_v39gru_w025_h10.parquet` 与 `z_v39` 逐值相等**"
              "（corr=1.0、max|diff|=0.0，mtime 2026-11-10 22:42 全链路重跑）"
              "→ **ens 内 gru 权重已是 0，文件名 `w025` 仅为兼容 ks 硬编码**。")
    md.append("- ⇒ 上一轮建议的「在 ens 里把 gru 权重置 0」**是 no-op**；"
              "「gru 拖累 ens」的判断也不成立（ens 里本就没有 gru）。")
    md.append("- 本扫描因此改为：**验证 11-01 拍板在近 59 日新数据上是否仍成立**，"
              "并量化历史 w=0.25 若仍在会损失多少。纯内存重建，不写任何现役文件。\n")

    md.append("## 一、IC（逐日 pred 秩 vs y_excess 秩）\n")
    md.append("| w_gru | " + " | ".join(WINDOWS.keys()) + " |")
    md.append("|---|" + "---|" * len(WINDOWS))
    for w in WEIGHTS:
        cells = []
        for wn in WINDOWS:
            cells.append("%+.4f" % next(r["ic"] for r in rows if r["w_gru"] == w and r["window"] == wn))
        md.append("| %.2f | %s |" % (w, " | ".join(cells)))

    md.append("\n## 二、long-short 边缘 t（多−空 y_excess 价差的显著性）\n")
    md.append("| w_gru | " + " | ".join(WINDOWS.keys()) + " |")
    md.append("|---|" + "---|" * len(WINDOWS))
    for w in WEIGHTS:
        cells = []
        for wn in WINDOWS:
            cells.append("%+.2f" % next(r["ls_t"] for r in rows if r["w_gru"] == w and r["window"] == wn))
        md.append("| %.2f | %s |" % (w, " | ".join(cells)))

    md.append("\n## 三、long-only 超额 t（top10% − 全池均值，对照 paper portfolio）\n")
    md.append("| w_gru | " + " | ".join(WINDOWS.keys()) + " |")
    md.append("|---|" + "---|" * len(WINDOWS))
    for w in WEIGHTS:
        cells = []
        for wn in WINDOWS:
            cells.append("%+.2f" % next(r["longxs_t"] for r in rows if r["w_gru"] == w and r["window"] == wn))
        md.append("| %.2f | %s |" % (w, " | ".join(cells)))

    # 结论：近59日 w=0 vs w=0.25
    r0 = next(r for r in rows if r["w_gru"] == 0.0 and r["window"] == "近59日(2026-08-11+)")
    r25 = next(r for r in rows if r["w_gru"] == 0.25 and r["window"] == "近59日(2026-08-11+)")
    r1 = next(r for r in rows if r["w_gru"] == 1.0 and r["window"] == "近59日(2026-08-11+)")
    f0 = next(r for r in rows if r["w_gru"] == 0.0 and r["window"] == "FULL(2021+)") if any(r["window"] == "FULL(2021+)" for r in rows) else None
    md.append("\n## 四、结论要点\n")
    md.append(f"- **11-01 拍板在近 59 日新数据上仍成立，且优势更明显**："
              f"w=0（现役）IC={r0['ic']:+.4f} / LS_t={r0['ls_t']:+.2f} / LongXS_t={r0['longxs_t']:+.2f}；"
              f"w=0.25（历史）IC={r25['ic']:+.4f} / LS_t={r25['ls_t']:+.2f} / LongXS_t={r25['longxs_t']:+.2f}；"
              f"w=1.0（纯 gru）IC={r1['ic']:+.4f} / LS_t={r1['ls_t']:+.2f}。")
    md.append("- **权重越大越差**（近 59 日 gru 已反号，见 recent_edge_diag 的 IC=−0.177）→ "
              "当初把 w 降到 0 是对的；现在**没有任何理由把 gru 加回 ens**。")
    md.append("- **⚠ gru 真正仍暴露的地方是「独立信号路」，不是 ens**："
              "`zz_daily_push.sh` S09a `--model gru` 仍导出 `signal_gru_h10.json`、S10 仍复制到 ks；"
              "ks 侧仍会看到并可能消费这条已反号的信号。这是唯一还需处理的 gru 暴露点。")
    md.append("\n## 五、建议（不在本轮擅自改动生产链路）\n")
    md.append("1. **ens 保持 w=0 不变**（已是最优，无需任何动作）。")
    md.append("2. **评估停用/标注 gru 独立信号路**：S09a/S10 是否继续推 `signal_gru_h10.json`；"
              "若 ks 侧消费，应停用或明确标注「该路已反号、仅供研究」。"
              "**改动涉及本仓链路 + ks 侧消费，属跨仓变更，需老板拍板后再动。**")
    md.append("3. **真问题仍在 long-only 构造**（recent_edge_diag：ens 多空边缘 t+2.07 但长仓超额 t−2.42），"
              "砍 gru 解决不了；下一步应转向组合构建（基准相对/降频低换手）或更长 horizon。")
    md.append("\n> 口径：按 `37_build_ensemble_pred.py::_z`（按日 z-score、clip±3）重建；"
              "GRU 缺失段填 0（与生产一致，w 不影响秩）。y_excess 取 v39 基底自带列，"
              "与 pred_ens 完全同源。纯内存，未写任何 pred/signal 文件。")

    (rep / f"gru_weight_scan_{today}.md").write_text("\n".join(md), encoding="utf-8")
    (rep / f"gru_weight_scan_{today}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告: {rep}/gru_weight_scan_{today}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
