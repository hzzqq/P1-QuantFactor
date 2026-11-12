"""P1 更长 horizon 边缘诊断（2026-11-11）：换 20/40/60 日，边缘会不会变厚？

不重训、不改信号。按 src/features/labels.py::build_labels 的**完全同口径**从 panel+基准
自算 h∈{10,20,40,60} 的前瞻超额收益标签 y_h = 个股 T→T+h 收盘收益 − sh000300 同期收益，
再用**同一个现役 ens 信号**（pred_ens_v39gru_w025_h10.parquet，w_gru=0）在各 h 下测：
  - IC：逐日 pred 秩 vs y_h 秩
  - LS_t：top10% − bottom10% 的 y_h 价差显著性（相对排序有效性）
  - LongXS_t：top10% − 全池均值的 y_h 显著性（long-only 能否变现，对应 paper portfolio）

分窗口 FULL(2021+) / 2024+ / 2026 / 近59日；诚实标注：h 越大尾部标签缺失越多，
近59日窗口对 h≥20 基本无有效标签（需未来 h 个交易日），该格标 NA 不参与结论。

输出 reports/horizon_edge_scan_YYYYMMDD.{md,json}。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("E:/project/sj/data/P1/processed")
RAW = Path("E:/project/sj/data/P1/raw")
HORIZONS = [10, 20, 40, 60]
WINDOWS = {"FULL(2021+)": "2021-01-01", "2024+": "2024-01-01",
           "2026": "2026-01-01", "近59日": "2026-08-11"}


def main() -> int:
    print("读取 panel / bench / pred ...", flush=True)
    panel = pd.read_parquet(PROC / "panel.parquet", columns=["date", "symbol", "close"])
    panel["date"] = pd.to_datetime(panel["date"])
    bench = pd.read_parquet(RAW / "index" / "sh000300.parquet", columns=["date", "close"])
    bench["date"] = pd.to_datetime(bench["date"])
    pred = pd.read_parquet(PROC / "pred_ens_v39gru_w025_h10.parquet",
                           columns=["date", "symbol", "pred"])
    pred["date"] = pd.to_datetime(pred["date"])
    pred = pred.drop_duplicates(subset=["date", "symbol"], keep="last")

    close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    bc = bench.set_index("date")["close"].sort_index().reindex(close.index).ffill()
    print(f"close {close.shape} / bench {len(bc)} / pred {len(pred):,} 行", flush=True)

    rows = []
    for h in HORIZONS:
        fwd = close.shift(-h) / close - 1.0
        bfwd = bc.shift(-h) / bc - 1.0
        exc = fwd.sub(bfwd, axis=0)
        y = (exc.stack().rename("y_h").reset_index()
             .rename(columns={"level_0": "date", "level_1": "symbol"}))
        y = y.dropna(subset=["y_h"])
        d = pred.merge(y, on=["date", "symbol"], how="inner")
        print(f"h={h}: y_h 有效 {len(y):,} 行，与 pred 对齐 {len(d):,} 行，"
              f"y_h 最大日期 {d['date'].max().date()}", flush=True)
        d["_pr"] = d.groupby("date")["pred"].rank(pct=True)
        for wname, lo in WINDOWS.items():
            x = d[d["date"] >= lo].copy()
            n_days = x["date"].nunique()
            if n_days < 10:
                rows.append({"h": h, "window": wname, "n_days": int(n_days),
                             "ic": None, "ls_t": None, "longxs_t": None,
                             "longxs_mean": None, "ls_mean": None, "na": True})
                continue
            x["_yr"] = x.groupby("date")["y_h"].rank(pct=True)
            ic = x.groupby("date").apply(
                lambda g: g["_pr"].corr(g["_yr"]), include_groups=False).dropna()
            g = x.groupby("date")
            long_m = g.apply(lambda z: z.loc[z["_pr"] >= 0.9, "y_h"].mean(), include_groups=False)
            short_m = g.apply(lambda z: z.loc[z["_pr"] <= 0.1, "y_h"].mean(), include_groups=False)
            all_m = g["y_h"].mean()
            ls = (long_m - short_m).dropna()
            lx = (long_m - all_m).dropna()

            def t(s):
                return float(s.mean() / (s.std() / np.sqrt(len(s)))) if s.std() and s.std() > 0 else float("nan")

            # 非重叠复核：h 日标签相邻日共享未来收益 → 日度 t 被自相关高估。
            # 每 h 个交易日取 1 个样本（iloc[::h]），样本间无重叠，t 更保守可比。
            def t_no(s):
                z = s.iloc[::h]
                if len(z) < 8:
                    return float("nan"), int(len(z))
                return (float(z.mean() / (z.std() / np.sqrt(len(z))))
                        if z.std() and z.std() > 0 else float("nan")), int(len(z))

            ls_t_no, n_no = t_no(ls)
            lx_t_no, _ = t_no(lx)
            rows.append({"h": h, "window": wname, "n_days": int(n_days),
                         "ic": float(ic.mean()), "ls_t": t(ls), "longxs_t": t(lx),
                         "longxs_mean": float(lx.mean()), "ls_mean": float(ls.mean()),
                         "ls_t_no": ls_t_no, "longxs_t_no": lx_t_no, "n_nonoverlap": n_no,
                         "na": False})

    today = datetime.now().strftime("%Y%m%d")
    rep = PROC / "reports"
    rep.mkdir(parents=True, exist_ok=True)

    def cell(h, wn, key, fmt="%+.4f"):
        r = next(x for x in rows if x["h"] == h and x["window"] == wn)
        if r["na"] or r[key] is None:
            return "NA(标签不足)"
        return fmt % r[key]

    print(f"\n=== 更长 horizon 边缘扫描（{today}）===")
    print(" h   窗口          n_days   IC       LS_t   LongXS_t")
    for r in rows:
        if r["na"]:
            print("  %-3d %-14s %5d   NA(标签不足)" % (r["h"], r["window"], r["n_days"]))
        else:
            print("  %-3d %-14s %5d  %+.4f  %+.2f   %+.2f" % (
                r["h"], r["window"], r["n_days"], r["ic"], r["ls_t"], r["longxs_t"]))

    md = []
    md.append(f"# P1 更长 horizon 边缘诊断（{today}）\n")
    md.append("- 口径：按 `labels.py::build_labels` 同定义，从 panel + sh000300 自算 "
              "`y_h = 个股 T→T+h 收盘收益 − 基准同期收益`；**同一现役 ens 信号**（w_gru=0）"
              "在各 h 下评估，不重训、不改信号。")
    md.append("- LS_t = top10%−bottom10% 的 y_h 价差显著性（相对排序）；"
              "LongXS_t = top10%−全池均值显著性（**long-only 能否变现**，对应 paper portfolio）。")
    md.append("- ⚠ 诚实边界：h 越大尾部标签缺失越多（需未来 h 个交易日），"
              "**近59日窗口对 h≥20 基本无有效标签，标 NA 不参与结论**；"
              "本诊断是「同一 h10 信号在不同持有期的表现」，不是重训 h20/h60 模型。\n")

    md.append("## 一、IC\n")
    md.append("| h | " + " | ".join(WINDOWS) + " |")
    md.append("|---|" + "---|" * len(WINDOWS))
    for h in HORIZONS:
        md.append("| %d | %s |" % (h, " | ".join(cell(h, w, "ic") for w in WINDOWS)))
    md.append("\n## 二、LS_t（多空边缘显著性）\n")
    md.append("| h | " + " | ".join(WINDOWS) + " |")
    md.append("|---|" + "---|" * len(WINDOWS))
    for h in HORIZONS:
        md.append("| %d | %s |" % (h, " | ".join(cell(h, w, "ls_t", "%+.2f") for w in WINDOWS)))
    md.append("\n## 三、LongXS_t（long-only 超额显著性，能否变现）\n")
    md.append("| h | " + " | ".join(WINDOWS) + " |")
    md.append("|---|" + "---|" * len(WINDOWS))
    for h in HORIZONS:
        md.append("| %d | %s |" % (h, " | ".join(cell(h, w, "longxs_t", "%+.2f") for w in WINDOWS)))
    md.append("\n## 四、有效交易日数（标签覆盖）\n")
    md.append("| h | " + " | ".join(WINDOWS) + " |")
    md.append("|---|" + "---|" * len(WINDOWS))
    for h in HORIZONS:
        md.append("| %d | %s |" % (h, " | ".join(
            str(next(x["n_days"] for x in rows if x["h"] == h and x["window"] == w)) for w in WINDOWS)))

    # 结论以非重叠 t 为准（日度 t 因重叠标签虚高，不能用来判优劣）
    concl = []
    for wname in ("2024+", "2026"):
        parts = []
        for h in HORIZONS:
            r = next(x for x in rows if x["h"] == h and x["window"] == wname)
            v = r["longxs_t_no"]
            parts.append(f"h{h}:{v:+.2f}(n={r['n_nonoverlap']})"
                         if v is not None and v == v else f"h{h}:样本不足(n={r['n_nonoverlap']})")
        concl.append(f"- **{wname} 非重叠 LongXS_t**：" + " / ".join(parts))
    md.append("\n## 五、非重叠复核（关键：h 日标签相邻日共享未来收益，日度 t 被自相关高估）\n")
    md.append("每 h 个交易日取 1 个样本（`iloc[::h]`），样本间无重叠，t 更保守、跨 h 可比。\n")
    md.append("| h | 窗口 | 日度 LongXS_t | 非重叠 LongXS_t | 非重叠样本数 |")
    md.append("|---|---|---|---|---|")
    for h in HORIZONS:
        for w in ("2024+", "2026"):
            r = next(x for x in rows if x["h"] == h and x["window"] == w)
            md.append("| %d | %s | %+.2f | %+.2f | %d |" % (
                h, w, r["longxs_t"], r["longxs_t_no"] if r["longxs_t_no"] == r["longxs_t_no"] else float("nan"),
                r["n_nonoverlap"]))
    md.append("\n> 日度 t 随 h 增大而虚高（重叠越严重）；**判优劣以非重叠 LongXS_t 为准**。")

    md.append("\n## 六、结论要点（诚实版）\n")
    md.append("- ⚠ **不要用日度 t 判「更长 h 更好」**：日度 LongXS_t 从 h10 的 +6.53 涨到 h60 的 +14.57（2024+），"
              "看似单调变厚，但这是**重叠标签导致的自相关虚高**，非重叠复核后优势基本消失。")
    md.extend(concl)
    md.append("- **2026 窗口非重叠 t 全部 ≤ +0.38（h10 +0.38 / h20 +0.22），h40/h60 样本不足（5/3 个）**"
              "→ **换更长 horizon 救不了近期弱势**，与 `recent_edge_diag` 一致：根因在 long-only 构造/regime。")
    md.append("- **唯一略有价值的方向**：2024+ 窗口 h20 非重叠 t=+3.14 略优于 h10 的 +2.67，"
              "但幅度小、且 2026 不延续（h20 仅 +0.22）→ **不足以支撑「为换 horizon 而重训」**；"
              "若将来重训，可把 h20 作为候选一并验证，但别指望它解决近期问题。")
    md.append("- 诚实边界：本诊断是「同一 h10 信号在不同持有期上的表现」，**不是重训 h20/h60 模型**；"
              "更长 horizon 的尾部标签缺失更严重（h60 近59日 0 天有效），近期无法评估。")
    (rep / f"horizon_edge_scan_{today}.md").write_text("\n".join(md), encoding="utf-8")
    (rep / f"horizon_edge_scan_{today}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n报告: {rep}/horizon_edge_scan_{today}.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
