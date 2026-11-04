"""阶段 ZZ：ens 信号导出（S09 专用，替代推送链内联 python）。

2026-11-01 起 ens 生产权重 w_gru=0.00（37_build_ensemble_pred.py 默认），
信号 json 文件名保持 signal_ens_v39gru_w025_h10.json 兼容下游（ks gen_morning_event_sentiment 等）。

可选能力（方案二，老板拍板"都做"落地）：--exclude-gru-unique
  - 规则：每日 top20 中剔除"gru 当日 top10 且 baseline/ev/fusion 均未选"的票（gru 独有票），
    剔除后按原分补足。反事实显示该规则有增量（+9.94%→+11.31%，10/49 日、11 笔）。
  - 注意：w_gru=0.00 时 ens 退化为纯 v39 序，gru 独有票不在 ens 排名前列，规则自然不触发；
    仅当 --w-gru>0 保留部分 gru 时该开关才生效。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths  # noqa: E402
from src.signal import export_to_file  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
SIGNAL_DIR = PROCESSED / "signals"
ENS_PRED = PROCESSED / "pred_ens_v39gru_w025_h10.parquet"
ENS_JSON = SIGNAL_DIR / "signal_ens_v39gru_w025_h10.json"
SIGNAL_FILES = {
    "gru": "signal_gru_h10.json", "baseline": "signal_baseline_h10.json",
    "ev": "signal_ev_h10.json", "fusion": "signal_fusion_h10.json",
}
K = 10


def topk_set(daily: list[dict], date: str, k: int = K) -> set[str]:
    rows = [d for d in daily if d["date"] == date]
    rows.sort(key=lambda x: -x["score"])
    return {r["symbol"] for r in rows[:k]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude-gru-unique", action="store_true",
                    help="剔除 ens top20 中属 gru 独有票的成员（w_gru>0 时生效；默认关）")
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ens = pd.read_parquet(ENS_PRED)
    ens["date"] = pd.to_datetime(ens["date"]).dt.normalize()
    out_path = Path(args.out) if args.out else ENS_JSON

    if args.exclude_gru_unique:
        gd = json.loads((SIGNAL_DIR / SIGNAL_FILES["gru"]).read_text(encoding="utf-8"))["daily"]
        others = {k: json.loads((SIGNAL_DIR / v).read_text(encoding="utf-8"))["daily"]
                  for k, v in SIGNAL_FILES.items() if k != "gru"}
        dates = sorted(ens["date"].dt.strftime("%Y-%m-%d").unique())
        drop_rows: set[tuple[str, str]] = set()
        n_excl, n_days = 0, 0
        ens_rows = [{"date": d.strftime("%Y-%m-%d"), "symbol": s, "score": v}
                    for d, s, v in zip(ens["date"], ens["symbol"], ens["pred"])]
        for dt in dates:
            g = topk_set(gd, dt)
            other_union = set()
            for _, od in others.items():
                other_union |= topk_set(od, dt)
            gru_unique = g - other_union
            ens_top = topk_set(ens_rows, dt, args.top_n)
            hit = gru_unique & ens_top
            n_excl += len(hit)
            n_days += (1 if hit else 0)
            drop_rows |= {(dt, s) for s in hit}
        print(f"[ens] 剔除 gru 独有票：{n_days} 个信号日触发、共剔除 {n_excl} 笔 top{args.top_n} 成员")
        if drop_rows:
            ens["dstr"] = ens["date"].dt.strftime("%Y-%m-%d")
            drops = pd.DataFrame(sorted(drop_rows), columns=["dstr", "symbol"])
            ens = ens.merge(drops, on=["dstr", "symbol"], how="left", indicator=True)
            ens = ens[ens["_merge"] == "left_only"].drop(columns=["_merge", "dstr"])

    sig = export_to_file(ens, out_path, top_n=args.top_n,
                         model="ens_v39gru_w025", horizon=10)
    print(f"[ens] 信号已导出: {out_path}")
    print(f"[ens] 最新交易日 {sig['latest_date']} | Top 看多 {len(sig['top_long'])} | Top 看空 {len(sig['top_short'])}")
    for r in sig["top_long"][:10]:
        print(f"  {r['symbol']}  分数={r['pred']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
