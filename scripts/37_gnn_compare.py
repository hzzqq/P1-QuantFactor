"""阶段 3x：因子图卷积网络(GNN) 与 GRUAttention/Transformer 的 head-to-head 对照（开题清单 #2）。

用法：
    python scripts/37_gnn_compare.py
    python scripts/37_gnn_compare.py --gnn pred_gnn_h10.parquet \
        --gru pred_gru_h10.parquet --transformer pred_transformer_h10.parquet --anchor 0.491

对照指标（与 36_transformer_compare.py 同口径）：
    - 方向命中率：sign(pred)==sign(y_excess)，平盘不计（锚 49.1%）。
    - IC / ICIR：来自 src.eval.metrics。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                       # noqa: E402
import pandas as pd                       # noqa: E402

from shared import paths                  # noqa: E402
from src.eval import metrics             # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
ANCHOR = 0.491  # StockSignal 论文方向命中率（1722/3507）


def direction_hit(df: pd.DataFrame) -> dict:
    """方向命中率：sign(pred)==sign(y_excess)，平盘(y≈0)不计。"""
    pred = df["pred"].to_numpy(dtype=float)
    y = df["y_excess"].to_numpy(dtype=float)
    mask = np.isfinite(pred) & np.isfinite(y) & (y != 0)
    if int(mask.sum()) == 0:
        return {"n": 0, "hit": float("nan")}
    hit = np.sign(pred[mask]) == np.sign(y[mask])
    return {"n": int(mask.sum()), "hit": float(hit.mean())}


def _row(name: str, hit: dict, m: dict, anchor: float) -> dict:
    delta = hit["hit"] - anchor
    return {
        "model": name,
        "n": hit["n"],
        "direction_hit": round(hit["hit"], 4),
        "vs_anchor_delta": round(delta, 4),
        "beats_anchor": bool(delta > 0),
        "ic_mean": round(float(m["ic_mean"]), 4),
        "icir": round(float(m["icir"]), 4),
        "ic_positive_rate": round(float(m["ic_positive_rate"]), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", default="pred_gnn_h10.parquet")
    ap.add_argument("--gru", default="pred_gru_h10.parquet")
    ap.add_argument("--transformer", default="pred_transformer_h10.parquet")
    ap.add_argument("--anchor", type=float, default=ANCHOR)
    args = ap.parse_args()

    table = []
    if (PROCESSED / args.gnn).exists():
        gnn = pd.read_parquet(PROCESSED / args.gnn)
        table.append(_row("FactorGraphGCN", direction_hit(gnn),
                          metrics.summarize(gnn, "pred", "y_excess"), args.anchor))
    else:
        print("跳过 GNN：找不到", PROCESSED / args.gnn)
    if (PROCESSED / args.gru).exists():
        gru = pd.read_parquet(PROCESSED / args.gru)
        table.append(_row("GRUAttention", direction_hit(gru),
                          metrics.summarize(gru, "pred", "y_excess"), args.anchor))
    else:
        print("跳过 GRU：找不到", PROCESSED / args.gru)
    if (PROCESSED / args.transformer).exists():
        t = pd.read_parquet(PROCESSED / args.transformer)
        table.append(_row("TransformerSignal", direction_hit(t),
                          metrics.summarize(t, "pred", "y_excess"), args.anchor))
    else:
        print("跳过 Transformer：找不到", PROCESSED / args.transformer)

    if not table:
        print("无任何预测文件可对照")
        return 1

    out = {"anchor": args.anchor, "compare": table}
    print("\n" + "=" * 70)
    print("  因子图卷积(GNN) vs 基线 · 方向命中率对照（锚=%.3f）" % args.anchor)
    print("=" * 70)
    for r in table:
        print(f"  {r['model']:<18} 命中={r['direction_hit']:.4f} "
              f"(Δ锚 {r['vs_anchor_delta']:+.4f})  IC={r['ic_mean']:.4f} "
              f"ICIR={r['icir']:.4f}  IC正占比={r['ic_positive_rate']:.3f}")
    print("=" * 70 + "\n")

    rp = PROCESSED / "report_gnn_vs_gru.json"
    with rp.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("对照报告已保存:", rp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
