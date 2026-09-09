"""阶段 6：把 P1 预测导出为 StockSignal 可 ingestion 的信号文件。

用法：
    python scripts/06_export_signal.py
    python scripts/06_export_signal.py --pred data/P1/processed/pred_gru_h10.parquet --model gru
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths
from shared.logging_utils import get_logger

from src.signal import export_to_file

logger = get_logger("P1.export_signal")
PROCESSED = paths.DATA / "P1" / "processed"


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 → StockSignal 信号导出")
    ap.add_argument("--pred", default=None)
    ap.add_argument("--model", default="baseline_lgb")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()

    _model_file = {"gru": "gru", "baseline": "baseline",
                   "fusion": "fusion"}.get(args.model, "baseline")
    pred_path = Path(args.pred) if args.pred else (
        PROCESSED / f"pred_{_model_file}_h{args.horizon}.parquet"
    )
    if not pred_path.exists():
        logger.error("找不到 %s", pred_path)
        return 1

    preds = pd.read_parquet(pred_path)
    out = PROCESSED / "signals" / f"signal_{args.model}_h{args.horizon}.json"
    sig = export_to_file(preds, out, top_n=args.top_n,
                         model=args.model, horizon=args.horizon)
    logger.info("信号已导出: %s", out)
    logger.info("最新交易日 %s | Top 看多 %d 只 | Top 看空 %d 只",
                sig["latest_date"], len(sig["top_long"]), len(sig["top_short"]))
    print(f"\n最新交易日 {sig['latest_date']} · Top 看多个股：")
    for r in sig["top_long"][:10]:
        print(f"  {r['symbol']}  分数={r['pred']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
