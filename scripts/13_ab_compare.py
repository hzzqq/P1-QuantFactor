"""阶段 13：GRU vs regime 融合 信号 A/B 对比（供 StockSignal ingestion 决策）。

比较 `signal_gru_h10.json`（incumbent）与 `signal_fusion_h10.json`（regime 门控候选）：
- 最新交易日 top_long / top_short 的重叠度（Jaccard）
- 重叠个股在两份信号里的排序相关性（pct-rank 的 Pearson，等价于 Spearman）
- 分歧个股（GRU 看多但融合不看多，反之亦然）
- 近期 daily 信号的「非中性占比」——直接反映 regime 门控是否如预期在湍流期收缩仓位

输出：report_ab_compare.json（结构化的 A/B 指标）+ 终端可读表格。

用法：
    python scripts/13_ab_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths

PROCESSED = paths.DATA / "P1" / "processed"
SIG = PROCESSED / "signals"


def load(name: str) -> dict:
    return json.loads((SIG / name).read_text(encoding="utf-8"))


def top_set(sig: dict, kind: str) -> set[str]:
    return {r["symbol"] for r in sig.get(kind, [])}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> int:
    gru = load("signal_gru_h10.json")
    fus = load("signal_fusion_h10.json")

    gru_long, fus_long = top_set(gru, "top_long"), top_set(fus, "top_long")
    gru_short, fus_short = top_set(gru, "top_short"), top_set(fus, "top_short")

    # 重叠个股的 pct-rank 排序相关性
    gru_lr = {r["symbol"]: r["rank"] for r in gru["top_long"]}
    fus_lr = {r["symbol"]: r["rank"] for r in fus["top_long"]}
    ov = set(gru_lr) & set(fus_lr)
    rank_corr = (pearson(np.array([gru_lr[s] for s in ov]),
                         np.array([fus_lr[s] for s in ov]))
                 if ov else float("nan"))

    # 分歧
    gru_only_long = sorted(gru_long - fus_long)
    fus_only_long = sorted(fus_long - gru_long)
    gru_only_short = sorted(gru_short - fus_short)
    fus_only_short = sorted(fus_short - gru_short)

    # daily 非中性占比（近期窗口）
    def non_neutral_frac(sig: dict) -> float:
        daily = sig.get("daily", [])
        if not daily:
            return float("nan")
        nn = sum(1 for d in daily if d["signal"] != "中性")
        return nn / len(daily)

    gru_nn = non_neutral_frac(gru)
    fus_nn = non_neutral_frac(fus)

    report = {
        "latest_date": gru["latest_date"],
        "horizon": gru.get("horizon"),
        "top_long_jaccard": jaccard(gru_long, fus_long),
        "top_short_jaccard": jaccard(gru_short, fus_short),
        "overlap_long_n": len(gru_long & fus_long),
        "overlap_short_n": len(gru_short & fus_short),
        "rank_corr_overlap_long": rank_corr,
        "gru_only_long": gru_only_long,
        "fus_only_long": fus_only_long,
        "gru_only_short": gru_only_short,
        "fus_only_short": fus_only_short,
        "gru_non_neutral_frac": gru_nn,
        "fus_non_neutral_frac": fus_nn,
        "fusion_position_shrink_ratio": (fus_nn / gru_nn) if gru_nn else None,
    }

    out = PROCESSED / "report_ab_compare.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print("\n## GRU vs regime 融合 信号 A/B 对比（StockSignal ingestion 决策）\n")
    print(f"最新交易日：{report['latest_date']} ｜ horizon={report['horizon']}")
    print(f"  top_long  Jaccard = {report['top_long_jaccard']:.3f}  "
          f"（重叠 {report['overlap_long_n']} 只）")
    print(f"  top_short Jaccard = {report['top_short_jaccard']:.3f}  "
          f"（重叠 {report['overlap_short_n']} 只）")
    print(f"  重叠个股排序相关性(rank corr) = {rank_corr:.3f}")
    print(f"  daily 非中性占比：GRU {gru_nn:.3f} → 融合 {fus_nn:.3f}  "
          f"（仓位收缩比 {report['fusion_position_shrink_ratio']:.3f}）")
    print(f"\n  GRU 独有看多（融合未选）：{gru_only_long}")
    print(f"  融合独有看多（GRU 未选）：{fus_only_long}")
    print(f"  GRU 独有看空（融合未选）：{gru_only_short}")
    print(f"  融合独有看空（GRU 未选）：{fus_only_short}")
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
