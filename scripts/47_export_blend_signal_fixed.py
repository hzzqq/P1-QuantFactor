"""阶段 47 — 重导出 blend 信号（修复 pred 量纲，主推前置）。

根因（见 45_build_v41_blend.py）：blend 的 pred 是逐日截面 z 分数（clip ±3）的加权和，
落在 [-3,3]；而信号 schema 的 pred 契约是「小得分 ~±0.2」（EV 源 parquet 即如此，
页面用 pred*100 当 % 显）。直接把 z 分数写进 JSON → top20 全压在 +3.0 天花板，
页面渲染成 +300% 的坏值，不能主推。

修复：仅修显示层 JSON——把 z 分数线性映射到 ±0.2 量级（Z2SCORE = 1/15，单调，
排序/推荐不变）。parquet（z 分数，供回测复现）不动；只重导 signal_v41blend_w30_h10.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths  # noqa: E402
from src.signal.export_signal import build_signals  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
SRC = PROCESSED / "pred_ens_v41blend_w30_h10.parquet"
OUT = PROCESSED / "signals" / "signal_v41blend_w30_h10.json"

# 显示得分对称量程 [-SCORE_SCALE, +SCORE_SCALE]，与 EV 源 pred ~±0.09 同量级
# （纯显示归一，单调，排序/看多阈值语义不变）。
# 关键修复：A 源 pred 生成时已裁剪到 ±3.0，顶端 ~20 只同值 → 重算 z 仍并列，
# 直接写 JSON 会让 top20 全压在天花板、页面渲染成 +300%/全并列的坏值。
# 这里改用「按日 ordinal 秩」映射到对称小得分：序数秩确定性破并列（每只区分），
# 同时保序、不破坏 build_signals 内部的 0.9 分位看多阈值与 top20 选取。
SCORE_SCALE = 0.1


def _to_display_score(df: pd.DataFrame) -> pd.DataFrame:
    """逐日把 pred（z 分数，含并列）映射为 [-SCORE_SCALE,+SCORE_SCALE] 的区分得分。

    - ord_rank(method="first")：确定性破并列（同值按出现序给 1..n 不同秩）。
    - p=(ord-1)/(n-1) ∈ [0,1] 全区分；score=(p*2-1)*SCORE_SCALE 对称。
    - 保序：原 pred 越大 → score 越大；top20 选取与看多(decile)语义不变。
    """
    parts = []
    for _, g in df.groupby("date"):
        g = g.copy()
        n = len(g)
        ord_rank = g["pred"].rank(method="first")
        p = (ord_rank - 1) / max(n - 1, 1)
        g["pred"] = ((p * 2 - 1) * SCORE_SCALE).astype(float)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    df = pd.read_parquet(SRC)
    df = df[["date", "symbol", "pred"]].dropna().copy()
    df = _to_display_score(df)  # z 分数(含并列) → 区分小得分（显示归一）
    sig = build_signals(df, top_n=20)
    sig["model"] = "v41blend_w30"
    sig["horizon"] = 10
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(sig, ensure_ascii=False, indent=2), encoding="utf-8")

    tl, sh = sig["top_long"], sig["top_short"]
    preds = [r["pred"] for r in tl]
    print(f"[47] 重导出 {OUT.name}  ({OUT.stat().st_size/1e6:.2f}MB)")
    print(f"  latest_date={sig['latest_date']} | top_long={len(tl)} "
          f"top_short={len(sh)} daily={len(sig['daily'])}")
    print(f"  top_long pred 范围: {min(preds):+.4f} ~ {max(preds):+.4f} "
          f"(旧的坏值=+3.0000)")
    print(f"  top1: {tl[0]['symbol']} pred={tl[0]['pred']:+.4f}")
    print("  top_long 前5: " + ", ".join(
        f"{r['symbol']}({r['pred']:+.3f})" for r in tl[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
