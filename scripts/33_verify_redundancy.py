"""阶段 45 实证核验：②b 新增因子（log_mktcap_real / turnover_daily）是否冗余。

背景：09-14 裁定 ②b 终止，理由——
  (a) log_mktcap_real ≡ log_amount − log(turnover_daily) − 常数（代数共线）
  (b) turnover_daily 在 h10 为负
但用户 09-15「启动」又要求 baostock 拉真实估值→补市值/换手因子。
本脚本读取 32 刚生成的 v40（含 log_mktcap_real + turnover_daily），做实证冗余/价值核验；
不重复无效劳动、不粉饰。

核验三项：
  1. 三因子 pooled Pearson/Spearman 相关矩阵
  2. 冗余证明：用 log_amount + log(turnover_daily) 线性重建 log_mktcap_real 的 R²
  3. 每个因子对标签（forward excess return）的 IC/ICIR（独立信息量）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.eval.metrics import daily_ic, ic_summary  # noqa: E402

PROC = ROOT / "data" / "P1" / "processed"
SRC = PROC / "dataset_h10_v40.parquet"


def main() -> int:
    print(f"[33] 读 {SRC.name} schema ...", flush=True)
    schema = pq.read_schema(SRC)
    cols = [f.name for f in schema]
    labels = [c for c in cols if c.startswith("y_")]
    ycol = "y_excess" if "y_excess" in cols else (labels[0] if labels else None)
    print(f"[33] 候选标签列: {labels}  -> 采用 y_col={ycol}", flush=True)
    if ycol is None:
        print("[33] 无标签列，跳过 IC 核验")
        return 1

    need = ["date", ycol, "log_amount", "log_mktcap_real", "turnover_daily"]
    df = pd.read_parquet(SRC, columns=[c for c in need if c in cols])
    print(f"[33] 载入 {len(df):,} 行", flush=True)

    fac = ["log_amount", "log_mktcap_real", "turnover_daily"]
    sub = df[fac].astype(float)

    print("\n=== (1) pooled Pearson 相关 ===")
    print(sub.corr().round(4).to_string())
    print("\n=== (1b) pooled Spearman 相关 ===")
    print(sub.corr(method="spearman").round(4).to_string())

    # (2) 冗余证明
    print("\n=== (2) 冗余证明：log_mktcap_real 能否被 log_amount+log(turnover) 线性重建 ===")
    lt = np.log(sub["turnover_daily"].clip(lower=1e-6)).to_numpy()
    la = sub["log_amount"].to_numpy()
    yv = sub["log_mktcap_real"].to_numpy()
    X = np.vstack([np.ones_like(la), la, lt]).T
    mask = np.isfinite(X).all(1) & np.isfinite(yv)
    coef, *_ = np.linalg.lstsq(X[mask], yv[mask], rcond=None)
    pred = X[mask] @ coef
    ss_res = float(((yv[mask] - pred) ** 2).sum())
    ss_tot = float(((yv[mask] - yv[mask].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"  log_mktcap_real ≈ {coef[0]:.4f} + {coef[1]:.4f}*log_amount "
          f"+ {coef[2]:.4f}*log(turnover_daily)")
    print(f"  R² = {r2:.6f}  （≈1.0 ⇒ 完全冗余：模型已有 log_amount 即涵盖 log_mktcap_real）")

    # (3) 各因子独立 IC
    print(f"\n=== (3) 各因子 vs {ycol} 的 IC（独立信息量）===")
    rows = []
    for fc in fac:
        ic = daily_ic(df, pred_col=fc, y_col=ycol)
        s = ic_summary(ic)
        rows.append((fc, s["ic_mean"], s["icir"], s["icir_annual"],
                     s["ic_positive_rate"], s["t_stat"], s["n_days"]))
        print(f"  {fc:18s} IC={s['ic_mean']:+.4f} ICIR={s['icir']:+.4f} "
              f"年化ICIR={s['icir_annual']:+.2f} 正占比={s['ic_positive_rate']:.3f} "
              f"t={s['t_stat']:.2f} n={s['n_days']}")

    # 结论判定
    print("\n=== 结论判定 ===")
    redundant = r2 > 0.999
    print(f"  log_mktcap_real 冗余（R²>{0.999}）: {redundant}")
    # turnover_daily 是否仍有独立 IC（>0.02 视为有信息）
    t_ic = dict((r[0], r[1]) for r in rows)["turnover_daily"]
    print(f"  turnover_daily 独立 IC={t_ic:+.4f} "
          f"（|IC|>0.02 视为有独立信息，否则与终止裁定一致）")
    if redundant and abs(t_ic) <= 0.02:
        print("  >>> 与 09-14 ②b 终止裁定一致：两因子均无独立增量，不应并入模型。")
    elif redundant and abs(t_ic) > 0.02:
        print("  >>> log_mktcap_real 仍冗余，但 turnover_daily 有独立 IC ⇒ 仅该因子可部分复活 ②b。")
    else:
        print("  >>> 与 09-14 裁定不符，需人工复核。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
