"""阶段 48e — 真实换手率(turn/turnover_daily) 与现有全量因子的冗余核验（扩展版）。

背景
----
`33_verify_redundancy.py` 只对比 log_amount / log_mktcap_real / turnover_daily 三者，
不足以回答决策级问题：「turn 与现有 38(39) 因子是否冗余、是否值得纳入 P1 因子体系」。
本脚本做**全因子**冗余核验，输出决策所需证据：

  (1) turnover_daily / log_mktcap_real 与 **v39 全部既有因子** 的 pooled Pearson/Spearman 相关
      → 找出与 turn 最相关的既有因子（若 |r| 极高 ⇒ turn 是既有因子的线性变换 ⇒ 冗余）
  (2) 逐日横截面平均相关（pooled 会把时间趋势混进来，横截面相关才是选股意义上的冗余）
  (3) R² 冗余证明：用既有因子线性重建 turnover_daily，看能解释多少
  (4) **残差 IC**（决策核心）：把 turnover_daily 对既有因子做逐日横截面 OLS 投影、取残差，
      再算残差对 y_excess 的 IC。
      ⇒ 残差 IC ≈ 0 ⇒ turn 的信息已被既有因子完全覆盖 ⇒ 不值得纳入；
         残差 IC 显著 ⇒ turn 有独立信息，值得纳入。
      （这一步是"是否值得纳入"的直接证据，不依赖任何测试集挑选。）
  (5) 与 log_amount 的 R² 对照（对照 09-14 裁定的代数共线预言）

数据来源与口径由调用方保证（本脚本只读 v40 数据集，不自己取数）。
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

from src.eval.metrics import daily_ic, ic_summary  # noqa: E402

PROC = ROOT / "data" / "P1" / "processed"


def _cs_residual(df: pd.DataFrame, target: str, feats: list[str]) -> pd.DataFrame:
    """逐日横截面 OLS：target ~ 1 + feats，返回残差（未解释部分）。

    逐日做是因为选股因子是横截面概念；pooled 回归会被时间趋势污染。
    """
    out = []
    for _, g in df.groupby("date", sort=False):
        g = g.dropna(subset=[target] + feats)
        if len(g) < len(feats) + 5:
            continue
        X = np.column_stack([np.ones(len(g))] + [g[c].to_numpy(dtype=float) for c in feats])
        y = g[target].to_numpy(dtype=float)
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = y - X @ coef
        sub = g[["date", target]].copy()
        sub["resid"] = resid
        out.append(sub)
    if not out:
        return pd.DataFrame(columns=["date", target, "resid"])
    return pd.concat(out, ignore_index=True)


def main() -> int:
    ds = "dataset_h10_v40.parquet"
    src = PROC / ds
    meta_p = PROC / "dataset_h10_v40_meta.json"
    if not src.exists():
        print(f"[33b] 缺少 {src}；请先跑 32_derive_real_factors.py")
        return 1
    meta = json.load(open(meta_p, encoding="utf-8"))
    all_factors = list(meta["factor_names"])
    new_factors = [f for f in ("log_mktcap_real", "turnover_daily") if f in all_factors]
    old_factors = [f for f in all_factors if f not in new_factors]
    print(f"[33b] 数据集 {ds} | 既有因子 {len(old_factors)} | 新增因子 {new_factors}")

    import pyarrow.parquet as pq
    schema = pq.read_schema(src)
    cols = [f.name for f in schema]
    labels = [c for c in cols if c.startswith("y_")]
    ycol = "y_excess" if "y_excess" in cols else (labels[0] if labels else None)
    if ycol is None:
        print("[33b] 无标签列，无法做 IC 核验")
        return 1
    print(f"[33b] 标签列 = {ycol}")

    need = ["date", ycol] + old_factors + new_factors
    need = [c for c in need if c in cols]
    df = pd.read_parquet(src, columns=need)
    print(f"[33b] 载入 {len(df):,} 行 | 日期 {df['date'].min().date()}~{df['date'].max().date()}"
          f" | 标的 {df['symbol'].nunique() if 'symbol' in df.columns else '?'} 只")

    # ---------- (1) pooled 相关 ----------
    print("\n=== (1) 新因子 vs 既有因子：pooled Pearson 相关（取 |r| 最大的 12 个）===")
    for nf in new_factors:
        sub = df[[nf] + old_factors].astype(float)
        pear = sub.corr()[nf].drop(nf).sort_values(key=abs, ascending=False)
        top = pear.head(12)
        print(f"  [{nf}] pooled Pearson |r| Top12:")
        for k, v in top.items():
            print(f"      {k:22s} r={v:+.4f}  |r|={abs(v):.4f}")

    # ---------- (2) 逐日横截面平均相关 ----------
    print("\n=== (2) 逐日横截面平均相关（Spearman，逐日算后取均值；选股意义上的冗余）===")
    cs_rows = {}
    for nf in new_factors:
        vals = {}
        for oc in old_factors:
            s = df[["date", nf, oc]].dropna()
            if s.empty:
                continue
            cs = s.groupby("date", sort=False).apply(
                lambda g: g[nf].corr(g[oc], method="spearman")
                if len(g) >= 20 else np.nan, include_groups=False)
            vals[oc] = float(np.nanmean(cs))
        cs_rows[nf] = vals
        ser = pd.Series(vals).sort_values(key=abs, ascending=False)
        print(f"  [{nf}] 横截面 Spearman |r| Top12（均值）:")
        for k, v in ser.head(12).items():
            print(f"      {k:22s} r={v:+.4f}  |r|={abs(v):.4f}")

    # ---------- (3)+(5) R² 冗余证明 ----------
    print("\n=== (3) R² 冗余证明 ===")
    r2_log_amount = {}
    for nf in new_factors:
        X = np.column_stack([np.ones(len(df)), df["log_amount"].to_numpy(float)])
        yv = df[nf].to_numpy(float)
        mask = np.isfinite(X).all(1) & np.isfinite(yv)
        coef, *_ = np.linalg.lstsq(X[mask], yv[mask], rcond=None)
        pred = X[mask] @ coef
        ss_res = float(((yv[mask] - pred) ** 2).sum())
        ss_tot = float(((yv[mask] - yv[mask].mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        r2_log_amount[nf] = r2
        print(f"  {nf} ≈ {coef[0]:.4f} + {coef[1]:.4f}*log_amount   →  R² = {r2:.6f}")

    # 全因子重建 turnover_daily（pooled，多因子）
    for nf in new_factors:
        cols_needed = [nf] + old_factors
        sub = df[cols_needed].astype(float)
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 1000:
            print(f"  [{nf}] 有效样本不足，跳过多因子 R²")
            continue
        X = np.column_stack([np.ones(len(sub))] + [sub[c].to_numpy() for c in old_factors])
        yv = sub[nf].to_numpy()
        coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        pred = X @ coef
        ss_res = float(((yv - pred) ** 2).sum())
        ss_tot = float(((yv - yv.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  [{nf}] 用**全部 {len(old_factors)} 个既有因子**线性重建 → R² = {r2:.6f}"
              f"（即既有因子已能解释 {(1-r2)*100:.1f}% 的方差）")

    # ---------- (4) 残差 IC：决策核心 ----------
    print("\n=== (4) 残差 IC（决策核心）：逐日横截面把新因子投影到既有因子后，取残差算 IC ===")
    results = {}
    for nf in new_factors:
        ic_raw = daily_ic(df, pred_col=nf, y_col=ycol)
        s_raw = ic_summary(ic_raw)
        print(f"  [{nf}] 原始 IC={s_raw['ic_mean']:+.4f} ICIR={s_raw['icir']:+.4f} "
              f"年化ICIR={s_raw['icir_annual']:+.2f} 正占比={s_raw['ic_positive_rate']:.3f} "
              f"t={s_raw['t_stat']:.2f} n_days={s_raw['n_days']}")
        res = _cs_residual(df, nf, old_factors)
        if res.empty:
            print(f"  [{nf}] 残差计算失败（样本不足）")
            continue
        res = res.rename(columns={"resid": nf + "_resid"})
        ic_res = daily_ic(res, pred_col=nf + "_resid", y_col=ycol)
        s_res = ic_summary(ic_res)
        print(f"  [{nf}] 残差 IC={s_res['ic_mean']:+.4f} ICIR={s_res['icir']:+.4f} "
              f"年化ICIR={s_res['icir_annual']:+.2f} 正占比={s_res['ic_positive_rate']:.3f} "
              f"t={s_res['t_stat']:.2f} n_days={s_res['n_days']}")
        results[nf] = {"raw": s_raw, "resid": s_res, "r2_log_amount": r2_log_amount.get(nf)}

    # ---------- 结论 ----------
    print("\n=== 结论判定 ===")
    for nf in new_factors:
        r = results.get(nf)
        if not r:
            continue
        raw_ic = r["raw"]["ic_mean"]
        res_ic = r["resid"]["ic_mean"]
        res_t = r["resid"]["t_stat"]
        r2 = r["r2_log_amount"]
        print(f"  [{nf}] 原始 IC={raw_ic:+.4f} → 残差 IC={res_ic:+.4f} (t={res_t:+.2f})"
              f" | 与 log_amount 的 R²={r2:.4f}")
        if abs(res_ic) <= 0.005 or abs(res_t) < 2.0:
            print(f"      ⇒ 残差 IC 近乎 0（|IC|<=0.005 或 |t|<2）：既有因子已覆盖其信息 ⇒ **不值得纳入**")
        else:
            print(f"      ⇒ 残差 IC 显著非 0：**有独立信息，值得纳入**（仍需重训验证 IC 增量）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
