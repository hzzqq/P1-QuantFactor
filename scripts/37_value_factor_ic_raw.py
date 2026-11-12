"""阶段 50 - alpha 再生：估值因子(EP/BTOP) rank-IC（新干净源，从未用过）。

背景
----
②b 治本已证：真实换手率信息已被现有 log_amount 代理覆盖 → 不重训。
但 raw baostock_daily 还躺着 **peTTM / pbMRQ** 两个从没进过因子体系的估值字段。
估值因子（EP=1/PE、BTOP=1/PB）是经典价值因子，与现有 38 个量价时序因子正交，
是 alpha 再生最干净的候选源。本脚本回答：估值因子有没有独立可预测信号？值不值得并入重训？

口径（对齐 36 / ②b 方法论）
----
- 前向收益 fwd10（h=10）+ 截面等权市场中性化（对齐 P1 去 beta 意图）。
- 价值因子：ep = 1/peTTM（仅 peTTM>0）；btop = 1/pbMRQ（仅 pbMRQ>0）。
  方向约定：ep/btop 越大=越便宜；若价值效应成立，预期 IC>0（便宜跑赢）。
- 两层结论：
    Tier1（RAW 直算）：ep/btop 对 fwd10_neu 的逐日截面 Spearman rank-IC（均值/std/ICIR/t），
           窗口 FULL/2024+/2026/recent59td；并做对 log_amount 的残差 IC（增量 vs 规模代理）。
    Tier2（决策级，v39 数据集）：把 ep/btop merge 进 dataset_h10_v39，对**全部 39 个现有因子**
           做逐日截面 OLS 取残差，算残差 IC vs 模型自身 y_excess（=现有模型未覆盖的增量）。
- 护栏：各价值因子 top-1 占比 >20% 即拒绝（防坏数据假结论，沿用 33b/36 门禁）。

ST/停牌过滤沿用 36 修复（tradestatus/isST 是 large_string，须 pd.to_numeric）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.eval.metrics import daily_ic, ic_summary  # noqa: E402

RAW = ROOT / "data" / "P1" / "raw" / "baostock_daily" / "baostock_daily.parquet"
V39 = ROOT / "data" / "P1" / "processed" / "dataset_h10_v39.parquet"
HORIZON = 10
RECENT_TD = 59


def _read_clean() -> pd.DataFrame:
    t0 = time.time()
    cols = ["symbol", "date", "close", "peTTM", "pbMRQ", "tradestatus", "isST"]
    df = pd.read_parquet(RAW, columns=cols)
    df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce")
    df["isST"] = pd.to_numeric(df["isST"], errors="coerce")
    n0 = len(df)
    df = df[df["isST"].fillna(0) != 1]
    df = df[df["tradestatus"].fillna(1) == 1]
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    # 估值因子（仅正 PE/PB）
    pe = df["peTTM"].astype(float)
    pb = df["pbMRQ"].astype(float)
    df["ep"] = np.where(pe > 0, 1.0 / pe, np.nan)
    df["btop"] = np.where(pb > 0, 1.0 / pb, np.nan)
    df["fwd10"] = df.groupby("symbol", sort=False)["close"].shift(-HORIZON) / df["close"] - 1.0
    df["fwd10_neu"] = df["fwd10"] - df.groupby("date", sort=False)["fwd10"].transform("mean")
    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"[37] 读+估值因子构建 {len(df):,} 行（剔 ST/停牌 {n0-len(df):,}）"
          f" {time.time()-t0:.1f}s", flush=True)
    return df


def _sanity_gate(df: pd.DataFrame) -> bool:
    ok = True
    for c in ["ep", "btop"]:
        s = df[c].dropna()
        vc = s.value_counts()
        top1 = float(vc.iloc[0]) / len(s) if len(s) else 1.0
        print(f"[37] 门禁 {c}: top-1 占比 {top1*100:.4f}%, unique={s.nunique():,}, "
              f"null={df[c].isna().mean()*100:.2f}%, median={s.median():.4g}")
        if top1 > 0.20:
            print(f"[37] ❌ {c} 疑似恒值填充，拒绝")
            ok = False
    return ok


def _window_mask(dates: pd.Series, kind: str) -> pd.Series:
    if kind == "FULL":
        return pd.Series(True, index=dates.index)
    if kind == "2024+":
        return dates >= pd.Timestamp("2024-01-01")
    if kind == "2026":
        return dates >= pd.Timestamp("2026-01-01")
    if kind == "recent59td":
        uniq = np.sort(pd.unique(dates))
        cutoff = uniq[-RECENT_TD] if len(uniq) > RECENT_TD else uniq[0]
        return dates >= cutoff
    raise ValueError(kind)


def _report_ic(df: pd.DataFrame, feats: list[str]) -> None:
    windows = ["FULL", "2024+", "2026", "recent59td"]
    for f in feats:
        print(f"\n=== 特征 {f} (Tier1 RAW rank-IC vs fwd10_neu) ===")
        sub = pd.DataFrame({"date": df["date"].to_numpy(),
                            "pred": df[f].to_numpy(),
                            "fwd10_neu": df["fwd10_neu"].to_numpy()})
        for w in windows:
            m = _window_mask(df["date"], w)
            ic = daily_ic(sub[m], pred_col="pred", y_col="fwd10_neu")
            s = ic_summary(ic)
            print(f"  [{w:10s}] IC={s['ic_mean']:+.4f}  std={s['ic_std']:.4f}  "
                  f"ICIR={s['icir']:+.3f}  正占比={s['ic_positive_rate']:.3f}  "
                  f"t={s['t_stat']:+.2f}  n_days={s['n_days']}")


def _residual_ic(df: pd.DataFrame, feats: list[str], proxy: str = "log_amount") -> None:
    print(f"\n=== Tier1 残差 IC（对 {proxy} 残差 vs fwd10_neu）===")
    windows = ["FULL", "2024+", "2026", "recent59td"]
    # log_amount 需从 baostock amount 派生
    amt = pd.read_parquet(RAW, columns=["symbol", "date", "amount"])
    amt["log_amount"] = np.where(amt["amount"].astype(float) > 0,
                                 np.log(amt["amount"].astype(float)), np.nan)
    df2 = df.merge(amt[["symbol", "date", "log_amount"]], on=["symbol", "date"], how="left")
    for f in feats:
        print(f"\n--- {f} 残差 IC ---")
        out = []
        for d, g in df2.groupby("date", sort=False):
            gg = g[[f, "log_amount", "fwd10_neu"]].dropna()
            if len(gg) < 10:
                continue
            X = np.column_stack([np.ones(len(gg)), gg["log_amount"].to_numpy(float)])
            yv = gg[f].to_numpy(float)
            try:
                coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            except np.linalg.LinAlgError:
                continue
            res = yv - X @ coef
            out.append(pd.DataFrame({"date": d, "resid": res,
                                     "y": gg["fwd10_neu"].to_numpy(float)}))
        res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "resid", "y"])
        for w in windows:
            m = _window_mask(res["date"], w)
            ic = daily_ic(res[m].rename(columns={"resid": "pred", "y": "fwd10_neu"}),
                          pred_col="pred", y_col="fwd10_neu")
            s = ic_summary(ic)
            sig = "显著" if (abs(s["ic_mean"]) > 0.005 and abs(s["t_stat"]) >= 2.0) else "≈0(冗余)"
            print(f"  [{w:10s}] 残差IC={s['ic_mean']:+.4f}  ICIR={s['icir']:+.3f}  "
                  f"t={s['t_stat']:+.2f}  n_days={s['n_days']}  → {sig}")


def _tier2_vs_all(df: pd.DataFrame) -> None:
    """决策级：把价值因子 merge 进 v39，对全部现有因子做残差 IC（模型未覆盖的增量）。"""
    if not V39.exists():
        print("[37] 跳过 Tier2：缺少 dataset_h10_v39.parquet")
        return
    print("\n=== Tier2 残差 IC（对 v39 全部现有因子残差 vs 模型 y_excess）===")
    t0 = time.time()
    ds = pd.read_parquet(V39)
    meta_p = V39.with_name("dataset_h10_v39_meta.json")
    meta = __import__("json").load(open(meta_p, encoding="utf-8"))
    factors = list(meta["factor_names"])
    ycol = "y_excess"
    need = ["date", "symbol", ycol] + factors
    ds = ds[[c for c in need if c in ds.columns]].copy()
    val = df[["symbol", "date", "ep", "btop"]]
    ds = ds.merge(val, on=["symbol", "date"], how="left")
    print(f"[37] v39 载入 {len(ds):,} 行 | 现有因子 {len(factors)} | 标签 {ycol} "
          f"| 估值覆盖 ep={ds['ep'].notna().mean()*100:.1f}% btop={ds['btop'].notna().mean()*100:.1f}% "
          f"{time.time()-t0:.1f}s", flush=True)

    windows = ["FULL", "2024+", "2026", "recent59td"]
    ds_dates = pd.to_datetime(ds["date"])
    for f in ["ep", "btop"]:
        print(f"\n--- {f}（先算原始 IC，再算对全因子残差 IC）---")
        for w in windows:
            m = _window_mask(ds_dates, w)
            sub = ds[m]
            ic_raw = daily_ic(sub, pred_col=f, y_col=ycol)
            s_raw = ic_summary(ic_raw)
            # 残差：逐日截面 OLS f ~ 1 + 全部现有因子
            out = []
            for d, g in sub.groupby("date", sort=False):
                gg = g[[f] + factors + [ycol]].dropna()
                if len(gg) < len(factors) + 10:
                    continue
                X = np.column_stack([np.ones(len(gg))] +
                                   [gg[c].to_numpy(float) for c in factors])
                yv = gg[f].to_numpy(float)
                try:
                    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                res = yv - X @ coef
                out.append(pd.DataFrame({"date": d, "resid": res,
                                         "y": gg[ycol].to_numpy(float)}))
            if not out:
                print(f"  [{w:10s}] 残差计算失败（样本不足）")
                continue
            res = pd.concat(out, ignore_index=True)
            ic_r = daily_ic(res.rename(columns={"resid": "pred", "y": ycol}),
                            pred_col="pred", y_col=ycol)
            s_r = ic_summary(ic_r)
            sig = "显著增量" if (abs(s_r["ic_mean"]) > 0.005 and abs(s_r["t_stat"]) >= 2.0) else "≈0(被覆盖)"
            print(f"  [{w:10s}] 原始IC={s_raw['ic_mean']:+.4f}(t{s_raw['t_stat']:+.1f})  "
                  f"→ 残差IC={s_r['ic_mean']:+.4f}(t{s_r['t_stat']:+.2f}, n={s_r['n_days']})  "
                  f"→ {sig}")


def main() -> int:
    print("[37] alpha 再生：估值因子(EP/BTOP) rank-IC", flush=True)
    df = _read_clean()
    if not _sanity_gate(df):
        return 2
    _report_ic(df, ["ep", "btop"])
    _residual_ic(df, ["ep", "btop"])
    _tier2_vs_all(df)
    print("\n[37] 完成。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
