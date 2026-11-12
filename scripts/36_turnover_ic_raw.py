"""阶段 49 - ②b 治本：从 RAW baostock_daily（干净 turn）直接计算换手率因子 rank-IC。

背景
----
v40 的 turnover_daily 来自 baostock 但被恒值填充污染（top-1 占比 57.65%，坏数据，
33b 已门禁拒绝出结论）。本次从**原始未污染** baostock_daily.parquet 直接取 turn，
绕开 v40 坏数据，回答决策级问题：
  「真实逐日换手率，对 10 日市场中性化前向收益的 rank-IC 是否显著？是否值得纳入重训？」

口径（对齐 P1 既有方法论）
----
- 前向收益 fwd10 = close.shift(-10)/close - 1（per symbol，T 买 T+10 卖），horizon=10 对齐 P1。
- 市场中性化（对齐 P1 y_excess 去除 beta 的意图）：fwd10_neu = fwd10 - 截面等权均值(date)。
  这是标准横截面因子 IC 做法，去除大盘共同成分；比固定基准指数更贴合「横截面选股」语义。
- 换手率特征（三个角度）：
    turn_ln     = log1p(turn)                       —— 水平因子（流动性/规模维度）
    turn_z60    = (turn - 60日均)/60日std（per symbol 时序 z）  —— 异动/活跃度异常
    turn_surge20= turn / 20日均 - 1                  —— 短期放量偏离
- 对照代理：log_amount = log(amount)（baostock 成交额；与阶段27 log_amount 同义的
  规模/流动性代理，同源更公平）。
- rank-IC：每日截面 Spearman(特征, fwd10_neu)，再取 均值 / std / ICIR(未年化) / t。
- 窗口：FULL / 2024+ / 2026 / recent59td（最近 59 个交易日）。
- 决策核心：把换手率特征对 log_amount 做逐日截面 OLS 取残差，算**残差 rank-IC**——
  残差 IC≈0 ⇒ 信息已被 log_amount 覆盖 ⇒ 不值得纳入；显著 ⇒ 有独立增量信息。

护栏（沿用 33b 数据健全性门禁）：turn top-1 占比 >20% 即拒绝（防坏数据假结论）。
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

SRC = ROOT / "data" / "P1" / "raw" / "baostock_daily" / "baostock_daily.parquet"

HORIZON = 10          # 对齐 P1 h=10
Z_WIN = 60
SURGE_WIN = 20
RECENT_TD = 59


def _read_clean() -> pd.DataFrame:
    t0 = time.time()
    cols = ["symbol", "date", "close", "turn", "amount", "tradestatus", "isST"]
    df = pd.read_parquet(SRC, columns=cols)
    print(f"[36] 读 raw baostock_daily {len(df):,} 行 "
          f"{df['date'].min().date()}~{df['date'].max().date()} "
          f"{df['symbol'].nunique()} 只 {time.time()-t0:.1f}s", flush=True)
    # —— 修复 2026-11 卡住 bug：tradestatus/isST 是 large_string('0'/'1')，
    #    不能用整数 1 比较，否则字符串 '1' != 1 恒 True → 全过滤 → 0 行。
    df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce")
    df["isST"] = pd.to_numeric(df["isST"], errors="coerce")
    n0 = len(df)
    df = df[df["isST"].fillna(0) != 1]              # 剔除 ST
    df = df[df["tradestatus"].fillna(1) == 1]       # 仅保留交易日（剔除停牌）
    print(f"[36] 过滤 ST/停牌：{n0:,} → {len(df):,} 行 "
          f"（剔除 {n0-len(df):,}，{ (n0-len(df))/n0:.3%}）", flush=True)
    df["symbol"] = df["symbol"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return df


def _sanity_gate(df: pd.DataFrame) -> bool:
    s = df["turn"].dropna()
    vc = s.value_counts()
    top1 = float(vc.iloc[0]) / len(s)
    nun = int(s.nunique())
    print(f"[36] 数据健全性门禁  turn: top-1 占比 {top1*100:.4f}%, unique={nun:,}, "
          f"std={float(s.std()):.6g}, null={df['turn'].isna().mean()*100:.3f}%", flush=True)
    if top1 > 0.20:
        print("[36] ❌ turn 存在恒值填充（top-1>20%），拒绝出结论")
        return False
    return True


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.time()
    g = df.groupby("symbol", sort=False)
    # 前向收益（T 买 T+H 卖）
    df["fwd10"] = g["close"].shift(-HORIZON) / df["close"] - 1.0
    # 市场中性化：减截面等权均值
    df["fwd10_neu"] = df["fwd10"] - df.groupby("date", sort=False)["fwd10"].transform("mean")
    # 换手率特征
    df["turn_ln"] = np.log1p(df["turn"].clip(lower=0))
    mean60 = g["turn"].transform(lambda s: s.rolling(Z_WIN, min_periods=Z_WIN).mean())
    std60 = g["turn"].transform(lambda s: s.rolling(Z_WIN, min_periods=Z_WIN).std())
    df["turn_z60"] = (df["turn"] - mean60) / std60
    mean20 = g["turn"].transform(lambda s: s.rolling(SURGE_WIN, min_periods=SURGE_WIN).mean())
    df["turn_surge20"] = df["turn"] / mean20 - 1.0
    # 对照代理：log_amount（同源成交额）
    df["log_amount"] = np.where(df["amount"] > 0, np.log(df["amount"]), np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"[36] 特征构建完成 {time.time()-t0:.1f}s", flush=True)
    return df


def _window_mask(dates: pd.Series, kind: str) -> pd.Series:
    if kind == "FULL":
        return pd.Series(True, index=dates.index)
    if kind == "2024+":
        return dates >= pd.Timestamp("2024-01-01")
    if kind == "2026":
        return dates >= pd.Timestamp("2026-01-01")
    if kind == "recent59td":
        uniq = np.sort(pd.unique(dates))
        if len(uniq) <= RECENT_TD:
            cutoff = uniq[0]
        else:
            cutoff = uniq[-RECENT_TD]
        return dates >= cutoff
    raise ValueError(kind)


def _report_ic(df: pd.DataFrame, feats: list[str]) -> None:
    windows = ["FULL", "2024+", "2026", "recent59td"]
    for f in feats:
        print(f"\n=== 特征 {f} ===")
        sub = pd.DataFrame({
            "date": df["date"].to_numpy(),
            "pred": df[f].to_numpy(),
            "fwd10_neu": df["fwd10_neu"].to_numpy(),
        })
        for w in windows:
            m = _window_mask(df["date"], w)
            ic = daily_ic(sub[m], pred_col="pred", y_col="fwd10_neu")
            s = ic_summary(ic)
            print(f"  [{w:10s}] IC={s['ic_mean']:+.4f}  std={s['ic_std']:.4f}  "
                  f"ICIR={s['icir']:+.3f}  正占比={s['ic_positive_rate']:.3f}  "
                  f"t={s['t_stat']:+.2f}  n_days={s['n_days']}")


def _residual_ic(df: pd.DataFrame, feats: list[str], proxy: str = "log_amount") -> None:
    """决策核心：把换手率特征对 log_amount 做逐日截面 OLS，取残差算 rank-IC。"""
    print(f"\n=== 残差 IC（增量信息检验）：特征 ~ 截面OLS({proxy}) 残差 vs fwd10_neu ===")
    windows = ["FULL", "2024+", "2026", "recent59td"]
    for f in feats:
        print(f"\n--- {f} 残差 IC ---")
        out = []
        for d, g in df.groupby("date", sort=False):
            gg = g[[f, proxy, "fwd10_neu"]].dropna()
            if len(gg) < 10:
                continue
            X = np.column_stack([np.ones(len(gg)), gg[proxy].to_numpy(float)])
            yv = gg[f].to_numpy(float)
            try:
                coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid = yv - X @ coef
            sub = pd.DataFrame({"date": d, "resid": resid, "y": gg["fwd10_neu"].to_numpy(float)})
            out.append(sub)
        res = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "resid", "y"])
        for w in windows:
            m = _window_mask(res["date"], w)
            ic = daily_ic(res[m].rename(columns={"resid": "pred", "y": "fwd10_neu"}),
                          pred_col="pred", y_col="fwd10_neu")
            s = ic_summary(ic)
            sig = "显著" if (abs(s["ic_mean"]) > 0.005 and abs(s["t_stat"]) >= 2.0) else "≈0(冗余)"
            print(f"  [{w:10s}] 残差IC={s['ic_mean']:+.4f}  ICIR={s['icir']:+.3f}  "
                  f"t={s['t_stat']:+.2f}  n_days={s['n_days']}  → {sig}")


def main() -> int:
    print("[36] ②b 治本：RAW 换手率因子 rank-IC", flush=True)
    df = _read_clean()
    if not _sanity_gate(df):
        return 2
    df = _build_features(df)
    feats = ["turn_ln", "turn_z60", "turn_surge20", "log_amount"]
    _report_ic(df, feats)
    _residual_ic(df, ["turn_ln", "turn_z60", "turn_surge20"], proxy="log_amount")
    print("\n[36] 完成。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
