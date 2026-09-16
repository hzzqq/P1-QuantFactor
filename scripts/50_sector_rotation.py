"""阶段 50：免费行业/ETF 轮动模块（腾讯源，零成本）。

复用 src/data/sources 的腾讯 fetcher 拉一篮子宽基指数 + 行业 ETF 的日线，
做「动量轮动」长仓策略：
  - 每只标的算 trailing `mom_days` 日收益率（动量）；
  - 每 `rebal` 个交易日调仓，做多截面动量前 `top_k` 名，等权；
  - 长仓 ETF 成本 = 佣金 0.03% + 滑点 0.1%（ETF 无印花税）；
  - 不做空、不施加涨跌停约束（指数/ETF 流动性充足，涨跌停罕见）。

产出：
  - data/P1/processed/rotation/panel_rotation.parquet  (缓存，7天刷新)
  - data/P1/processed/report_rotation.json            (回测报告)
  - data/P1/processed/signals/signal_rotation.json    (最新一期做多名单)

注：本模块完全免费，是 P1 多因子日频框架之外的「宏观/行业择时」延伸玩法。
"""
from __future__ import annotations
import sys, time, json, pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(r"E:/project/sj"); PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data.sources import fetch_kline, fetch_index
from src.backtest import _apply_vol_target
from shared.logging_utils import get_logger
logger = get_logger("P1.rotation")

PROC = ROOT / "data" / "P1" / "processed"
CACHE = PROC / "rotation"; CACHE.mkdir(parents=True, exist_ok=True)
SIGNAL_DIR = PROC / "signals"; SIGNAL_DIR.mkdir(parents=True, exist_ok=True)

# 一篮子标的：宽基指数 + 行业 ETF（覆盖风格/行业，免费腾讯源可得）
BASKET = [
    ("sh000300", "沪深300"), ("sh000905", "中证500"), ("sh000016", "上证50"),
    ("sz399006", "创业板指"), ("sh000688", "科创50"),
    ("510050", "上证50ETF"), ("510300", "沪深300ETF"), ("510500", "中证500ETF"),
    ("159915", "创业板ETF"), ("512010", "医药ETF"), ("512660", "军工ETF"),
    ("512880", "证券ETF"), ("159928", "消费ETF"), ("512760", "芯片ETF"),
    ("515030", "新能源ETF"), ("512800", "银行ETF"),
]

START, END = "2018-01-01", "2026-09-01"
MOM_LOOK = 252         # 动量长窗（≈12 个月）
MOM_SKIP = 21          # 跳过近 1 个月（12-1 动量，避开短期反转）
MA_WIN = 200           # 趋势过滤均线
REBAL = 21             # 月度调仓（交易日）
TOP_K = 5              # 做多前 K 名（合格且最强国）
COMM = 0.0003          # 佣金
SLIP = 0.001           # 滑点（ETF 无印花税）


def _is_index(code8: str) -> bool:
    """6 位代码段以 000/399/930 开头视为指数（无复权），其余为 ETF（前复权）。"""
    six = code8[2:] if code8.startswith(("sh", "sz", "bj")) else code8
    return six.startswith(("000", "399", "930"))


def load_panel() -> pd.DataFrame:
    """拉取或读缓存的轮动面板。"""
    import os
    cf = CACHE / "panel_rotation.parquet"
    if cf.exists() and (time.time() - os.path.getmtime(cf)) < 7 * 86400:
        logger.info("读缓存 %s", cf)
        return pd.read_parquet(cf)
    frames = []
    for code, name in BASKET:
        try:
            if _is_index(code):
                df = fetch_index(code, START, END)
            else:
                df = fetch_kline(code, START, END, adjust="qfq")
            if df.empty:
                logger.warning("空: %s %s", code, name); continue
            df = df[["symbol", "date", "open", "close"]].copy()
            df["name"] = name
            frames.append(df)
            logger.info("拉到 %s %s  %s 行", code, name, len(df))
        except Exception as e:
            logger.warning("失败 %s %s: %s", code, name, str(e)[:80])
    if not frames:
        raise RuntimeError("全部标的拉取失败（网络或端点被限）")
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    panel.to_parquet(cf, index=False)
    logger.info("面板落盘 %s %s 行", cf, len(panel))
    return panel


def backtest_rotation(panel: pd.DataFrame, vol_target: float | None = None,
                      vol_lookback: int = 20) -> dict:
    """动量轮动长仓回测（12-1 动量 + 200日趋势过滤 + 无合格标的空仓）。

    经典稳健配方：
      - 动量 = P_{t-21} / P_{t-252} - 1（12 个月收益减近 1 个月，避开短期反转）；
      - 合格 = 动量有效 且 收盘价 > 200 日均线（只做上升趋势）；
      - 每月调仓，做多合格者中动量前 TOP_K（等权）；
      - 若当月无合格标的 → 空仓（当日收益 0，不计成本）；
      - 成本仅对变动部分计费（ETF 无印花税）。
    """
    panel = panel.sort_values(["symbol", "date"]).copy()
    g = panel.groupby("symbol")["close"]
    # 12-1 动量：P_{t-21}/P_{t-252} - 1
    panel["mom"] = g.transform(lambda s: s.shift(MOM_SKIP) / s.shift(MOM_LOOK) - 1.0)
    panel["ma"] = g.transform(lambda s: s.rolling(MA_WIN).mean())
    panel["trend_up"] = panel["close"] > panel["ma"]
    dates = sorted(panel["date"].unique())
    px = panel.pivot(index="date", columns="symbol", values="close")
    mom = panel.pivot(index="date", columns="symbol", values="mom")
    trend = panel.pivot(index="date", columns="symbol", values="trend_up")

    eq = 1.0
    eq_curve = []
    held = set()
    last_rebal = -10**9
    all_rets = []
    n_cash = 0
    for i, d in enumerate(dates):
        valid_mom = mom.loc[d].dropna()
        eligible = valid_mom[trend.loc[d].reindex(valid_mom.index).fillna(False).astype(bool)]
        cost = 0.0
        if i - last_rebal >= REBAL:
            if len(eligible) > 0:
                top = set(eligible.nlargest(TOP_K).index)
                sell_set = held - top
                buy_set = top - held
                cost = (len(sell_set) + len(buy_set)) * (COMM + SLIP) / max(len(top), 1)
                held = top
            else:
                held = set()  # 空仓
            last_rebal = i
        if held:
            syms = list(held)
            p0 = px.shift(1).loc[d, syms]
            p1 = px.loc[d, syms]
            ok = p0.notna() & p1.notna() & (p0 > 0)
            if ok.any():
                day_ret = float((p1[ok] / p0[ok] - 1.0).mean()) - cost
                eq *= (1.0 + day_ret)
                all_rets.append(day_ret)
        else:
            n_cash += 1
            all_rets.append(0.0)
        eq_curve.append(eq)

    # B 任务技术迁移：组合层波动率目标化（阶段49 验证于 h20；此处只去杠杆）
    if vol_target is not None:
        all_rets = _apply_vol_target(list(all_rets), 1, vol_target, vol_lookback, 1.0)
        # 重新展开 eq_curve（按缩放后的日收益重算权益）
        eq = 1.0
        eq_curve = []
        for r in all_rets:
            eq *= (1.0 + r)
            eq_curve.append(eq)

    a = np.array(all_rets, dtype=float)
    n = len(a)
    years = n / 252.0
    total = float(eq - 1)
    ann = float((1 + total) ** (1 / max(years, 1e-9)) - 1) if years > 0 else np.nan
    sharpe = float(a.mean() / a.std() * np.sqrt(252)) if a.std() > 0 else np.nan
    peak = np.maximum.accumulate(eq_curve)
    mdd = float((np.array(eq_curve) / peak - 1).min())
    return {
        "strategy": "momentum_rotation_12_1_trendfilter",
        "basket_size": len(BASKET),
        "mom_look": MOM_LOOK, "mom_skip": MOM_SKIP, "ma_window": MA_WIN,
        "rebal_days": REBAL, "top_k": TOP_K,
        "start": str(dates[0].date()), "end": str(dates[-1].date()),
        "total_return": total, "annual_return": ann,
        "sharpe": sharpe,         "max_drawdown": mdd,
        "n_days": n, "n_cash_days": n_cash,
        "cash_ratio": n_cash / n if n else 0.0,
        "vol_target": vol_target,
    }


def latest_signal(panel: pd.DataFrame) -> dict:
    """产出最新一期做多名单（12-1 动量 + 趋势过滤，与回测一致）。"""
    panel = panel.sort_values(["symbol", "date"]).copy()
    g = panel.groupby("symbol")["close"]
    panel["mom"] = g.transform(lambda s: s.shift(MOM_SKIP) / s.shift(MOM_LOOK) - 1.0)
    panel["ma"] = g.transform(lambda s: s.rolling(MA_WIN).mean())
    panel["trend_up"] = panel["close"] > panel["ma"]
    last_date = panel["date"].max()
    row = panel[panel["date"] == last_date].copy()
    row = row.dropna(subset=["mom"])
    row = row[row["trend_up"] == True]  # 仅上升趋势
    row = row.sort_values("mom", ascending=False)
    top = row.head(TOP_K)
    return {
        "as_of": str(last_date.date()),
        "method": "12_1_momentum + 200d_trend_filter",
        "top_long": [
            {"symbol": r.symbol, "name": r.name,
             "mom_ret": round(float(r.mom), 4), "close": round(float(r.close), 3)}
            for r in top.itertuples()
        ],
        "all_ranked": [
            {"symbol": r.symbol, "name": r.name, "mom_ret": round(float(r.mom), 4)}
            for r in row.itertuples()
        ],
    }


def main() -> int:
    t0 = time.time()
    logger.info("=== 阶段50 行业/ETF 轮动 ===")
    panel = load_panel()
    logger.info("面板 %s 行 | %s 标的", f"{len(panel):,}", panel['symbol'].nunique())

    # 基线 + vol_target 扫描（B 技术迁移：只去杠杆压回撤）
    rows = []
    rows.append(backtest_rotation(panel, vol_target=None))
    for vt in (0.08, 0.10, 0.12):
        rows.append(backtest_rotation(panel, vol_target=vt))
    stats = rows[0]   # 基线用于信号导出
    sig = latest_signal(panel)

    rep = CACHE.parent / "report_rotation.json"
    rep.write_text(json.dumps({"scans": rows, "signal": sig},
                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    sig_path = SIGNAL_DIR / "signal_rotation.json"
    sig_path.write_text(json.dumps(sig, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 轮动回测（vol_target 扫描）===")
    print(f"{'volt':>5} {'Sharpe':>7} {'cumRet':>9} {'MDD':>8} {'cash%':>6}")
    for r in rows:
        vt = "-" if r["vol_target"] is None else f"{r['vol_target']:.2f}"
        print(f"{vt:>5} {r['sharpe']:7.3f} {r['total_return']:9.2%} "
              f"{r['max_drawdown']:8.2%} {r['cash_ratio']:6.1%}")
    print(f"\n=== 最新做多名单 (as_of {sig['as_of']}, 12-1动量+趋势过滤) ===")
    for r in sig["top_long"]:
        print(f"  {r['symbol']:10} {r['name']:8} 动量 {r['mom_ret']:+.2%}")
    print(f"\n报告: {rep}\n信号: {sig_path}\n耗时 {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
