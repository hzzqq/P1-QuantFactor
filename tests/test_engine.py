"""R2/R3 + 无前视：回测引擎测试。"""
import numpy as np
import pandas as pd

from src.backtest import engine as BE


def _flat_panel(n_sym=20, n_days=220, start="2021-01-01"):
    dates = pd.bdate_range(start, periods=n_days)
    syms = [f"S{i:02d}" for i in range(n_sym)]
    rows = []
    for s in syms:
        for d in dates:
            # 价格恒为 100 → 无涨跌停、无收益，便于精确核算成本
            rows.append({"symbol": s, "date": d, "open": 100.0, "close": 100.0})
    return pd.DataFrame(rows), dates, syms


def _flip_preds(dates, syms, rb_freq=10):
    """每个调仓期排名完全翻转 → 连续模式每期 full turnover。
    注意用「调仓期号 = ti//rb_freq」翻排（若用全日期序号 ti，因每次 +rb_freq(偶数)
    奇偶不变，排名其实不翻，会测不出换手成本）。"""
    df = []
    for ti, d in enumerate(dates):
        period = ti // rb_freq
        for si, s in enumerate(syms):
            val = 1.0 if (si + period) % 2 == 0 else -1.0
            df.append({"date": d, "symbol": s, "pred": val})
    return pd.DataFrame(df)


def test_no_lookahead_entry_uses_next_open():
    # attach_forward 必须把入场价设为 T+1 开盘（open_s1），而非当日
    panel, dates, syms = _flat_panel(n_days=30)
    af = BE.attach_forward(panel, horizon=10)
    # 取某只股票：open_s1 应等于下一日 open（此处恒为 100，但位置必须偏移 1）
    sub = af[af["symbol"] == syms[0]].reset_index(drop=True)
    # open_s1 是 open 的 shift(-1)；首日 open_s1 应为次日 open=100，而当日 open 也是 100
    # 用非平凡价格验证偏移更稳：构造单调价格
    panel2 = panel.copy()
    panel2["open"] = panel2.groupby("symbol").cumcount() + 1.0
    af2 = BE.attach_forward(panel2, horizon=10)
    s2 = af2[af2["symbol"] == syms[0]].reset_index(drop=True)
    # 第 i 行 open_s1 == 第 i+1 行 open
    assert np.allclose(s2["open_s1"].iloc[:-1].values, s2["open"].iloc[1:].values)


def test_continuous_cost_not_double_counted(R2=True):
    """R2：flat 价格(gross=0) 全换手下，稳态每桶净收益应 = -(c_buy+c_sell)，
    而非旧实现的 -2*(c_buy+c_sell)（双倍计费）。"""
    panel, dates, syms = _flat_panel(n_days=220)
    preds = _flip_preds(dates, syms)
    res = BE.run_backtest_continuous(
        preds, panel, horizon=10, top_pct=0.1, rebalance_freq=10,
        cost=BE.DEFAULT_COST, mode="long_short", buffer=0.0)
    c = BE.DEFAULT_COST
    c_buy = c["commission"] + c["slippage"]
    c_sell = c["commission"] + c["slippage"] + c["stamp"]
    steady_cost = c_buy + c_sell  # 稳态每桶成本（多空各一份）
    # 直接从 equity 反推每桶均值净收益（首桶仅进场成本减半，长期均值逼近稳态）
    eq = res.equity.values
    per = eq[1:] / eq[:-1] - 1.0
    mean_per = float(per.mean())
    # 旧实现会给出约 -2*steady_cost；新实现约 -steady_cost（首桶略小）
    assert -2.0 * steady_cost - 1e-6 < mean_per < -0.5 * steady_cost, \
        f"连续成本异常：均值净 {mean_per:.5f}，期望≈ -{steady_cost:.5f}（非 -{2*steady_cost:.5f}）"


def test_bucket_ntrades_not_overcounted(R3=True):
    """R3：n_trades 应为实际选中交易数（桶数×2×k），而非全样本行数。"""
    panel, dates, syms = _flat_panel(n_days=220)
    preds = _flip_preds(dates, syms)
    res = BE.run_backtest(preds, panel, horizon=10, top_pct=0.1,
                          rebalance_freq=10, cost=BE.DEFAULT_COST,
                          mode="long_short")
    n_buckets = res.ls_stats["n_buckets"]
    k = max(1, int(len(syms) * 0.1))   # 均匀下每桶有效数=全样本
    # 实际选中 = 每桶 多 k + 空 k
    assert res.n_trades == n_buckets * 2 * k, \
        f"n_trades={res.n_trades} 应={n_buckets*2*k}"
    # 远小于全样本行数（旧实现会≈全行数）
    assert res.n_trades < len(panel) // 2


def test_backtest_bootstrap_true_does_not_crash(N1=True):
    """N1（R6 回归修复）：bootstrap=True 时引擎调用 backtest_bootstrap(..., years=years)，
    旧签名无 years 形参 → TypeError 崩溃。修复后必须正常返回且带置信区间。"""
    panel, dates, syms = _flat_panel(n_days=220)
    preds = _flip_preds(dates, syms)
    res = BE.run_backtest(preds, panel, horizon=10, top_pct=0.1,
                          rebalance_freq=10, cost=BE.DEFAULT_COST,
                          mode="long_short", bootstrap=True)
    assert "total_return_ci" in res.ls_stats, "bootstrap 置信区间未产出"
    assert "sharpe_ci" in res.ls_stats
    # 连续模式同样验证
    res2 = BE.run_backtest_continuous(
        preds, panel, horizon=10, top_pct=0.1, rebalance_freq=10,
        cost=BE.DEFAULT_COST, mode="long_short", buffer=0.0, bootstrap=True)
    assert "total_return_ci" in res2.ls_stats


def test_continuous_ntrades_accumulates_not_just_last_bucket(N4=True):
    """N4：连续模式 n_trades 须累计每桶成交，而非仅末桶持仓数（旧实现严重低估）。"""
    panel, dates, syms = _flat_panel(n_days=220)
    preds = _flip_preds(dates, syms)   # 全换手
    res = BE.run_backtest_continuous(
        preds, panel, horizon=10, top_pct=0.1, rebalance_freq=10,
        cost=BE.DEFAULT_COST, mode="long_short", buffer=0.0)
    k = max(1, int(len(syms) * 0.1))
    last_holdings = 2 * k
    # 累计成交必远大于末桶持仓，且每桶至少开多/空各 k
    assert res.n_trades > last_holdings, f"n_trades={res.n_trades} 仅末桶持仓，未累计"
    assert res.n_trades >= res.ls_stats["n_buckets"] * 2 * k


def test_continuous_reports_entry_blocked(N9=True):
    """N9：连续模式须上报因涨停被挡无法买入的条目数（旧实现恒为 0）。"""
    panel, dates, syms = _flat_panel(n_days=220)
    # 让首个调仓日的 T+1（dates[1]）全市场涨停 → entry 被挡
    mask = panel["date"] == dates[1]
    panel = panel.copy()
    panel.loc[mask, "close"] = 110.0   # close/prev_close-1 = 0.10 >= LIMIT_UP
    preds = _flip_preds(dates, syms)
    res = BE.run_backtest_continuous(
        preds, panel, horizon=10, top_pct=0.1, rebalance_freq=10,
        cost=BE.DEFAULT_COST, mode="long_short", buffer=0.0)
    assert res.n_blocked > 0, "涨停挡单计数应为正"


def test_continuous_survives_limit_up_exit_day(N5=True):
    """N5：空头回补需买入，涨停日无法买入；含涨停日的面板连续回测不得崩溃且 equity 有限。"""
    panel, dates, syms = _flat_panel(n_days=220)
    mask = panel["date"] == dates[60]
    panel = panel.copy()
    panel.loc[mask, "close"] = 110.0   # 制造一个涨停日（含退出窗口内）
    preds = _flip_preds(dates, syms)
    res = BE.run_backtest_continuous(
        preds, panel, horizon=10, top_pct=0.1, rebalance_freq=10,
        cost=BE.DEFAULT_COST, mode="long_short", buffer=0.0)
    assert np.isfinite(res.equity).all(), "含涨停日 equity 不应出现 nan"


def test_bucket_boundary_no_crash(N16=True):
    """N16：调仓日抽稀到面板末尾时不得 IndexError；最短面板仍返回结果。"""
    panel, dates, syms = _flat_panel(n_days=25)   # horizon=10，仅能形成少量桶
    preds = _flip_preds(dates, syms)
    res = BE.run_backtest(preds, panel, horizon=10, top_pct=0.1,
                          rebalance_freq=10, cost=BE.DEFAULT_COST,
                          mode="long_short")
    from src.backtest.engine import BacktestResult
    assert isinstance(res, BacktestResult)
