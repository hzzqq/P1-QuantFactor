"""R4/R5：评估指标（metrics）测试。

覆盖：
- summarize 同时返回 ic / ic_mean（R4 别名修复，避免下游 .get("ic") 取 None）
- daily_ic 对单调预测给出显著 > 0 的 IC
- ic_summary 基本统计正确（均值/ICIR/正率）
"""
import numpy as np
import pandas as pd

from src.eval import metrics


def _mono_panel(n_days: int = 60, n_sym: int = 30, seed: int = 0) -> pd.DataFrame:
    """构造预测与未来超额收益强单调相关的面板（应有显著正 IC）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        preds = rng.normal(size=n_sym)
        # 未来收益 = 预测 + 噪声 → 强正相关
        y = preds + rng.normal(scale=0.3, size=n_sym)
        for s in range(n_sym):
            rows.append({"date": f"2024-01-{d+1:02d}", "symbol": f"S{s}",
                         "pred": preds[s], "y_excess": y[s]})
    return pd.DataFrame(rows)


def test_summarize_returns_ic_alias():
    df = _mono_panel()
    out = metrics.summarize(df, "pred", "y_excess")
    # R4：ic 别名必须存在且等于 ic_mean
    assert "ic" in out, "summarize 缺少 ic 别名键"
    assert "ic_mean" in out
    assert out["ic"] == out["ic_mean"], "ic 别名应等于 ic_mean"
    assert out["ic_mean"] > 0.3, "单调预测应给出强正 IC"


def test_daily_ic_positive_for_monotonic():
    df = _mono_panel()
    ic = metrics.daily_ic(df, "pred", "y_excess")
    assert len(ic) > 0
    assert ic.dropna().mean() > 0.3, "单调面板日均 IC 应显著为正"


def test_ic_summary_stats():
    s = pd.Series(np.linspace(0.01, 0.09, 50))  # 全正、均值 0.05、std≈0.0236
    out = metrics.ic_summary(s)
    assert abs(out["ic_mean"] - 0.05) < 1e-9
    assert out["ic_positive_rate"] == 1.0
    # 线性递增序列 std>0 → ICIR=均值/std≈2.10
    assert out["icir"] > 2.0
    assert out["t_stat"] > 0  # 单样本 t 检验显著


def test_ic_summary_varying_series_finite_icir():
    s = pd.Series(np.linspace(0.01, 0.09, 50))
    out = metrics.ic_summary(s)
    assert np.isfinite(out["icir"])  # 变序列 std>0 → ICIR 有限
    assert np.isfinite(out["t_stat"])
    assert out["icir"] > 2.0


def test_ic_summary_short_series_is_nan():
    s = pd.Series([0.1])  # 单点，不足以统计
    out = metrics.ic_summary(s)
    assert out["ic_mean"] != out["ic_mean"]  # np.nan


def test_bootstrap_ci_covers_point_estimate():
    rng = np.random.default_rng(1)
    # 稳定正收益序列：点估计 +0.01/桶，bootstrap 区间应横跨 0.01 且 lo<hi
    rets = rng.normal(0.01, 0.03, size=200)
    lo, hi = metrics.bootstrap_ci(rets, lambda x: float(x.mean()), n_boot=200, seed=2)
    assert lo < hi
    # 点估计应落在 (lo, hi) 内
    assert lo < 0.01 < hi


def test_backtest_bootstrap_returns_cis():
    rng = np.random.default_rng(3)
    rets = rng.normal(0.012, 0.04, size=150).tolist()
    out = metrics.backtest_bootstrap(rets, rebalance_freq=10, n_boot=200, seed=4)
    assert "total_return_ci" in out and "sharpe_ci" in out
    tlo, thi = out["total_return_ci"]
    slo, shi = out["sharpe_ci"]
    assert tlo < thi and slo < shi
    # 累计收益点估计应落在 bootstrap 区间内
    pt_total = (1 + np.array(rets)).prod() - 1
    assert tlo < pt_total < thi


def test_backtest_bootstrap_small_sample_empty():
    out = metrics.backtest_bootstrap([0.01, 0.02], n_boot=10)
    assert out == {}


def test_daily_ic_guard_zero_variance_returns_nan_not_inf():
    """N2：某日预测为常数 → 方差 0，旧实现 cov/sqrt(0)=inf/nan 污染 IC 序列。
    修复后该日 IC 必须为 nan（下游 dropna），绝不能出现 inf。"""
    df = pd.DataFrame({
        "date": ["2024-01-01"] * 12,
        "pred": [0.5] * 12,                     # 常数预测
        "y_excess": np.random.default_rng(0).normal(size=12),
    })
    ic = metrics.daily_ic(df, "pred", "y_excess")
    assert len(ic) == 1
    assert not np.isfinite(ic.iloc[0]), "常数预测日 IC 应为 nan"
    assert not np.isinf(ic.iloc[0]), "常数预测日 IC 不得为 inf"


def test_summarize_spread_sharpe_uses_horizon():
    """N3：spread_sharpe 年化必须用传入 horizon，而非硬编码 /10。
    horizon=5 与 horizon=10 的 sharpe 应差 sqrt(2) 倍。"""
    df = _mono_panel(n_days=120, n_sym=40, seed=7)
    ss5 = metrics.summarize(df, "pred", "y_excess", horizon=5)["spread_sharpe"]
    ss10 = metrics.summarize(df, "pred", "y_excess", horizon=10)["spread_sharpe"]
    assert np.isfinite(ss5) and np.isfinite(ss10)
    # 年化因子 sqrt(252/5) / sqrt(252/10) = sqrt(2)
    assert abs(ss5 / ss10 - np.sqrt(2)) < 1e-6, "spread_sharpe 未按 horizon 年化"


def test_bootstrap_nan_guarded():
    """N8：含 NaN 的收益序列不得污染 bootstrap 统计量（不得出现 nan 区间或崩溃）。"""
    rets = [0.01, 0.02, np.nan, 0.015, 0.0, 0.03, np.nan, 0.012]
    lo, hi = metrics.bootstrap_ci(rets, lambda x: float(x.mean()), seed=1)
    assert np.isfinite(lo) and np.isfinite(hi)
    out = metrics.backtest_bootstrap(rets, rebalance_freq=10, seed=1)
    if out:
        assert np.isfinite(out["total_return_ci"][0])


def test_decile_monotonicity_detects_monotonic():
    """N11：强单调面板应判为单调向上、首尾组差为正。"""
    rng = np.random.default_rng(2)
    rows = []
    for s in range(50):
        # 预测越高，未来收益越高（强单调）
        pred = s / 50.0
        y = pred + rng.normal(scale=0.05)
        rows.append({"date": "2024-03-01", "symbol": f"S{s}", "pred": pred, "y_excess": y})
    df = pd.DataFrame(rows)
    m = metrics.decile_monotonicity(df, n_groups=5)
    assert m["monotonic"] is True
    assert m["spread"] > 0
    assert m["slope"] > 0


def test_summarize_exposes_icir_annual_and_ndays():
    """N18：summarize 透传 ic_summary 的 icir_annual / n_days 键。"""
    df = _mono_panel()
    out = metrics.summarize(df, "pred", "y_excess")
    assert "icir_annual" in out and np.isfinite(out["icir_annual"])
    assert "n_days" in out and out["n_days"] > 0


def test_top_bottom_spread_min_n_param():
    """N19：min_n 控制最少截面样本数；小于阈值返回空。"""
    df = _mono_panel(n_days=1, n_sym=8, seed=5)   # 单日仅 8 只
    empty = metrics.top_bottom_spread(df, min_n=10)
    assert empty.dropna().empty
    ok = metrics.top_bottom_spread(df, min_n=5)
    assert not ok.dropna().empty
