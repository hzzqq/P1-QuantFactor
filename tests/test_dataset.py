"""R1：时序切分标签泄漏钳制测试。"""
import numpy as np
import pandas as pd

from src.models import dataset as ds_mod


def _make_dates():
    return pd.date_range("2020-01-01", "2022-12-31", freq="B").values


def test_make_masks_label_horizon_clamps_train():
    dates = _make_dates()
    n = len(dates)
    train_end = dates[n // 3]
    valid_end = dates[2 * n // 3]
    h = 10
    tr, va, te, _ = ds_mod.make_masks(dates, train_end, valid_end,
                                      label_horizon=h)
    d = pd.DatetimeIndex(dates)
    pos = d.get_indexer([pd.Timestamp(train_end)])[0]
    cutoff = d[max(0, pos - h)]
    # 训练样本最晚日期须 <= train_end 前 h 个交易日 → 标签窗口不越界
    assert d[tr].max() <= cutoff, f"标签泄漏：训练最晚 {d[tr].max()} > 钳制 {cutoff}"


def test_make_masks_label_horizon_clamps_valid():
    dates = _make_dates()
    n = len(dates)
    train_end = dates[n // 3]
    valid_end = dates[2 * n // 3]
    h = 10
    tr, va, te, _ = ds_mod.make_masks(dates, train_end, valid_end,
                                      label_horizon=h)
    d = pd.DatetimeIndex(dates)
    posv = d.get_indexer([pd.Timestamp(valid_end)])[0]
    cutoffv = d[max(0, posv - h)]
    assert d[va].max() <= cutoffv, f"valid 标签泄漏：最晚 {d[va].max()} > {cutoffv}"


def test_make_masks_default_no_clamp_backward_compat():
    dates = _make_dates()
    n = len(dates)
    train_end = dates[n // 3]
    valid_end = dates[2 * n // 3]
    h = 10
    tr0, _, _, _ = ds_mod.make_masks(dates, train_end, valid_end)  # 默认 0
    d = pd.DatetimeIndex(dates)
    pos = d.get_indexer([pd.Timestamp(train_end)])[0]
    cutoff = d[max(0, pos - h)]
    # 旧行为允许训练样本临近边界（标签窗口越界）
    assert d[tr0].max() > cutoff
    # 默认返回仍为 bool 数组且三类互斥覆盖
    assert tr0.dtype == bool


def test_make_masks_mutually_exclusive():
    dates = _make_dates()
    n = len(dates)
    tr, va, te, _ = ds_mod.make_masks(dates, dates[n // 3], dates[2 * n // 3],
                                      label_horizon=10)
    assert not (tr & va).any()
    assert not (va & te).any()
    assert not (tr & te).any()


def test_make_masks_sym_always_bool_array(N7=True):
    """N7：即使不传 symbols，返回的 sym 也必须是布尔数组（不得为 None 导致下游崩溃）。"""
    dates = _make_dates()
    n = len(dates)
    tr, va, te, sym = ds_mod.make_masks(dates, dates[n // 3], dates[2 * n // 3])
    assert sym is not None, "sym 不得为 None"
    assert isinstance(sym, np.ndarray)
    assert sym.dtype == bool


def test_build_3d_float32_dtype(N10=True):
    """N10：build_3d 输出 X3d 必须为 float32（落盘与回读内存减半）。"""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=20, freq="B")
    syms = [f"S{i}" for i in range(6)]
    rows = []
    for s in syms:
        for d in dates:
            rows.append({"date": d, "symbol": s,
                         "f1": rng.normal(), "f2": rng.normal(),
                         "y_excess": rng.normal()})
    df = pd.DataFrame(rows)
    X3d, y3d, dts, sms = ds_mod.build_3d(df, ["f1", "f2"], y_col="y_excess")
    assert X3d.dtype == np.float32, f"X3d 应为 float32，实得 {X3d.dtype}"
    assert y3d.dtype == np.float32
