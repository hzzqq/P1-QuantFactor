"""回测引擎包。"""
from .engine import (
    LIMIT_UP,
    LIMIT_DOWN,
    DEFAULT_COST,
    attach_forward,
    BacktestResult,
    run_backtest,
    run_backtest_continuous,
)

__all__ = [
    "LIMIT_UP",
    "LIMIT_DOWN",
    "DEFAULT_COST",
    "attach_forward",
    "BacktestResult",
    "run_backtest",
    "run_backtest_continuous",
]
