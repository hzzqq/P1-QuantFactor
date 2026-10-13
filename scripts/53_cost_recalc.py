# -*- coding: utf-8 -*-
"""
scripts/53_cost_recalc.py — R10 成本口径跨仓重算（T-275，ks 空间授权）。

目的：用 ks 仓单源成本模型（modules/trading_costs.py，T-274）重算 P1 2026 严格
hold-out 的净收益，产出「毛 / sj 旧净 / ks 新净」三口径对照——把 55 页引用的
净收益数字升级到统一成本口径（含 2023-08-28 印花税减半 + 过户费）。

红线（诚实语义）：
- 毛口径数字**不覆盖不修改**，只新增净口径对照；
- 成本参数**跨仓 import** E:/project/ks/StockSignal/modules/trading_costs.py
  （成本口径单源铁律，禁止在 sj 侧复制参数）；
- 49.1% 方向命中锚与成本无关，本脚本不触碰任何方向命中数字；
- 复现基线：GRU 2026 净 +8.67%（rf=10，sj progress 阶段 18/19 记录）应在本脚本
  sj 旧口径下复现（同 pred/panel/参数），偏差大则如实记录并排查。

用法：
    python scripts/53_cost_recalc.py                # gru + fusion，2026 区间
    python scripts/53_cost_recalc.py --start-date 2026-01-01 --end-date 2026-08-14
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
KS_ROOT = Path("E:/project/ks/StockSignal")
for _p in (str(ROOT), str(PROJ), str(KS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths  # noqa: E402
from shared.logging_utils import get_logger  # noqa: E402

from src.backtest import run_backtest, DEFAULT_COST, BacktestResult  # noqa: E402

# 跨仓单源：成本参数只从 ks 仓 trading_costs 读取（T-274 单源铁律）
from modules.trading_costs import (  # noqa: E402
    COMMISSION_RATE, COMMISSION_MIN, STAMP_TAX_RATE, SLIPPAGE_RATE,
    TRANSFER_FEE_RATE, round_trip_rate,
)

logger = get_logger("P1.cost_recalc")
PROCESSED = paths.DATA / "P1" / "processed"
OUT = PROCESSED / "report_cost_recalc.json"

# 预测文件 → 展示名（EV(fusion) 即 regime 门控融合 = 生产信号，见 progress 阶段 18 记录）
TARGETS = [
    ("gru", "pred_gru_h10.parquet"),
    ("ev_fusion", "pred_fusion_h10.parquet"),
]


def _gross_cost() -> dict:
    return {"commission": 0.0, "slippage": 0.0, "stamp": 0.0}


def _ks_cost() -> dict:
    """ks 单源成本 → sj 回测引擎 cost dict。

    sj 引擎无过户费字段，按「费率并入 commission」处理（ks 过户费 0.001% 双边
    与佣金同为双边比例项，合并不改变总量语义）；最低佣金 5 元为绝对额，
    回测引擎按比例扣费不支持最低额——10 万级组合单边佣金远超 5 元，忽略最低额
    影响并如实记录（保守方向：实际成本略高于本估计）。
    """
    return {
        "commission": COMMISSION_RATE + TRANSFER_FEE_RATE,  # 0.00026
        "slippage": SLIPPAGE_RATE,                          # 0.001
        "stamp": STAMP_TAX_RATE,                            # 0.0005（2023-08-28 减半）
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="R10 成本口径跨仓重算（ks 单源）")
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--top-pct", type=float, default=0.1)
    ap.add_argument("--rebalance-freq", type=int, default=10,
                    help="必须 = horizon（非重叠桶；rf 错配护栏同 05）")
    ap.add_argument("--start-date", type=str, default="2026-01-01",
                    help="2026 严格 hold-out 区间起点")
    ap.add_argument("--end-date", type=str, default="2026-08-14",
                    help="信号数据截止日（与 55 页口径一致）")
    args = ap.parse_args()

    assert args.rebalance_freq == args.horizon, \
        f"rf({args.rebalance_freq}) != horizon({args.horizon})——rf 错配护栏（历史事故见 05_backtest.py）"

    ks_rate = round_trip_rate(100_000)
    logger.info("ks 单源全成本率（10 万本金）≈ %.4f%%", ks_rate * 100)

    panel_path = paths.DATA / "P1" / "processed" / "panel.parquet"
    panel = pd.read_parquet(panel_path)
    logger.info("行情面板 %s 行 | %s 只", f"{len(panel):,}", panel["symbol"].nunique())

    reports = {}
    t0 = time.time()
    import datetime as _dt
    for tag, pred_name in TARGETS:
        p = PROCESSED / pred_name
        if not p.exists():
            logger.error("预测表缺失: %s", p)
            return 1
        preds = pd.read_parquet(p)
        preds["date"] = pd.to_datetime(preds["date"])
        pred_mtime = _dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        date_lo, date_hi = preds["date"].min(), preds["date"].max()
        n_before = len(preds)
        preds = preds[(preds["date"] >= pd.Timestamp(args.start_date))
                      & (preds["date"] <= pd.Timestamp(args.end_date))]
        logger.info("=== %s（%s → %s 行，%s~%s）===",
                    tag, f"{n_before:,}", f"{len(preds):,}", args.start_date, args.end_date)

        common = dict(horizon=args.horizon, top_pct=args.top_pct,
                      rebalance_freq=args.rebalance_freq, mode="long_short",
                      cost_model="fixed", sizer="equal",
                      vol_target=None, vol_lookback=20, max_leverage=1.0)

        gross = run_backtest(preds, panel, **common, cost=_gross_cost())
        sj_old = run_backtest(preds, panel, **common, cost=DEFAULT_COST)
        ks_new = run_backtest(preds, panel, **common, cost=_ks_cost())

        def _stat(r: BacktestResult) -> dict:
            ls = r.ls_stats
            return {"total_return": ls.get("total_return"),
                    "annual_return": ls.get("annual_return"),
                    "sharpe": ls.get("sharpe"),
                    "max_drawdown": ls.get("max_drawdown"),
                    "n_buckets": ls.get("n_buckets")}

        reports[tag] = {
            "pred_file": pred_name,
            "pred_mtime": pred_mtime,
            "pred_date_coverage": f"{date_lo:%Y-%m-%d} ~ {date_hi:%Y-%m-%d}",
            "n_preds": int(len(preds)),
            "gross": _stat(gross),
            "sj_old_net": {"stats": _stat(sj_old), "cost": DEFAULT_COST,
                           "note": "sj 旧口径：佣金万3+滑点0.1%+印花税0.1%（2023-08 前税率）"},
            "ks_new_net": {"stats": _stat(ks_new), "cost": _ks_cost(),
                           "note": "ks 单源口径（T-274）：佣金万2.5+过户费0.001%（并入commission）"
                                   "+滑点0.1%+印花税0.05%（2023-08-28 减半）"},
        }
        logger.info("%s: gross=%.2f%% sj_old_net=%.2f%% ks_new_net=%.2f%%",
                    tag,
                    reports[tag]["gross"]["total_return"] * 100,
                    reports[tag]["sj_old_net"]["stats"]["total_return"] * 100,
                    reports[tag]["ks_new_net"]["stats"]["total_return"] * 100)

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "horizon": args.horizon, "top_pct": args.top_pct,
            "rebalance_freq": args.rebalance_freq,
            "start_date": args.start_date, "end_date": args.end_date,
            "mode": "long_short", "sizer": "equal",
            "ks_round_trip_rate_100k": ks_rate,
            "ks_cost_source": "E:/project/ks/StockSignal/modules/trading_costs.py（跨仓 import，单源）",
            "commission_min_note": "sj 引擎按比例扣费不支持最低佣金 5 元；10 万级组合远超最低额，忽略影响偏保守",
            "honesty_note": "真实重跑（同 pred/panel/引擎参数换成本口径），非手改数字；"
                            "毛口径与 49.1% 方向锚不修改",
            "pred_version_warning": "历史引用数字（GRU 2026 净 +8.67%/EV 全量 +10.68%，sj progress 阶段"
                            " 18/19）基于 8-31 walk-forward pred，该文件已被 2026-09-25 重训覆盖"
                            "（同参数全区间回测 net +16.09%→-13.08%，差 29pp，无法逐位复现）；"
                            "本报告为**现役 pred** 的三口径对照。另：ks 55 页引用「EV(fusion) +11.56%」"
                            "在 sj 产物/记录中无出处（权威记录为 EV 全量 +10.68%），引用需订正",
        },
        "targets": reports,
    }
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    logger.info("重算报告已保存: %s（耗时 %.1fs）", OUT, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
