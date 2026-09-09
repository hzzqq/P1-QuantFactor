"""阶段 15：事件因子(EV 47维) · 收尾完善 —— 一键完成
  ① 合并 主CSV(t01~t11) + 分片A(t12~t20子集) + 分片B → report_tune_grid_ev.csv(20组)
  ② 打印 EV(47维) vs 基线(38维) 对比表（复用 report_tune_grid.csv）
  ③ 选最优配置（按 2026 净收益）
  ④ 全量 1427 确认：用最优 tag 在 --max-symbols 0 下重跑，落 report_tune_ev_full.csv
  ⑤ 导出正式 EV 生产信号（覆盖全 1427 只）→ signals/signal_ev_h10.json

用法：
  python scripts/15_finalize_ev.py            # 全量收尾（含全量确认，约 +25min）
  python scripts/15_finalize_ev.py --no-full # 只做 ①②③ 合并与对比，不跑全量确认
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths  # noqa: E402
from shared.logging_utils import get_logger  # noqa: E402

logger = get_logger("P1.finalize_ev")
PROCESSED = paths.DATA / "P1" / "processed"
VENV_PY = ROOT / "env" / "Scripts" / "python.exe"

# 动态加载 14，复用其 run_one / load_ev_dataset / GRID / 常量
_spec = importlib.util.spec_from_file_location(
    "ev14", PROJ / "scripts" / "14_event_factor_iterate.py")
ev14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev14)


def merge_grid() -> pd.DataFrame:
    """合并主 CSV + 分片 A/B/C，按 tag 去重，落盘 report_tune_grid_ev.csv。"""
    files = ["report_tune_grid_ev.csv",
             "report_tune_ev_shardA.csv",
             "report_tune_ev_shardB.csv",
             "report_tune_ev_shardC.csv"]
    rows: dict[str, dict] = {}
    for f in files:
        p = PROCESSED / f
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for r in df.to_dict(orient="records"):
            rows[r["tag"]] = r  # 后者覆盖前者（同 tag 不会冲突，分片互斥）
    merged = pd.DataFrame(list(rows.values()))
    merged = merged.sort_values("tag").reset_index(drop=True)
    merged.to_csv(PROCESSED / "report_tune_grid_ev.csv", index=False,
                  float_format="%.6f")
    logger.info("合并完成：%d 组 → %s", len(merged),
                PROCESSED / "report_tune_grid_ev.csv")
    return merged


def print_comparison(ev: pd.DataFrame) -> None:
    base_p = PROCESSED / "report_tune_grid.csv"
    if not base_p.exists():
        logger.warning("基线报告缺失 %s，跳过对比", base_p)
        return
    base = pd.read_csv(base_p)
    evv = ev[ev["error"].isin([None, float("nan")])] if "error" in ev else ev
    print("\n" + "=" * 86)
    print("  EV(47维=38+牧羊人9) vs 基线(38维) · 同口径(800子集 / 严格2026 hold-out)")
    print("=" * 86)
    table = [
        ("均值 IC", evv["ic_2026"].mean(), base["ic_2026"].mean()),
        ("均值 ICIR", evv["icir_2026"].mean(), base["icir_2026"].mean()),
        ("均值 2026净", evv["net_2026"].mean(), base["net_2026"].mean()),
        ("均值 夏普", evv["sharpe_2026"].mean(), base["sharpe_2026"].mean()),
        ("最优(按净)净", evv["net_2026"].max(), base["net_2026"].max()),
    ]
    print(f"  {'指标':<14}{'EV 47维':>14}{'基线 38维':>14}{'判读':>20}")
    for name, a, b in table:
        delta = (a - b)
        print(f"  {name:<14}{a:>14.4f}{b:>14.4f}{delta:>+14.4f}")
    best_ev = evv.sort_values("net_2026", ascending=False).iloc[0]
    best_base = base.sort_values("net_2026", ascending=False).iloc[0]
    print(f"  EV 最优: {best_ev['tag']} 净 {best_ev['net_2026']*100:.2f}% / "
          f"夏普 {best_ev['sharpe_2026']:.2f} / IC {best_ev['ic_2026']:.4f}")
    print(f"  基线最优: {best_base['tag']} 净 {best_base['net_2026']*100:.2f}% / "
          f"夏普 {best_base['sharpe_2026']:.2f} / IC {best_base['ic_2026']:.4f}")
    print("=" * 86)


def pick_best(ev: pd.DataFrame) -> dict:
    ok = ev[~ev.get("error", pd.Series([None]*len(ev))).notna()] \
        if "error" in ev else ev
    best = ok.sort_values("net_2026", ascending=False).iloc[0].to_dict()
    logger.info("最优配置(按2026净): %s → 净 %.2f%% / 夏普 %.2f",
                best["tag"], best["net_2026"] * 100, best["sharpe_2026"])
    return best


def full_confirm(best: dict, horizon: int, seed: int) -> None:
    """全量 1427 确认：最优 tag 在 max_symbols=0 下重跑，落 report_tune_ev_full.csv。"""
    full_csv = PROCESSED / "report_tune_ev_full.csv"
    done = set()
    if full_csv.exists():
        try:
            done = set(pd.read_csv(full_csv)["tag"].tolist())
        except Exception:
            done = set()
    if best["tag"] in done:
        logger.info("全量确认已存在 %s，跳过", best["tag"])
        return

    logger.info("加载全量 EV 数据集 (max_symbols=0) ...")
    _, X3d, y3d, dates, symbols, _ = ev14.load_ev_dataset(horizon, 0.5)
    cfg = next(c for c in ev14.GRID if c["tag"] == best["tag"])
    args = types.SimpleNamespace(
        horizon=horizon, epochs=15, batch_size=1024, stride=5,
        max_symbols=0, threads=8, label_clip=0.5,
        train_end="2024-12-31", valid_end="2025-12-31",
        top_pct=0.1, rebalance_freq=15,
    )
    logger.info("全量重跑 %s（1427 只，约 20~25min）...", cfg["tag"])
    r = ev14.run_one(cfg, X3d, y3d, dates, symbols, None, args, seed)
    rows = []
    if full_csv.exists():
        rows = pd.read_csv(full_csv).to_dict(orient="records")
    rows.append(r)
    pd.DataFrame(rows).to_csv(full_csv, index=False, float_format="%.6f")
    logger.info("[全量 %s] 完成 → IC=%.4f ICIR=%.4f | 净 %.2f%% 夏普 %.2f",
                cfg["tag"], r["ic_2026"], r["icir_2026"],
                (r["net_2026"] or 0) * 100, r["sharpe_2026"])


def export_signal(best: dict, horizon: int) -> None:
    pred = PROCESSED / f"pred_gru_ev_{best['tag']}_h{horizon}.parquet"
    if not pred.exists():
        logger.error("全量预测缺失 %s，无法导出", pred)
        return
    cmd = [
        str(VENV_PY),
        str(PROJ / "scripts" / "06_export_signal.py"),
        "--pred", str(pred), "--model", "ev", "--horizon", str(horizon),
    ]
    logger.info("导出正式 EV 信号：%s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-full", action="store_true", help="只合并+对比，不跑全量确认")
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()

    cfg0 = ev14.load_config(ev14.CONFIG_PATH)
    seed = cfg0.get_path("seed", 42)
    horizon = args.horizon or cfg0.get_path("label.horizon", 10)

    ev = merge_grid()
    print_comparison(ev)
    best = pick_best(ev)

    if not args.no_full:
        full_confirm(best, horizon, seed)
        export_signal(best, horizon)
        logger.info("收尾完成：正式信号已覆盖全 1427 只")
    else:
        logger.info("（--no-full）跳过全量确认与信号导出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
