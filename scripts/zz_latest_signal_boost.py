"""T-337：P1 四路信号追平 boost（消除 dataset_h10 标签滞后导致的信号日期缺口）。

背景：
    dataset_h10 需要 y_excess 标签（T+1~T+10 前视），永远只含到「当前日-10 交易日」，
    而 S01 全量重建面板后 day-1 的新数据要等下一次自动化全链才进信号 → 信号尾部
    存在最长约 10 个交易日的真空。本脚本在**不重训、不改任何现有文件**的前提下
    （T-336 授权边界：additive-only，仅新增本文件），对最新 panel 用与 02 完全相同的
    因子口径补算 boost 日期的因子行（y_excess 如实置 NaN，不编造标签），然后：

      1. GRU 路   ：拼 dataset_h10 + 新行 → 尾窗 60 交易日 → 复用生产权重
                    gru_best_h10_2026.pt（seq20）→ 追加 pred_gru_h10.parquet
      2. EV 路    ：新行 + 9 牧羊人列（复刻 14 build_event_dataset：ffill→全历史
                    z-score→按日广播）→ 生产权重 gru_ev_t10_seq40_h128_lr1e-3_h10.pt
                    （seq40）→ 追加 pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet
      3. baseline v39（S06 口径）：dataset_h10_v39（39 维，log_amount 从缓存 merge），
                    train=_year∈[2021,2023] va=[2024,2025]（IC 早停，seed42/deterministic，
                    与 28+walk_forward 逐参数一致）→ 追加 pred_baseline_h10_v39.parquet
      4. baseline 原版（S07 口径）：dataset_h10（38 维）同上 → 追加 pred_baseline_h10.parquet
      之后由 zz_daily_push.sh S08（ensemble）→ S09（导出）→ S10（复制到 ks）收尾。

口径要点（与生产逐一对齐）：
    - 因子：iter_factors + winsorize_mad(5.0) + cross_sectional_zscore（按日截面，
      切窗不影响值）+ 行过滤 min_feature_ratio=0.6（非 NaN 因子 ≥ 60%），与
      pipeline.build_dataset 完全同参。
    - GRU/EV 尾窗内历史行复刻 S04/S05 的预处理：replace inf→NaN → dropna(y_excess)
      → clip |y|≤0.5（行移除 → X3d 对应格为 NaN，窗口内容与生产一致）；新行不过
      标签过滤（y 本就是 NaN）。SequenceDataset 用 y3d_dummy=全 0 通过 y_valid 检查，
      再覆写 indices 只留「新行真实存在的 (symbol,date) 对」（停牌股无当日行，不产样本；
      窗口 NaN≤0.2 检查仍基于真实 X3d）。
    - 已知微漂移（备案）：EV 的 9 列 z-score 用当前全历史 mean/std（与 10-12 构建
      时刻略有差异）；LGBM 每次重训本就与 S06 每日重训同口径，非漂移。
    - 幂等：每个 pred 文件 append 前先删已存在的 boost 日期行再拼（可安全重跑）；
      覆写前 .bak 备份。

用法：
    python scripts/zz_latest_signal_boost.py --threads 8
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]      # E:/project/sj
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared import paths  # noqa: E402
from shared.logging_utils import get_logger  # noqa: E402
from src.features.pipeline import (  # noqa: E402
    _stack, cross_sectional_zscore, iter_factors, winsorize_mad,
)
from src.models import baseline_lgb  # noqa: E402
from src.models import dataset as ds_mod  # noqa: E402
from src.models import gru_attn  # noqa: E402
from src.models import trainer  # noqa: E402

logger = get_logger("P1.boost_latest")

PROCESSED = paths.DATA / "P1" / "processed"
MODELS_DIR = paths.MODELS / "P1"
SHEPHERD_CSV = Path(r"E:/project/ks/StockSignal/data/shepherd_history.csv")
LA_PATH = PROCESSED / "_log_amount.parquet"

SHEEP_COLS = ["up_count", "down_count", "flat_count", "limit_up", "limit_down",
              "red_ratio", "connect_hl", "zt_fail_ratio", "zt_prev_ret"]

# 与 config/default.yaml + pipeline.build_dataset 默认值逐一对齐
WINDOWS = [5, 10, 20, 60, 120]
WINSOR = 5.0
MIN_FEATURE_RATIO = 0.6
PANEL_TAIL_DAYS = 200        # 因子复算窗口（mom_120 需 ≥121 日历史，200 有余量）
SEQ_WINDOW_DAYS = 60         # GRU/EV 尾窗（seq40 需 39 日历史，60 有余量）
LABEL_CLIP = 0.5             # 与 S04/S05/28 一致

GRU_CKPT = MODELS_DIR / "gru_best_h10_2026.pt"
GRU_SEQ = 20
EV_CKPT = MODELS_DIR / "gru_ev_t10_seq40_h128_lr1e-3_h10.pt"
EV_SEQ = 40

PRED_GRU = PROCESSED / "pred_gru_h10.parquet"
PRED_EV = PROCESSED / "pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet"
PRED_V39 = PROCESSED / "pred_baseline_h10_v39.parquet"
PRED_BASE = PROCESSED / "pred_baseline_h10.parquet"

TEST_YEAR = 2026             # boost 只补当前年；train/valid 年份由 walk_forward 公式推出


def _load_sd(ckpt: Path) -> dict:
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def detect_boost_dates() -> list[pd.Timestamp]:
    """boost 日期 = panel 交易日 > dataset_h10 最大日期（数据集永远滞后于面板）。"""
    ds_max = pd.to_datetime(pd.read_parquet(
        PROCESSED / "dataset_h10.parquet", columns=["date"])["date"]).max()
    panel_dates = pd.DatetimeIndex(np.sort(pd.to_datetime(pd.read_parquet(
        PROCESSED / "panel.parquet", columns=["date"])["date"]).unique()))
    boost = [d for d in panel_dates if d > ds_max]
    logger.info("dataset_h10 max=%s | panel max=%s | boost %s 个交易日: %s ~ %s",
                ds_max.date(), panel_dates.max().date(), len(boost),
                boost[0].date() if boost else "-", boost[-1].date() if boost else "-")
    return boost


def compute_new_factor_rows(panel: pd.DataFrame, boost_dates: list[pd.Timestamp],
                            fn38: list[str]) -> pd.DataFrame:
    """从 panel 尾窗复算 boost 日期的 38 因子行（口径 = pipeline.build_dataset）。"""
    t0 = time.time()
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    dup = int(panel.duplicated(subset=["date", "symbol"]).sum())
    if dup:
        logger.warning("面板尾窗含 %s 个重复 (date,symbol) 行，已去重（保留末条）", dup)
        panel = panel.drop_duplicates(subset=["date", "symbol"], keep="last")

    factor_cols: dict[str, pd.Series] = {}
    for name, mat in iter_factors(panel, WINDOWS):
        processed = cross_sectional_zscore(winsorize_mad(mat, WINSOR))
        factor_cols[name] = _stack(processed)
        del processed, mat

    X = pd.DataFrame(factor_cols)
    del factor_cols
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.reset_index()
    X["date"] = pd.to_datetime(X["date"])
    X = X[X["date"].isin(pd.DatetimeIndex(boost_dates))]

    # 行过滤：与 build_dataset 相同的 min_feature_ratio 口径（不设标签条件）
    thresh = max(1, int(len(fn38) * MIN_FEATURE_RATIO))
    before = len(X)
    X = X[X[fn38].notna().sum(axis=1) >= thresh]
    logger.info("新因子行：boost 日截面 %s 行 → 因子≥%s/%s 过滤后 %s 行",
                f"{before:,}", thresh, len(fn38), f"{len(X):,}")

    out = X[["date", "symbol"] + fn38].copy()
    out["y_excess"] = np.nan          # 诚实语义：无标签，绝不编造
    out["y_cls"] = np.nan
    out["y_fwd"] = np.nan
    out[fn38] = out[fn38].astype("float32")
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    logger.info("新因子行就绪 %s 行 (%s ~ %s)，%.1fs",
                f"{len(out):,}", out["date"].min().date(), out["date"].max().date(),
                time.time() - t0)
    return out


def gru_boost_predict(ds_name: str, meta_name: str, new_rows: pd.DataFrame,
                      seq_len: int, ckpt: Path, threads: int, tag: str) -> pd.DataFrame:
    """GRU/EV 共用：拼历史+新行 → 尾窗 build_3d → 生产权重推理 boost 日期。"""
    t0 = time.time()
    meta = json.load(open(PROCESSED / meta_name, encoding="utf-8"))
    fn = meta["factor_names"]
    cols = ["date", "symbol"] + fn + ["y_excess"]

    # 边界下推读取：历史部分只读尾窗所需日期
    ds_dates = pd.DatetimeIndex(np.sort(pd.to_datetime(pd.read_parquet(
        PROCESSED / ds_name, columns=["date"])["date"]).unique()))
    ext_dates = pd.DatetimeIndex(np.sort(np.unique(np.concatenate(
        [ds_dates.values, new_rows["date"].unique()]))))
    tail_dates = ext_dates[-SEQ_WINDOW_DAYS:]

    hist = pd.read_parquet(PROCESSED / ds_name, columns=cols,
                           filters=[("date", ">=", tail_dates[0])])
    # 复刻 S04/S05 历史行预处理（行移除 → X3d 对应格为 NaN，与生产窗口一致）
    hist = hist.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    hist = hist[hist["y_excess"].abs() <= LABEL_CLIP]

    data = pd.concat([hist, new_rows[cols]], ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["date"].isin(tail_dates)]
    logger.info("[%s] 尾窗 %s 行（历史 %s + 新 %s）| %s 天 × %s 因子",
                tag, f"{len(data):,}", f"{len(hist):,}", f"{len(new_rows):,}",
                len(tail_dates), len(fn))

    X3d, y3d, dates, symbols = ds_mod.build_3d(data, fn)
    del data, hist
    gc.collect()
    n_f = X3d.shape[2]

    sd = _load_sd(ckpt)
    w_dim = int(sd["input_norm.weight"].shape[0])
    if w_dim != n_f:
        logger.error("[%s] 权重 in_dim %s != 数据 %s —— 拒绝推理", tag, w_dim, n_f)
        raise SystemExit(1)
    hid = int(sd["gru.weight_ih_l0"].shape[0] // 3)
    nl = sum(1 for k in sd if k.startswith("gru.weight_ih_l"))

    # y3d_dummy=0 让 y_valid 全过；窗口 NaN≤0.2 检查仍基于真实 X3d
    ds = ds_mod.SequenceDataset(
        X3d, np.zeros_like(y3d), seq_len, None,
        (pd.to_datetime(pd.Series(dates)) >= tail_dates[0]).to_numpy(), date_stride=1)
    idx = ds.indices
    if len(idx) == 0:
        logger.error("[%s] 尾窗无候选样本", tag)
        raise SystemExit(1)
    keys = pd.MultiIndex.from_arrays(
        [pd.to_datetime(dates[idx[:, 1]]), np.asarray(symbols)[idx[:, 0]]])
    new_keys = pd.MultiIndex.from_arrays([new_rows["date"], new_rows["symbol"]])
    ds.indices = idx[keys.isin(new_keys)]
    logger.info("[%s] 候选 %s → 命中新行 %s 样本 | arch h=%s l=%s seq=%s",
                tag, f"{len(idx):,}", f"{len(ds.indices):,}", hid, nl, seq_len)

    model = gru_attn.GRUAttention(n_f, hidden=hid, n_layers=nl)
    model.load_state_dict(sd)
    pred = trainer.predict(model, ds, num_threads=threads)
    fidx = ds.indices
    out = pd.DataFrame({
        "date": pd.to_datetime(dates[fidx[:, 1]]),
        "symbol": np.asarray(symbols)[fidx[:, 0]],
        "pred": np.asarray(pred, dtype=np.float32),
        "y_excess": np.full(len(fidx), np.nan, dtype=np.float32),
        "year": np.full(len(fidx), TEST_YEAR, dtype=np.int32),
    })
    logger.info("[%s] 推理完成 %s 行，%.1fs", tag, f"{len(out):,}", time.time() - t0)
    return out


def build_sheep_z() -> pd.DataFrame:
    """复刻 14 build_event_dataset 的牧羊人列预处理（z-score 用当前全历史 mean/std）。"""
    sheep = pd.read_csv(SHEPHERD_CSV)
    sheep["date"] = pd.to_datetime(sheep["date"])
    sheep = sheep[["date"] + SHEEP_COLS].sort_values("date").drop_duplicates("date")
    sheep[SHEEP_COLS] = sheep[SHEEP_COLS].ffill()
    for c in SHEEP_COLS:
        m, s = sheep[c].mean(), sheep[c].std()
        sheep[c] = (sheep[c] - m) / s if (s and s == s and s > 0) else 0.0
    sheep[SHEEP_COLS] = sheep[SHEEP_COLS].fillna(0.0)
    return sheep


def run_baseline(ds_name: str, meta_name: str, new_rows: pd.DataFrame,
                 tag: str) -> pd.DataFrame:
    """复刻 28+walk_forward 的 2026 重训口径，对新行出预测（S06/S07 同源）。"""
    t0 = time.time()
    fn = json.load(open(PROCESSED / meta_name, encoding="utf-8"))["factor_names"]
    lo = pd.Timestamp(year=TEST_YEAR - 5, month=1, day=1)
    hi = pd.Timestamp(year=TEST_YEAR, month=12, day=31)
    data = pd.read_parquet(PROCESSED / ds_name, columns=["date", "symbol"] + fn + ["y_excess"],
                           filters=[("date", ">=", lo), ("date", "<=", hi)])
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["y_excess"])
    data = data[data["y_excess"].abs() <= LABEL_CLIP]
    data[fn + ["y_excess"]] = data[fn + ["y_excess"]].astype(np.float32)
    data["_year"] = pd.to_datetime(data["date"]).dt.year

    y = TEST_YEAR
    tr = data[(data["_year"] >= y - 5) & (data["_year"] <= y - 3)]
    va = data[(data["_year"] >= y - 2) & (data["_year"] <= y - 1)]
    if len(tr) < 2000:
        logger.error("[%s] 训练样本不足 %s", tag, f"{len(tr):,}")
        raise SystemExit(1)
    logger.info("[%s] LGBM 重训：train %s 行(%s~%s) valid %s 行(%s~%s) | %s 因子",
                tag, f"{len(tr):,}", tr["_year"].min(), tr["_year"].max(),
                f"{len(va):,}", va["_year"].min(), va["_year"].max(), len(fn))
    model = baseline_lgb.train(tr[fn], tr["y_excess"], va[fn], va["y_excess"],
                               None, 400, eval_metric="ic")
    X_te = new_rows[fn].astype(np.float32)
    pred = model.predict(X_te, num_iteration=model.best_iteration)
    out = pd.DataFrame({
        "date": new_rows["date"].values,
        "symbol": new_rows["symbol"].values,
        "pred": np.asarray(pred, dtype=np.float32),
        "y_excess": np.full(len(new_rows), np.nan, dtype=np.float32),
        "year": np.full(len(new_rows), y, dtype=np.int32),
    })
    logger.info("[%s] LGBM 完成：best_iter=%s，%s 行，%.1fs",
                tag, model.best_iteration, f"{len(out):,}", time.time() - t0)
    return out


def append_pred(out_path: Path, new_df: pd.DataFrame,
                boost_dates: list[pd.Timestamp], tag: str) -> None:
    """幂等追加：先删已存在 boost 日期行，再拼新行；覆写前 .bak 备份。"""
    old = pd.read_parquet(out_path)
    old["date"] = pd.to_datetime(old["date"])
    n_old, old_max = len(old), old["date"].max().date()
    old = old[~old["date"].isin(pd.DatetimeIndex(boost_dates))]
    dropped = n_old - len(old)
    merged = (pd.concat([old, new_df], ignore_index=True)
              .drop_duplicates(subset=["date", "symbol"], keep="last")
              .sort_values(["date", "symbol"]).reset_index(drop=True))
    bak = out_path.parent / (out_path.name + ".bak")
    shutil.copy2(out_path, bak)
    merged.to_parquet(out_path, index=False)
    logger.info("[%s] %s：%s 行(max %s) → %s 行(max %s)，删旧 boost 行 %s，备份 %s",
                tag, out_path.name, f"{n_old:,}", old_max, f"{len(merged):,}",
                merged["date"].max().date(), dropped, bak.name)


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 四路信号追平 boost（T-337）")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()

    boost_dates = detect_boost_dates()
    if not boost_dates:
        logger.info("面板不比 dataset_h10 新，无需 boost。退出。")
        return 0

    fn38 = json.load(open(PROCESSED / "dataset_h10_meta.json",
                          encoding="utf-8"))["factor_names"]
    logger.info("=== 1/4 复算 %s 个 boost 交易日的 38 因子行 ===", len(boost_dates))
    panel = pd.read_parquet(PROCESSED / "panel.parquet")
    udates = pd.DatetimeIndex(np.sort(pd.to_datetime(panel["date"]).unique()))
    tail = panel[pd.to_datetime(panel["date"]) >= udates[-PANEL_TAIL_DAYS]].copy()
    del panel
    new38 = compute_new_factor_rows(tail, boost_dates, fn38)
    del tail
    gc.collect()

    logger.info("=== 2/4 GRU 路（seq=%s，%s）===", GRU_SEQ, GRU_CKPT.name)
    gru_new = gru_boost_predict("dataset_h10.parquet", "dataset_h10_meta.json",
                                new38, GRU_SEQ, GRU_CKPT, args.threads, "gru")
    append_pred(PRED_GRU, gru_new, boost_dates, "gru")
    del gru_new
    gc.collect()

    logger.info("=== 3/4 EV 路（seq=%s，%s）===", EV_SEQ, EV_CKPT.name)
    fn47 = json.load(open(PROCESSED / "dataset_h10_ev_meta.json",
                          encoding="utf-8"))["factor_names"]
    sheep = build_sheep_z()
    new47 = new38.merge(sheep, on="date", how="left")
    for c in SHEEP_COLS:
        new47[c] = new47[c].fillna(0.0)
    new47 = new47[["date", "symbol"] + fn47 + ["y_excess"]]
    ev_new = gru_boost_predict("dataset_h10_ev.parquet", "dataset_h10_ev_meta.json",
                               new47, EV_SEQ, EV_CKPT, args.threads, "ev")
    append_pred(PRED_EV, ev_new, boost_dates, "ev")
    del ev_new, new47, sheep
    gc.collect()

    logger.info("=== 4/4 baseline 双路 LGBM 重训（S06/S07 口径）===")

    la = pd.read_parquet(LA_PATH)
    la["date"] = pd.to_datetime(la["date"])
    new39 = new38.merge(la[["date", "symbol", "log_amount"]],
                        on=["date", "symbol"], how="left")
    n_missing = int(new39["log_amount"].isna().sum())
    if n_missing:
        med = float(new39["log_amount"].median())
        logger.warning("log_amount 缺失 %s 行，用全局中位数 %.4f 填充（27 同款兜底）",
                       n_missing, med)
        new39["log_amount"] = new39["log_amount"].fillna(med)
    del la

    v39_new = run_baseline("dataset_h10_v39.parquet", "dataset_h10_v39_meta.json",
                           new39, "v39")
    append_pred(PRED_V39, v39_new, boost_dates, "v39")
    del v39_new, new39
    gc.collect()

    base_new = run_baseline("dataset_h10.parquet", "dataset_h10_meta.json",
                            new38, "base")
    append_pred(PRED_BASE, base_new, boost_dates, "base")
    del base_new, new38
    gc.collect()

    print("\n" + "=" * 78)
    print("  T-337 boost 完成（四路 pred 尾部）")
    print("=" * 78)
    for p, name in [(PRED_GRU, "gru"), (PRED_EV, "ev"),
                    (PRED_V39, "baseline_v39"), (PRED_BASE, "baseline_orig")]:
        d = pd.to_datetime(pd.read_parquet(p, columns=["date"])["date"])
        print(f"  {name:<14} {p.name:<48} max={d.max().date()} rows={len(d):,}")
    print(f"  boost 交易日：{boost_dates[0].date()} ~ {boost_dates[-1].date()}"
          f"（{len(boost_dates)} 天）| 总耗时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
