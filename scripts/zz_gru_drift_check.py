"""阶段 ZZ：gru 特征分布漂移检查（2026-11-01）。

gru 生产推理（zz_predict_only_gru.py）输入 = dataset_h10.parquet 的 39 个因子，
按年用静态权重 gru_best_h10_{y}.pt（2026 窗口的权重训练于 date<=2024-12-31、验证 2025）。

本脚本对比各因子在 训练窗口（<=2024-12-31） vs 2026 推理窗口（2026-01~10-26）的分布：
  - 逐特征 mean/std/p1/p50/p99；
  - PSI（Population Stability Index，10 桶，以训练期分位建边界）：>0.10 显著漂移、>0.25 剧变；
  - 列出漂移 Top-N 特征，为「2026 样本外退化是否源于特征分布漂移」提供证据（描述性，非因果）。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import zz_paper_portfolio as zz  # noqa: E402

FEATURES = [
    "mom_5", "rev_5", "vol_5", "vr_5", "vstd_5", "pos_5", "bias_5",
    "mom_10", "rev_10", "vol_10", "vr_10", "vstd_10", "pos_10", "bias_10",
    "mom_20", "rev_20", "vol_20", "vr_20", "vstd_20", "pos_20", "bias_20",
    "mom_60", "rev_60", "vol_60", "vr_60", "vstd_60", "pos_60", "bias_60",
    "mom_120", "rev_120", "vol_120", "vr_120", "vstd_120", "pos_120", "bias_120",
    "ampl_1", "gap_1", "ret_1",
]


def psi(actual: np.ndarray, expected: np.ndarray, buckets: int = 10) -> float:
    """PSI：以 expected 分位数建桶边界，对比两分布占比。"""
    edges = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    ea, _ = np.histogram(expected, bins=edges)
    aa, _ = np.histogram(actual, bins=edges)
    p_e = ea / max(ea.sum(), 1)
    p_a = aa / max(aa.sum(), 1)
    p_a = np.clip(p_a, 1e-6, None)
    p_e = np.clip(p_e, 1e-6, None)
    return float(np.sum((p_a - p_e) * np.log(p_a / p_e)))


def daily_rank_ic(d: pd.DataFrame, f: str) -> float:
    """日均截面 rank IC（逐日 Spearman，再取均值）；向量化实现。"""
    s = d[["date", f, "y_excess"]].copy()
    s[f + "_r"] = s.groupby("date")[f].rank()
    s["y_r"] = s.groupby("date")["y_excess"].rank()
    g = s.groupby("date")
    mf = g[f + "_r"].transform("mean")
    my = g["y_r"].transform("mean")
    c = s[f + "_r"] - mf
    cy = s["y_r"] - my
    num = (c * cy).groupby(d["date"]).sum()
    den = np.sqrt((c * c).groupby(d["date"]).sum() * (cy * cy).groupby(d["date"]).sum())
    ic = (num / den.replace(0.0, np.nan)).dropna()
    return float(ic.mean()) if len(ic) else float("nan")


def main() -> int:
    d = pd.read_parquet(zz.ROOT / "data" / "P1" / "processed" / "dataset_h10.parquet")
    d = d.replace([np.inf, -np.inf], np.nan)
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()

    train = d[d["date"] <= pd.Timestamp("2024-12-31")]
    val = d[(d["date"] >= pd.Timestamp("2025-01-01")) & (d["date"] <= pd.Timestamp("2025-12-31"))]
    test = d[d["date"] >= pd.Timestamp("2026-01-01")]
    print(f"样本: 训练 {len(train):,} / 验证 {len(val):,} / 2026推理 {len(test):,}")

    rows = []
    for f in FEATURES:
        t = train[f].dropna().to_numpy()
        s = test[f].dropna().to_numpy()
        if len(t) < 1000 or len(s) < 1000:
            continue
        rows.append({
            "feature": f,
            "psi": round(psi(s, t), 4),
            "train_mean": round(float(t.mean()), 4), "test_mean": round(float(s.mean()), 4),
            "train_std": round(float(t.std()), 4), "test_std": round(float(s.std()), 4),
            "train_p50": round(float(np.median(t)), 4), "test_p50": round(float(np.median(s)), 4),
            "train_p99": round(float(np.percentile(t, 99)), 4), "test_p99": round(float(np.percentile(s, 99)), 4),
            "n_train": int(len(t)), "n_test": int(len(s)),
        })

    df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    n_sig = int((df["psi"] > 0.1).sum())
    n_sev = int((df["psi"] > 0.25).sum())
    print(f"显著漂移 PSI>0.10: {n_sig}/{len(df)} 个特征 | 剧变 PSI>0.25: {n_sev} 个")
    print("\nTop-12 漂移特征（PSI 降序）:")
    print(df.head(12).to_string(index=False))

    # ---- IC 对比：因子-收益关系（训练窗口 vs 2025 vs 2026） ----
    print("\n[IC] 逐因子日均截面 rank IC：训练(<=2024) vs 2025 vs 2026")
    ic_rows = []
    for f in FEATURES:
        it = daily_rank_ic(train, f)
        iv = daily_rank_ic(val, f)
        is_ = daily_rank_ic(test, f)
        ic_rows.append({"feature": f, "ic_train": round(it, 4), "ic_2025": round(iv, 4),
                        "ic_2026": round(is_, 4), "ic_delta_2026_vs_train": round(is_ - it, 4)})
    ict = pd.DataFrame(ic_rows)
    print(ict.sort_values("ic_delta_2026_vs_train").to_string(index=False))
    n_flip = int(((ict["ic_train"].abs() > 0.01) & (ict["ic_2026"] * ict["ic_train"] < 0)).sum())
    n_decay = int(((ict["ic_train"].abs() > 0.01) & (ict["ic_2026"].abs() < 0.5 * ict["ic_train"].abs())).sum())
    print(f"训练期有效(|IC|>0.01)且 2026 反转符号的特征: {n_flip} 个 | 衰减到不足一半: {n_decay} 个")

    out = {
        "generated_at": pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"),
        "train_window": "<=2024-12-31", "test_window": "2026-01-01~max",
        "n_features": int(len(df)), "n_significant_psi0_1": n_sig, "n_severe_psi0_25": n_sev,
        "psi_table": rows,
        "ic_table": ic_rows,
        "n_ic_flip_2026": n_flip, "n_ic_decay_half": n_decay,
    }
    op = zz.REPORT_DIR / "gru_drift_check_20261101.json"
    op.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
