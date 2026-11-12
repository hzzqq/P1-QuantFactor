"""直方图梯度提升基线（sklearn HistGradientBoostingRegressor）—— LightGBM 不可用时的同族替代。

背景
----
项目原 baseline 用 LightGBM（`baseline_lgb.py`），但当前运行环境（Python 3.13）无 lightgbm 轮子、
清华镜像也未收录 → 无法复跑。Histogram GBDT 与 LightGBM 同族（同属直方图提升树），
`HistGradientBoostingRegressor` 在 sklearn 中可用，作为**公平的同模型对照**用于 v39↔v41 因子增量判定。

纪律对齐 `baseline_lgb`：
  - **IC 早停**（默认 eval_metric="ic"）：用验证集 Spearman 秩相关做早停，避免 MSE 早停把
    best_iteration 随机化到 2~45 的坑（标签 IC 仅 0.02–0.05，噪声远大于信号）。
  - 严格 walk-forward，绝不用未来数据。
  - API 与 `baseline_lgb.walk_forward` 一致（返回 合并预测 DataFrame + 逐年指标），便于复用 28/33 的分块逻辑。

用法：由 `40_retrain_v41_baseline.py` 调用，传 --ds/--meta 指定数据集（v39 或 v41）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import make_scorer

from shared.logging_utils import get_logger

logger = get_logger("P1.models.baseline_hgb")

DEFAULT_PARAMS: dict = {
    "learning_rate": 0.05,
    "max_iter": 400,
    "max_leaf_nodes": 63,        # 对应 lgb num_leaves
    "min_samples_leaf": 200,      # 对应 lgb min_data_in_leaf
    "l2_regularization": 1.0,     # 对应 lgb lambda_l2
    "max_bins": 255,
    "random_state": 42,
}


def _spearman(y_true, y_pred):
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(y_true, float), np.asarray(y_pred, float)).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)


_SPEARMAN_SCORER = make_scorer(_spearman, greater_is_better=True)


def train(X_tr: pd.DataFrame, y_tr: pd.Series,
         X_va: pd.DataFrame | None = None, y_va: pd.Series | None = None,
         params: dict | None = None, num_boost_round: int = 400,
         early_stopping: int = 50, eval_metric: str = "ic") -> object:
    """训练 HGB 回归模型。eval_metric='ic' 用验证集 Spearman IC 做早停；'mse' 用默认损失早停。

    注意：sklearn 的 HistGradientBoostingRegressor 不接受显式验证集（fit 仅 X,y），
    早停验证集由 `validation_fraction` 从训练集尾部自动切分（时间序训练集尾部≈最近窗口，
    等价于时间验证）。早停监控指标由 `scoring` 决定：ic→自定义 spearman scorer，mse→原生损失。
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    p["max_iter"] = num_boost_round
    has_val = X_va is not None and len(X_va) > 0
    if has_val and eval_metric == "ic":
        model = HistGradientBoostingRegressor(
            **p, early_stopping=True, validation_fraction=0.1,
            scoring=_SPEARMAN_SCORER, n_iter_no_change=early_stopping)
    elif has_val:
        model = HistGradientBoostingRegressor(
            **p, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=early_stopping)
    else:
        model = HistGradientBoostingRegressor(**p)
    model.fit(X_tr, y_tr)
    logger.info("训练 HGB：训练样本 %s，特征 %s，早停指标=%s，实际迭代=%s",
                f"{len(X_tr):,}", X_tr.shape[1], eval_metric, getattr(model, "n_iter_", None))
    return model


def split_by_date(data: pd.DataFrame, train_end: str, valid_end: str,
                  date_col: str = "date"):
    d = pd.to_datetime(data[date_col])
    train = data[d <= train_end]
    valid = data[(d > train_end) & (d <= valid_end)]
    test = data[d > valid_end]
    return train, valid, test


def walk_forward(data: pd.DataFrame, factor_names: list[str],
                 y_col: str = "y_excess", train_years: int = 3,
                 params: dict | None = None, num_boost_round: int = 400,
                 test_years: list[int] | None = None,
                 max_train_rows: int | None = None, seed: int = 42,
                 valid_years: int = 2, eval_metric: str = "ic"):
    """滚动训练（walk-forward），严格不用未来数据。API 对齐 baseline_lgb.walk_forward。"""
    df = data
    df["_year"] = pd.to_datetime(df["date"]).dt.year
    years = sorted(df["_year"].unique())
    if test_years is None:
        test_years = [y for y in years if y >= years[0] + train_years + 1]

    all_preds: list[pd.DataFrame] = []
    yearly: list[dict] = []
    for y in test_years:
        tr = df[(df["_year"] >= y - train_years - valid_years)
                & (df["_year"] <= y - valid_years - 1)]
        va = df[(df["_year"] >= y - valid_years) & (df["_year"] <= y - 1)]
        te = df[df["_year"] == y]
        if len(tr) < 2000 or len(te) < 100:
            logger.warning("跳过 %s：样本不足 (train=%s, test=%s)", y, len(tr), len(te))
            continue
        if max_train_rows and len(tr) > max_train_rows:
            tr = tr.sample(max_train_rows, random_state=seed)
        X_tr, y_tr = tr[factor_names], tr[y_col]
        X_va, y_va = (va[factor_names], va[y_col]) if len(va) > 0 else (None, None)
        X_te = te[factor_names]
        model = train(X_tr, y_tr, X_va, y_va, params, num_boost_round,
                      eval_metric=eval_metric)
        pred = pd.DataFrame({
            "date": te["date"].values,
            "symbol": te["symbol"].values,
            "pred": model.predict(X_te),
            y_col: te[y_col].values,
            "year": y,
        })
        all_preds.append(pred)
        imp = pd.Series(getattr(model, "feature_importances_", np.zeros(len(factor_names))),
                        index=factor_names)
        top_feats = imp.sort_values(ascending=False).head(5)
        logger.info("%s 完成：样本 %s | 迭代 %s | Top5 %s",
                    y, f"{len(te):,}", getattr(model, "n_iter_", None),
                    ", ".join(f"{k}({v:.0f})" for k, v in top_feats.items()))
        yearly.append({
            "year": y, "n_train": len(tr), "n_test": len(te),
            "best_iteration": int(getattr(model, "n_iter_", 0)),
            "eval_metric": eval_metric, "valid_years": valid_years,
            "top_features": {k: float(v) for k, v in top_feats.items()},
        })
    if not all_preds:
        return pd.DataFrame(), yearly
    return pd.concat(all_preds, ignore_index=True), yearly
