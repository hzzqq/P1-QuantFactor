"""LightGBM 基线模型。

为什么必须先建树模型基线（这一步绝不能省）：
    在量化领域，梯度提升树通常极强，很多"深度学习选股"的论文都败给了
    没调参的 LightGBM。如果本项目里的神经网络打不过基线，
    说明时序建模没带来增量信息 —— 这个结论本身就是极有价值的发现。
    跳过基线直接上神经网络，是自欺欺人。

    因此本模块是全项目的"裁判"，后续所有神经网络模型都要和它比。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.logging_utils import get_logger

logger = get_logger("P1.models.baseline_lgb")

DEFAULT_PARAMS: dict = {
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 200,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "num_threads": 8,
    "seed": 42,
    "deterministic": True,
}


def _ic_eval(y_pred, data):
    """LightGBM 自定义评估：验证集 IC（Spearman 秩相关）。

    模型真实价值是**排序能力（IC）**，而非 MSE。旧范式用单年 MSE 早停，
    而标签 IC 仅 0.02–0.05（噪声远大于信号），MSE 早停会把 best_iteration
    随机化到 2~45（命中 2 即严重欠拟合）。改为用 IC 早停，方向与价值一致。

    LightGBM feval 签名：(preds, data)，data 为 Dataset，标签用 data.get_label()。
    """
    yt = np.asarray(data.get_label(), dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    try:
        from scipy.stats import spearmanr
        ic = spearmanr(yt, yp).correlation
    except Exception:
        rt = np.argsort(np.argsort(yt))
        rp = np.argsort(np.argsort(yp))
        ic = float(np.corrcoef(rt, rp)[0, 1])
    if ic is None or np.isnan(ic):
        ic = 0.0
    return "ic", float(ic), True


def train(X_tr: pd.DataFrame, y_tr: pd.Series,
          X_va: pd.DataFrame | None = None, y_va: pd.Series | None = None,
          params: dict | None = None, num_boost_round: int = 400,
          early_stopping: int = 50, eval_metric: str = "ic") -> object:
    """训练 LightGBM 回归模型（预测未来超额收益）。

    eval_metric:
        "ic"  -> 用验证集 IC（Spearman）早停（默认，方向对齐真实价值）。
        "mse" -> 旧范式，验证集 MSE 早停（保留以便 A/B 对照）。
    """
    import lightgbm as lgb

    params = {**DEFAULT_PARAMS, **(params or {})}
    feval = None
    if eval_metric == "ic":
        # 关掉内置 mse 指标，避免与自定义 IC 评估冲突
        params = {**params, "metric": "None"}
        feval = _ic_eval

    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)

    valid_sets = [dtrain]
    if X_va is not None and len(X_va) > 0:
        valid_sets.append(lgb.Dataset(X_va, label=y_va, reference=dtrain,
                                      free_raw_data=True))

    callbacks = [lgb.log_evaluation(period=100)]
    if len(valid_sets) > 1:
        # first_metric_only：多指标时只对首个（IC）做早停判定
        callbacks.append(lgb.early_stopping(early_stopping, verbose=False,
                                            first_metric_only=True))

    logger.info("训练 LightGBM：训练样本 %s，特征 %s，早停指标=%s",
                f"{len(X_tr):,}", X_tr.shape[1], eval_metric)
    model = lgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        valid_sets=valid_sets, feval=feval, callbacks=callbacks,
    )
    return model


def split_by_date(data: pd.DataFrame, train_end: str, valid_end: str,
                  date_col: str = "date"):
    """按日期切分训练/验证/测试。"""
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
    """滚动训练（walk-forward），严格不用未来数据。

    对每个测试年份 Y：
        训练集 = [Y-train_years-valid_years, Y-valid_years-1]
        验证集 = [Y-valid_years, Y-1]   # 默认 2 年，降低单年 IC 噪声
        测试集 = Y

    valid_years=1 时退化为旧范式（验证集仅 Y-1）。
    eval_metric="ic" 用 IC 早停（默认，修正 best_iteration 2~45 乱跳）；
    eval_metric="mse" 为旧范式（对照用）。

    Returns:
        (合并后的预测 DataFrame, 逐年指标 list)
    """
    import lightgbm as lgb

    # 不 copy 整个数据集（省内存）：直接在传入的 data 上加 _year 列。
    # 调用方传入的 data 本就是本次 walk_forward 专用，原地加列无副作用。
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
            logger.warning("跳过 %s：样本不足 (train=%s, test=%s)",
                           y, len(tr), len(te))
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
            "pred": model.predict(X_te, num_iteration=model.best_iteration),
            y_col: te[y_col].values,
            "year": y,
        })
        all_preds.append(pred)

        imp = pd.Series(model.feature_importance("gain"), index=factor_names)
        top_feats = imp.sort_values(ascending=False).head(5)

        logger.info(
            "%s 完成：样本 %s | 最优迭代 %s | Top5 因子 %s",
            y, f"{len(te):,}", model.best_iteration,
            ", ".join(f"{k}({v:.0f})" for k, v in top_feats.items()),
        )
        yearly.append({
            "year": y, "n_train": len(tr), "n_test": len(te),
            "best_iteration": model.best_iteration,
            "eval_metric": eval_metric, "valid_years": valid_years,
            "top_features": {k: float(v) for k, v in top_feats.items()},
        })

    if not all_preds:
        return pd.DataFrame(), yearly
    return pd.concat(all_preds, ignore_index=True), yearly
