"""阶段 3x：强化学习仓位决策策略（开题清单 #3）。

与 #1/#2 同口径：walk-forward 2022-2026，预测 horizon=10 日超额收益。
本脚本**不直接训预测模型**，而是在已有信号 pred_gru_h10.parquet 之上，
学一个「仓位决策策略」π(state)→position∈[-1,1]：
    state = [信号z, mom_z, vol_z, pos_z, bias_z]（决策日截面 z-score）
    奖励  r = a·y_excess − cost·|Δa|（Δa=相邻决策换手）
用 REINFORCE（高斯策略 + 批均值基线）训练，与静态基线对比：
    A) sign(pred)       —— 满仓多空
    B) tanh(2·pred_z)   —— 固定平滑缩放（"不用 RL，只缩放信号"）
指标：每决策日做多 top20%w / 做空 bottom20%w，算组合 Sharpe/总收益/胜率/最大回撤；
并报告平均换手 |Δa|（RL 应更低，因其奖励含换手惩罚项）。

诚实口径：RL 策略价值在于「用 vol/mom/pos/bias 等状态对信号做非线性、风险感知的
仓位调制」，而非凭空造 alpha。若 RL 未稳定优于 tanh 基线，须如实表述为
「边缘/持平」，不夸大。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                       # noqa: E402
import pandas as pd                       # noqa: E402
import torch                             # noqa: E402
import torch.nn as nn                    # noqa: E402

from shared import paths                  # noqa: E402
from src.models import rl_position as rlm  # noqa: E402

PROCESSED = paths.DATA / "P1" / "processed"
COST = 0.001    # 换手成本（每单位换手）
SIGMA = 0.10    # 探索噪声
FEATURES = ["mom_20", "vol_20", "pos_20", "bias_20"]


def load_data(pred_file: str) -> pd.DataFrame:
    pred = pd.read_parquet(PROCESSED / pred_file)
    ds = pd.read_parquet(
        paths.DATA / "P1" / "processed" / "dataset_h10.parquet",
        columns=["date", "symbol"] + FEATURES)
    df = pred.merge(ds, on=["date", "symbol"], how="inner")
    return df


def make_state(df: pd.DataFrame):
    """决策日截面 z-score，并把 z 列写回 df，返回 (df, 状态矩阵, 列名)。"""
    df = df.copy()
    df["pred_z"] = df.groupby("date")["pred"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-9))
    for f in FEATURES:
        df[f + "_z"] = df.groupby("date")[f].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-9))
    cols = ["pred_z"] + [f + "_z" for f in FEATURES]
    S = df[cols].to_numpy(dtype=np.float32)
    S = np.nan_to_num(S, 0.0)
    return df, S, cols


def prev_action(a: np.ndarray, symbol: np.ndarray, date: np.ndarray) -> np.ndarray:
    """按 (symbol,date) 排序后，同 symbol 内 shift，得到上一决策的 action。"""
    order = np.lexsort((date, symbol))      # 主排序 symbol，次 date
    a_sorted = a[order]
    sym_sorted = symbol[order]
    tmp = pd.DataFrame({"sym": sym_sorted, "a": a_sorted})
    tmp["prev"] = tmp.groupby("sym")["a"].shift(1)
    prev = np.full_like(a_sorted, 0.0)
    prev[order] = tmp["prev"].to_numpy()
    return np.nan_to_num(prev, 0.0)


def train_policy(S, symbol_arr, date_arr, y, epochs=15, lr=1e-3,
                 batch=4096, sigma=0.05):
    """Actor-Critic（REINFORCE + 价值基线）：降低策略梯度方差。

    - 价值网络 ValueNet 回归到「确定性动作 μ 的奖励」r_target=μ·y−cost·|μ−μ_prev|，
      作为 low-variance 基线 b(s)；
    - 策略每步 advantage = r − b(s)，r = a·y − cost·|a−a_prev|（a 为采样动作）；
    - a_prev 用确定性 μ 在同 (symbol,date) 序列上 shift，每轮更新（向量化、无泄漏）。
    相比批均值 EMA 基线，价值基线显著降低方差，使策略更易收敛到有效的
    「信号→仓位」映射，而非被噪声奖励带偏。
    """
    model = rlm.PolicyNet(state_dim=S.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    value_net = rlm.ValueNet(state_dim=S.shape[1])
    vopt = torch.optim.Adam(value_net.parameters(), lr=lr)
    X = torch.tensor(S)
    Y = torch.tensor(y, dtype=torch.float32)
    n = len(S)
    for ep in range(epochs):
        with torch.no_grad():
            mu_det = model(X).numpy()
        a_prev_det = prev_action(mu_det, symbol_arr, date_arr)
        Aprev = torch.tensor(a_prev_det, dtype=torch.float32)
        r_target = (torch.tensor(mu_det, dtype=torch.float32) * Y
                    - COST * torch.abs(torch.tensor(mu_det, dtype=torch.float32)
                                       - torch.tensor(a_prev_det, dtype=torch.float32)))
        # 1) 训练价值基线（回归到确定性奖励）
        for _ in range(2):
            perm = torch.randperm(n)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                vopt.zero_grad()
                loss_v = ((value_net(X[idx]) - r_target[idx]) ** 2).mean()
                loss_v.backward()
                vopt.step()
        # 2) 策略更新：advantage = r − b(s)
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            s = X[idx]
            yy = Y[idx]
            ap = Aprev[idx]
            mu = model(s)
            eps = torch.randn_like(mu)
            a = torch.clamp(mu + sigma * eps, -1, 1)
            logp = -0.5 * ((a - mu) / sigma) ** 2
            b = value_net(s).detach()
            r = a * yy - COST * torch.abs(a - ap)
            adv = (r - b)
            loss = -(logp * adv).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def portfolio_metrics(df: pd.DataFrame, w_col: str):
    """每决策日：做多 w 最高 20%、做空最低 20%，算组合指标。"""
    df = df.copy()
    df["grp"] = df.groupby("date")[w_col].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    lon = df[df["grp"] == 4].groupby("date")["y_excess"].mean()
    sho = df[df["grp"] == 0].groupby("date")["y_excess"].mean()
    ret = (lon - sho).dropna()
    if len(ret) < 5:
        return None
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252 / 10)) if ret.std() > 0 else float("nan")
    total = float((1 + ret).prod() - 1)
    win = float((ret > 0).mean())
    cum = np.cumprod(1 + ret.to_numpy())
    dd = 1 - cum / np.maximum.accumulate(cum)
    mdd = float(dd.max())
    return {"sharpe": sharpe, "total_return": total, "win_rate": win,
            "max_dd": mdd, "n_days": int(len(ret))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="pred_gru_h10.parquet")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--out", default="pred_rl_position_h10.parquet")
    args = ap.parse_args()

    if not (PROCESSED / args.pred).exists():
        print("找不到信号文件:", PROCESSED / args.pred)
        return 1

    t0 = time.time()
    df = load_data(args.pred)
    if args.max_symbols:
        keep = df["symbol"].value_counts().index[:args.max_symbols]
        df = df[df["symbol"].isin(keep)]
    df, S, _ = make_state(df)
    years = sorted(df["year"].dropna().unique().tolist())
    print("年份:", years, "样本:", len(df))

    all_rows, yearly = [], []
    for y in years:
        test_mask = (df["year"] == y).to_numpy()
        train_mask = (df["year"] < y).to_numpy()
        if int(train_mask.sum()) < 1000 or int(test_mask.sum()) < 50:
            print(f"跳过 {y}: 训练/测试样本不足")
            continue
        St = S[train_mask]
        yt = df.loc[train_mask, "y_excess"].to_numpy(dtype=float)
        sym_t = df.loc[train_mask, "symbol"].to_numpy()
        date_t = df.loc[train_mask, "date"].to_numpy()
        model = train_policy(St, sym_t, date_t, yt, epochs=args.epochs)

        with torch.no_grad():
            a_test = model(torch.tensor(S[test_mask])).numpy()
        sub = df[test_mask].copy()
        sub["w_rl"] = np.clip(a_test, -1, 1)
        sub["w_sign"] = np.sign(sub["pred"].to_numpy())
        sub["w_tanh"] = np.tanh(2 * sub["pred_z"].to_numpy())
        pa = prev_action(sub["w_rl"].to_numpy(), sub["symbol"].to_numpy(),
                        sub["date"].to_numpy())
        turnover = float(np.abs(sub["w_rl"].to_numpy() - pa).mean())

        row = {"year": int(y)}
        for wcol in ["w_rl", "w_sign", "w_tanh"]:
            m = portfolio_metrics(sub, wcol)
            if m:
                row[wcol] = m
                yearly.append({"year": int(y), "method": wcol, **m})
        row["turnover_rl"] = turnover
        print(f"  {y}: RL sharpe={row.get('w_rl',{}).get('sharpe'):.3f} "
              f"sign={row.get('w_sign',{}).get('sharpe'):.3f} "
              f"tanh={row.get('w_tanh',{}).get('sharpe'):.3f} turnover_rl={turnover:.3f}")
        out_cols = ["date", "symbol", "pred", "w_rl", "w_sign",
                    "w_tanh", "y_excess", "year"]
        out_sub = sub[out_cols].copy()
        out_sub["turnover_rl"] = turnover
        all_rows.append(out_sub)

    if not all_rows:
        print("无产出")
        return 1
    preds = pd.concat(all_rows, ignore_index=True)
    preds.to_parquet(PROCESSED / args.out, index=False)

    overall = {}
    for wcol in ["w_rl", "w_sign", "w_tanh"]:
        overall[wcol] = portfolio_metrics(preds, wcol)
    report = {"by_year": yearly, "overall": overall,
              "avg_turnover_rl": float(preds["turnover_rl"].mean()),
              "elapsed_sec": round(time.time() - t0, 1)}
    rp = PROCESSED / "report_rl_position.json"
    with rp.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n=== 总体（RL vs 静态基线）===")
    for wcol in ["w_rl", "w_sign", "w_tanh"]:
        m = overall[wcol]
        if m:
            print(f"  {wcol:<7} Sharpe={m['sharpe']:.3f} 总收益={m['total_return']:+.3f} "
                  f"胜率={m['win_rate']:.3f} 最大回撤={m['max_dd']:.3f}")
    print(f"  平均换手(RL)={report['avg_turnover_rl']:.3f}")
    print("报告:", rp, "耗时", report["elapsed_sec"], "秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
