"""T-265 A 路：RL 仓位策略 v2——乘性调制 + 基线锚定 advantage。

背景（38_rl_position.py 诚实结论）：自由动作 RL（PolicyNet 直接输出 position）
全期 Sharpe -0.741，完败于 tanh(2·pred_z) 基线（+1.145），平均换手仅 0.026
（策略塌缩到近恒定小幅仓位，丢失满仓暴露）。

v2 的三点结构性修正（针对失效机制，非超参补丁）：
  V1 乘性调制参数化：a = clip(b0·(1+κ·gθ(s)))，b0=tanh(2·pred_z) 为基线动作，
     gθ∈[-1,1] 由零初始化末层策略网络输出 → **训练起点严格等于基线**，
     策略只学「相对基线的增减仓调制」，满仓暴露永不丢失（保底设计）；
  V2 基线锚定 advantage：adv = [r(a) − r(b0)] − V(s)，r(b0) 为基线动作在同
     (symbol,date) shift 链下的奖励（预计算常数）——策略只有跑赢基线才获正梯度，
     从机制上消除「偏离基线即扣换手费却无收益」的坍缩激励；
  V3 状态扩展：state = [pred_z, mom_z, vol_z, pos_z, bias_z, |pred_z|]，
     新增信号自身置信度 |pred_z|（调制策略理应学「何时信任信号」）。

另附 **oracle 调制上限诊断**（诚实口径）：λ 网格 × vol_20_z 三分位桶，
λ* 由训练年选出、测试年应用——估计「若状态确有调制信息」的可学上限；
若 oracle 也仅与基线持平，则说明该信号上状态调制本身无信息，RL 无从赢起。

同口径：walk-forward 2022-2026（<38 同参）、COST=0.001、组合构建/指标函数
直接复用 38_rl_position.py（importlib 按路径加载，保证 head-to-head 公平）。
产出：pred_rl_mod_h10.parquet + report_rl_position_v2.json。
"""
from __future__ import annotations

import argparse
import importlib.util
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

# ── 复用 38 的数据/口径函数（同脚本链路，保证 head-to-head 公平）──
_spec = importlib.util.spec_from_file_location(
    "rl38", str(PROJ / "scripts" / "38_rl_position.py"))
rl38 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rl38)

PROCESSED = paths.DATA / "P1" / "processed"
COST = rl38.COST        # 0.001
KAPPA = 0.8             # 调制幅度上限：a = clip(b0·(1±0.8))
ORACLE_LAMBDAS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def build_states(df: pd.DataFrame):
    """6 维状态：38 的 5 维 + |pred_z|（信号置信度）。返回 (df, S, cols)。"""
    df, S5, cols5 = rl38.make_state(df)
    df = df.copy()
    df["pred_abs_z"] = df["pred_z"].abs()
    S = np.concatenate([S5, df[["pred_abs_z"]].to_numpy(dtype=np.float32)], axis=1)
    S = np.nan_to_num(S, 0.0)
    return df, S, cols5 + ["pred_abs_z"]


def baseline_reward(b0: np.ndarray, y: np.ndarray, symbol: np.ndarray, date: np.ndarray) -> np.ndarray:
    """基线动作奖励 r(b0)=b0·y − COST·|Δb0|（同 (symbol,date) shift 链，无泄漏）。"""
    b0_prev = rl38.prev_action(b0, symbol, date)
    return b0 * y - COST * np.abs(b0 - b0_prev)


def train_mod_policy(S, b0, r_base, symbol_arr, date_arr, y,
                     epochs=40, lr=1e-3, batch=4096, sigma=0.05, kappa=KAPPA):
    """残差 Actor-Critic：策略只学调制 g，advantage 锚定基线奖励。"""
    policy = rlm.PolicyNet(state_dim=S.shape[1])
    value = rlm.ValueNet(state_dim=S.shape[1])
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    vopt = torch.optim.Adam(value.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    X = torch.tensor(S)
    B0 = torch.tensor(b0, dtype=torch.float32)
    RB = torch.tensor(r_base, dtype=torch.float32)
    Y = torch.tensor(y, dtype=torch.float32)
    n = len(S)
    for ep in range(epochs):
        with torch.no_grad():
            g_det = policy(X).numpy()
        a_det = np.clip(b0 * (1.0 + kappa * g_det), -1.0, 1.0)
        a_prev = rl38.prev_action(a_det, symbol_arr, date_arr)
        r_det = a_det * y - COST * np.abs(a_det - a_prev) - r_base   # 残差奖励
        Rt = torch.tensor(r_det, dtype=torch.float32)
        AP = torch.tensor(a_prev, dtype=torch.float32)
        # 1) 价值基线回归残差奖励
        for _ in range(2):
            perm = torch.randperm(n)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                vopt.zero_grad()
                loss_v = ((value(X[idx]) - Rt[idx]) ** 2).mean()
                loss_v.backward()
                vopt.step()
        # 2) 策略更新：adv = 残差奖励(采样) − 残差奖励(确定性,μ_det) − V(s)
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            s, b0b, yb = X[idx], B0[idx], Y[idx]
            ap = AP[idx]
            r_det_b = Rt[idx].detach()
            mu = policy(s)                                    # g ∈ [-1,1]
            eps = torch.randn_like(mu)
            g = torch.clamp(mu + sigma * eps, -1, 1)
            a = torch.clamp(b0b * (1.0 + kappa * g), -1, 1)
            logp = -0.5 * ((g - mu) / sigma) ** 2
            r_sample = a * yb - COST * torch.abs(a - ap) - RB[idx]
            adv = (r_sample - r_det_b) - value(s).detach()
            loss = -(logp * adv).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return policy


def apply_mod(policy, S, b0, kappa=KAPPA):
    with torch.no_grad():
        g = policy(torch.tensor(S)).numpy()
    return np.clip(b0 * (1.0 + kappa * g), -1.0, 1.0), g


def oracle_weights(vol_z_tr: np.ndarray, vol_z_te: np.ndarray,
                   b0_tr: np.ndarray, y_tr: np.ndarray, sym_tr: np.ndarray,
                   date_tr: np.ndarray, b0_te: np.ndarray) -> np.ndarray:
    """严格无泄漏 oracle 上限：vol_20_z 三分位桶（边界由训练年分位定），
    每桶 λ* 在训练年按决策日平均奖励选出，测试年按同桶应用。
    返回测试期 w_oracle。这是「若状态确有调制信息、且允许直接选参」的可学上限。"""
    tr_ok = np.isfinite(vol_z_tr)
    if tr_ok.sum() < 100:
        return b0_te.copy()
    q1, q2 = np.quantile(vol_z_tr[tr_ok], [1 / 3, 2 / 3])
    b_tr = np.digitize(np.nan_to_num(vol_z_tr), [q1, q2])
    b_te = np.digitize(np.nan_to_num(vol_z_te), [q1, q2])
    w = b0_te.copy()
    for b in (0, 1, 2):
        m_b = (b_tr == b) & tr_ok
        if m_b.sum() < 50:
            continue
        best_r, best_l = -np.inf, 1.0
        for lam in ORACLE_LAMBDAS:
            wl = np.clip(b0_tr * lam, -1.0, 1.0)
            wl_prev = rl38.prev_action(wl, sym_tr, date_tr)
            r_lam = wl * y_tr - COST * np.abs(wl - wl_prev)
            score = float(np.nanmean(r_lam[m_b]))
            if score > best_r:
                best_r, best_l = score, lam
        w[b_te == b] = np.clip(b0_te[b_te == b] * best_l, -1.0, 1.0)
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="pred_gru_h10.parquet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--kappa", type=float, default=KAPPA)
    ap.add_argument("--out", default="pred_rl_mod_h10.parquet")
    args = ap.parse_args()

    if not (PROCESSED / args.pred).exists():
        print("找不到信号文件:", PROCESSED / args.pred)
        return 1

    t0 = time.time()
    df = rl38.load_data(args.pred)
    if args.max_symbols:
        keep = df["symbol"].value_counts().index[:args.max_symbols]
        df = df[df["symbol"].isin(keep)].reset_index(drop=True)
    df, S, cols = build_states(df)
    df = df.reset_index(drop=True)
    years = sorted(df["year"].dropna().unique().tolist())
    print("年份:", years, "样本:", len(df), "状态:", cols)

    all_rows, yearly = [], []
    for y in years:
        test_mask = (df["year"] == y).to_numpy()
        train_mask = (df["year"] < y).to_numpy()
        if int(train_mask.sum()) < 1000 or int(test_mask.sum()) < 50:
            print(f"跳过 {y}: 训练/测试样本不足")
            continue
        sym_t = df.loc[train_mask, "symbol"].to_numpy()
        date_t = df.loc[train_mask, "date"].to_numpy()
        y_tr = df.loc[train_mask, "y_excess"].to_numpy(dtype=float)
        b0_tr = np.tanh(2.0 * df.loc[train_mask, "pred_z"].to_numpy(dtype=float))
        r_base_tr = baseline_reward(b0_tr, y_tr, sym_t, date_t)

        policy = train_mod_policy(S[train_mask], b0_tr, r_base_tr, sym_t, date_t,
                                  y_tr, epochs=args.epochs, kappa=args.kappa)
        b0_te = np.tanh(2.0 * df.loc[test_mask, "pred_z"].to_numpy(dtype=float))
        w_mod, g_mean = apply_mod(policy, S[test_mask], b0_te, kappa=args.kappa)
        vol_z_tr = df.loc[train_mask, "vol_20_z"].to_numpy(dtype=float)
        vol_z_te = df.loc[test_mask, "vol_20_z"].to_numpy(dtype=float)
        y_te = df.loc[test_mask, "y_excess"].to_numpy(dtype=float)
        w_oracle = oracle_weights(vol_z_tr, vol_z_te, b0_tr, y_tr, sym_t, date_t, b0_te)
        sub = df[test_mask].copy()
        sub["w_mod"] = w_mod
        sub["w_oracle"] = w_oracle
        sub["w_sign"] = np.sign(sub["pred"].to_numpy())
        sub["w_tanh"] = np.tanh(2 * sub["pred_z"].to_numpy())
        pa = rl38.prev_action(w_mod, sub["symbol"].to_numpy(), sub["date"].to_numpy())
        turnover = float(np.abs(w_mod - pa).mean())

        row = {"year": int(y), "g_mean_abs": round(float(np.abs(g_mean).mean()), 4)}
        for wcol in ["w_mod", "w_oracle", "w_sign", "w_tanh"]:
            m = rl38.portfolio_metrics(sub, wcol)
            if m:
                row[wcol] = m
                yearly.append({"year": int(y), "method": wcol, **m})
        row["turnover_mod"] = turnover
        print(f"  {y}: mod={row.get('w_mod', {}).get('sharpe', float('nan')):.3f} "
              f"oracle={row.get('w_oracle', {}).get('sharpe', float('nan')):.3f} "
              f"tanh={row.get('w_tanh', {}).get('sharpe', float('nan')):.3f} "
              f"|g|={row['g_mean_abs']:.3f} turnover={turnover:.3f}", flush=True)
        out_cols = ["date", "symbol", "pred", "w_mod", "w_oracle", "w_sign",
                    "w_tanh", "y_excess", "year"]
        all_rows.append(sub[out_cols].copy())
        all_rows[-1]["turnover_mod"] = turnover

    if not all_rows:
        print("无产出")
        return 1
    preds = pd.concat(all_rows, ignore_index=True)
    preds.to_parquet(PROCESSED / args.out, index=False)

    overall = {}
    for wcol in ["w_mod", "w_oracle", "w_sign", "w_tanh"]:
        overall[wcol] = rl38.portfolio_metrics(preds, wcol)
    report = {
        "design": "v2 multiplicative modulation: a=clip(b0*(1+kappa*g(s))), "
                  "adv anchored to baseline reward b0=tanh(2*pred_z), state+|pred_z|",
        "kappa": args.kappa, "epochs": args.epochs,
        "oracle_note": "oracle=可学调制上限（vol 三分位桶 × λ 网格，λ* 由训练年选出、测试年应用，严格无泄漏）",
        "by_year": yearly, "overall": overall,
        "avg_turnover_mod": float(preds["turnover_mod"].mean()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    rp = PROCESSED / "report_rl_position_v2.json"
    with rp.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n=== 总体（v2 调制 vs oracle vs 静态基线）===")
    for wcol in ["w_mod", "w_oracle", "w_sign", "w_tanh"]:
        m = overall[wcol]
        if m:
            print(f"  {wcol:<9} Sharpe={m['sharpe']:+.3f} 总收益={m['total_return']:+.3f} "
                  f"胜率={m['win_rate']:.3f} 回撤={m['max_dd']:.3f}")
    print(f"  平均换手(mod)={report['avg_turnover_mod']:.3f}")
    print("报告:", rp, "耗时", report["elapsed_sec"], "秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
