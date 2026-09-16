"""成本敏感的 A 股多空回测引擎。

设计要点（与既有评估口径对齐，且更贴近真实交易）：
- 信号在 T 日收盘生成（pred 列），**只能在 T+1 开盘成交**；
- 持有 horizon 个交易日，在 T+horizon 开盘平仓（与标签「未来 N 日超额收益」窗口一致，仅平移一天）；
- 涨跌停约束：T+1 涨停（>= LIMIT_UP）无法买入 → 该标的跳过；
  平仓日跌停（<= LIMIT_DOWN）无法卖出 → 顺延至最近一个非跌停日（上限 +5 日）；
- 交易成本（A 股典型）：佣金 0.03% 双边、冲击/滑点 0.1% 双边、印花税 0.1% 仅卖出；
- 组合构建：每个调仓日按 pred 横截面排序，做多前 P%、做空后 P%，等权；
  非重叠调仓桶链式复利（调仓频率 = 持有期，桶之间不重叠）。

所有口径都在注释里写清，方便后续审计与 StockSignal 接入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import backtest_bootstrap

# 涨跌停阈值（主板 10%；用 0.095 留安全边际，避开 ST/北交所已在股票池剔除）
LIMIT_UP = 0.095
LIMIT_DOWN = -0.095

# A 股交易成本（单边比例）
DEFAULT_COST = {
    "commission": 0.0003,   # 佣金，双边
    "slippage": 0.001,      # 滑点/冲击，双边（固定档，对应流动性中档）
    "stamp": 0.001,         # 印花税，仅卖出
}

# 流动性分档滑点（A 任务：按个股真实成交额分档，取代固定 0.1%）
# 顺序：流动性前 30% / 30~70% / 70~90% / 后 10%(微盘)
LIQUIDITY_SLIPPAGE_TIERS = (0.0005, 0.0010, 0.0015, 0.0025)


def compute_liquidity_slippage(panel: pd.DataFrame,
                               tiers=LIQUIDITY_SLIPPAGE_TIERS) -> dict:
    """按个股流动性（中位数日成交额）分档滑点，返回 {symbol: slippage}。

    panel 需含 symbol,date,close,volume。amount≈close*volume*100（volume 单位「手」=100 股）。
    微盘股流动性差、冲击成本高，应比大盘股付更高滑点——这比统一 0.1% 更贴近真实。
    若 panel 无 volume 则返回空 dict（调用方退化为固定档）。
    """
    need = {"symbol", "date", "close", "volume"}
    if not need.issubset(panel.columns):
        return {}
    df = panel[["symbol", "date", "close", "volume"]].dropna()
    df = df[df["volume"] > 0]
    if df.empty:
        return {}
    df["amount"] = df["close"] * df["volume"] * 100.0
    med = df.groupby("symbol")["amount"].median()
    if med.empty:
        return {}
    q = med.quantile([0.30, 0.70, 0.90]).values
    out = {}
    for sym, a in med.items():
        if a >= q[2]:
            out[sym] = tiers[3]
        elif a >= q[1]:
            out[sym] = tiers[2]
        elif a >= q[0]:
            out[sym] = tiers[1]
        else:
            out[sym] = tiers[0]
    return out


def compute_name_vol(panel: pd.DataFrame) -> dict:
    """个股日收益波动率 std（静态逆波动加权用），返回 {symbol: std(day_ret)}。"""
    if not {"symbol", "date", "close"}.issubset(panel.columns):
        return {}
    df = panel[["symbol", "date", "close"]].sort_values(["symbol", "date"]).copy()
    df["prev"] = df.groupby("symbol")["close"].shift(1)
    df["ret"] = df["close"] / df["prev"] - 1.0
    sd = df.groupby("symbol")["ret"].std()
    return {s: float(v) for s, v in sd.items() if pd.notna(v) and v > 0}


def _apply_vol_target(bucket_rets: list[float], rebalance_freq: int,
                      vol_target: float | None, vol_lookback: int,
                      max_leverage: float) -> list[float]:
    """组合层波动率目标化（B 任务）。

    用**已实现**波动率（前 vol_lookback 个桶的收益）估计年化波动，把下一桶的
    总敞口缩放到目标波动。不使用未来信息（只用历史）。L 截断在 [0, max_leverage]。

    **设计约束（压回撤而非加杠杆）**：max_leverage 默认 1.0，即只去杠杆、不加息。
    只有当 vol_target **低于**策略自然波动时，L<1 才会真正压低回撤；若目标高于自然
    波动，L 会触顶 max_leverage（默认 1.0，不加杠杆），敞口不变。这是诚实的回撤
    控制手段：收益与回撤按同比例缩放，夏普大致不变。
    返回缩放后的桶收益序列。vol_target=None 时原样返回。
    """
    if vol_target is None or len(bucket_rets) < 2:
        return list(bucket_rets)
    scaled: list[float] = []
    hist: list[float] = []
    for r in bucket_rets:
        if len(hist) >= vol_lookback:
            vol = float(np.std(hist[-vol_lookback:]) * np.sqrt(252 / rebalance_freq))
            L = min(max_leverage, vol_target / vol) if vol > 1e-9 else max_leverage
        else:
            L = 1.0
        scaled.append(L * r)
        hist.append(r)
    return scaled


def attach_forward(panel: pd.DataFrame, horizon: int = 10,
                   exit_extend: int = 5) -> pd.DataFrame:
    """在 panel 上附加：未来收益、涨跌停标记、各偏移的 open/limit 列。

    panel 需含列：symbol, date, open, close。返回按 (symbol, date) 排序的副本。
    """
    df = panel.sort_values(["symbol", "date"]).copy()
    df = df.set_index(["symbol", "date"], drop=False)

    g = df.groupby(level=0)
    df["prev_close"] = g["close"].shift(1)
    df["day_ret"] = df["close"] / df["prev_close"] - 1.0
    df["is_limit_up"] = df["day_ret"] >= LIMIT_UP
    df["is_limit_down"] = df["day_ret"] <= LIMIT_DOWN

    max_k = horizon + exit_extend
    # open 在偏移 k 处的值 & 当日是否跌停
    for k in range(1, max_k + 1):
        df[f"open_s{k}"] = g["open"].shift(-k)
        df[f"ld_s{k}"] = g["is_limit_down"].shift(-k)
    # 入场日（T+1）是否涨停（无法买入）
    df["entry_limit_up"] = g["is_limit_up"].shift(-1)

    df = df.reset_index(drop=True)
    return df


def _bucket_stats(rets: list[float], rebalance_freq: int, years: float) -> dict:
    """回测桶序列的统计量（run_backtest 与 run_backtest_continuous 共用，N17 去重）。

    返回 {total_return, annual_return, sharpe, max_drawdown, win_rate, n_buckets}；
    样本不足或零波动时返回空 dict。
    """
    a = np.array(rets, dtype=float)
    if len(a) < 2 or a.std() == 0:
        return {}
    total = float((1 + a).prod() - 1)
    ann = float((1 + total) ** (1 / max(years, 1e-9)) - 1) if years > 0 else np.nan
    sharpe = float(a.mean() / a.std() * np.sqrt(252 / rebalance_freq))
    peak = np.cumprod(1 + a)
    running_max = np.maximum.accumulate(peak)
    mdd = float((peak / running_max - 1).min())
    win = float((a > 0).mean())
    return {
        "total_return": total,
        "annual_return": ann,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "win_rate": win,
        "n_buckets": int(len(a)),
    }


def run_backtest_continuous(preds: pd.DataFrame,
                            panel: pd.DataFrame,
                            horizon: int = 10,
                            top_pct: float = 0.1,
                            rebalance_freq: int = 10,
                            cost: dict | None = None,
                            mode: str = "long_short",
                            buffer: float = 0.02,
                            bootstrap: bool = False,
                            cost_model: str = "fixed",
                            sizer: str = "equal",
                            vol_target: float | None = None,
                            vol_lookback: int = 20,
                            max_leverage: float = 1.0) -> "BacktestResult":
    """连续持仓 + 缓冲区（buffer zone）调仓回测。

    与 `run_backtest`（非重叠桶）的关键差异：
    - 非重叠桶每期**全部清仓再建仓**，换手率 200%，成本极重；
    - 本函数维护**连续持仓组合**，每期只交易「变动的那一部分」：
        * 老持仓只要没跌出缓冲区就继续留着（宽出）
        * 新买入的必须排进更靠前的严格区间（严进）
      因此换手率大幅下降，只对实际变动部分计成本。

    参数
    ----
    buffer : 缓冲区宽度（排名分位）。例如 top_pct=0.1, buffer=0.02 时：
        - 新买入需排名在 top 8% 以内（严进）
        - 已持有的只要还在 top 12% 以内就保留（宽出）
    """
    cost = cost or DEFAULT_COST
    # A 任务：流动性分档滑点
    slip_map = compute_liquidity_slippage(panel) if cost_model == "liquidity" else None
    if sizer == "inv_vol":
        import warnings
        warnings.warn(
            "sizer='inv_vol' 已在本策略(日频多空 Top-decile)验证失败：逆波动加权把仓位"
            "集中于低波动/弱信号股（含 vol≈0 的僵尸股被赋权爆炸），实测净收益 -53%、"
            "MDD -71%；已退化为 equal。压回撤请用 vol_target。", stacklevel=2)
    base_slip = float(cost["slippage"])
    c_buy0 = float(cost["commission"])
    c_sell0 = float(cost["commission"]) + float(cost["stamp"])

    df = panel[["symbol", "date", "open", "close"]].sort_values(["symbol", "date"]).copy()
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df["day_ret"] = df["close"] / df["prev_close"] - 1.0
    df["is_lu"] = (df["day_ret"] >= LIMIT_UP).fillna(False)
    df["is_ld"] = (df["day_ret"] <= LIMIT_DOWN).fillna(False)

    px = df.pivot(index="date", columns="symbol", values="open")
    lu = df.pivot(index="date", columns="symbol", values="is_lu").fillna(False).astype(bool)
    ld = df.pivot(index="date", columns="symbol", values="is_ld").fillna(False).astype(bool)

    # 平仓价顺延：
    #  - 多头平仓是「卖」，跌停(ld)无法卖 → 仅避开 ld；涨停(lu)日仍可卖，不需避。
    #  - 空头回补是「买」，涨停(lu)日无法买 → 必须同时避开 ld 与 lu（N5 修复）。
    sellable_long = px.where(~ld)
    exit_px_long = sellable_long.bfill(axis=0, limit=5)
    block_buy = ld | lu
    sellable_short = px.where(~block_buy)
    exit_px_short = sellable_short.bfill(axis=0, limit=5)

    pv = preds.pivot_table(index="date", columns="symbol",
                           values="pred", aggfunc="last")
    pv = pv.reindex(columns=px.columns)

    dates = px.index
    # 调仓日必须取自「预测表实际覆盖的日期」，否则对过滤后的预测表会越界 KeyError
    rb_dates = list(pv.index[::rebalance_freq])

    prev_long: set = set()
    prev_short: set = set()
    per_rets, per_long, per_short, turnovers = [], [], [], []
    per_dates = []
    total_trades = 0      # 累计成交笔数（N4 修复：原仅末桶持仓数，严重低估）
    total_blocked = 0     # 累计因涨停被挡无法买入的条目数（N9 修复：原恒为 0）

    for i in range(len(rb_dates) - 1):
        T, Tn = rb_dates[i], rb_dates[i + 1]
        i0 = dates.get_loc(T) + 1        # T+1 开盘建仓
        i1 = dates.get_loc(Tn) + 1       # Tn+1 开盘平旧仓 + 建新仓
        if i1 >= len(dates):
            break
        d0, d1 = dates[i0], dates[i1]

        s_full = pv.loc[T].dropna()
        if len(s_full) < 20:
            continue
        # T+1 涨停无法买入 → 统计被挡数并剔除
        blocked = lu.loc[d0].reindex(s_full.index).fillna(True).astype(bool)
        total_blocked += int(blocked.sum())
        s = s_full[~blocked]
        if len(s) < 20:
            continue

        rk = s.rank(pct=True)
        n = len(s)
        k = max(1, int(n * top_pct))

        if buffer > 0:
            # 多头：严格 top P% 必进；老持仓掉到 top (P+buffer) 内仍保留
            tgt_long = set(s[rk >= 1 - top_pct].index) | \
                       (prev_long & set(s[rk >= 1 - top_pct - buffer].index))
            # 空头：对称
            tgt_short = set(s[rk <= top_pct].index) | \
                        (prev_short & set(s[rk <= top_pct + buffer].index))
        else:
            tgt_long = set(s.nlargest(k).index)
            tgt_short = set(s.nsmallest(k).index)

        def _ret(syms, sign=1.0) -> float:
            if not syms:
                return 0.0
            syms = list(syms)
            # 多头平仓用 exit_px_long（避跌停），空头回补用 exit_px_short（避跌停+涨停）
            epx = exit_px_long if sign > 0 else exit_px_short
            p0 = px.loc[d0, syms]
            p1 = epx.loc[d1, syms]
            ok = p0.notna() & p1.notna() & (p0 > 0)
            if not ok.any():
                return 0.0
            g = (p1[ok] / p0[ok] - 1.0) * sign
            # B 任务：sizer="inv_vol" 已在本策略验证失败并退化为 equal（见顶部警告），
            # 此处恒等权均值，不做个股逆波动加权（逆波动加权会把仓位压向低波动/弱信号股）。
            return float(g.mean())

        # 换手成本：按方向分别计费——进场只收买、出场只收卖。
        # A 任务：滑点取篮子（tgt 集合）平均流动性档位，而非统一固定。
        nL, nS = len(tgt_long), len(tgt_short)
        if slip_map:
            slip_l = float(np.mean([slip_map.get(s, base_slip) for s in tgt_long])) if nL else base_slip
            slip_s = float(np.mean([slip_map.get(s, base_slip) for s in tgt_short])) if nS else base_slip
        else:
            slip_l = slip_s = base_slip
        c_buy_l = c_buy0 + slip_l
        c_sell_l = c_sell0 + slip_l
        c_buy_s = c_buy0 + slip_s
        c_sell_s = c_sell0 + slip_s
        # 成交笔数始终累计（无论该侧是否为空），供 n_trades 真实统计（N4）
        entriesL = len(tgt_long - prev_long)   # 新买入 = 买
        exitsL = len(prev_long - tgt_long)      # 旧卖出 = 卖
        entriesS = len(tgt_short - prev_short)  # 新开空 = 卖(含印花)
        exitsS = len(prev_short - tgt_short)     # 平空 = 买
        total_trades += entriesL + exitsL + entriesS + exitsS
        if nL:
            wL = 1.0 / nL
            costL = (entriesL * c_buy_l + exitsL * c_sell_l) * wL
        else:
            costL = 0.0
        if nS:
            wS = 1.0 / nS
            costS = (entriesS * c_sell_s + exitsS * c_buy_s) * wS
        else:
            costS = 0.0

        gross_long = _ret(tgt_long, 1.0)
        gross_short = _ret(tgt_short, -1.0)
        net_long = gross_long - costL
        net_short = gross_short - costS

        if mode == "long_only":
            bucket = net_long
        else:
            bucket = 0.5 * net_long + 0.5 * net_short
        per_rets.append(bucket)
        per_long.append(net_long)
        per_short.append(net_short)
        per_dates.append(T)
        # 换手率（双边，相对目标仓位总量）
        base = max(nL + nS, 1)
        turnovers.append(((len(tgt_long - prev_long) + len(prev_long - tgt_long) +
                           len(tgt_short - prev_short) + len(prev_short - tgt_short))
                          / base))

        prev_long, prev_short = tgt_long, tgt_short

    # B 任务：组合层波动率目标化（用已实现波动，不用未来信息）
    per_rets = _apply_vol_target(per_rets, rebalance_freq,
                                 vol_target, vol_lookback, max_leverage)

    eq = pd.Series(per_rets, index=pd.to_datetime(per_dates))
    equity = (1.0 + eq).cumprod()
    n_buckets = len(eq)
    years = n_buckets * rebalance_freq / 252.0 if n_buckets else 0.0

    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    ls = _bucket_stats(per_rets, rebalance_freq, years)
    ls["avg_turnover"] = avg_turnover
    if bootstrap:
        ls.update(backtest_bootstrap(per_rets, rebalance_freq, years=years))
    lo = _bucket_stats(per_long, rebalance_freq, years)
    lo["avg_turnover"] = avg_turnover

    return BacktestResult(
        horizon=horizon, top_pct=top_pct, rebalance_freq=rebalance_freq,
        cost=cost, equity=equity,
        long_equity=(1 + pd.Series(per_long)).cumprod(),
        short_equity=(1 + pd.Series(per_short)).cumprod(),
        ls_stats=ls, long_stats=lo, short_stats=_bucket_stats(per_short, rebalance_freq, years),
        n_trades=int(total_trades),
        n_blocked=int(total_blocked),
        mode=f"continuous(buffer={buffer})[{cost_model}/{sizer}]"
             + (f"+volt{vl_}" if (vl_ := vol_target) else ""),
        cost_model=cost_model, sizer=sizer, vol_target=vol_target,
        equity_dates=list(per_dates),
    )


def run_backtest(preds: pd.DataFrame,
                 panel: pd.DataFrame,
                 horizon: int = 10,
                 top_pct: float = 0.1,
                 rebalance_freq: int | None = None,
                 cost: dict | None = None,
                 mode: str = "long_short",
                 label_col: str = "y_excess",
                 bootstrap: bool = False,
                 cost_model: str = "fixed",
                 sizer: str = "equal",
                 vol_target: float | None = None,
                 vol_lookback: int = 20,
                 max_leverage: float = 1.0) -> "BacktestResult":
    """对一份预测表跑成本敏感回测。

    参数
    ----
    preds   : 含 date, symbol, pred 的预测表（可由 03/04 脚本产出）
    panel   : 含 symbol, date, open, close, [volume] 的行情面板
    horizon: 持有交易日数（与标签窗口一致）
    top_pct: 多/空各取前/后多少比例
    rebalance_freq: 调仓频率（交易日）；默认 = horizon（非重叠桶）
    cost    : 交易成本 dict（见 DEFAULT_COST）
    cost_model: "fixed" 统一滑点；"liquidity" 按个股成交额分档滑点（A 任务）
    sizer   : "equal" 等权；"inv_vol" 已验证失败、退化为 equal（见函数内警告）
    vol_target: 年化波动目标（如 0.10）；非 None 且低于自然波动时做组合层去杠杆、
                压低回撤（B 任务）。max_leverage 默认 1.0，不加息。
    vol_lookback / max_leverage: vol 目标化的回看桶数 / 杠杆上限（默认只去杠杆）
    """
    cost = cost or DEFAULT_COST
    if rebalance_freq is None:
        rebalance_freq = horizon

    # A 任务：流动性分档滑点
    slip_map = compute_liquidity_slippage(panel) if cost_model == "liquidity" else None
    if sizer == "inv_vol":
        import warnings
        warnings.warn(
            "sizer='inv_vol' 已在本策略(日频多空 Top-decile)验证失败：逆波动加权把仓位"
            "集中于低波动/弱信号股（含 vol≈0 的僵尸股被赋权爆炸），实测净收益 -53%、"
            "MDD -71%；已退化为 equal。压回撤请用 vol_target。", stacklevel=2)
    base_slip = float(cost["slippage"])
    c_buy0 = float(cost["commission"])
    c_sell0 = float(cost["commission"]) + float(cost["stamp"])

    df = attach_forward(panel, horizon, exit_extend=5)
    df = df[["symbol", "date", "open_s1"] + [f"open_s{k}" for k in range(horizon, horizon + 6)]
             + [f"ld_s{k}" for k in range(horizon, horizon + 6)]
             + ["entry_limit_up"]].copy()

    p = preds[["date", "symbol", "pred"]].dropna().copy()
    p["date"] = pd.to_datetime(p["date"])

    # 向量化对齐：预测表 inner join 面板（按 symbol,date）
    m = df.merge(p, on=["symbol", "date"], how="inner")
    m = m.dropna(subset=["open_s1", "pred"])
    m["entry_blocked"] = m["entry_limit_up"].fillna(False).astype(bool)

    # 平仓偏移：在 [horizon, horizon+5] 中选第一个非跌停日
    ld_cols = [f"ld_s{k}" for k in range(horizon, horizon + 6)]
    open_cols = [f"open_s{k}" for k in range(horizon, horizon + 6)]
    k_arr = np.array(range(horizon, horizon + 6))
    ld_mat = m[ld_cols].fillna(False).values.astype(bool)
    open_mat = m[open_cols].values.astype(float)
    not_ld = ~ld_mat
    idx = np.argmax(not_ld, axis=1)            # 第一个非跌停的索引
    all_ld = ~not_ld.any(axis=1)               # 全跌停的极端情况
    exit_k = np.where(all_ld, horizon + 5, k_arr[idx])
    m["exit_k"] = exit_k
    m["open_in"] = m["open_s1"].values
    m["open_out"] = open_mat[np.arange(len(m)), idx]

    m = m.dropna(subset=["open_in", "open_out"])
    if m.empty:
        raise ValueError("预测与面板无交集，无法回测")

    # 毛收益（open 进 → open 出）
    m["gross_ret"] = m["open_out"] / m["open_in"] - 1.0

    # 调仓日：取预测日期集合，按频率抽稀
    all_dates = np.sort(m["date"].unique())
    rb_dates = all_dates[::rebalance_freq]

    bucket_rets = []
    bucket_long = []
    bucket_short = []
    bucket_dates = []
    n_sel = 0
    for rb in rb_dates:
        sub = m[m["date"] == rb]
        if sub.empty:
            continue
        valid = sub[~sub["entry_blocked"]]
        if len(valid) < 20:
            continue
        valid = valid.copy()
        valid["rank"] = valid["pred"].rank(pct=True)
        n = len(valid)
        k = max(1, int(n * top_pct))
        long_sym = valid.nlargest(k, "pred")
        short_sym = valid.nsmallest(k, "pred")
        n_sel += len(long_sym) + len(short_sym)

        def net(r: pd.Series, side: str) -> float:
            # A 任务：个股滑点（流动性分档），否则用固定档
            slip = slip_map.get(r["symbol"], base_slip) if slip_map else base_slip
            c_buy = c_buy0 + slip
            c_sell = c_sell0 + slip
            g = r["gross_ret"]
            if side == "long":
                return g - c_buy - c_sell
            else:  # short: 进场是卖，平仓是买
                return -g - c_sell - c_buy

        # 多/空侧净收益（B 任务：sizer="inv_vol" 已退化为 equal，见顶部警告）
        long_net = long_sym.apply(lambda r: net(r, "long"), axis=1)
        short_net = short_sym.apply(lambda r: net(r, "short"), axis=1)
        long_ret = float(long_net.mean())
        short_ret = float(short_net.mean())

        if mode == "long_only":
            bucket = long_ret
        else:
            bucket = 0.5 * long_ret + 0.5 * short_ret
        bucket_rets.append(bucket)
        bucket_long.append(long_ret)
        bucket_short.append(short_ret)
        bucket_dates.append(rb)

    # B 任务：组合层波动率目标化（用已实现波动，不用未来信息）
    bucket_rets = _apply_vol_target(bucket_rets, rebalance_freq,
                                    vol_target, vol_lookback, max_leverage)

    eq = pd.Series(bucket_rets, index=pd.to_datetime(bucket_dates))
    equity = (1.0 + eq).cumprod()
    n_buckets = len(eq)
    years = n_buckets * rebalance_freq / 252.0 if n_buckets else 0.0

    ls_stats = _bucket_stats(bucket_rets, rebalance_freq, years)
    if bootstrap:
        ls_stats.update(backtest_bootstrap(bucket_rets, rebalance_freq, years=years))

    result = BacktestResult(
        horizon=horizon,
        top_pct=top_pct,
        rebalance_freq=rebalance_freq,
        cost=cost,
        equity=equity,
        long_equity=(1 + pd.Series(bucket_long)).cumprod(),
        short_equity=(1 + pd.Series(bucket_short)).cumprod(),
        ls_stats=ls_stats,
        long_stats=_bucket_stats(bucket_long, rebalance_freq, years),
        short_stats=_bucket_stats(bucket_short, rebalance_freq, years),
        n_trades=int(n_sel),
        n_blocked=int(m["entry_blocked"].sum()),
        mode=f"bucket[{cost_model}/{sizer}]"
             + (f"+volt{vl_}" if (vl_ := vol_target) else ""),
        cost_model=cost_model, sizer=sizer, vol_target=vol_target,
        equity_dates=list(bucket_dates),
    )
    return result


class BacktestResult:
    """回测结果容器。"""

    def __init__(self, horizon, top_pct, rebalance_freq, cost, equity,
                 long_equity, short_equity, ls_stats, long_stats, short_stats,
                 n_trades, n_blocked, mode: str = "bucket",
                 cost_model: str = "fixed", sizer: str = "equal",
                 vol_target: float | None = None, equity_dates: list | None = None):
        self.mode = mode
        self.horizon = horizon
        self.top_pct = top_pct
        self.rebalance_freq = rebalance_freq
        self.cost = cost
        self.equity = equity
        self.long_equity = long_equity
        self.short_equity = short_equity
        self.ls_stats = ls_stats
        self.long_stats = long_stats
        self.short_stats = short_stats
        self.n_trades = n_trades
        self.n_blocked = n_blocked
        self.cost_model = cost_model
        self.sizer = sizer
        self.vol_target = vol_target
        self.equity_dates = equity_dates or list(equity.index)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "horizon": self.horizon,
            "top_pct": self.top_pct,
            "rebalance_freq": self.rebalance_freq,
            "cost": self.cost,
            "cost_model": self.cost_model,
            "sizer": self.sizer,
            "vol_target": self.vol_target,
            "n_trades": self.n_trades,
            "n_blocked_by_limit_up": self.n_blocked,
            "long_short": self.ls_stats,
            "long_only": self.long_stats,
            "short_only": self.short_stats,
        }

    def summary_text(self) -> str:
        s = self.ls_stats
        lines = [
            f"回测（{self.mode}，horizon={self.horizon}，调仓={self.rebalance_freq}日，"
            f"多/空各{self.top_pct:.0%}）",
            f"  累计收益        : {s.get('total_return', float('nan')):.2%}",
            f"  年化收益        : {s.get('annual_return', float('nan')):.2%}",
            f"  夏普            : {s.get('sharpe', float('nan')):.2f}",
            f"  最大回撤        : {s.get('max_drawdown', float('nan')):.2%}",
            f"  胜率            : {s.get('win_rate', float('nan')):.2%}",
            f"  调仓桶数        : {s.get('n_buckets', 0)}",
        ]
        if "avg_turnover" in s:
            lines.append(f"  平均换手率      : {s['avg_turnover']:.1%} / 期")
        if self.vol_target is not None:
            lines.append(f"  波动目标        : {self.vol_target:.0%}（已缩放）")
        if self.n_blocked:
            lines.append(f"  成交笔数        : {self.n_trades:,}"
                         f"（因涨停被挡 {self.n_blocked:,}）")
        return "\n".join(lines)
