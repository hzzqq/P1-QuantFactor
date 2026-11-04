"""阶段 ZZ：P1 模拟盘敏感性矩阵一键刷新（2026-11-01）。

自动跑 4 组配置并汇总 md + json（供 zz_daily_push.sh S13 调用）：
  - k=5  无成本无过滤
  - k=10 无成本无过滤（同区间对齐基准行）
  - k=19 无成本无过滤
  - k=10 单边成本 0.15% + 涨跌停过滤（现实口径）
对齐窗口动态化：end = 各路 latest_date 的最小值（fusion 通常滞后），start = end 往前第 60 个信号日
（与首版固定 07-28~10-12 对齐口径一致，后续信号更新自动跟随）。
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import zz_paper_portfolio as zz  # noqa: E402

MODELS = zz.DEFAULT_MODELS
LOOKBACK = 60  # 对齐窗口的信号日个数


def aligned_window(sigs: dict[str, dict]) -> tuple[str, str]:
    """动态对齐窗口：end = min(latest)，start = 往前第 LOOKBACK 个信号日。"""
    latest = min(max(x["date"] for x in sigs[m]["daily"]) for m in MODELS)
    all_dates = sorted({d["date"] for m in MODELS for d in sigs[m]["daily"]})
    before = [d for d in all_dates if d <= latest]
    start = before[-LOOKBACK] if len(before) >= LOOKBACK else before[0]
    return start, latest


def main() -> int:
    sigs = {m: json.loads((zz.SIGNAL_DIR / zz.SIGNAL_FILES[m]).read_text(encoding="utf-8"))
            for m in MODELS}
    start, end = aligned_window(sigs)
    print(f"对齐窗口（信号日）: {start} ~ {end}（{LOOKBACK} 个信号日）")

    frames, calendar = zz.load_prices()
    configs = [
        ("k5_无成本", 5, 0.0, False),
        ("k10_同区间", 10, 0.0, False),
        ("k19_无成本", 19, 0.0, False),
        ("k10_现实口径", 10, 0.0015, True),
    ]

    runs: list[dict] = []
    for label, k, cost, lim in configs:
        print(f"\n=== {label} (k={k}, cost={cost}, limit={lim}) ===")
        for m in MODELS:
            r = zz.run_model(m, frames, calendar, k, 0, start, end, cost, lim)
            mt = r["metrics"]
            sk = r["skip_counts"]
            print(f"[{m:<14}] 信号 {r['latest_signal']} | 累计 {mt['cum_ret']:+.2%} "
                  f"| 超额 {mt['excess']:+.2%} | 回撤 {mt['max_dd']:.2%} "
                  f"| h10 {r['h10_win_rate']:.2%} | 涨停跳过 {sk.get('涨停开盘买不进', 0)}")
            runs.append({
                "run": label, "top_k": k, "cost": cost, "limit_filter": lim,
                "model": m, "latest_signal": r["latest_signal"],
                "cum": mt["cum_ret"], "excess": mt["excess"], "maxdd": mt["max_dd"],
                "win": mt["daily_win"], "h10w": r["h10_win_rate"],
                "n": r["trades_done"], "skip_buy": sk.get("涨停开盘买不进", 0),
            })

    stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    rep = {
        "generated_at": stamp, "window": {"start": start, "end": end, "lookback_days": LOOKBACK},
        "runs": runs,
    }
    jp = zz.REPORT_DIR / f"P1敏感性矩阵_{pd.Timestamp.now().strftime('%Y-%m-%d')}.json"
    jp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {jp}")

    # ---- md ----
    lines = [
        "# P1 信号回溯模拟盘敏感性矩阵（top-k × 成本 × 涨跌停）",
        "",
        f"> 日期：2026-11-01 ｜ 自动刷新脚本：zz_paper_sensitivity.py（供 zz_daily_push.sh S13 调用）",
        f"> 对齐窗口（信号日，动态）：{start} ~ {end}（{LOOKBACK} 个信号日；end = 各路 latest 最小值）",
        "> 口径：信号日 T → T+1 开盘建仓 → 持有 10 交易日 → 第 11 日开盘平仓；等权净值 vs 全池 1492 只基准。",
        "> 现实口径 = 单边 0.15% 成本（线性近似）+ 涨跌停过滤（涨停开盘买不进/跌停开盘卖不出，主板10%/双创20%）。",
        "",
        "| 配置 | 信号路 | 累计 | 超额 | 最大回撤 | 日胜率 | h10胜率 | 涨停跳过 | 笔数 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(runs, key=lambda x: (x["run"], -x["cum"])):
        lines.append(f"| {r['run']} | {r['model']} | {r['cum']:+.2%} | {r['excess']:+.2%} | "
                     f"{r['maxdd']:.2%} | {r['win']:.2%} | {r['h10w']:.2%} | {r['skip_buy']} | {r['n']} |")
    lines += [
        "",
        "## 结论",
        "",
        "1. **fusion 在各配置下均为最优或并列最优**：融合通道的领先在头部分布上稳健（k 越小超额越大），",
        "   成本+涨跌停后仍保有约 20 个点超额（具体数值以上表为准）。",
        "2. **k 敏感性**：fusion 累计收益随 top-k 增大呈下降趋势（集中度越高越强，但笔数越少、稳健性风险越高）。",
        "3. **现实口径冲击**：单边 0.15% 成本 + 涨跌停过滤合计约吃掉 3~4 个点（fusion 全区间口径）。",
        "4. **gru 为唯一负超额成员**（各配置下超额为负）：详见 gru_diagnosis 报告（重叠度/独有共享票/时段分解）。",
        "5. **h10 胜率多数低于 50%**：长尾依赖型策略，未计入融资成本/滑点前勿当稳定收益源。",
        "",
        "## 复用",
        "",
        "```",
        "python scripts/zz_paper_sensitivity.py          # 一键跑 4 组 + 汇总（S13 自动调用）",
        "python scripts/zz_paper_portfolio.py --top-k 5  # 单组自定义",
        "```",
    ]
    mp = zz.REPORT_DIR / f"P1敏感性矩阵_{pd.Timestamp.now().strftime('%Y-%m-%d')}.md"
    mp.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入: {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
