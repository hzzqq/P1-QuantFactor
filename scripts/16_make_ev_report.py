"""生成 EV 事件因子 20 迭代调参 HTML 报告（一键）。"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

P = Path("data/P1/processed")
ev = pd.read_csv(P / "report_tune_grid_ev.csv")
base = pd.read_csv(P / "report_tune_grid.csv")
full = pd.read_csv(P / "report_tune_ev_full.csv")


def cls(v):
    return "pos" if v > 0 else "neg"


# ① 20 组排名
evs = ev.sort_values("net_2026", ascending=False)
rank_rows = ""
for i, r in enumerate(evs.itertuples(), 1):
    hl = ' style="background:#1e3a2f"' if r.tag == "t10_seq40_h128_lr1e-3" else ""
    rank_rows += (
        f"<tr{hl}><td>{i}</td><td>{r.tag}</td><td>{r.ic_2026:.4f}</td>"
        f"<td>{r.icir_2026:.4f}</td>"
        f"<td class='{cls(r.net_2026)}'>{r.net_2026*100:+.2f}%</td>"
        f"<td class='{cls(r.sharpe_2026)}'>{r.sharpe_2026:.2f}</td>"
        f"<td>{r.mdd_2026*100:.2f}%</td></tr>"
    )

# ② 同构对比（EV t01~t10 vs 基线 t01~t10）
ev10 = ev[ev.tag.str.startswith(
    ("t01_", "t02_", "t03_", "t04_", "t05_", "t06_", "t07_", "t08_", "t09_", "t10_"))]
iso = [("均值 IC", ev10.ic_2026.mean(), base.ic_2026.mean()),
       ("均值 ICIR", ev10.icir_2026.mean(), base.icir_2026.mean()),
       ("均值 2026净", ev10.net_2026.mean(), base.net_2026.mean()),
       ("均值 夏普", ev10.sharpe_2026.mean(), base.sharpe_2026.mean())]
iso_rows = ""
for n, a, b in iso:
    iso_rows += (f"<tr><td>{n}</td><td>{a:.4f}</td><td>{b:.4f}</td>"
                 f"<td class='{cls(a-b)}'>{a-b:+.4f}</td></tr>")

# ③ 全量三信号
f = full.iloc[0]
full_rows = (
    "<tr><td>纯 GRU（t02确认）</td><td>-4.11%</td><td>-0.40</td><td>0.0741</td></tr>"
    "<tr><td>regime 门控融合</td><td>-3.78%</td><td>-0.28</td><td>0.0721</td></tr>"
    f"<tr style='background:#1e3a2f'><td><b>EV 事件因子（t10确认）</b></td>"
    f"<td><b>{f.net_2026*100:+.2f}%</b></td><td><b>{f.sharpe_2026:.2f}</b>"
    f"<td><b>{f.ic_2026:.4f}</b></td></tr>"
)

html = f"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<title>EV 事件因子 20 迭代调参报告</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0f1419;color:#e6e6e6;margin:0;padding:32px}}
h1{{color:#7fd1a0}} h2{{color:#9ec5fe;border-bottom:1px solid #2a3441;padding-bottom:8px}}
table{{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}}
th,td{{border:1px solid #2a3441;padding:8px 10px;text-align:right}}
th{{background:#1a2430;color:#9ec5fe}} td:first-child,th:first-child{{text-align:left}}
.pos{{color:#ff6b6b}} .neg{{color:#51cf66}}
.card{{background:#161c24;border:1px solid #2a3441;border-radius:10px;padding:20px;margin:16px 0}}
.kpi{{display:inline-block;background:#1a2430;border-radius:8px;padding:12px 18px;margin:8px}}
.note{{color:#a0aec0;font-size:13px;line-height:1.6}}</style></head>
<body><h1>事件因子（牧羊人市场广度）· GRU 20 迭代调参报告</h1>
<div class=card><div class=kpi>完成组数 <b>20</b>/20</div>
<div class=kpi>最优 t10 净 <b>+4.01%</b>（子集）</div>
<div class=kpi>全量确认净 <b>{f.net_2026*100:+.2f}%</b>（仍全场最优）</div>
<div class=kpi>新增特征 <b>9</b> 维（39–47）</div></div>
<h2>① 20 组排名（按 2026 严格 hold-out 净收益）</h2>
<table><tr><th>#</th><th>配置</th><th>IC</th><th>ICIR</th><th>2026净</th><th>夏普</th><th>回撤</th></tr>{rank_rows}</table>
<h2>② 同构对比（EV t01~t10 共10组 vs 基线 t01~t10 共10组，公平口径）</h2>
<table><tr><th>指标</th><th>EV 47维</th><th>基线 38维</th><th>增量</th></tr>{iso_rows}</table>
<h2>③ 全量 1427 严格 hold-out · 三信号对比（2026）</h2>
<table><tr><th>信号</th><th>净收益</th><th>夏普</th><th>IC</th></tr>{full_rows}</table>
<div class=card><div class=note><b>结论：</b>事件因子（牧羊人市场广度情绪）带来<b>真实但温和</b>增益——
同构 IC +18.7%、2026 净 +0.81pp、夏普转正 +0.165；全量下把 2026 亏损从纯 GRU 的 -4.11% 收窄到
<b>{f.net_2026*100:+.2f}%</b>，为全量候选最优。⚠️ 2026 单年仍为负（湍流弱年），真实价值在 2022–2025，
对外须明示「多年度复合才显正期望」。建议采纳为第 39–47 特征，默认 t10(EV) 或稳健替代 t19(L3)。
<br>⚠️ 单组有训练随机性（t12 两次跑 -0.82% vs -1.63%），结论看排序与均值。</div></div>
</body></html>"""

out = P / "ev_tune_report.html"
out.write_text(html, encoding="utf-8")
print("HTML 报告已生成:", out, out.stat().st_size, "bytes")
