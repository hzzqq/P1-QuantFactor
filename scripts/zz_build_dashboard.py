"""生成 P1 组合构建诊断看板（单文件 HTML，2026-11-11）。

读取：
- reports/portfolio_equity_curves_20261111.json  （权益曲线）
- reports/portfolio_construction_diag_20261111.json（网格 + 基准相对）

产出：frontend/p1_construction_dashboard.html（自包含，可直接浏览器打开）。
遵循区域惯例：A 股红涨绿跌（盈利=红、亏损=绿）。
"""
from __future__ import annotations

import json
from pathlib import Path

PROC = Path("E:/project/sj/data/P1/processed/reports")
OUT = Path("E:/project/sj/frontend/p1_construction_dashboard.html")

eq = json.loads((PROC / "portfolio_equity_curves_20261111.json").read_text(encoding="utf-8"))
diag = json.loads((PROC / "portfolio_construction_diag_20261111.json").read_text(encoding="utf-8"))

# 提取 rf=10 buf=0 的 net 窗口统计（核心对照）
def grab(mode):
    for r in diag["grid_results"]:
        if r["mode"] == mode and r["cost"] == "net" and r["rf"] == 10 and r["buffer"] == 0.0:
            return r["ls"]
    return {}

lo_w = grab("long_only")
ls_w = grab("long_short")
windows = ["FULL", "2024+", "2026", "recent59td"]
long_only_net = [round(lo_w[w]["total_return"] * 100, 1) if lo_w.get(w) else None for w in windows]
long_short_net = [round(ls_w[w]["total_return"] * 100, 1) if ls_w.get(w) else None for w in windows]

# 基准相对 rf=10
br = {r["window"]: r["excess_total_return"] for r in diag["benchmark_relative"] if r["rf"] == 10}
alpha_decay = [round(br.get(w, float("nan")) * 100, 1) for w in windows]

# 换手：rf=10 buf=0 vs buf=0.1（净）
def turn(buf):
    for r in diag["grid_results"]:
        if r["mode"] == "long_short" and r["cost"] == "net" and r["rf"] == 10 and r["buffer"] == buf:
            return round(r["avg_turnover"] * 100)
    return None
turn_buf0, turn_buf1 = turn(0.0), turn(0.1)

DATA = {
    "dates": eq["long_only_net"]["dates"],
    "idx2026": eq["idx_2026"],
    "longOnly": eq["long_only_net"]["values"],
    "longShort": eq["long_short_net"]["values"],
    "benchmark": eq["benchmark_gross"]["values"],
    "windows": windows,
    "longOnlyNet": long_only_net,
    "longShortNet": long_short_net,
    "alphaDecay": alpha_decay,
    "turnBuf0": turn_buf0,
    "turnBuf1": turn_buf1,
}

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>P1 量化信号 · 组合构建诊断看板</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Sora:wght@400;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg0:#0a0e1a; --bg1:#121a2e; --bg2:#1a2440; --glass:rgba(255,255,255,.05);
  --stroke:rgba(255,255,255,.12); --txt:#e8edf7; --mut:#90a0c0;
  --up:#ff4d5e; --down:#21c08b; --accent:#5b8cff; --warn:#ffb547;
  --up-soft:rgba(255,77,94,.15); --down-soft:rgba(33,192,139,.15);
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  font-family:'Sora',system-ui,sans-serif; color:var(--txt); line-height:1.6;
  background:
    radial-gradient(1200px 600px at 15% -10%, #1c2a52 0%, transparent 55%),
    radial-gradient(900px 500px at 90% 0%, #2a1c44 0%, transparent 50%),
    linear-gradient(180deg, var(--bg0), var(--bg1) 60%, var(--bg0));
  background-attachment:fixed; min-height:100vh; padding-bottom:60px;
}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
nav{position:sticky;top:0;z-index:20;backdrop-filter:blur(14px);
  background:rgba(10,14,26,.72);border-bottom:1px solid var(--stroke);}
nav .row{max-width:1080px;margin:0 auto;padding:14px 20px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
nav .logo{font-family:'Space Grotesk';font-weight:700;font-size:18px;letter-spacing:.5px}
nav .tag{font-size:12px;color:var(--mut);border:1px solid var(--stroke);padding:3px 10px;border-radius:999px}
section{padding:46px 0 8px}
h2{font-family:'Space Grotesk';font-size:clamp(22px,4vw,30px);font-weight:700;margin-bottom:6px}
.sub{color:var(--mut);font-size:15px;margin-bottom:22px}
.hero{padding-top:54px}
.hero h1{font-family:'Space Grotesk';font-size:clamp(28px,6vw,46px);font-weight:700;line-height:1.15;
  background:linear-gradient(90deg,#fff,#9db4ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero p.lead{color:var(--mut);font-size:17px;max-width:760px;margin-top:14px}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:30px}
.card{background:var(--glass);border:1px solid var(--stroke);border-radius:18px;padding:22px;
  backdrop-filter:blur(8px);position:relative;overflow:hidden}
.card .k{font-size:13px;color:var(--mut);margin-bottom:8px}
.card .v{font-family:'Space Grotesk';font-size:clamp(26px,5vw,38px);font-weight:700}
.card .d{font-size:13px;color:var(--mut);margin-top:6px}
.pos{color:var(--up)} .neg{color:var(--down)} .neu{color:var(--accent)}
.panel{background:var(--glass);border:1px solid var(--stroke);border-radius:20px;padding:22px;
  backdrop-filter:blur(8px);margin-top:18px}
.chart-box{position:relative;height:380px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.badge{display:inline-block;font-size:12px;padding:4px 12px;border-radius:999px;font-weight:600}
.badge.warn{background:rgba(255,181,71,.16);color:var(--warn);border:1px solid rgba(255,181,71,.4)}
.badge.up{background:var(--up-soft);color:var(--up);border:1px solid rgba(255,77,94,.4)}
.badge.down{background:var(--down-soft);color:var(--down);border:1px solid rgba(33,192,139,.4)}
.warnbar{margin-top:22px;border:1px solid rgba(255,181,71,.45);background:rgba(255,181,71,.08);
  border-radius:16px;padding:20px 22px;display:flex;gap:16px;align-items:flex-start}
.warnbar .ic{font-size:26px;line-height:1}
.warnbar h3{font-family:'Space Grotesk';color:var(--warn);font-size:19px;margin-bottom:6px}
.rec{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:8px}
.rec .card h4{font-family:'Space Grotesk';font-size:18px;margin-bottom:8px}
.rec .card ul{margin-left:18px;color:var(--mut);font-size:14px}
.rec .card ul li{margin:5px 0}
footer{margin-top:50px;color:var(--mut);font-size:13px;border-top:1px solid var(--stroke);padding-top:22px}
.note{font-size:13px;color:var(--mut);margin-top:10px;line-height:1.5}
@media(max-width:720px){
  .kpis,.grid2,.rec{grid-template-columns:1fr}
  .chart-box{height:320px}
  nav .tag{display:none}
}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*{transition:none!important}}
canvas{max-width:100%}
</style>
</head>
<body>
<nav><div class="row">
  <span class="logo">P1 量化信号</span>
  <span class="tag">组合构建诊断看板</span>
  <span class="tag">2026-11-11</span>
</div></nav>

<div class="wrap">
  <section class="hero">
    <h1>近期弱势根因：alpha 衰减，<br/>不是构造费，也不是换手费</h1>
    <p class="lead">诊断结论——把纯多头组合改为<b>市场中性（多空）</b>构造，能在 2026 把多空边缘（LS_t 仍 +2.07）兑现为收益；但<b>提高调仓周期（降频）反而磨掉近期边缘</b>。真正要治的是 alpha 再生。</p>
    <div class="kpis">
      <div class="card">
        <div class="k">2026 净收益 · 市场中性</div>
        <div class="v pos">+10.5%</div>
        <div class="d">夏普 +0.83（rf=10, buffer=0）— 回血杠杆</div>
      </div>
      <div class="card">
        <div class="k">2026 净收益 · 纯多头</div>
        <div class="v neg">−1.8%</div>
        <div class="d">夏普 +0.02 — 全频率转负</div>
      </div>
      <div class="card">
        <div class="k">多头 alpha 衰减</div>
        <div class="v neu">+47% → +3.7%</div>
        <div class="d">相对等权全市场：2024+ → 近59日</div>
      </div>
    </div>
  </section>

  <section>
    <h2>一、净值曲线：beta 顺风 vs 中性稳健</h2>
    <div class="sub">长期多头借市场 beta 冲到 3.34×，但 regime 脆弱；市场中性仅 1.69× 却三段全正、2026 逆势上行。阴影区为 2026 近期弱势段。</div>
    <div class="panel"><div class="chart-box"><canvas id="eqChart"></canvas></div></div>
    <div class="note">A 股惯例：红=涨/盈利，绿=跌/亏损。基准=等权全市场（零成本，作市场参照）。</div>
  </section>

  <section>
    <h2>二、分时段收益：市场中性全正，纯多头 2026 翻绿</h2>
    <div class="sub">同一 rf=10、buffer=0、净成本口径，跨四个时段对照（recent59td 仅 3 个调仓桶，样本极小，仅供参考）。</div>
    <div class="panel"><div class="chart-box"><canvas id="winChart"></canvas></div></div>
  </section>

  <section>
    <h2>三、alpha 衰减：多头相对全市场的超额在蒸发</h2>
    <div class="sub">基准相对 = 纯多头 gross − 等权全市场 gross。从 2024+ 的 +47% 塌到近 59 交易日的 +3.7%——近期亏损的底层根因。</div>
    <div class="panel"><div class="chart-box"><canvas id="alphaChart"></canvas></div></div>
  </section>

  <div class="warnbar">
    <div class="ic">⚠️</div>
    <div>
      <h3>诚实边界：组合构建改造是止血，不是治愈</h3>
      <p style="color:var(--mut);font-size:14px">市场中性把「残余正 alpha 利差」兑现、绕开 beta 拖累，但<b>救不回已衰减的 alpha 本身</b>。治本仍靠 alpha 再生（补换手率/流通市值因子 + 排查训练不稳定）。此外，市场中性需 A 股做空能力，<b>零售端 StockSignal 落地被现实卡住</b>；recent59td 窗口样本极小、不可当统计显著。</p>
    </div>
  </div>

  <section>
    <h2>四、降换手是免费午餐</h2>
    <div class="sub">在 rf=10 下把 buffer 从 0 提到 0.1（连续持有），换手从 <b id="t0"></b> 降到 <b id="t1"></b>，全样本净收益不降反升（+68.6%→+72.0%）。应保留——但别为提高降换手而加大 rf（那会磨掉近期边缘）。</div>
    <div class="grid2" style="margin-top:16px">
      <div class="card"><div class="k">建议保留</div><div class="v neu">buffer = 0.1</div><div class="d">连续持有降换手，几乎无代价</div></div>
      <div class="card"><div class="k">禁止</div><div class="v neg">rf &gt; 10</div><div class="d">降频磨掉 2026 多空边缘</div></div>
    </div>
  </section>

  <section>
    <h2>五、落地建议（待老板拍板）</h2>
    <div class="rec">
      <div class="card">
        <h4 class="pos">✓ 可做 · 零售端</h4>
        <ul>
          <li>维持纯多头信号，但 UI 增加<b>基准相对展示</b></li>
          <li>明确标注<b>近期 alpha 衰减风险</b>徽标</li>
          <li>P1 管线采用 <b>buffer=0.1</b> 降换手</li>
          <li>复用现有信号 <code>top_long</code>/<code>top_short</code> 做多空参考视图</li>
        </ul>
      </div>
      <div class="card">
        <h4 class="neu">△ 待评估 · 多空产品</h4>
        <ul>
          <li>市场中性需 A 股<b>做空能力</b>（两融/合格投资者）</li>
          <li>需先在 37 / zz_daily_push 增加 mode 透传</li>
          <li>做空腿可行性评估 + 门禁后再动生产</li>
          <li>当前仅作研究参考，不自动交易</li>
        </ul>
      </div>
    </div>
  </section>

  <footer>
    <div>数据：<code>pred_ens_v39gru_w025_h10.parquet</code>（生产集成，全历史 2019-01-02~2026-11-09）· 引擎 <code>src/backtest/engine.py::run_backtest_continuous</code></div>
    <div style="margin-top:8px">窗口 FULL / 2024+ / 2026 / recent59td · 成本 net=真实 A 股成本，gross=零成本 · 基准=等权全市场（面板无指数）</div>
    <div style="margin-top:8px">未改动生产 pred/signal/ks。本看板为只读诊断可视化，投产前请人工复核数据口径与无障碍合规。</div>
  </footer>
</div>

<script>
const DATA = /*__DATA__*/;
const UP='#ff4d5e', DOWN='#21c08b', ACCENT='#5b8cff', MUT='#90a0c0', GRID='rgba(255,255,255,.08)';
function fmtPct(v){ return v==null?'—':(v>=0?'+':'')+v.toFixed(1)+'%'; }
const ctxEq=document.getElementById('eqChart').getContext('2d');
const labels=DATA.dates;
const region=DATA.idx2026;
// 2026 阴影：用 annotation 区带（手动画背景）
const eqChart=new Chart(ctxEq,{
  type:'line',
  data:{labels,
    datasets:[
      {label:'纯多头 (net)',data:DATA.longOnly,borderColor:UP,backgroundColor:'rgba(255,77,94,.08)',fill:true,borderWidth:2,pointRadius:0,tension:.15},
      {label:'市场中性 (net)',data:DATA.longShort,borderColor:ACCENT,backgroundColor:'rgba(91,140,255,.08)',fill:true,borderWidth:2,pointRadius:0,tension:.15},
      {label:'等权全市场 (gross)',data:DATA.benchmark,borderColor:MUT,borderDash:[6,4],borderWidth:1.5,pointRadius:0,tension:.15}
    ]},
  options:{responsive:true,maintainAspectRatio:false,interaction:{mode:'index',intersect:false},
    plugins:{legend:{labels:{color:MUT,font:{family:'Sora'}}},
      tooltip:{callbacks:{label:c=>c.dataset.label+': '+(c.parsed.y).toFixed(2)+'×'}}},
    scales:{x:{ticks:{color:MUT,maxTicksLimit:12,font:{family:'Sora'}},grid:{color:GRID}},
      y:{ticks:{color:MUT,font:{family:'Sora'},callback:v=>v.toFixed(1)+'×'},grid:{color:GRID}}}},
  plugins:[{id:'band',beforeDatasetsDraw(chart){
    const {ctx,chartArea,scales}=chart; if(!chartArea) return;
    const x=scales.x; const n=labels.length; if(n<2) return;
    const left=x.getPixelForValue(0), right=x.getPixelForValue(n-1);
    const xa=left+(right-left)*(DATA.idx2026/(n-1));
    if(!isFinite(xa)) return;
    ctx.save(); ctx.fillStyle='rgba(255,181,71,.10)';
    ctx.fillRect(xa,chartArea.top,chartArea.right-xa,chartArea.bottom-chartArea.top);
    ctx.restore();
  }}]
});

new Chart(document.getElementById('winChart'),{
  type:'bar',
  data:{labels:DATA.windows,datasets:[
    {label:'纯多头 net',data:DATA.longOnlyNet,backgroundColor:UP},
    {label:'市场中性 net',data:DATA.longShortNet,backgroundColor:ACCENT}
  ]},
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{labels:{color:MUT,font:{family:'Sora'}}},
      tooltip:{callbacks:{label:c=>c.dataset.label+': '+fmtPct(c.parsed.y)}}},
    scales:{x:{ticks:{color:MUT,font:{family:'Sora'}},grid:{color:GRID}},
      y:{ticks:{color:MUT,font:{family:'Sora'},callback:v=>v+'%'},grid:{color:GRID}}}}
});

new Chart(document.getElementById('alphaChart'),{
  type:'bar',
  data:{labels:DATA.windows,datasets:[
    {label:'基准相对 alpha (多头 gross − 全市场 gross)',data:DATA.alphaDecay,
     backgroundColor:DATA.alphaDecay.map(v=>v>=0?UP:DOWN)}
  ]},
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{labels:{color:MUT,font:{family:'Sora'}}},
      tooltip:{callbacks:{label:c=>'超额: '+fmtPct(c.parsed.y)}}},
    scales:{x:{ticks:{color:MUT,font:{family:'Sora'}},grid:{color:GRID}},
      y:{ticks:{color:MUT,font:{family:'Sora'},callback:v=>v+'%'},grid:{color:GRID}}}}
});
document.getElementById('t0').textContent=DATA.turnBuf0+'%';
document.getElementById('t1').textContent=DATA.turnBuf1+'%';
</script>
</body>
</html>
"""

html = HTML.replace("/*__DATA__*/", json.dumps(DATA, ensure_ascii=False))
OUT.write_text(html, encoding="utf-8")
print(f"看板已生成: {OUT}  ({len(html):,} 字节)")
print(f"  窗口对照 longOnlyNet={long_only_net} longShortNet={long_short_net}")
print(f"  alphaDecay={alpha_decay}  换手 buf0={turn_buf0}% buf1={turn_buf1}%")
