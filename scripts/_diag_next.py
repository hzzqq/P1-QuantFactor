"""查证下一步方向的依据：回测/GRU/IC衰减现状（只读诊断）"""
from __future__ import annotations

import json
from pathlib import Path

PROJ = Path(r"E:/project/sj/P1-QuantFactor")
DATA = Path(r"E:/project/sj/data/P1/processed")
ROOT = Path(r"E:/project/sj")

print("=" * 66, flush=True)
print("[A] 脚本清单（看有哪些能力已具备）", flush=True)
for p in sorted((PROJ / "scripts").glob("*.py")):
    print(f"  {p.name}", flush=True)

print("=" * 66, flush=True)
print("[B] 回测/收益类产出（是否验证过'IC→真钱'）", flush=True)
hits = []
for pat in ("*backtest*", "*bt_*", "*equity*", "*pnl*", "*return*"):
    hits += list(DATA.glob(pat))
    hits += list((PROJ / "results").glob(pat)) if (PROJ / "results").exists() else []
    hits += list(PROJ.glob(pat))
seen = set()
for p in hits:
    if p.name in seen:
        continue
    seen.add(p.name)
    print(f"  {p.name}  ({p.parent.name})  {p.stat().st_size/1024:.1f}KB", flush=True)
if not seen:
    print("  !! 未找到任何回测/净值产出 → 尚未验证过 IC 能否变成钱", flush=True)

print("=" * 66, flush=True)
print("[C] GRU/神经网络现状", flush=True)
gru = [p for p in sorted((PROJ / "scripts").glob("*.py")) if any(
    k in p.name.lower() for k in ("gru", "nn", "torch", "lstm", "deep", "neural"))]
print(f"  GRU 相关脚本: {[p.name for p in gru] or '无'}", flush=True)
mdl = ROOT / "models"
if mdl.exists():
    pts = sorted(mdl.rglob("*.pt"))
    print(f"  models/ 下 .pt 模型 {len(pts)} 个，样例: {[p.name for p in pts[:8]]}", flush=True)
    gru_pt = [p for p in pts if any(k in str(p).lower() for k in ("gru", "attn", "lstm", "torch"))]
    print(f"  其中疑似神经网络权重: {len(gru_pt)} → {[p.name for p in gru_pt[:6]]}", flush=True)

print("=" * 66, flush=True)
print("[D] 生产基线 v39 的逐年 IC（衰减形状）", flush=True)
for p in sorted(DATA.glob("yearly*.json"))[:12]:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        print(f"  {p.name}: {d}", flush=True)
    except Exception as e:
        print(f"  {p.name} 读取失败 {e!r}", flush=True)
for p in sorted(DATA.glob("report*.json"))[:8]:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        s = json.dumps(d, ensure_ascii=False)
        print(f"  {p.name}: {s[:400]}", flush=True)
    except Exception as e:
        print(f"  {p.name} 读取失败 {e!r}", flush=True)

print("=" * 66, flush=True)
print("[E] 预测产出（pred_baseline_*.parquet）", flush=True)
for p in sorted(DATA.glob("pred_baseline_*.parquet"))[:12]:
    print(f"  {p.name}  {p.stat().st_size/1e6:.1f}MB", flush=True)
