"""核实 A/B 两个方向的家底：GRU 预测是否还在、GRU 权重有哪些、②b 进度。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(r"E:/project/sj")
PROC = ROOT / "data" / "P1" / "processed"

print("=" * 66, flush=True)
print("[1] processed/ 下所有 pred_*.parquet", flush=True)
preds = sorted(PROC.glob("pred_*.parquet"))
for p in preds:
    print(f"  {p.name}  {p.stat().st_size/1e6:.1f}MB", flush=True)
print(f"  合计 {len(preds)} 个", flush=True)

print("=" * 66, flush=True)
print("[2] 全盘搜 GRU 相关产出（pred/parquet）", flush=True)
found = []
for base in (ROOT / "data", ROOT / "models"):
    if not base.exists():
        continue
    for p in base.rglob("*"):
        if p.is_file() and "gru" in p.name.lower():
            found.append(p)
print(f"  命中 {len(found)} 个:", flush=True)
for p in found[:40]:
    print(f"    {p.relative_to(ROOT)}  {p.stat().st_size/1e6:.2f}MB", flush=True)
if len(found) > 40:
    print(f"    ...(还有 {len(found)-40} 个)", flush=True)

print("=" * 66, flush=True)
print("[3] GRU 权重 .pt（可用于重新推理出预测）", flush=True)
mdl = ROOT / "models"
pts = sorted(mdl.rglob("*.pt")) if mdl.exists() else []
gru_pt = [p for p in pts if "gru" in p.name.lower()]
print(f"  .pt 总数 {len(pts)}，其中 gru 权重 {len(gru_pt)}:", flush=True)
for p in gru_pt:
    print(f"    {p.relative_to(mdl)}  {p.stat().st_size/1024:.0f}KB", flush=True)

print("=" * 66, flush=True)
print("[4] ②b 进度（30b_run.log 末 12 行）", flush=True)
lg = ROOT / "data" / "P1" / "raw" / "baostock_daily" / "30b_run.log"
if lg.exists():
    lines = lg.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    for line in lines[-12:]:
        print(f"  {line}", flush=True)
else:
    print("  (无日志)", flush=True)

print("=" * 66, flush=True)
print("[5] 推理/预测类脚本（重出 GRU 预测用）", flush=True)
for p in sorted((ROOT / "P1-QuantFactor" / "scripts").glob("*.py")):
    n = p.name.lower()
    if any(k in n for k in ("predict", "walkforward", "train_nn", "infer", "gru", "ensemble")):
        print(f"  {p.name}", flush=True)
