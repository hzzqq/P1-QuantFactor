"""诊断 ②b 现状：封禁? 进度? v40 因子是否真有新信息?（一次性只读诊断脚本）"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

OUTDIR = Path(r"E:/project/sj/data/P1/raw/baostock_daily")
PROJ = Path(r"E:/project/sj/P1-QuantFactor")
DATA = Path(r"E:/project/sj/data/P1/processed")

print("=" * 60, flush=True)
print("[1] baostock 封禁状态", flush=True)
try:
    import baostock as bs

    lg = bs.login()
    print(f"  login error_code={lg.error_code} msg={lg.error_msg}", flush=True)
    if lg.error_code == "0":
        rs = bs.query_history_k_data_plus(
            "sh.600000",
            "date,close,turn,amount",
            start_date="2024-01-01",
            end_date="2024-01-10",
            frequency="d",
        )
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        print(f"  UNBANNED! fetched {len(rows)} rows, sample={rows[:2]}", flush=True)
        bs.logout()
    else:
        print("  → 仍在黑名单，②b 无法继续拉取", flush=True)
except Exception as e:
    print(f"  probe exception: {e!r}", flush=True)

print("=" * 60, flush=True)
print("[2] ②b 进度", flush=True)
if OUTDIR.exists():
    finals = sorted(
        p.name for p in OUTDIR.glob("baostock_daily_batch_*.parquet") if "partial" not in p.name
    )
    partials = sorted(p.name for p in OUTDIR.glob("*partial*"))
    print(f"  FINAL {len(finals)}/15: {finals}", flush=True)
    print(f"  partial {len(partials)}: {partials}", flush=True)
    st = OUTDIR / "pipeline_status.json"
    print(f"  status: {st.read_text(encoding='utf-8') if st.exists() else '(无)'}", flush=True)
else:
    print(f"  OUTDIR 不存在: {OUTDIR}", flush=True)

print("=" * 60, flush=True)
print("[3] v40 因子公式（32_derive_real_factors.py）", flush=True)
f32 = PROJ / "scripts" / "32_derive_real_factors.py"
if f32.exists():
    for i, line in enumerate(f32.read_text(encoding="utf-8").splitlines(), 1):
        if any(
            k in line
            for k in ("log_mktcap_real", "turnover_daily", "float_shares", "turn", "amount")
        ):
            print(f"  {i:4d}| {line}", flush=True)
else:
    print(f"  缺 {f32}", flush=True)

print("=" * 60, flush=True)
print("[4] v39 数据集特征（确认 log_amount 是否已在）", flush=True)
for meta in sorted(DATA.glob("dataset_h10_v39*meta*.json"))[:1]:
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
        feats = m.get("features") or m.get("feature_names") or m.get("cols")
        if feats:
            print(f"  {meta.name} 共 {len(feats)} 特征:", flush=True)
            print(f"  {feats}", flush=True)
            print(
                f"  → log_amount 已在? {'log_amount' in feats}; "
                f"turnover 类已在? "
                f"{[f for f in feats if 'turn' in f.lower() or 'amount' in f.lower()]}",
                flush=True,
            )
    except Exception as e:
        print(f"  meta 读取失败 {meta}: {e!r}", flush=True)
if not list(DATA.glob("dataset_h10_v39*meta*.json")):
    print(f"  未找到 v39 meta（在 {DATA}）", flush=True)

print("=" * 60, flush=True)
print("[5] v40 数据集是否存在", flush=True)
for p in sorted(DATA.glob("dataset_*v40*"))[:6]:
    print(f"  {p.name}  {p.stat().st_size/1e6:.1f}MB", flush=True)
