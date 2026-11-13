#!/usr/bin/env bash
# Tier-1: 刷新 h20 baseline 数据集 + 重训 + 导出并部署到 ks。
# 背景运行；日志见 data/P1/processed/h20_refresh_log.txt
export PATH="/usr/bin:/bin:/usr/local/bin:$PATH"
export H20_KEEP_TAIL=1   # 27b 保留尾部无标签行，供 28 --live 出最新日(2026-11-09)预测
PY=~/.workbuddy/binaries/python/envs/default/Scripts/python.exe
ROOT=E:/project/sj
PROJ=$ROOT/P1-QuantFactor
PROC=$ROOT/data/P1/processed
KS=E:/project/ks/StockSignal/data/p1_signals
LOG=$PROC/h20_refresh_log.txt
exec > >(tee -a "$LOG") 2>&1

echo "=== [$(date)] h20 baseline refresh START ==="

# 0) 备份旧产物（防回滚）
cp -p "$PROC/dataset_h20_v39.parquet"        "$PROC/dataset_h20_v39.parquet.bak_20261113" 2>/dev/null || echo "WARN: dataset_h20 backup skip"
cp -p "$PROC/pred_baseline_h20_v39.parquet"  "$PROC/pred_baseline_h20_v39.parquet.bak_20261113" 2>/dev/null || echo "WARN: pred_h20 backup skip"

# 1) 27b: 基于当前 dataset_h10_v39(→2026-11-09) 重建 dataset_h20_v39
echo "=== [$(date)] 27b rebuild dataset_h20_v39 ==="
$PY "$PROJ/scripts/27b_build_h20_labels.py" || { echo "FATAL 27b"; exit 1; }

# 2) 28: 重训 h20 baseline（--live 保留尾部无标签交易日供最新日预测）
#    分三片（含此前缺失的 2022），各片独立以便容错
echo "=== [$(date)] 28 retrain h20 partA 2019-2022 (live) ==="
$PY "$PROJ/scripts/28_retrain_v39_baseline.py" --horizon 20 --ds dataset_h20_v39.parquet --meta dataset_h20_v39_meta.json \
    --test-years 2019,2020,2021,2022 --label-horizon 20 --live \
    --out-part pred_baseline_h20_v39_refA.parquet --yearly-part yearly_h20_v39_refA.json \
    || echo "WARN: partA failed"
echo "=== [$(date)] 28 retrain h20 partB 2023-2024 (live) ==="
$PY "$PROJ/scripts/28_retrain_v39_baseline.py" --horizon 20 --ds dataset_h20_v39.parquet --meta dataset_h20_v39_meta.json \
    --test-years 2023,2024 --label-horizon 20 --live \
    --out-part pred_baseline_h20_v39_refB.parquet --yearly-part yearly_h20_v39_refB.json \
    || echo "WARN: partB failed"
echo "=== [$(date)] 28 retrain h20 partC 2025-2026 (live) ==="
$PY "$PROJ/scripts/28_retrain_v39_baseline.py" --horizon 20 --ds dataset_h20_v39.parquet --meta dataset_h20_v39_meta.json \
    --test-years 2025,2026 --label-horizon 20 --live \
    --out-part pred_baseline_h20_v39_refC.parquet --yearly-part yearly_h20_v39_refC.json \
    || echo "WARN: partC failed"

# 3) 合并三片 → 最终 pred_baseline_h20_v39.parquet
echo "=== [$(date)] consolidate ==="
$PY - <<'PYEOF'
import pandas as pd, os
PROC="E:/project/sj/data/P1/processed"
parts=["pred_baseline_h20_v39_refA.parquet","pred_baseline_h20_v39_refB.parquet","pred_baseline_h20_v39_refC.parquet"]
dfs=[pd.read_parquet(os.path.join(PROC,p)) for p in parts if os.path.exists(os.path.join(PROC,p))]
if not dfs:
    raise SystemExit("FATAL: no h20 parts produced")
df=pd.concat(dfs,ignore_index=True)
# 去重（同 date,symbol 仅保留一条，按 year 升序后末次覆盖）
df=df.drop_duplicates(subset=["date","symbol"], keep="last")
df.to_parquet(os.path.join(PROC,"pred_baseline_h20_v39.parquet"),index=False)
print("consolidated",len(df),"rows; years",sorted(df["year"].dropna().unique().tolist()),"date max",df["date"].max())
PYEOF

# 4) 导出信号（独立 model key=baseline20，避免与 ks 现有 baseline(h10) 撞 key 覆盖生产信号）
echo "=== [$(date)] 06 export signal_baseline20_h20 ==="
$PY "$PROJ/scripts/06_export_signal.py" --pred "$PROC/pred_baseline_h20_v39.parquet" --model baseline20 --horizon 20 --top-n 20 \
    || { echo "FATAL export"; exit 1; }

# 5) 部署到 ks（自托管目录，优先于 fallback 被加载器发现）
echo "=== [$(date)] copy to ks ==="
cp -p "$PROC/signals/signal_baseline20_h20.json" "$KS/signal_baseline20_h20.json" \
    || { echo "FATAL copy"; exit 1; }
$PY - <<'PYEOF'
import json,os
p="E:/project/ks/StockSignal/data/p1_signals/signal_baseline20_h20.json"
d=json.load(open(p,encoding="utf-8"))
print("ks signal OK: model=%s horizon=%s latest=%s top_long=%d top_short=%d daily=%d size=%dKB"
      %(d.get("model"),d.get("horizon"),d.get("latest_date"),
        len(d.get("top_long",[])),len(d.get("top_short",[])),len(d.get("daily",[])),os.path.getsize(p)//1024))
PYEOF

echo "=== [$(date)] h20 baseline refresh DONE ==="
