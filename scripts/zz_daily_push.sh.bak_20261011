#!/usr/bin/env bash
# P1 每日信号推进驱动脚本（自动化用）
# 步骤：重建面板 → 重建 dataset_h10 → 并入 log_amount(v39) → GRU 仅推理 →
#       baseline(v39) 2026 增补 → baseline(原) 2026 增补 → 合并 → 集成 → 导出 → 复制到 ks
set -u
PY="E:/project/sj/env/Scripts/python.exe"
PROJ="E:/project/sj/P1-QuantFactor"
PD="E:/project/sj/data/P1/processed"
KS="E:/project/ks/StockSignal/data/p1_signals"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="E:/project/ks/StockSignal/logs/p1_daily_push_${TS}.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== P1 daily push START $TS ==="
cd "$PROJ" || { echo "cd PROJ fail"; exit 1; }

step(){ echo "-------------------------------------------------[ $1 ]-------------------------------------------------"; }

# [1] 重建面板（来自已扩展的 kline）+ dataset_h10
step "1/8 rebuild panel + dataset_h10"
"$PY" scripts/02_build_features.py --rebuild-panel || { echo "STEP1 FAIL"; exit 1; }

# [2] 并入 log_amount(v39)；先删过期缓存，强制从扩展后 kline 重建
step "2/8 augment log_amount (v39) -- rebuild cache"
rm -f "$PD/_log_amount.parquet"
"$PY" scripts/27_add_logamount_factor.py || { echo "STEP2 FAIL"; exit 1; }

# [3] GRU 仅推理（不重训）
step "3/8 gru predict-only"
"$PY" scripts/zz_predict_only_gru.py --horizon 10 --seq-len 20 --threads 8 || { echo "STEP3 FAIL"; exit 1; }

# [4] baseline v39 —— 仅跑 2026，落 part 后合并进 consolidated
step "4/8 baseline v39 (2026 only)"
"$PY" scripts/28_retrain_v39_baseline.py --test-years 2026 \
    --out-part pred_baseline_h10_v39_2026_new.parquet \
    --yearly-part yearly_v39_2026_new.json || { echo "STEP4 FAIL"; exit 1; }
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, 'E:/project/sj'); sys.path.insert(0, 'E:/project/sj/P1-QuantFactor')
import pandas as pd
from pathlib import Path
PD = Path('E:/project/sj/data/P1/processed')
old = pd.read_parquet(PD/'pred_baseline_h10_v39.parquet')
new = pd.read_parquet(PD/'pred_baseline_h10_v39_2026_new.parquet')
old_pre = old[pd.to_datetime(old['date']).dt.year < 2026]
merged = pd.concat([old_pre, new], ignore_index=True).drop_duplicates(subset=['date','symbol'], keep='last')
merged.to_parquet(PD/'pred_baseline_h10_v39.parquet', index=False)
print('baseline_v39 merged rows', len(merged), 'datemax', pd.to_datetime(merged['date']).max().date())
PYEOF

# [5] baseline 原版(38因子) —— 用 28 指向 dataset_h10，仅跑 2026，合并
step "5/8 baseline original (2026 only)"
"$PY" scripts/28_retrain_v39_baseline.py --ds dataset_h10.parquet --meta dataset_h10_meta.json \
    --test-years 2026 --out-part pred_baseline_h10_2026_new.parquet \
    --yearly-part yearly_2026_new.json || { echo "STEP5 FAIL"; exit 1; }
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, 'E:/project/sj'); sys.path.insert(0, 'E:/project/sj/P1-QuantFactor')
import pandas as pd
from pathlib import Path
PD = Path('E:/project/sj/data/P1/processed')
old = pd.read_parquet(PD/'pred_baseline_h10.parquet')
new = pd.read_parquet(PD/'pred_baseline_h10_2026_new.parquet')
old_pre = old[pd.to_datetime(old['date']).dt.year < 2026]
merged = pd.concat([old_pre, new], ignore_index=True).drop_duplicates(subset=['date','symbol'], keep='last')
merged.to_parquet(PD/'pred_baseline_h10.parquet', index=False)
print('baseline_original merged rows', len(merged), 'datemax', pd.to_datetime(merged['date']).max().date())
PYEOF

# [6] 集成预测
step "6/8 build ensemble pred"
"$PY" scripts/37_build_ensemble_pred.py || { echo "STEP6 FAIL"; exit 1; }

# [7] 导出信号（gru / baseline / ens）
step "7/8 export signals"
"$PY" scripts/06_export_signal.py --model gru --horizon 10 || { echo "STEP7a FAIL"; exit 1; }
"$PY" scripts/06_export_signal.py --model baseline --horizon 10 || { echo "STEP7b FAIL"; exit 1; }
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, 'E:/project/sj'); sys.path.insert(0, 'E:/project/sj/P1-QuantFactor')
import pandas as pd
from pathlib import Path
from src.signal import export_to_file
PD = Path('E:/project/sj/data/P1/processed')
ens = pd.read_parquet(PD/'pred_ens_v39gru_w025_h10.parquet')
out = PD/'signals'/'signal_ens_v39gru_w025_h10.json'
sig = export_to_file(ens, out, top_n=20, model='ens_v39gru_w025', horizon=10)
print('ens signal latest_date', sig['latest_date'], 'n_long', len(sig['top_long']), 'n_short', len(sig['top_short']))
PYEOF

# [8] 复制到 ks p1_signals
step "8/8 copy to ks p1_signals"
cp -f "$PD/signals/signal_gru_h10.json" "$KS/" || { echo "COPY gru FAIL"; exit 1; }
cp -f "$PD/signals/signal_baseline_h10.json" "$KS/" || { echo "COPY baseline FAIL"; exit 1; }
cp -f "$PD/signals/signal_ens_v39gru_w025_h10.json" "$KS/" || { echo "COPY ens FAIL"; exit 1; }
echo "=== P1 daily push DONE $(date) ==="
