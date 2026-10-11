#!/usr/bin/env bash
# P1 每日信号推进驱动脚本 v2（自动化用；T-251，2026-10-11）
# 步骤：重建面板+dataset_h10 → EV 数据集重建(新) → 并入 log_amount(v39/v48) →
#       GRU 仅推理 → EV predict-only(新，复用生产权重不重训) → baseline(v39) 2026 增补 →
#       baseline(原) 2026 增补 → 合并×2 → 集成 → 导出四路(gru/baseline/ens/ev) → 复制到 ks
# T-251 加固：每步 exit 码+心跳落盘（heartbeat），失败即停；bash zz_daily_push.sh [S0x] 可断点续跑。
# 前置：18_bridge_kline.py 先行刷新 K 线（cf399ccd 自动化职责）。
set -u
PY="E:/project/sj/env/Scripts/python.exe"
PROJ="E:/project/sj/P1-QuantFactor"
PD="E:/project/sj/data/P1/processed"
KS="E:/project/ks/StockSignal/data/p1_signals"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="E:/project/ks/StockSignal/logs/p1_daily_push_${TS}.log"
HB="E:/project/ks/StockSignal/logs/p1_push_heartbeat.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== P1 daily push START $TS ==="
cd "$PROJ" || { echo "cd PROJ fail"; exit 1; }

FROM="${1:-S01}"
need(){ [ "${1}" \> "$FROM" ] || [ "${1}" = "$FROM" ]; }
step(){ echo "-------------------------------------------------[ $1 ]-------------------------------------------------"; }
ok(){ echo "$(date '+%F %T') OK $1" >> "$HB"; }
ng(){ echo "$(date '+%F %T') FAIL $1" >> "$HB"; }

# [S01] 重建面板（来自已扩展的 kline）+ dataset_h10
if need S01; then
  step "S01/10 rebuild panel + dataset_h10"
  "$PY" scripts/02_build_features.py --rebuild-panel || { echo "STEP1 FAIL"; ng S01; exit 1; }
  ok S01
fi

# [S02] EV 数据集重建（T-244：读刷新后的 dataset_h10 + ks 牧羊人 CSV → 47 维，新增）
if need S02; then
  step "S02/10 rebuild EV dataset"
  "$PY" scripts/14_event_factor_iterate.py --build-only --horizon 10 || { echo "STEP2 FAIL"; ng S02; exit 1; }
  ok S02
fi

# [S03] 并入 log_amount(v39/v48)；先删过期缓存，强制从扩展后 kline 重建
if need S03; then
  step "S03/10 augment log_amount"
  rm -f "$PD/_log_amount.parquet"
  "$PY" scripts/27_add_logamount_factor.py || { echo "STEP3 FAIL"; ng S03; exit 1; }
  ok S03
fi

# [S04] GRU 仅推理（不重训）
if need S04; then
  step "S04/10 gru predict-only"
  "$PY" scripts/zz_predict_only_gru.py --horizon 10 --seq-len 20 --threads 8 || { echo "STEP4 FAIL"; ng S04; exit 1; }
  ok S04
fi

# [S05] EV predict-only（T-244：生产权重 gru_ev_t10_seq40_h128_lr1e-3，不重训，新增）
if need S05; then
  step "S05/10 EV predict-only"
  "$PY" scripts/zz_predict_only_ev.py --threads 8 || { echo "STEP5 FAIL"; ng S05; exit 1; }
  ok S05
fi

# [S06] baseline v39 —— 仅跑 2026，落 part 后合并进 consolidated
if need S06; then
  step "S06/10 baseline v39 (2026 only)"
  "$PY" scripts/28_retrain_v39_baseline.py --test-years 2026 \
      --out-part pred_baseline_h10_v39_2026_new.parquet \
      --yearly-part yearly_v39_2026_new.json || { echo "STEP6 FAIL"; ng S06; exit 1; }
  "$PY" - <<'PYEOF' || { echo "STEP6 MERGE FAIL"; ng S06m; exit 1; }
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
  ok S06
fi

# [S07] baseline 原版(38因子) —— 用 28 指向 dataset_h10，仅跑 2026，合并
if need S07; then
  step "S07/10 baseline original (2026 only)"
  "$PY" scripts/28_retrain_v39_baseline.py --ds dataset_h10.parquet --meta dataset_h10_meta.json \
      --test-years 2026 --out-part pred_baseline_h10_2026_new.parquet \
      --yearly-part yearly_2026_new.json || { echo "STEP7 FAIL"; ng S07; exit 1; }
  "$PY" - <<'PYEOF' || { echo "STEP7 MERGE FAIL"; ng S07m; exit 1; }
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
  ok S07
fi

# [S08] 集成预测
if need S08; then
  step "S08/10 build ensemble pred"
  "$PY" scripts/37_build_ensemble_pred.py || { echo "STEP8 FAIL"; ng S08; exit 1; }
  ok S08
fi

# [S09] 导出信号（gru / baseline / ens / ev 四路）
if need S09; then
  step "S09/10 export signals"
  "$PY" scripts/06_export_signal.py --model gru --horizon 10 || { echo "STEP9a FAIL"; ng S09a; exit 1; }
  "$PY" scripts/06_export_signal.py --model baseline --horizon 10 || { echo "STEP9b FAIL"; ng S09b; exit 1; }
  "$PY" - <<'PYEOF' || { echo "STEP9c FAIL"; ng S09c; exit 1; }
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
  "$PY" scripts/06_export_signal.py --pred "$PD/pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet" \
      --model ev --horizon 10 || { echo "STEP9d FAIL"; ng S09d; exit 1; }
  ok S09
fi

# [S10] 复制到 ks p1_signals + latest_date 终检
if need S10; then
  step "S10/10 copy to ks p1_signals"
  cp -f "$PD/signals/signal_gru_h10.json" "$KS/" || { echo "COPY gru FAIL"; ng S10; exit 1; }
  cp -f "$PD/signals/signal_baseline_h10.json" "$KS/" || { echo "COPY baseline FAIL"; ng S10; exit 1; }
  cp -f "$PD/signals/signal_ens_v39gru_w025_h10.json" "$KS/" || { echo "COPY ens FAIL"; ng S10; exit 1; }
  cp -f "$PD/signals/signal_ev_h10.json" "$KS/" || { echo "COPY ev FAIL"; ng S10; exit 1; }
  "$PY" - <<'PYEOF' || { echo "VERIFY FAIL"; ng S10v; exit 1; }
import json
from pathlib import Path
KS = Path('E:/project/ks/StockSignal/data/p1_signals')
lds = {}
for f in ['signal_gru_h10.json', 'signal_baseline_h10.json',
          'signal_ens_v39gru_w025_h10.json', 'signal_ev_h10.json']:
    sig = json.load(open(KS/f, encoding='utf-8'))
    lds[f] = sig['latest_date']
    print('[verify]', f, 'latest_date=', sig['latest_date'],
          '| top_long=', len(sig.get('top_long', [])), '| top_short=', len(sig.get('top_short', [])))
    assert sig.get('top_long') and sig.get('top_short'), f + ' top 榜为空'
uniq = sorted(set(lds.values()))
assert len(uniq) == 1, '四路 latest_date 不一致: %s' % lds
print('[verify] 四路 latest_date 一致 =', uniq[0])
PYEOF
  ok S10
fi

echo "=== P1 daily push DONE $(date) ==="
ok "ALL_DONE $TS"
