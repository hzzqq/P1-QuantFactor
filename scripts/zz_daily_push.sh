#!/usr/bin/env bash
# P1 每日信号推进驱动脚本 v6（自动化用；T-251，2026-10-11；v3: T-352 增 S11 fusion 环节，2026-10-30；v4: 2026-11-01 增 S12 绩效实证刷新；v5: 2026-11-01 增 S13 敏感性矩阵刷新；v6: 2026-11-10 实时信号模式 live：S01 保留尾部无标签交易日，S04/S05/S06/S07 对最新日一并出分，latest_date 推至真实最新交易日 11-09；最末日为模型预测、尚无实现收益验证）
# 诚实边界：fusion(S11) 用前瞻 y_excess 标签，仍滞后于四路(设计性)，不参与 latest_date 一致性门禁。
# 步骤：重建面板+dataset_h10 → EV 数据集重建(新) → 并入 log_amount(v39/v48) →
#       GRU 仅推理 → EV predict-only(新，复用生产权重不重训) → baseline(v39) 2026 增补 →
#       baseline(原) 2026 增补 → 合并×2 → 集成 → 导出四路(gru/baseline/ens/ev) →
#       融合+导出 fusion(S11) → 复制到 ks（含 fusion 信号）→ 绩效实证刷新(S12) → 敏感性矩阵刷新(S13)
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
  step "S01/11 rebuild panel + dataset_h10 (live: 保留尾部无标签交易日)"
  "$PY" scripts/02_build_features.py --rebuild-panel --keep-unlabeled || { echo "STEP1 FAIL"; ng S01; exit 1; }
  ok S01
fi

# [S02] EV 数据集重建（T-244：读刷新后的 dataset_h10 + ks 牧羊人 CSV → 47 维，新增）
if need S02; then
  step "S02/11 rebuild EV dataset"
  "$PY" scripts/14_event_factor_iterate.py --build-only --horizon 10 || { echo "STEP2 FAIL"; ng S02; exit 1; }
  ok S02
fi

# [S03] 并入 log_amount(v39/v48)；先删过期缓存，强制从扩展后 kline 重建
if need S03; then
  step "S03/11 augment log_amount"
  rm -f "$PD/_log_amount.parquet"
  "$PY" scripts/27_add_logamount_factor.py || { echo "STEP3 FAIL"; ng S03; exit 1; }
  ok S03
fi

# [S04] GRU 仅推理（不重训）
if need S04; then
  step "S04/11 gru predict-only (live)"
  "$PY" scripts/zz_predict_only_gru.py --horizon 10 --seq-len 20 --threads 8 --live || { echo "STEP4 FAIL"; ng S04; exit 1; }
  ok S04
fi

# [S05] EV predict-only（T-244：生产权重 gru_ev_t10_seq40_h128_lr1e-3，不重训，新增）
if need S05; then
  step "S05/11 EV predict-only (live)"
  "$PY" scripts/zz_predict_only_ev.py --threads 8 --live || { echo "STEP5 FAIL"; ng S05; exit 1; }
  ok S05
fi

# [S06] baseline v39 —— 仅跑 2026，落 part 后合并进 consolidated
if need S06; then
  step "S06/11 baseline v39 (2026 only, live)"
  "$PY" scripts/28_retrain_v39_baseline.py --test-years 2026 --live \
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
  step "S07/11 baseline original (2026 only, live)"
  "$PY" scripts/28_retrain_v39_baseline.py --ds dataset_h10.parquet --meta dataset_h10_meta.json \
      --test-years 2026 --live --out-part pred_baseline_h10_2026_new.parquet \
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

# [S08] 集成预测（2026-11-01 拍板：--w-gru 0.00，ens 内 gru 权重下调至 0）
if need S08; then
  step "S08/11 build ensemble pred"
  "$PY" scripts/37_build_ensemble_pred.py --w-gru 0.00 || { echo "STEP8 FAIL"; ng S08; exit 1; }
  ok S08
fi

# [S09] 导出信号（gru / baseline / ens / ev 四路；ens 走 zz_export_ens.py，可 --exclude-gru-unique）
# 2026-11-11：gru 独立信号路近 59 日已反号（Rank-IC≈-0.177，显著为负），
# 但 S09a/S10 仍把它推给 ks 侧作「灰度参考」，有误导人工决策的风险。
# 处理 = 保留推送以维持四路 latest_date 一致性门禁，但固化 quality 标注：
# 「仅供研究、禁止用于投资决策」，让 ks 消费端可识别。证据见
# data/P1/processed/reports/recent_edge_diag_20261111.md 与 gru_weight_scan_20261111.md。
# 注：ens 集成内 gru 权重已于 2026-11-01 置 0，本独立路不并入 ens，不影响 ens 信号。
GRU_QUALITY='{"status":"research_only","usable_for_decision":false,"warning":"GRU 独立信号路近期已反号，仅供研究对照，禁止用于投资决策","evidence":{"window":"2026-08-11~2026-11-09(59交易日)","rank_ic_recent":-0.177,"ls_t_recent":-4.37,"reports":["reports/recent_edge_diag_20261111.md","reports/gru_weight_scan_20261111.md"]},"annotated_at":"2026-11-11","note":"ens 集成内 gru 权重已于 2026-11-01 置 0；本独立路未并入 ens，其反号不影响 ens 信号，但作为独立展示信号有误导风险"}'
if need S09; then
  step "S09/11 export signals"
  "$PY" scripts/06_export_signal.py --model gru --horizon 10 --quality-json "$GRU_QUALITY" || { echo "STEP9a FAIL"; ng S09a; exit 1; }
  "$PY" scripts/06_export_signal.py --model baseline --horizon 10 || { echo "STEP9b FAIL"; ng S09b; exit 1; }
  "$PY" scripts/zz_export_ens.py || { echo "STEP9c FAIL"; ng S09c; exit 1; }
  "$PY" scripts/06_export_signal.py --pred "$PD/pred_gru_ev_t10_seq40_h128_lr1e-3_h10.parquet" \
      --model ev --horizon 10 || { echo "STEP9d FAIL"; ng S09d; exit 1; }
  ok S09
fi

# [S10] 复制到 ks p1_signals + latest_date 终检
if need S10; then
  step "S10/11 copy to ks p1_signals"
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

# [S11] 融合信号（T-352 防复发：四路导出后再融合+导出+复制；fusion 依赖 y_excess 前瞻标签，latest_date 设计性滞后于四路，不要求一致）
if need S11; then
  step "S11/11 build + export fusion signal"
  "$PY" scripts/11_conditional_fusion.py || { echo "STEP11a FAIL"; ng S11a; exit 1; }
  "$PY" scripts/06_export_signal.py --model fusion || { echo "STEP11b FAIL"; ng S11b; exit 1; }
  cp -f "$PD/signals/signal_fusion_h10.json" "$KS/" || { echo "COPY fusion FAIL"; ng S11c; exit 1; }
  "$PY" - <<'PYEOF' || { echo "VERIFY fusion FAIL"; ng S11v; exit 1; }
import json
from pathlib import Path
KS = Path('E:/project/ks/StockSignal/data/p1_signals')
sig = json.load(open(KS/'signal_fusion_h10.json', encoding='utf-8'))
print('[verify] signal_fusion_h10.json latest_date=', sig['latest_date'],
      '| top_long=', len(sig.get('top_long', [])), '| top_short=', len(sig.get('top_short', [])))
assert sig.get('top_long') and sig.get('top_short'), 'fusion top 榜为空'
print('[verify] fusion latest_date 与四路存在 y_excess 设计性滞后，不要求一致')
PYEOF
  ok S11
fi

# [S12] 绩效实证刷新（2026-11-01：信号更新后自动重跑回溯模拟盘，top-k=10 全区间，
#   结果落 reports/paper_portfolio_*.json，供绩效跟踪基线；断点可 bash zz_daily_push.sh S12）
if need S12; then
  step "S12/12 refresh paper portfolio"
  "$PY" scripts/zz_paper_portfolio.py --top-k 10 || { echo "STEP12 FAIL"; ng S12; exit 1; }
  ok S12
fi

# [S13] 敏感性矩阵刷新（2026-11-01：k∈{5,10,19}×成本/涨跌停 4 组 + 动态对齐窗口汇总，
#   结果落 reports/P1敏感性矩阵_*.md/.json；断点可 bash zz_daily_push.sh S13）
if need S13; then
  step "S13/13 refresh sensitivity matrix"
  "$PY" scripts/zz_paper_sensitivity.py || { echo "STEP13 FAIL"; ng S13; exit 1; }
  ok S13
fi

echo "=== P1 daily push DONE $(date) ==="
ok "ALL_DONE $TS"
