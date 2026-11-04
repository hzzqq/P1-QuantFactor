# -*- coding: utf-8 -*-
"""20_export_findings.py — 导出毕设三算法诚实证据 → experiment_findings.json。

职责：把已落盘的真实 head-to-head 报告数字（非硬编码）映射为 StockSignal 侧
``modules/experiment_improvements.load_live_findings`` 可消费的合并 JSON：

    <PROCESSED>/experiment_findings.json  =  {key: {...}, ...}

数据来源（全部真实文件，缺失即跳过该 key，绝不编造）：
  - report_gnn_vs_gru_full.json → transformer_signal / gnn_factor
    （compare 数组：TransformerSignal / GRUAttention / FactorGraphGCN 三行）
  - report_rl_position.json     → rl_position（overall: w_rl / w_sign / w_tanh）

运行：python scripts/20_export_findings.py
输出路径与 06_export_signal.py 同目录约定（shared.paths.DATA / P1 / processed）。
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # E:/project/sj（shared 所在层）
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.paths import DATA  # noqa: E402  (E:/project/sj/shared/paths.py)

PROCESSED = DATA / "P1" / "processed"


def _load(name: str) -> dict | None:
    fp = PROCESSED / name
    if not fp.is_file():
        print(f"[skip] {name} 不存在")
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[skip] {name} 解析失败: {e}")
        return None


def _num(v):
    return float(v) if isinstance(v, (int, float)) else None


def _from_head_to_head(rep: dict) -> tuple[dict, dict]:
    """从 gnn_vs_gru_full 的 compare 数组提取 #1/#2（基线=GRUAttention 行）。"""
    rows = {r.get("model"): r for r in rep.get("compare", [])}
    base = rows.get("GRUAttention") or {}
    dhb, icb, icirb = _num(base.get("direction_hit")), _num(base.get("ic_mean")), _num(base.get("icir"))
    n_base = base.get("n")
    prov_base = f"GRUAttention 基线行 (n={n_base})" if n_base else "GRUAttention 基线行"

    out = {}
    t = rows.get("TransformerSignal")
    if t and _num(t.get("direction_hit")) is not None and dhb is not None:
        out["transformer_signal"] = {
            "direction_hit": _num(t.get("direction_hit")),
            "direction_hit_baseline": dhb,
            "ic": _num(t.get("ic_mean")),
            "ic_baseline": icb,
            "icir": _num(t.get("icir")),
            "icir_baseline": icirb,
            "provenance": f"report_gnn_vs_gru_full.json TransformerSignal (n={t.get('n')}) + {prov_base}",
        }
    g = rows.get("FactorGraphGCN")
    if g and _num(g.get("direction_hit")) is not None and dhb is not None:
        out["gnn_factor"] = {
            "direction_hit": _num(g.get("direction_hit")),
            "direction_hit_baseline": dhb,
            "ic": _num(g.get("ic_mean")),
            "ic_baseline": icb,
            "icir": _num(g.get("icir")),
            "icir_baseline": icirb,
            "provenance": f"report_gnn_vs_gru_full.json FactorGraphGCN (n={g.get('n')}) + {prov_base}",
        }
    return out


def _from_rl(rep: dict) -> dict:
    """从 rl_position 的 overall 提取 #3（w_rl vs w_sign/w_tanh）。"""
    ov = rep.get("overall") or {}
    rl, sign, tanh = ov.get("w_rl") or {}, ov.get("w_sign") or {}, ov.get("w_tanh") or {}
    sh = _num(rl.get("sharpe"))
    if sh is None:
        return {}
    out = {"sharpe": sh}
    if _num(sign.get("sharpe")) is not None:
        out["sharpe_baseline_sign"] = _num(sign.get("sharpe"))
    if _num(tanh.get("sharpe")) is not None:
        out["sharpe_baseline_tanh"] = _num(tanh.get("sharpe"))
    if _num(rep.get("avg_turnover_rl")) is not None:
        out["turnover"] = _num(rep.get("avg_turnover_rl"))
    out["provenance"] = "report_rl_position.json overall (w_rl vs w_sign/w_tanh)"
    return {"rl_position": out}


def main() -> int:
    findings: dict = {}
    rep_h2h = _load("report_gnn_vs_gru_full.json")
    if rep_h2h:
        findings.update(_from_head_to_head(rep_h2h))
    rep_rl = _load("report_rl_position.json")
    if rep_rl:
        findings.update(_from_rl(rep_rl))

    if not findings:
        print("[fail] 无任何真实报告可导出，拒绝写出空/编造文件")
        return 1

    findings["_meta"] = {
        "exported_at": date.today().isoformat(),
        "exporter": "scripts/20_export_findings.py",
        "sources": ["report_gnn_vs_gru_full.json", "report_rl_position.json"],
    }
    out_fp = PROCESSED / "experiment_findings.json"
    out_fp.write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {out_fp} keys={sorted(k for k in findings if k != '_meta')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
