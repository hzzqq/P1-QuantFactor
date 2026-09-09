"""P1 → ONNX 导出：把训练好的 GRUAttention 权重导出为 ONNX，供 P5 部署。

用法：
    python scripts/07_export_onnx.py \
        --checkpoint models/P1/gru_best_h10_2026.pt \
        --out models/P5/gru_p1_h10.onnx
"""
from __future__ import annotations

import argparse
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # E:/project/sj
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models.gru_attn import GRUAttention         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 GRU 导出 ONNX")
    ap.add_argument("--checkpoint", required=True,
                    help="训练好的权重 .pt（GRUAttention state_dict）")
    ap.add_argument("--out", required=True, help="输出 .onnx 路径")
    ap.add_argument("--n-features", type=int, default=38)
    ap.add_argument("--seq-len", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--attn-dim", type=int, default=32)
    args = ap.parse_args()

    model = GRUAttention(
        n_features=args.n_features, hidden=args.hidden,
        n_layers=args.n_layers, dropout=0.2, attn_dim=args.attn_dim,
    )
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    model.eval()

    dummy = torch.randn(1, args.seq_len, args.n_features)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(out),
        input_names=["input"], output_names=["score"],
        dynamic_axes={"input": {0: "batch"}, "score": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"已导出 P1 GRU ONNX: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
