"""阶段 51：P1 信号 → 券商模拟盘适配器（QMT / miniQMT / xt_trader）。

把 P1 产出的 signal_*.json 转换成任意券商模拟盘（miniQMT/xt_trader）可消费的
目标持仓文件（CSV + JSON）。沙箱无券商 API，故本脚本：
  1) 始终落盘 orders.csv / target_positions.json（可直接被你的下单脚本读取）；
  2) 若运行环境装了 xt_trader/xttype，可选地推送（--push，需你机器有 miniQMT）；
  3) 不依赖 StockTrader 仓库源码，是通用桥接，真跑在你本机。

A 股零售模拟盘一般不能对单股做空，故默认 long_only（只取 top_long）；
top_short 作为「对冲候选」单列输出，加 --allow-short 才生成做空委托（仅当账户支持）。

用法：
  python scripts/51_signal_to_broker.py --signal data/P1/processed/signals/signal_ens_v39gru_w025_h10.json
  python scripts/51_signal_to_broker.py --signal ... --top-k 10 --allow-short --push
"""
from __future__ import annotations
import argparse
import json
import sys
import pathlib

ROOT = pathlib.Path(r"E:/project/sj"); PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.logging_utils import get_logger
logger = get_logger("P1.broker_bridge")

PROC = ROOT / "data" / "P1" / "processed"
SIGNAL_DIR = PROC / "signals"
OUT_DIR = PROC / "broker_orders"


def code_to_broker(symbol: str) -> tuple[str, str]:
    """sh600165 -> ('600165', 'SH')；sz300684 -> ('300684', 'SZ')。"""
    s = symbol.lower()
    if s.startswith("sh"):
        return s[2:], "SH"
    if s.startswith("sz"):
        return s[2:], "SZ"
    if s.startswith("bj"):
        return s[2:], "BJ"
    return s, "SH"


def main() -> int:
    ap = argparse.ArgumentParser(description="P1 信号 → 券商模拟盘适配器")
    ap.add_argument("--signal", type=str, default=None,
                    help="信号 JSON 路径；默认取 signal_ens_v39gru_w025_h10.json")
    ap.add_argument("--top-k", type=int, default=20,
                    help="多/空各取前 K 名（默认 20）")
    ap.add_argument("--mode", choices=["long_only", "long_short"], default="long_only",
                    help="long_only=只做多 top_long；long_short=额外生成 top_short 做空委托")
    ap.add_argument("--allow-short", action="store_true",
                    help="允许生成做空委托（仅当券商账户支持融券/做空时生效）")
    ap.add_argument("--out", type=str, default=None, help="输出目录")
    ap.add_argument("--push", action="store_true",
                    help="尝试推送至 miniQMT（需本机安装 xt_trader/xttype 且已登录）")
    args = ap.parse_args()

    sig_path = pathlib.Path(args.signal) if args.signal else \
        SIGNAL_DIR / "signal_ens_v39gru_w025_h10.json"
    if not sig_path.exists():
        logger.error("信号文件不存在: %s", sig_path); return 1
    sig = json.loads(sig_path.read_text(encoding="utf-8"))

    top_long = (sig.get("top_long") or [])[: args.top_k]
    top_short = (sig.get("top_short") or [])[: args.top_k]
    k = max(len(top_long), 1)
    w = 1.0 / k  # 等权

    orders = []
    for r in top_long:
        code, mkt = code_to_broker(r["symbol"])
        orders.append({
            "code": code, "market": mkt, "full": f"{code}.{mkt}",
            "action": "BUY", "side": "LONG", "weight": round(w, 4),
            "score": round(r.get("pred", 0.0), 4),
            "name": r.get("name", ""),
        })
    short_orders = []
    if args.mode == "long_short" and args.allow_short:
        ks = max(len(top_short), 1)
        ws = 1.0 / ks
        for r in top_short:
            code, mkt = code_to_broker(r["symbol"])
            o = {
                "code": code, "market": mkt, "full": f"{code}.{mkt}",
                "action": "SELL", "side": "SHORT", "weight": round(ws, 4),
                "score": round(r.get("pred", 0.0), 4),
                "name": r.get("name", ""),
            }
            orders.append(o); short_orders.append(o)
    elif top_short:
        # 默认不生成做空委托，但把 top_short 作为对冲候选记录
        logger.info("top_short 共 %s 只作为对冲候选单列（未生成做空委托；"
                    "需 --mode long_short --allow-short 才下单）", len(top_short))

    OUT_DIR = pathlib.Path(args.out) if args.out else (PROC / "broker_orders")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) CSV（可被任意下单脚本 pd.read_csv 消费）
    import csv
    csv_path = OUT_DIR / "orders.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=["code", "market", "full", "action",
                                            "side", "weight", "score", "name"])
        wri.writeheader()
        for o in orders:
            wri.writerow(o)

    # 2) JSON（目标持仓，适合 QMT target_pos 风格程序读取）
    target = {
        "source_signal": str(sig_path),
        "generated_at": sig.get("generated_at"),
        "latest_date": sig.get("latest_date"),
        "mode": args.mode,
        "long_only": [o for o in orders if o["side"] == "LONG"],
        "short_only": short_orders,
        "n_long": len(top_long),
        "n_short": len(short_orders),
    }
    json_path = OUT_DIR / "target_positions.json"
    json_path.write_text(json.dumps(target, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"信号: {sig_path}")
    print(f"模式: {args.mode} | 做多 {len(top_long)} 只"
          f"{f' / 做空 {len(short_orders)} 只' if short_orders else ''}")
    print(f"等权权重: {w:.4f} / 只")
    print(f"多仓前 5: " + ", ".join(f"{o['full']}({o['score']:.2f})"
                                     for o in orders[:5] if o['side'] == 'LONG'))
    print(f"落盘: {csv_path}\n      {json_path}")

    if args.push:
        try:
            import xt_trader  # type: ignore
            import xttype    # type: ignore
            logger.warning("检测到 xt_trader，推送逻辑需你按账户配置填充"
                           "（此处仅占位，避免误下单）")
            # 真实推送需：session_id / 资金账号 / 调用 xt_trader.AssetTrader；
            # 见 docs/signal_to_broker.md。本脚本默认只落盘，不自动下单。
            print("[push] 已检测到 xt_trader，但为安全默认不自动下单；"
                  "请按 docs/signal_to_broker.md 在人工确认后调用。")
        except Exception as e:
            logger.warning("未检测到 xt_trader（沙箱/未安装）：%s", str(e)[:80])
            print("[push] 本机未安装 xt_trader，已跳过真实推送；"
                  "文件已落盘，可在 miniQMT 机器上读取。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
