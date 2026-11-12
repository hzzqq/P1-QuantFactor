"""P1 信号推进终检：四路源/ks md5 一致 + latest_date 一致且 > 10-21 + top 榜非空 + fusion 非空。

退出码 0 = 通过（新信号可保留）；非 0 = 不合格（应回滚）。
"""
import json
import hashlib
import sys
from pathlib import Path

SRC = Path("E:/project/sj/data/P1/processed/signals")
KS = Path("E:/project/ks/StockSignal/data/p1_signals")
FOUR = [
    "signal_gru_h10.json",
    "signal_baseline_h10.json",
    "signal_ens_v39gru_w025_h10.json",
    "signal_ev_h10.json",
]


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    lds = {}
    for f in FOUR:
        a, b = SRC / f, KS / f
        if not a.exists() or not b.exists():
            print("FINAL MISSING", f, flush=True)
            return 1
        ma, mb = _md5(a), _md5(b)
        s = json.loads(b.read_text(encoding="utf-8"))
        ld = s.get("latest_date")
        lds[f] = ld
        tl = len(s.get("top_long", []))
        ts = len(s.get("top_short", []))
        print(
            "FINAL %s ld=%s tl=%d ts=%d src_ks_match=%s"
            % (f, ld, tl, ts, ma == mb),
            flush=True,
        )
        if not tl or not ts or ma != mb:
            return 1
    if len(set(lds.values())) != 1:
        print("FINAL LATEST_DATE_MISMATCH", lds, flush=True)
        return 1
    ld = str(list(lds.values())[0])
    if ld <= "2026-10-21":
        print("FINAL NO_PROGRESS latest_date=", ld, flush=True)
        return 1
    fus = KS / "signal_fusion_h10.json"
    if not fus.exists():
        print("FINAL MISSING fusion", flush=True)
        return 1
    s = json.loads(fus.read_text(encoding="utf-8"))
    tl = len(s.get("top_long", []))
    ts = len(s.get("top_short", []))
    print(
        "FINAL FUSION ld=%s tl=%d ts=%d" % (s.get("latest_date"), tl, ts),
        flush=True,
    )
    if not tl or not ts:
        return 1
    print("FINAL_SIGNAL_CHECK_OK latest_date=", ld, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
