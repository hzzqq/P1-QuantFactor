import sys, pathlib, socket, time

ROOT = pathlib.Path(r"E:/project/sj")
PROJ = ROOT / "P1-QuantFactor"
for _p in (str(ROOT), str(PROJ)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 1) pandas / numpy / engine import
try:
    import pandas as pd, numpy as np
    print("PANDAS_OK", pd.__version__, "NUMPY", np.__version__)
except Exception as e:
    print("PANDAS_FAIL", repr(e)); sys.exit(1)

try:
    from src.backtest import run_backtest, run_backtest_continuous, DEFAULT_COST
    print("ENGINE_OK")
except Exception as e:
    print("ENGINE_FAIL", repr(e))

# 2) panel columns (liquidity source?)
panel_path = ROOT / "data" / "P1" / "processed" / "panel.parquet"
if panel_path.exists():
    p = pd.read_parquet(panel_path)
    print("PANEL_ROWS", len(p), "COLS", list(p.columns))
else:
    print("PANEL_MISSING", str(panel_path))

# 3) predictions present?
pred_dir = ROOT / "data" / "P1" / "processed"
for nm in ("pred_ens_v39gru_w025_h10.parquet", "pred_baseline_h10.parquet"):
    fp = pred_dir / nm
    print("PRED", nm, "EXISTS" if fp.exists() else "MISSING")

# 4) Tencent reachability (for C)
host = "web.ifzq.gtimg.cn"
t0 = time.time()
try:
    ip = socket.gethostbyname(host)
    s = socket.create_connection((host, 80), timeout=5)
    s.close()
    print("TENCENT_REACHABLE", ip, f"{time.time()-t0:.2f}s")
except Exception as e:
    print("TENCENT_UNREACHABLE", repr(e))
