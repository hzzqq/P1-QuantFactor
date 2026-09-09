import sys, time
sys.path.insert(0, r"E:/project/sj/P1-QuantFactor")
try:
    import baostock as bs
    print("baostock import OK, version:", getattr(bs, "__version__", "?"))
except Exception as e:
    print("IMPORT FAIL:", repr(e)); sys.exit(1)

t0 = time.time()
try:
    lg = bs.login()
    print("login:", lg.error_code, lg.error_msg, round(time.time() - t0, 2), "s")
except Exception as e:
    print("LOGIN FAIL:", repr(e)[:200]); sys.exit(1)
if lg.error_code != "0":
    print("LOGIN NOT OK, exit"); sys.exit(1)

t1 = time.time()
rs = bs.query_history_k_data_plus(
    "sh.600000",
    "date,code,open,high,low,close,volume,amount,turn,peTTM,pbMRQ",
    start_date="2024-01-01", end_date="2024-12-31", frequency="d",
)
print("query error_code:", rs.error_code, rs.error_msg)
rows = []
while (rs.error_code == "0") and rs.next():
    rows.append(rs.get_row_data())
print("fetched rows:", len(rows), "in", round(time.time() - t1, 2), "s")
if rows:
    print("sample:", rows[0])
    print("cols:", ["date", "code", "open", "high", "low", "close", "volume", "amount", "turn", "peTTM", "pbMRQ"])
bs.logout()
print("logout OK")
