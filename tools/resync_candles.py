"""Drop one timeframe's candles of a market and download them again (e.g. after the
candle alignment changed).   python tools/resync_candles.py fx 4h"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import markets  # noqa: E402
from core.context import activate, mkt  # noqa: E402
from core.storage import Storage  # noqa: E402

market, tf = sys.argv[1], sys.argv[2]
activate(markets.get(market))
st = Storage()
n = st.db.execute("DELETE FROM candles WHERE tf=?", (tf,)).rowcount
st.db.commit()
print(f"{n} {tf} лаа устгав")
from markets.fx import data  # noqa: E402
for s in list(mkt().symbols) + list(getattr(mkt(), "aux_symbols", [])):
    print(s, data.sync_candles(st, s, tf, mkt().history_days))
print(st.db.execute("SELECT strftime('%H', ts/1000, 'unixepoch') h, COUNT(*) FROM candles WHERE tf=? "
                    "GROUP BY h", (tf,)).fetchall())
