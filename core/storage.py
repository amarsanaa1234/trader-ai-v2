"""SQLite storage, one database per market (data/<market>/bot.db): candles, news,
trades, equity, events and small key/value state. A market adds its own
tables through Market.schema."""
import json
import sqlite3
import time

from .context import mkt

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT, tf TEXT, ts INTEGER, open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (symbol, tf, ts));
CREATE TABLE IF NOT EXISTS news (
    id TEXT PRIMARY KEY, ts INTEGER, source TEXT, title TEXT, summary TEXT, scored INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS sentiment (
    ts INTEGER, asset TEXT, score REAL, confidence REAL, reason TEXT);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, mode TEXT, symbol TEXT, side TEXT, qty REAL,
    entry REAL, stop REAL, tp REAL, risk_usd REAL, opened_ts INTEGER, opened_bar_ts INTEGER,
    status TEXT, closed_ts INTEGER, exit REAL, pnl REAL, r_multiple REAL, reward REAL,
    exit_reason TEXT, signal_json TEXT);
CREATE TABLE IF NOT EXISTS events (ts INTEGER, level TEXT, msg TEXT);
CREATE TABLE IF NOT EXISTS equity_log (ts INTEGER, mode TEXT, equity REAL, dopamine REAL);
CREATE TABLE IF NOT EXISTS carry (
    id INTEGER PRIMARY KEY AUTOINCREMENT, mode TEXT, symbol TEXT, qty REAL, spot_entry REAL, perp_entry REAL,
    opened_ts INTEGER, status TEXT, closed_ts INTEGER, funding REAL DEFAULT 0, fees REAL DEFAULT 0,
    pnl REAL, last_funding_ts INTEGER, note TEXT);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


def now_ms() -> int:
    return int(time.time() * 1000)


class Storage:
    def __init__(self, path=None):
        market = mkt()
        self.db = sqlite3.connect(path or market.db_path, timeout=30, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA + market.schema)
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(trades)")}
        for name, kind in (("dist", "REAL"), ("best", "REAL"), ("partial_done", "INTEGER DEFAULT 0"),
                           ("review", "TEXT"), ("regime", "INTEGER")):
            if name not in cols:
                self.db.execute(f"ALTER TABLE trades ADD COLUMN {name} {kind}")
        self.db.commit()

    # ---- key/value state ----
    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO state(key, value) VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()

    def event(self, level, msg):
        self.db.execute("INSERT INTO events VALUES (?, ?, ?)", (now_ms(), level, msg))
        self.db.commit()

    # ---- market data ----
    def last_candle_ts(self, symbol, tf):
        row = self.db.execute("SELECT MAX(ts) m FROM candles WHERE symbol=? AND tf=?", (symbol, tf)).fetchone()
        return row["m"]

    def insert_candles(self, symbol, tf, rows):
        self.db.executemany(
            "INSERT OR REPLACE INTO candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(symbol, tf, *r) for r in rows])
        self.db.commit()

    # ---- news ----
    def insert_news(self, rows):
        self.db.executemany(
            "INSERT OR IGNORE INTO news(id, ts, source, title, summary) VALUES (?, ?, ?, ?, ?)", rows)
        self.db.commit()

    def unscored_news(self, since_ms, limit=40):
        return self.db.execute(
            "SELECT * FROM news WHERE scored=0 AND ts>=? ORDER BY ts DESC LIMIT ?", (since_ms, limit)).fetchall()

    def mark_scored(self, ids):
        self.db.executemany("UPDATE news SET scored=1 WHERE id=?", [(i,) for i in ids])
        self.db.commit()

    def insert_sentiment(self, rows):
        self.db.executemany("INSERT INTO sentiment VALUES (?, ?, ?, ?, ?)", rows)
        self.db.commit()

    def recent_sentiment(self, asset, since_ms):
        return self.db.execute(
            "SELECT * FROM sentiment WHERE asset IN (?, 'MARKET') AND ts>=?", (asset, since_ms)).fetchall()

    # ---- trades ----
    def open_trades(self, mode):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM trades WHERE status='open' AND mode=?", (mode,)).fetchall()]

    def insert_trade(self, t: dict) -> int:
        cols = ", ".join(t.keys())
        qs = ", ".join("?" for _ in t)
        cur = self.db.execute(f"INSERT INTO trades ({cols}) VALUES ({qs})", list(t.values()))
        self.db.commit()
        return cur.lastrowid

    def update_trade(self, trade_id, **fields):
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE trades SET {sets} WHERE id=?", [*fields.values(), trade_id])
        self.db.commit()

    def trades_since(self, mode, since_ms):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM trades WHERE mode=? AND opened_ts>=? ORDER BY opened_ts", (mode, since_ms)).fetchall()]

    def log_equity(self, mode, equity, dopamine):
        self.db.execute("INSERT INTO equity_log VALUES (?, ?, ?, ?)", (now_ms(), mode, equity, dopamine))
        self.db.commit()

    def equity_history(self, mode, since_ms=0):
        return [dict(r) for r in self.db.execute(
            "SELECT ts, equity, dopamine FROM equity_log WHERE mode=? AND ts>=? ORDER BY ts", (mode, since_ms))]

    # ---- carry ----
    def open_carries(self, mode):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM carry WHERE status='open' AND mode=?", (mode,)).fetchall()]

    def insert_carry(self, c: dict) -> int:
        cols = ", ".join(c.keys())
        cur = self.db.execute(f"INSERT INTO carry ({cols}) VALUES ({', '.join('?' for _ in c)})", list(c.values()))
        self.db.commit()
        return cur.lastrowid

    def update_carry(self, carry_id, **fields):
        sets = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE carry SET {sets} WHERE id=?", [*fields.values(), carry_id])
        self.db.commit()

    def closed_trades(self, mode, limit=50):
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM trades WHERE mode=? AND status='closed' ORDER BY closed_ts DESC LIMIT ?",
            (mode, limit)).fetchall()]
