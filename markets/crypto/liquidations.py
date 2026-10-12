"""Liquidation recorder (data collection only; the trading bot doesn't use it yet).

Binance publishes forced closes ("liquidations") of futures positions on a
public websocket, but keeps no history. This records them from now on into
data/liquidations.db, so in 1-2 months we can test whether liquidation
spikes are a useful signal.

Note: Binance sends at most one liquidation per symbol per second (the
largest), so totals are a lower bound, but spikes are still visible.
Runs as its own service (crypto-liq), separate from the trading bot.
"""
import asyncio
import json
import logging
import sqlite3
import time

import aiohttp

from core.settings import DATA_ROOT

log = logging.getLogger(__name__)
DB = DATA_ROOT / "crypto" / "liquidations.db"
URL = "wss://fstream.binance.com/market/ws/!forceOrder@arr"  # Binance moved market streams under /market
SCHEMA = """CREATE TABLE IF NOT EXISTS liquidations (
    ts INTEGER, symbol TEXT, liquidated TEXT, price REAL, qty REAL, usd REAL);
CREATE INDEX IF NOT EXISTS liq_ts ON liquidations(ts);
CREATE INDEX IF NOT EXISTS liq_sym_ts ON liquidations(symbol, ts);"""


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB, timeout=30)
    db.executescript(SCHEMA)
    return db


def parse(msg: dict):
    o = msg.get("o") or {}
    qty = float(o.get("z") or o.get("q") or 0)
    price = float(o.get("ap") or o.get("p") or 0)
    # a forced SELL closes a long position, a forced BUY closes a short
    liquidated = "long" if o.get("S") == "SELL" else "short"
    return int(o.get("T") or msg.get("E")), o.get("s", "?"), liquidated, price, qty, price * qty


async def _listen(db: sqlite3.Connection):
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(URL, heartbeat=60) as ws:
            log.info("Liquidation stream connected")
            batch, last_flush = [], time.time()
            async for m in ws:
                if m.type != aiohttp.WSMsgType.TEXT:
                    if m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    continue
                try:
                    batch.append(parse(json.loads(m.data)))
                except Exception as e:
                    log.debug("bad message: %s", e)
                if batch and (len(batch) >= 50 or time.time() - last_flush > 10):
                    db.executemany("INSERT INTO liquidations VALUES (?, ?, ?, ?, ?, ?)", batch)
                    db.commit()
                    batch, last_flush = [], time.time()


def run():
    """Forever: listen, store, reconnect with backoff (Binance drops streams every 24h)."""
    db = connect()
    delay = 5
    while True:
        started = time.time()
        try:
            asyncio.run(_listen(db))
        except Exception as e:
            log.warning("Liquidation stream error: %s", e)
        if time.time() - started > 300:
            delay = 5
        time.sleep(delay)
        delay = min(delay * 2, 300)


def summary(hours: int = 24) -> list[tuple]:
    """(symbol, long USD liquidated, short USD liquidated, count) for the last N hours, biggest first."""
    db = connect()
    since = int(time.time() * 1000) - hours * 3_600_000
    return db.execute("""SELECT symbol, SUM(CASE WHEN liquidated='long' THEN usd ELSE 0 END),
                                SUM(CASE WHEN liquidated='short' THEN usd ELSE 0 END), COUNT(*)
                         FROM liquidations WHERE ts >= ? GROUP BY symbol ORDER BY SUM(usd) DESC""",
                      (since,)).fetchall()
