"""Telegram commands from phones. A message is obeyed only if it comes from a
chat that belongs to a configured user (USER<n>_TELEGRAM_CHAT_ID); each person's
commands act on their own account only. Admins have a few extra commands."""
import json
import logging
import os
import sqlite3
import time

import requests

import markets

from . import users as users_mod
from .context import mkt
from .settings import DATA_ROOT, TELEGRAM_BOT_TOKEN

log = logging.getLogger(__name__)

HELP = """Commands:
/status – balance, positions, dopamine
/stop – stop opening new trades (open ones keep their stop/target)
/resume – start opening trades again
/closeall – close every position (asks you to confirm)
/report – 7-day report
/resetpeak – re-enable trading after the 30% drawdown stop (check first)
/help – this list
One market only: add its name, e.g. /stop fx, /status crypto (no name = every market)"""

ADMIN_HELP = """
👑 Admin:
/users – everyone and their accounts
/emergency – stop and close EVERYONE's trading (asks you to confirm)"""


def enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN)


# ---- one Telegram poller for all market processes ----
# Telegram allows only one getUpdates reader per bot. Each market runs as its own
# process, so they share a small hub database: whoever holds the lease reads
# Telegram and files every command into the inbox; every market process then
# reads the inbox itself. "/stop fx" acts on one market, "/stop" on all of them.
HUB_DB = DATA_ROOT / "telegram.db"
LEASE_SEC = 60


def _hub():
    db = sqlite3.connect(HUB_DB, timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS inbox (id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER, text TEXT, ts INTEGER);
        CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);""")
    return db


def _kv(db, key, default=None):
    row = db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def _set(db, key, value):
    db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))


def _take_lease(db, me: str) -> bool:
    now = time.time()
    with db:
        lease = _kv(db, "poller")
        if lease and lease["owner"] != me and lease["until"] > now:
            return False
        _set(db, "poller", {"owner": me, "until": now + LEASE_SEC})
    return True


def _fetch_into_inbox(db):
    offset = _kv(db, "offset", 0)
    try:
        r = requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                         params={"offset": offset, "timeout": 0}, timeout=15)
        r.raise_for_status()
        updates = r.json().get("result", [])
    except Exception as e:
        log.debug("telegram poll: %s", e)
        return
    rows = []
    for u in updates:
        offset = max(offset, u["update_id"] + 1)
        msg = u.get("message") or {}
        text = (msg.get("text") or "").strip()
        if text.startswith("/"):
            rows.append((msg.get("chat", {}).get("id"), text, int(time.time() * 1000)))
    with db:
        db.executemany("INSERT INTO inbox (chat_id, text, ts) VALUES (?, ?, ?)", rows)
        _set(db, "offset", offset)
        db.execute("DELETE FROM inbox WHERE ts < ?", (int((time.time() - 7 * 86400) * 1000),))


def parse(text: str, market: str):
    """'/stop@MyBot fx' -> '/stop' for fx; None if it targets another market."""
    parts = text.strip().lower().split()
    cmd = parts[0].split("@")[0]
    if len(parts) > 1 and parts[1] in markets.AVAILABLE and parts[1] != market:
        return None
    return cmd


def poll(st) -> list[tuple]:
    """New commands for this market since the last poll (non-blocking): [(User, "/command"), ...]."""
    if not enabled():
        return []
    db = _hub()
    try:
        if _take_lease(db, f"{mkt().name}:{os.getpid()}"):
            _fetch_into_inbox(db)
        last = st.get("tg_seen")
        if last is None:   # first start: don't replay old commands
            last = db.execute("SELECT COALESCE(MAX(id), 0) m FROM inbox").fetchone()["m"]
        rows = db.execute("SELECT id, chat_id, text FROM inbox WHERE id > ? ORDER BY id", (last,)).fetchall()
    finally:
        db.close()
    everyone = users_mod.load_users()
    cmds = []
    for r in rows:
        last = r["id"]
        who = users_mod.by_chat(everyone, r["chat_id"])
        cmd = parse(r["text"], mkt().name) if who else None
        if cmd:                      # strangers and other markets' commands are ignored
            cmds.append((who, cmd))
    st.set("tg_seen", last)
    return cmds
