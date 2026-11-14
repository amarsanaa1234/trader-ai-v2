"""The bot's messages. Each person gets their own account's messages; market-wide
notices (new strategy found, shadow promoted, errors) go to the admins.

Owner, 2026-11-15: Telegram is off (not used). Messages are kept in data/messages.db for
7 days and shown on the web ("📨 Мэдэгдэл"); older ones are deleted so it stays small."""
import logging
import sqlite3
import time

import requests

from . import users as users_mod
from .context import mkt
from .settings import DATA_ROOT, TELEGRAM_BOT_TOKEN

log = logging.getLogger(__name__)
_last = {}
MESSAGES_DB = DATA_ROOT / "messages.db"
KEEP_DAYS = 7
TELEGRAM_ON = False   # owner, 2026-11-15: Telegram not used any more


def _db():
    db = sqlite3.connect(MESSAGES_DB, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS msgs (id INTEGER PRIMARY KEY, ts INTEGER, market TEXT, "
               "user TEXT, text TEXT)")
    db.execute("CREATE INDEX IF NOT EXISTS msgs_ts ON msgs(ts)")
    return db


def store(market: str, user: str, text: str) -> None:
    """Keep one message for the web; drop anything older than KEEP_DAYS."""
    now = int(time.time() * 1000)
    try:
        db = _db()
        try:
            with db:
                db.execute("INSERT INTO msgs (ts, market, user, text) VALUES (?, ?, ?, ?)",
                           (now, market, user, text))
                db.execute("DELETE FROM msgs WHERE ts < ?", (now - KEEP_DAYS * 86_400_000,))
        finally:
            db.close()
    except Exception as e:
        log.warning("мэдэгдэл хадгалж чадсангүй: %s", e)


def recent(user: str, days: int = KEEP_DAYS, limit: int = 500) -> list[dict]:
    """This person's messages of the last `days` days, newest first."""
    if not MESSAGES_DB.exists():
        return []
    db = _db()
    try:
        rows = db.execute("SELECT ts, market, text FROM msgs WHERE user = ? AND ts >= ? ORDER BY ts DESC LIMIT ?",
                          (user, int(time.time() * 1000) - days * 86_400_000, limit)).fetchall()
    finally:
        db.close()
    return [dict(r) for r in rows]


def _recipients(to) -> list:
    """to: None = admins, a User, a user name, or a list of those."""
    everyone = users_mod.load_users()
    if to is None:
        return [u for u in everyone if u.is_admin]
    out = []
    for t in (to if isinstance(to, (list, tuple)) else [to]):
        u = t if isinstance(t, users_mod.User) else users_mod.by_name(everyone, str(t))
        if u:
            out.append(u)
    return out


def send(chat_id: str, text: str) -> None:
    if not (TELEGRAM_ON and TELEGRAM_BOT_TOKEN and chat_id):
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                      json={"chat_id": chat_id, "text": text}, timeout=15)
    except Exception as e:
        log.warning("Telegram илгээж чадсангүй: %s", e)


def notify(text: str, key: str | None = None, every_sec: int = 0, to=None) -> None:
    """key + every_sec: don't repeat the same kind of message to the same person too often."""
    m = mkt()
    tag = f"{m.label} · " + ("💵 LIVE" if m.mode == "live" else f"🧪 {m.mode}")
    for u in _recipients(to):
        k = f"{u.name}:{key}" if key else None
        if k and every_sec and time.time() - _last.get(k, 0) < every_sec:
            continue
        if k:
            _last[k] = time.time()
        msg = f"{text}\n\n{tag}" + ("" if u.legacy else f" · 👤 {u.name}")
        log.info("📣 → %s: %s", u.name, msg)
        store(m.name, u.name, text)
        send(u.telegram_chat_id, msg)
