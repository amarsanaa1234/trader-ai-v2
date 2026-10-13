"""Telegram messages. Each person gets their own account's messages; market-wide
notices (new strategy found, shadow promoted, errors) go to the admins."""
import logging
import time

import requests

from . import users as users_mod
from .context import mkt
from .settings import TELEGRAM_BOT_TOKEN

log = logging.getLogger(__name__)
_last = {}


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
    if not (TELEGRAM_BOT_TOKEN and chat_id):
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
        send(u.telegram_chat_id, msg)
