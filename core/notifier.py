"""Messages to you only (Telegram, optional). Nothing else leaves the laptop."""
import logging
import time

import requests

from .context import mkt
from .settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger(__name__)
_last = {}


def notify(text: str, key: str | None = None, every_sec: int = 0) -> None:
    """key + every_sec: don't repeat the same kind of message too often."""
    if key and every_sec and time.time() - _last.get(key, 0) < every_sec:
        return
    if key:
        _last[key] = time.time()
    m = mkt()
    tag = f"{m.label} · " + ("💵 LIVE" if m.mode == "live" else f"🧪 {m.mode}")
    msg = f"{text}\n\n{tag}"
    log.info("📣 %s", msg)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                          json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=15)
        except Exception as e:
            log.warning("Telegram илгээж чадсангүй: %s", e)
