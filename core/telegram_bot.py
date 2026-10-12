"""Telegram commands from your phone. Only messages from TELEGRAM_CHAT_ID are
obeyed; anything else is ignored."""
import logging

import requests

from .settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger(__name__)

HELP = """Командууд:
/status – equity, position, допамин
/stop – шинэ арилжаа нээхийг зогсооно (нээлттэйг stop/tp-ээр нь үлдээнэ)
/resume – дахин эхлүүлнэ
/closeall – бүх position-ыг хаах (баталгаажуулалт асууна)
/report – 7 хоногийн тайлан
/resetpeak – 30%-ийн drawdown зогсолтыг шалгаад дахин нээх
/help – энэ жагсаалт"""


def enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def poll(st) -> list[str]:
    """New commands since the last poll (non-blocking)."""
    if not enabled():
        return []
    offset = st.get("tg_offset", 0)
    try:
        r = requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                         params={"offset": offset, "timeout": 0}, timeout=15)
        r.raise_for_status()
        updates = r.json().get("result", [])
    except Exception as e:
        log.debug("telegram poll: %s", e)
        return []
    cmds = []
    for u in updates:
        offset = max(offset, u["update_id"] + 1)
        msg = u.get("message") or {}
        if str(msg.get("chat", {}).get("id")) != str(TELEGRAM_CHAT_ID):
            continue  # not you
        text = (msg.get("text") or "").strip().split("@")[0].lower()
        if text.startswith("/"):
            cmds.append(text)
    st.set("tg_offset", offset)
    return cmds
