"""Telegram commands from phones. A message is obeyed only if it comes from a
chat that belongs to a configured user (USER<n>_TELEGRAM_CHAT_ID); each person's
commands act on their own account only. Admins have a few extra commands."""
import logging

import requests

from . import users as users_mod
from .settings import TELEGRAM_BOT_TOKEN

log = logging.getLogger(__name__)

HELP = """Командууд:
/status – данс, position, допамин
/stop – шинэ арилжаа нээхийг зогсооно (нээлттэйг stop/tp-ээр нь үлдээнэ)
/resume – дахин эхлүүлнэ
/closeall – бүх position-ыг хаах (баталгаажуулалт асууна)
/report – 7 хоногийн тайлан
/resetpeak – 30%-ийн drawdown зогсолтыг шалгаад дахин нээх
/help – энэ жагсаалт"""

ADMIN_HELP = """
👑 Админ:
/users – хэрэглэгчид, тэдний данс
/emergency – БҮХ хүний бүх арилжааг зогсоож хаах (баталгаажуулалт асууна)"""


def enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN)


def poll(st) -> list[tuple]:
    """New commands since the last poll (non-blocking): [(User, "/command"), ...]."""
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
    everyone = users_mod.load_users()
    cmds = []
    for u in updates:
        offset = max(offset, u["update_id"] + 1)
        msg = u.get("message") or {}
        who = users_mod.by_chat(everyone, msg.get("chat", {}).get("id"))
        if who is None:
            continue  # a stranger: ignored
        text = (msg.get("text") or "").strip().split("@")[0].lower()
        if text.startswith("/"):
            cmds.append((who, text))
    st.set("tg_offset", offset)
    return cmds
