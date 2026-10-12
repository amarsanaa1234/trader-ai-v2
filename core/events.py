"""Economic calendar blackout: no new trades around high-impact US events
(CPI, FOMC, NFP...). Crypto often whipsaws hard right after them.

Sources: the free weekly ForexFactory feed, plus any events you add by hand in
events.json, e.g. [{"time": "2026-11-03T18:00:00Z", "title": "My event"}].
"""
import json
import logging
from datetime import datetime, timezone

import requests

from .settings import BLACKOUT_AFTER_MIN, BLACKOUT_BEFORE_MIN, ROOT
from .storage import Storage, now_ms

log = logging.getLogger(__name__)
FEED = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
MANUAL = ROOT / "events.json"
MIN_MS = 60_000


def _parse(t: str) -> int:
    return int(datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp() * 1000)


def refresh(st: Storage) -> int:
    """Download this week's high-impact USD events (call a few times a day)."""
    r = requests.get(FEED, timeout=20, headers={"User-Agent": "Mozilla/5.0 (crypto_ai calendar)"})
    r.raise_for_status()
    found = {(e["ts"], e["title"]): e for e in st.get("calendar", []) if e["ts"] > now_ms() - 7 * 86_400_000}
    for e in r.json():
        if e.get("impact") == "High" and e.get("country") == "USD":
            try:
                ts = _parse(e["date"])
            except Exception:
                continue
            title = e.get("title", "?")
            found[(ts, title)] = {"ts": ts, "title": title, "country": "USD", "impact": "High",
                                  "forecast": e.get("forecast") or "", "previous": e.get("previous") or ""}
    st.set("calendar", sorted(found.values(), key=lambda e: e["ts"]))
    return len(found)


def all_events(st: Storage) -> list[dict]:
    events = list(st.get("calendar", []))
    if MANUAL.exists():
        try:
            for e in json.loads(MANUAL.read_text(encoding="utf-8")):
                events.append({"ts": _parse(e["time"]), "title": e.get("title", "manual")})
        except Exception as ex:
            log.warning("events.json уншиж чадсангүй: %s", ex)
    return sorted(events, key=lambda e: e["ts"])


def blackout(st: Storage, at_ms: int | None = None) -> str | None:
    """Returns the event title if we're inside its no-trade window."""
    t = now_ms() if at_ms is None else at_ms
    for e in all_events(st):
        if e["ts"] - BLACKOUT_BEFORE_MIN * MIN_MS <= t <= e["ts"] + BLACKOUT_AFTER_MIN * MIN_MS:
            return e["title"]
    return None


def upcoming(st: Storage, hours: int = 72) -> list[dict]:
    t = now_ms()
    return [e for e in all_events(st) if t - BLACKOUT_AFTER_MIN * MIN_MS <= e["ts"] <= t + hours * 3_600_000]
