"""Telegram off (owner, 2026-11-15): messages go to data/messages.db, kept 7 days, shown on the web."""
import time

from core import notifier, telegram_bot


def test_store_keeps_7_days(tmp_path, monkeypatch):
    monkeypatch.setattr(notifier, "MESSAGES_DB", tmp_path / "m.db")
    notifier.store("crypto", "amar", "old")
    db = notifier._db()
    with db:
        db.execute("UPDATE msgs SET ts = ?", (int(time.time() * 1000) - 8 * 86_400_000,))
    db.close()
    notifier.store("crypto", "amar", "TRADE OPENED")
    notifier.store("fx", "other", "not mine")
    got = notifier.recent("amar")
    assert [r["text"] for r in got] == ["TRADE OPENED"]
    db = notifier._db()
    assert db.execute("SELECT COUNT(*) FROM msgs").fetchone()[0] == 2   # the 8-day-old one is gone
    db.close()


def test_telegram_off(monkeypatch):
    sent = []
    monkeypatch.setattr(notifier.requests, "post", lambda *a, **k: sent.append(a))
    monkeypatch.setattr(notifier, "TELEGRAM_BOT_TOKEN", "x")
    notifier.send("123", "hi")
    assert sent == [] and not telegram_bot.enabled()
