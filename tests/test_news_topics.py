"""News ideas registered in advance (owner, 2026-10-27): topics, skipped opinion
headlines, and the check that only counts headlines known after registration."""
import numpy as np

from core import ai, news, news_eval, news_topics, web
from core.storage import Storage, now_ms

HOUR = 3_600_000


def test_topics_and_skip():
    assert news_topics.topic("Fed holds rates steady, Powell says") == "central bank / rates"
    assert news_topics.topic("API: Crude, Product Inventories Continue to Draw") == "supply / inventory / weather"
    assert news_topics.topic("Bitcoin price prediction: could BTC hit $100k?") == "forecast / analysis / opinion"
    assert news_topics.topic("") == news_topics.OTHER
    assert news_topics.skip("Ethereum outlook for next week") and not news_topics.skip("Fed cuts rates")


def test_opinion_headlines_never_reach_the_ai(tmp_path, monkeypatch):
    st = Storage(tmp_path / "n.db")
    t = now_ms() - HOUR
    st.insert_news([("op", t, "x", "Bitcoin could hit $100k, analyst predicts", ""),
                    ("fed", t + 1, "x", "Fed cuts rates by 50bp", "")])
    seen = {}

    def ask_json(system, user, schema):
        seen["prompt"] = user
        return {"items": [{"id": 0, "asset": "BTC", "sentiment": "positive", "strength": 0.8, "reason": "r"}]}

    monkeypatch.setattr(ai, "enabled", lambda: True)
    monkeypatch.setattr(ai, "ask_json", ask_json)
    monkeypatch.setattr(ai, "model_name", lambda: "test")
    assert news.score_news(st) == 1
    assert "Fed cuts rates" in seen["prompt"] and "analyst predicts" not in seen["prompt"]
    assert st.unscored_news(0) == []                                   # the skipped one is marked done too
    assert st.db.execute("SELECT news_id FROM sentiment").fetchone()["news_id"] == "fed"


def test_hypotheses_count_only_new_headlines(tmp_path):
    st = Storage(tmp_path / "n.db")
    start = news_eval._registered_ms() - 10 * 24 * HOUR
    n = 60 * 24
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(0.002 + rng.normal(0, 0.003, n)))      # rises: positive scores are right
    rows = [("BTC/USDT:USDT", "1h", start + i * HOUR, c / 1.001, c * 1.003, c / 1.004, c, 1.0)
            for i, c in enumerate(close)]
    st.db.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?)", rows)
    news_rows, sent = [], []
    for k in range(40):                                    # 5 before registration (ignored), 35 after
        ts = news_eval._registered_ms() + (k - 5) * 12 * HOUR
        news_rows.append((f"n{k}", ts, "x", "Fed signals another rate cut", ""))
        sent.append((ts, "BTC", 0.8, 0.8, "r", ts + HOUR, "test", f"n{k}"))
    st.insert_news(news_rows)
    st.insert_sentiment(sent)
    hyps = {h["key"]: h for h in news_eval.hypotheses(st)}
    cb = hyps["central_bank_4h"]
    assert cb["n"] == 35 and cb["hit"] > 0.8 and cb["status"].startswith("✅")
    assert hyps["supply_24h"]["n"] == 0 and hyps["supply_24h"]["status"].startswith("⏳")
    st.set("news_eval", {**news_eval.evaluate(st, days=60), "run_at": now_ms()})
    html = web.news_check_html(st)
    assert "Урьдчилан бүртгэсэн" in html and "central bank / rates" in html
