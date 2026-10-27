"""Headline topics, and the news ideas registered in advance (owner, 2026-10-27).

The 10/09-10/27 look at 566 scored headlines (all markets) found the AI right
about half the time overall. Two topics looked better when chosen on the first
half of those days and checked on the second half, but on too few headlines to
trust:
  * central bank / rates -> 4h after the score: 65% right, +0.21 ATR after cost (40 headlines)
  * supply / inventory / weather -> 24h after: 53% right, +0.71 ATR (19 headlines)
They are written down HERE, before the new data exists, and core/news_eval.py
checks them only on headlines known from REGISTERED_FROM on. Choosing ideas
after seeing the data that tests them would flatter them.

Forecast / analysis / opinion headlines were right only 30% of the time at 24h,
so core/news.py no longer sends them to the AI.

The rules are plain keyword lists, kept exactly as measured; the first match wins.
"""
import re

TOPICS = [
    ("central bank / rates", r"\b(fed|fomc|ecb|boj|boe|rba|boc|snb|rate|rates|powell|lagarde|ueda|bailey|hike|cut|yield)"),
    ("data (CPI, jobs, GDP, PMI)", r"\b(cpi|inflation|payroll|nfp|jobs|unemployment|gdp|pmi|retail sales|ppi|claims)"),
    ("geopolitics / war / tariffs", r"\b(war|iran|israel|russia|ukraine|china|tariff|sanction|trump|attack|strike|opec)"),
    ("ETF / funds / flows", r"\b(etf|inflow|outflow|fund|blackrock|grayscale|treasur|buy(s|ing)? \$|holdings)"),
    ("regulation / court / SEC", r"\b(sec|regulat|court|law|bill|congress|lawsuit|cftc|ban|license)"),
    ("hack / exploit / scam", r"\b(hack|exploit|stolen|scam|drain|breach)"),
    ("forecast / analysis / opinion",
     r"\b(forecast|outlook|prediction|predict|analysis|could|may|might|eyes|target|technical|levels?|\?)"),
    ("supply / inventory / weather", r"\b(inventor|stock(pile|s)|supply|output|production|weather|harvest|crop|eia|api)"),
]
OTHER = "other"
SKIP = {"forecast / analysis / opinion"}          # not sent to the AI

REGISTERED_FROM = "2026-10-28"                    # only headlines known from this UTC day on count
HYPOTHESES = [
    {"key": "central_bank_4h", "topic": "central bank / rates", "hours": 4},
    {"key": "supply_24h", "topic": "supply / inventory / weather", "hours": 24},
]
MIN_N = 30                                        # fewer headlines than this: keep collecting

_COMPILED = [(name, re.compile(rx)) for name, rx in TOPICS]


def topic(title: str) -> str:
    t = (title or "").lower()
    for name, rx in _COMPILED:
        if rx.search(t):
            return name
    return OTHER


def skip(title: str) -> bool:
    return topic(title) in SKIP
