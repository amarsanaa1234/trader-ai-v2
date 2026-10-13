"""Mistake memory: learn from losing trades.

Every backtest trade is tagged with the situation it started in (strong
evidence against it, trend or range, weekday, the market's own tags such as
the FX session). If trades started in one situation keep losing - clearly,
not by bad luck (t-statistic) - the settings learn to stay out of it
("avoid" list). A lesson is only kept if it also improves the walk-forward
score; the vault and the shadow test still have to approve the result.

Only the research windows are looked at, never the vault.
"""
import logging

import numpy as np

from . import backtest, validation
from .strategy import compute, situation_tags

log = logging.getLogger(__name__)

MIN_TRADES = 30          # a situation needs this many trades before it can be judged
MAX_AVG_R = -0.15        # ... and must lose at least this much per trade on average
MAX_T = -2.0             # ... clearly (t-statistic), not by chance
MAX_LESSONS = 3          # at most this many new "avoid" rules per run
MIN_GAIN = 0.1           # walk-forward score must improve at least this much


def situation_stats(frames: dict, p: dict) -> dict:
    """{tag: [R of every research-window trade that started in that situation]}."""
    out = {}
    for df in frames.values():
        feats = compute(df, p)
        end = validation.research_end(len(df))
        res = backtest.run(df, p, end=end, feats=feats)
        tags = situation_tags(df, feats)
        for t in res.trades:
            i = t["open_bar"] - 1                     # the candle whose signal opened it
            if i < 0:
                continue
            for name, (bad_long, bad_short) in tags.items():
                if (bad_long if t["side"] == 1 else bad_short)[i]:
                    out.setdefault(name, []).append(t["r"])
    return out


def worst_situations(stats: dict, already=()) -> list[dict]:
    found = []
    for name, rs in stats.items():
        if name in already or len(rs) < MIN_TRADES:
            continue
        a = np.asarray(rs)
        t = a.mean() / (a.std(ddof=1) + 1e-9) * np.sqrt(len(a))
        if a.mean() <= MAX_AVG_R and t <= MAX_T:
            found.append({"tag": name, "trades": len(a), "avg_r": round(float(a.mean()), 3), "t": round(float(t), 2)})
    return sorted(found, key=lambda x: x["t"])


def learn(frames: dict, p: dict, wf: dict) -> tuple[dict, list]:
    """Returns (settings with new avoid rules, lessons learned). Unchanged if nothing helps."""
    avoid = list(p.get("avoid") or [])
    bad = worst_situations(situation_stats(frames, p), avoid)
    best, best_wf, learned = dict(p), wf, []
    for b in bad[:MAX_LESSONS * 2]:
        if len(learned) >= MAX_LESSONS:
            break
        trial = {**best, "avoid": list(best.get("avoid") or []) + [b["tag"]]}
        twf = validation.walk_forward(frames, trial)
        if twf["score"] >= best_wf["score"] + MIN_GAIN and twf["positive_folds"] >= best_wf["positive_folds"]:
            best, best_wf = trial, twf
            learned.append({**b, "wf_score": twf["score"]})
            log.info("Алдаанаас сурав: '%s' нөхцөлд %d арилжаа, дундаж %+.2fR -> цаашид орохгүй",
                     b["tag"], b["trades"], b["avg_r"])
    return best, learned
