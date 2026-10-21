"""The daily self-improvement message must read well whatever the improver returned."""
from core import messages as M


def test_improve_report_reads_well():
    rep = {
        "candidates": 1600, "dopamine": -0.21, "search_width": 0.20,
        "current": {"walk_forward": {"positive_folds": 2, "trades": 2061},
                    "vault": {"return": 0.0261, "trades": 360, "avg_r": 0.145}},
        "best_walk_forward": {"positive_folds": 4, "avg_r": 0.06},
        "checked": [{"rejected": "vault оноо одоогийнхоос дээр биш"}, {"rejected": "vault дээр арилжаа цөөн"},
                    {"rejected": None}, {"rejected": None}],
        "new_shadows": [{"id": "6e357722", "vault": {"return": 0.031, "trades": 340}}],
        "lessons": [[{"tag": "against_carry", "trades": 120, "avg_r": -0.21}]],
        "shadow_review": [{"id": "6e357722", "days": 0.4, "shadow": {"trades": 0, "return": 0.0}}],
    }
    text = M.improve_report(rep, 10)
    print(text)
    assert "1,600" in text and "2/4" in text and "+2.6%" in text
    assert "not better than live" in text and "2 passed" in text
    assert "against_carry" in text and "NOT changed today" in text and "10 days" in text
    assert "CHANGED TODAY" in M.improve_report({**rep, "shadow_promoted": "6e357722"}, 10)
    assert "DAILY SELF-IMPROVEMENT" in M.improve_report({}, 10)      # an empty report must not crash
