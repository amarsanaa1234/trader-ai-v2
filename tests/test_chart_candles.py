"""Price chart candles (owner, 2026-11-15): stale timeframes are left out or rebuilt from finer ones."""
from core import web


def test_resample_15m_to_1h():
    h = 3_600_000
    rows = [[h + i * 900_000, 10 + i, 20 + i, 5 + i, 11 + i] for i in range(6)]   # 01:00 .. 02:15
    out = web._resample(rows, h)
    assert out == [[h, 10, 23, 5, 14], [2 * h, 14, 25, 9, 16]]
