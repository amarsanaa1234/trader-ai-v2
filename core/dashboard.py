"""Read-only dashboard on 127.0.0.1:DASHBOARD_PORT. On the server it is published by
Caddy (see _allowed for who may see it). Runs inside the bot, or alone with `main.py dashboard`."""
import base64
import hmac
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import events, library, reward, shadow
from .context import mkt
from .settings import BLACKOUT_AFTER_MIN, BLACKOUT_BEFORE_MIN, DASHBOARD_PORT, SHADOW_MIN_DAYS
from .storage import Storage, now_ms

log = logging.getLogger(__name__)
HTML = Path(__file__).with_name("dashboard.html")
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "").strip()


def drift(st: Storage) -> dict:
    """Live results vs what the backtest promised (vault)."""
    trades = st.closed_trades(mkt().mode, 200)
    live_r = [t["r_multiple"] for t in trades if t["r_multiple"] is not None]
    promised = None
    if mkt().params_path.exists():
        m = json.loads(mkt().params_path.read_text(encoding="utf-8")).get("metrics", {})
        promised = (m.get("vault") or m.get("backtest", {}).get("vault") or {}).get("avg_r")
    live = round(sum(live_r) / len(live_r), 3) if live_r else None
    warn = bool(live is not None and promised is not None and len(live_r) >= 15 and live < promised - 0.3)
    return {"live_avg_r": live, "live_trades": len(live_r), "backtest_avg_r": promised, "warning": warn}


_prices = {"ts": 0.0, "data": {}}


def prices(symbols) -> dict:
    """Last traded prices, cached for 15 s so many viewers don't hammer Binance."""
    import time
    if symbols and time.time() - _prices["ts"] > 15:
        try:
            _prices["data"] = mkt().prices(symbols)
            _prices["ts"] = time.time()
        except Exception as e:
            log.debug("prices: %s", e)
    return _prices["data"]


def data() -> dict:
    st = Storage()
    MODE, SYMBOLS = mkt().mode, mkt().symbols
    params_path = mkt().params_path
    hist = st.equity_history(MODE, now_ms() - 30 * 86_400_000)
    step = max(1, len(hist) // 600)
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else {}
    dop = st.get("dopamine", 0.0)
    eq = hist[-1]["equity"] if hist else None
    peak = st.get("peak_equity")
    return {
        "market": mkt().name, "market_label": mkt().label, "mode": MODE, "symbols": SYMBOLS, "equity": eq, "peak": peak,
        "drawdown": round(1 - eq / peak, 4) if eq and peak else 0.0,
        "dopamine": dop, "mood": reward.mood(dop), "day": st.get("day"), "paused": st.get("paused", False),
        "paper_started": st.get("paper_started"),
        "equity_history": hist[::step],
        "open_trades": st.open_trades(MODE),
        "prices": prices({t["symbol"] for t in st.open_trades(MODE)}),
        "pending": {k[8:]: v for k, v in [(r["key"], json.loads(r["value"])) for r in st.db.execute(
            "SELECT key, value FROM state WHERE key LIKE 'pending:%'")] if v},
        "trades": st.closed_trades(MODE, 50),
        "params": {"note": params.get("note"), "updated": params.get("updated_utc"), "metrics": params.get("metrics"),
                   "pattern_edges": params.get("params", {}).get("pattern_edges")},
        "shadows": [{**{k: s.get(k) for k in ("id", "created", "forward", "live_forward", "backtest")},
                     "days": round((now_ms() - s["created"]) / 86_400_000, 1)} for s in st.get("shadows", [])],
        "shadow_rules": {"min_days": SHADOW_MIN_DAYS, "min_trades": shadow.MIN_FORWARD_TRADES,
                         "margin": shadow.PROMOTE_MARGIN, "max_age_days": shadow.MAX_AGE_DAYS},
        "library": [{k: s.get(k) for k in ("key", "name", "status", "source", "description", "long_entry",
                                           "short_entry", "stats", "added", "reviewed")} for s in library.load()],
        "carry_open": st.open_carries(MODE),
        "carry_closed": [dict(r) for r in st.db.execute(
            "SELECT * FROM carry WHERE status='closed' AND mode=? ORDER BY closed_ts DESC LIMIT 20", (MODE,))],
        "events": events.upcoming(st, hours=24 * 7),
        "blackout": {"before_min": BLACKOUT_BEFORE_MIN, "after_min": BLACKOUT_AFTER_MIN, "now": events.blackout(st)},
        "drift": drift(st),
        **mkt().dashboard_extras(st),   # market-specific panels (crypto: Strategy 2)
    }


def _allowed(handler) -> bool:
    """Public access (through the web proxy) is fine for a demo. With real money
    the dashboard needs DASHBOARD_PASSWORD, otherwise it is only reachable from
    the server itself / an SSH tunnel."""
    if DASHBOARD_PASSWORD:
        expected = "Basic " + base64.b64encode(f"admin:{DASHBOARD_PASSWORD}".encode()).decode()
        return hmac.compare_digest(handler.headers.get("Authorization", ""), expected)
    via_proxy = handler.headers.get("X-Forwarded-For") is not None
    return not (mkt().mode == "live" and via_proxy)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not _allowed(self):
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="crypto_ai"')
            self.end_headers()
            return
        if self.path in ("/", "/index.html"):
            body, ctype = HTML.read_bytes(), "text/html; charset=utf-8"
        elif self.path.startswith("/api/data"):
            body, ctype = json.dumps(data(), ensure_ascii=False, default=str).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def serve(background: bool = True):
    srv = ThreadingHTTPServer(("127.0.0.1", DASHBOARD_PORT), Handler)
    log.info("Dashboard: http://127.0.0.1:%d", DASHBOARD_PORT)
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    srv.serve_forever()
