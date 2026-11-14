"""The web app: login, a menu, one page per market, settings, admin, and a
public performance page. One process serves every market (`python main.py web`);
behind Caddy (HTTPS) on the server, on 127.0.0.1 only.

Everyone sees only their own account. Admins can open anyone's pages
(?user=name) and the admin page. All pages are read-only: trading is controlled
from Telegram, so a stolen web password can look but never trade.
"""
import contextlib
import html
import json
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

import markets

from . import auth, dashboard
from . import users as users_mod
from .context import activate, mkt
from .settings import NEWS_EVAL_FROM
from .storage import Storage

log = logging.getLogger(__name__)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
COOKIE = "tai_session"
_lock = threading.Lock()
MARKET_PAGE = Path(__file__).with_name("dashboard.html")


@contextlib.contextmanager
def market(name: str):
    """Core modules read the active market; the web serves several, so switch under a lock."""
    with _lock:
        activate(markets.get(name))
        yield mkt()


def current_user(request: Request):
    name = auth.read_session(request.cookies.get(COOKIE))
    return users_mod.by_name(users_mod.load_users(markets.AVAILABLE), name) if name else None


def visible_markets(user) -> list[str]:
    return list(markets.AVAILABLE) if user.is_admin else [m for m in markets.AVAILABLE if m in user.markets]


def target_user(request: Request, user) -> str | None:
    """Whose account to show: yourself, or (admins only) ?user=name."""
    wanted = request.query_params.get("user")
    if wanted and wanted != user.name:
        return wanted if user.is_admin and users_mod.by_name(users_mod.load_users(), wanted) else None
    return user.name


# ---------------- layout ----------------
NAV_CSS = """
.tai-nav { background:var(--card); border-bottom:1px solid var(--line); position:sticky; top:0; z-index:10; }
.tai-nav .in { max-width:1200px; margin:0 auto; padding:8px 16px; display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
.tai-nav .brand { font-weight:700; margin-right:10px; color:var(--text); text-decoration:none; }
.tai-nav a.item { color:var(--text); text-decoration:none; padding:5px 10px; border-radius:8px; font-size:14px; }
.tai-nav a.item:hover { background:var(--accent-soft, rgba(59,91,219,.14)); }
.tai-nav a.item.on { background:var(--accent-soft, rgba(59,91,219,.14)); color:var(--accent); font-weight:600; }
.tai-nav .who { margin-left:auto; color:var(--muted); font-size:13px; }
.tai-nav .who a { color:var(--muted); }
@media (max-width:640px) { .tai-nav .in { gap:2px; } .tai-nav a.item { padding:4px 7px; font-size:13px; } .tai-nav .who { width:100%; margin:0; } }
"""

CSS = NAV_CSS + """
:root { --bg:#f6f7f9; --card:#fff; --text:#16181d; --muted:#6b7280; --line:#e5e7eb; --accent:#3b5bdb;
        --accent-soft:#e7ecfd; --up:#0f9d58; --down:#d93025; --warn-bg:#fff4e5; --warn:#9a5b00; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0f1115; --card:#171a21; --text:#e8eaed; --muted:#9aa0a6; --line:#2a2f3a; --accent:#8ea2ff;
          --accent-soft:#1f2640; --up:#34c77b; --down:#ff6b5e; --warn-bg:#3a2a10; --warn:#ffc46b; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main.page { max-width:1100px; margin:0 auto; padding:20px 16px 60px; }
h1 { font-size:22px; margin:6px 0 14px; } h2 { font-size:16px; margin:22px 0 8px; }
.muted { color:var(--muted); } .up { color:var(--up); } .down { color:var(--down); }
.grid { display:grid; gap:12px; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; min-width:0; }
a.card { color:inherit; text-decoration:none; display:block; } a.card:hover { border-color:var(--accent); }
.big { font-size:24px; font-weight:650; font-variant-numeric:tabular-nums; }
.scroll { overflow-x:auto; } table { width:100%; border-collapse:collapse; font-size:14px; }
th,td { text-align:left; padding:8px; border-bottom:1px solid var(--line); white-space:nowrap; } th { color:var(--muted); font-size:12px; }
.note { background:var(--warn-bg); color:var(--warn); border-radius:10px; padding:10px 14px; margin:12px 0; }
.ok { background:var(--accent-soft); border-radius:10px; padding:10px 14px; margin:12px 0; }
form.box { max-width:380px; } label { display:block; font-size:13px; color:var(--muted); margin:10px 0 4px; }
input { width:100%; padding:10px; border-radius:8px; border:1px solid var(--line); background:var(--bg); color:var(--text); font-size:15px; }
button { margin-top:14px; padding:10px 16px; border:0; border-radius:8px; background:var(--accent); color:#fff; font-size:15px; cursor:pointer; }
code, pre { background:var(--accent-soft); border-radius:6px; padding:1px 6px; font-size:13px; }
pre { padding:10px; overflow-x:auto; }
"""


# Look (owner, 2026-11-12): modern and minimal, easy on the eyes but with colour. Loaded after each
# page's own styles (web pages and the market dashboards), so it restyles everything in one place.
FONT_LINK = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">')
THEME = """
:root { --bg:#f4f6fb; --card:#ffffff; --card2:#f8fafd; --text:#111827; --muted:#6b7488; --line:#e7ebf3;
        --accent:#5b5bd6; --accent2:#0ea5a4; --accent-soft:#ecebff; --up:#0f9f6e; --down:#e5484d;
        --warn-bg:#fff6e0; --warn:#9a6700; --shadow:0 1px 2px rgba(17,24,39,.04),0 6px 24px rgba(17,24,39,.05); }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --card:#141a23; --card2:#18202b; --text:#e6e9ef; --muted:#8b94a7; --line:#232c39;
          --accent:#8b8cff; --accent2:#2ec4b6; --accent-soft:rgba(139,140,255,.14); --up:#3ecf8e; --down:#ff6b81;
          --warn-bg:rgba(251,191,36,.12); --warn:#fbbf24; --shadow:none; } }
[hidden] { display:none !important; }   /* "hidden" must win over .grid / .row display rules */
html { -webkit-font-smoothing:antialiased; }
body { background:var(--bg) !important; color:var(--text);
       font-family:Inter,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif !important; letter-spacing:-.005em; }
main, main.page { max-width:1180px; }
h1 { font-size:24px !important; font-weight:700; letter-spacing:-.02em; }
h2 { font-size:15px !important; font-weight:600; letter-spacing:-.01em; }
.muted, .sub { color:var(--muted) !important; }
.card { background:var(--card) !important; border:1px solid var(--line) !important; border-radius:16px !important;
        box-shadow:var(--shadow); padding:18px !important; }
.card .card { box-shadow:none; background:var(--card2) !important; border-radius:12px !important; padding:14px !important; }
a.card:hover { border-color:var(--accent) !important; transform:translateY(-1px); transition:all .15s ease; }
.grid { gap:14px !important; }
.row { gap:14px !important; margin-bottom:14px !important; }
.kpi .label { font-size:11.5px !important; text-transform:uppercase; letter-spacing:.06em; font-weight:600; color:var(--muted); }
.kpi { overflow:hidden; }
.kpi .value, .big { font-size:clamp(18px, 1vw + 12px, 24px) !important; font-weight:700 !important; letter-spacing:-.02em;
                    margin-top:6px; line-height:1.2; overflow-wrap:break-word; }
.kpi .value .word { display:block; font-size:15px; font-weight:600; letter-spacing:0; margin-top:4px; color:var(--text); }
.up { color:var(--up) !important; } .down { color:var(--down) !important; }
table { font-size:13.5px; }
th { font-size:11px !important; text-transform:uppercase; letter-spacing:.06em; font-weight:600 !important;
     color:var(--muted) !important; border-bottom:1px solid var(--line) !important; padding:8px 10px !important; }
td { border-bottom:1px solid var(--line) !important; padding:9px 10px !important; }
tbody tr:last-child td { border-bottom:0 !important; }
tbody tr:hover td { background:var(--accent-soft); }
.banner, .note { background:var(--warn-bg) !important; color:var(--warn) !important; border:0; border-radius:12px !important; }
.ok { background:var(--accent-soft) !important; border-radius:12px !important; }
.badge, .pill, .chip { background:var(--accent-soft) !important; border:0 !important; color:var(--accent); font-weight:500; }
.badge.up { background:rgba(16,185,129,.12) !important; color:var(--up); }
.badge.down { background:rgba(244,63,94,.12) !important; color:var(--down); }
.bar { background:var(--line) !important; } .bar > span { background:linear-gradient(90deg,var(--accent),var(--accent2)) !important; }
.q { border-color:var(--line) !important; color:var(--muted) !important; background:var(--card2) !important; }
.q:hover { color:var(--accent) !important; border-color:var(--accent) !important; }
.help { background:var(--card2) !important; border:1px solid var(--line); border-radius:12px !important; }
.empty { color:var(--muted); font-size:13px; }
.sheet { border-radius:18px !important; background:var(--card) !important; }
input { border-radius:10px !important; border:1px solid var(--line) !important; background:var(--card2) !important; }
input:focus { outline:none; border-color:var(--accent) !important; box-shadow:0 0 0 3px var(--accent-soft); }
button:not(.q):not(.x) { border-radius:10px !important; font-weight:600;
         background:linear-gradient(135deg,var(--accent),#7c5cff) !important; color:#fff !important; }
a { color:var(--accent); }
a.btn { border-radius:10px !important; border-color:var(--line) !important; background:var(--card2); }
.tai-nav { background:color-mix(in srgb, var(--card) 82%, transparent) !important; backdrop-filter:saturate(1.4) blur(12px);
           -webkit-backdrop-filter:saturate(1.4) blur(12px); border-bottom:1px solid var(--line) !important; }
.tai-nav .in { padding:10px 16px !important; }
.tai-nav .brand { font-weight:700; letter-spacing:-.02em;
                  background:linear-gradient(90deg,var(--accent),var(--accent2)); -webkit-background-clip:text;
                  background-clip:text; color:transparent !important; }
.tai-nav a.item { border-radius:999px !important; color:var(--muted) !important; font-weight:500; }
.tai-nav a.item:hover { color:var(--text) !important; background:var(--card2) !important; }
.tai-nav a.item.on { background:var(--accent-soft) !important; color:var(--accent) !important; }
::selection { background:var(--accent-soft); }
@media (max-width:640px) {
  .tai-nav .in { flex-wrap:nowrap !important; overflow-x:auto; scrollbar-width:none; gap:4px !important; }
  .tai-nav .in::-webkit-scrollbar { display:none; }
  .tai-nav a.item, .tai-nav .brand { flex:none; }
  .tai-nav .who { width:auto !important; margin-left:8px !important; flex:none; }
  #kpis, [id$="-kpi"] { grid-template-columns:repeat(2, minmax(0, 1fr)) !important; gap:10px !important; }
  .kpi { padding:12px !important; } .kpi .value { font-size:18px !important; }
  .card { padding:14px !important; border-radius:14px !important; }
}
"""


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def nav(user, active: str) -> str:
    items = [("home", "/", "🏠 Нүүр")]
    for name in visible_markets(user):
        items.append((name, f"/m/{name}", markets.get(name).label))
    items += [("messages", "/messages", "📨 Мэдэгдэл"), ("reports", "/reports", "📊 Тайлан"),
              ("settings", "/settings", "⚙️ Тохиргоо")]
    if user.is_admin:
        items.append(("admin", "/admin", "👑 Админ"))
    links = "".join(f'<a class="item{" on" if k == active else ""}" href="{h}">{t}</a>' for k, h, t in items)
    return (f'<nav class="tai-nav"><div class="in"><a class="brand" href="/">🤖 Trader AI</a>{links}'
            f'<span class="who">👤 {esc(user.name)}{" 👑" if user.is_admin else ""} · <a href="/logout">гарах</a></span>'
            f"</div></nav>")


def page(title: str, body: str, user=None, active: str = "") -> HTMLResponse:
    top = nav(user, active) if user else ""
    return HTMLResponse(f"""<!doctype html><html lang="mn"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)} · Trader AI</title>
{FONT_LINK}<style>{CSS}{THEME}</style></head><body>{top}<main class="page">{body}</main></body></html>""",
                        headers={"Cache-Control": "no-store"})


def need_login():
    return RedirectResponse("/login", status_code=303)


def money(x) -> str:
    return "–" if x is None else f"{x:,.2f}"


def pct(x) -> str:
    return "–" if x is None else f'<span class="{"up" if x > 0 else "down" if x < 0 else ""}">{x:+.2%}</span>'


# ---------------- login ----------------
@app.get("/login")
def login_form(request: Request, err: str = ""):
    msg = {"bad": "Нэр эсвэл нууц үг буруу байна.",
           "locked": f"Олон удаа буруу оруулсан тул {auth.LOCK_MINUTES} минут түгжигдлээ."}.get(err, "")
    return page("Нэвтрэх", f"""<h1>🤖 Trader AI</h1>{f'<div class="note">{msg}</div>' if msg else ''}
<form class="box card" method="post" action="/login">
<label>Нэр</label><input name="username" autocomplete="username" autofocus required>
<label>Нууц үг</label><input name="password" type="password" autocomplete="current-password" required>
<button type="submit">Нэвтрэх</button></form>
<p class="muted"><a href="/public">🌍 Нийтийн гүйцэтгэлийн хуудас</a></p>""")


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    name = username.strip().lower()
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    if auth.locked(name):
        return RedirectResponse("/login?err=locked", status_code=303)
    known = users_mod.by_name(users_mod.load_users(markets.AVAILABLE), name)
    if not (auth.check_login(name, password, ip) and known):
        log.warning("login failed: %s from %s", name, ip)
        return RedirectResponse("/login?err=bad", status_code=303)
    resp = RedirectResponse("/", status_code=303)
    secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
    resp.set_cookie(COOKIE, auth.make_session(name), max_age=auth.SESSION_DAYS * 86400, httponly=True,
                    samesite="lax", secure=secure)
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(COOKIE)
    return resp


# ---------------- home ----------------
def _summary(name: str, who: str) -> dict:
    with market(name):
        try:
            return dashboard.summary(who)
        except Exception as e:
            log.warning("summary %s: %s", name, e)
            return {"market": name, "label": mkt().label, "mode": mkt().mode, "error": str(e)}


@app.get("/")
def home(request: Request):
    user = current_user(request)
    if not user:
        return need_login()
    cards = []
    for name in visible_markets(user):
        s = _summary(name, user.name)
        if s.get("equity") is None:
            body = '<div class="muted">Одоогоор өгөгдөл алга (бот асаагүй эсвэл таны данс холбогдоогүй)</div>'
        else:
            state = "⏸ зогссон" if s["paused"] else ("🛑 kill switch" if s["halted"] else "▶️ ажиллаж байна")
            body = (f'<div class="big">{money(s["equity"])}</div><div>Өнөөдөр {pct(s["today"])}</div>'
                    f'<div class="muted">{s["open"]} position · {s["pending"]} захиалга · {state}</div>'
                    f'<div class="muted">допамин {s["dopamine"]:+.2f} {esc(s["mood"])}</div>')
        cards.append(f'<a class="card" href="/m/{name}"><div class="muted">{esc(s["label"])} · {esc(s["mode"])}</div>{body}</a>')
    if not cards:
        cards.append('<div class="card muted">Танд нээлттэй зах зээл алга. Админд хандана уу.</div>')
    return page("Нүүр", f"<h1>Сайн байна уу, {esc(user.name)} 👋</h1><div class='grid'>{''.join(cards)}</div>",
                user, "home")


# ---------------- market pages ----------------
@app.get("/m/{name}")
def market_page(name: str, request: Request):
    user = current_user(request)
    if not user:
        return need_login()
    who = target_user(request, user)
    if name not in visible_markets(user) or who is None:
        return page("Хандах эрхгүй", "<h1>Хандах эрхгүй</h1>", user)
    api = f"/api/m/{name}/data?user={who}"
    candles_api = f"/api/m/{name}/candles"
    inject = (f"{FONT_LINK}<style>{NAV_CSS}{THEME}</style>{nav(user, name)}"
              f"<script>window.API_URL = {json.dumps(api)}; window.CANDLES_URL = {json.dumps(candles_api)};"
              f" window.MARKET_TF = {json.dumps('15m' if name == 'fx' else m_tf(name))};</script>")
    if who != user.name:
        inject += f'<div style="max-width:1200px;margin:8px auto;padding:0 16px" class="muted">👑 {esc(who)}-ийн данс</div>'
    text = MARKET_PAGE.read_text(encoding="utf-8")
    text = text.replace("<body>", "<body>" + inject, 1).replace('fetch("/api/data")', "fetch(window.API_URL)")
    m = markets.get(name)
    text = text.replace("<title>Crypto AI Dashboard</title>", f"<title>{esc(m.label)} · Trader AI</title>")
    text = text.replace("<h1>🤖 Crypto AI</h1>", f"<h1>{esc(m.label)}</h1>")
    text = text.replace('" USDT"', json.dumps(" " + m.currency))
    return HTMLResponse(text, headers={"Cache-Control": "no-store"})


@app.get("/api/m/{name}/data")
def market_data(name: str, request: Request):
    user = current_user(request)
    if not user:
        return JSONResponse({"error": "login"}, status_code=401)
    who = target_user(request, user)
    if name not in visible_markets(user) or who is None:
        return JSONResponse({"error": "forbidden"}, status_code=403)
    with market(name):
        d = dashboard.data(who)
    return JSONResponse(json.loads(json.dumps(d, default=str)), headers={"Cache-Control": "no-store"})


def m_tf(name: str) -> str:
    try:
        return markets.get(name).timeframe
    except Exception:
        return "4h"


@app.get("/api/m/{name}/candles")
def candles_api(name: str, request: Request, symbol: str, tf: str = "", until: int | None = None, limit: int = 300):
    """Candles for the price chart (owner 2026-11-15): the newest `limit` up to `until` (ms)."""
    user = current_user(request)
    if not user:
        return JSONResponse({"error": "login"}, status_code=401)
    if name not in visible_markets(user):
        return JSONResponse({"error": "forbidden"}, status_code=403)
    limit = max(20, min(int(limit), 1000))
    with market(name):
        db = Storage().db
        try:
            # Only timeframes that are still kept up to date: crypto 15m stopped 2026-10-11 and FX
            # 1h/4h on 2026-11-11 (nobody syncs them any more). A stale one is rebuilt from a fresh
            # finer one (FX 1h/4h from 15m) or left out.
            last = {r[0]: int(r[1]) for r in db.execute(
                "SELECT tf, MAX(ts) FROM candles WHERE symbol = ? GROUP BY tf", (symbol,)) if r[0] in TF_MS}
            newest = max(last.values(), default=0)
            fresh = sorted((t for t, ts in last.items() if ts >= newest - 2 * 86_400_000), key=TF_MS.get)
            source = {t: t for t in fresh}
            for t in TF_MS:
                if t not in source and fresh and TF_MS[fresh[0]] < TF_MS[t]:
                    source[t] = fresh[0]
            tfs = sorted(source, key=TF_MS.get)
            pref = "15m" if name == "fx" else mkt().timeframe
            tf = tf if tf in source else pref if pref in source else (tfs[-1] if tfs else "")
            rows = []
            if tf:
                base, k = source[tf], TF_MS[tf] // TF_MS[source[tf]]
                q, args = "SELECT ts, open, high, low, close FROM candles WHERE symbol = ? AND tf = ?", [symbol, base]
                if until:
                    q += " AND ts <= ?"
                    args.append(int(until))
                rows = db.execute(q + " ORDER BY ts DESC LIMIT ?", (*args, limit * k)).fetchall()
        finally:
            db.close()
    rows = [[int(r[0]), r[1], r[2], r[3], r[4]] for r in reversed(rows)]
    if tf and source[tf] != tf:
        rows = _resample(rows, TF_MS[tf])[-limit:]
    return JSONResponse({"symbol": symbol, "tf": tf, "tfs": tfs, "candles": rows}, headers={"Cache-Control": "no-store"})


TF_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}


def _resample(rows: list, ms: int) -> list:
    """Finer candles -> bigger ones on the UTC grid (ts, open, high, low, close)."""
    out = []
    for ts, o, h, l, c in rows:
        b = ts // ms * ms
        if out and out[-1][0] == b:
            x = out[-1]
            x[2], x[3], x[4] = max(x[2], h), min(x[3], l), c
        else:
            out.append([b, o, h, l, c])
    return out


# ---------------- messages (replaced Telegram, owner 2026-11-15) ----------------
@app.get("/messages")
def messages_page(request: Request):
    user = current_user(request)
    if not user:
        return need_login()
    from . import notifier
    mine = set(visible_markets(user))
    rows = [r for r in notifier.recent(user.name) if r["market"] in mine]
    labels = {n: markets.get(n).label for n in mine}
    chips = "".join(f'<button class="mchip" data-m="{esc(n)}">{esc(labels[n])} '
                    f'<span class="muted">{sum(1 for r in rows if r["market"] == n)}</span></button>'
                    for n in sorted(mine) if any(r["market"] == n for r in rows))
    items = "".join(
        f'<div class="msg" data-m="{esc(r["market"])}" data-ts="{r["ts"]}">'
        f'<div class="mhead"><span>{esc(labels.get(r["market"], r["market"]))}</span><span class="mt"></span></div>'
        f'<div class="mtext">{esc(r["text"])}</div></div>' for r in rows)
    body = f"""<h1>📨 Мэдэгдэл</h1>
<p class="muted">Ботын мэдэгдлүүд (арилжаа нээсэн/хаасан, stop шилжсэн, TP сунгасан, алдаа гэх мэт).
Сүүлийн {notifier.KEEP_DAYS} хоногийнх хадгалагдана, түүнээс хуучин нь автоматаар устна.</p>
<div class="mchips"><button class="mchip on" data-m="">Бүгд <span class="muted">{len(rows)}</span></button>{chips}</div>
<div id="msgs">{items or '<p class="muted">Сүүлийн 7 хоногт мэдэгдэл алга.</p>'}</div>
<style>
.mchips{{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}}
.mchip{{border:1px solid var(--line);background:var(--card);color:inherit;border-radius:999px;padding:5px 12px;cursor:pointer;font:inherit}}
.mchip.on{{border-color:var(--accent);color:var(--accent)}}
.mday{{margin:18px 0 6px;font-weight:600;color:var(--muted);font-size:.9em}}
.msg{{border:1px solid var(--line);background:var(--card);border-radius:12px;padding:10px 12px;margin:6px 0}}
.mhead{{display:flex;justify-content:space-between;gap:8px;font-size:.82em;color:var(--muted);margin-bottom:4px}}
.mtext{{white-space:pre-wrap;word-break:break-word;font-size:.92em;line-height:1.45}}
</style>
<script>
(function () {{
  const box = document.getElementById("msgs"), msgs = [...box.querySelectorAll(".msg")];
  let last = "";
  for (const el of msgs) {{
    const d = new Date(+el.dataset.ts);
    el.querySelector(".mt").textContent = d.toLocaleTimeString([], {{hour: "2-digit", minute: "2-digit"}});
    const day = d.toLocaleDateString([], {{weekday: "long", month: "numeric", day: "numeric"}});
    if (day !== last) {{
      const h = document.createElement("div"); h.className = "mday"; h.textContent = day; h.dataset.day = day;
      el.before(h); last = day;
    }}
    el.dataset.day = day;
  }}
  document.querySelectorAll(".mchip").forEach(b => b.onclick = () => {{
    document.querySelectorAll(".mchip").forEach(x => x.classList.toggle("on", x === b));
    const m = b.dataset.m, shown = new Set();
    for (const el of msgs) {{
      const ok = !m || el.dataset.m === m; el.hidden = !ok; if (ok) shown.add(el.dataset.day);
    }}
    box.querySelectorAll(".mday").forEach(h => h.hidden = !shown.has(h.dataset.day));
  }});
}})();
</script>"""
    return page("Мэдэгдэл", body, user, "messages")


# ---------------- reports ----------------
# The AI's weekly lessons take ~25 s per market on the server's CPU. They used to be written
# on every page load (~75 s, and other pages waited on the market lock meanwhile), so now
# they are made in the background and kept for WEEK_AI_TTL; the page shows what is ready.
WEEK_AI_TTL = 6 * 3600
_week_ai: dict = {}            # (market, user) -> {"ts": made, "text": bullets, "busy": bool}
_week_ai_lock = threading.Lock()


def _week_bullets(name: str, user: str, stats: dict) -> str:
    c = None
    with _week_ai_lock:
        c = _week_ai.setdefault((name, user), {"ts": 0, "text": "", "busy": False})
        if time.time() - c["ts"] > WEEK_AI_TTL and not c["busy"]:
            c["busy"] = True

            def work():
                from . import review
                text = review.weekly_bullets(stats, name)
                with _week_ai_lock:
                    c.update(ts=time.time(), text=text, busy=False)

            threading.Thread(target=work, daemon=True).start()
        text, busy = c["text"], c["busy"]
    if text:
        return "\n\n" + text
    return "\n\n⏳ AI-ийн дүгнэлт бэлдэж байна (~1 минут), дараа нь хуудсаа шинэчилнэ үү." if busy else ""


@app.get("/reports")
def reports(request: Request):
    user = current_user(request)
    if not user:
        return need_login()
    from . import review
    parts = []
    for name in visible_markets(user):
        with market(name):   # only the quick numbers under the shared lock
            try:
                stats = review.weekly_stats(Storage(), mkt().mode, user.name)
                text = review.weekly_head(stats) + _week_bullets(name, user.name, stats)
            except Exception as e:
                text = f"–  ({e})"
            parts.append(f"<h2>{esc(mkt().label)}</h2><pre>{esc(text)}</pre>")
    news_parts = []
    for name in visible_markets(user):
        with market(name):
            news_parts.append(news_check_html(Storage()))
    return page("Тайлан", "<h1>📊 7 хоногийн тайлан</h1>" + "".join(parts) +
                "<h1 style='margin-top:28px'>📰 Мэдээний шалгалт</h1>"
                "<p class='muted'>AI мэдээг \"эерэг/сөрөг\" гэж үнэлсний дараа үнэ тэр чиглэлд явсан уу? "
                "Мэдээ одоогоор шийдвэрт оролцдоггүй; энэ шалгалтаар буцааж оруулах эсэхийг шийднэ. "
                f"{esc(NEWS_EVAL_FROM)}-нээс эхлэн 7 хоног тутам автоматаар шинэчлэгдэнэ.</p>" + "".join(news_parts),
                user, "reports")


def news_check_html(st) -> str:
    r = st.get("news_eval")
    head = f"<h2>{esc(mkt().label)}</h2>"
    if not r:
        n = st.db.execute("SELECT COUNT(*) c FROM news").fetchone()["c"]
        k = st.db.execute("SELECT COUNT(*) c FROM sentiment").fetchone()["c"]
        return head + (f"<div class='card'>⏳ {esc(NEWS_EVAL_FROM)}-нд анх удаа шалгана. Одоогоор <b>{n}</b> гарчиг, "
                       f"<b>{k}</b> AI үнэлгээ цуглуулсан.</div>")
    from datetime import datetime, timezone
    when = datetime.fromtimestamp(r["run_at"] / 1000, timezone.utc).strftime("%Y-%m-%d")
    rows = "".join(f"<tr><td>{x['hours']} цагийн дараа</td><td>{esc(x['subset'])}</td><td>{x['n']}</td>"
                   f"<td>{x['hit']:.0%}</td><td>{x['avg_atr']:+.2f}</td><td>{x['t']:+.1f}</td></tr>" for x in r["rows"])
    table = ("<table><thead><tr><th>Хугацаа</th><th>Мэдээ</th><th>Тоо</th><th>Зөв чиглэл</th><th>Дундаж (ATR)</th>"
             f"<th>t</th></tr></thead><tbody>{rows}</tbody></table>") if rows else ""
    return head + (f"<div class='card'><b>{esc(r['verdict'])}</b><br><span class='muted'>{when} · сүүлийн {r['days']} хоног · "
                   f"{r['scored']} үнэлгээ · {esc(', '.join(r.get('models') or []))}</span>{table}"
                   "<p class='muted'>Зөв чиглэл &gt; 55% ба t &gt; 2 бол мэдээ ашигтай байж магадгүй.</p></div>"
                   + hypotheses_html(r.get("hypotheses") or []))


def hypotheses_html(hyps) -> str:
    """The news ideas written down in advance (core/news_topics.py), judged only on newer headlines."""
    if not hyps:
        return ""
    rows = "".join(
        f"<tr><td>{esc(h['topic'])}</td><td>{h['hours']} цаг</td><td>{h['n']}</td>"
        f"<td>{format(h['hit'], '.0%') if 'hit' in h else '–'}</td>"
        f"<td>{format(h['avg_atr'], '+.2f') if 'avg_atr' in h else '–'}</td>"
        f"<td>{format(h['t'], '+.1f') if 't' in h else '–'}</td><td>{esc(h['status'])}</td></tr>" for h in hyps)
    return ("<div class='card'><b>🧪 Урьдчилан бүртгэсэн 2 таамаг</b><br>"
            f"<span class='muted'>{esc(hyps[0]['since'])}-нээс хойшхи шинэ мэдээгээр л шалгана "
            "(хуучин өгөгдлөөр сонгоод тэр өгөгдлөөрөө шалгавал хэт сайн харагддаг).</span>"
            "<table><thead><tr><th>Сэдэв</th><th>Хугацаа</th><th>Тоо</th><th>Зөв</th><th>Дундаж (ATR)</th>"
            f"<th>t</th><th>Дүгнэлт</th></tr></thead><tbody>{rows}</tbody></table></div>")


# ---------------- settings ----------------
@app.get("/settings")
def settings_page(request: Request, msg: str = ""):
    user = current_user(request)
    if not user:
        return need_login()
    note = {"ok": '<div class="ok">✅ Нууц үг солигдлоо. Бусад төхөөрөмж дээр дахин нэвтрэх шаардлагатай.</div>',
            "bad": '<div class="note">Одоогийн нууц үг буруу байна.</div>',
            "mismatch": '<div class="note">Шинэ нууц үгүүд таарахгүй байна.</div>',
            "short": f'<div class="note">Шинэ нууц үг хамгийн багадаа {auth.MIN_PASSWORD} тэмдэгт байх ёстой.</div>'}.get(msg, "")
    tg = "унтраасан (мэдэгдэл вэб дээр: 📨 Мэдэгдэл, сүүлийн 7 хоног)"
    return page("Тохиргоо", f"""<h1>⚙️ Тохиргоо</h1>{note}
<div class="card"><b>Telegram:</b> {tg}<br><b>Зах зээл:</b> {esc(', '.join(user.markets) or '–')}
<br><b>Нэг арилжааны эрсдэл:</b> {f'{user.risk_per_trade:.2%}' if user.risk_per_trade else '0.5% (стандарт, зах зээл бүрийн дээд хязгаар)'}
<p class="muted">Арилжааг зогсоох/эхлүүлэх Telegram командууд (/stop, /resume, /closeall) Telegram-тай хамт унтарсан.</p></div>
<h2>Нууц үг солих</h2>
<form class="box card" method="post" action="/settings/password">
<label>Одоогийн нууц үг</label><input name="current" type="password" autocomplete="current-password" required>
<label>Шинэ нууц үг (≥ {auth.MIN_PASSWORD} тэмдэгт)</label><input name="new1" type="password" autocomplete="new-password" required>
<label>Шинэ нууц үг дахин</label><input name="new2" type="password" autocomplete="new-password" required>
<button type="submit">Солих</button></form>""", user, "settings")


@app.post("/settings/password")
def change_password(request: Request, current: str = Form(...), new1: str = Form(...), new2: str = Form(...)):
    user = current_user(request)
    if not user:
        return need_login()
    ip = request.client.host if request.client else "?"
    if not auth.check_login(user.name, current, ip):
        return RedirectResponse("/settings?msg=bad", status_code=303)
    if new1 != new2:
        return RedirectResponse("/settings?msg=mismatch", status_code=303)
    try:
        auth.set_password(user.name, new1)
    except ValueError:
        return RedirectResponse("/settings?msg=short", status_code=303)
    resp = RedirectResponse("/settings?msg=ok", status_code=303)
    resp.set_cookie(COOKIE, auth.make_session(user.name), max_age=auth.SESSION_DAYS * 86400, httponly=True,
                    samesite="lax", secure=request.headers.get("x-forwarded-proto", request.url.scheme) == "https")
    return resp


# ---------------- admin ----------------
@app.get("/admin")
def admin(request: Request):
    user = current_user(request)
    if not user:
        return need_login()
    if not user.is_admin:
        return page("Хандах эрхгүй", "<h1>Хандах эрхгүй</h1>", user)
    everyone = users_mod.load_users(markets.AVAILABLE)
    heads = "".join(f"<th>{esc(markets.get(n).label)}</th>" for n in markets.AVAILABLE)
    rows = []
    for u in everyone:
        cells = []
        for n in markets.AVAILABLE:
            with market(n):
                active = u.name in {t.name for t in mkt().traders(everyone)}
            if n not in u.markets:
                cells.append('<td class="muted">—</td>')
            elif not active:
                cells.append('<td class="down">түлхүүр алга</td>')
            else:
                s = _summary(n, u.name)
                cells.append(f'<td><a href="/m/{n}?user={esc(u.name)}">{money(s.get("equity"))}</a> · '
                             f'{s.get("open", 0)} pos{" ⏸" if s.get("paused") else ""}</td>')
        rows.append(f"<tr><td><b>{esc(u.name)}</b>{' 👑' if u.is_admin else ''}</td>"
                    f"<td>{'✅' if auth.has_password(u.name) else '❌'}</td>"
                    f"<td>{'✅' if u.telegram_chat_id else '❌'}</td>"
                    f"<td>{f'{u.risk_per_trade:.2%}' if u.risk_per_trade else '1%'}</td>{''.join(cells)}</tr>")
    beats = []
    for n in markets.AVAILABLE:
        s = _summary(n, user.name)
        hb = s.get("heartbeat")
        age = (time.time() * 1000 - hb) / 60000 if hb else None
        state = "❓ хэзээ ч асаагүй" if age is None else ("✅" if age < 5 else f"⚠️ {age:.0f} мин чимээгүй")
        beats.append(f"<tr><td>{esc(markets.get(n).label)}</td><td>{esc(s.get('mode'))}</td><td>{state}</td></tr>")
    body = f"""<h1>👑 Админ</h1>
<h2>Хэрэглэгчид</h2><div class="card scroll"><table>
<tr><th>Нэр</th><th>Нууц үг</th><th>Telegram</th><th>Эрсдэл</th>{heads}</tr>{''.join(rows)}</table></div>
<h2>Ботууд</h2><div class="card scroll"><table><tr><th>Зах зээл</th><th>Горим</th><th>Сүүлд ажилласан</th></tr>{''.join(beats)}</table></div>
<h2>Шинэ хэрэглэгч нэмэх</h2><div class="card">
<p>1. Сервер дээрх <code>.env</code> файлд нэмнэ:</p>
<pre>USER{len(everyone) + 1}_NAME=bat
USER{len(everyone) + 1}_MARKETS=crypto
USER{len(everyone) + 1}_TELEGRAM_CHAT_ID=...
USER{len(everyone) + 1}_CRYPTO_BINANCE_KEY=...
USER{len(everyone) + 1}_CRYPTO_BINANCE_SECRET=...</pre>
<p>2. Нууц үг: <code>python main.py user password bat</code></p>
<p>3. Бот 5 минутын дотор шинэ хүнийг автоматаар олж, Telegram-аар мэдэгдэнэ.</p></div>"""
    return page("Админ", body, user, "admin")


# ---------------- public ----------------
@app.get("/public")
def public_page():
    everyone = users_mod.load_users(markets.AVAILABLE)
    lead = users_mod.primary(everyone)
    cards = []
    for n in markets.AVAILABLE:
        with market(n):
            try:
                p = dashboard.public(lead.name)
            except Exception as e:
                log.warning("public %s: %s", n, e)
                continue
        if p["return"] is None:
            continue
        since = time.strftime("%Y-%m-%d", time.gmtime(p["since"] / 1000)) if p["since"] else "–"
        cards.append(f"""<div class="card"><div class="muted">{esc(p['label'])} · {esc(p['mode'])}</div>
<div class="big">{pct(p['return'])}</div><div class="muted">{since}-аас хойш</div>
<div>{p['trades']} арилжаа · ялалт {('–' if p['win_rate'] is None else f"{p['win_rate']:.0%}")}
· дундаж {('–' if p['avg_r'] is None else f"{p['avg_r']:+.2f}R")}</div></div>""")
    body = ("<h1>🌍 Trader AI · гүйцэтгэл</h1><p class='muted'>Дүнг хувиар харуулав. Мөнгөн дүн, хувийн мэдээлэл "
            "харагдахгүй. Demo үр дүн нь бодит мөнгөний үр дүнг баталгаажуулахгүй. Санхүүгийн зөвлөгөө биш.</p>"
            f"<div class='grid'>{''.join(cards) or '<div class=card>Одоогоор өгөгдөл алга</div>'}</div>"
            "<p><a href='/login'>Нэвтрэх</a></p>")
    return page("Гүйцэтгэл", body)


def serve(host: str = "127.0.0.1", port: int | None = None):
    import uvicorn

    from .settings import DASHBOARD_PORT
    uvicorn.run(app, host=host, port=port or DASHBOARD_PORT, log_level="warning", proxy_headers=True,
                forwarded_allow_ips="127.0.0.1")
