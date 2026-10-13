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


def esc(x) -> str:
    return html.escape(str(x if x is not None else ""))


def nav(user, active: str) -> str:
    items = [("home", "/", "🏠 Нүүр")]
    for name in visible_markets(user):
        items.append((name, f"/m/{name}", markets.get(name).label))
    items += [("reports", "/reports", "📊 Тайлан"), ("settings", "/settings", "⚙️ Тохиргоо")]
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
<style>{CSS}</style></head><body>{top}<main class="page">{body}</main></body></html>""",
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
    inject = f"<style>{NAV_CSS}</style>{nav(user, name)}<script>window.API_URL = {json.dumps(api)};</script>"
    if who != user.name:
        inject += f'<div style="max-width:1200px;margin:8px auto;padding:0 16px" class="muted">👑 {esc(who)}-ийн данс</div>'
    text = MARKET_PAGE.read_text(encoding="utf-8")
    text = text.replace("<body>", "<body>" + inject, 1).replace('fetch("/api/data")', "fetch(window.API_URL)")
    text = text.replace("<title>Crypto AI Dashboard</title>", f"<title>{esc(markets.get(name).label)} · Trader AI</title>")
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


# ---------------- reports ----------------
@app.get("/reports")
def reports(request: Request):
    user = current_user(request)
    if not user:
        return need_login()
    from . import review
    parts = []
    for name in visible_markets(user):
        with market(name):
            try:
                text = review.weekly_report(Storage(), mkt().mode, user.name)
            except Exception as e:
                text = f"–  ({e})"
            parts.append(f"<h2>{esc(mkt().label)}</h2><pre>{esc(text)}</pre>")
    return page("Тайлан", "<h1>📊 7 хоногийн тайлан</h1>" + "".join(parts), user, "reports")


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
    tg = "✅ холбогдсон" if user.telegram_chat_id else "❌ холбогдоогүй (админд chat ID-гаа өгнө үү)"
    return page("Тохиргоо", f"""<h1>⚙️ Тохиргоо</h1>{note}
<div class="card"><b>Telegram:</b> {tg}<br><b>Зах зээл:</b> {esc(', '.join(user.markets) or '–')}
<br><b>Нэг арилжааны эрсдэл:</b> {f'{user.risk_per_trade:.2%}' if user.risk_per_trade else '1% (стандарт)'}
<p class="muted">Арилжааг зогсоох/эхлүүлэх нь Telegram-аар: /stop, /resume, /closeall</p></div>
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
