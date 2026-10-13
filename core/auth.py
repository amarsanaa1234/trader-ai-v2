"""Web login: salted scrypt password hashes, signed session cookies, lockout.

Stored in data/app.db (shared by all markets). Nothing here is reversible to a
password. A session cookie carries the user name and the time the password
was last set, so changing a password logs out every old session.
"""
import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import time

from .settings import DATA_ROOT

DB = DATA_ROOT / "app.db"
SECRET_FILE = DATA_ROOT / "app_secret.key"
SESSION_DAYS = 14
MAX_FAILS, LOCK_MINUTES = 5, 15
MIN_PASSWORD = 10


def _db() -> sqlite3.Connection:
    db = sqlite3.connect(DB, timeout=30)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS passwords (user TEXT PRIMARY KEY, salt BLOB, hash BLOB, updated INTEGER);
        CREATE TABLE IF NOT EXISTS login_attempts (ts INTEGER, user TEXT, ip TEXT, ok INTEGER);""")
    return db


def _secret() -> bytes:
    if not SECRET_FILE.exists():
        SECRET_FILE.write_bytes(secrets.token_bytes(32))
        try:
            SECRET_FILE.chmod(0o600)
        except OSError:
            pass
    return SECRET_FILE.read_bytes()


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)


def set_password(user: str, password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"Нууц үг хамгийн багадаа {MIN_PASSWORD} тэмдэгт байх ёстой")
    salt = secrets.token_bytes(16)
    with _db() as db:
        db.execute("INSERT OR REPLACE INTO passwords VALUES (?, ?, ?, ?)",
                   (user, salt, _hash(password, salt), int(time.time() * 1000)))


def has_password(user: str) -> bool:
    with _db() as db:
        return db.execute("SELECT 1 FROM passwords WHERE user=?", (user,)).fetchone() is not None


def _version(user: str) -> int | None:
    with _db() as db:
        row = db.execute("SELECT updated FROM passwords WHERE user=?", (user,)).fetchone()
    return row["updated"] if row else None


def locked(user: str) -> bool:
    since = int((time.time() - LOCK_MINUTES * 60) * 1000)
    with _db() as db:
        fails = db.execute("SELECT COUNT(*) c FROM login_attempts WHERE user=? AND ok=0 AND ts>=?",
                           (user, since)).fetchone()["c"]
    return fails >= MAX_FAILS


def check_login(user: str, password: str, ip: str) -> bool:
    """True if the password is right and the account isn't locked. Every try is recorded."""
    if locked(user):
        return False
    with _db() as db:
        row = db.execute("SELECT salt, hash FROM passwords WHERE user=?", (user,)).fetchone()
        ok = bool(row) and hmac.compare_digest(_hash(password, row["salt"]), row["hash"])
        if not row:
            _hash(password, b"0" * 16)  # same work either way: don't reveal which names exist
        db.execute("INSERT INTO login_attempts VALUES (?, ?, ?, ?)", (int(time.time() * 1000), user, ip, int(ok)))
    return ok


def make_session(user: str) -> str:
    body = base64.urlsafe_b64encode(json.dumps(
        {"u": user, "v": _version(user), "exp": int(time.time()) + SESSION_DAYS * 86400}).encode()).decode()
    sig = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def read_session(cookie: str | None) -> str | None:
    """User name for a valid, unexpired cookie whose password hasn't changed since."""
    if not cookie or "." not in cookie:
        return None
    body, sig = cookie.rsplit(".", 1)
    if not hmac.compare_digest(hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest(), sig):
        return None
    try:
        data = json.loads(base64.urlsafe_b64decode(body.encode()))
    except Exception:
        return None
    if data.get("exp", 0) < time.time() or data.get("v") != _version(data.get("u", "")):
        return None
    return data["u"]
