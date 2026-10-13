"""Users come from .env:

    USER1_NAME=amar            # login name: a-z, 0-9, _ ; first user is the main admin
    USER1_ROLE=admin           # admin | user (default user)
    USER1_MARKETS=crypto,fx    # which markets trade for this person
    USER1_TELEGRAM_CHAT_ID=... # where this person's messages go
    USER1_RISK_PER_TRADE=0.005 # optional, may only be LOWER than the global 1%
    USER1_CRYPTO_BINANCE_KEY=..., USER1_CRYPTO_BINANCE_SECRET=...   # market credentials:
    USER1_FX_OANDA_TOKEN=...                                         #   USER<n>_<MARKET>_<KEY>

.env is re-read on every call, so adding USER3_... while the bot runs makes
a new person appear within minutes (the bot announces it). Passwords never
live in .env: `python main.py user password <name>` stores a salted hash.

With no USER<n>_NAME at all, a single admin "admin" is used with the v1-style
keys (CRYPTO_BINANCE_KEY / BINANCE_API_KEY, TELEGRAM_CHAT_ID) so old setups
keep working.
"""
import os
import re
from dataclasses import dataclass, field

from dotenv import dotenv_values

from .settings import ROOT, TELEGRAM_CHAT_ID

NAME_RE = re.compile(r"^[a-z0-9_]{2,24}$")


@dataclass
class User:
    id: int
    name: str
    role: str = "user"
    markets: list = field(default_factory=list)
    telegram_chat_id: str = ""
    risk_per_trade: float | None = None
    env: dict = field(default_factory=dict, repr=False)   # this user's raw USER<n>_ values
    legacy: bool = False

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def cred(self, market: str, key: str) -> str:
        """USER<n>_<MARKET>_<KEY>; the legacy single user reads <MARKET>_<KEY> instead."""
        k = f"{market}_{key}".upper()
        if self.legacy:
            return (os.getenv(k) or "").strip()
        return (self.env.get(k) or "").strip()


def _env() -> dict:
    """Process environment, then the .env file read fresh (so edits show up without a restart)."""
    fresh = {k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None} if (ROOT / ".env").exists() else {}
    return {**{k: v for k, v in os.environ.items() if k.startswith("USER")}, **fresh}


def load_users(all_markets: tuple = ()) -> list[User]:
    env = _env()
    ids = sorted({int(m.group(1)) for k in env if (m := re.fullmatch(r"USER(\d+)_NAME", k))})
    users, seen = [], set()
    for n in ids:
        prefix = f"USER{n}_"
        mine = {k[len(prefix):]: v for k, v in env.items() if k.startswith(prefix)}
        name = (mine.get("NAME") or "").strip().lower()
        if not NAME_RE.match(name) or name in seen:
            continue  # bad or duplicate name: ignored (doctor command reports it)
        seen.add(name)
        rpt = mine.get("RISK_PER_TRADE")
        users.append(User(
            id=n, name=name, role=(mine.get("ROLE") or "user").strip().lower(),
            markets=[m.strip() for m in (mine.get("MARKETS") or "").split(",") if m.strip()],
            telegram_chat_id=(mine.get("TELEGRAM_CHAT_ID") or "").strip(),
            risk_per_trade=float(rpt) if rpt else None, env=mine))
    if users:
        if not any(u.is_admin for u in users):
            users[0].role = "admin"  # there is always an admin: the first user
        return users
    return [User(id=0, name="admin", role="admin", markets=list(all_markets),
                 telegram_chat_id=TELEGRAM_CHAT_ID, legacy=True)]


def primary(users: list[User]) -> User:
    """The main admin (lowest number). Market-wide messages and the learning
    signal (dopamine for the improver) follow this account."""
    return next(u for u in users if u.is_admin)


def by_name(users: list[User], name: str) -> User | None:
    return next((u for u in users if u.name == name), None)


def by_chat(users: list[User], chat_id) -> User | None:
    return next((u for u in users if u.telegram_chat_id and u.telegram_chat_id == str(chat_id)), None)
