"""Thin OANDA v20 REST client (requests only).

Practice (demo) and live accounts live on different hosts, and a practice token
only works on the practice host, so a demo token can never touch real money.
"""
import logging
import os
import time

import requests

log = logging.getLogger(__name__)
HOSTS = {"practice": "https://api-fxpractice.oanda.com/v3", "live": "https://api-fxtrade.oanda.com/v3"}


class OandaError(RuntimeError):
    pass


class Client:
    def __init__(self, token: str, account: str = "", env: str = "practice"):
        if not token:
            raise OandaError("OANDA token алга (.env: USER<n>_FX_OANDA_TOKEN)")
        self.base, self.account = HOSTS[env], account
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Accept-Datetime-Format": "UNIX",
                               "Content-Type": "application/json"})

    def req(self, method: str, path: str, **kw):
        for attempt in range(4):
            r = self.s.request(method, self.base + path, timeout=20, **kw)
            if r.status_code == 429 or r.status_code >= 500:   # rate limit / server hiccup: back off
                time.sleep(0.5 * 2 ** attempt)
                continue
            if not r.ok:
                raise OandaError(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
            return r.json()
        raise OandaError(f"{method} {path}: OANDA хариу өгөхгүй байна")

    def get(self, path, **params):
        return self.req("GET", path, params=params)

    def acct(self, path: str = "") -> str:
        return f"/accounts/{self.account}{path}"


_data_client = None


def data_client() -> Client:
    """Client for market data: FX_OANDA_TOKEN, else the first user with an fx token."""
    global _data_client
    if _data_client is None:
        token, account = os.getenv("FX_OANDA_TOKEN", "").strip(), os.getenv("FX_OANDA_ACCOUNT", "").strip()
        if not token:
            from core.users import load_users
            for u in load_users(("fx",)):
                if u.cred("fx", "oanda_token"):
                    token, account = u.cred("fx", "oanda_token"), u.cred("fx", "oanda_account")
                    break
        # prices are the same on both hosts; a practice token is the safe default for data
        _data_client = Client(token, account, os.getenv("FX_OANDA_ENV", "practice").strip().lower())
    return _data_client
