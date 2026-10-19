"""People, passwords and who may see what. Uses a throw-away database and a
fake .env, never the real ones."""
import pytest
from fastapi.testclient import TestClient

from core import auth
from core import users as users_mod

ENV = {
    "USER1_NAME": "amar", "USER1_ROLE": "admin", "USER1_MARKETS": "crypto",
    "USER2_NAME": "bat", "USER2_MARKETS": "crypto", "USER2_RISK_PER_TRADE": "0.005",
    "USER2_CRYPTO_BINANCE_KEY": "k2", "USER2_CRYPTO_BINANCE_SECRET": "s2",
    "USER3_NAME": "Bad Name!",                      # invalid: ignored
    "USER4_NAME": "amar",                           # duplicate: ignored
}


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(users_mod, "_env", lambda: dict(ENV))


@pytest.fixture
def authdb(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "DB", tmp_path / "app.db")
    monkeypatch.setattr(auth, "SECRET_FILE", tmp_path / "secret.key")


def test_users_from_env(env):
    us = users_mod.load_users(("crypto",))
    assert [u.name for u in us] == ["amar", "bat"]
    assert us[0].is_admin and not us[1].is_admin
    assert us[1].risk_per_trade == 0.005
    assert us[1].cred("crypto", "binance_key") == "k2"
    assert us[0].cred("crypto", "binance_key") == ""


def test_first_user_becomes_admin(monkeypatch):
    monkeypatch.setattr(users_mod, "_env", lambda: {"USER1_NAME": "a1", "USER2_NAME": "b2"})
    assert users_mod.load_users()[0].is_admin


def test_legacy_single_user(monkeypatch):
    monkeypatch.setattr(users_mod, "_env", lambda: {})
    us = users_mod.load_users(("crypto",))
    assert len(us) == 1 and us[0].legacy and us[0].is_admin and us[0].markets == ["crypto"]


def test_who_trades_a_market(env, monkeypatch):
    from core.context import mkt
    monkeypatch.setenv("CRYPTO_MODE", "demo")          # demo needs keys
    traders = [u.name for u in mkt().traders(users_mod.load_users(("crypto",)))]
    assert traders == ["bat"]                           # amar has no keys
    monkeypatch.setenv("CRYPTO_MODE", "signal")        # paper needs none
    assert [u.name for u in mkt().traders(users_mod.load_users(("crypto",)))] == ["amar", "bat"]


def test_user_risk_can_only_be_lower():
    from core import risk_guard as rg
    base = dict(equity=1000.0, entry=100.0, stop=98.0, side=1, open_positions=0, open_notional=0.0,
                trades_today=0, halted=False, peak_equity=1000.0)
    assert rg.check_entry(**base, risk_per_trade=0.0025).qty == pytest.approx(rg.check_entry(**base).qty / 2)
    assert rg.check_entry(**base, risk_per_trade=0.05).qty == pytest.approx(rg.check_entry(**base).qty)


def test_passwords_and_lockout(authdb):
    with pytest.raises(ValueError):
        auth.set_password("amar", "short")
    auth.set_password("amar", "correct horse battery")
    assert auth.check_login("amar", "correct horse battery", "1.1.1.1")
    for _ in range(auth.MAX_FAILS):
        assert not auth.check_login("amar", "wrong", "1.1.1.1")
    assert auth.locked("amar")
    assert not auth.check_login("amar", "correct horse battery", "1.1.1.1")   # locked even with the right one


def test_sessions(authdb):
    auth.set_password("amar", "correct horse battery")
    cookie = auth.make_session("amar")
    assert auth.read_session(cookie) == "amar"
    assert auth.read_session(cookie[:-1] + ("0" if cookie[-1] != "0" else "1")) is None   # tampered
    auth.set_password("amar", "another long password")
    assert auth.read_session(cookie) is None            # a new password logs old sessions out


def _login(client, name, pw):
    return client.post("/login", data={"username": name, "password": pw}, follow_redirects=False)


def test_web_access_rules(env, authdb, monkeypatch):
    from core import dashboard, web
    monkeypatch.setattr(dashboard, "data", lambda user: {"user": user})
    monkeypatch.setattr(dashboard, "summary", lambda user: {"label": "x", "mode": "signal", "equity": None})
    auth.set_password("amar", "admin password 123")
    auth.set_password("bat", "user password 123")

    c = TestClient(web.app)
    assert c.get("/", follow_redirects=False).status_code == 303              # must log in
    assert c.get("/api/m/crypto/data").status_code == 401
    assert _login(c, "bat", "nope").headers["location"].endswith("err=bad")
    assert _login(c, "bat", "user password 123").headers["location"] == "/"
    assert c.get("/api/m/crypto/data").json() == {"user": "bat"}             # own account
    assert c.get("/api/m/crypto/data?user=amar").status_code == 403          # not someone else's
    assert "Хандах эрхгүй" in c.get("/admin").text                           # not admin
    assert "Админ" not in c.get("/").text.split("<main")[0]                  # no admin menu item

    a = TestClient(web.app)
    _login(a, "amar", "admin password 123")
    assert a.get("/api/m/crypto/data?user=bat").json() == {"user": "bat"}    # admins may look
    assert "Хэрэглэгчид" in a.get("/admin").text
    assert a.get("/public").status_code == 200
