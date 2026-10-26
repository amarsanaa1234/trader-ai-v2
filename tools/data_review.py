"""Everything the bots have collected, checked against what the price did next (read-only).

Run on the server from /opt/trader_ai_v2 (the data lives there):
    python tools/data_review.py

Each check is an event study or an information coefficient (IC = correlation of a
signal with the following return), with a p-value from shuffling (how often a
random signal looks at least as good). Small samples are flagged: two weeks of
liquidations can hint, not prove.
"""
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
RNG = np.random.default_rng(3)
CRYPTO = {"BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT", "BNB": "BNB/USDT:USDT", "XRP": "XRP/USDT:USDT",
          "SOL": "SOL/USDT:USDT"}


def closes(db, symbol, tf):
    d = pd.read_sql_query("SELECT ts, close FROM candles WHERE symbol=? AND tf=? ORDER BY ts", db, params=(symbol, tf))
    return d.set_index("ts")["close"]


def fwd(c: pd.Series, at_ms, bars: int):
    """Log return from the first close at/after each time to `bars` closes later."""
    idx = np.searchsorted(c.index.to_numpy(), np.asarray(at_ms))
    ok = idx + bars < len(c)
    out = np.full(len(idx), np.nan)
    v = np.log(c.to_numpy())
    out[ok] = v[idx[ok] + bars] - v[idx[ok]]
    return out


def perm_p(x, y, stat, n=2000):
    """p-value: how often shuffled pairs give a statistic at least as large."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 10:
        return np.nan, len(x)
    real = stat(x, y)
    hits = sum(stat(RNG.permutation(x), y) >= real for _ in range(n))
    return (hits + 1) / (n + 1), len(x)


def ic(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def hit(x, y):
    return float(np.mean(np.sign(x) == np.sign(y)))


def line(name, x, y, unit="", note=""):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        print(f"  {name:46} too few ({ok.sum()})")
        return
    p, n = perm_p(x, y, ic)
    print(f"  {name:46} n={n:6}  IC {ic(x[ok], y[ok]):+.3f}  sign right {hit(x[ok], y[ok]):.0%}  p={p:.3f} {note}")


def section(t):
    print(f"\n===== {t}")


# ---------------- crypto ----------------
def liquidations():
    section("1. Crypto liquidations (2 weeks): is a spike a reversal signal?")
    lq = sqlite3.connect("data/crypto/liquidations.db")
    bot = sqlite3.connect("data/crypto/bot.db")
    d = pd.read_sql_query("SELECT ts, symbol, liquidated, usd FROM liquidations", lq)
    d["coin"] = d["symbol"].str.extract(r"^(BTC|ETH|BNB|XRP|SOL)")[0]
    d = d.dropna(subset=["coin"])
    d["hour"] = d["ts"] // 3_600_000 * 3_600_000
    rows = []
    for coin, g in d.groupby("coin"):
        h = g.pivot_table(index="hour", columns="liquidated", values="usd", aggfunc="sum").fillna(0.0)
        full = pd.RangeIndex(int(h.index.min()), int(h.index.max()) + 3_600_000, 3_600_000)
        h = h.reindex(full, fill_value=0.0)
        c = closes(bot, CRYPTO[coin], "1h")   # (the 15m history stopped on 10/10)
        for side in ("long", "short"):
            if side not in h:
                continue
            s = h[side]
            base = s.rolling(72, min_periods=24).median().shift(1)
            spike = (s > 5 * base.clip(lower=1)) & (s > s.quantile(0.95))
            at = (s.index[spike] + 3_600_000).to_numpy()            # decided at the hour's end
            for bars, lab in ((1, "1h"), (4, "4h"), (12, "12h"), (24, "24h")):
                r = fwd(c, at, bars)
                base_r = fwd(c, (s.index + 3_600_000).to_numpy(), bars)
                rows.append((coin, side, lab, r, base_r))
    print("  spike = hourly liquidations > 5x the previous 3 days' median and in the top 5%")
    print("  longs liquidated = forced selling (price fell); 'reversal' means price then goes UP (and the mirror)")
    for side in ("long", "short"):
        for lab in ("1h", "4h", "12h", "24h"):
            rs = np.concatenate([r for _, sd, l, r, _b in rows if sd == side and l == lab])
            bs = np.concatenate([b for _, sd, l, _r, b in rows if sd == side and l == lab])
            rs, bs = rs[np.isfinite(rs)], bs[np.isfinite(bs)]
            want = 1 if side == "long" else -1                      # reversal direction
            rev = want * rs
            # p: how often random hours give an average reversal at least this big
            sims = [want * RNG.choice(bs, len(rs)).mean() for _ in range(2000)] if len(rs) else [0]
            p = (np.sum(np.array(sims) >= rev.mean()) + 1) / (len(sims) + 1) if len(rs) else np.nan
            print(f"  {side:5} liq spike -> next {lab:3}: {len(rs):3} spikes | avg move in reversal direction "
                  f"{rev.mean() * 100 if len(rs) else 0:+.2f}% | reversed {np.mean(rev > 0) if len(rs) else 0:.0%} | p={p:.3f}")


def crypto_positioning():
    section("2. Crypto crowd positioning (Binance long/short ratios, open interest; ~5 weeks hourly)")
    bot = sqlite3.connect("data/crypto/bot.db")
    pos = pd.read_sql_query("SELECT * FROM positioning ORDER BY ts", bot)
    for col, lab, sign in (("global_ls", "all accounts long/short ratio (contrarian)", -1),
                           ("top_ls", "top traders long/short ratio", 1),
                           ("oi_value", "open interest 24h change", 1)):
        xs, ys = [], []
        for coin, sym in CRYPTO.items():
            g = pos[pos.symbol == sym].set_index("ts")[col].dropna()
            if len(g) < 100:
                continue
            x = g.pct_change(24) if col == "oi_value" else (g - g.rolling(168, min_periods=48).mean()) / g.rolling(168, min_periods=48).std()
            c = closes(bot, sym, "1h")
            xs.append(sign * x.to_numpy())
            ys.append(fwd(c, x.index.to_numpy() + 3_600_000, 24))
        if xs:
            line(lab + " -> next 24h", np.concatenate(xs), np.concatenate(ys), note="(hourly, overlapping)")


def crypto_funding_fng_premium():
    section("3. Crypto funding, Fear & Greed, Coinbase premium (2 years)")
    bot = sqlite3.connect("data/crypto/bot.db")
    xs, ys = [], []
    for coin, sym in CRYPTO.items():
        f = pd.read_sql_query("SELECT ts, rate FROM funding WHERE symbol=? ORDER BY ts", bot, params=(sym,)).set_index("ts")["rate"]
        c = closes(bot, sym, "1h")
        xs.append(-f.to_numpy())                                  # contrarian: high funding -> down
        ys.append(fwd(c, f.index.to_numpy(), 72))
    x, y = np.concatenate(xs), np.concatenate(ys)
    line("funding (contrarian) -> next 3 days", x, y)
    q = np.nanquantile(-x, 0.9)
    m = -x >= q
    print(f"  {'top 10% funding (crowded longs) -> next 3 days':46} avg {np.nanmean(y[m]) * 100:+.2f}% vs all {np.nanmean(y) * 100:+.2f}%")
    fng = pd.read_sql_query("SELECT ts, value FROM fng ORDER BY ts", bot).set_index("ts")["value"]
    c = closes(bot, CRYPTO["BTC"], "4h")
    fng = fng[fng.index >= c.index[0]]
    at = fng.index.to_numpy() + 86_400_000                          # published the next day
    line("Fear&Greed (contrarian) -> BTC next 7 days", -(fng.to_numpy() - 50), fwd(c, at, 42))
    for name, cond in (("extreme fear (<20)", fng < 20), ("extreme greed (>80)", fng > 80)):
        r = fwd(c, at[cond.to_numpy()], 42)
        print(f"  {name + ' -> BTC next 7 days':46} n={np.isfinite(r).sum():4} avg {np.nanmean(r) * 100:+.2f}%")
    cb = pd.read_sql_query("SELECT product, ts, close FROM coinbase_candles ORDER BY ts", bot)
    for prod, sym in (("BTC-USD", CRYPTO["BTC"]), ("ETH-USD", CRYPTO["ETH"])):
        s = cb[cb["product"] == prod].set_index("ts")["close"]
        b = closes(bot, sym, "1h").reindex(s.index)
        prem = np.log(s / b)
        z = (prem - prem.rolling(168, min_periods=48).mean()) / prem.rolling(168, min_periods=48).std()
        line(f"Coinbase premium z ({prod}) -> next 24h", z.to_numpy(), fwd(closes(bot, sym, "1h"), z.index.to_numpy() + 3_600_000, 24))


def news(market, symbols_tf):
    bot = sqlite3.connect(f"data/{market}/bot.db")
    s = pd.read_sql_query("SELECT ts, asset, score FROM sentiment WHERE asset != 'MARKET' AND score != 0", bot)
    if s.empty:
        print(f"  {market}: no scored headlines")
        return
    s["day"] = s["ts"] // 86_400_000
    g = s.groupby(["asset", "day"]).agg(score=("score", "mean"), ts=("ts", "max")).reset_index()   # one vote per asset-day
    xs, ys = [], []
    for asset, gg in g.groupby("asset"):
        sym = symbols_tf(asset)
        if sym is None:
            continue
        c = closes(bot, *sym)
        if c.empty:
            continue
        bars = 24 if sym[1] == "1h" else 6
        xs.append(gg["score"].to_numpy())
        ys.append(fwd(c, gg["ts"].to_numpy(), bars))
    if xs:
        line(f"{market}: AI headline score -> next 24h (per asset-day)", np.concatenate(xs), np.concatenate(ys))


def all_news():
    section("4. AI news scores (crypto since 10/09, FX 10/17, commodities 10/19)")
    news("crypto", lambda a: (CRYPTO[a], "1h") if a in CRYPTO else None)
    news("fx", lambda a: (a, "4h") if "_" in a else None)
    news("commodities", lambda a: (a, "4h") if "_" in a else None)


def oanda_crowd():
    section("5. OANDA clients' positions and waiting orders (2 years, every 6h)")
    for market in ("fx", "commodities"):
        db = sqlite3.connect(f"data/{market}/bot.db")
        for table, col, lab, sign in (("book", "long_pct", "clients % long (contrarian)", -1),
                                      ("orders", "imbalance", "waiting orders imbalance", 1)):
            t = pd.read_sql_query(f"SELECT symbol, ts, {col} v FROM {table} ORDER BY ts", db)
            xs, ys = [], []
            for sym, g in t.groupby("symbol"):
                c = closes(db, sym, "4h")
                if c.empty:
                    continue
                v = g.set_index("ts")["v"]
                z = (v - v.rolling(120, min_periods=30).mean()) / v.rolling(120, min_periods=30).std()
                xs.append(sign * z.to_numpy())
                ys.append(fwd(c, v.index.to_numpy(), 6))
            if xs:
                line(f"{market}: {lab} -> next 24h", np.concatenate(xs), np.concatenate(ys))


def cot():
    section("6. COT: big speculators' futures positions (10 years, weekly)")
    for market, pairs in (("fx", {"EUR": "EUR_USD", "GBP": "GBP_USD", "AUD": "AUD_USD", "NZD": "NZD_USD",
                                    "JPY": ("USD_JPY", -1), "CAD": ("USD_CAD", -1), "CHF": ("USD_CHF", -1)}),
                          ("commodities", {"XAU": "XAU_USD", "XAG": "XAG_USD", "XPT": "XPT_USD", "XCU": "XCU_USD",
                                           "BCO": "BCO_USD", "NATGAS": "NATGAS_USD", "CORN": "CORN_USD",
                                           "WHEAT": "WHEAT_USD"})):
        db = sqlite3.connect(f"data/{market}/bot.db")
        k = pd.read_sql_query("SELECT ccy, ts, net_pct FROM cot ORDER BY ts", db)
        for horizon, bars, lab in ((24, 30, "next 5 days"), (120, 120, "next 4 weeks")):
            xs_c, xs_m, ys = [], [], []
            for ccy, sym in pairs.items():
                sign = 1
                if isinstance(sym, tuple):
                    sym, sign = sym
                c = closes(db, sym, "4h")
                g = k[k.ccy == ccy].set_index("ts")["net_pct"]
                if c.empty or len(g) < 60:
                    continue
                z = (g - g.rolling(156, min_periods=52).mean()) / g.rolling(156, min_periods=52).std()
                r = sign * fwd(c, g.index.to_numpy(), bars)
                xs_c.append(-z.to_numpy())
                xs_m.append(g.diff(4).to_numpy())
                ys.append(r)
            line(f"{market}: speculators extreme (contrarian) -> {lab}", np.concatenate(xs_c), np.concatenate(ys))
            line(f"{market}: speculators adding (4-week change) -> {lab}", np.concatenate(xs_m), np.concatenate(ys))


def swap_vs_curve():
    section("7. Commodities: OANDA's real swap vs the futures curve the backtest uses (1 week)")
    db = sqlite3.connect("data/commodities/bot.db")
    fh = pd.read_sql_query("SELECT symbol, ts, long_rate, short_rate FROM financing_hist ORDER BY ts", db)
    cv = pd.read_sql_query("SELECT symbol, ts, carry FROM curve ORDER BY ts", db)
    for sym, g in fh.groupby("symbol"):
        last = g.iloc[-1]
        cc = cv[cv.symbol == sym]
        curve_now = cc["carry"].iloc[-1] if len(cc) else np.nan
        sym_h = (last["short_rate"] - last["long_rate"]) / 2
        print(f"  {sym:11} OANDA long {last['long_rate']:+.1%}/yr short {last['short_rate']:+.1%}/yr "
              f"(symmetric {-sym_h:+.1%}) | curve {curve_now:+.1%}")


def vix_spikes():
    section("8. VIX jumps (fear) -> what FX and gold did over the next 5 days (8 years)")
    db = sqlite3.connect("data/fx/bot.db")
    v = pd.read_sql_query("SELECT ts, v FROM vix ORDER BY ts", db).set_index("ts")["v"]
    v = v[v.index >= 1_540_000_000_000]
    jump = v / v.shift(5) > 1.25
    at = v.index[jump].to_numpy()
    for sym, lab in (("USD_JPY", "USD/JPY (yen as a safe haven: down?)"), ("AUD_USD", "AUD/USD (risk currency: down?)"),
                     ("XAU_USD", "gold")):
        c = closes(db, sym, "4h")
        if c.empty:
            continue
        r = fwd(c, at, 30)
        base = fwd(c, v.index.to_numpy(), 30)
        print(f"  {lab:46} n={np.isfinite(r).sum():3} avg {np.nanmean(r) * 100:+.2f}% (any day {np.nanmean(base) * 100:+.2f}%)")


if __name__ == "__main__":
    for fn in (liquidations, crypto_positioning, crypto_funding_fng_premium, all_news, oanda_crowd, cot,
               swap_vs_curve, vix_spikes):
        try:
            fn()
        except Exception as e:
            print(f"  ! {fn.__name__} failed: {e}")
