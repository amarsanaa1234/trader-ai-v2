"""Carry, more ways (owner, 2026-10-26).

FX: the live FX Strategy 2 (markets/fx/strategy2.py: weekly, 3 long / 3 short by
interest rate, out when VIX jumps) next to variants:
  k     - 1, 2 or 3 currencies on each side
  hold  - re-ranked weekly (5 days) or monthly (21 days)
  guard - none / VIX jump / trend (keep a leg only if the currency's own 60-day
          move agrees with it - the classic fix for carry crashes) / both
Commodities: the same idea on the futures curve - long the 3 commodities in the
steepest backwardation (shortage, longs are paid), short the 3 in the deepest
contango (glut), using the curve history the commodities bot already keeps.

Settings are chosen on the research years (first 85%) only; the vault (newest
15%) and random rankings (same k / hold / costs) then judge the choice.

    python tools/carry_variants_test.py
"""
import itertools
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import markets  # noqa: E402
from core.context import activate  # noqa: E402

activate(markets.get("fx"))
from core.storage import Storage  # noqa: E402
from markets.fx import strategy2 as s2  # noqa: E402

YEAR, VAULT, RANDOM_RUNS = 252, 0.15, 300


def stats(r):
    if len(r) < 20 or r.std() == 0:
        return dict(ret=0.0, sharpe=0.0, dd=0.0)
    eq = r.cumsum()
    return dict(ret=float(r.mean() * YEAR), sharpe=float(r.mean() / r.std() * np.sqrt(YEAR)),
                dd=float((eq.cummax() - eq).max()))


def rank_weights(score: pd.DataFrame, k: int, hold: int, rng=None) -> pd.DataFrame:
    cols = list(score.columns)
    w = np.zeros(score.shape)
    last = np.zeros(len(cols))
    vals = score.to_numpy()
    for i in range(len(score)):
        if i % hold == 0 and np.isfinite(vals[i]).all():
            order = np.argsort(vals[i]) if rng is None else rng.permutation(len(cols))
            last = np.zeros(len(cols))
            last[order[-k:]] = 1.0 / k
            last[order[:k]] = -1.0 / k
        w[i] = last
    return pd.DataFrame(w, index=score.index, columns=cols)


def vol_target(raw: pd.Series) -> pd.Series:
    vol = raw.rolling(60, min_periods=20).std().shift(1) * np.sqrt(YEAR)
    return (raw * (0.08 / vol).clip(upper=s2.MAX_SCALE)).fillna(0.0)


def judge(name, r, cut):
    a, b = stats(r[r.index < cut]), stats(r[r.index >= cut])
    return {"name": name, "research": a, "vault": b, "years": {y: float(g.sum()) for y, g in r.groupby(r.index.year)}}


def show(rows, title):
    print(f"\n===== {title}")
    print(f"{'variant':34} | {'RESEARCH ret/yr Sharpe maxDD':29} | {'VAULT ret/yr Sharpe maxDD':26}")
    for x in rows:
        a, b = x["research"], x["vault"]
        print(f"{x['name']:34} | {a['ret']:+8.1%} {a['sharpe']:+6.2f} {a['dd']:6.1%}   | "
              f"{b['ret']:+8.1%} {b['sharpe']:+6.2f} {b['dd']:6.1%}")


def random_p(score, k, hold, returns_fn, cut, real, post=None):
    """Share of random rankings (same k, hold, guards, costs) at least as good."""
    rng = np.random.default_rng(5)
    res, vau = [], []
    for _ in range(RANDOM_RUNS):
        w = rank_weights(score, k, hold, rng)
        if post is not None:
            w = post(w)
        r = returns_fn(w)
        res.append(stats(r[r.index < cut])["sharpe"])
        vau.append(stats(r[r.index >= cut])["sharpe"])
    return float(np.mean(np.array(res) >= real["research"]["sharpe"])), \
        float(np.mean(np.array(vau) >= real["vault"]["sharpe"]))


def fx_part():
    st = Storage()
    lp, rt, vix = s2.load(st)
    cut = lp.index[int(len(lp) * (1 - VAULT))]
    fear = (vix / vix.shift(s2.VIX_DAYS) > s2.VIX_JUMP).shift(1, fill_value=False)
    calm = (~fear).astype(float)
    mom = lp - lp.shift(60)
    rel = mom.sub(mom.mean(axis=1), axis=0)

    def returns(w):
        return s2.daily_returns(lp, rt, w)[0]

    def guarded(guard):
        def post(w):
            if "trend" in guard:
                w = w.where(np.sign(w) == np.sign(rel), 0.0)
            if "vix" in guard:
                w = w.mul(calm, axis=0)
            return w
        return post

    rows = []
    for k, hold, guard in itertools.product((1, 2, 3), (5, 21), ("none", "vix", "trend", "vix+trend")):
        w = guarded(guard)(rank_weights(rt, k, hold))
        name = f"k{k} {'weekly' if hold == 5 else 'monthly'} {guard}" + ("  <- LIVE" if (k, hold, guard) == (3, 5, "vix") else "")
        rows.append((judge(name, returns(w), cut), k, hold, guard))
    rows.sort(key=lambda x: -x[0]["research"]["sharpe"])
    show([x[0] for x in rows], f"FX carry ({len(lp)} days {lp.index[0].date()}..{lp.index[-1].date()}, vault from {cut.date()})")
    best, k, hold, guard = rows[0]
    live = next(x for x in rows if x[1:] == (3, 5, "vix"))
    for label, (res, kk, hh, gg) in (("best on research", rows[0]), ("live", live)):
        pr, pv = random_p(rt, kk, hh, returns, cut, res, guarded(gg))
        print(f"{label:16} {res['name']}: by year {', '.join(f'{y} {v:+.1%}' for y, v in res['years'].items())}")
        print(f"{'':16} random rankings: research p={pr:.3f}, vault p={pv:.3f}")


def commodities_part():
    activate(markets.get("commodities"))
    from markets.commodities import curve
    m = markets.get("commodities")
    st = Storage()
    closes, carry = {}, {}
    for s in m.symbols:
        df = pd.read_sql_query("SELECT ts, close FROM candles WHERE symbol=? AND tf='4h' ORDER BY ts", st.db,
                               params=(s,))
        c = df.set_index(pd.to_datetime(df["ts"], unit="ms", utc=True))["close"]
        closes[s] = c.groupby(c.index.normalize()).last()
        k = curve.load(st, s)
        if not k.empty:
            k.index = pd.to_datetime(k.index, unit="ms", utc=True).normalize()
            carry[s] = k.groupby(k.index).last()
    px = pd.DataFrame(closes).dropna(how="all")
    px = px[px.index.dayofweek < 5].ffill()
    cy = pd.DataFrame(carry).reindex(px.index).ffill(limit=10)
    keep = cy.notna().sum(axis=1) >= 6
    px, cy = px[keep], cy[keep]
    cy = cy.apply(lambda row: row.fillna(row.median()), axis=1)   # a missing curve ranks in the middle
    cut = px.index[int(len(px) * (1 - VAULT))]
    ret = np.log(px).diff().fillna(0.0)
    spread = pd.Series({s: 2 * m.costs_for(s)["maker"] for s in px.columns})
    mom = np.log(px) - np.log(px).shift(60)

    def returns(w):
        wy = w.shift(1).fillna(0.0)
        raw = (wy * ret).sum(axis=1) + (wy * cy).sum(axis=1) / 365 * (7 / 5) \
            - (w.diff().abs().fillna(0.0) * spread).sum(axis=1)
        return vol_target(raw)

    def guarded(guard):
        return (lambda w: w.where(np.sign(w) == np.sign(mom), 0.0)) if guard == "trend" else (lambda w: w)

    rows = []
    for k, hold, guard in itertools.product((1, 2, 3), (5, 21), ("none", "trend")):
        w = guarded(guard)(rank_weights(cy, k, hold))
        rows.append((judge(f"k{k} {'weekly' if hold == 5 else 'monthly'} {guard}", returns(w), cut), k, hold, guard))
    rows.sort(key=lambda x: -x[0]["research"]["sharpe"])
    show([x[0] for x in rows], f"Commodities curve carry ({len(px)} days {px.index[0].date()}..{px.index[-1].date()}, "
                               f"vault from {cut.date()})")
    res, k, hold, guard = rows[0]
    pr, pv = random_p(cy, k, hold, returns, cut, res, guarded(guard))
    print(f"best on research {res['name']}: by year {', '.join(f'{y} {v:+.1%}' for y, v in res['years'].items())}")
    print(f"{'':16} random rankings: research p={pr:.3f}, vault p={pv:.3f}")


if __name__ == "__main__":
    fx_part()
    commodities_part()
