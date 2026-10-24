"""Self-improvement with three safety gates:

1. Walk-forward: candidates are ranked on 4 consecutive history windows by
   how well and how consistently they did (see validation.py).
2. Vault: the best few must also beat the live settings on the newest 15% of
   history, which ranking never looks at.
3. Shadow: survivors are forward-tested on new live candles (shadow.py) and
   only take over after they beat the live settings there too.

Search width depends on dopamine: low dopamine = explore wider.
"""
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor

from . import library, mistakes, patterns, portfolio, shadow, significance, validation
from .features import add_static_features
from .context import mkt
from .strategy import clamp_params, load_params, param_space, save_params

log = logging.getLogger(__name__)

MIN_VAULT_TRADES = 10
MAX_VAULT_DD = 0.20
PROMOTE_MARGIN = 0.25
MIN_POSITIVE_FOLDS = 3
ENSEMBLE_SIZE = 5


def min_vault_trades() -> int:
    """A market may demand more vault trades (config validation.min_vault_trades)."""
    return int(((getattr(mkt(), "cfg", None) or {}).get("validation") or {}).get("min_vault_trades", MIN_VAULT_TRADES))


def extra_gates(frames: dict, cand: dict) -> tuple[bool, dict]:
    """Market-requested checks beyond the per-symbol vault (config "validation"):
    portfolio: the shared-account vault must also make money with a tolerable drawdown;
    max_p_value: the edge must beat random entries (significance.py)."""
    cfg = (getattr(mkt(), "cfg", None) or {}).get("validation") or {}
    out, ok = {}, True
    if cfg.get("portfolio"):
        pv = portfolio.vault(frames, cand)
        out["portfolio"] = pv
        ok &= pv["return"] > 0 and pv["max_dd"] < MAX_VAULT_DD
    if ok and cfg.get("max_p_value") is not None:
        sig = significance.random_benchmark(frames, cand, n=int(cfg.get("random_runs", 100)))
        out["significance"] = sig
        ok &= sig["p_value"] <= float(cfg["max_p_value"])
    return bool(ok), out


def mutate(p: dict, scale: float, rng: random.Random) -> dict:
    q = {k: v for k, v in p.items() if k != "ensemble"}
    for k, (lo, hi, _) in param_space().items():
        if rng.random() < 0.4:
            q[k] = q[k] + rng.gauss(0, scale * (hi - lo))
    return clamp_params(q)


def random_params(rng: random.Random, base: dict) -> dict:
    q = {k: v for k, v in base.items() if k != "ensemble"}
    for k, (lo, hi, _) in param_space().items():
        q[k] = rng.uniform(lo, hi)
    return clamp_params(q)


def load_frames(st, symbols) -> dict:
    frames = {s: mkt().load_frame(st, s) for s in symbols}
    frames = {s: f for s, f in frames.items() if len(f) > 3000}
    library.ensure_seeded()
    library.review(frames)
    strats = library.approved()
    return {s: add_static_features(f, strats) for s, f in frames.items()}


# ---- parallel walk-forward: each worker process prepares the frames once ----
_WORKER = {}


def _init_worker(market_name, symbols):
    # a worker is a fresh Python process: activate the same market first
    import markets
    from .context import activate
    from .storage import Storage
    activate(markets.get(market_name))
    st = Storage()
    frames = {s: mkt().load_frame(st, s) for s in symbols}
    strats = library.approved()
    _WORKER["frames"] = {s: add_static_features(f, strats) for s, f in frames.items() if len(f) > 3000}


def _walk_forward_worker(params):
    return validation.walk_forward(_WORKER["frames"], params)


def walk_forward_many(frames: dict, candidates: list) -> list:
    """walk_forward() for every candidate, spread over the CPU cores
    (keeps 4 cores free so the laptop and the live bot stay responsive)."""
    workers = min(20, max(1, (os.cpu_count() or 2) - 4))
    if workers == 1 or len(candidates) < 40:
        return [validation.walk_forward(frames, c) for c in candidates]
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(mkt().name, list(frames))) as pool:
            return list(pool.map(_walk_forward_worker, candidates, chunksize=max(1, len(candidates) // (workers * 4))))
    except Exception as e:  # never let parallelism break the daily job
        log.warning("Зэрэгцээ тооцоолол амжилтгүй (%s), нэг цөмөөр үргэлжлүүлж байна", e)
        return [validation.walk_forward(frames, c) for c in candidates]


def improve(st, symbols, dopamine: float = 0.0, seed: int | None = None) -> dict:
    rng = random.Random(seed)
    frames = load_frames(st, symbols)
    if not frames:
        return {"promoted": False, "reason": "түүхэн өгөгдөл хангалтгүй (эхлээд `sync` ажиллуул)"}

    # Which chart patterns actually worked (never measured on the vault).
    edges = patterns.measure_edges(frames, 1 - validation.VAULT_FRACTION)

    if dopamine > 0.3:
        scale, n = 0.05, 600
    elif dopamine > -0.2:
        scale, n = 0.10, 1000
    else:
        scale, n = 0.20, 1600
    log.info("Сайжруулалт: допамин=%.2f -> %d хувилбар, хайлтын өргөн=%.2f", dopamine, n, scale)

    current = load_params()
    current["pattern_edges"] = edges
    cur_wf = validation.walk_forward(frames, current)
    cur_vault = validation.vault(frames, current)

    candidates = [mutate(current, scale, rng) for _ in range(int(n * 0.8))]
    candidates += [random_params(rng, current) for _ in range(n - len(candidates))]
    for c in candidates:
        c["pattern_edges"] = edges
    results = walk_forward_many(frames, candidates)
    scored = sorted(zip(results, candidates), key=lambda x: -x[0]["score"])

    # Mistake memory: for the best few, look at where their losing trades cluster
    # (research windows only) and try staying out of those situations.
    lessons = []
    for wf, cand in list(scored[:3]):
        better, learned = mistakes.learn(frames, cand, wf)
        if learned:
            scored.append((validation.walk_forward(frames, better), better))
            lessons.append(learned)
    scored.sort(key=lambda x: -x[0]["score"])

    # Ensemble: the 5 best settings vote (3 of 5 must agree). One "best" setting is often
    # partly luck; a majority of good ones is steadier.
    members = [{k: v for k, v in c.items() if k != "ensemble"} for _, c in scored[:ENSEMBLE_SIZE]]
    if len(members) == ENSEMBLE_SIZE:
        ens = {**members[0], "ensemble": members, "ensemble_votes": ENSEMBLE_SIZE // 2 + 1}
        scored.append((validation.walk_forward(frames, ens), ens))
        scored.sort(key=lambda x: -x[0]["score"])

    passed, checked = [], []
    for wf, cand in scored[:10]:
        if wf["score"] <= cur_wf["score"] or wf["positive_folds"] < MIN_POSITIVE_FOLDS:
            continue
        v = validation.vault(frames, cand)
        why = None
        if v["score"] <= cur_vault["score"] + PROMOTE_MARGIN:
            why = "vault оноо одоогийнхоос дээр биш"
        elif v["trades"] < min_vault_trades():
            why = "vault дээр арилжаа цөөн"
        elif v["return"] <= 0 or v["max_dd"] >= MAX_VAULT_DD:
            why = "vault алдагдалтай / уналт их"
        else:
            ok, extra = extra_gates(frames, cand)
            v.update(extra)
            if ok:
                passed.append((wf, v, cand))
            else:
                why = "нэг данс эсвэл санамсаргүйтэй харьцуулалтад тэнцээгүй"
        checked.append({"walk_forward": wf, "vault": v, "ensemble": bool(cand.get("ensemble")), "rejected": why})
        if len(passed) >= 3:
            break

    report = {"current": {"walk_forward": cur_wf, "vault": cur_vault}, "candidates": n,
              "dopamine": round(dopamine, 2), "search_width": scale,
              "best_walk_forward": scored[0][0] if scored else None, "pattern_edges": edges,
              "library_approved": [s["name"] for s in library.approved()], "new_shadows": [], "checked": checked,
              "lessons": lessons}

    if not mkt().params_path.exists() and passed:
        # very first run: nothing live to protect yet, use the best candidate directly
        wf, v, cand = passed[0]
        save_params(cand, {"walk_forward": wf, "vault": v}, note="initial settings")
        report["promoted"] = True
    else:
        for wf, v, cand in passed[:2]:
            s = shadow.add(st, cand, {"walk_forward": wf, "vault": v})
            if s is None:       # every place taken by a shadow still being tested
                report["shadows_full"] = report.get("shadows_full", 0) + 1
                continue
            report["new_shadows"].append({"id": s["id"], "walk_forward": wf, "vault": v})
        report["promoted"] = False
        # keep the live pattern edges fresh if that alone doesn't hurt the vault
        live = load_params()
        if edges != live.get("pattern_edges") and cur_vault["score"] >= validation.vault(frames, live)["score"]:
            save_params(current, {"walk_forward": cur_wf, "vault": cur_vault}, note="pattern edges refreshed")
            report["edges_refreshed"] = True

    # forward-test the shadows on new candles; one may take over
    sh = shadow.review(st, frames)
    report["shadow_review"] = sh["shadows"]
    if sh["promoted"]:
        report["promoted"] = True
        report["shadow_promoted"] = sh["promoted"]["id"]
    return report
