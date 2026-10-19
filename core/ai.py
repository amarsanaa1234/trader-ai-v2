"""One place to ask the AI for JSON: local Ollama or the Claude API.
The answer is always forced into a JSON schema, so the AI can only fill in
data, never run code or place orders."""
import json
import logging
import os

import requests

from .settings import ANTHROPIC_API_KEY, CLAUDE_MODEL, NEWS_AI, OLLAMA_MODEL, OLLAMA_URL

log = logging.getLogger(__name__)
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")


_warned = False


def model_name() -> str:
    """Which AI scores the news (stored with every score, to compare AIs later)."""
    return f"claude:{CLAUDE_MODEL}" if NEWS_AI == "claude" else f"ollama:{OLLAMA_MODEL}" if NEWS_AI == "ollama" else "off"


def enabled() -> bool:
    """True if an AI backend is configured and reachable."""
    global _warned
    if NEWS_AI == "claude":
        return bool(ANTHROPIC_API_KEY)
    if NEWS_AI == "ollama":
        try:
            requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).raise_for_status()
            return True
        except Exception:
            if not _warned:
                log.warning("Ollama ажиллахгүй байна (%s). Мэдээ/стратеги судлах AI алгасагдана. "
                            "https://ollama.com суулгаад `ollama pull %s` ажиллуулна уу.", OLLAMA_URL, OLLAMA_MODEL)
                _warned = True
            return False
    return False


def _ollama(system: str, user: str, schema: dict):
    options = {"temperature": 0}
    if os.getenv("OLLAMA_THREADS"):  # on a small server: leave a core free for the bot
        options["num_thread"] = int(os.getenv("OLLAMA_THREADS"))
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=1800, json={
        "model": OLLAMA_MODEL, "stream": False, "format": schema, "options": options,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    })
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def _claude(system: str, user: str, schema: dict):
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise RuntimeError("NEWS_AI=claude боловч ANTHROPIC_API_KEY байхгүй")
    client = anthropic.Anthropic()
    kwargs = {}
    if CLAUDE_MODEL in FALLBACK_MODELS:  # server-side refusal fallback
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    resp = client.beta.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=system,
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": user}],
        **kwargs,
    )
    if resp.stop_reason == "refusal":
        log.warning("Claude татгалзлаа: %s", getattr(resp, "stop_details", None))
        return None
    if resp.stop_reason == "max_tokens":
        log.warning("Claude хариу max_tokens-д тасарлаа")
        return None
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


def ask_json(system: str, user: str, schema: dict):
    """Returns the parsed JSON object, or None if the AI is off or declined."""
    if NEWS_AI == "claude":
        return _claude(system, user, schema)
    if NEWS_AI == "ollama":
        return _ollama(system, user, schema)
    return None
