"""Global settings shared by every market (loaded from .env).

Market-specific things (symbols, timeframe, mode, keys, costs) live in
markets/<name>/config.yaml and the market adapter. Risk limits live in
risk_guard.py."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_ROOT = ROOT / "data"
DATA_ROOT.mkdir(exist_ok=True)

LIVE_CONFIRM = os.getenv("LIVE_CONFIRM", "").strip()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# News / research AI: "ollama" (local), "claude" (Anthropic API), or "off"
NEWS_AI = os.getenv("NEWS_AI", "ollama").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5-5").strip()
CRYPTOPANIC_TOKEN = os.getenv("CRYPTOPANIC_TOKEN", "").strip()
# Small local models write poor Mongolian, so with Ollama reviews default to English.
REVIEW_LANG = os.getenv("REVIEW_LANG", "Mongolian (Cyrillic)" if NEWS_AI == "claude" else "English").strip()

SHADOW_MIN_DAYS = float(os.getenv("SHADOW_MIN_DAYS", "10"))
BLACKOUT_BEFORE_MIN = int(os.getenv("BLACKOUT_BEFORE_MIN", "60"))
BLACKOUT_AFTER_MIN = int(os.getenv("BLACKOUT_AFTER_MIN", "60"))
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8765"))

LOOP_SECONDS = int(os.getenv("LOOP_SECONDS", "30"))
NEWS_EVERY_MIN = int(os.getenv("NEWS_EVERY_MIN", "30"))
IMPROVE_HOUR_UTC = int(os.getenv("IMPROVE_HOUR_UTC", "1"))


def base_asset(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTC', 'EUR/USD' -> 'EUR'."""
    return symbol.split("/")[0]

# From this date the bots check weekly whether the AI news scores were right
# (core/news_eval.py); the web report page shows the result. Owner, 2026-10-19.
NEWS_EVAL_FROM = os.getenv("NEWS_EVAL_FROM", "2026-11-19")
