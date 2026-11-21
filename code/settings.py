"""All tunable settings in one place so they can be quoted in the prompt sheets."""
import os
from pathlib import Path

def _load_dotenv():
    for f in [
        Path(__file__).resolve().parent.parent / ".env",
        Path.cwd() / ".env",
    ]:
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    val = v.strip().strip('"').strip("'")
                    cur = os.environ.get(k, "")
                    if not cur or cur in ("your-key-here", "YOUR_KEY"):
                        if val and val not in ("your-key-here", "YOUR_KEY"):
                            os.environ[k] = val

_load_dotenv()


MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Final-solution settings per prompt (quoted in the Section 2 prompt sheets)
SETTINGS = {
    "P1": dict(temperature=0.2, json_mode=True),
    "P2": dict(temperature=0.7, json_mode=True, votes=3),   # votes = self-consistency runs
    "P3": dict(temperature=0.3, json_mode=True),
}

# Settings used ONLY by scripts/run_eval.py to replay the iteration log
VERSION_SETTINGS = {
    "v1":    dict(temperature=None, json_mode=False),
    "v2":    dict(temperature=None, json_mode=False),
    "v3":    dict(temperature=0.2,  json_mode=False),
    "final": dict(temperature=0.7,  json_mode=True),
}

MAX_REVIEWS = 60       # cost / time guard
MAX_CHARS = 1500       # per review
THEME_SAMPLE = 40      # reviews shown to P1
SENTIMENTS = ("positive", "negative", "neutral", "mixed")
