"""Thin LLM layer: Gemini for real runs, MockLLM for offline plumbing tests."""
import json
import os
import sys
import re
import time

from settings import MODEL


class LLMError(Exception):
    pass


class GeminiLLM:
    def __init__(self, model=None, api_key=None):
        from google import genai  # lazy import so tests run without the package
        from settings import _load_dotenv
        _load_dotenv()
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key or key in ("your-key-here", "YOUR_KEY"):
            raise LLMError("GEMINI_API_KEY is not set (see .env.example)")
        from google.genai import types as _t
        self.client = genai.Client(api_key=key, http_options=_t.HttpOptions(timeout=60000))
        self.model = model or MODEL
        self.calls = 0

    def generate(self, system, user, temperature=0.2, json_mode=True):
        from google.genai import types
        kwargs = dict(system_instruction=system or None)
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        cfg = types.GenerateContentConfig(**kwargs)
        last = None
        for attempt in range(6):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=user, config=cfg)
                self.calls += 1
                time.sleep(float(os.getenv('LLM_DELAY', '0')))
                return resp.text or ""
            except Exception as exc:  # network / quota: back off and retry
                last = exc
                print(f'[LLM retry {attempt + 1}/6] {str(exc)[:120]}', file=sys.stderr)
                time.sleep(min(45, 4 * 2 ** attempt))
        raise LLMError(f"Gemini call failed after 6 attempts: {last}")


def parse_json(text, salvage=True):
    """Parse model output. salvage=False is the strict mode used by the eval script."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        if not salvage:
            raise
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


# ---------------- Mock (keyword based; plumbing tests only, NOT for submission results) ----------------
_THEME_KEYS = {
    "Battery life": ["battery", "charge", "charging"],
    "Sound quality": ["sound", "bass", "audio", "music"],
    "Shipping & delivery": ["deliver", "shipping", "arrived"],
    "Packaging": ["packag", "packing", "box"],
    "Product quality": ["quality", "broke", "stopped working", "build"],
    "Price/value": ["price", "value", "expensive", "cheap", "paisa"],
    "Customer service": ["seller", "support", "service", "refund"],
}
_POS = ["amazing", "great", "good", "love", "fast", "excellent", "achhi", "badiya", "best", "fantastic"]
_NEG = ["dies", "broke", "bad", "poor", "late", "kharab", "worst", "terrible", "damaged", "tuta", "slow", "waste"]


class MockLLM:
    calls = 0

    def generate(self, system, user, temperature=0.2, json_mode=True):
        MockLLM.calls += 1
        if "PROMPT_ID: P1" in (system or ""):
            return json.dumps({"themes": [{"name": n, "definition": f"Comments about {n.lower()}"}
                                          for n in list(_THEME_KEYS)[:6]]})
        if "PROMPT_ID: P3" in (system or ""):
            stats = json.loads(user.split("Theme statistics (JSON):")[1])
            sug = []
            for s in stats:
                ev = (s.get("negative_evidence") or [])[:1]
                sug.append({"theme": s["theme"], "priority": "high" if s["complaint_share"] >= 0.3 else "medium",
                            "action": f"Review and improve {s['theme'].lower()}.",
                            "why": f"{s['complaint_share']:.0%} of mentions were negative.",
                            "evidence": ev, "expected_impact": "Fewer complaints"})
            return json.dumps({"summary": "Mock summary.", "suggestions": sug[:5]})
        return self._classify(user)

    def _classify(self, user):
        m = re.search(r"<review>(.*?)</review>", user, re.S) or re.search(r"Review:\s*(.*)", user, re.S)
        review = m.group(1).strip()
        low = review.lower()
        if len(low.split()) < 3 or not re.search(r"[a-z]{3,}", low):
            return json.dumps({"is_valid_review": False, "language": "English", "aspects": [],
                               "overall_sentiment": "neutral"})
        aspects = []
        for sent in re.split(r"[.!?,;]| but | par | lekin ", review):
            s = sent.strip()
            if not s:
                continue
            for theme, keys in _THEME_KEYS.items():
                if any(k in s.lower() for k in keys):
                    pos = any(w in s.lower() for w in _POS)
                    neg = any(w in s.lower() for w in _NEG)
                    sentiment = "mixed" if pos and neg else "positive" if pos else "negative" if neg else "neutral"
                    if not any(a["theme"] == theme for a in aspects):
                        aspects.append({"theme": theme, "sentiment": sentiment,
                                        "evidence": " ".join(s.split()[:12])})
        pos = sum(a["sentiment"] == "positive" for a in aspects)
        neg = sum(a["sentiment"] == "negative" for a in aspects)
        overall = "mixed" if pos and neg else "positive" if pos else "negative" if neg else "neutral"
        return json.dumps({"is_valid_review": True, "language": "English", "aspects": aspects,
                           "overall_sentiment": overall})
