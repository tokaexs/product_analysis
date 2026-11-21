"""Product Review Analysis pipeline.

Stages (see docs/diagrams):
  clean_reviews -> P1 discover_themes -> P2 classify_review (x votes, majority vote)
  -> aggregate (pure Python) -> P3 suggest -> report
"""
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from llm import LLMError, parse_json
from prompts_io import load_prompt, render
from settings import (MAX_CHARS, MAX_REVIEWS, SENTIMENTS, SETTINGS, THEME_SAMPLE)

DEFAULT_THEMES = [{"name": n, "definition": d} for n, d in [
    ("Product quality", "How well the product is made and works"),
    ("Durability", "How long the product lasts, breakage"),
    ("Price/value", "Cost compared with what you get"),
    ("Shipping & delivery", "Speed and condition on arrival"),
    ("Packaging", "Box, wrapping and protection"),
    ("Customer service", "Seller support, refunds, replacements"),
    ("Description accuracy", "Product matches photos and listing"),
    ("Other", "Anything that fits no other theme")]]

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------- cleaning (no LLM)
def clean_reviews(raw):
    """raw: str (one review per line / blank-line separated) or list[str]. Returns (reviews, stats)."""
    if isinstance(raw, str):
        parts = re.split(r"\n\s*\n", raw.strip()) if "\n\n" in raw.strip() else raw.splitlines()
    else:
        parts = list(raw)
    stats = {"received": len(parts), "empty": 0, "duplicates": 0, "truncated": 0, "capped": 0}
    seen, out = set(), []
    for p in parts:
        t = " ".join(str(p).split())
        if not t:
            stats["empty"] += 1
            continue
        key = re.sub(r"[^\w\s]", "", t.lower()).strip()   # ignore case + punctuation
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        if len(t) > MAX_CHARS:
            t = t[:MAX_CHARS]
            stats["truncated"] += 1
        out.append(t)
    if len(out) > MAX_REVIEWS:
        stats["capped"] = len(out) - MAX_REVIEWS
        out = out[:MAX_REVIEWS]
    stats["kept"] = len(out)
    return out, stats


# ---------------------------------------------------------------- P1
def discover_themes(llm, reviews, business_context=""):
    system, user = load_prompt("P1_theme_discovery.txt")
    numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(reviews[:THEME_SAMPLE]))
    user = render(user, business_context=business_context or "Not provided", reviews=numbered)
    cfg = SETTINGS["P1"]
    for _ in range(2):
        try:
            data = parse_json(llm.generate(system, user, cfg["temperature"], cfg["json_mode"]))
            themes, names = [], set()
            for t in data["themes"]:
                name = str(t["name"]).strip()
                if name and name.lower() not in names and name.lower() != "other":
                    names.add(name.lower())
                    themes.append({"name": name, "definition": str(t.get("definition", "")).strip()})
            if len(themes) >= 2:
                return themes[:8] + [{"name": "Other", "definition": "Anything that fits no other theme"}]
        except (ValueError, KeyError, TypeError):
            continue
    return DEFAULT_THEMES   # fallback: P1 found no recurring themes (e.g. unrelated text)


# ---------------------------------------------------------------- P2
def validate_classification(obj, theme_names, review):
    """Normalise one P2 output. Raises ValueError if it breaks the schema."""
    if not isinstance(obj, dict) or not isinstance(obj.get("is_valid_review"), bool):
        raise ValueError("is_valid_review missing or not boolean")
    overall = str(obj.get("overall_sentiment", "neutral")).lower()
    if overall not in SENTIMENTS:
        raise ValueError(f"bad overall_sentiment {overall!r}")
    aspects, used = [], set()
    low_review = review.lower()
    for a in obj.get("aspects") or []:
        theme = str(a.get("theme", "")).strip()
        sent = str(a.get("sentiment", "")).lower()
        if theme not in theme_names:
            theme = "Other" if "Other" in theme_names else None
        if theme is None or sent not in SENTIMENTS or theme in used:
            continue
        ev = str(a.get("evidence", "")).strip()
        used.add(theme)
        aspects.append({"theme": theme, "sentiment": sent, "evidence": ev,
                        "evidence_verified": bool(ev) and ev.lower() in low_review})
    if not obj["is_valid_review"]:
        aspects, overall = [], "neutral"
    return {"is_valid_review": obj["is_valid_review"], "language": str(obj.get("language", "")),
            "aspects": aspects, "overall_sentiment": overall}


def _majority(labels):
    """Majority label; a tie becomes 'mixed'. Returns (label, share_of_votes_for_it)."""
    counts = Counter(labels)
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "mixed", top[0][1] / len(labels)
    return top[0][0], top[0][1] / len(labels)


def classify_once(llm, review, themes_json, theme_names, prompt_version="final", settings=None):
    system, user = load_prompt("P2_classify.txt", prompt_version)
    user = render(user, themes=themes_json, review=review)
    cfg = settings or SETTINGS["P2"]
    for attempt in range(2):  # one retry on bad JSON
        raw = llm.generate(system, user, cfg["temperature"], cfg["json_mode"])
        try:
            return validate_classification(parse_json(raw), theme_names, review), attempt
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return None, 1


def classify_review(llm, review, themes, votes=None):
    """Self-consistency: run P2 `votes` times and keep what the majority agrees on."""
    votes = votes or SETTINGS["P2"]["votes"]
    names = [t["name"] for t in themes]
    themes_json = json.dumps(names, ensure_ascii=False)
    runs = []
    for _ in range(votes):
        res, _retry = classify_once(llm, review, themes_json, names)
        if res:
            runs.append(res)
    if not runs:
        return {"review": review, "failed": True, "is_valid": False}
    valid_votes = [r for r in runs if r["is_valid_review"]]
    if len(valid_votes) * 2 <= len(runs):
        return {"review": review, "failed": False, "is_valid": False,
                "agreement": 1 - len(valid_votes) / len(runs) if valid_votes else 1.0}
    overall, overall_agree = _majority([r["overall_sentiment"] for r in valid_votes])
    aspects = []
    for theme in names:
        mentions = [a for r in valid_votes for a in r["aspects"] if a["theme"] == theme]
        if len(mentions) * 2 <= len(valid_votes):   # needs a strict majority of runs
            continue
        sent, agree = _majority([a["sentiment"] for a in mentions])
        ev = next((a["evidence"] for a in mentions
                   if a["sentiment"] == sent and a["evidence_verified"]), "")
        aspects.append({"theme": theme, "sentiment": sent, "agreement": round(agree, 2), "evidence": ev})
    return {"review": review, "failed": False, "is_valid": True, "overall": overall,
            "agreement": round(overall_agree, 2), "aspects": aspects}


# ---------------------------------------------------------------- aggregation (no LLM)
def aggregate(results, themes):
    valid = [r for r in results if r.get("is_valid")]
    stats = []
    for t in themes:
        rows = [(r, a) for r in valid for a in r["aspects"] if a["theme"] == t["name"]]
        if not rows:
            continue
        c = Counter(a["sentiment"] for _, a in rows)
        n = len(rows)
        complaints = c["negative"] + c["mixed"]
        stats.append({
            "theme": t["name"], "definition": t["definition"], "mentions": n,
            "share_of_reviews": round(n / len(valid), 2) if valid else 0,
            "positive": c["positive"], "negative": c["negative"],
            "neutral": c["neutral"], "mixed": c["mixed"],
            "complaint_share": round(complaints / n, 2),
            "dominant": _majority([a["sentiment"] for _, a in rows])[0],
            "avg_agreement": round(sum(a["agreement"] for _, a in rows) / n, 2),
            "negative_evidence": [a["evidence"] for _, a in rows
                                  if a["sentiment"] in ("negative", "mixed") and a["evidence"]][:4],
            "positive_evidence": [a["evidence"] for _, a in rows
                                  if a["sentiment"] == "positive" and a["evidence"]][:2],
        })
    stats.sort(key=lambda s: s["mentions"], reverse=True)
    overall = Counter(r["overall"] for r in valid)
    return stats, dict(overall)


# ---------------------------------------------------------------- P3
def suggest(llm, theme_stats, n_reviews, business_context=""):
    problem = [s for s in theme_stats if s["negative"] + s["mixed"] > 0]
    if not problem:   # nothing to fix: skip the LLM call
        return {"summary": "No recurring complaints were found in these reviews.", "suggestions": []}
    payload = [{k: s[k] for k in ("theme", "mentions", "complaint_share", "negative", "mixed",
                                  "positive", "negative_evidence")} for s in problem]
    system, user = load_prompt("P3_improvements.txt")
    user = render(user, business_context=business_context or "Not provided", n_reviews=n_reviews,
                  theme_stats=json.dumps(payload, ensure_ascii=False, indent=1))
    cfg = SETTINGS["P3"]
    allowed_quotes = {q for s in problem for q in s["negative_evidence"]}
    allowed_themes = {s["theme"] for s in problem}
    for _ in range(2):
        try:
            data = parse_json(llm.generate(system, user, cfg["temperature"], cfg["json_mode"]))
            sugs = []
            for s in data["suggestions"]:
                pr = str(s.get("priority", "")).lower()
                if s.get("theme") not in allowed_themes or pr not in PRIORITY_ORDER:
                    continue
                sugs.append({"theme": s["theme"], "priority": pr, "action": str(s["action"]),
                             "why": str(s.get("why", "")),
                             "evidence": [q for q in s.get("evidence", []) if q in allowed_quotes],
                             "expected_impact": str(s.get("expected_impact", ""))})
            sugs.sort(key=lambda s: PRIORITY_ORDER[s["priority"]])
            return {"summary": str(data.get("summary", "")), "suggestions": sugs[:5]}
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    raise LLMError("Improvement generation failed twice (P3 returned invalid JSON)")


# ---------------------------------------------------------------- orchestrator
def run_pipeline(llm, raw_reviews, business_context="", votes=None, progress=None, workers=4):
    reviews, clean_stats = clean_reviews(raw_reviews)
    if not reviews:
        return {"error": "No usable reviews found. Paste at least one review (one per line).",
                "clean_stats": clean_stats}
    notify = progress or (lambda *_: None)
    notify(0.05, "Discovering themes (P1)")
    themes = discover_themes(llm, reviews, business_context)
    results = [None] * len(reviews)

    def work(i):
        return i, classify_review(llm, reviews[i], themes, votes)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, res in pool.map(work, range(len(reviews))):
            results[i] = res
            done += 1
            notify(0.05 + 0.75 * done / len(reviews), f"Classifying reviews (P2) {done}/{len(reviews)}")
    theme_stats, overall = aggregate(results, themes)
    n_valid = sum(1 for r in results if r.get("is_valid"))
    if n_valid == 0:
        return {"error": "None of the inputs looked like genuine product reviews.",
                "clean_stats": clean_stats, "results": results}
    notify(0.85, "Writing improvement suggestions (P3)")
    sug = suggest(llm, theme_stats, n_valid, business_context)
    notify(1.0, "Done")
    return {"clean_stats": clean_stats, "themes": themes, "results": results,
            "theme_stats": theme_stats, "overall": overall, "n_valid": n_valid,
            "n_invalid": len(results) - n_valid, "n_failed": sum(1 for r in results if r.get("failed")),
            "summary": sug["summary"], "suggestions": sug["suggestions"]}
