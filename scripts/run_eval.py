"""Replay the prompt iteration log with REAL runs (parallel, saves after every version).

  python3 -W ignore scripts/run_eval.py --runs 3 --workers 4
"""
import argparse, csv, json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from llm import GeminiLLM, MockLLM, parse_json            # noqa: E402
from pipeline import _majority, validate_classification    # noqa: E402
from prompts_io import load_prompt, render                 # noqa: E402
from settings import MODEL, SENTIMENTS, VERSION_SETTINGS   # noqa: E402

EVAL_THEMES = ["Sound quality", "Battery life", "Shipping & delivery", "Packaging",
               "Product quality", "Sizing/fit", "Price/value", "Customer service",
               "Description accuracy", "Other"]


def extract(text):
    """Return (json_ok, schema_ok, parsed_or_None, plain_label_or_None)."""
    try:
        obj = parse_json(text, salvage=False)
    except Exception:
        words = [w for w in SENTIMENTS if re.search(r"\b" + w + r"\b", (text or "").lower())]
        return False, False, None, (words[0] if len(words) == 1 else None)
    try:
        return True, True, validate_classification(obj, EVAL_THEMES, ""), None
    except Exception:
        return True, False, None, None


def run_version(llm, ver, rows, runs, workers, delay):
    system, user_t = load_prompt("P2_classify.txt", ver)
    cfg = VERSION_SETTINGS[ver]

    def call(task):
        r, k = task
        user = render(user_t, themes=json.dumps(EVAL_THEMES), review=r["review"])
        try:
            text = llm.generate(system, user, cfg["temperature"], cfg["json_mode"])
        except Exception as exc:       # API failure: reported separately, not counted as a prompt failure
            print("   API call failed:", str(exc)[:90], file=sys.stderr)
            text = None
        if delay:
            time.sleep(delay)
        return r, k, text

    tasks = [(r, k) for r in rows for k in range(runs)]
    outs = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, o in enumerate(pool.map(call, tasks), 1):
            outs.append(o)
            if i % 12 == 0 or i == len(tasks):
                print(f"  {ver}: {i}/{len(tasks)} calls done", flush=True)

    m = dict(n=0, json=0, schema=0, ev_ok=0, ev_n=0, acc=0, acc_n=0, val=0, val_n=0, api_fail=0)
    labels_by_review, raw = {}, []
    for r, k, text in outs:
        raw.append({"version": ver, "id": r["id"], "run": k, "raw": text})
        if text is None:
            m["api_fail"] += 1
            continue
        j_ok, s_ok, parsed, word = extract(text)
        m["n"] += 1; m["json"] += j_ok; m["schema"] += s_ok
        if parsed:
            for asp in parsed["aspects"]:
                m["ev_n"] += 1
                m["ev_ok"] += bool(asp["evidence"]) and asp["evidence"].lower() in r["review"].lower()
            m["val_n"] += 1
            m["val"] += (parsed["is_valid_review"] == (r["valid"] == "1"))
        label = parsed["overall_sentiment"] if parsed and parsed["is_valid_review"] else word
        if r["valid"] == "1":
            m["acc_n"] += 1; m["acc"] += (label == r["overall_gold"])
            labels_by_review.setdefault(r["id"], {"gold": r["overall_gold"], "labels": []})["labels"].append(label)
    stab = [len(set(v["labels"])) == 1 for v in labels_by_review.values()]
    maj = [_majority([l or "none" for l in v["labels"]])[0] == v["gold"] for v in labels_by_review.values()]
    pct = lambda x, y: round(100 * x / y, 1) if y else None
    row = {"version": ver, "model": MODEL, "temperature": cfg["temperature"], "json_mode": cfg["json_mode"],
           "runs_per_review": runs, "calls": m["n"], "api_failures": m["api_fail"],
           "json_valid_%": pct(m["json"], m["n"]), "schema_valid_%": pct(m["schema"], m["n"]),
           "overall_accuracy_%": pct(m["acc"], m["acc_n"]), "validity_accuracy_%": pct(m["val"], m["val_n"]),
           "stability_%": pct(sum(stab), len(stab)), "evidence_verbatim_%": pct(m["ev_ok"], m["ev_n"]),
           "majority_accuracy_%": pct(sum(maj), len(maj))}
    return row, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["v1", "v2", "v3", "final"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delay", type=float, default=0.0)
    ap.add_argument("--mock", action="store_true", help="plumbing test only; results are meaningless")
    a = ap.parse_args()
    llm = MockLLM() if a.mock else GeminiLLM()
    print("model:", MODEL)
    rows = list(csv.DictReader(open(ROOT / "test-data" / "labeled_reviews.csv", encoding="utf-8")))
    res = ROOT / "test-data" / "results"
    res.mkdir(exist_ok=True)
    f = res / "prompt_eval.csv"
    existing = {}
    if f.exists():
        existing = {r["version"]: r for r in csv.DictReader(open(f, encoding="utf-8"))}
    for ver in a.versions:
        row, raw = run_version(llm, ver, rows, a.runs, a.workers, a.delay)
        print(row, flush=True)
        existing[ver] = row
        merged = [existing[v] for v in ("v1", "v2", "v3", "final") if v in existing]
        with open(f, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row)); w.writeheader(); w.writerows(merged)
        json.dump(raw, open(res / ("prompt_eval_raw_" + ver + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("saved", res / "prompt_eval.csv")


if __name__ == "__main__":
    main()
