import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from llm import MockLLM, parse_json
from pipeline import clean_reviews, classify_review, run_pipeline, validate_classification, _majority

REVIEWS = [
    "Sound is amazing but the battery dies in 3 hours.",
    "Battery is terrible, charging takes forever.",
    "Great bass and excellent sound quality.",
    "Delivery was late and the box was damaged.",
    "idk",
]


def test_clean_dedupes_and_drops_empty():
    out, st = clean_reviews("Good phone\n\n  good PHONE \n\n\n")
    assert out == ["Good phone"] and st["duplicates"] == 1


def test_clean_empty_input():
    out, st = clean_reviews("   \n  ")
    assert out == [] and st["kept"] == 0


def test_parse_json_fences_and_salvage():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_validate_rejects_bad_schema():
    import pytest
    with pytest.raises(ValueError):
        validate_classification({"is_valid_review": "yes"}, ["A"], "x")


def test_validate_flags_unverified_evidence():
    obj = {"is_valid_review": True, "overall_sentiment": "negative",
           "aspects": [{"theme": "A", "sentiment": "negative", "evidence": "made up quote"}]}
    out = validate_classification(obj, ["A", "Other"], "the real review text")
    assert out["aspects"][0]["evidence_verified"] is False


def test_majority_tie_is_mixed():
    assert _majority(["positive", "negative"])[0] == "mixed"


def test_full_pipeline_with_mock():
    res = run_pipeline(MockLLM(), "\n".join(REVIEWS), votes=3)
    assert res["n_valid"] == 4 and res["n_invalid"] == 1
    battery = next(s for s in res["theme_stats"] if s["theme"] == "Battery life")
    assert battery["mentions"] == 2 and battery["negative"] + battery["mixed"] == 2
    assert res["suggestions"], "expected at least one suggestion"


def test_all_invalid_returns_error():
    res = run_pipeline(MockLLM(), "idk\nok\n??", votes=3)
    assert "error" in res


def test_empty_input_makes_no_llm_call():
    before = MockLLM.calls
    res = run_pipeline(MockLLM(), "", votes=3)
    assert "error" in res and MockLLM.calls == before
