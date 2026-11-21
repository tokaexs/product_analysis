"""Streamlit UI.  Run:  streamlit run code/app.py"""
import json
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from llm import GeminiLLM, LLMError, MockLLM          # noqa: E402
from pipeline import run_pipeline                      # noqa: E402
from settings import MODEL, SETTINGS, _load_dotenv   # noqa: E402

_load_dotenv()

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "test-data", "sample_headphones.txt")
ICON = {"high": "🔴", "medium": "🟠", "low": "🟢"}

st.set_page_config(page_title="Product Review Analysis", page_icon="🛍️", layout="wide")
st.title("🛍️ Product Review Analysis")
st.caption("Paste your customer reviews → get themes, sentiment per theme, and what to fix first.")

with st.sidebar:
    st.header("Settings")
    saved_key = os.getenv("GEMINI_API_KEY", "")
    key = st.text_input("Gemini API key", type="password", value=saved_key,
                        help="Stays in memory only. Never commit keys to GitHub.")
    if saved_key:
        st.caption("🟢 API key loaded from .env")
    else:
        st.caption("🟡 No key in .env — enter one above or set in .env")
    votes = st.slider("Self-consistency votes per review", 1, 5, SETTINGS["P2"]["votes"],
                      help="Each review is classified this many times; the majority answer wins.")
    demo = st.checkbox("Offline demo mode (mock model, not real AI)", value=False)
    st.caption(f"Model: {MODEL}")

context = st.text_input("What do you sell? (optional)", placeholder="e.g. Wireless earbuds, Amazon India, budget segment")
if "text" not in st.session_state:
    st.session_state.text = ""
c1, c2 = st.columns([1, 1])
if c1.button("Load sample reviews"):
    st.session_state.text = open(SAMPLE, encoding="utf-8").read()
up = c2.file_uploader("…or upload a CSV with a 'review' column", type=["csv"])
if up is not None:
    df_up = pd.read_csv(up)
    col = "review" if "review" in df_up.columns else df_up.columns[0]
    st.session_state.text = "\n".join(df_up[col].astype(str).tolist())
text = st.text_area("Reviews (one per line)", key="text", height=220)

if st.button("Analyse reviews", type="primary"):
    try:
        active_key = (key or "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
        llm = MockLLM() if demo else GeminiLLM(api_key=active_key or None)
        bar = st.progress(0.0, text="Starting")
        res = run_pipeline(llm, text, context, votes=votes, workers=2,
                           progress=lambda p, m: bar.progress(min(p, 1.0), text=m))
        bar.empty()
        st.session_state.res = res
    except LLMError as exc:
        st.error(f"AI call failed: {exc}")
        st.session_state.res = None

res = st.session_state.get("res")
if res:
    if "error" in res:
        st.warning(res["error"])
        st.stop()
    cs = res["clean_stats"]
    m = st.columns(5)
    m[0].metric("Reviews analysed", res["n_valid"])
    m[1].metric("Ignored (not real reviews)", res["n_invalid"])
    m[2].metric("Duplicates removed", cs["duplicates"])
    m[3].metric("Positive", res["overall"].get("positive", 0))
    m[4].metric("Negative", res["overall"].get("negative", 0))
    if res["n_failed"]:
        st.info(f"{res['n_failed']} review(s) could not be processed after retries and were skipped.")
    st.subheader("Summary")
    st.write(res["summary"])

    st.subheader("Themes and sentiment")
    df = pd.DataFrame(res["theme_stats"])
    if not df.empty:
        st.bar_chart(df.set_index("theme")[["positive", "neutral", "mixed", "negative"]])
        st.dataframe(df[["theme", "mentions", "positive", "negative", "mixed", "neutral",
                         "complaint_share", "avg_agreement"]], hide_index=True, width="stretch")
        st.caption("avg_agreement = how often the repeated AI runs agreed (1.0 = fully consistent).")
        for s in res["theme_stats"]:
            if s["negative_evidence"] or s["positive_evidence"]:
                with st.expander(f"Evidence: {s['theme']}"):
                    for q in s["negative_evidence"]: st.markdown(f"- ❌ “{q}”")
                    for q in s["positive_evidence"]: st.markdown(f"- ✅ “{q}”")

    st.subheader("What to improve first")
    if not res["suggestions"]:
        st.success("No recurring complaints found.")
    for sg in res["suggestions"]:
        with st.container(border=True):
            st.markdown(f"**{ICON[sg['priority']]} {sg['priority'].title()} priority — {sg['theme']}**")
            st.write(sg["action"])
            st.caption(f"Why: {sg['why']}  ·  Expected impact: {sg['expected_impact']}")
            for q in sg["evidence"]: st.markdown(f"> “{q}”")

    st.download_button("Download full report (JSON)", json.dumps(res, ensure_ascii=False, indent=2),
                       file_name="review_analysis.json", mime="application/json")
