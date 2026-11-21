# Technical Report - Product Review Analysis (1-2 pages; export to PDF as TeamName_TechnicalReport.pdf)

## 1. Problem
Small e-commerce sellers receive many free-text reviews but cannot manually find recurring issues. Star ratings hide *why* customers are unhappy.

## 2. Architecture
Streamlit UI -> Python pipeline -> Gemini API. Pipeline: clean (code) -> P1 theme discovery -> P2 review classifier run 3 times per review -> majority vote and aggregation (code) -> P3 improvement advisor. See docs/diagrams/4_system_architecture.png.

## 3. Tools and models
Google Gemini 2.5 Flash (JSON output mode), Python 3, Streamlit, pandas, pytest. Free-tier friendly; all prompts are plain text files in prompts/.

## 4. Prompt design
- **Classification prompts**: fixed label sets (positive/negative/neutral/mixed) and an exact-theme whitelist, so outputs can be counted.
- **Structured output**: strict JSON schema, JSON mode, and code-side validation with one retry.
- **Self-consistency**: P2 runs 3 times at temperature 0.7; the majority label wins, ties become "mixed", and an agreement score is shown to the seller.
- **Grounding**: evidence quotes must appear verbatim in the review (checked in code); P3 may only cite quotes it was given.
- **Safety**: reviews sit inside <review> tags and are declared untrusted data (prompt-injection guard).
- **Division of labour**: the LLM judges language; Python does counting, ranking and percentages.

## 5. Test results
TODO: paste your real numbers: tests passed X/14 (list the failures), prompt-version results (JSON validity, accuracy, stability, majority vs single-run accuracy) from test-data/results/.

## 6. Limitations
TODO: write the limitations you actually observed (e.g. language coverage beyond English/Hinglish, 3x API cost for voting, themes are discovered from at most 40 reviews, sarcasm still fails sometimes).
