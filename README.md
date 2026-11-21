# Product Review Analysis
Turns a pile of customer reviews into **themes, sentiment per theme and prioritised improvements**, built for small sellers on e-commerce platforms who have no data team.

**Prompt-engineering techniques used:** classification prompts, structured (JSON) output with validation, self-consistency (majority vote over repeated runs), few-shot examples, injection-resistant delimiters.

## How it works
`clean (code) -> P1 theme discovery -> P2 review classifier x3 votes -> majority vote + aggregation (code) -> P3 improvement advisor -> report`
Diagrams are in `docs/diagrams/`. Final prompts are in `prompts/final/`, earlier drafts in `prompts/v1..v3/`.

## Run it
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...            # free key from Google AI Studio (never commit it)
streamlit run code/app.py            # tick "Load sample reviews" then "Analyse reviews"
```
No key? Tick **Offline demo mode** in the sidebar to check the UI (mock model, not real AI).

## Reproduce the numbers in the report
```bash
python scripts/run_eval.py --versions v1 v2 v3 final --runs 3   # prompt iteration log numbers
python scripts/run_tests.py                                      # 14-row test table -> docs/test_table.md
pytest tests                                                     # unit tests (no API key needed)
```
## Team and links
Team: Tech_deck · Prototype: https://prdctanalysis.streamlit.app/ · Reel: https://youtu.be/CIMnPZJWv48?si=pCeSepBjNE2meH_L
