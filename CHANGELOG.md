# Changelog
Add one dated line every time you change something. Evaluators look for progress since Milestone 1.

## 2026-11-20 (Milestone 2 build)
- Added 3-prompt pipeline (P1 themes, P2 classifier with self-consistency, P3 improvements)
- Added Streamlit UI, validators (schema, theme whitelist, verbatim-evidence check)
- Added labeled evaluation set (24 reviews), prompt-version eval script, 14-case test runner, unit tests
- Added diagrams (prompt workflow, decomposition, I/O flow, architecture, self-consistency loop)
- 2026-11-21: Test 14 (unrelated text) crashed theme discovery; added default-theme fallback so P2 flags such input as invalid
