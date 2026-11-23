# Handoff
status: active
date: 2026-11-23
from-chat-tokens: ~190k

## Goal
Bring Stock Analysis System (Streamlit, sector-specific stock valuation) to a finished, verified
state in this folder, merging everything useful from the old copy in
`E:\Main\Dev\Python\In Development\Stock_Valuation` (branch `chore/audit`).

## State
- Done: unified `docs/SPEC.md`, `docs/plan.md`, `docs/QA.md`, `docs/STATE.md`; In Development docs copied to `docs/sources/in-development/` with a module-by-module comparison in `docs/sources/IN_DEVELOPMENT_SUMMARY.md`.
- Done: plan tasks T1-T8 (sidebar nav, live smoke of all pages, pip-audit, localhost binding, Docker, Streamlit deprecations, keys guard test, Financials FCF messages).
- Running: Streamlit dev server on http://localhost:8501 from the previous chat (may be gone; restart with the command in Verify).
- Branch/commit: claude/peaceful-euler-aq2j08 @ 856131d (+ this note), tree clean, nothing pushed.

## Decisions
- Nothing from `chore/audit` code was ported except the keys guard test - its library modules are covered here in tested form; two ideas went to the v2 backlog.
- Owner tasks (key rotation, branches, push) are postponed by the user until the work is finished - do not ask about them again before then.
- Server binds to localhost in `.streamlit/config.toml`; Docker overrides with 0.0.0.0.

## Next
1. Read `docs/plan.md`: all "Now" tasks are done; start the v2 backlog item "Saved custom screener templates" (Stock Screener page, `src/pages/screening/stock_screener.py`) with a test.
2. Then "Multi-sector comparison view" on Sector Analysis, with a test.
3. When the user says the work is finished, go through owner items O1-O4 with them (rotation, PR into master, scope of SPEC section 5, old folder).

## Open questions
- Scope of SPEC section 5 (APV, sentiment, EDGAR, stress testing): default is "not in MVP" until the user decides.

## Traps
- Global `~/.gitignore_global` ignores `*_secret*` and `.env.*`: name files accordingly (that is why the test is `test_no_hardcoded_keys.py`).
- Server listens on 127.0.0.1 only: PowerShell `Invoke-WebRequest http://localhost` may hang on IPv6; use `curl http://127.0.0.1:8501/_stcore/health`.
- Live smoke script lives outside the repo (hits network); offline equivalents are `tests/test_app.py` and `tests/test_pages_interaction.py`.
- Old Alpha Vantage / FMP keys are still in git history - never print them.

## Verify
- `.venv\Scripts\python.exe -m pytest -q` (expect 282 passed) and `.venv\Scripts\python.exe -m ruff check StockAnalysisSystem`
- `.venv\Scripts\python.exe StockAnalysisSystem\run.py` then `curl http://127.0.0.1:8501/_stcore/health` -> ok
