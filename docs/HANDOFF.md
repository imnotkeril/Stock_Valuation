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
- Done: V5-V6 News tab (Finnhub + Alpha Vantage sentiment).
- Done: V4 scenario analysis on the Risk tab.
- Done: V3 SOTP in every sector model.
- Done: V2 APV tab (valuation/apv.py).
- Done: V1 SEC EDGAR (offline-tested; live needs SEC_EDGAR_USER_AGENT).
- Done: v2 backlog items: saved screener templates, Compare sectors view (13 new tests).
- Done: plan tasks T1-T8 (sidebar nav, live smoke of all pages, pip-audit, localhost binding, Docker, Streamlit deprecations, keys guard test, Financials FCF messages).
- Running: Streamlit dev server on http://localhost:8501 (may be gone; restart with the command in Verify).
- Branch/commit: claude/peaceful-euler-aq2j08, latest local commit, tree clean, nothing pushed.

## Decisions
- Nothing from `chore/audit` code was ported except the keys guard test - its library modules are covered here in tested form; two ideas went to the v2 backlog.
- Owner decided O3: build ALL original README ideas (SPEC section 5) as v2 V1-V8, paid/keyed ones too.
- Owner tasks (key rotation, branches, push) are postponed by the user until the work is finished - do not ask about them again before then.
- Server binds to localhost in `.streamlit/config.toml`; Docker overrides with 0.0.0.0.

## Next
1. V7 in `docs/plan.md`: precedent transactions - curated CSV of public M&A deals by sector (`src/industry/data/precedent_deals.csv`, sourced public figures with date, acquirer, target, EV, EV/Revenue, EV/EBITDA) + optional FMP M&A feed with `FINANCIAL_MODELING_PREP_API_KEY`; median deal multiples applied to the company on the Valuation tab; tests.
2. V8 optional deep-learning forecast (`requirements-ml.txt`, lazy import, hidden when not installed), then V9 user guide + CHANGELOG.
3. Ask the user for `SEC_EDGAR_USER_AGENT`, `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `FINANCIAL_MODELING_PREP_API_KEY` to live-check V1, V5-V7.

## Open questions
- Keys for V5-V7 (Finnhub, Alpha Vantage, FMP): the user will add them to `.env`; values must never be printed.

## Traps
- SEC EDGAR rejects anonymous clients (403); never invent a contact User-Agent - the user must set SEC_EDGAR_USER_AGENT.
- Global `~/.gitignore_global` ignores `*_secret*` and `.env.*`: name files accordingly (that is why the test is `test_no_hardcoded_keys.py`).
- Server listens on 127.0.0.1 only: PowerShell `Invoke-WebRequest http://localhost` may hang on IPv6; use `curl http://127.0.0.1:8501/_stcore/health`.
- Live smoke script lives outside the repo (hits network); offline equivalents are `tests/test_app.py` and `tests/test_pages_interaction.py`.
- Old Alpha Vantage / FMP keys are still in git history - never print them.

## Verify
- `.venv\Scripts\python.exe -m pytest -q` (expect 352 passed) and `.venv\Scripts\python.exe -m ruff check StockAnalysisSystem`
- `.venv\Scripts\python.exe StockAnalysisSystem\run.py` then `curl http://127.0.0.1:8501/_stcore/health` -> ok
