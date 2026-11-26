# STATE — Stock Analysis System

Append-only progress log. One line per completed step: `- [x] <phase>/<task id> <what> — <check evidence>`. Next work = first item below "Next".
On a rate-limit cut or new session: read this file, then `docs/plan.md`, continue with "Next".
Router: `finance/screener` (Streamlit variant) · Modules: legal-compliance · Branch: `claude/review-fixes-195b14`.

## Log
History before adoption (free format, kept as written):
- [x] history 2026-11-20 sector models, valuation/risk tabs, forecasting engine, 280 offline tests (cloud session).
- [x] history 2026-11-23 cloned locally, venv + deps, app runs at http://localhost:8501.
- [x] history 2026-11-23 merged docs from the In Development copy (`chore/audit`) into `docs/sources/`; unified SPEC, plan, summary.
- [x] history 2026-11-23 T1-T8: sidebar nav, live smoke of all pages, pip-audit clean, localhost binding, Docker build+health, `use_container_width` removed, secrets guard test, Financials FCF messages — 282 tests, ruff clean.
- [x] history 2026-11-23 v2: saved screener templates, Compare sectors view — 13 new tests, browser check.
- [x] history 2026-11-23 V1 SEC EDGAR provider + Overview section — 9 offline tests, 304 pass.
- [x] history 2026-11-24 V2 APV tab — 8 tests incl. hand calculation; live AAPL/XOM/JPM/PLD ok.
- [x] history 2026-11-24 V3 SOTP in all sector models — 22 tests, 334 pass, live 6 sectors ok.
- [x] history 2026-11-24 V4 scenario analysis (Risk tab) — 6 tests, 340 pass, live AAPL/JPM/XOM/NEE ok.
- [x] history 2026-11-24 V5+V6 Finnhub/Alpha Vantage news sentiment + News tab — 12 tests, 352 pass.
- [x] history 2026-11-24 V7 precedent transactions (deal editor, CSV import, FMP feed) — 5 tests.
- [x] history 2026-11-24 V8 MLP price model — 5 tests.
- [x] history 2026-11-24 V9 USER_GUIDE + CHANGELOG — 363 tests pass.
- [x] history 2026-11-24 branch pushed (owner yes); PR #1 opened; keys moved to `.env`, old folder recycled (O4); FMP free-plan limit fix — 364 pass.
- [x] history 2026-11-26 FRED macro via public CSV, SPEC refreshed — live CPI/UNRATE/DGS10 ok.
- [x] history 2026-11-26 owner rotated keys, SEC User-Agent set — live SEC, Finnhub, FMP, Alpha Vantage ok.
- [x] history 2026-11-26 PR #1 merged into master (a92c061) after green CI 3.11-3.13 (owner approved).
- [x] history 2026-11-26 product review `docs/review/PRODUCT_REVIEW.md` + 19 screenshots (merged from `claude/product-review-aea4ae`).

Framework (from adoption on):
- [x] adopted (finance/screener, owner confirmed) — baseline: ruff "All checks passed!", pytest "366 passed in 124.98s"; apply-router exit 0 (5 skills linked, 0 missing); CLAUDE.md, AGENTS.md, ARCHITECTURE.md, adr/001-006 created

- [x] brief/plan - 4 items read (process/plan.md, finance/_family.md, finance/screener.md, known-traps.md; stack/python.md); legal-compliance build/qa inserts judged not applicable (no personal data), Next.js stack rows n/a (ADR-001)
- [x] plan/draft - 21 tasks T01-T21 in 4 phases (P1 T01-T08, review T09-T10, P2 T11-T17, hardening T18-T21), no parallel groups (shared files); SPEC F20/F21 with AC20.1-AC21.7 added for traceability
- [x] plan/lint - `doc-check --step plan`: 9 pass, 0 fail, 3 judge (exit 0)
- [x] tools/live_valuation_check.py baseline (rf 3 %): AAPL fair 193.91 -45.5 % Sell (wacc 8.91 %), JPM 310.39 -2.0 % Hold, XOM 95.47 -39.6 % Sell (wacc 6.00 % floor), MSFT -29.1 %, PGR +195 % (excess_return 1071), CHTR +232 % (wacc 6.00 % floor)
- [x] note: auditor model fable failed (no usage credits, HTTP 429) -> sonnet per rules/audit.md model table

- [x] audit/plan - FIX, 4 findings (4 valid: A-01 docs_check owner, A-02 sql-queries n/a, A-04 security-reviewer fixed; A-03 kept, noted) [shadow]
- [x] brief/T01 - 5 items read (blueprints/finance-screener, stack/python, stack/pandas, TDD skill, known-traps)
- [x] build/T01 utils/rates.py risk_free_rate() (env override -> FRED DGS10 cached 1 day + stale -> fallback 4.5 %), DataLoader.get_fred_series — 26 passed (test_rates + test_data_loader), ruff clean; live "4.91 % (FRED DGS10, as of 2026-11-24)"

- [x] brief/T02 - 3 items (stack/python, TDD skill, known-traps; read in T01, rechecked); owns + config.py (legacy RISK_FREE_RATE removed as T01 promised)
- [x] build/T02 valuation/capital.py one cost_of_capital (Blume beta, Kd rf+0.5..8 %, WACC 6-20 %, ERP 4.4 % Damodaran Jul-2026, live rf) used by generic, sector, APV, stress, Sharpe; ADR-007 — 402 passed, ruff clean; live rf 4.91 %: AAPL 184.47 -48.1 % Sell (wacc 9.49 %), JPM 278.54 -12.1 % Reduce, XOM 86.30 -45.4 % Sell (wacc 6.59 %), MSFT -31.4 %, PGR +54.5 %, CHTR +232 % (wacc 6 % floor, T05)

- [x] brief/T03 - 2 items (stack/python, known-traps); owns valuation.py + test_pages_interaction.py
- [x] build/T03 DCF slider default = shared WACC, APV inputs (tax, Kd, beta, rf) from capital.py, rf caption, "Why do DCF, APV and the Sector Model differ?" expander on DCF + APV — 35 passed (pages_interaction + app), ruff clean; live AAPL DCF default 9.49 % (was table 12 %), XOM 6.59 % (was 9 %), caption "rf 4.91 % (FRED DGS10, as of 2026-11-24)"

- [x] brief/T04 - 2 items (stack/python, known-traps); sector widgets already ticker-scoped, DCF/APV/Risk were not
- [x] build/T04 main.reset_company_widgets() drops val_*/risk_* state when the analysed ticker changes — 22 passed (widget_state + pages_interaction), red check: 2 fail without the fix; ruff clean

- [x] brief/T05 - 2 items (stack/pandas, known-traps)
- [x] build/T05 Valuation Filter: default approach = sector model (DCF alias), Method = model + heaviest method, method range/spread -> confidence, |upside| > 100 % flagged "check inputs" and hidden unless ticked — 15 passed (valuation_filter) + 7 page-button tests, ruff clean; live: CHTR +232 % flagged Low, PGR "Financial: Excess Return" $235-$524 Low, AAPL "Technology: DCF (SBC-adjusted FCF)" High

## Next
- [ ] build/T06 - run `step-brief --task T06` first; bank Risk tab + Peer Comparison N/A / 0.00 %

## Needed from owner
- none
