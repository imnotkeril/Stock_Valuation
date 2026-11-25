# STATE

Branch `claude/peaceful-euler-aq2j08` (this folder). Spec: `docs/SPEC.md`. Plan: `docs/plan.md`. QA: `docs/QA.md`.

- 2026-11-20 sector models, valuation/risk tabs, forecasting engine, 280 offline tests (cloud session).
- 2026-11-23 cloned locally, venv + deps, app runs at http://localhost:8501.
- 2026-11-23 merged docs from the In Development copy (`chore/audit`) into `docs/sources/`; unified SPEC, plan, summary incl. module-by-module comparison.
- 2026-11-23 T1-T8 done: sidebar nav, live smoke of all pages (10 sectors, all buttons), pip-audit clean, localhost binding, Docker build+health ok, `use_container_width` removed, secrets guard test, Financials FCF messages. 282 tests pass, ruff clean.
- 2026-11-23 v2: saved custom screener templates (save/reapply/delete, JSON in src/data/user) and Compare sectors view on Sector Analysis (ETF performance, median fundamentals, leaders). 13 new tests; checked live in browser.

- 2026-11-23 V1 SEC EDGAR: utils/sec_edgar.py (CIK, filings, 10-year XBRL history, day cache), Overview section on demand, 9 offline tests; 304 tests pass. Live check pending SEC_EDGAR_USER_AGENT from the user.

- 2026-11-24 V2 APV: valuation/apv.py + APV tab on Valuation (n/a for Financials), 8 tests incl. hand calculation; live AAPL/XOM/JPM/PLD ok.

- 2026-11-24 V3 SOTP in all sector models: common.parametric_sum_of_parts / blended_sotp, off by default (segment share 0), 22 tests; 334 pass, live 6 sectors ok.

- 2026-11-24 V4 scenario analysis: models/stress_testing.py + Risk tab section, 6 tests; 340 pass, live AAPL/JPM/XOM/NEE ok.

- 2026-11-24 V5+V6: utils/news_sentiment.py (Finnhub news/metrics/peers, Alpha Vantage NEWS_SENTIMENT, lexicon scorer, 6h cache, keys never in cache names) + News tab; 12 tests; 352 pass. Live needs keys.

- 2026-11-24 V7 precedent transactions: models/precedent_transactions.py + Valuation > Precedents (deal editor, CSV import, deal multiples, FMP feed); 5 tests.

- 2026-11-24 V8 neural network (MLP) price model in forecast_engine + Forecasting page; 5 tests.

- 2026-11-24 V9 docs/USER_GUIDE.md + CHANGELOG.md, linked from README. All v2 tasks V1-V9 done; 363 tests pass.

- 2026-11-24 branch claude/peaceful-euler-aq2j08 pushed to origin (user approved); no PR, master untouched.

- 2026-11-24 owner skipped live checks of V1/V5-V7 (no keys in .env); moving to finalize (O1, O2, O4). 363 tests pass, ruff clean.

- 2026-11-24 O2: PR https://github.com/imnotkeril/Stock_Valuation/pull/1 (already open) updated with latest commits; merge is the owner's. O4 held: old folder has untracked StockAnalysisSystem/.env.txt with API keys + 13 cache files, owner to decide.

- 2026-11-24 keys moved from old .env.txt to .env; old In Development folder sent to Recycle Bin (O4). Live: Finnhub ok, FMP fixed (free-plan limit 5), AV news is premium, SEC needs owner's User-Agent. 364 pass.

- 2026-11-26 FRED macro via public CSV (pandas-datareader dropped), 2 tests, live CPI/UNRATE/DGS10 ok; SPEC brought up to date (APV, SEC, FMP, env list, §5 statuses).

- 2026-11-26 owner rotated keys and set SEC_EDGAR_USER_AGENT; live: SEC (filings + XBRL history, UI ok), Finnhub, FMP, Alpha Vantage all ok.

- 2026-11-26 PR #1 merged into master (a92c061) after green CI 3.11-3.13 (owner approved). Product review running in session product-review.

## Next
- Done. Proposed: live risk-free rate from FRED DGS10; MacroLens ideas (see HANDOFF).

## Needed from owner
- none (O1-O4 closed).
