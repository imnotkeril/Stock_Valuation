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

## Next
- Owner decided O3: build all original ideas. Next: V8 (neural-network forecast) in docs/plan.md, then V9 docs.

## Needed from owner
- O1 rotate Alpha Vantage and FMP keys (still in git history). O2 branch decision / push. O3 scope of SPEC §5. O4 old In Development folder.
