# STATE

Branch `claude/peaceful-euler-aq2j08` (this folder). Spec: `docs/SPEC.md`. Plan: `docs/plan.md`. QA: `docs/QA.md`.

- 2026-11-20 sector models, valuation/risk tabs, forecasting engine, 280 offline tests (cloud session).
- 2026-11-23 cloned locally, venv + deps, app runs at http://localhost:8501.
- 2026-11-23 merged docs from the In Development copy (`chore/audit`) into `docs/sources/`; unified SPEC, plan, summary incl. module-by-module comparison.
- 2026-11-23 T1-T8 done: sidebar nav, live smoke of all pages (10 sectors, all buttons), pip-audit clean, localhost binding, Docker build+health ok, `use_container_width` removed, secrets guard test, Financials FCF messages. 282 tests pass, ruff clean.
- 2026-11-23 v2: saved custom screener templates (save/reapply/delete, JSON in src/data/user) and Compare sectors view on Sector Analysis (ETF performance, median fundamentals, leaders). 13 new tests; checked live in browser.

## Next
- Owner decided O3: build all original ideas. Start v2 V1 (SEC EDGAR) in docs/plan.md, then V2-V9.

## Needed from owner
- O1 rotate Alpha Vantage and FMP keys (still in git history). O2 branch decision / push. O3 scope of SPEC §5. O4 old In Development folder.
