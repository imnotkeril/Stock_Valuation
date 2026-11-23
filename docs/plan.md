# Plan

Based on `docs/SPEC.md` (2026-11-23). Earlier work: `docs/sources/IN_DEVELOPMENT_SUMMARY.md`.
Each task: owner · check. Tick when the check passes.

## Done (history)
- [x] Readiness audit, `.gitignore`, `.env.example`, keys out of code (In Development, 2026-10-24)
- [x] App shell, unified valuation contract, offline test infrastructure (2026-11-04)
- [x] All 8 sector models on the shared mixin; Valuation and Risk tabs; forecasting engine;
      page fixes; 280 offline tests (2026-11-20)
- [x] App shell polish, matplotlib in requirements (2026-11-20)
- [x] Local run verified: venv, deps, `run.py`, health `ok`, Home renders (2026-11-23)
- [x] Unified SPEC, plan and In Development summary (2026-11-23)

## Now
- [x] T1 Duplicate sidebar navigation: Streamlit auto-lists `src/pages/` above the app menu.
      Fix: `client.showSidebarNavigation = false` in `.streamlit/config.toml`.
      · dev · browser: only the app menu is shown
- [x] T2 Live smoke test: one ticker per sector (AAPL, JPM, XOM, NEE, AMZN, PG, CAT, PLD, JNJ,
      GOOGL), every Company Analysis tab, Peer/Sector/Screener/Filter/Forecasting/Reports;
      fix what breaks. · qa · notes in `docs/QA.md`, no tracebacks
- [x] T3 Dependency audit: `pip-audit -r StockAnalysisSystem/requirements.txt`; raise
      minimums where CVEs are found. · dev · pip-audit clean
- [x] T4 Bind to localhost by default (logs show Network/External URLs; `enableCORS=false`).
      Keep Docker on `0.0.0.0` via its own flag. · dev · local run shows Local URL only;
      Docker still reachable
- [x] T5 Verify Docker image: `docker build` + `docker run`, health `ok`. · devops · output pasted
- [x] T6 Replace deprecated `use_container_width` (33 places) · dev · no deprecation warnings
- [x] T7 Port hard-coded-keys guard test from `chore/audit` · dev · `tests/test_no_hardcoded_keys.py` passes
- [x] T8 Financials: hide or explain FCF DCF on the generic Valuation tab for banks/insurers
      (QA observation) · dev · JPM Valuation tab shows no negative-FCF warning

## Owner (needs your decision or action)
- [ ] O1 **Rotate Alpha Vantage and FMP keys**: they were hard-coded in old `config.py` and
      are still in git history on GitHub (`master` and both branches). Optionally rewrite
      history afterwards (owner decision).
- [ ] O2 Branches: make `claude/peaceful-euler-aq2j08` the main line (PR into `master`);
      archive `chore/audit` without merging (its work is superseded, see summary).
      Push/PR only on your yes.
- [x] O3 Scope of SPEC §5: owner decided 2026-11-23 - build ALL original ideas (v2 below),
      including the ones that need paid or keyed data sources.
- [ ] O4 Old folder `E:\Main\Dev\Python\In Development\Stock_Valuation`: delete or archive
      after O2 (its docs are copied to `docs/sources/in-development/`).

## v2 - original README ideas (owner decision O3: build all)
Order: free sources first. Every provider is optional: without its key the feature shows
"add <KEY> to .env" instead of failing. Each task: tests offline with recorded fixtures.
- [x] Saved custom screener templates (idea from In Development `screener_engine.py`)
- [x] Multi-sector comparison view (idea from In Development `industry_benchmarking.py`)
- [x] V1 SEC EDGAR provider (free, needs only a User-Agent with contact): company facts
      (XBRL) as a fallback/extra source for statements; filings list on Overview · dev
      Done: `utils/sec_edgar.py`, Overview section, 9 offline tests. Live run pending the user's
      `SEC_EDGAR_USER_AGENT` (SEC rejects anonymous clients with 403).
- [x] V2 APV (adjusted present value) method in the generic valuation (unlevered DCF +
      PV of tax shield) · quant · hand-calculated test
- [ ] V3 SOTP for the remaining sectors (Technology, Financials, Manufacturing, Real Estate,
      Healthcare, Communication) via segment data where available · quant
- [ ] V4 Stress testing / market risk simulation: rate, margin and multiple shocks, historical
      scenarios (2008, 2020, 2022) on price and fair value · quant
- [ ] V5 Finnhub provider (free key `FINNHUB_API_KEY`): peers, metrics, company news · dev
- [ ] V6 News sentiment: Finnhub news + Alpha Vantage NEWS_SENTIMENT (keys), score per
      company over time; new tab on Company Analysis · ai/dev
- [ ] V7 Precedent transactions: M&A deals by sector (FMP key, `FINANCIAL_MODELING_PREP_API_KEY`;
      fallback: curated CSV of public deals) → EV/Revenue, EV/EBITDA deal multiples · quant
- [ ] V8 Deep-learning forecasts as an optional extra (`requirements-ml.txt`: torch-based model,
      lazy import; page hides it when not installed) · quant
- [ ] V9 User guide and CHANGELOG (after V1-V8)
