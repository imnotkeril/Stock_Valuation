# Plan — Stock Analysis System

Source: `docs/ARCHITECTURE.md`, `docs/review/PRODUCT_REVIEW.md` section 5 (review fixes, 2026-11-26). Ticked by whoever finishes a task: checkbox here + one line in `docs/STATE.md` + local commit.

Legend: `[ ]` open · `[x]` done (evidence in STATE) · `[!]` blocked (reason on the line).
Fields: `layer` data|backend|frontend|ai|infra|qa|docs · `tier` A|B|C · `owner` agent id · `model` haiku|sonnet|opus · `owns` files this task may edit · `depends` · `uses` · `check` ONE command, exit 0 = done · `done` acceptance (cites review ids R*/U*).
`$PY` = `"E:/Main/Dev/Python/Done/Stock Valuation/.venv/Scripts/python.exe"` (venv lives in the main checkout). `$T` = `$PY -m pytest -o addopts="" -p no:warnings -q`. `$LIVE` = `$PY StockAnalysisSystem/tools/live_valuation_check.py AAPL JPM XOM`.
Every valuation task: offline tests + ruff + `$LIVE` pasted into STATE (baseline 2026-11-26: AAPL fair 193.91 -45.5 % Sell, JPM 310.39 -2.0 % Hold, XOM 95.47 -39.6 % Sell, MSFT -29.1 %, PGR +195 %, CHTR +232 %).
Module `legal-compliance`: its build/qa inserts (privacy pages, consent, deletion flow) do not apply, the app collects no personal data; only the "not investment advice" disclaimer is kept (checked in T20).
Phase 0 (scaffold) is not needed: the app exists and builds (adoption baseline 366 passed, ruff clean).
Router build-row skill `skill:sql-queries` is n/a (no SQL/DuckDB layer; file cache only, ADR-002). T02 is kept as one task because all consumers must switch to the shared module together; it commits per consumer so a cut is resumable.

## Phase 1 — P1: numbers that can be trusted (review section 5, P1)
- [x] T01 Live risk-free rate service (R1, part 1)
  layer: data · tier: B · owner: quant-research · model: sonnet
  owns: StockAnalysisSystem/src/utils/rates.py, StockAnalysisSystem/src/utils/data_loader.py, StockAnalysisSystem/src/config.py, StockAnalysisSystem/tests/test_rates.py
  depends: -
  uses: blueprint:finance-screener, stack:python, stack:pandas, mcp:context7, skill:test-driven-development
  check: `$T StockAnalysisSystem/tests/test_rates.py StockAnalysisSystem/tests/test_data_loader.py && $PY -m ruff check StockAnalysisSystem`
  done: `risk_free_rate()` returns FRED DGS10 (latest valid observation, percent -> fraction) with as-of date and source; cached 1 day; falls back to stale cache, then to a documented constant; offline mode never calls the network; override via env `STOCK_ANALYSIS_RISK_FREE_RATE`.
- [x] T02 One cost-of-capital module + re-baseline (R1 part 2, R2 core)
  layer: backend · tier: B · owner: quant-research · model: opus
  owns: StockAnalysisSystem/src/valuation/capital.py, StockAnalysisSystem/src/valuation/base_valuation.py, StockAnalysisSystem/src/valuation/sector_specific/common.py, StockAnalysisSystem/src/valuation/apv.py, StockAnalysisSystem/src/models/stress_testing.py, StockAnalysisSystem/src/pages/company_analysis/risk.py, StockAnalysisSystem/src/pages/screening/stock_screener.py, StockAnalysisSystem/tests/test_capital.py, StockAnalysisSystem/tests/test_sector_models.py, StockAnalysisSystem/tests/test_apv.py, StockAnalysisSystem/tests/test_stress_testing.py, docs/adr/007-cost-of-capital.md
  depends: T01
  uses: stack:python, skill:test-driven-development
  check: `$T && $PY -m ruff check StockAnalysisSystem && $LIVE`
  done: one `cost_of_capital()` (one beta rule, one Kd band, one ERP, one WACC band, live rf) used by generic and sector models, APV, stress base discount and Sharpe; ERP re-baselined in ADR-007; a test proves generic and sector WACC are equal for the same inputs; results carry rf value + as-of date in `assumptions`; live AAPL/JPM/XOM before/after in STATE.
- [x] T03 DCF/APV tab defaults from the shared WACC + "why methods differ" box (R2 UI)
  layer: frontend · tier: C · owner: quant-research · model: sonnet
  owns: StockAnalysisSystem/src/pages/company_analysis/valuation.py, StockAnalysisSystem/tests/test_pages_interaction.py
  depends: T02
  uses: stack:python
  check: `$T StockAnalysisSystem/tests/test_pages_interaction.py StockAnalysisSystem/tests/test_app.py && $PY -m ruff check StockAnalysisSystem`
  done: DCF slider default = shared WACC (not the static sector table); APV uses the same rf/ERP; caption "rf x.xx % (FRED DGS10, as of <date>)"; expander explains why DCF, APV and Sector Model differ.
- [x] T04 Ticker-scoped widget state (R3 / U1)
  layer: frontend · tier: B · owner: quant-research · model: sonnet
  owns: StockAnalysisSystem/src/main.py, StockAnalysisSystem/src/pages/company_analysis/valuation.py, StockAnalysisSystem/src/pages/company_analysis/sector_valuation.py, StockAnalysisSystem/src/pages/company_analysis/risk.py, StockAnalysisSystem/tests/test_widget_state.py
  depends: T03
  uses: stack:python
  check: `$T StockAnalysisSystem/tests/test_widget_state.py StockAnalysisSystem/tests/test_pages_interaction.py && $PY -m ruff check StockAnalysisSystem`
  done: after AAPL then JPM the DCF/APV/sector sliders show JPM defaults; keys of the previous ticker are cleared on ticker change (test).
- [x] T05 Valuation Filter sanity layer (R4 / U2)
  layer: backend · tier: B · owner: quant-research · model: sonnet
  owns: StockAnalysisSystem/src/pages/screening/valuation_filter.py, StockAnalysisSystem/tests/test_valuation_filter.py
  depends: T02
  uses: stack:pandas
  check: `$T StockAnalysisSystem/tests/test_valuation_filter.py && $PY -m ruff check StockAnalysisSystem`
  done: results beyond +-100 % flagged "check inputs" and not counted as undervalued by default; Method column shows the model actually used; spread across methods shown; default approach = sector model.
- [x] T06 Bank health score, duplicate Altman text, bank stress message, Peer Comparison N/A and 0 % (R5 / U4, U5)
  layer: backend · tier: B · owner: quant-research · model: sonnet
  owns: StockAnalysisSystem/src/pages/company_analysis/risk.py, StockAnalysisSystem/src/pages/comparison/peer_comparison.py, StockAnalysisSystem/tests/test_risk_page.py, StockAnalysisSystem/tests/test_peer_comparison.py
  depends: T04
  uses: stack:python
  check: `$T StockAnalysisSystem/tests/test_risk_page.py StockAnalysisSystem/tests/test_peer_comparison.py && $PY -m ruff check StockAnalysisSystem`
  done: Financials health score excludes the FCF component; Altman paragraph printed once; revenue stress test explains the bank case; Peer Comparison shows the real method and rates instead of N/A / 0.00 %.
- [x] T07 Confidence on the recommendation badge (R6)
  layer: backend · tier: B · owner: quant-research · model: sonnet
  owns: StockAnalysisSystem/src/valuation/base_valuation.py, StockAnalysisSystem/src/pages/company_analysis/valuation.py, StockAnalysisSystem/tests/test_recommendation.py
  depends: T04
  uses: stack:python
  check: `$T StockAnalysisSystem/tests/test_recommendation.py StockAnalysisSystem/tests/test_sector_models.py && $PY -m ruff check StockAnalysisSystem && $LIVE`
  done: recommendation carries `confidence` (from method dispersion, terminal-value share, data completeness), shown on the badge; label softened to "Below/Above model value" when confidence is low.
- [x] T08 Phase 1 gate
  layer: qa · tier: C · owner: qa-tester · model: haiku · owns: - · depends: T01, T02, T03, T04, T05, T06, T07
  uses: none
  check: `$T && $PY -m ruff check StockAnalysisSystem && $LIVE`
  done: full suite green; live AAPL/JPM/XOM + PGR/CHTR pasted; app opened in the browser (Valuation, Risk, Valuation Filter, Peer Comparison).

## Phase 2 — Review step (process/review.md) after P1
- [x] T09 Review with `reviewer` + `security-reviewer`; fix valid findings
  layer: qa · tier: B · owner: reviewer · model: opus
  owns: docs/REVIEW.md (+ files named by valid findings)
  depends: T08
  uses: skill:verification-before-completion, agent:security-reviewer (owner of the security pass: FRED fetch, env override, deps, secrets)
  check: `$T && $PY -m ruff check StockAnalysisSystem`
  done: `docs/REVIEW.md` lists findings with dispositions; every valid P0/P1 fixed and re-tested.
- [x] T10 Phase 2 gate
  layer: qa · tier: C · owner: qa-tester · model: haiku · owns: - · depends: T09
  uses: none
  check: `$T && $PY -m ruff check StockAnalysisSystem && $LIVE`
  done: green output after review fixes pasted.

## Phase 3 — P2: the product explains and matches itself
- [x] T11 Golden-value tests: hand-computed DCF, APV, WACC, residual income (R13)
  layer: qa · tier: B · owner: qa-tester · model: sonnet
  owns: StockAnalysisSystem/tests/test_golden_values.py
  depends: T09
  uses: none
  check: `$T StockAnalysisSystem/tests/test_golden_values.py`
  done: each formula checked against a hand calculation written in the test.
- [x] T12 Relative valuation: live universe medians, method relevance filter, benchmark date (R8)
  layer: backend · tier: B · owner: quant-research · model: sonnet
  owns: StockAnalysisSystem/src/industry/benchmarks.py, StockAnalysisSystem/src/industry/universe.py, StockAnalysisSystem/src/pages/company_analysis/valuation.py, StockAnalysisSystem/tests/test_benchmarks.py
  depends: T09
  uses: stack:pandas
  check: `$T StockAnalysisSystem/tests/test_benchmarks.py && $PY -m ruff check StockAnalysisSystem && $LIVE`
  done: Relative tab uses medians from the cached universe when available (static table as labelled fallback, with date); P/B dropped for asset-light firms, P/S for profitable ones.
- [x] T13 Generic DCF: growth fade, SBC toggle, dead DCF variants exposed or removed (R9)
  layer: backend · tier: B · owner: quant-research · model: sonnet
  owns: StockAnalysisSystem/src/pages/company_analysis/valuation.py, StockAnalysisSystem/src/pages/company_analysis/page_helpers.py, StockAnalysisSystem/src/valuation/dcf_models.py, StockAnalysisSystem/tests/test_dcf_fade.py
  depends: T12
  uses: stack:python
  check: `$T StockAnalysisSystem/tests/test_dcf_fade.py && $PY -m ruff check StockAnalysisSystem && $LIVE`
  done: linear growth fade to terminal; SBC deduction toggle (default on for Technology); multi-stage/Monte Carlo/scenario DCF either reachable or removed with the doc claims.
- [x] T14 UX U6-U9: empty-state hints + main-area run buttons, one-step company pick, static Sector Outlook labelled, clearer labels (R10)
  layer: frontend · tier: C · owner: frontend-dev · model: sonnet
  owns: StockAnalysisSystem/src/main.py, StockAnalysisSystem/src/pages/comparison/peer_comparison.py, StockAnalysisSystem/src/pages/comparison/sector_analysis.py, StockAnalysisSystem/src/pages/company_analysis/overview.py, StockAnalysisSystem/src/pages/company_analysis/valuation.py, StockAnalysisSystem/tests/test_pages_interaction.py
  depends: T13
  uses: skill:frontend-design
  check: `$T StockAnalysisSystem/tests/test_pages_interaction.py StockAnalysisSystem/tests/test_app.py && $PY -m ruff check StockAnalysisSystem`
  done: Peer/Sector pages show a hint and a main-area run button; Enter on a unique match selects it; "Sector notes (static)" label; "P/x" and "Change since" labels replaced; beta/correlation explained.
- [x] T15 Speed: thread-pool fetch for screener/valuation filter, progress for Peer Comparison (R11)
  layer: backend · tier: B · owner: backend-dev · model: sonnet
  owns: StockAnalysisSystem/src/pages/screening/stock_screener.py, StockAnalysisSystem/src/pages/screening/valuation_filter.py, StockAnalysisSystem/tests/test_parallel_fetch.py
  depends: T14
  uses: stack:python
  check: `$T StockAnalysisSystem/tests/test_parallel_fetch.py StockAnalysisSystem/tests/test_valuation_filter.py && $PY -m ruff check StockAnalysisSystem`
  done: 4-8 workers with a shared rate limit; same results as sequential on fixtures; cold screener time measured and pasted.
- [x] T16 Precedents: labelled sample data, clear weight status (R12 / U10)
  layer: backend · tier: B · owner: quant-research · model: sonnet
  owns: StockAnalysisSystem/src/pages/company_analysis/valuation.py, StockAnalysisSystem/src/models/precedent_transactions.py, StockAnalysisSystem/tests/test_precedent_transactions.py
  depends: T15
  uses: none
  check: `$T StockAnalysisSystem/tests/test_precedent_transactions.py && $PY -m ruff check StockAnalysisSystem`
  done: tab explains it is user-owned data, offers an "illustrative" sample; states whether it counts in fair value.
- [x] T17 Phase 3 gate
  layer: qa · tier: C · owner: qa-tester · model: haiku · owns: - · depends: T11, T12, T13, T14, T15, T16
  uses: none
  check: `$T && $PY -m ruff check StockAnalysisSystem && $LIVE`
  done: green output pasted.

## Phase 4 — Hardening (review fixes -> QA -> docs -> release prep)
- [x] T27 Methodology research (owner request, before T18 sign-off): three research agents (DCF/WACC/APV/DDM, sector models, distress + risk metrics) against Damodaran / CFA / McKinsey / Nareit / original papers -> `docs/review/METHODOLOGY_RESEARCH.md`; real discrepancies fixed, each with a test
  layer: qa · tier: B · owner: quant-research · model: opus
  owns: docs/review/METHODOLOGY_RESEARCH.md (+ files named by valid findings), StockAnalysisSystem/tests/test_methodology_fixes.py
  depends: T17
  uses: none
  check: `$T && $PY -m ruff check StockAnalysisSystem && $LIVE`
  done: claim -> source -> code -> verdict -> fix table; fixes committed with tests; open items listed.
- [x] T18 Review fixes for Phase 3 (light `reviewer` pass over the P2 diff, valid findings fixed)
  layer: qa · tier: B · owner: reviewer · model: opus
  owns: docs/REVIEW.md (+ files named by valid findings)
  depends: T17
  uses: skill:verification-before-completion
  check: `$T && $PY -m ruff check StockAnalysisSystem`
  done: REVIEW.md round 2 with dispositions.
## Phase 3b — Owner knowledge-base gaps, small rows (SPEC F22, `docs/sources/KNOWLEDGE_BASE_GAP.md`)
- [x] T22 Liquidation value method: orderly + forced recovery rates per asset class, editable, floor only (AC22.1)
  layer: backend · tier: B · owner: quant-research · model: sonnet
  owns: StockAnalysisSystem/src/valuation/liquidation.py, StockAnalysisSystem/src/pages/company_analysis/valuation.py, StockAnalysisSystem/tests/test_liquidation.py
  depends: T18
  uses: stack:python
  check: `$T StockAnalysisSystem/tests/test_liquidation.py && $PY -m ruff check StockAnalysisSystem && $LIVE`
  done: hand-computed golden test; section on the Valuation tab; not in the weighted fair value (stated).
- [x] T23 Risk-adjusted ratios on the Risk tab: Sortino, Treynor, Calmar, Information ratio, Jensen's alpha (AC22.2)
  layer: backend · tier: B · owner: quant-research · model: sonnet
  owns: StockAnalysisSystem/src/models/performance_ratios.py, StockAnalysisSystem/src/pages/company_analysis/risk.py, StockAnalysisSystem/tests/test_performance_ratios.py
  depends: T22
  uses: stack:python
  check: `$T StockAnalysisSystem/tests/test_performance_ratios.py StockAnalysisSystem/tests/test_risk_page.py && $PY -m ruff check StockAnalysisSystem`
  done: golden values on a hand-made return series; N/A rules tested; definitions in captions.
- [x] T24 Insider transactions via Finnhub (key optional, missing key = explanation) (AC22.3)
  layer: data · tier: B · owner: backend-dev · model: sonnet
  owns: StockAnalysisSystem/src/utils/insider.py, StockAnalysisSystem/src/pages/company_analysis/overview.py, StockAnalysisSystem/tests/test_insider.py
  depends: T18
  uses: stack:python
  check: `$T StockAnalysisSystem/tests/test_insider.py && $PY -m ruff check StockAnalysisSystem`
  done: recorded-shape fixture parsed; no key -> info text, no request; HTTP error -> empty table, no exception.
- [x] T25 A1-A4 / P1-P4 balance liquidity grouping on Financials > Ratios (AC22.4)
  layer: backend · tier: B · owner: quant-research · model: sonnet
  owns: StockAnalysisSystem/src/models/balance_liquidity.py, StockAnalysisSystem/src/pages/company_analysis/financials.py, StockAnalysisSystem/tests/test_balance_liquidity.py
  depends: T22
  uses: stack:python
  check: `$T StockAnalysisSystem/tests/test_balance_liquidity.py && $PY -m ruff check StockAnalysisSystem`
  done: hand-computed grouping test; four conditions + verdict shown; missing rows handled.
- [x] T26 Phase 3b gate
  layer: qa · tier: C · owner: qa-tester · model: haiku · owns: - · depends: T22, T23, T24, T25
  uses: none
  check: `$T && $PY -m ruff check StockAnalysisSystem && $LIVE`
  done: green output pasted.

## Phase 4 (cont.) — QA, docs, release
- [x] T19 QA: browser pass of every page for AAPL/JPM/XOM, `docs/QA.md` updated
  layer: qa · tier: C · owner: qa-tester · model: sonnet
  owns: docs/QA.md
  depends: T26
  uses: none
  check: `$T && $LIVE`
  done: QA.md lists pages checked, timings, issues; no tracebacks.
- [x] T20 Docs: METHODOLOGY.md (how numbers are made), README + USER_GUIDE with screenshots, SPEC/CHANGELOG refresh, disclaimer (R7)
  layer: docs · tier: C · owner: tech-writer · model: sonnet
  owns: README.md, docs/METHODOLOGY.md, docs/USER_GUIDE.md, docs/SPEC.md, CHANGELOG.md, docs/screenshots/**, StockAnalysisSystem/tools/docs_check.py (created here: README links screenshots + METHODOLOGY, disclaimer present)
  depends: T16
  uses: module:legal-compliance#qa
  check: `$PY StockAnalysisSystem/tools/docs_check.py`
  done: methodology covers WACC build, weights, thresholds, confidence, each sector model, benchmark source, known biases; screenshots taken after the fixes are linked from README and USER_GUIDE; feature table current; HTML export documented; disclaimer present.
- [x] T21 Release prep (no push): final gate, CHANGELOG entry, PR proposal to the owner
  layer: infra · tier: C · owner: devops · model: haiku · owns: CHANGELOG.md · depends: T20
  uses: none
  check: `$T && $PY -m ruff check StockAnalysisSystem && $LIVE`
  done: green output pasted; PR text drafted; push/PR only on the owner's yes.

## Backlog (not planned now; owner decides)
- Methodology leftovers (METHODOLOGY_RESEARCH.md): RONIC terminal value, marginal tax rate for the debt shield, minority interest / preferred in the DCF-tab bridge, VZ DCF tab (WACC floor + telecom growth).
- P3: R14 wider ruff + narrow broad excepts + split giant modules; R15 watchlist / "what changed"; R16 sensitivity tornado in PDF; R17 second price source.
- MacroLens ideas (review 5.1): yield-curve 2s10s/10s3m card, recession probability (probit), regime quadrant + sector tilt table (informational only, never in fair value).
- `repo-polish` after T21, only on the owner's yes.

## Traceability
| SPEC AC / review item | Tasks |
|---|---|
| AC20.1 R1 | T01, T02, T03 |
| AC20.2 R2 | T02, T03, T11 |
| AC20.3 R3 / U1 | T04 |
| AC20.4 R4 / U2 | T05 |
| AC20.5 R5 / U4, U5 | T06 |
| AC20.6 R6 | T07 |
| AC21.1 R7 | T20 |
| AC21.2 R8 | T12 |
| AC21.3 R9 | T13 |
| AC21.4 R10 / U6-U9 | T14 |
| AC21.5 R11 | T15 |
| AC21.6 R12 / U10 | T16 |
| AC21.7 R13 | T11 |
| AC22.1 | T22 |
| AC22.2 | T23 |
| AC22.3 | T24 |
| AC22.4 | T25 |

# History (pre-framework plan, kept as written)

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
- [x] O1 **Rotate Alpha Vantage and FMP keys** (done by owner 2026-11-26): they were hard-coded in old `config.py` and
      are still in git history on GitHub (`master` and both branches). Optionally rewrite
      history afterwards (owner decision).
- [x] O2 Branches: make `claude/peaceful-euler-aq2j08` the main line (PR into `master`);
      archive `chore/audit` without merging (its work is superseded, see summary).
      Push/PR only on your yes. 2026-11-24: PR #1 open and up to date; merge is yours.
- [x] O3 Scope of SPEC §5: owner decided 2026-11-23 - build ALL original ideas (v2 below),
      including the ones that need paid or keyed data sources.
- [x] O4 Old folder `E:\Main\Dev\Python\In Development\Stock_Valuation`: delete or archive
      after O2 (its docs are copied to `docs/sources/in-development/`). Done 2026-11-24: keys moved to `.env`, folder sent to Recycle Bin.

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
- [x] V3 SOTP for the remaining sectors (Technology, Financials, Manufacturing, Real Estate,
      Healthcare, Communication) via segment data where available · quant
- [x] V4 Stress testing / market risk simulation: rate, margin and multiple shocks, historical
      scenarios (2008, 2020, 2022) on price and fair value · quant
- [x] V5 Finnhub provider (free key `FINNHUB_API_KEY`): peers, metrics, company news · dev
- [x] V6 News sentiment: Finnhub news + Alpha Vantage NEWS_SENTIMENT (keys), score per
      company over time; new tab on Company Analysis · ai/dev
- [x] V7 Precedent transactions: M&A deals by sector (FMP key, `FINANCIAL_MODELING_PREP_API_KEY`;
      fallback: curated CSV of public deals)
      Done as: user-maintained deal table (editor + CSV import, no bundled figures to avoid
      unverified numbers), median/IQR deal multiples -> implied value; FMP feed of recent M&A → EV/Revenue, EV/EBITDA deal multiples · quant
- [x] V8 Deep-learning forecasts as an optional extra (`requirements-ml.txt`: torch-based model,
      lazy import; page hides it when not installed) · quant
      Done as: "Neural Network (MLP)" price model (scikit-learn MLPRegressor, two hidden layers,
      no new dependency); torch/TensorFlow not added - too heavy for the gain
- [x] V9 User guide and CHANGELOG (after V1-V8)

## Known debt (recorded at adoption 2026-11-26, not fixed yet)
- [ ] D1 Two cost-of-capital implementations and hard-coded 3 % risk-free (review R1/R2) -> planned below
- [ ] D2 Narrow ruff rule set; 125 broad `except Exception` (review R14)
- [ ] D3 pip without lock file; finance family default is uv + `uv.lock` (ADR-005)
- [ ] D4 No golden textbook tests for DCF/WACC/APV (review R13)
- [ ] D5 Git history still holds old (rotated) keys from `config.py` (owner decision on history rewrite)
