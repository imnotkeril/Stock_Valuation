# ARCHITECTURE — Stock Analysis System

Status: reverse-engineered from the code on 2026-11-26 (adopt-project), not designed up front. Router `finance/screener`;
deviations from the router defaults are recorded as ADRs in `docs/adr/`.

## 1. Stack and versions
| Layer | Choice | Version | Source |
|---|---|---|---|
| Language | Python | 3.11 local (`.venv`), CI 3.11-3.13, Docker 3.12-slim | ADR-005 |
| Packaging | pip + `StockAnalysisSystem/requirements.txt` (minimums), `requirements-dev.txt` | - | ADR-005 (router default is uv) |
| UI | Streamlit | >= 1.50 (tested 1.66) | ADR-001 (router allows Streamlit for single user) |
| Data | pandas >= 2.2, numpy >= 1.26, scipy, statsmodels, scikit-learn | see requirements | router |
| Charts | plotly (UI), matplotlib (PDF) | - | - |
| Market data | yfinance (primary), requests for REST/CSV providers | yfinance >= 0.2.54 | ADR-002 |
| Export | fpdf2, openpyxl, xlsxwriter | - | - |
| Tests / lint | pytest, ruff (narrow rule set) | - | ADR-004 |

## 2. System overview
Browser -> `src/main.py` (Streamlit shell: sidebar company search + page menu, `st.session_state` holds the selected
ticker and computed results) -> page modules in `src/pages/` -> domain code:
`utils/data_loader.py` (all provider I/O + file cache) -> `utils/financial_data.py` (normalised statements dict) ->
`models/` (ratios, bankruptcy, stress tests, forecasting) and `valuation/` (generic `BaseValuation`, `apv.py`,
`dcf_models.py`, sector models in `valuation/sector_specific/` built on `common.py`). Every valuation returns one
schema (`build_valuation_result`): methods, weights, fair value, upside, recommendation, assumptions.
No server, no database: single user on `127.0.0.1:8501`; Docker binds `0.0.0.0` inside the container.

## 3. Folder layout
```
StockAnalysisSystem/
  run.py                 launcher (streamlit run src/main.py, localhost)
  src/config.py          constants: sectors, aliases, DCF defaults, RISK_FREE_RATE, env/offline flags
  src/main.py            app shell and navigation
  src/pages/             home, company_analysis/* (overview, financials, valuation, sector_valuation, risk, news),
                         comparison/* (peer_comparison, sector_analysis, sector_compare), screening/* (stock_screener,
                         valuation_filter), forecasting, reports
  src/utils/             data_loader, financial_data, sec_edgar, news_sentiment, export, visualization
  src/models/            ratio_analysis, bankruptcy_models, financial_statements, stress_testing, precedent_transactions,
                         forecasting/ (time_series, ml_models, forecast_engine)
  src/valuation/         base_valuation (contract + generic methods + WACC), apv, dcf_models, sector_factor,
                         sector_specific/ (common + 8 sector models)
  src/industry/          benchmarks (static sector medians), sector_mapping, sector_ratios, universe (~132 tickers)
  src/data/              cache/, sector_data/, user/ (git-ignored runtime data)
  tests/                 offline pytest suite, fixtures/ (recorded AAPL, MSFT), synthetic.py, offline.py
docs/                    SPEC, ARCHITECTURE, plan, STATE, QA, USER_GUIDE, HANDOFF, adr/, review/, sources/
```

## 4. Data model
No database. File cache under `src/data/cache` (pickle/JSON per provider call, keyed without API keys), lifetimes:
prices `CACHE_EXPIRY_DAYS`, info/statements 7 days, search 3 days, SEC/FRED 1 day. Order: fresh cache -> fetch ->
stale cache. User data (screener templates, precedent deals) as JSON in `src/data/user/`.

## 5. Interfaces
Streamlit pages only; no public API. Provider endpoints: Yahoo (yfinance), Alpha Vantage REST, Finnhub REST,
FMP REST, SEC EDGAR (`data.sec.gov`, declared User-Agent), FRED `fredgraph.csv?id=<series>`.

## 6. Modules wiring
| Module | Depth | `requires` | Touches | Env vars |
|---|---|---|---|---|
| legal-compliance | minimal | - | disclaimer in README, UI footer/home, reports; no personal data collected | - |
Router defaults `background-jobs` and `notifications` are not used: scans run on demand, no alerts (ADR-001).

## 7. Cross-cutting
- Config and secrets: `.env` (git-ignored) loaded by python-dotenv; names in `.env.example`; guard test
  `tests/test_no_hardcoded_keys.py`. Keys never appear in cache file names.
- Errors: providers return empty frames/dicts instead of raising; pages show "not available" text. Many broad
  `except Exception` blocks (review R14, backlog).
- Offline mode: `STOCK_ANALYSIS_OFFLINE=1` disables all network calls (CI and tests).
- Cost of capital: currently two implementations (`base_valuation._wacc_components`, `common.cost_of_capital`) and a
  constant `RISK_FREE_RATE = 3 %`; to be unified (plan R1/R2, ADR-007).
- Security: localhost binding, no auth (single user), no user-supplied URLs; tickers go to provider APIs only.

## 8. Environments and ports
Local only: `127.0.0.1:8501`. Docker: `-p 8501:8501`, health check `/_stcore/health`.

## 9. Testing strategy
Tier A: none (no money movement). Tier B: valuation math, WACC/APV, ratios, distress models, data adapters with recorded
fixtures and synthetic statements. Tier C: Streamlit `AppTest` page smoke and interaction tests. All offline;
live provider checks are manual scripts recorded in STATE.

## 10. Risks and applied traps
| Trap / risk | Where | Decision |
|---|---|---|
| Free vendor terms (Yahoo personal/research use) | data_loader | personal research tool only; stated in README |
| Survivorship / static universe | `industry/universe.py` | today's ~132 large caps; screener is a snapshot tool, no backtests |
| Inconsistent discount rates between tabs | valuation | fix in R1/R2 |
| Outlier fair values ranked as opportunities | valuation_filter | fix in R4 |
| "Recommendation" read as advice | base_valuation badge | disclaimer + confidence (R6) |

## 11. ADRs
- ADR-001 Streamlit single-user app instead of FastAPI + Next.js -> `docs/adr/001-streamlit-single-user.md`
- ADR-002 Yahoo primary, keyed providers optional, file cache with stale fallback -> `docs/adr/002-data-providers-and-cache.md`
- ADR-003 Unified valuation contract and model-declared parameters -> `docs/adr/003-unified-valuation-contract.md`
- ADR-004 Offline test suite with recorded fixtures -> `docs/adr/004-offline-tests.md`
- ADR-005 pip + requirements.txt instead of uv -> `docs/adr/005-pip-requirements.md`
- ADR-006 FRED via public fredgraph CSV -> `docs/adr/006-fred-public-csv.md`
- ADR-007 One cost-of-capital module, live rf, ERP 4.4 % -> `docs/adr/007-cost-of-capital.md`
