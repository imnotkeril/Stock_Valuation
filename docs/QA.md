# QA report — 2026-11-23

Branch `claude/peaceful-euler-aq2j08`, Windows 11, Python 3.11.9, Streamlit 1.67.0, live Yahoo
Finance data (no API keys).

## What was run

| Check | Result |
|---|---|
| Offline suite `pytest` | 282 passed (280 + 2 new secrets tests) |
| `ruff check StockAnalysisSystem` | All checks passed |
| Live smoke (AppTest, real network): Company Analysis for AAPL, JPM, XOM, NEE, AMZN, PG, CAT, PLD, JNJ, GOOGL | 10/10 OK, 15 tabs each, correct sector model picked (Utilities → Energy, Staples → Retail) |
| Live smoke: every other page opened and every button clicked (Home ×17, Peer Comparison, Sector Analysis, Stock Screener, Valuation Filter, Forecasting ×3, Reports) | all OK, no exceptions, no `st.error` |
| Browser: Home, AAPL Company Analysis → Valuation tab | renders; no app errors in console |
| `pip-audit` on the project venv | No known vulnerabilities (after upgrading venv `setuptools` 65.5 → latest) |
| Docker: `docker build -f StockAnalysisSystem/Dockerfile`, `docker run -p 8502:8501` | builds; `/_stcore/health` → `ok` |
| Local server binding | listens on `127.0.0.1:8501` only; log shows a single local URL |

Script for the live smoke run is not part of the repo (it hits the network); the same flows
are covered offline by `tests/test_app.py` and `tests/test_pages_interaction.py`.

## Fixed during QA
- Duplicate sidebar navigation (Streamlit auto-listed `src/pages/`): `showSidebarNavigation = false`.
- 33 × deprecated `use_container_width=True` → `width="stretch"` (Streamlit announced removal).
- Server listened on all interfaces with `enableCORS = false`: now `address = "localhost"`,
  CORS default restored. Docker keeps `--server.address=0.0.0.0`.
- Ported guard test against hard-coded API keys (`tests/test_no_hardcoded_keys.py`).
- Financials (JPM): the Valuation → DCF tab and the Financials → FCF sustainability block no longer
  warn about negative free cash flow; they explain that FCF is not meaningful for banks and insurers.
  Re-checked live: JPM and AAPL render with no warnings.

## Observations (not bugs, tracked in plan)
- Stock Screener first run takes ~150 s live (downloads the whole ~130-company universe);
  later runs use the cache.
- AAPL weighted fair value $193.96 vs price $342.56 (−43%): conservative defaults; values are
  assumption-driven by design (see SPEC §3.3).
- Macro forecast (FRED) needs optional `pandas-datareader`; not installed, button reports it.

## Not covered
- Alpha Vantage fallback (no key used).
- Optional `prophet` and `pandas-datareader` paths.
- Linux/macOS run (CI covers tests on Ubuntu only).

## v2 re-check — 2026-11-24

| Check | Result |
|---|---|
| Offline suite | 363 passed |
| `ruff check` | clean |
| Live smoke, Company Analysis (XOM, AAPL, JPM, PLD, JNJ, CAT, GOOGL, NEE, AMZN, PG) | 10/10 OK, 18 tabs each (new: APV, Precedents, News) |
| Browser: Sector Analysis > Compare sectors (4 sectors, live) | OK: XLK/XLF/XLE/XLV vs SPY, leaders table |
| Browser: XOM Valuation > APV | renders; default unlevered cost was 3.7 % from a very low beta - now floored at 6 % (APV $109 vs price) |

Not live-checked (need the user's settings): SEC EDGAR (`SEC_EDGAR_USER_AGENT`), News
(`FINNHUB_API_KEY` / `ALPHA_VANTAGE_API_KEY`), FMP M&A feed. All three are covered offline with
responses in the providers' JSON formats.
