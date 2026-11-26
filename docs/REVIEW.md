
## Security reviewer — P1

Scope: `git diff 9e99d37..HEAD -- StockAnalysisSystem` (T01-T08: utils/rates.py, DataLoader.get_fred_series, valuation/capital.py, Valuation / Risk / Peer / Valuation Filter pages). Local single-user app; no auth, no money paths, no uploads.

### Findings

| id | severity | file:line | finding | evidence | fix |
|---|---|---|---|---|---|
| - | - | - | No verified findings in this diff. | See checks below. | - |

### Checks (category: result)

| # | Category | Result |
|---|---|---|
| 1 | Access control | n.a. - single-user local app, no roles or per-user data. |
| 2 | Auth/session | n.a. - no auth. `reset_company_widgets` (main.py) only deletes own `st.session_state` keys with `val_`/`risk_` prefixes. |
| 3 | Injection | checked. `STOCK_ANALYSIS_RISK_FREE_RATE` -> `rates.parse_rate`: `float()` only, nan/inf/negative/>20 % rejected by `PLAUSIBLE_RANGE`; warning log does not echo the value. FRED series id is the constant `DGS10`; cache name goes through `_cache_path` sanitiser (data_loader.py:138-140). No shell/SQL/path sinks added. |
| 4 | XSS (unsafe_allow_html) | checked. valuation.py:117-121 badge interpolates `label`/`color` from the `RECOMMENDATION_THRESHOLDS` constants and `confidence` from `assess_confidence` (base_valuation.py:139, literal High/Medium/Low); `_confidence_line` (valuation.py:150-155) emits only that literal. `confidence_reasons` go to `st.caption` (markdown, no HTML) and are built from numbers plus internal method names. risk.py:161 level/colour come from bankruptcy_models.py:433-434 constants. No external string (company name, Yahoo field) reaches an HTML sink. |
| 5 | SSRF | checked. `_fetch_fred` (data_loader.py:585-589): fixed HTTPS host `fred.stlouisfed.org`, params are a constant id and dates, `timeout=30`, `raise_for_status`. No user-controlled URL. |
| 6 | Deserialization | checked. FRED CSV parsed with `pd.read_csv` + `to_numeric(errors="coerce")`; only the resulting DataFrame is pickled. Pickle cache (`_read_cache`, data_loader.py:147-155) is pre-existing and local (git-ignored `src/data/cache/*`); writing it already requires local code-exec level access. |
| 7 | Money/credits | n.a. - research tool, no payments or orders. Data-integrity of the rate: implausible FRED values fall back to the constant (rates.py:73-75). |
| 8 | Secrets/data exposure | checked. `.env` ignored (`.gitignore:20`), never committed (`git log --all -- .env` empty). Each `.env` value searched in tracked files and in all commits of the range: no match. `git log -p 9e99d37..HEAD` grep for api_key/secret/password/token: only `from-chat-tokens` counters in docs. Keys present (names only): ALPHA_VANTAGE_API_KEY, FINNHUB_API_KEY, FINANCIAL_MODELING_PREP_API_KEY, SEC_EDGAR_USER_AGENT. |
| 9 | Abuse/DoS | checked. FRED call cached for a day with stale fallback; no new costly endpoint. |
| 10 | Deps/config | checked. `requirements.txt` unchanged in range; pip-audit clean. `.streamlit/config.toml` binds `localhost`; Dockerfile:28 `--server.address=0.0.0.0` is pre-existing, container-internal, runs as `USER app` (exposure then depends on `docker run -p`; publish as `127.0.0.1:8501:8501`). |

Tool output:
```
$ python -m pip_audit -r StockAnalysisSystem/requirements.txt
No known vulnerabilities found
```

### Dropped candidates
- CSV formula injection in Valuation Filter export (valuation_filter.py:355-357): new columns `Method range` / `Check` are formatted numbers and constants; company names come from Yahoo, not an attacker-controlled channel. Dropped.
- Unbounded FRED response size: fixed trusted HTTPS host; no realistic exploit. Dropped.
- Pickle cache RCE: requires write access to the local cache dir; pre-existing, outside the diff. Dropped.

Trap-worthy: none new.

## Reviewer (correctness) — P1

Scope: `git diff b6b2045..7efbd38 -- StockAnalysisSystem` plus `afec7ae` (utils/rates.py). Checked against SPEC F20 AC20.1-AC20.6, PRODUCT_REVIEW §5 R1-R6, ADR-007.

| id | sev | file:line | finding | evidence | suggested fix |
|---|---|---|---|---|---|
| R-01 | P1 | StockAnalysisSystem/src/valuation/base_valuation.py:127-130 | `assess_confidence` counts "weighted methods" from `get_method_weights(subsector)`, which falls back to `DEFAULT_METHOD_WEIGHTS` (base_valuation.py:341-344) holding the methods of *every* subsector. Methods never run because they do not apply to the subsector count as unusable, so the company loses a confidence point and can be pushed to Low, which softens the badge (AC20.6). This hits XOM, one of the required live-check tickers. | Offline probe, synthetic `oil_gas` (Energy, `integrated`): 4 methods run, `excluded == {}`, yet the reasons include `'4 of 8 weighted methods usable'`. Energy weights = 8 keys (`reserve_nav`, `contracted_dcf`, `regulated_asset_base`, `project_dcf`, ...), available methods = 4. Retail `specialty_retail`: `dividend_discount` is weighted but not available, so it reports "2 of 5" where 2 of 4 is right. | Planned = `[k for k, w in weights.items() if w and w > 0 and k in methods]` (methods that ran, or `get_available_methods(subsector)`). Add a test: a sector with cross-subsector default weights and every applicable method usable must not get this reason. |
| R-02 | P2 | StockAnalysisSystem/src/pages/screening/valuation_filter.py:745 | The "Discount Rate Adjustment" adds to the sector table rate (`default_discount_rate`), not to the shared WACC the model now uses by default. So 0 % gives the WACC (e.g. 8.3 %) but +0.25 % jumps to the table rate + 0.25 % (e.g. 10.25 %). The setting is not continuous and goes around the single cost of capital (AC20.2). Existed before this diff, but T05 touched this path. | `dcf_overrides()` builds `overrides["discount_rate"] = defaults["default_discount_rate"] + discount_rate_adj` only when adj != 0. | Base the adjustment on `capital.cost_of_capital_for(financial_data, default)["wacc"]`, or pass it as a delta the model applies to its own WACC. |
| R-03 | P2 | StockAnalysisSystem/src/pages/screening/valuation_filter.py:910, 986 | (a) Combined approach: `method_values or valuation_components` picks the sector-model method values when both exist, so the "Method range" and spread ignore the 40 % relative leg, while the spread is divided by the blended fair value. (b) `method_values_of` keeps methods whose normalised weight is 0 (`key in weights`), which can widen the range and give a false Low confidence. | Code reading. `summarize_valuations` puts zero-weight methods in `weights` but not in `methods_used`. | Combined: pass `valuation_components` (or merge it with the relative value). Filter with `weights.get(key, 0) > 0` or `key in summary["methods_used"]`. |
| R-04 | P2 | StockAnalysisSystem/src/pages/company_analysis/valuation.py:324 | The APV "Cost of debt" slider runs from 1-15 %, but Kd is now capped at rf + 8 %. A user override or a FRED rate above 7 % (allowed up to 20 % by `PLAUSIBLE_RANGE`) gives a default above 15 %, and `st.slider` raises StreamlitAPIException. | `DEBT_SPREAD_BOUNDS = (0.005, 0.08)`; rates.py `PLAUSIBLE_RANGE = (0.0, 0.20)`. | Clamp the default to the slider range, or set max = max(15, rf + 8 %). |
| R-05 | P2 | StockAnalysisSystem/src/pages/comparison/peer_comparison.py:664-668 | `valuation_method_label`, `valuation_rates` and `_pct_or_na` are defined after the `if __name__ == "__main__": run_peer_comparison_page()` block. If the file is run directly it raises NameError. Imports via main.py work. | File layout. | Move the helpers above the `__main__` guard. |
| R-06 | P2 | StockAnalysisSystem/src/pages/company_analysis/risk.py:101, 302 | `sector == "Financials"` drops the cash-flow health component and the revenue stress test for the whole sector. That includes payment networks and asset managers (V, MA, BLK, sector "Financials"), whose FCF and operating leverage do matter. The caption calls every one of them "Bank / insurer". | Code reading. The test covers only the synthetic bank. | Key off the financial sector model's detected subsector (bank / insurer) rather than the sector name. |
| R-07 | P2 | StockAnalysisSystem/src/pages/company_analysis/valuation.py:356-370 | The "Why do DCF, APV and the Sector Model differ?" text says the gap is "not from the discount rate". That is wrong once the user moves the DCF discount or APV Ku slider, and for banks the sector model discounts at Ke, not at the WACC the text quotes. | Code reading. | Say "by default all three start from the same cost of capital" and mention that equity models (banks, DDM) use the cost of equity. |

Verified as not bugs: zero debt (Kd = rf + 2 %, equity weight 1), missing beta (1.0, `beta_assumed`), zero or negative equity (sector default WACC), NaN inputs (`to_float`/`safe_divide`), non-positive fair values (filtered in `summarize_valuations`), upside 0 with Low confidence (stays Hold), `get_recommendation` has no other callers, the reset of `val_*`/`risk_*` keys covers both ticker-change paths (sidebar button and cross-page `nav.go_to`; both go through `_set_request`), no other page uses the `val_`/`risk_` key prefixes, and the sector `cost_of_capital` keeps the old debt definition (`capital_structure` == `total_debt`).

Pointers for other reviewers: none.

AC coverage (6/6 implemented): AC20.1 rates.py + rf caption/label (tests test_rates.py). AC20.2 capital.py used by base/sector/APV/stress/Sharpe (test_capital.py). AC20.3 reset_company_widgets (test_widget_state.py). AC20.4 sanity_check/method_label (test_valuation_filter.py). AC20.5 risk.py/peer_comparison.py (test_risk_page.py, test_peer_comparison.py). AC20.6 assess_confidence/get_recommendation (test_recommendation.py). Weak point: R-01 can soften the label wrongly.

Commands:
- `python -m pytest -o addopts="" -p no:warnings -q` -> `447 passed in 150.46s (0:02:30)` (run at 7efbd38, before 58aa9f4/T12)
- `python -m ruff check StockAnalysisSystem` -> `All checks passed!`
- Offline probe (scratchpad) over all synthetic profiles with `STOCK_ANALYSIS_RISK_FREE_RATE=4.5`: oil_gas `['methods disagree (range 66% of the fair value)', '4 of 8 weighted methods usable']`, excluded `{}`.

Trap-worthy for known-traps.md: "Default weight tables that span every subsector must be intersected with the methods actually run before any 'n of m methods' ratio is computed."

## Dispositions (build session, 2026-11-26)

| Id | Severity | Disposition |
|---|---|---|
| S-* | - | Security reviewer: nothing found (0 P0/P1/P2); pip-audit clean; `.env` never committed. |
| R-01 | P1 | Valid, fixed: `assess_confidence` counts only weighted methods that ran; test `test_weights_of_methods_that_never_ran_are_ignored`. |
| R-02 | P2 | Valid, open: Valuation Filter discount adjustment should shift the shared WACC (fix in T18). |
| R-03 | P2 | Valid, open: Combined range should include the relative part and skip zero-weight methods (T18). |
| R-04 | P2 | Valid, open: APV Kd slider max 15 % < rf + 8 % when rf > 7 % (T18). |
| R-05 | P2 | Valid, open: peer_comparison helpers defined after the `__main__` guard (T18). |
| R-06 | P2 | Valid, open: bank special case also hits payment networks / asset managers (V, MA, BLK) (T18). |
| R-07 | P2 | Valid, open: "not from the discount rate" wording; banks use cost of equity (T18). |

# Round 2 (P2 diff T11-T16, 2026-11-26)

### Reviewer (correctness) round 2 - P2 changes T11-T16 (58aa9f4..dd6e02f)

| ID | Sev | File:line | Finding | Evidence | Suggested fix |
|---|---|---|---|---|---|
| R-08 | P1 | StockAnalysisSystem/src/pages/screening/valuation_filter.py:842 (with :743) | The Valuation Filter's relative leg never runs. `sector_benchmarks` comes from `ratio_analyzer.get_sector_benchmarks(...)`, which returns `{category: {...}}` (keys `valuation`, `profitability`, ...) for one sector. Looking it up with `sector_benchmarks.get(stock_sector, {})` therefore always gives `{}`. The result: approach "Relative" never returns a row, and "Combined" always falls back to the sector model only (the 60/40 blend never happens). The bug dates from 1ccff22, but T15's parity test hides it: `tests/test_parallel_fetch.py:83` skips the non-empty assert for Relative ("offline fixtures carry no sector multiples"), so it compares `[] == []`. | Probe `scratchpad/probe_relative.py`: keys are `['valuation','profitability','liquidity']` and `lookup('Technology')` is `None` for both sector="All" and "Technology". `perform_valuation_analysis(..., "Relative", "Technology", margin_of_safety=-10, include_outliers=True)` gives 0 rows. `calculate_relative_valuation` with `get_sector_benchmarks("Technology")["valuation"]` returns a result. | Use `sector_multiples = ratio_analyzer.get_sector_benchmarks(stock_sector).get("valuation", {})`, precomputed per sector before the pool (or reuse `relative_benchmarks()` from the Relative tab). In the test, require rows for Relative and Combined and check that Combined values blend both legs. |
| R-09 | P2 | StockAnalysisSystem/src/pages/screening/valuation_filter.py:1127-1175 | Once R-08 is fixed, the filter's relative leg ignores T12's `irrelevant_multiples`. It would value Technology names on P/B, and banks on P/S and EV/EBITDA, which the Relative tab now marks "Used: No". The filter and the tab would give different relative fair values for the same company. | `calculate_relative_valuation` uses all four multiples with no sector check. The tab filters with `key not in excluded` (valuation.py `_render_relative`). | Drop the components whose key is in `irrelevant_multiples(stock_sector)` (map labels to keys) before averaging. |
| R-10 | P2 | StockAnalysisSystem/src/pages/screening/stock_screener.py:647 | `parallel_map` cannot be interrupted. If the main thread raises while it consumes results (Streamlit raises RerunException/StopException from `bar.progress` in `progress_callback` when the user changes a widget or presses Stop), `ThreadPoolExecutor.__exit__` calls `shutdown(wait=True)` and runs every queued download first. With 132 universe tickers the app is stuck for tens of seconds before the rerun starts. The old sequential loop stopped at once. | Probe `scratchpad/probe_abort.py`: callback raises on the first result, and the output is `aborted after 1.65s, tasks run: 30/30`. | Wrap the consume loop: `except BaseException: pool.shutdown(wait=False, cancel_futures=True); raise`. You can also check a stop flag in `run()`. |
| R-11 | P2 | StockAnalysisSystem/src/industry/benchmarks.py:430-459 | `live_valuation_medians` takes any cached info file whatever its age, and `as_of` is the newest file's mtime. One freshly analysed company makes the caption read "data as of <today>" even when the other n-1 multiples are weeks old. The cache keeps info for 7 days and stale files indefinitely. | Code: `newest = stamp if ... stamp > newest`. There is no `_is_fresh` check. The caption is in valuation.py `relative_benchmarks`. | Report the oldest date or a range ("as of A to B"), or skip files older than a bound, e.g. `INFO_CACHE_DAYS` x 4. |
| R-12 | P2 | StockAnalysisSystem/src/main.py:357-368, 381, 453 | The one-step pick also fires on blur, not only on Enter. `text_input.on_change` runs when the box loses focus. Say a user types an exact ticker that is also a prefix of other results ("C", "T", "MA", "V") and then clicks the "Matches" selectbox to pick, for example, CRM. The blur rerun analyses C at once. Afterwards the sidebar can show a different company than the one analysed. The callback docstring admits the blur case, but the help text promises "Press Enter". | `unique_match` returns the exact symbol even with many results: `search_universe('C')` gives `['C','CRM','CSCO',...]`. Streamlit commits text_input on blur. | Auto-run only when the result list has exactly one entry, or when the analysed ticker is also written into `ca_search_choice` so the sidebar and the analysis agree. Change the help text to match. |
| R-13 | P2 | StockAnalysisSystem/src/models/precedent_transactions.py:39 | The illustrative sample gives Financials EV/Revenue 3.0 and EV/EBITDA 10.0. For JPM-type companies the Precedents tab then shows a per-share value from deal EV/EBITDA (`implied_values`). In the same commit range (T12), the Relative tab declares EV/EBITDA and EV/Revenue "not meaningful for banks and insurers". | `sample_deals("Financials")` produces five rows with `ev_ebitda` 7.5-13. `_render_precedents` has no sector guard. | For Financials, show an info note instead of the sample, or omit the EV-based sample rows. A deal P/B or P/TBV column would be needed for a meaningful bank sample. |

Checked, nothing found:
- `perform_valuation_analysis` refactor: all 4 `continue` paths became `return None`, the except path returns None, outlier and row order is preserved (`parallel_map` keeps input order).
- Shared `DataLoader` cache files are per symbol. The FRED file is warmed before the pool.
- The `ValuationFactory._instances` lazy init race is benign, and model `get_valuation` does not mutate instance state.
- `yfinance` 1.8 `download` no longer uses the shared `_DFS`.
- `st.cache_data` from workers is fine with the attached context.
- DCF fade math: year 1 = g, final year = terminal g, hand-checked by test_dcf_fade.
- SBC checkbox state is reset on company change via the `val_` prefix.
- Peer/sector main-area buttons set the same `*_active` flags as the sidebar buttons.

Tests run: `pytest test_parallel_fetch test_benchmarks test_dcf_fade test_golden_values test_precedent_transactions test_pages_interaction test_recommendation` gives 76 passed in 95.57s.

## Round 2 dispositions

| Id | Severity | Disposition |
|---|---|---|
| R-08 | P1 | Valid, fixed 6c04793: relative leg uses `get_sector_benchmarks(stock_sector)["valuation"]`; T15 parity test now requires Relative rows. |
| R-09 | P2 | Valid, fixed: `calculate_relative_valuation(..., sector=)` drops `irrelevant_multiples`. |
| R-10 | P2 | Valid, fixed: `parallel_map` cancels queued futures on any BaseException from the progress callback. |
| R-11 | P2 | Valid, fixed: cache entries older than 30 days skipped; `as_of` = oldest file used. |
| R-12 | P2 | Valid, fixed: auto-run only on a single search result; exact symbol preselected in Matches. |
| R-13 | P2 | Valid, fixed: no EV-multiple sample for Financials, note instead. |
