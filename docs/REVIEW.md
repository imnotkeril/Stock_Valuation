
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
