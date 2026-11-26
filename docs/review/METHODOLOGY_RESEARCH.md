# Methodology research (2026-11-26)

Owner request: wherever the valuation or risk analysis looked doubtful, check it against authoritative sources
(Damodaran / NYU Stern, CFA Institute curriculum, McKinsey "Valuation", Nareit, the original papers) instead of
guessing. Three research passes, one per area, each row: claim -> source -> what the code does -> verdict -> fix.
Fix status of every discrepancy is in the "Dispositions" table at the end; `docs/METHODOLOGY.md` cites this file.

Caveat recorded by the research agents: a few source URLs were cited from memory and not opened
(Damodaran `firmeq.html`, `pbv.html`, `riskfree.pdf`, `ratings.html`, `capstructure.pdf`, the real-estate chapter).

## 1. DCF, WACC, APV, DDM, residual income

### Methodology audit: DCF, WACC, APV, DDM, residual income

Scope: `StockAnalysisSystem/src` in worktree `review-fixes-finish-2bedc3` (read-only review, 2026-11-26).
Paths below are relative to `StockAnalysisSystem/src/`. Verdicts: **matches** / **acceptable simplification** / **discrepancy**.

### Summary of discrepancies (by severity)

| # | Severity | Where | Problem |
|---|---|---|---|
| D1 | High | `pages/company_analysis/valuation.py:202,208,235-247`, `page_helpers.py:194-208,648-677` | DCF tab discounts levered FCF (after interest) at WACC and then deducts net debt: interest counted twice |
| D2 | Medium | `apv.py:59-72` | APV terminal tax shield grows at g but is discounted at Kd (mixes constant-debt and constant-leverage assumptions) |
| D3 | Medium | `base_valuation.py:864-889` | Generic/sector FCFF engine keeps SBC added back (reported FCF); DCF tab deducts it by default |
| D4 | Medium | `dcf_models.py:228-229`, `base_valuation.py:945-953`, `page_helpers.py:673` | Terminal value grows the last FCF instead of tying reinvestment to g/RONIC |
| D5 | Low-Med | `capital.py:69-79,114` vs `base_valuation.py:810-819,889`; `dcf_models.py:166` | Two tax-rate rules (ADR-007 says one); effective rate used for the debt tax shield instead of marginal |
| D6 | Low | `pages/company_analysis/page_helpers.py:677` | DCF tab equity bridge omits minority interest and preferred stock (engine subtracts them) |
| D7 | Low | `valuation.py:230,334`, `base_valuation.py:320`, `financial_sector.py:91` | Terminal growth allowed up to 5 % with no check against rf |
| D8 | Low | `utils/financial_data.py:80`, `financial_sector.py:210-211`, `base_valuation.py:1210` | RI / justified P/B / ROE use "Stockholders Equity" (includes preferred) and NI before preferred dividends |
| D9 | Low | `valuation.py:210,235` | SBC deduction uses the latest year even when the base FCF is a 3-year average |

### (1) DCF

| claim | source (URL) | code (file:line) | verdict | fix |
|---|---|---|---|---|
| FCFF (pre-interest) must be discounted at WACC; FCFE at Ke. Mixing levered cash flow with WACC and then subtracting debt double-counts the cost of debt | Damodaran, FCF primer https://pages.stern.nyu.edu/~adamodar/pdfiles/blog/FreeCF.pdf ; McKinsey *Valuation* ch. 8-10 (enterprise DCF) | Engine: `base_valuation.py:882-889` adds back after-tax interest to reported FCF -> FCFF, discounted at WACC (`:1008`, `:978`), debt deducted in bridge (`:771`). **Matches.** DCF tab: `valuation.py:202` uses `free_cash_flow_series` (`page_helpers.py:194-208`, OCF - capex = after interest), discounts at the WACC slider (`valuation.py:226,247`) and deducts net debt (`page_helpers.py:677`) | **discrepancy (D1)** | In `_render_dcf` build the base from unlevered FCF: `fcf + abs(interest_expense) * (1 - components["tax_rate"])` per period (same as `apv.unlevered_fcf` / `BaseValuation._unlevered_fcf_series`), or call `BaseValuation._unlevered_fcf_series`. Relabel the slider "Unlevered FCF". |
| Gordon terminal value TV = CF(n+1)/(r - g), requires r > g | Damodaran, Growth Rates and Terminal Value https://pages.stern.nyu.edu/~adamodar/pdfiles/ovhds/dam2ed/growthandtermvalue.pdf | `base_valuation.py:945-953`, `:968-974`; `page_helpers.py:673`; `dcf_models.py:282` (MC) | matches | - |
| Stable growth should not exceed nominal economy growth; rf is the proxy cap | Same PDF ("One simple proxy for the nominal growth rate of the economy is the riskfree rate"); https://pages.stern.nyu.edu/~adamodar/podcasts/valspr24/session11slides.pdf | Defaults 1-3 % (`config.py:179,186-190`) are below rf (~4-4.9 %). But sliders/params allow 5 % with no rf check: `valuation.py:230,334`, `base_valuation.py:320`, `financial_sector.py:91`; only r > g is checked (`valuation.py:244`) | **discrepancy (D7)**, low | Warn (or cap) when `terminal > components["risk_free_rate"]`; in the engine clamp `terminal_growth` to `min(g, rf)` in `_dcf_inputs` (`base_valuation.py:1009`) with a note. |
| Terminal reinvestment must be consistent with growth: reinvestment rate = g / RONIC; TV = NOPAT(n+1)(1 - g/RONIC)/(WACC - g) | Damodaran session slides (above, "growth has to be earned"); McKinsey *Valuation* value-driver formula (ch. "Continuing value") | Forecast-driven DCF has NOPAT and reinvestment per year but TV = last FCF x (1+g)/(r-g) (`dcf_models.py:228-229` -> `base_valuation.py:969,953`); generic DCF and DCF tab grow FCF with no reinvestment link (`base_valuation.py:1020`, `page_helpers.py:665-673`) | **discrepancy (D4)**, medium (FCF-growth DCFs: acceptable simplification; driver DCF: real, as it has the inputs) | In `forecast_driven_dcf` pass `terminal_cash_flow = nopat_n*(1+g)*(1 - g/RONIC)` to `_discount_cash_flows`, RONIC default = WACC (McKinsey's neutral case) or last-year ROIC capped. |
| Growth fade toward stable growth (multi-stage) | Damodaran, https://pages.stern.nyu.edu/~adamodar/pdfiles/ovhds/dam2ed/growthandtermvalue.pdf (three-stage model) | `base_valuation.py:193-198` (linear fade year 1 -> g_terminal at year n), `dcf_models.py:119` (3-stage), `page_helpers.py:661-662` (optional, default on `valuation.py:218`) | matches | - |
| Mid-year convention: cash flows arrive through the year; McKinsey discounts at t-0.5 (or multiplies by (1+WACC)^0.5) | McKinsey *Valuation* 7th ed. ch. 10 (mid-year adjustment) | Year-end discounting everywhere: `base_valuation.py:978,984`, `page_helpers.py:669`, `dcf_models.py:281`, `apv.py:62-63`; valuation date = last fiscal year end | acceptable simplification (understates EV by ~(1+WACC)^0.5 - 1 ~ 4-5 %; conservative and consistent across models) | Optional: `factor = (1+r)**-(year-0.5)` for explicit flows and TV x (1+r)^0.5. |
| SBC is a real expense; adding it back (as reported OCF/FCF does) overstates value | Damodaran FCF primer https://pages.stern.nyu.edu/~adamodar/pdfiles/blog/FreeCF.pdf ; blog https://aswathdamodaran.blogspot.com/2014/02/stock-based-employee-compensation-value.html | DCF tab deducts SBC by default (`valuation.py:221,235`); Technology sector model deducts (`dcf_models.py:522`, `tech_sector.py:246`); generic engine used by every other sector keeps SBC added back (`base_valuation.py:871-889`) | **discrepancy (D3)**, medium | In `_unlevered_fcf_series` subtract `stock_based_compensation` (aligned by period, as in `dcf_models.py:522`); keep a flag to disable. Then the Technology model need not do it separately. |
| SBC deduction must match the period of the base FCF | (consistency) | `valuation.py:210` takes latest-year SBC; `:235` subtracts it from either latest FCF or 3-year average FCF (`:208`) | **discrepancy (D9)**, low | Build an SBC series and subtract the same window (`sbc.tail(3).mean()` when the average is chosen). |
| Net debt bridge: Equity = EV - debt + cash - minority interest - preferred (+ non-operating assets) | Damodaran, Investment Valuation ch. "From firm value to equity value" https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/firmeq.html ; McKinsey ch. 16 | Engine: `base_valuation.py:767-777` (debt incl. yfinance leases, cash + ST investments, minority, preferred) | matches (non-operating LT investments not added: acceptable simplification, no mapping in `utils/financial_data.py`) | - |
| Same bridge in the DCF tab | (consistency with engine) | `page_helpers.py:677`: `equity = EV - net_debt` only | **discrepancy (D6)**, low | Subtract `minority_interest` and `preferred_stock` (reuse `BaseValuation._equity_bridge`). |
| Driver FCFF = EBIT(1-t) + D&A - capex - ΔNWC; ΔNWC = NWC% x ΔRevenue | McKinsey *Valuation* ch. 9; CFA L2 "Free Cash Flow Valuation" | `dcf_models.py:193-199`, NWC% excludes cash and ST debt (`:156-158`) | matches | - |

### (2) Cost of capital

| claim | source (URL) | code (file:line) | verdict | fix |
|---|---|---|---|---|
| US rf = 10-year Treasury (no default spread) | Damodaran, riskfree rate paper https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/riskfree.pdf | `utils/rates.py:24,66-77` FRED DGS10, override, fallback 4.5 % (`config.py:172`) | matches | - |
| ERP ~ implied ERP; Damodaran implied ERP 4.41 % (1 Nov 2026, trailing 12m with adjusted payout), computed vs the 10-yr T-bond | https://pages.stern.nyu.edu/~adamodar/ (home page, monthly update) | `capital.py:25` = 4.4 % (ADR-007 cites July 2026 4.42 %) | matches (constant; ADR already notes yearly review) | - |
| CAPM Ke = rf + beta x ERP | CFA L1 "Cost of Capital"; Damodaran https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/capstr.html | `capital.py:49-51`, bounds 4.5-25 % | matches | - |
| Blume adjusted beta = 0.67 x raw + 0.33 (Bloomberg convention, Blume 1971/1975) | Blume, M. (1975) "Betas and their regression tendencies", J. Finance 30(3) | `capital.py:42-46`, 1.0 when unknown, clamp 0.3-3 | matches (Damodaran prefers bottom-up betas: acceptable simplification) | - |
| Kd = current market pre-tax borrowing rate (rf + default spread, synthetic rating from interest coverage when unrated) | Damodaran ratings/spreads https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ratings.html | `capital.py:107-111`: interest / total debt clamped to rf + 0.5 %..8 %, default rf + 2 % | acceptable simplification (historical coupon, floored at rf + 0.5 % so stale low coupons cannot dominate) | Optional: synthetic rating from EBIT/interest. |
| Tax rate for the debt tax benefit should be the marginal rate (interest saves tax at the margin); effective rate for operating taxes | Damodaran Q&A https://pages.stern.nyu.edu/adamodar/New_Home_Page/valquestions/a3.htm ; https://pages.stern.nyu.edu/~adamodar/pdfiles/country/Brvaln01.pdf ("tax rate used to compute the after-tax cost of debt ... marginal") | Effective rate (latest reported, 0-35 %) used for Kd(1-t) `capital.py:114,122,152` and APV shield `valuation.py:341`/`apv.py:59` | **discrepancy (D5a)**, low-medium (e.g. AAPL effective ~16 % vs marginal ~21-25 %) | Add `MARGINAL_TAX_RATE = 0.25` (21 % federal + ~4 % state) for Kd(1-t), Hamada and APV shields; keep effective rate for NOPAT. |
| One company, one tax rule (ADR-007 "Tax rate: the sector-model rule") | `docs/adr/007-cost-of-capital.md` | WACC: `capital.effective_tax_rate` (latest, `capital.py:69-79`). FCFF interest add-back and driver NOPAT: `BaseValuation._effective_tax_rate` (3-year average, `base_valuation.py:810-819`, used `:889`, `dcf_models.py:166`) | **discrepancy (D5b)**, low | Make `_effective_tax_rate` delegate to `capital.effective_tax_rate(income)`. |
| Market-value weights E/(D+E) | McKinsey ch. 15; Damodaran capstr lecture (above) | `capital.py:87-96,112-114`: equity at market cap; debt at book | acceptable simplification (book debt standard proxy) | - |
| Bounding WACC | (practice) | `capital.py:31,118` clamp 6-20 % with `unbounded_wacc` exposed | acceptable simplification (floor documented; flagged rows per ADR-007) | - |

### (3) APV, DDM, residual income, justified P/B

| claim | source (URL) | code (file:line) | verdict | fix |
|---|---|---|---|---|
| Hamada: beta_u = beta_L / (1 + (1-t) D/E) | Damodaran, Applied Corp. Finance ch. 4; https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/capstr.html | `apv.py:17-21`; Blume first, then unlever (`apv.py:32`) | matches (Hamada implies constant debt and tax shields at Kd risk) | Use marginal tax rate (D5a). |
| APV = unlevered value at ku + PV(tax shields) (- expected distress costs) | Damodaran, APV in Applied Corp. Finance ch. 8 https://pages.stern.nyu.edu/~adamodar/pdfiles/acf3E/presentations/capstructure.pdf | `apv.py:41-78`; no distress costs | acceptable simplification (distress costs omitted; note in UI) | - |
| Tax-shield discount rate: Kd if debt is fixed in amount (Myers; perpetuity = t x D); ku if debt grows with the firm / constant leverage (Harris-Pringle 1985, Miles-Ezzell 1980; McKinsey APV) | Fernandez, "The value of tax shields is NOT equal to the present value of tax shields" https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=290727 ; https://www.iese.edu/media/research/pdfs/DI-0663-E.pdf | Forecast shields constant (fixed debt) at Kd `apv.py:59-65` (consistent). Terminal shield grows at g yet discounted at Kd `apv.py:70-72` | **discrepancy (D2)**, medium (growing perpetuity at Kd overstates the shield; e.g. Kd 6 %, g 3 % doubles it vs t x D) | Either (a) fixed debt: `pv_terminal_shield = shield / cost_of_debt / (1+kd)**years` (no growth), or (b) debt grows with the firm: discount the terminal shield at `unlevered_cost`: `shield*(1+g)/(ku-g)/(1+ku)**years`. Pick (a) to stay consistent with Hamada. |
| Unlevered FCF = levered FCF + interest x (1-t) | McKinsey ch. on APV; Damodaran FCFF definition | `apv.py:36-38`, `valuation.py:346` | matches | - |
| Two-stage DDM; stable payout = 1 - g/ROE; Gordon TV at Ke | Damodaran DDM https://pages.stern.nyu.edu/~adamodar/pdfiles/ovhds/dam2ed/growthandtermvalue.pdf ("Stable Growth Payout Ratio = 1 - g/ROE") | `base_valuation.py:1248-1311` (fade, stable payout `:1293`, TV via `:1298`); near-term g = ROE x retention `financial_sector.py:295` | matches | - |
| Residual income: V0 = B0 + Σ (ROE_t - Ke) B_(t-1)/(1+Ke)^t + TV; clean-surplus book roll-forward | CFA L2 "Residual Income Valuation" | `dcf_models.py:437-476`: excess on opening book (`:462`), clean surplus (`:464`), terminal excess on opening book of year n+1 discounted from n (`:469-472`) | matches. Note: default sustainable ROE = recent ROE capped 25 % (`financial_sector.py:228`), so excess returns persist forever (CFA persistence factor ω = 1): acceptable but aggressive | Optional: default sustainable ROE as a fade toward Ke (e.g. midpoint of ROE and Ke). |
| Justified P/B = (ROE - g)/(Ke - g) | CFA L2 "Market-Based Valuation: P/B"; Damodaran https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/pbv.html | `financial_sector.py:272-289` | matches | - |
| RI / P/B / ROE are defined on **common** equity and net income to common | CFA L2 "Residual Income Valuation" (book value of common equity) | `total_equity` resolves "Stockholders Equity" first (`utils/financial_data.py:80`), which includes preferred; net income before preferred dividends (`financial_sector.py:210-211`, `base_valuation.py:1209-1210`) | **discrepancy (D8)**, low (JPM: ~$20 bn preferred, ~6 % of book) | For financials use "Common Stock Equity" first and net income to common (`Net Income Common Stockholders`), or subtract `preferred_stock` from book. |

Read: `valuation/capital.py`, `valuation/apv.py`, `valuation/dcf_models.py`, `valuation/base_valuation.py`, `pages/company_analysis/valuation.py`, `pages/company_analysis/page_helpers.py`, `valuation/sector_specific/financial_sector.py`, `utils/rates.py`, `utils/financial_data.py`, `config.py`, `docs/adr/007-cost-of-capital.md`

## 2. Sector models

### Sector valuation methodology audit

Scope: `StockAnalysisSystem/src/valuation/sector_specific/*.py`, `valuation/base_valuation.py`, `valuation/dcf_models.py` (excess return, SBC FCF) and `utils/financial_data.py` (row mapping that feeds them). The code was only read, never edited. Paths below are relative to `StockAnalysisSystem/src/`.

Verdicts: **D** = discrepancy (formula or data error with code evidence), **S** = acceptable simplification, **M** = matches.

### Discrepancies (fix these)

| # | Claim (source says) | Source URL | Code file:line (code does) | Verdict | Fix |
|---|---|---|---|---|---|
| D1 | You can treat leases as debt, or leave them as operating costs, but never both. If lease liabilities sit in debt, add the lease expense (its interest part) back to operating income. If you leave operating income after rent, keep the leases out of debt. | https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valquestions/oplease.htm ; https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/oplev.pdf | `valuation/sector_specific/common.py:207` and `valuation/base_valuation.py:737-743` take debt from yfinance `Total Debt` (`utils/financial_data.py:88`). That row includes operating-lease liabilities: Yahoo's Total Debt = debt + `Capital Lease Obligations`. For example, GOOGL 2021 shows Total Debt 28.4bn = 14.8bn of debt + 15.6bn of leases (https://stackoverflow.com/questions/70090315). FCFF (`common.py:260`) and EBITDA stay after rent. `retail_sector.py:174-176` says so in its own note: "FCFF is after store rent; lease liabilities ... are deducted". | **D**: the lease is counted twice in every EV-to-equity bridge (`equity_bridge` `common.py:220`, `_equity_bridge` `base_valuation.py:759`) and in EV/EBITDA. This matters most for retail, restaurants, airlines and telecom towers. | Subtract the yfinance lease rows (`Long Term Capital Lease Obligation`, `Current Capital Lease Obligation`, or `Capital Lease Obligations`) from debt in `capital_structure` / `_total_debt`. Keep finance leases if you can split them out. The other option is to add lease interest (≈ lease liability × pre-tax cost of debt) after tax to FCFF and EBITDA. Then fix the note at `retail_sector.py:175`. |
| D2 | Nareit FFO = GAAP net income, **excluding gains/losses from sales** of depreciable real estate, **plus** real-estate depreciation and amortisation, **plus** impairments of depreciable real estate, plus adjustments for unconsolidated JVs. | https://www.reit.com/glossary/funds-operation-ffo ; Nareit FFO White Paper, 2018 Restatement (reit.com) ; issuer example: https://www.sec.gov/Archives/edgar/data/895417/000089541724000014/supplementalinformationtof.htm | `valuation/sector_specific/real_estate.py:137-141`: `ffo = net_income + da`. It does not subtract property-sale gains, does not add back impairments, and uses total D&A instead of real-estate D&A. `utils/financial_data.py` maps no gain-on-sale or impairment row. | **D**: FFO and AFFO are overstated in years with disposal gains (common for apartment and office REITs) and understated in impairment years. The error carries into P/FFO value at `real_estate.py:152-153`. | Map the yfinance rows `Gain On Sale Of Ppe` / `Gain On Sale Of Business` / `Gain On Sale Of Security` and `Asset Impairment Charge` / `Impairment Of Capital Assets`. Then compute `ffo = net_income - gains + impairments + da`, and subtract preferred dividends to get FFO per common share. |
| D3 | Excess-return and P/B models for banks value **common** equity. Start from common book equity (Damodaran uses book value of equity, net of preferred) and net income to common stockholders, after preferred dividends. | https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/finfirm09.pdf ; https://pages.stern.nyu.edu/~adamodar/pdfiles/eqnotes/finsvc.pdf | `utils/financial_data.py:80` puts `Stockholders Equity` first, which is common + preferred. `financial_data.py:66` puts `Net Income` first, which is before preferred dividends. These rows feed `financial_sector.py:210-211` (book value, net income), `:232` (BVPS) and `base_valuation.py:1206-1216` (ROE). `equity_value` (`dcf_models.py:473`) is then divided by common shares. | **D**: common value per share is overstated by roughly the preferred stock. JPM carries about $20bn of preferred (≈6% of book), and COF, SCHW, ALL, MET and others carry more. Justified P/B and P/E are inflated the same way. | For equity models, use `Common Stock Equity` (or `total_equity - preferred_stock`) and `Net Income Common Stockholders`. The simplest route is a `common_equity` key in the mapping that the financial model and `_return_on_equity` use. |
| D4 | Customer or subscriber value = per-period **cash flow** per customer (after the capex that keeps the customer served) × lifetime factor `1/(churn + r − g)`. Telecom is capital-intensive (capex ≈ 15-20% of revenue), so EBITDA is not cash flow. | https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/intangibles.pdf ; McKinsey, Valuation (7th ed.), ch. on capex intensity / telecom (book) | `valuation/sector_specific/communication_sector.py:136`: `arpu*12*margin*(1-tax)` with `margin` = **EBITDA margin** (`:121-123`). There is no capex deduction, and tax is charged on EBITDA. The same applies to `user_based` at `:155-156`. The EV then passes through the debt bridge. | **D**: with 38% EBITDA, 18% capex and D&A ≈ capex, the code takes 30% of revenue as cash where the right figure is about 16%. Subscriber value comes out about 2x too high for telecom and cable. | Use the average FCFF margin (`average_margin(ctx, "fcff", 3)`) or `(EBITDA − capex)` margin taxed as `EBIT×(1−t)+D&A−capex`. For `user_based`, also deduct SBC so it matches the tech model. |

### Checked and acceptable (no action needed)

| Claim (source says) | Source URL | Code file:line (code does) | Verdict | Fix / note |
|---|---|---|---|---|
| Banks and insurers are valued on equity only (no EV/FCFF DCF), with cash flows discounted at the cost of equity. | https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/finfirm09.pdf | `financial_sector.py:61-72`: bank, insurance and diversified use excess return, justified P/B, P/E and DDM, all at Ke; `fcf_dcf`/`ev_ebitda` weight 0. Ke = CAPM (`base_valuation.py:849-859`). | M | - |
| Excess return: equity = BV0 + Σ (ROE_t − Ke)·BV_{t−1}/(1+Ke)^t + terminal (ROE_s − Ke)·BV_n/(Ke − g)/(1+Ke)^n; stable payout = 1 − g/ROE. | finfirm09.pdf (above) | `dcf_models.py:437-476` | M | Correct timing (opening BV) and Ke > g guard. |
| Justified P/B = (ROE − g)/(Ke − g). | finsvc.pdf | `financial_sector.py:272-289` | M | Practitioners often use P/TBV with ROTCE. Book includes goodwill (S). Optional: add a tangible variant `(total_equity − goodwill_and_intangibles)` with ROTCE. |
| Bank DDM: g = ROE × retention; Gordon terminal at stable payout 1 − g/ROE. | finsvc.pdf | `financial_sector.py:291-300`, `base_valuation.py:1248-1313` | M | The payout uses dividends only and ignores buybacks (S). |
| Fee-based financials (asset managers, payments) may use FCFF DCF and EV/EBITDA. | finfirm09.pdf | `financial_sector.py:64-65,78-80` | M | - |
| REIT NAV = NOI / cap rate + cash − debt − preferred (− minority). | Damodaran, Investment Valuation ch. "Valuing Real Estate" (pages.stern.nyu.edu/~adamodar, URL not verified in this session) | `real_estate.py:113-133` uses EBITDA as the NOI proxy; the bridge is at `common.py:220-227`. | S | EBITDA ≈ NOI − G&A, which is roughly the same as capitalising G&A. It ignores land, development work in progress and JV NAV. Leases on REIT balance sheets are small, but D1 still applies. |
| AFFO = FFO − recurring (maintenance) capex − straight-line rent adjustment. | Nareit (above); REALPAC AFFO paper https://www.sfu.ca/~poitras/realpac_on_ffoaff-2019.pdf | `real_estate.py:142`: AFFO = FFO − 25% × D&A. There is no straight-line rent adjustment. | S | No recurring-capex or straight-line rent data is available. `affo × multiple × 1.15` (`:153`) ends up close to the FFO value, so it is harmless but adds little. Fixing D2 matters more. |
| Upstream NAV: PV of reserve cash flows (SEC PV-10 = future net cash flows at a 12-month average price, discounted at 10%). Analysts re-price at strip or mid-cycle prices. | https://www.sec.gov/rules/final/2008/33-8995.pdf (Modernization of Oil and Gas Reporting; ASC 932 standardised measure) | `energy_sector.py:198-231`: mid-cycle NOPAT (average EBITDA margin) is rescaled by `oil_price/75` with 2x leverage, then paid as an annuity over the reserve life at WACC. | S | There is no reserve volume or PV-10 input, and the $75 reference price is fixed (`:62`) rather than the period's realised price. Optional: read the 10-K standardised measure, or rescale from the realised price of the latest year. |
| Mid-cycle normalisation for commodity and cyclical businesses. | McKinsey, Valuation, cyclical companies chapter | `energy_sector.py:194-196`, `manufacturing.py:132`: FCFF margin averaged over all years | M | - |
| Integrated majors: SOTP (upstream NAV + downstream/chemicals at EV/EBITDA). | McKinsey, Valuation (SOTP) | `energy_sector.py:233-253` | S | 65% upstream share by default; it can be overridden. |
| Midstream: contracted cash flows at a low rate, merchant at a premium, re-contracting haircut. | Practitioner convention | `energy_sector.py:255-285` | S | The terminal value applies the base rate to merchant volumes too, which is a minor inconsistency. |
| Regulated utility: equity value ≈ rate base × equity ratio × (allowed ROE − g)/(Ke − g), i.e. justified P/B on regulated equity. | finsvc.pdf (justified P/B); practitioner RAB convention | `energy_sector.py:287-316`: net PP&E is the rate-base proxy, book equity/(equity + debt) the ratio, Ke = CAPM, g capped at terminal + 2pp. | S | Net PP&E includes construction work in progress and other non-rate-base assets, and holdco debt is not split out. Value goes straight to equity, so there is no double debt deduction. Correct. |
| Tech: SBC is a real expense and should be deducted from FCF (or treated as share dilution, never both). | https://pages.stern.nyu.edu/~adamodar/pdfiles/papers/intangibles.pdf | `dcf_models.py:518-523`: unlevered FCF (reported FCF + after-tax interest, `base_valuation.py:882-889`) − 100% SBC. Current share count, no option overhang. | M | Rule of 40 uses reported FCF margin before SBC (`tech_sector.py:243-260`). That is the industry convention (S). |
| Unprofitable growth: margins converge to a target, reinvestment = ΔRevenue / sales-to-capital, terminal reinvestment = g/ROIC. | Damodaran, young growth firms | `tech_sector.py:288-340` | M | Tax losses carried forward are ignored, which the code notes. |
| R&D capitalisation: unamortised asset = Σ R&D_{−k}·(N−k)/N; adjusted EBIT = EBIT + R&D − amortisation. | intangibles.pdf | `healthcare.py:118-148` | S | Amortisation includes the current year and drops year −N, which is minor. The FCFF value is unaffected because the adjustment is informational. |
| rNPV = Σ probability × PV(programme cash flows). | Practitioner convention | `healthcare.py:150-183` | S | Remaining development costs are not risk-weighted or deducted. Pipeline is added to a base DCF that already expenses current R&D, which is acceptable. |
| Retail: same-store growth on the mature base, plus NPV of new units. | McKinsey, Valuation | `retail_sector.py:182-242`: FCF = NOPAT·(1 − g/ROIC), new store NPV = value/(1+r) − build cost | M | The EV bridge carries D1 (leases). |
| Telecom: EV/EBITDA with capex-intensity awareness. | McKinsey, Valuation | `communication_sector.py:39-44,117` (FCFF DCF captures capex; EV/EBITDA 7x) | M (DCF) | D4 covers the subscriber method only. |
| Manufacturing EPV = normalised NOPAT / WACC; replacement value of the asset base. | Greenwald, Value Investing (EPV) | `manufacturing.py:144-172` | M | - |
| Method weights blend only the methods that ran; missing weight defaults to 1.0. | - | `base_valuation.py:527-535`; SOTP weights are added explicitly at `common.py:1065-1066,1112-1114` | M | - |
| FCFF = EBIT(1−t) + D&A − capex + ΔWC (cash-flow sign). | McKinsey / Damodaran | `common.py:259-262` | M | The fallback `OCF − capex` is after interest (FCFE-like), which is minor. |

## 3. Distress scores, ratios, risk metrics, liquidation value

### Methodology audit: distress models, ratios, market risk, liquidation, performance ratios, A1-A4/P1-P4

Code root: `StockAnalysisSystem/src` (worktree `review-fixes-finish-2bedc3`). Read-only review, 2026-11-26.
State at review time: `valuation/liquidation.py` exists (untracked). `models/performance_ratios.py` and
`models/balance_liquidity.py` do **not** exist yet, so the textbook definitions they must follow are given in sections 6 and 7.

Verdicts: **matches** / **acceptable simplification** / **DISCREPANCY**.

### 1. Bankruptcy models (`models/bankruptcy_models.py`)

| Claim | Source URL | Code file:line | Verdict | Fix |
|---|---|---|---|---|
| Altman 1968 Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E, D = market value of equity / book total liabilities | https://en.wikipedia.org/wiki/Altman_Z-score ; Altman 2000 "Revisiting the Z-Score and ZETA" https://moodle2.units.it/pluginfile.php/322514/mod_resource/content/0/Zscores.pdf | bankruptcy_models.py:41-43, 198, 214 | matches | - |
| Z cut-offs: safe > 2.99, grey 1.81-2.99, distress < 1.81 | same | :42, :218 | matches | - |
| Z' (private) = 0.717A + 0.847B + 3.107C + 0.420D + 0.998E, D = book equity / TL; cut-offs 2.90 / 1.23 | same | :44-46 | matches | - |
| Z'' (non-manufacturers) = 6.56A + 3.26B + 6.72C + 1.05D, no sales term; D = book equity / TL; cut-offs 2.60 / 1.10 | same; https://www.wallstreetprep.com/knowledge/altman-z-score | :47-49, :200-201, :217 (coef 0 skips E) | matches | - |
| Emerging-market EM score = 3.25 + Z'' with zones 5.85 / 4.35 | Zscores.pdf (Altman 2000/2005) | :51 alias `"emerging" -> non_manufacturing` | acceptable simplification (US Z'' cut-offs used; equivalent because EM = Z''+3.25 and 4.35-3.25 = 1.10, 5.85-3.25 = 2.60) | Optional: note in UI that "emerging" uses the Z'' scale without the 3.25 constant |
| Variant choice: Z for public manufacturers, Z' when no market value, Z'' for non-manufacturers; not for financials | Altman 2000; Altman et al. 2017 (J. Int. Fin. Mgmt & Acct.) recommend Z'' for non-manufacturers | :55-67, :146-159 | acceptable simplification (sector map + industry keywords; Technology defaults to Z but "software"/"internet" switch to Z'') | - |
| Springate 1978: S = 1.03 WC/TA + 3.07 EBIT/TA + 0.66 EBT/CL + 0.40 Sales/TA; S < 0.862 = failing | https://metricgate.com/docs/springate-score/ (secondary; original Springate 1978 MBA thesis, Simon Fraser) | :244, :250-257 | matches | - |
| Zmijewski 1984 X = -4.336 - 4.513 NI/TA + 5.679 TL/TA + 0.004 CA/CL | https://arxiv.org/pdf/2606.05623 ; https://core.ac.uk/download/357313344.pdf | :285-290 | matches (coefficients) | - |
| Zmijewski 1984 is a **probit** model: P(distress) = Phi(X), the standard normal CDF (Ohlson 1980 is the logit) | https://arxiv.org/pdf/2606.05623 ("P(bankruptcy) = Phi(ZMJ)"); https://core.ac.uk/download/357313344.pdf ("Zmijewski (1984) used probit") | :22 (docstring), :279 (formula string), :291 `1/(1+exp(-X))` | **DISCREPANCY** - logistic link applied to probit coefficients. Classification at 0.5 is unchanged (both cross 0.5 at X = 0), but the displayed probability is wrong: e.g. X = -1 -> logistic 26.9 % vs probit 15.9 %; X = 1 -> 73.1 % vs 84.1 % | `probability = scipy.stats.norm.cdf(score)` (or `0.5 * math.erfc(-score / math.sqrt(2))`); update docstring :22 and formula string :279 to `P = Phi(X)` |
| Distress models not applicable to banks/insurers | Altman 2000 (estimated on non-financial firms) | :56, :189-192, :240, :272 | matches | - |
| Interest coverage red flag = EBIT / interest | CFA Institute FRA curriculum | :370 | matches | - |

### 2. Ratios (`models/ratio_analysis.py`, `models/financial_statements.py`)

| Claim | Source URL | Code file:line | Verdict | Fix |
|---|---|---|---|---|
| Current ratio = current assets / current liabilities | CFA Institute FRA curriculum (Financial Analysis Techniques) https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/financial-analysis-techniques | ratio_analysis.py:369; financial_statements.py:275 | matches | - |
| Quick ratio = (cash + ST marketable securities + receivables) / CL | CFA FRA (same) | ratio_analysis.py:371; financial_statements.py:257-259, 276 | acceptable simplification (CA - inventory also includes prepaid/other current assets; slightly overstates) | Optional: use cash_and_short_term_investments + receivables when both are reported |
| ROE = net income / average total equity | CFA FRA | ratio_analysis.py:299, 360 | matches | - |
| ROA = net income / average total assets | CFA FRA | ratio_analysis.py:301, 361 | matches | - |
| ROIC = NOPAT / average invested capital, NOPAT = EBIT (1 - t); invested capital = debt + equity (- cash in Damodaran's version) | https://pages.stern.nyu.edu/~adamodar/ (Damodaran, "Return on Capital, ROIC and ROE") | ratio_analysis.py:310-312, 363 | acceptable simplification (fallback uses latest, not average, debt + equity; cash not netted) | - |
| Interest coverage = EBIT / interest expense | CFA FRA | ratio_analysis.py:382; financial_statements.py:209 | matches | - |
| Debt / equity = total debt / total equity (negative equity -> undefined) | CFA FRA | ratio_analysis.py:377; financial_statements.py:278 | matches | - |

### 3. Market risk (`pages/company_analysis/risk.py::market_risk_metrics`, TRADING_DAYS = 252 at :53)

| Claim | Source URL | Code file:line | Verdict | Fix |
|---|---|---|---|---|
| Volatility annualised as daily std x sqrt(252) | CFA Institute Portfolio Mgmt curriculum | risk.py:375 | matches | - |
| Sharpe (1994) = mean excess return / std of excess returns; annualised mean x 252 over std x sqrt(252) | Sharpe 1994 https://web.stanford.edu/~wfsharpe/art/sr/sr.htm | risk.py:377, 385 | acceptable simplification (numerator is geometric CAGR minus rf, denominator arithmetic vol; CAGR ~ arithmetic mean - sigma^2/2, so Sharpe is biased low by ~sigma/2, e.g. -0.15 at 30 % vol) | Optional for consistency with the coming Sortino/IR: `(returns.mean()*252 - rf) / volatility` |
| Historical 1-day VaR 95 % / 99 % = 5th / 1st percentile of daily returns | Jorion, *Value at Risk*; CFA | risk.py:379 | matches | - |
| CVaR (expected shortfall) = mean of returns <= VaR | Rockafellar & Uryasev 2000; CFA | risk.py:389 | matches | - |
| Max drawdown = min(P / running max - 1) | Young 1991 / standard | risk.py:378, 387 | matches | - |

### 4. Liquidation value (`valuation/liquidation.py`, exists, untracked)

| Claim | Source URL | Code file:line | Verdict | Fix |
|---|---|---|---|---|
| Liquidation value = sum(asset class book x recovery) - liquidation costs - all liabilities; equity floored at 0 | Damodaran, *The Dark Side of Valuation* (distressed firms; liquidation/fire-sale value) https://pages.stern.nyu.edu/~adamodar/ ; Graham & Dodd 1934 liquidation-value table | liquidation.py:7-10, 137-141 | matches | - |
| Recovery rates: Graham & Dodd ranges cash 100 %, receivables 75-90 %, inventory 50-75 %, fixed/misc assets 1-50 % (avg ~15 %); intangibles 0 | Graham & Dodd, *Security Analysis* (1934), ch. "Significance of the current-asset value" | liquidation.py:67-70 | acceptable simplification (all within textbook ranges; orderly PP&E 50 % is at the top of the G&D fixed-asset range, fine for "orderly"; forced inventory 35 % below G&D but in line with fire-sale) | - |
| Orderly vs forced: same balance sheet, lower rates and higher costs for forced | Damodaran (fire-sale discount) | :67-73, :156-159 | matches | - |
| Asset classes sum to total assets; "other" residual, no double counting | - | :98-122 | matches | - |
| Not for banks/insurers | Damodaran (financial service firms) | :40-42 | matches (docstring; enforce in caller) | - |
| Minority interest: if `total_liabilities` excludes NCI (yfinance "Total Liabilities Net Minority Interest"), NCI claims rank ahead of parent equity in a break-up | - | :153 | acceptable simplification (NCI usually small) | Optional: subtract `minority_interest` when reported |

### 5. Performance ratios (`models/performance_ratios.py` - NOT PRESENT at review time; definitions it must follow)

| Ratio | Textbook definition (source) | Pitfalls to avoid |
|---|---|---|
| Sortino | (R_p - MAR) / DD, DD = sqrt( (1/N) * sum over ALL N periods of min(0, r_t - MAR)^2 ). Sortino & Price 1994 "Performance measurement in a downside risk framework", J. Investing; CFA/GIPS note https://rpc.cfainstitute.org/-/media/documents/code/gips/the-sortino-ratio.pdf ; Rollinger & Hoffman (CME) https://www.cmegroup.com/content/dam/cmegroup/education/files/sortino-a-sharper-ratio.pdf | Divide by N (all periods), not by the count of negative periods; do NOT use `returns[returns<0].std()` (that subtracts the mean of negatives and changes the denominator). Daily MAR = annual MAR / 252. Annualise: numerator x 252, DD x sqrt(252). |
| Treynor | (R_p - R_f) / beta_p (CFA Portfolio Mgmt) | Annualised returns; beta from the same window and frequency as returns. |
| Information ratio | (R_p - R_b) / TE, TE = std(R_p - R_b) (CFA "Active return / active risk") | Annualise active mean x 252, TE x sqrt(252). Benchmark = index (e.g. SPY), not risk-free. |
| Jensen's alpha | alpha = R_p - [R_f + beta (R_m - R_f)] (Jensen 1968; CFA) | Annualised; same beta and window as Treynor. Equivalent: intercept of the excess-return regression x 252. |
| Calmar | annualised return (CAGR) over the trailing 36 months / abs(max drawdown over the same 36 months) (Young 1991, *Futures*) https://en.wikipedia.org/wiki/Calmar_ratio | Use a 36-month window (full history = MAR ratio, label it so if used); denominator positive (abs of drawdown). |

### 6. Russian balance-sheet liquidity grouping (`models/balance_liquidity.py` - NOT PRESENT; definitions it must follow)

Sources: Sheremet A.D., Negashev E.V. *Metodika finansovogo analiza*; Savitskaya G.V. *Analiz khozyaistvennoi deyatelnosti*; https://buh.ru/articles/analiz-platezhesposobnosti-peregruppirovka-statey-bukhgalterskogo-balansa.html ; https://www.klerk.ru/buh/articles/506407

| Group | RAS lines (Russian balance sheet) | US-GAAP mapping for yfinance |
|---|---|---|
| A1 most liquid | 1250 cash + 1240 short-term financial investments | cash_and_short_term_investments |
| A2 quickly realisable | 1230 receivables due within 12 months | receivables |
| A3 slowly realisable | 1210 inventories + 1220 VAT + 1260 other current assets (+ long-term receivables in some editions) | inventory + other current assets |
| A4 hard to realise | 1100 non-current assets | total non-current assets |
| P1 most urgent liabilities | 1520 accounts payable | accounts payable |
| P2 short-term liabilities | 1510 short-term borrowings + 1550 other short-term liabilities | current debt + other current liabilities |
| P3 long-term liabilities | 1400 long-term liabilities (+ 1530 deferred income, 1540 provisions in some editions; Sheremet puts 1530/1540 into P4) | total non-current liabilities |
| P4 permanent liabilities | 1300 equity (+ 1530 + 1540 in Sheremet) | total equity |

Absolute liquidity conditions: A1 >= P1, A2 >= P2, A3 >= P3, A4 <= P4 (the fourth follows from the first three since the balance sheet balances; check it as a consistency test). Derived: current liquidity TL = (A1+A2) - (P1+P2); prospective liquidity PL = A3 - P3. Pitfalls: A-groups must sum to total assets and P-groups to total liabilities + equity; do not put long-term debt into P2; state the edition used for 1530/1540 and 1260.

## Dispositions

| Finding | Disposition |
|---|---|
| Risk: Zmijewski probability used the logistic | Fixed 6c04793: P = Phi(X) (probit), test `test_zmijewski_probability_is_probit`. |
| Risk: Sortino / IR / Calmar definitions | Followed in T23 (`models/performance_ratios.py`). |
| DCF/WACC D1-D9, Sectors D1-D4 | See `methodology-fixes` section below (filled when the fixes land). |
