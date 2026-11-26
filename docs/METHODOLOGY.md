# Methodology: how every number is made

This document describes what the Stock Analysis System computes, from which inputs, with which
defaults and bounds. It describes the code as it is on branch `claude/review-fixes-finish-2bedc3`
(2026-11-26). Where a choice is a simplification or a known bias, it says so in
[Known biases and open items](#10-known-biases-and-open-items).

Research and education only. Every output is a model estimate built on public, sometimes delayed
or incomplete data and on assumptions you can change. It is not investment advice, and it is not
a forecast of any share price.

Sources for the choices below: [`review/METHODOLOGY_RESEARCH.md`](review/METHODOLOGY_RESEARCH.md)
(three source-checked audits and their dispositions), [`adr/007-cost-of-capital.md`](adr/007-cost-of-capital.md),
and the code paths named in each section. Primary references: Damodaran, NYU Stern
(<https://pages.stern.nyu.edu/~adamodar/>); McKinsey, *Valuation*; CFA Institute curricula; Blume (1975),
*J. Finance*; Hamada (1972); Myers (1974); Harris & Pringle (1985); Altman (1968, 2000); Springate (1978);
Zmijewski (1984); Sortino & Price (1994); Sharpe (1994); Jensen (1968); Young (1991); Graham & Dodd (1934);
Nareit FFO White Paper (2018 restatement); Sheremet & Negashev, *Metodika finansovogo analiza*.

## 1. Data

| Input | Source | Notes |
|---|---|---|
| Prices, company info, statements | Yahoo Finance (`yfinance`) | Unofficial API; statements cover 4-5 years. Canonical line items in `utils/financial_data.py` |
| 10-year Treasury yield | FRED `DGS10` (public CSV, no key) | Risk-free rate, see section 2 |
| SEC filings and 10-year XBRL history | SEC EDGAR (`SEC_EDGAR_USER_AGENT`) | Overview tab only |
| Insider transactions | Finnhub `stock/insider-transactions` (`FINNHUB_API_KEY`) | Section 9 |
| News sentiment | Finnhub, Alpha Vantage (keys) | Lexicon score on headlines; context, not a signal |
| Precedent deals | User table, optional FMP M&A feed | Section 6 |
| Sector benchmarks | Static table in `industry/benchmarks.py` | Section 8 |

Cache: `StockAnalysisSystem/src/data/cache`. When a provider fails, the last cached copy is used.
Offline mode (`STOCK_ANALYSIS_OFFLINE=1`) reads the cache only.

## 2. Risk-free rate

Code: `utils/rates.py`, `risk_free_rate()`. Order of sources:

1. User override: environment variable `STOCK_ANALYSIS_RISK_FREE_RATE`, as a fraction (`0.045`) or a
   percent (`4.5`). Values outside 0-20 % are ignored with a warning.
2. FRED `DGS10`, latest valid observation, percent divided by 100. Cached for a day; a stale cached copy
   is used if FRED is down. An implausible value (outside 0-20 %) is rejected.
3. Fallback constant `config.RISK_FREE_RATE_FALLBACK` = 4.5 %.

The rate, its source and its as-of date are stored with every result (`risk_free_rate`,
`risk_free_rate_source`, `risk_free_rate_as_of`) and shown in the UI. The rate is the same on every tab.
At the T19 live run (2026-11-26) FRED gave 4.91 % as of 2026-11-24.

## 3. Cost of capital (one module for every model)

Code: `valuation/capital.py` (ADR-007). The generic DCF, every sector model, APV, the stress tests and the
Sharpe-type ratios read their rates from this module, so one company has one discount rate.

| Item | Rule | Constant |
|---|---|---|
| Equity risk premium (ERP) | Constant, Damodaran's implied US ERP (4.42 % in the July 2026 update), rounded | 4.4 % (`EQUITY_RISK_PREMIUM`) |
| Raw beta | `yfinance` beta (`market_data.beta`, else `company_info.beta`) | - |
| Adjusted beta | Blume: `0.67 x raw + 0.33`, clamped to 0.3-3.0. 1.0 when the raw beta is unknown | `BETA_BOUNDS` |
| Cost of equity (Ke) | CAPM: `rf + beta x ERP`, clamped to 4.5-25 % | `COST_OF_EQUITY_BOUNDS` |
| Pre-tax cost of debt (Kd) | Interest expense / total debt, clamped to `[rf + 0.5 %, rf + 8 %]`. Without the inputs: `rf + 2 %` | `DEBT_SPREAD_BOUNDS`, `DEFAULT_DEBT_SPREAD` |
| After-tax Kd | `Kd x (1 - t)` | - |
| Tax rate (t) | Latest reported effective rate (`tax_rate` row; else tax provision / pretax income). 21 % when the rate is not between 0 and 50 %. Capped at 35 % | `DEFAULT_TAX_RATE`, `MAX_TAX_RATE` |
| Debt (D) | Borrowings from one balance-sheet period: long-term plus current borrowings; else Total Debt minus Capital Lease Obligations. Lease liabilities are excluded (see below) | `utils/financial_data.debt_excluding_leases` |
| Equity (E) | Market cap; else shares x price; else book equity; else no market value (sector default WACC) | - |
| WACC | `E/(D+E) x Ke + D/(D+E) x Kd x (1 - t)`, clamped to 6-20 %. The unclamped value is kept as `unbounded_wacc` | `WACC_BOUNDS` |

Rules that matter for reading the output:

- **Leases are not debt.** Operating-lease liabilities are inside Yahoo's Total Debt. Cash flows and
  EBITDA are already after rent, so counting the leases as debt as well would count them twice. The same
  debt figure is used in the WACC, the equity bridges, and the multiples' enterprise value.
- **Banks and insurers use common equity.** Book value is Common Stock Equity (preferred stock
  excluded) and net income is Net Income Common Stockholders, when reported. Without those rows the
  total equity and net income rows are used.
- **The WACC floor binds for some companies.** A highly levered company can sit on the 6 % WACC floor
  (for example CHTR in the ADR-007 table). The unclamped WACC is kept in the result as `unbounded_wacc`.
- **Terminal growth is capped at rf.** Stable growth cannot exceed the risk-free rate (a proxy for
  nominal economic growth, Damodaran). A capped value is noted in the result and in the DCF tab's warning.

Effect (ADR-007): moving the risk-free rate from 3 % to about 4.9 % raises the cost of equity for a beta-1
company from 8.5 % to about 9.3 %. The ADR-007 table gives AAPL's fair value as 193.91 at rf 3 % and 184.47 at
the current rate.

## 4. Discounted cash flow

### 4.1 Generic DCF engine (`valuation/base_valuation.py`, `dcf_models.py`)

- **Cash flow base: unlevered FCF (FCFF).** Reported free cash flow, plus after-tax interest
  (`|interest| x (1 - t)`), minus stock-based compensation (SBC) per period. If FCF had to fall back to
  net income (already after SBC), SBC is not deducted again. Stock-based compensation is deducted once.
- **Growth fade.** Growth falls linearly from the first forecast year to the terminal growth rate in the
  final year. Defaults: generic 2 %, sector
  defaults 1-3 % (`config.py`), all capped at rf.
- **Terminal value.** Gordon growth on the final cash flow: `CF(n) x (1 + g) / (WACC - g)`. The model
  requires WACC > g.
- **Discounting.** Year-end convention. Equity value = EV - debt + cash - minority interest - preferred
  stock. Value per share = equity / shares outstanding.
- **Variants** (`dcf_models.py`): multi-stage DCF (growth fades in stages), forecast-driven DCF (revenue,
  margin and reinvestment drivers: NOPAT, sales-to-capital), Monte Carlo DCF (distribution of the inputs),
  scenario DCF, and the sector DCF used by the tech and retail models.

### 4.2 DCF tab (`pages/company_analysis/valuation.py`, `page_helpers.py`)

The tab is an interactive DCF with a sensitivity grid. Its defaults:

- **Starts from unlevered FCF.** `UFCF = FCF + |interest| x (1 - t)` per period, using the same tax rate
  as the WACC. The chart and radio are labelled "unlevered". This replaced the earlier levered base, which
  counted interest twice (once in the cash flow, once through the net-debt deduction).
- **Base:** latest year, or a 3-year average (selectable).
- **SBC deducted** by default when reported. With the 3-year average base, the 3-year average SBC is
  deducted (same window).
- **Growth fades** to the terminal rate by default (switchable).
- **Terminal growth** above rf shows a warning; the slider value is kept.
- **Equity bridge:** EV minus net debt (see known bias K4: minority interest and preferred stock are not
  deducted on this tab).

### 4.3 Sector cash-flow variants

Sector models use the same engine with their own inputs; see section 5 for each.

## 5. Adjusted present value (`valuation/apv.py`)

- **Unlevered cost of equity** `ku = rf + beta_u x ERP`. `beta_u` is the Blume-adjusted beta unlevered with
  Hamada: `beta_u = beta_L / (1 + (1 - t) x D/E)`.
- **Unlevered cash flow** = levered FCF + after-tax interest (same as the DCF base).
- **Unlevered value** = PV of the unlevered cash flows at `ku`, with a Gordon terminal value.
- **Tax shields** = `t x D` each year, discounted at Kd. Debt is treated as fixed in amount (Myers 1974,
  consistent with Hamada).
- **Terminal shield:** a level perpetuity, `t x D / Kd`, discounted at Kd. Growing the shield at g while
  discounting at Kd overstated it (research finding W-D2, fixed in a6bde27): with Kd 6 % and g 3 % the old
  value was twice `t x D`. The total PV of shields is now `t x D` under the fixed-debt assumption.
- **Equity value** = APV - debt + cash. Distress costs are not modelled.

APV is not shown for banks and insurers (the tab says so).

## 6. Sector models and blending weights

Every sector model returns the same result: one value per method, a weighted fair value, upside versus the
price, and a recommendation (section 7). **Weights are normalised over the methods that actually ran**
(`base_valuation.summarize_valuations`). A method that cannot be computed (missing inputs) is excluded
and listed, so the table below need not sum to 1 for every company. Subsector tables override the default.

| Sector model (file) | Default weights | Subsector overrides (examples) |
|---|---|---|
| Generic (`base_valuation.DEFAULT_METHOD_WEIGHTS`) | DCF 0.5, relative 0.3, asset-based 0.1, dividend discount 0.1 | - |
| Technology (`tech_sector.py`) | DCF 0.50, revenue DCF 0.50, Rule of 40 0.20, EV/Revenue 0.10, EV/EBITDA 0.20, P/E 0.20 | SaaS: DCF 0.45, Rule of 40 0.25, EV/Revenue 0.10, EV/EBITDA 0.20. Semiconductors: DCF 0.45, EV/EBITDA 0.30, P/E 0.25 |
| Financials (`financial_sector.py`) | Excess return 0.35, justified P/B 0.25, P/E 0.20, dividend discount 0.20; FCF DCF and EV/EBITDA 0 | Bank, insurance and diversified use the default; asset managers and payments use FCF DCF and multiples |
| Energy, Utilities (`energy_sector.py`) | DCF 0.40, relative 0.25, dividend discount 0.15, reserve NAV 0.35, sum of the parts 0.30, contracted DCF 0.35, regulated asset base 0.35, project DCF 0.35 | Downstream: DCF 0.6, relative 0.4. Utilities: DCF 0.25, regulated asset base 0.4, dividend discount 0.2, relative 0.15 |
| Consumer (`retail_sector.py`) | DCF 0.5, relative 0.3, unit economics 0.2, sum of the parts 0.2, dividend discount 0.15 | E-commerce: DCF 0.6, relative 0.4, sum of the parts 0.2. Grocery and staples have their own tables |
| Industrials, Materials (`manufacturing.py`) | DCF 0.4, relative 0.3, earnings power 0.2, replacement value 0.1, dividend discount 0.1 | Metals and mining: DCF 0.3, relative 0.25, earnings power 0.2, replacement value 0.25. Automotive: DCF 0.3, relative 0.3, earnings power 0.2, replacement value 0.2 |
| Real Estate (`real_estate.py`) | NAV 0.35, P/FFO 0.30, DCF 0.20, dividend discount 0.15, relative 0.40 | Real estate services: DCF 0.6, relative 0.4 |
| Healthcare (`healthcare.py`) | R&D-adjusted DCF 0.5, relative 0.3, pipeline rNPV 0.3, dividend discount 0.1 | Biotechnology: R&D DCF 0.3, relative 0.2, rNPV 0.5 |
| Communication (`communication_sector.py`) | DCF 0.45, relative 0.3, subscriber 0.25, user-based 0.25, dividend discount 0.15 | Telecom: DCF 0.45, relative 0.25, subscriber 0.15, dividend discount 0.15 |

Method notes that change the number:

- **Financials.** Ke is the CAPM cost of equity (section 3). Excess return: `BV0 + sum (ROE - Ke) x BV(t-1) /
  (1 + Ke)^t` plus a terminal excess (opening book of year n). Justified P/B = `(ROE - g) / (Ke - g)`.
  Dividend discount: Gordon at Ke with stable payout `1 - g/ROE`; near-term g = ROE x retention. Fee
  businesses may use FCF DCF and EV/EBITDA. Distress scores do not apply.
- **Energy.** Reserve NAV: mid-cycle NOPAT rescaled from the reference price to the mid-cycle oil price
  (`oil_price` parameter, default $65/bbl) and paid as an annuity over the reserve life at WACC. Regulated utilities: rate base (net PP&E
  proxy) x equity ratio x `(allowed ROE - g) / (Ke - g)`.
- **Technology.** SBC is a real expense (deducted in the DCF). Rule of 40 uses the reported FCF margin
  before SBC (the industry convention).
- **Communication.** Subscriber and user-based values use a cash margin per customer:
  `EBIT x (1 - t) + D&A - capex` (3-year averages; capex defaults to D&A), not the EBITDA margin
  (finding S-D4, fixed in 5ba4387).
- **Real estate.** NAV = NOI / cap rate + cash - debt - preferred - minority, with EBITDA as the NOI proxy.
  FFO = net income to common - gains on sale + impairments + D&A (Nareit definition), with missing rows
  counted as zero and a note (fixed in 5ba4387; the real-estate-only D&A is not in Yahoo's data, so total
  D&A is used).
- **Healthcare.** R&D is capitalised (amortised over N years) for the adjusted EBIT. rNPV = sum of
  probability-weighted programme cash flows. Remaining development costs are not risk-weighted.
- **Manufacturing.** Earnings power value = normalised NOPAT / WACC; replacement value = the asset base at
  replacement cost. Cyclical margins are averaged over all years.
- **Retail.** Same-store growth on the mature base plus the NPV of new units (value / (1 + r) minus build cost).

## 7. Relative valuation, recommendation and confidence

### 7.1 Multiples used for the relative leg

- **Live sector medians** (preferred): median P/E, P/S, P/B and EV/EBITDA of the sector's universe companies
  whose company info is already cached (the app never downloads for this). Needs at least 5 companies per
  multiple. Cached info older than 30 days (`LIVE_MAX_AGE_DAYS`) is not used. The date of the oldest file
  used is shown as `as_of`.
- **Static medians** (fallback): long-run medians by sector in `valuation/base_valuation.SECTOR_MULTIPLES`
  (for example Technology P/E 25, EV/EBITDA 18; Financials P/E 12, P/B 1.3). The source is named in the
  UI. Defaults for an unknown sector: P/E 16, EV/EBITDA 11, EV/Revenue 2.
- **Irrelevant multiples are dropped** (`industry/benchmarks.irrelevant_multiples`). P/B is used only in
  Financials, Real Estate, Utilities, Energy and Materials. For Financials, P/S and EV/EBITDA are also
  dropped. The "Used" column shows what was applied; averages use only the multiples used.
- **Implausible values** outside 0.01x-20x of the share price are treated as failed inputs, not as valuations.

### 7.2 Recommendation thresholds (`base_valuation.RECOMMENDATION_THRESHOLDS`)

Upside = fair value / price - 1 (weighted fair value).

| Upside | Label |
|---|---|
| at least +25 % | Strong Buy |
| at least +10 % | Buy |
| at least -10 % | Hold |
| at least -25 % | Reduce |
| below -25 % | Sell |

Upside above +-100 % is reported as implausible and asks the user to check the inputs.

### 7.3 Confidence (`base_valuation.assess_confidence`)

Each condition adds penalty points. 0 points = High, 1 = Medium, 2 or more = Low.

- Methods spread more than 30 % of the fair value (1 point), or more than 60 % (2 points); spread counted only
  with two or more methods. One usable method: 1 point.
- A DCF method whose terminal value is above 80 % of its value: 1 point.
- Fewer than 60 % of the weighted methods usable (only methods that ran count): 1 point.
- Fewer than three annual statements: 1 point.
- Upside beyond +-100 %: 2 points.

With Low confidence the Buy and Sell labels are softened to "Below model value" and "Above model value";
Hold is unchanged. The reasons are listed next to the badge.

### 7.4 Precedent transactions (not weighted)

Valuation > Precedents applies the median and interquartile range of deal EV/Revenue and EV/EBITDA to the
company's revenue and EBITDA. The result is a **cross-check only: it is not part of the weighted fair value**
(finding R12, T16). Deal data is the user's own table (editor or CSV). A sample of five illustrative,
fictional deals can be loaded on request; the sample is labelled and not saved until you save it. Deal
multiples include a control premium, so they sit above trading multiples. The optional FMP feed returns the
latest five M&A announcements on the free plan.

### 7.5 Liquidation value (`valuation/liquidation.py`)

A floor for asset-heavy or distressed companies. It is never part of the weighted fair value.

- Gross recovery = sum over asset classes of book value x recovery rate.
- Liquidation cost = cost % x gross recovery.
- Equity = gross recovery - liquidation cost - **all** liabilities at 100 % of book; per share = max(equity, 0)
  / shares outstanding.
- Asset classes come from the latest balance sheet. "Other" is the residual of total assets, so nothing is
  counted twice.

| Asset class | Orderly | Forced |
|---|---|---|
| Cash & short-term investments | 100 % | 100 % |
| Receivables | 85 % | 70 % |
| Inventory | 60 % | 35 % |
| Long-term investments | 90 % | 75 % |
| PP&E (net) | 50 % | 25 % |
| Goodwill & intangibles | 0 % | 0 % |
| Other assets | 30 % | 10 % |
| Liquidation costs (% of gross recovery) | 5 % | 10 % |

Orderly = sale over 6-12 months to strategic buyers; forced = fire sale within weeks. Rates are editable on
the Valuation tab (Asset-Based). Not computed for banks and insurers.

## 8. Sector benchmarks and comparison

- Sector ratio benchmarks (`industry/benchmarks.SECTOR_BENCHMARKS`) are approximate large-cap sector medians,
  written into the app. They are not per-number sourced. An optional override file in the sector data folder
  is written only by `update_benchmarks`.
- Sector ETFs (`industry/universe.SECTOR_ETFS`) are the performance benchmark for a company and for the
  sector comparison.
- Sector notes (Sector Analysis tab) are static reference text, not data.
- The universe is about 130 large US companies (`industry/universe.py`). Screener and valuation-filter
  results are limited to it.

## 9. Ratios, risk and performance

### 9.1 Financial ratios (`models/ratio_analysis.py`, `models/financial_statements.py`)

- Current ratio = current assets / current liabilities. Quick ratio = (cash + short-term investments +
  receivables) / current liabilities (an approximation when some current-asset rows are missing).
- ROE = net income / average total equity. ROA = net income / average total assets.
- ROIC = NOPAT / average invested capital, NOPAT = EBIT x (1 - t) (fallback uses latest debt + equity).
- Interest coverage = EBIT / interest expense. Debt / equity = total debt / total equity (undefined when
  equity is negative).

### 9.2 Market risk (`pages/company_analysis/risk.py`, `models/performance_ratios.py`)

Daily simple returns, 252 trading days a year.

| Measure | Definition |
|---|---|
| Annual return | Compound annual growth rate over the price history |
| Volatility | Standard deviation of daily returns x sqrt(252) |
| Sharpe | (annual return - rf) / volatility. rf is the rate in section 2 |
| Max drawdown | Minimum of price / running maximum - 1 |
| VaR 95 % / 99 % | Historical 1-day: 5th / 1st percentile of daily returns |
| CVaR 95 % | Mean of the daily returns at or below VaR 95 % |
| Sortino | `mean(r - rf_d) x 252 / (DD x sqrt(252))`, with downside deviation `DD = sqrt(mean over ALL N days of min(r - rf_d, 0)^2)`, rf_d = rf / 252 (Sortino & Price 1994). Divides by N, not by the count of down days |
| Beta | OLS slope of daily excess returns on the S&P 500 (SPY) excess returns, same window |
| Treynor | (annual return - rf) / beta. N/A when beta <= 0 |
| Jensen's alpha | annual return - [rf + beta x (benchmark return - rf)] (Jensen 1968) |
| Information ratio | mean(r - benchmark) x 252 / (std(r - benchmark) x sqrt(252)). Tracking error is against SPY, not rf |
| Calmar | CAGR / abs(max drawdown) over the trailing 36 months (756 trading days), or the whole history when shorter (labelled). N/A without a drawdown |

Benchmark: S&P 500 (SPY), `MARKET_BENCHMARK` in `risk.py`. When SPY history is missing, beta, Treynor,
Jensen and IR show N/A rather than a guess. Each ratio has a one-line definition in the UI.

### 9.3 Scenario and stress tests

Revenue stress and the five macro shocks (rates +200 bp, recession, margin squeeze, de-rating,
stagflation) change EV/EBITDA and show interest cover. The 2008, 2020 and 2022 crises are shown as
beta-implied drawdowns and as actual drawdowns. The stress-test discount rate is `rf + ERP` for a beta-1
company (`capital.market_discount_rate`), not a fixed constant.

### 9.4 Distress models (`models/bankruptcy_models.py`)

Not applied to banks or insurers (the models were estimated on non-financial firms).

- **Altman Z (1968)**, public manufacturers: `1.2 A + 1.4 B + 3.3 C + 0.6 D + 1.0 E`, D = market equity /
  total liabilities. Safe above 2.99; distress below 1.81.
- **Altman Z' (private firms)**: `0.717 A + 0.847 B + 3.107 C + 0.420 D + 0.998 E`, D = book equity /
  total liabilities. Cut-offs 2.90 / 1.23.
- **Altman Z'' (non-manufacturers)**: `6.56 A + 3.26 B + 6.72 C + 1.05 D` (no sales term). Cut-offs 2.60 /
  1.10. The "emerging" label uses this score (the 3.25 emerging-market constant is not added; the zones are
  equivalent once shifted).
- Variant choice by sector: Z for manufacturers with market value, Z' when there is no market value, Z''
  for non-manufacturers and for software or internet names.
- **Springate (1978)**: `S = 1.03 WC/TA + 3.07 EBIT/TA + 0.66 EBT/CL + 0.40 Sales/TA`. S below 0.862 is failing.
- **Zmijewski (1984)**: `X = -4.336 - 4.513 NI/TA + 5.679 TL/TA + 0.004 CA/CL`. A probit model:
  `P = Phi(X)`, the standard normal CDF (fixed in 6c04793; it had used a logistic link). Classification at
  P = 0.5 (X = 0).

### 9.5 Liquidity grouping A1-A4 / P1-P4 (`models/balance_liquidity.py`)

Russian balance-sheet method (Sheremet & Negashev; Savitskaya), mapped onto the yfinance (US GAAP) balance
sheet, latest period.

| Group | Meaning | Mapping |
|---|---|---|
| A1 | Most liquid assets | Cash and short-term investments |
| A2 | Quickly realisable | Receivables |
| A3 | Slowly realisable | Rest of current assets (inventory, prepaid and other) |
| A4 | Hard to realise | Non-current assets |
| P1 | Most urgent liabilities | Payables |
| P2 | Short-term liabilities | Rest of current liabilities (short-term debt, deferred revenue, other) |
| P3 | Long-term liabilities | Total liabilities - current liabilities |
| P4 | Permanent liabilities | Total equity including minority interest |

A3 and P2 are residuals, so the groups add up to total assets and to liabilities plus equity. Verdict:
absolutely liquid when A1 >= P1, A2 >= P2, A3 >= P3 and A4 <= P4 (the last follows from the others on a
balanced sheet and is checked as a consistency test). Also shown: current liquidity (A1 + A2) - (P1 + P2)
and prospective liquidity A3 - P3.

### 9.6 Insider transactions (`utils/insider.py`)

- Requires `FINNHUB_API_KEY`. Without it no request is made and the section explains how to enable it.
  Provider errors show an empty table and never raise.
- Lists Form 4 transactions from the last 365 days (date, name, code, shares, price, value, holdings after).
- The 90-day summary counts only open-market trades: code **P** (purchase) and code **S** (sale). Value =
  shares x transaction price. Grants (A), option exercises (M, X), tax withholding (F) and gifts (G) are
  listed but not counted as buying or selling.
- Responses are cached like the news feed; the key is never part of a cache file name.

## 10. Known biases and open items

From `review/METHODOLOGY_RESEARCH.md` (dispositions table) and the QA/backlog notes. Fixed items are listed
for traceability.

Open:

- **K1 Terminal value and reinvestment (W-D4).** The forecast-driven DCF grows the last cash flow instead of
  tying reinvestment to growth (`g / RONIC`). Growth-based DCFs share this simplification. Low impact on the
  checked companies.
- **K2 Debt tax shield uses the effective tax rate (W-D5a).** The marginal rate (about 21-25 % US federal
  plus state) would be more correct for Kd (1 - t), Hamada and the APV shield. Effective rates are lower
  (AAPL about 16 % effective), so debt shields are understated slightly.
- **K3 Discounting is year-end.** Mid-year discounting would raise EV by roughly 4-5 %. The convention is
  consistent across models and conservative.
- **K4 DCF tab bridge omits minority interest and preferred stock (W-D6).** The engine subtracts them; the tab
  does not. Small for most of the universe.
- **K5 Telecom default assumptions.** VZ's DCF tab (about 235 per share against a price near 47 on the
  research date) is driven by the 6 % WACC floor and the telecom default growth. Backlog: review the telecom
  default growth and the WACC floor.
- **K6 Fixed equity risk premium.** 4.4 % is a constant. Revisit at each Damodaran update (January and July).
- **K7 Beta is the provider's regression beta,** not re-estimated; the Blume adjustment is applied on top.
- **K8 Debt at book value, equity at market value** in the WACC weights (standard practice, not exact).
- **K9 Sharpe numerator uses CAGR** (geometric), denominator arithmetic volatility. This biases Sharpe low by
  roughly sigma/2 at high volatility. Sortino, IR and Treynor use arithmetic means.
- **K10 Sector benchmarks and static multiples are approximate** and not per-number sourced; live medians are
  preferred when the universe cache is available.
- **K11 Liquidation value ignores non-controlling interest** when total liabilities exclude it; the value is
  a floor and not part of the fair value.
- **K12 APV has no distress costs** and does not deduct SBC on its tab. Shields use fixed debt (Myers).
- **K13 Banks: book value includes goodwill.** P/TBV is not offered; justified P/B uses total common book.
- **K14 Distress scores are not for financial companies,** and the Zmijewski and Altman zones are estimated on
  historical US samples.
- **K15 Peer and screening universe** is about 130 large US companies; results describe that set.

Fixed in the review-fix round (2026-11-26; see `CHANGELOG.md` and `review/METHODOLOGY_RESEARCH.md`):
DCF tab on unlevered FCF (D1); SBC window (W-D9); leases out of debt and one debt period (S-D1);
common equity and net income to common for banks (W-D8 / S-D3); generic FCFF deducts SBC once (W-D3); APV
terminal shield at fixed debt (W-D2); telecom cash margin (S-D4); REIT FFO per Nareit (S-D2); one tax rule
(W-D5b); terminal growth capped at rf (W-D7); Zmijewski probit (distress); Blume and CAPM rules unified (ADR-007).

## 11. Disclaimer

For research and education only. Nothing in this document, in the application or in its exports is
investment advice, a recommendation to buy or sell any security, or a forecast. Models depend on
assumptions and on data that can be incomplete, delayed or wrong. Check every number against the filings and
decide for yourself.
