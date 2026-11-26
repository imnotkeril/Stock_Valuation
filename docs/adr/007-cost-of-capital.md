# ADR-007: One cost-of-capital module with a live risk-free rate and a re-baselined ERP

- Status: accepted
- Date: 2026-11-26 (plan task T02; review findings R1, R2)
- Deciders: build session (finance/screener router), owner to confirm in review

## Context and problem
The product review (`docs/review/PRODUCT_REVIEW.md` §5, R1-R2) found two CAPM/WACC implementations with
different rules, both on a fixed 3 % risk-free rate while the 10-year Treasury yield is ~4.9 %:

| Rule | Generic DCF (`base_valuation`) | Sector models (`sector_specific/common`) |
|---|---|---|
| Beta | raw, clamped 0.3-3 | Blume 0.67 x raw + 0.33 |
| Cost of debt | [rf + 0.5 %, 15 %] | [rf + 1 %, rf + 8 %] |
| WACC band | 5-20 % | 6-20 % |
| Tax rate | 3-year average | aggregated history |
| ERP | 5.5 % (also in `apv.py`) | 5.5 % |

So the same company got different discount rates on different tabs, and the stress test used a fixed 9 %.
Moving rf alone from 3 % to 4.9 % with an ERP of 5.5 % would raise every cost of equity by ~1.9 points:
that ERP was calibrated against low rates, so together they would overstate the expected market return.

## Decision drivers
- one company, one discount rate on every tab
- rf matches the market today, with its source and date visible
- the expected market return (rf + ERP) stays close to published implied estimates
- no new dependency, works offline

## Considered options
1. Keep both implementations, only swap the rf constant.
2. One module, live rf, keep ERP 5.5 %.
3. One module, live rf, ERP re-baselined to an implied-ERP estimate.

## Decision
Chosen: option 3, `StockAnalysisSystem/src/valuation/capital.py`.

- rf: `utils/rates.risk_free_rate()` (user override -> FRED DGS10 cached a day + stale copy -> 4.5 % constant).
- ERP: **4.4 %**, Damodaran's implied US equity risk premium of 4.42 % (July 2026 update,
  https://pages.stern.nyu.edu/~adamodar/), rounded. His 1 January 2026 S&P 500 expected return was 8.41 %.
  With rf 4.91 % a beta-1 cost of equity is 9.3 %, against 8.5 % before (3 % + 5.5 %).
- Beta: Blume-adjusted, clamped 0.3-3; 1.0 when unknown.
- Cost of debt: interest / debt in [rf + 0.5 %, rf + 8 %]; rf + 2 % without data.
- WACC band 6-20 %. Equity: market cap, then shares x price, then book equity. With no equity, the sector default rate is used.
- Tax rate: the sector-model rule (latest reported rate, else aggregated tax / pretax income, 0-35 %).
- Users of the module: generic DCF (`BaseValuation._wacc_components`), every sector model
  (`ValuationContext.cost_of_capital`), APV (`apv.unlevered_cost_of_equity`, Blume beta, then Hamada),
  stress tests (`market_discount_rate()` = rf + ERP replaces the fixed 9 %), Sharpe ratios
  (Risk tab, screener, sector compare).
- Every valuation result carries `risk_free_rate`, `risk_free_rate_source`, `risk_free_rate_as_of` and
  `equity_risk_premium` in `assumptions`.

## Consequences
- Good: one set of rules, provable by test; rf has a visible source and date; the ERP is tied to a cited estimate.
- Bad: cost of equity rises by about 0.8 points at today's rf, so fair values drop.

  | Ticker | Fair value, rf 3 % | Fair value, now | Upside, now | Recommendation |
  |---|---|---|---|---|
  | AAPL | 193.91 | 184.47 | -48.1 % | Sell |
  | JPM | 310.39 | 278.54 | -12.1 % | Reduce |
  | XOM | 95.47 | 86.30 | -45.4 % | Sell |

  XOM's WACC is now 6.59 %, off the floor. PGR falls from +195 % to +54.5 %. The remaining undervaluation
  comes from static sector multiples and DCF growth without fade; R8, R9 and the sanity layer (R4) address those.
- Bad: the ERP is a constant to revisit yearly (Damodaran publishes in January and July).
- Neutral: a highly levered company can still sit on the 6 % floor (CHTR); T05 flags such rows.

## Confirmation
- `tests/test_capital.py`: hand-computed WACC, the bands, rf metadata, and generic WACC equal to sector WACC
  for AAPL, MSFT and every synthetic profile.
- `tools/live_valuation_check.py AAPL JPM XOM MSFT PGR CHTR`: before and after values are in `docs/STATE.md`.
