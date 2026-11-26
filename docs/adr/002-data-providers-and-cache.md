# ADR-002: Yahoo Finance primary, keyed providers optional, file cache with stale fallback

- Status: accepted
- Date: 2026-11-26 (recorded during adopt-project; decision predates it)
- Deciders: owner / earlier sessions

## Context and problem
Fundamentals and prices are needed for ~132 US large caps without paid data. Free providers fail or rate-limit unpredictably.

## Decision drivers
- zero cost
- works without any key
- survives provider outages

## Considered options
1. paid vendor (Polygon/Tiingo)
2. Yahoo via yfinance + optional Alpha Vantage/Finnhub/FMP/SEC/FRED
3. SEC XBRL only

## Decision
Chosen: yfinance as primary source, every other provider optional and degrading to an explanatory message; on-disk cache (fresh -> fetch -> stale) and an offline flag.

## Consequences
- Good: runs with no keys; outages show last good data.
- Bad / debt: Yahoo is unofficial and for personal/research use only; 4-5 annual periods; sequential fetches are slow (screener ~3 min cold).

## Confirmation
`STOCK_ANALYSIS_OFFLINE=1` run of the test suite passes; README keys table lists each optional provider.
