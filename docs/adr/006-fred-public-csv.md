# ADR-006: FRED macro series via the public fredgraph CSV

- Status: accepted
- Date: 2026-11-26 (recorded during adopt-project; decision predates it)
- Deciders: owner / earlier sessions

## Context and problem
Macro indicators (CPI, unemployment, DGS10) were read with pandas-datareader, which is unmaintained and broke with pandas 3.

## Decision drivers
- no extra dependency
- no API key

## Considered options
1. pandas-datareader
2. FRED API with key
3. public `fredgraph.csv?id=` endpoint

## Decision
Chosen: Public CSV endpoint read with requests + pandas, cached by DataLoader.

## Consequences
- Good: no key, no dependency.
- Bad: endpoint is undocumented as an API; failure falls back to cache.

## Confirmation
`tests/test_data_loader.py` FRED tests; live CPI/UNRATE/DGS10 check 2026-11-26.
