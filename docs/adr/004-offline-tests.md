# ADR-004: Offline test suite with recorded fixtures

- Status: accepted
- Date: 2026-11-26 (recorded during adopt-project; decision predates it)
- Deciders: owner / earlier sessions

## Context and problem
Tests must run in CI and locally without keys or network, and must not flake on vendor data.

## Decision drivers
- deterministic
- no keys in CI

## Considered options
1. live API tests
2. recorded fixtures + synthetic statements + offline flag

## Decision
Chosen: Recorded AAPL/MSFT fixtures, synthetic statements (`tests/synthetic.py`), network blocked via `tests/offline.py`; live checks are manual and logged in STATE.

## Consequences
- Good: 366 tests in ~2 min, CI 3.11-3.13.
- Bad / debt: no golden textbook values for DCF/WACC yet (review R13); provider code paths only live-checked by hand.

## Confirmation
`pytest -o addopts="" -p no:warnings` green with no network.
