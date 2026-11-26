# ADR-003: Unified valuation contract and model-declared parameters

- Status: accepted
- Date: 2026-11-26 (recorded during adopt-project; decision predates it)
- Deciders: owner / earlier sessions

## Context and problem
Eight sector models plus a generic model must render in the same Valuation tab, Valuation Filter, Peer Comparison and reports.

## Decision drivers
- one UI for all sectors
- comparable outputs
- cheap to add a sector

## Considered options
1. per-sector UI code
2. one result schema (`build_valuation_result`) + `PARAMETERS`/`METHODS`/weights declared on the class

## Decision
Chosen: One schema returned by every model; UI controls are generated from the model's `PARAMETERS`; method blend weights declared per class/subsector.

## Consequences
- Good: new sector costs little UI code; tests check the schema for all models.
- Bad / debt: generic and sector models still compute cost of capital differently (to be fixed by a shared module).

## Confirmation
`tests/test_sector_models.py` checks every model returns the contract.
