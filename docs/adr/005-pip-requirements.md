# ADR-005: pip + requirements.txt instead of uv

- Status: accepted
- Date: 2026-11-26 (recorded during adopt-project; decision predates it)
- Deciders: owner / earlier sessions

## Context and problem
The finance family defaults to uv with a committed lock file. The project predates adoption and uses pip with minimum versions; CI and Docker install from `requirements.txt`.

## Decision drivers
- no rewrite during adoption
- Docker and CI already work

## Considered options
1. migrate to uv + uv.lock
2. keep pip + requirements minimums

## Decision
Chosen: Keep pip for now (adopt-project forbids dependency changes); uv migration stays in the backlog.

## Consequences
- Good: nothing breaks.
- Bad / debt: installs are not reproducible to the patch level.

## Confirmation
CI installs from requirements and passes on 3.11-3.13.
