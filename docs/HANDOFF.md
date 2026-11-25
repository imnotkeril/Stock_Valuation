# Handoff
status: active
date: 2026-11-26
from-chat-tokens: ~265k

## Goal
Fix the product-review findings (`docs/review/PRODUCT_REVIEW.md` section 5) through the coding-brain framework:
P1 (T01-T08), review step with reviewer + security-reviewer (T09-T10), P2 (T11-T17), hardening (T18-T21). Plan: `docs/plan.md`.

## State
- Done: adoption, plan + audit, T01 (live rf), T02 (`valuation/capital.py`, ERP 4.4 %, ADR-007), T03 (DCF/APV tabs use the shared WACC), T04 (widget state reset on ticker change).
- Running: T05 (Valuation Filter sanity layer).
- Branch/commit: claude/review-fixes-195b14 @ d1ed114, nothing pushed.

## Decisions
- Owner (2026-11-26, away): "делай до конца без моих подтверждений" - continue through all tasks without asking; still never push/PR/merge.
- ERP 4.4 % = Damodaran July 2026 implied US ERP 4.42 % (ADR-007); one Blume beta, Kd rf+0.5..8 %, WACC 6-20 %.
- Per-company widget prefixes `val_`, `risk_` (main.COMPANY_WIDGET_PREFIXES).

## Next
1. Run `node E:/Main/Dev/coding-brain/scripts/step-brief.mjs --project . --task T05`, then build T05 (Valuation Filter: flag |upside| > 100 %, real method instead of `DCF`, spread across methods, default approach = sector model; PRODUCT_REVIEW U2).
2. T06 (bank Risk tab: no FCF penalty, single Altman text, Peer Comparison N/A / 0.00 %), T07 (confidence on the badge), T08 P1 gate.
3. T09 review with `reviewer` + `security-reviewer` agents, then T10-T21; final: propose a PR to the owner (no push).

## Open questions
- none

## Traps
- `$PY` = `E:/Main/Dev/Python/Done/Stock Valuation/.venv/Scripts/python.exe`; `.env` copied into this worktree (git-ignored, never print).
- Bash heredocs with nested quotes break through the `rtk` wrapper: write scripts to the scratchpad with Write and run them.
- Tests: CACHE_DIR is a tmp dir, so rf in tests = fallback 4.5 % ("fallback constant").
- CHTR WACC sits on the 6 % floor (heavy leverage); T05 should flag it.

## Verify
- `$PY -m pytest -o addopts="" -p no:warnings -q` (expect ~407 passed)
- `$PY -m ruff check StockAnalysisSystem`
