# Handoff
status: active
date: 2026-11-26
from-chat-tokens: ~330k

## Goal
Fix the product-review findings (`docs/review/PRODUCT_REVIEW.md` section 5) through the coding-brain framework:
P1 (T01-T08), review step with reviewer + security-reviewer (T09-T10), P2 (T11-T17), hardening (T18-T21). Plan: `docs/plan.md`.

## State
- Done: adoption, plan + audit, P1 T01-T08 (live rf, capital.py + ADR-007, DCF/APV defaults, widget reset, Valuation Filter sanity, bank risk/peer fixes, confidence badge, P1 gate 446 passed + browser check).
- Running: T09 review: `reviewer` and `security-reviewer` agents write `docs/REVIEW.md` (sections "Reviewer (correctness) - P1", "Security reviewer - P1").
- Branch/commit: claude/review-fixes-195b14 @ 7efbd38, nothing pushed.

## Decisions
- Owner (2026-11-26, away): "делай до конца без моих подтверждений" - continue through all tasks without asking; still never push/PR/merge.
- ERP 4.4 % = Damodaran July 2026 implied US ERP 4.42 % (ADR-007); one Blume beta, Kd rf+0.5..8 %, WACC 6-20 %.
- Per-company widget prefixes `val_`, `risk_` (main.COMPANY_WIDGET_PREFIXES).

## Next
1. Read `docs/REVIEW.md`; if a section is missing, rerun that agent on `git diff b6b2045..HEAD`. Add a disposition column; fix every valid P0/P1 (one commit each), re-run `$T` + ruff, tick T09, STATE line.
2. `step-brief --task T10` (gate), then P2 T11-T17 (golden tests, live medians, DCF fade/SBC, UX U6-U9, parallel fetch, precedents).
3. Hardening T18-T21; final: propose a PR to the owner (no push).

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
