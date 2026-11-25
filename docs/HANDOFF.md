# Handoff
status: active
date: 2026-11-26
from-chat-tokens: ~205k

## Goal
Fix the product-review findings (`docs/review/PRODUCT_REVIEW.md` section 5) through the coding-brain framework:
P1 (T01-T08), review step with reviewer + security-reviewer (T09-T10), P2 (T11-T17), hardening (T18-T21). Plan: `docs/plan.md`.

## State
- Done: project adopted (router finance/screener, owner confirmed; CLAUDE.md, AGENTS.md, ARCHITECTURE.md, ADR-001..006).
- Done: plan written, linted (doc-check exit 0), audited (shadow, 4 valid findings handled, `docs/checks/plan.md`).
- Done: T01 `utils/rates.py` live risk-free rate (FRED DGS10, live 4.91 %); config keeps legacy `RISK_FREE_RATE = 0.03` until T02.
- Running: nothing.
- Branch/commit: claude/review-fixes-195b14 @ afec7ae, tree clean, nothing pushed.

## Decisions
- Fallback rf constant 4.5 % (`config.RISK_FREE_RATE_FALLBACK`); override env `STOCK_ANALYSIS_RISK_FREE_RATE`.
- Auditor model: fable has no credits -> sonnet (rules/audit.md table).
- Session was started by another session (local_0404ad89...); report back there; never push/PR/merge without the owner's yes.

## Next
1. Run `node E:/Main/Dev/coding-brain/scripts/step-brief.mjs --project . --task T02`, then build T02: new `valuation/capital.py` with one `cost_of_capital()` (live rf from `utils.rates`, one ERP, Blume beta, one Kd band, one WACC band); switch `base_valuation._wacc_components`, `sector_specific/common.cost_of_capital`, `apv.py`, `stress_testing._BASE_DISCOUNT`, Sharpe in `risk.py` / `stock_screener.py`; write ADR-007 (ERP re-baseline: rf ~4.9 % with ERP 5.5 % overstates Ke; consider implied ERP ~4.3-4.5 %); test generic == sector WACC.
2. Live check before/after: `$PY StockAnalysisSystem/tools/live_valuation_check.py AAPL JPM XOM MSFT PGR CHTR` (baseline in STATE), tick plan, STATE line, commit.
3. Continue T03-T08 in order, each with its step-brief, then T09 review with `reviewer` + `security-reviewer` agents.

## Open questions
- none

## Traps
- `.venv` is only in the main checkout: `$PY` = `E:/Main/Dev/Python/Done/Stock Valuation/.venv/Scripts/python.exe`; `.env` was copied into this worktree (git-ignored, never print).
- Bash heredocs with nested quotes break through the `rtk` wrapper: write scripts to the scratchpad and run them.
- Tests patch DataLoader public methods (`tests/offline.py`); new loader methods run for real with `offline=True`, so rf in tests = fallback 4.5 %.
- XOM/CHTR WACC hits the 6 % floor in sector models; PGR excess-return value 1071 (insurer on the bank model) drives +195 %.

## Verify
- `$PY -m pytest -o addopts="" -p no:warnings -q` (expect 376 passed)
- `$PY -m ruff check StockAnalysisSystem`
