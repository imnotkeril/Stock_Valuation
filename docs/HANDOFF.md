# Handoff
status: active
date: 2026-11-27
from-chat-tokens: ~170k

## Goal
Independent full-system re-test of master 747b6a2: every bug gets root cause + fix + regression test,
then push branch `claude/system-retest-722b8c`, PR into master, CI green, merge (owner allowed push, PR and merge).

## State
- Done: offline suite 533 passed, ruff clean; live providers all OK (FRED, SEC EDGAR, Finnhub news/insider,
  FMP M&A, Alpha Vantage quote, yfinance); live valuation AAPL/JPM/XOM/O/NEE recorded in docs/QA.md.
- Fixed RT-01 (commit 345dc9e): subsector terminal-growth default was dead; override sliders now start at the
  values the model used and only moved sliders override. Tests: tests/test_retest_fixes.py (5).
- Running: none in the new chat (a qa-tester AppTest live pass was started in the old chat; its scripts and
  outputs, if any, are in the old session scratchpad `qa/` folder, not in the repo).
- Not done: browser pass of every page for the 5 tickers, reviewer agent on the diff, push/PR/merge.

## Decisions
- WACC stays the derived discount rate even when a subsector default exists (ADR-007); only terminal_growth and
  forecast_years take subsector defaults over the static sector table.
- NEE fair value ~$40 vs price $85 is a model limitation (RAB on net PP&E, DDM), not a code bug; note only.

## Next
1. Live AppTest pass (or browser, `.claude/launch.json` "stock-analysis", port 8502): Company Analysis all tabs for
   AAPL, JPM, XOM, O, NEE; Valuation Filter, Screener, Peer/Sector Comparison, Forecasting, Reports. Collect
   exceptions, st.error, N/A where data exists, rates differing between tabs.
2. Each confirmed bug: test first, fix, `docs/QA.md` line under "Full-system re-test", STATE line, local commit.
3. `reviewer` agent on `git diff master...HEAD`; full suite; push; `gh pr create` into master; bind with
   ccd_pr `bind_pr`, monitor; when CI green `gh pr merge --merge`; Russian report to parent session.

## Open questions
- none

## Traps
- `$PY` = `E:/Main/Dev/Python/Done/Stock Valuation/.venv/Scripts/python.exe`; pytest needs `-o addopts=""`.
- `.env` in the worktree is a git-ignored copy with real keys: never print or commit it.
- Do not touch README, repo root layout, `.github/` community files (repo-polish session owns them).
- Untracked `co_AAPL.json` at repo root came from a QA script: do not commit it.
- Live diagnostic: `$PY StockAnalysisSystem/tools/live_valuation_check.py AAPL JPM XOM O NEE`.

## Verify
- `$PY -m pytest -o addopts="" -p no:warnings -q` (expect 540 passed)
- `$PY -m ruff check StockAnalysisSystem`
