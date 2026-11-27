# Handoff
status: active
date: 2026-11-27
from-chat-tokens: ~170k

## Goal
Independent full-system re-test of master 747b6a2: every bug gets root cause + fix + regression test,
then push branch `claude/system-retest-722b8c`, PR into master, CI green, merge (owner allowed push, PR and merge).

## State
- Done: RT-01 (345dc9e), RT-02 + RT-03 (cbfbf0d), review follow-ups; full suite 543 passed, ruff clean; live
  AppTest pass of every page for AAPL/JPM/XOM/O/NEE: 0 exceptions/errors/warnings (docs/QA.md "Full-system re-test").
- PR https://github.com/imnotkeril/Stock_Valuation/pull/3 open, bound in the app, Auto-fix on, CI running.
- Auto-merge was denied by the session permission classifier.

## Decisions
- WACC stays derived; only terminal_growth / forecast_years take subsector defaults over the sector table.
- NEE (~$40 vs $85) and O DCF-tab growth (19 %) are methodology backlog for the owner, not code bugs.

## Next
1. `mcp__ccd_pr__get_status`: when CI is green on 3.11-3.13, merge PR #3 with `gh pr merge 3 --merge` if the
   owner confirms (or permission allows); never force, never admin bypass.
2. If CI fails: `gh run view --log-failed`, fix, run the suite, commit, push.
3. After merge: STATE line with the merge sha, set this note to `status: consumed`, short Russian report to the
   parent session `local_dd7a30fa-49d1-4161-aa51-b96c1e32339f`.

## Open questions
- none

## Traps
- `$PY` = `E:/Main/Dev/Python/Done/Stock Valuation/.venv/Scripts/python.exe`; pytest needs `-o addopts=""`.
- `.env` in the worktree is a git-ignored copy with real keys: never print or commit it.
- Do not touch README, repo root layout, `.github/` community files (repo-polish session owns them).
- Untracked `co_AAPL.json` at repo root came from a QA script: do not commit it.
- Live diagnostic: `$PY StockAnalysisSystem/tools/live_valuation_check.py AAPL JPM XOM O NEE`.

## Verify
- `$PY -m pytest -o addopts="" -p no:warnings -q` (expect 543 passed)
- `$PY -m ruff check StockAnalysisSystem`
