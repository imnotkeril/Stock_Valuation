# Handoff
status: active
date: 2026-11-26
from-chat-tokens: ~270k

## Goal
Finish the product-review fix plan (`docs/plan.md`) through the coding-brain framework, then push branch
`claude/review-fixes-finish-2bedc3`, open a PR into `master`, get CI green, and **merge it** (owner allowed merge,
relayed by the parent session: "мерж я тож разрешаю, я ваще все разрешаю тебе с этим проектом сча работать, пока я сплю";
merge commit or squash, never force-push master; record the merge in STATE). Short Russian report at the end.

## State
- Done + committed: T10, T14, T15, T16, T17, T24; R-02..R-07 fixed (b05fbbb, tests `test_review_round1_fixes.py`);
  plan Phase 3b (T22-T26) + SPEC F22 added.
- Running when this chat stopped (background agents of the old session; they commit / write files themselves):
  - quant-research agent building T22 (liquidation), T23 (Sortino/Treynor/Calmar/IR/Jensen), T25 (A1-A4/P1-P4),
    one commit each. Check `git log` and `docs/plan.md` ticks; finish whatever is not ticked yourself.
  - 3 methodology research agents writing `research-dcf-wacc.md`, `research-sectors.md`, `research-risk.md` into the
    notes folder `C:/Users/kykyl/AppData/Roaming/Claude/side-session-notes/local_dd7a30fa-49d1-4161-aa51-b96c1e32339f`.
    If a file is missing after the others land, redo that area (general-purpose agent with web search).
- Round-2 review (reviewer agent) done: `review-round2.md` in the same notes folder, R-08 (P1) .. R-13 (P2). Not fixed yet.
- Branch/commit: claude/review-fixes-finish-2bedc3 @ 92553a9 (+ whatever the quant agent committed), nothing pushed.

## Next
1. Wait until the quant agent's commits are in (T22, T23, T25 ticked) or finish them; T26 gate (`$T`, ruff, `$LIVE`).
2. Methodology research step (new, owner): merge the three research files into `docs/review/METHODOLOGY_RESEARCH.md`
   (claim -> source -> code -> verdict -> fix); add it to plan.md as a task before T18's tick; fix real discrepancies, each with a test.
3. T18: copy review-round2.md into `docs/REVIEW.md` as round 2; fix R-08 (filter relative leg: use
   `ratio_analyzer.get_sector_benchmarks(stock_sector)["valuation"]`; make `tests/test_parallel_fetch.py` require Relative rows),
   R-09..R-13; dispositions table; then `security-reviewer` agent final pass; tick T18.
4. T19 QA (browser AAPL/JPM/XOM, `qa-tester`), T20 docs (METHODOLOGY.md citing METHODOLOGY_RESEARCH.md, README, USER_GUIDE
   with screenshots, `tech-writer`), T21 release prep.
5. Push, PR into master, bind with ccd_pr tools, CI green, merge, STATE line, report to the parent session
   (`local_dd7a30fa-49d1-4161-aa51-b96c1e32339f`).

## Open questions
- none

## Traps
- `$PY` = `E:/Main/Dev/Python/Done/Stock Valuation/.venv/Scripts/python.exe`; `.env` is copied into this worktree (git-ignored, never print).
- No pytest-xdist: run the full suite in the background (~3 min, 478+ tests).
- Bash heredocs with nested quotes break: write patch scripts with Write into the scratchpad and run them.
- Helper `tick.py TID 'state line'` (ticks plan.md, adds a STATE line before `## Next`) lived in the old scratchpad; rewrite if needed.
- Tests: CACHE_DIR is tmp, rf = fallback 4.5 %; conftest patches DataLoader; company tabs must show no st.warning offline.
- Never `git add -A` while another agent works in the worktree.

## Verify
- `$PY -m pytest -o addopts="" -p no:warnings -q` (expect ~500 passed)
- `$PY -m ruff check StockAnalysisSystem`
