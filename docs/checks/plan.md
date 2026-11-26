# Audit: plan (docs/plan.md)
mode: shadow
author-model: unknown · auditor-model: sonnet

## Verdict
FIX
One P1: the T20 check command points to a script that does not exist; plan.7 lists a router skill used by no task.

## Checklist
| # | Item | Result | Evidence |
|---|---|---|---|
| plan.1 | no leftover markers | PASS | 0 markers [script] |
| plan.2 | every task has all fields, check is a command | PASS | 21 tasks [script] |
| plan.3 | depends ids exist, acyclic, T01 no deps | PASS | docs/plan.md:16 "depends: -" [script] |
| plan.4 | [P gN] groups have disjoint owns | PASS | 0 groups checked [script] |
| plan.5 | chosen modules in uses | PASS | no modules chosen [script] |
| plan.6 | blueprint in T01 uses | PASS | docs/plan.md:17 "blueprint:finance-screener" [script] |
| plan.7 | every router phase-skill of build in some uses | FAIL | router build row lists skill:sql-queries; no task uses it (docs/plan.md:17 has only skill:test-driven-development; T14 only skill:frontend-design) |
| plan.8 | each phase ends with a gate | PASS | 4 phases [script] |
| plan.9 | every P1 AC in plan | PASS | 6 ACs mapped [script] |
| plan.10 | Hardening phase exists, last, ordered | PASS | docs/plan.md:131 "## Phase 4 — Hardening" [script] |
| plan.11 | vertical slices, tasks small | PASS | slices are vertical per review item; T02 is large (see A-02, P2) |
| plan.12 | tier and model match risk | PASS | docs/plan.md:22 T02 opus for the shared WACC; no money/auth/migration tasks |

## Findings
- A-01 | P1 | source: step:plan | at: docs/plan.md:151 | quote: "check: `$PY StockAnalysisSystem/tools/docs_check.py`" | scenario: tools/ holds only live_valuation_check.py and no task owns or creates docs_check.py, so T20 check fails with file-not-found and the docs task can never be ticked done | fix: add a task or owns entry that creates docs_check.py (links, screenshots, disclaimer), or use an existing command
- A-02 | P1 | source: checklist:7 | at: docs/plan.md:17 | quote: "uses: blueprint:finance-screener, stack:python, stack:pandas, mcp:context7, skill:test-driven-development" | scenario: the router build row lists skill:sql-queries and no task has it in uses, so the plan lint row fails as written; the app has no SQL layer, so the real fix is an explicit n/a | fix: add skill:sql-queries to a task uses, or record a waiver line in the plan header for sql-queries
- A-03 | P2 | source: checklist:11 | at: docs/plan.md:22 | quote: "owns: StockAnalysisSystem/src/valuation/capital.py, StockAnalysisSystem/src/valuation/base_valuation.py" | scenario: T02 owns 12 paths (new module, 5 consumers, 4 test files, ADR) and is one opus task; a rate-limit cut mid-task leaves half the consumers on the old WACC | fix: split into capital module + ADR and a consumer migration task
- A-04 | P2 | source: step:plan | at: docs/plan.md:69 | quote: "Review with `reviewer` + `security-reviewer`; fix valid findings" | scenario: owner is only reviewer and uses lists no security-reviewer, so the new FRED network fetch and env override may get no security pass | fix: name security-reviewer in owner or done

## Notes (non-blocking)
- N-01 | opinion | at: docs/plan.md:25 | riskiest assumption: re-baselining ERP/rf in T02 will not flip many recommendations; baseline numbers are only checked by eye through $LIVE, which needs the network.
- N-02 | opinion | at: docs/plan.md:60 | several checks end with $LIVE (network); a provider outage makes a done task fail non-deterministically.

## Disposition
| id | verdict | by | note |
|---|---|---|---|

Read: E:/Main/Dev/coding-brain/rules/audit.md, E:/Main/Dev/coding-brain/process/plan.md, docs/plan.md, docs/SPEC.md (F20, F21), doc-check and step-brief output, docs/checks/plan.md
| A-01 | valid | orchestrator | fixed: T20 owns now creates StockAnalysisSystem/tools/docs_check.py |
| A-02 | valid | orchestrator | fixed: header line records skill:sql-queries n/a (no SQL layer) |
| A-03 | valid | orchestrator | not split (P2, shadow): consumers must switch together; T02 commits per consumer, noted in header |
| A-04 | valid | orchestrator | fixed: T09 uses agent:security-reviewer as owner of the security pass |
