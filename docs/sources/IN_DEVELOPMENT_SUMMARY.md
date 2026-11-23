# What was done in the "In Development" copy (branch `chore/audit`)

Source folder: `E:\Main\Dev\Python\In Development\Stock_Valuation`
Branch: `chore/audit` (11 commits, 2026-10-24 … 2026-11-02). On GitHub up to `5a5f5b6`;
the last commit `32259f4` (handoff note) exists only locally.
Forked from the old commit `0844670` (2025-04-10), **not** from `master`.
This copy (`claude/peaceful-euler-aq2j08`) was forked from `master` (`f6e0541`), so neither
branch contains the other's work.

The original documents are in `in-development/` next to this file, unchanged:

| File | What it is |
|---|---|
| `AUDIT_2026-10-24.md` | Readiness audit (Russian): ~30% ready, security, broken UI and DCF, findings ranked by severity |
| `HANDOFF.md` | Handoff note 2026-11-02: state, decisions, next steps, traps |
| `STATE.md` | Short state log and "needed from owner" list |
| `ORIGINAL_README.md` | The original README: the first feature wish list of the project (it served as the spec) |
| `LEGACY_UI_README.md` | Note on the old broken UI moved to `legacy_ui/` |

There was no separate spec or plan document: the original README's feature list is the
de-facto spec, and the audit plus handoff are the plan.

## Commits on `chore/audit`

| Commit | Change |
|---|---|
| `ffdd793` | `.gitignore` (venv, pycache, `.env*`, caches, logs) |
| `15c0b21` | `.env.example` with placeholder key names |
| `d956e43` | Removed broken self-import in `industry/sector_ratios.py` |
| `850571f` | README: real clone URL and status note |
| `6cb5a8d` | Readiness audit document |
| `b66f8a5` | Snapshot commit of the v2 library layout; API keys read from env (`load_dotenv`), legacy UI pages restored from HEAD |
| `5ddc029` | Implemented missing DCF methods (`_estimate_initial_growth_rate`, `_forecast_financial_statements`), transition-phase fix |
| `464d7be` | `PeerComparison` methods wrongly nested inside another method, dedented |
| `6dc6ba8` | 25 offline unit tests (DCF, ratios, bankruptcy, loader, sectors); sector analyzer constructor/registration fixes |
| `5a5f5b6` | New small Streamlit UI (never run live); old 250 KB UI moved to `legacy_ui/` |
| `32259f4` | Handoff note (local only) |

## Features and fixes vs. this copy (checked 2026-11-23)

| Item from In Development | State in this copy | Action |
|---|---|---|
| `.gitignore` | Present, broader | none |
| `.env.example`, keys from env + `load_dotenv`, `*_API_KEY` names | Present (`src/config.py`) | none |
| `sector_ratios` self-import fix | Present (no self-import) | none |
| DCF missing methods | Rewritten: `_forecast_financial_statements` exists, multi-stage / Monte Carlo / scenario DCF tested | none |
| `PeerComparison` nesting fix | Page rewritten and covered by AppTest tests | none |
| `run.py` crash without `logs/` | `run.py` rewritten as a thin launcher, no file logging | none |
| 25 offline tests | Superseded by 280 offline tests | none |
| New small Streamlit UI | Superseded by the full UI (8 pages, company tabs) | none |
| `legacy_ui/` (old broken UI) | Not needed | dropped |
| `create_project.py` (scaffold generator) | Obsolete | dropped |
| Root `requirements.txt` freeze with CVEs | This copy uses a minimal list in `StockAnalysisSystem/requirements.txt` | pip-audit still to run (plan) |
| Hard-coded Alpha Vantage and FMP keys | Removed from code, **still in git history** of `master` and both branches | owner must rotate (plan) |
| Audit §2 "README claims features that do not exist" | Rewritten README is honest; unimplemented ideas listed in `docs/SPEC.md` §5 | decide scope (plan) |

Conclusion: nothing from `chore/audit` needs to be ported in code. Its value is the audit
findings and the owner tasks, carried into `docs/SPEC.md` and `docs/plan.md`.
