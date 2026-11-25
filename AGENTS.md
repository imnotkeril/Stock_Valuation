# AGENTS.md — Stock Analysis System

Instructions for any coding agent (Cursor, Codex, Claude Code).

- Goal: local single-user Streamlit app for sector-specific valuation, screening and risk analysis of US large caps.
- Stack: Python 3.11+, Streamlit, pandas/numpy, scikit-learn, statsmodels, plotly, yfinance; pytest + ruff.
- Read first: `docs/STATE.md` (progress), `docs/SPEC.md`, `docs/ARCHITECTURE.md`, `docs/plan.md`.
- Shared conventions: `E:/Main/Dev/coding-brain/rules/` (only the files relevant to your task) and `E:/Main/Dev/coding-brain/known-traps.md`.
- Work one task from `docs/plan.md` at a time; verify (offline tests + ruff; live check for valuation changes), tick it, add a line to `docs/STATE.md`, commit locally.
- Tests never call the network: use fixtures in `StockAnalysisSystem/tests/fixtures` and `STOCK_ANALYSIS_OFFLINE=1`.
- Never commit secrets (`.env*`). Never push or deploy without the owner's approval.
- Code, comments and docs in English. Not investment advice: keep the disclaimer.
