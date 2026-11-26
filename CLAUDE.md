# Stock Analysis System

Local single-user Streamlit app for sector-specific valuation, screening and risk analysis of US large caps.

- Router: finance/screener
- Modules: legal-compliance
- Stack: Python 3.11+ (pip + `StockAnalysisSystem/requirements.txt`, `.venv`), Streamlit, pandas/numpy/scipy, scikit-learn, statsmodels, plotly, yfinance, fpdf2/openpyxl; pytest + ruff; GitHub Actions CI (3.11-3.13); Dockerfile
- Services: Yahoo Finance (primary, no key), FRED public CSV (no key), SEC EDGAR (User-Agent), optional keys: Finnhub, Alpha Vantage, Financial Modeling Prep (`.env.example`)
- Decisions: `docs/adr/` · Spec: `docs/SPEC.md` · Architecture: `docs/ARCHITECTURE.md` · Plan: `docs/plan.md` · Progress: `docs/STATE.md` · Review: `docs/review/PRODUCT_REVIEW.md`

(`Router:` and `Modules:` lines are read by `E:/Main/Dev/coding-brain/scripts/step-brief.mjs`; keep them in this exact form. Update `Modules:` whenever the spec adds or drops a module.)

## Commands
- Run: `.venv\Scripts\python.exe StockAnalysisSystem\run.py` -> http://127.0.0.1:8501 (health `/_stcore/health`)
- Tests (offline): `.venv\Scripts\python.exe -m pytest -o addopts="" -p no:warnings`
- Lint: `.venv\Scripts\python.exe -m ruff check StockAnalysisSystem`
- Live scripts import `StockAnalysisSystem.src...` with the repo root on `sys.path`.

## Working rules
- Progress lives in `docs/STATE.md`. On session start, resume from its last unchecked step.
- Before each process step and before each plan task, generate its brief and follow it:
  `node E:/Main/Dev/coding-brain/scripts/step-brief.mjs --project . --step <step>` / `--task <id>`.
- After each task: run its check -> tick in `docs/plan.md` -> one line in `docs/STATE.md` -> local commit.
- Valuation changes need a live check (AAPL, JPM, XOM at least) besides offline tests.
- `.env` holds real keys: never print or commit values.
- No push / deploy / PR / merge without the user's explicit yes.
- Research and educational tool, not investment advice: keep the disclaimer in README, UI and reports.
