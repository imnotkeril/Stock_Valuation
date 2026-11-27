# Contributing

Thanks for taking a look. The most useful changes are small, tested fixes and clearly described valuation corrections with a source.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r StockAnalysisSystem/requirements.txt -r StockAnalysisSystem/requirements-dev.txt
python StockAnalysisSystem/run.py    # http://localhost:8501
```

Copy `.env.example` to `.env` only if you need API keys. Never commit `.env*` files or keys. Every key is optional.

## Checks

Run these before you open a pull request. CI runs the lint and test steps on Python 3.11, 3.12 and 3.13.

```bash
python -m ruff check StockAnalysisSystem
STOCK_ANALYSIS_OFFLINE=1 python -m pytest -o addopts="" -p no:warnings
python StockAnalysisSystem/tools/docs_check.py
```

On Windows PowerShell, set the variable with `$env:STOCK_ANALYSIS_OFFLINE = "1"` first.

The test suite is offline and must stay that way: it uses recorded fixtures and synthetic data, never the network.

## Pull requests

- Branch from `master`; one topic per pull request.
- Add or update a test for every behaviour change.
- Valuation changes need a golden or hand-computed test, and a matching update to [docs/METHODOLOGY.md](docs/METHODOLOGY.md).
- Keep the "not investment advice" disclaimer in the README, the UI and any report or export text.
- Lint, tests and the docs check must pass.

## Reporting problems

Use the issue templates. For security problems follow [SECURITY.md](SECURITY.md) instead of opening a public issue.
