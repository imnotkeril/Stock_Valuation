## What and why

## Checks
- [ ] `python -m ruff check StockAnalysisSystem` passes
- [ ] `STOCK_ANALYSIS_OFFLINE=1 python -m pytest -o addopts="" -p no:warnings` passes
- [ ] `python StockAnalysisSystem/tools/docs_check.py` passes
- [ ] Tests added or updated for behaviour changes; valuation changes have a golden or hand-computed test
- [ ] `docs/METHODOLOGY.md` updated if a valuation or model method changed
- [ ] README / USER_GUIDE / CHANGELOG updated if behaviour changed
- [ ] No secrets, `.env*` files, cache or database files committed
