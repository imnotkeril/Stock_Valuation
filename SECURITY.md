# Security Policy

## Scope

Stock Analysis System is a local, single-user Streamlit app for research. It has no accounts, no logins, no trading and no order routing. It stores no personal data beyond a local cache of downloaded market data.

- **Network:** the app binds to `localhost` by default. The Docker image binds to `0.0.0.0` inside the container, so only publish port 8501 on your own machine (`-p 127.0.0.1:8501:8501`). Do not expose the Streamlit port to the internet or a shared network: the app has no authentication.
- **Optional API keys:** `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY`, `FINANCIAL_MODELING_PREP_API_KEY`. They are read from the environment or `.env` only, and are not part of cache file names. Keep `.env` out of version control and out of screenshots and bug reports.
- **SEC EDGAR:** `SEC_EDGAR_USER_AGENT` holds your name and e-mail, which the SEC requires. It is sent to `data.sec.gov`. Do not paste it into public issues.
- **Outbound requests:** ticker symbols and the keys above go to the providers that use them (Yahoo Finance, FRED, SEC EDGAR, and the key-based providers). `STOCK_ANALYSIS_OFFLINE=1` disables all outbound calls.
- **Local cache:** downloaded data is kept under `StockAnalysisSystem/src/data/cache` (git-ignored). Delete it to remove everything the app has stored.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub private reporting: **Security -> Report a vulnerability** at https://github.com/imnotkeril/Stock_Valuation/security/advisories/new.
Include the affected file or endpoint, steps to reproduce and the impact. Expect an acknowledgement within a few days. Maintained on a best-effort basis.

## Supported versions

Only the latest commit on `master`.

## Hardening notes

- Run the app on `localhost` only. Use the Docker port mapping `127.0.0.1:8501:8501` if you run the container.
- Never commit `.env` files; only `.env.example` (names, no values) is tracked.
- If a key was ever committed, rotate it at the provider. Removing it from the working tree does not remove it from git history.
