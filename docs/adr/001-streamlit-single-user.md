# ADR-001: Streamlit single-user app instead of FastAPI + Next.js

- Status: accepted
- Date: 2026-11-26 (recorded during adopt-project; decision predates it)
- Deciders: owner / earlier sessions

## Context and problem
The finance/screener router defaults to a FastAPI service plus a Next.js table UI, with Streamlit allowed for a single-user quick version. The app is a personal research tool run locally by one person.

## Decision drivers
- one user, local machine
- fast iteration on analytical views
- Python-only codebase

## Considered options
1. FastAPI + Next.js (router default)
2. Streamlit
3. Dash

## Decision
Chosen: Streamlit, because it keeps the whole product in Python and the router allows it for 'just for me' tools; scans run on demand, so background jobs and alert delivery are not needed.

## Consequences
- Good: one language, quick pages, AppTest for page tests.
- Bad / debt: session-state bugs (stale widget keys), large page modules, no API for other clients.

## Confirmation
App starts with `run.py` on 127.0.0.1:8501; health endpoint answers ok.
