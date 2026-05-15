# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Milkweed is a self-hosted personal finance dashboard built on Plotly Dash (with a Flask backend). It implements the "Conscious Spending Plan" philosophy from *I Will Teach You To Be Rich*. Users authenticate via Firebase, and financial data is stored in Firestore with transactions sourced from Monarch Money.

## Development Commands

```bash
# Run locally (prod — writes to real Firestore)
python app.py

# Run locally against the Firebase emulator (safe, no prod writes)
firebase emulators:start                          # terminal 1 — UI at http://localhost:4000
ENV_PATH=secrets/env-file.dev uv run python app.py  # terminal 2

# Run with Docker Compose (prod — mirrors Cloud Run)
docker compose up --build

# Run with Docker Compose against the Firebase emulator
firebase emulators:start                          # terminal 1
docker compose --profile dev up --build           # terminal 2

# Deploy to Cloud Run via Cloud Build
gcloud builds submit --config cloudbuild.yaml

# Install/sync dependencies
uv sync
uv export --no-hashes -o requirements.txt  # regenerate requirements.txt for Docker

# Run utility scripts (must be run from repo root so relative imports resolve)
python -m scripts.download_monarch
python -m scripts.process_transactions
python -m scripts.download_config
```

There are no automated tests beyond `scripts/test_downloads.py`.

### Firebase Emulator

The Firestore emulator runs on port 8090; the emulator UI runs on port 4000. Auth is not emulated — login still uses real Firebase credentials. The `firebase-admin` SDK redirects all Firestore reads/writes to the emulator automatically when `FIRESTORE_EMULATOR_HOST` is set.

Env-file variants in `secrets/` (gitignored):
- `env-file` — prod (no emulator var set)
- `env-file.dev` — local dev (`FIRESTORE_EMULATOR_HOST=localhost:8090`)
- `env-file.docker-dev` — Docker dev (`FIRESTORE_EMULATOR_HOST=host.docker.internal:8090`)

## Environment / Secrets

Credentials live in `secrets/env-file` (loaded via `python-dotenv`). Required vars:
- `FLASK_SECRET_KEY`
- `FIREBASE_API_KEY`, `FIREBASE_AUTH_DOMAIN`, `FIREBASE_PROJECT_ID`, `FIREBASE_APP_ID`
- `MILKWEED_DEVICE_UUID` — required by the Monarch Money client for session headers
- `FIREBASE_CREDENTIALS` (Cloud Run) or `GOOGLE_APPLICATION_CREDENTIALS` (local path to service account JSON)

Firebase initialisation in `core/services/firebase.py` tries three strategies in order: JSON string env var → file path → Application Default Credentials (ADC for Cloud Run).

## Architecture

### Request / Auth Flow

1. Flask serves `templates/index.html`, which handles Firebase Authentication client-side.
2. On sign-in, the browser POSTs a Firebase ID token to `/login`; Flask verifies it with `firebase_admin.auth` and stores `user_id` in the server-side session.
3. `/dash/` redirects unauthenticated users back to `/`; `app.layout` is a function (`protected_layout`) that checks the Flask session before rendering Dash.

### Dash Page Structure

`app.py` mounts Dash at `/dash/` using `use_pages=True`. Pages register themselves with `dash.register_page(__name__, path=...)`. Shared client-side state uses `dcc.Store`:
- `config-store` (session) — serialised `SessionData` JSON
- `transaction-data-store` (memory) — full processed transactions DataFrame as JSON
- `transaction-subset-store` (memory) — date-filtered subset
- `monarch-session-store` (session) — serialised `MonarchMoney` object (pickle + base64)

### Data / Config Model

`SessionData` (in `core/models/session.py`) is the central config object. It loads per-user and per-household Firestore documents into Pydantic-validated `UserConfig` / `HouseholdConfig` models. It is serialised to/from JSON for `dcc.Store` persistence.

Firestore collections:
- `users/{uid}` — individual config + sub-collection `budgets/{year}-{month}`
- `households/{hid}` — household config (contains `members` list) + sub-collection `budgets/{year}-{month}`
- `users/{uid}/transactions`, `households/{hid}/transactions` — processed transaction documents

### Transaction Pipeline

Raw Monarch transactions → `convert_raw_transactions_to_dataframe` → `preprocess_transactions` (flatten nested `category`/`account` dicts) → `process_and_attribute_transactions` (splits by account owner, applies CSP category mapping from config) → Firestore via `update_firestore_transactions` (batch delete + batch set within the requested date window).

Firestore writes use `commit_in_batches` (batch size 400, hard limit 500) to avoid the Firestore 500-writes-per-batch limit.

### Financial Models (`core/models/`)

Beyond the Dash app, `core/models/` contains a standalone financial modelling library for retirement scenario planning:
- `core.py` — `Stream` and base classes for projected income/expense time series
- `transactions.py` — `Transactions` wrapper around a DataFrame with filter/aggregate helpers
- `individual.py`, `household.py`, `business.py`, `healthcare.py`, `portfolio.py`, `retirement.py` — composable entities that produce `pd.Series` projections by year

These models are used in `pages/trends.py` and are intended to back the future retirement planning page. They are independent of Firestore and can be exercised in notebooks.

## Key Conventions

- **CSP labels**: `fixed`, `investments`, `savings`, `guilt-free`, `income`. The mapping from Monarch category → CSP label is stored per-user in Firestore config (`csp_from_group`, `csp_from_category`, `csp_labels`).
- **Account ownership**: Transactions are attributed to a user or household based on the `accounts` list in each config. Joint accounts live under the household document.
- **`uv` for dependency management**: `pyproject.toml` + `uv.lock` are the source of truth; `requirements.txt` is generated from them for Docker.
- **Cloud Run service name**: `budgetbaby` (the older project name; the repo is `milkweed`).
