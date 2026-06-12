# Milkweed 🌱

**A self-hosted personal finance tool for living your Rich Life.**

Milkweed is a personal finance application designed to help you implement the "Conscious Spending Plan" philosophy popularized by Ramit Sethi in *I Will Teach You To Be Rich*. Milkweed helps you track your fixed costs and investments so you can enjoy your **Guilt-Free Spending** without worry.

Currently, this project is intended for developers or tech-savvy users to fork and host on their own infrastructure (Firebase + Google Cloud). Integration with the Monarch Money API supports transaction import (using the excellent [monarchmoney](https://github.com/hammem/monarchmoney) library).

> Do you have transactions but need them categorized? Try out my [transaction classifier](https://github.com/eriktuck/txn-classifier), a fine-tuned BERT classifier for IWT categories. 

## Features

### 📊 Planned vs. Actuals

Visualize your spending to date at a glance. The dashboard features a bar chart comparing your **Planned** allocation against your **Actual** spending for the month. Instantly see if you are hitting your targets for Fixed Costs, Investments, Savings, and Guilt-Free Spending.

![img](https://storage.googleapis.com/ei-dev-assets/assets/milkweed-pva.png)

### 💰 Budget

A spreadsheet-style interface where you can input your monthly income and expenses. The **Guilt-Free Button** at the top automatically assigns any unallocated budget to "Guilt-Free Spending." This is the money you can spend on dining out, travel, or whatever you love—guilt-free—because your obligations are already met.

![img](https://storage.googleapis.com/ei-dev-assets/assets/milkweed-budget.png)

### 💸 Conscious Spending Plan

Automatically populate a **Conscious Spending Plan** with planned or actual amounts from transaction data or your budget. Make changes directly in the plan to explore scenarios.

## Tech Stack

- **Frontend/Hosting:** Google Cloud Run (Containerized Application)
- **Backend/Database:** Google Firestore (NoSQL)
- **Authentication:** Firebase Authentication

## High-Level Setup

Milkweed is designed to be self-hosted. You will own your data completely.

> **Note:** A detailed, step-by-step guide on how to deploy this app on Firebase and Google Cloud Platform will be published on Medium soon. [Link coming soon].

### Prerequisites

**Accounts & services**

- A Google Cloud Platform (GCP) Account.
- A Firebase project (Authentication + Cloud Firestore enabled).
- A [Monarch Money](https://www.monarchmoney.com/) account (for transaction import).

**Local toolchain**

| Tool | Purpose | Install (macOS / Homebrew) |
|------|---------|----------------------------|
| [`uv`](https://docs.astral.sh/uv/) | Python dependency management; also provisions a managed Python ≥3.10 | `brew install uv` |
| Firebase CLI | Running the local Firestore emulator | `npm install -g firebase-tools` |
| `gcloud CLI` | Deploying to Cloud Run | `brew install --cask gcloud-cli` |
| Node.js + npm | Host for the Firebase CLI | `brew install node` |
| Docker Desktop | Optional — running the app via `docker compose` | `brew install --cask docker-desktop` |

> On non-macOS platforms, use the equivalent installer for each tool (e.g. the standalone Firebase CLI binary, the Google Cloud SDK installer, and Docker Desktop / Docker Engine).

**Credentials**

Create a `secrets/` directory (gitignored) containing an `env-file` with your Firebase config and other secrets, plus a service-account JSON for Firestore access. See `CLAUDE.md` for the full list of required environment variables and the emulator-specific `env-file.dev` variant.

After installing, sync dependencies and authenticate:

```bash
uv sync                                  # install Python deps into a managed venv
gcloud auth login                        # for deploys
gcloud auth application-default login    # for local Application Default Credentials
firebase login                           # for the Firestore emulator
```

### Quick Start Overview

- **Fork this Repository:** Clone the repo to your local machine.
- **Create a Firebase Project**
  - Go to the [Firebase Console](https://console.firebase.google.com/) and create a new project.
  - Enable **Authentication** (Google Sign-in or Email/Password).
  - Enable **Cloud Firestore** and start in "Production mode" (you will need to set up security rules).
- **Configure GCP**
  - Ensure the Google Cloud Run API is enabled for your project.Install the dependencies and build the container image.
- **Deploy**
  - Deploy the service to **Google Cloud Run**.
  - Set your environment variables (Firebase config keys) in the Cloud Run console.
- **Run:** Open the URL provided by Cloud Run, create an account, and start building your Conscious Spending Plan.

### Run Locally

Once the [prerequisites](#prerequisites) are installed and your `secrets/` directory is in place, you can run the app on your machine. The recommended workflow uses the **Firebase emulator** so you never read or write production data.

**Against the Firebase emulator (recommended — no prod writes):**

```bash
# Terminal 1 — start the emulator (Firestore on :8090, emulator UI on :4000)
firebase emulators:start --import=./emulator-data --export-on-exit

# Terminal 2 — run the app against the emulator
ENV_PATH=secrets/env-file.dev uv run python app.py
```

The first time, seed the emulator from prod (with the emulator already running, before launching the app):

```bash
uv run python -m scripts.seed_emulator                          # transactions from 2025-01-01
uv run python -m scripts.seed_emulator --start-date 2024-01-01  # wider range
firebase emulators:export ./emulator-data                       # persist for next session
```

> **Note:** Auth is not emulated — login still uses your real Firebase credentials. Only Firestore reads/writes are redirected to the emulator (automatically, when `FIRESTORE_EMULATOR_HOST` is set in `env-file.dev`).

**Against production (writes to real Firestore):**

```bash
python app.py
```

**With Docker Compose:**

```bash
firebase emulators:start                   # terminal 1 (emulator)
docker compose --profile dev up --build    # terminal 2 (app against emulator)

docker compose --profile prod up --build   # or, mirror Cloud Run against prod
```

The app is served at `http://localhost:8080`.

## Roadmap

I am actively developing new features including:

- **Spending Trends:** Historical views and projections to see how your "Rich Life" is evolving month-to-month
- **Retirement Scenario Planning:** Go beyond the simple compound interest calculators to model a non-traditional approach to retirement. The scenario planning module supports complex scenarios while maintaining an intuitive user interface.
- **Rental Income Analysis:** Tools for tracking income from real estate investments. 
- **Net Worth & Portfolio Tracker:** automatically updates investment account values and supports analysis of diversification.