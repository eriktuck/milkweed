"""Seed Erik's Profile fields (demographics + income segments) into the emulator.

Derives gross-income *salary-change segments* from notebooks/payroll_data.csv:
each distinct paycheck amount that repeats (a steady salary level) becomes one
segment, dated at its first occurrence and annualized to a yearly rate
(semi-monthly cadence ⇒ ×24). One-off bonuses / partial paychecks (amounts that
appear only once) are treated as noise and dropped. A final $0 segment marks the
end of employment, so income winds down to ~0 by 2025.

This is a *starting point*: Erik refines exact DOB and adds earlier history /
part-time periods through the Profile UI.

Usage (emulator must already be running):
  firebase emulators:start --import=./emulator-data        # terminal 1
  uv run python -m scripts.seed_profile                    # writes to users/{erik}
  # then, to persist:  firebase emulators:export ./emulator-data
"""

import os

import firebase_admin
import pandas as pd
from firebase_admin import credentials, firestore

SERVICE_ACCOUNT = "secrets/firebase-service-account"
EMULATOR_HOST = "localhost:8090"
PAYROLL_CSV = "notebooks/payroll_data.csv"

PAY_PERIODS_PER_YEAR = 24  # semi-monthly (15th + month-end)
# Employment winds down after the last steady-salary paycheck; sporadic small
# paychecks after this are treated as marginal. Held as a judgment call (not
# derivable from the noisy tail), so income → 0 from here.
EMPLOYMENT_END = "2024-06-01"

# Static demographics from the worked example (core/models/__init__.py).
# birth_date year is real (1986); month/day are placeholders Erik edits in the UI.
DEMOGRAPHICS = {
    "birth_date": "1986-01-01",
    "coast_age": 50,
    "retirement_age": 67,
    "claim_age": 70,
    "death_age": 90,
    "income_growth_rate": 0.03,
}


def _emulator_db():
    os.environ["FIRESTORE_EMULATOR_HOST"] = EMULATOR_HOST
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    app = firebase_admin.initialize_app(cred, name="seed_profile")
    return firestore.client(app=app)


def derive_income_segments(csv_path: str = PAYROLL_CSV) -> list[dict]:
    """Build annualized salary-change segments from per-paycheck payroll data."""
    df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")

    # Steady salary levels = paycheck amounts that occur more than once.
    counts = df["Gross Income"].value_counts()
    steady = set(counts[counts >= 2].index)

    segments = []
    prev_level = None
    for _, row in df.iterrows():
        amount = row["Gross Income"]
        if amount not in steady:
            continue  # one-off bonus / partial period → noise
        if amount == prev_level:
            continue  # de-dupe: same salary level carried forward
        segments.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "amount": round(float(amount) * PAY_PERIODS_PER_YEAR, 2),
        })
        prev_level = amount

    # Wind-down to $0 at end of employment.
    segments.append({"date": EMPLOYMENT_END, "amount": 0.0})
    return segments


def find_user(db, name: str = "erik") -> str | None:
    for doc in db.collection("users").stream():
        if (doc.to_dict() or {}).get("name", "").lower() == name.lower():
            return doc.id
    return None


def seed():
    db = _emulator_db()

    uid = find_user(db, "erik")
    if not uid:
        raise SystemExit("No user named 'erik' found in the emulator.")

    segments = derive_income_segments()
    payload = {**DEMOGRAPHICS, "income_segments": segments}

    print(f"Seeding profile for users/{uid} (erik):")
    for k, v in DEMOGRAPHICS.items():
        print(f"  {k}: {v}")
    print("  income_segments:")
    for s in segments:
        print(f"    {s['date']}  ${s['amount']:>12,.2f}/yr")

    db.collection("users").document(uid).set(payload, merge=True)
    print("\n✓ Written (merge=True). Export with: firebase emulators:export ./emulator-data")


if __name__ == "__main__":
    seed()
