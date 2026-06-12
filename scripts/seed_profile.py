"""Seed Erik's Profile fields (demographics + income segments) into the emulator.

Derives gross-income segments from the official SSA *Earnings Record* in
data/payroll-data.csv (per-year taxed earnings). Each year becomes a Jan 1
income segment; consecutive identical years collapse. Uses
`core.utils.functions.ssa_earnings_to_segments` — the same converter the Profile
page's uploader uses, so seed and UI can't drift.

This is a *starting point*: Erik refines exact DOB and layers in mid-year raises
through the Profile UI.

Usage (emulator must already be running):
  firebase emulators:start --import=./emulator-data        # terminal 1
  uv run python -m scripts.seed_profile                    # writes to users/{erik}
  # then, to persist:  firebase emulators:export ./emulator-data
"""

import os

import firebase_admin
from firebase_admin import credentials, firestore

from core.utils.functions import ssa_earnings_to_segments

SERVICE_ACCOUNT = "secrets/firebase-service-account"
EMULATOR_HOST = "localhost:8090"
PAYROLL_CSV = "data/payroll-data.csv"

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
    """Build yearly income segments from the SSA earnings-record CSV."""
    with open(csv_path) as f:
        return ssa_earnings_to_segments(f.read())


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

    existing = db.collection("users").document(uid).get().to_dict() or {}
    imported = derive_income_segments()

    # Overwrite the earnings history through the record's last year, but keep any
    # existing segment dated after it — those are forward-looking edits (a planned
    # raise / new job) the record can't know about. Mirrors the Profile uploader.
    cutoff = f"{imported[-1]['date'][:4]}-12-31"
    preserved = [s for s in (existing.get("income_segments") or [])
                 if s.get("date") and str(s["date"])[:10] > cutoff]
    segments = sorted(imported + preserved, key=lambda s: s["date"])

    # Only fill demographics the user hasn't already set, so a customized birth
    # date / ages are never clobbered.
    payload = {"income_segments": segments}
    for k, v in DEMOGRAPHICS.items():
        if not existing.get(k):
            payload[k] = v

    print(f"Seeding profile for users/{uid} (erik):")
    for k in DEMOGRAPHICS:
        note = "" if k in payload else "  (kept existing)"
        print(f"  {k}: {payload.get(k, existing.get(k))}{note}")
    print(f"  income_segments: {len(imported)} imported + {len(preserved)} preserved "
          f"(after {cutoff})")
    for s in segments:
        print(f"    {s['date']}  ${s['amount']:>12,.2f}/yr")

    db.collection("users").document(uid).set(payload, merge=True)
    print("\n✓ Written (merge=True). Export with: firebase emulators:export ./emulator-data")


if __name__ == "__main__":
    seed()
