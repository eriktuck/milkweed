"""Migrate Profile + income (and optionally Retirement assumptions) from the
running Firestore emulator into **prod** Firestore.

The emulator is seeded from prod with matching doc IDs (see seed_emulator.py), so
the user's emulator uid equals their prod uid. This copies a small, explicit set
of fields from emulator → prod with merge=True, so the rest of the prod user doc
(accounts, csp_plans, transactions, …) is never touched.

Defaults to a DRY RUN — it prints a field-by-field diff and writes nothing.
Pass --commit to actually write to prod.

Usage (emulator must be running with your data):
  uv run python -m scripts.migrate_profile_to_prod                 # dry run
  uv run python -m scripts.migrate_profile_to_prod --commit        # write to prod
  uv run python -m scripts.migrate_profile_to_prod --with-retirement --commit
"""

import argparse
import os

import firebase_admin
from firebase_admin import credentials, firestore

SERVICE_ACCOUNT = "secrets/firebase-service-account"
EMULATOR_HOST = "localhost:8090"
USER_NAME = "erik"

# Profile demographics + income (mirrors core.services.firebase.PROFILE_FIELDS).
PROFILE_FIELDS = (
    "birth_date",
    "coast_age",
    "retirement_age",
    "claim_age",
    "death_age",
    "income_growth_rate",
    "income_segments",
)


def _prod_db():
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    app = firebase_admin.initialize_app(cred, name="prod")
    return firestore.client(app=app)


def _emulator_db():
    # Must be set before the emulator Firestore client is created. Scoped to this
    # client only (the prod client is created first, without it).
    os.environ["FIRESTORE_EMULATOR_HOST"] = EMULATOR_HOST
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    app = firebase_admin.initialize_app(cred, name="emulator")
    return firestore.client(app=app)


def _find_user(db, name: str) -> str | None:
    for doc in db.collection("users").stream():
        if (doc.to_dict() or {}).get("name", "").lower() == name.lower():
            return doc.id
    return None


def _fmt(v) -> str:
    if isinstance(v, list):
        return f"[{len(v)} items]"
    return repr(v)


def migrate(commit: bool, with_retirement: bool):
    # Prod client first (no emulator env), then the emulator client.
    prod = _prod_db()
    emulator = _emulator_db()

    uid = _find_user(emulator, USER_NAME)
    if not uid:
        raise SystemExit(f"No user named '{USER_NAME}' found in the emulator.")

    prod_doc = prod.collection("users").document(uid).get()
    if not prod_doc.exists:
        raise SystemExit(
            f"users/{uid} does not exist in prod — refusing to create a new doc.")
    prod_data = prod_doc.to_dict() or {}
    if (prod_data.get("name", "").lower() != USER_NAME.lower()):
        raise SystemExit(
            f"Safety check failed: prod users/{uid} name is "
            f"{prod_data.get('name')!r}, expected {USER_NAME!r}.")

    emu_data = emulator.collection("users").document(uid).get().to_dict() or {}

    # Build the payload from the fields present in the emulator (skip absent ones).
    payload = {k: emu_data[k] for k in PROFILE_FIELDS if k in emu_data}
    if with_retirement and "retirement" in emu_data:
        payload["retirement"] = emu_data["retirement"]

    print(f"Migrating users/{uid} ({USER_NAME}) emulator → prod")
    print(f"{'FIELD':<20} {'EMULATOR (new)':<28} {'PROD (current)'}")
    print("-" * 78)
    for k, v in payload.items():
        cur = prod_data.get(k, "<absent>")
        changed = "" if cur == v else "  *"
        print(f"{k:<20} {_fmt(v):<28} {_fmt(cur)}{changed}")
    if with_retirement and "retirement" not in emu_data:
        print("retirement           <absent in emulator — nothing to migrate>")

    # Show income segments in detail (the bulk of the data).
    segs = payload.get("income_segments")
    if segs:
        print(f"\nincome_segments ({len(segs)}):")
        for s in segs:
            print(f"    {s.get('date')}  ${float(s.get('amount', 0)):>12,.2f}/yr")

    if not commit:
        print("\nDRY RUN — nothing written. Re-run with --commit to write to prod.")
        return

    prod.collection("users").document(uid).set(payload, merge=True)
    print(f"\n✓ Wrote {len(payload)} field(s) to prod users/{uid} (merge=True).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Migrate Profile/income emulator → prod.")
    p.add_argument("--commit", action="store_true",
                   help="Actually write to prod (default: dry run).")
    p.add_argument("--with-retirement", action="store_true",
                   help="Also migrate the users/{uid}.retirement assumptions.")
    args = p.parse_args()
    migrate(args.commit, args.with_retirement)
