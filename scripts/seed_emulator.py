"""
Seed the local Firestore emulator with a subset of production data.

Copies all user/household configs and budgets, plus transactions on or
after --start-date, from prod Firestore into the running emulator.

Usage (emulator must already be running):
  firebase emulators:start                        # terminal 1
  uv run python -m scripts.seed_emulator          # defaults to 2025-01-01
  uv run python -m scripts.seed_emulator --start-date 2024-01-01
"""

import os
import argparse
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, firestore

SERVICE_ACCOUNT = "secrets/firebase-service-account"
EMULATOR_HOST = "localhost:8090"


def _prod_db():
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    app = firebase_admin.initialize_app(cred, name="prod")
    return firestore.client(app=app)


def _emulator_db():
    # Must be set before the Firestore client is created
    os.environ["FIRESTORE_EMULATOR_HOST"] = EMULATOR_HOST
    cred = credentials.Certificate(SERVICE_ACCOUNT)
    app = firebase_admin.initialize_app(cred, name="emulator")
    return firestore.client(app=app)


def seed(start_date_str: str = "2025-01-01"):
    print(f"Seeding emulator from prod (transactions >= {start_date_str}) …")
    start_dt = datetime.fromisoformat(start_date_str).replace(tzinfo=timezone.utc)

    prod = _prod_db()
    emulator = _emulator_db()

    for collection in ("users", "households"):
        for doc in prod.collection(collection).stream():
            data = doc.to_dict()
            emulator.collection(collection).document(doc.id).set(data)
            print(f"  {collection}/{doc.id}")

            src_ref = prod.collection(collection).document(doc.id)
            dst_ref = emulator.collection(collection).document(doc.id)

            # Budgets sub-collection
            budget_docs = list(src_ref.collection("budgets").stream())
            for bdoc in budget_docs:
                dst_ref.collection("budgets").document(bdoc.id).set(bdoc.to_dict())
            if budget_docs:
                print(f"    {len(budget_docs)} budget docs")

            # CSP snapshots sub-collection
            csp_docs = list(src_ref.collection("csp_snapshots").stream())
            for cdoc in csp_docs:
                dst_ref.collection("csp_snapshots").document(cdoc.id).set(cdoc.to_dict())
            if csp_docs:
                print(f"    {len(csp_docs)} csp_snapshot docs")

            # Investment holdings snapshot sub-collection
            holding_docs = list(src_ref.collection("investments").stream())
            for hdoc in holding_docs:
                dst_ref.collection("investments").document(hdoc.id).set(hdoc.to_dict())
            if holding_docs:
                print(f"    {len(holding_docs)} investment holding docs")

            # Investment transactions sub-collection
            inv_txn_docs = list(src_ref.collection("investment_transactions").stream())
            for idoc in inv_txn_docs:
                dst_ref.collection("investment_transactions").document(idoc.id).set(idoc.to_dict())
            if inv_txn_docs:
                print(f"    {len(inv_txn_docs)} investment transaction docs")

            # Transactions sub-collection (date-filtered)
            txns = list(
                src_ref.collection("transactions")
                .where("date", ">=", start_dt)
                .stream()
            )
            for txn in txns:
                dst_ref.collection("transactions").document(txn.id).set(txn.to_dict())
            if txns:
                print(f"    {len(txns)} transactions")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Firestore emulator from prod.")
    parser.add_argument(
        "--start-date",
        default="2025-01-01",
        help="Copy transactions on or after this date (YYYY-MM-DD). Default: 2025-01-01",
    )
    args = parser.parse_args()
    seed(args.start_date)
