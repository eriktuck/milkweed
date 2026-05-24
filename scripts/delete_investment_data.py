"""Delete investment data (which contains full account numbers) from Firestore.

Scope — for each target owner this touches ONLY:
  {collection}/{uid}/investments/*              (holdings snapshot subcollection)
  {collection}/{uid}/investment_transactions/*  (investment txn history)
  {collection}/{uid}.investment_accounts          (field — keys are account numbers)
  {collection}/{uid}.investment_account_nicknames (field — keys are account numbers)

It NEVER touches `transactions`, `budgets`, or `csp_snapshots`.

Usage (run from repo root, with PROD credentials — NO emulator var set):
  uv run python -m scripts.delete_investment_data                 # dry run, reports counts
  uv run python -m scripts.delete_investment_data --commit        # actually delete subcollections
  uv run python -m scripts.delete_investment_data --commit --strip-config  # also remove the two config fields
"""

import argparse
import os
import sys

from firebase_admin import firestore

from core.services.firebase import db, commit_in_batches

# (collection, doc_id) pairs to clean.
TARGETS: list[tuple[str, str]] = [
    ("users", "Ij893k3NoQSUc5aFmIwj4xKNKzP2"),
    ("households", "iL5PXLzdhlp8s2pMubTR"),
]

INVESTMENT_SUBCOLLECTIONS = ["investments", "investment_transactions"]
CONFIG_FIELDS = ["investment_accounts", "investment_account_nicknames"]


def _delete(batch, doc):
    batch.delete(doc.reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", action="store_true",
                        help="Actually delete. Without this flag the script only reports (dry run).")
    parser.add_argument("--strip-config", action="store_true",
                        help="Also remove investment_accounts / investment_account_nicknames "
                             "fields from the parent doc (their keys are account numbers).")
    args = parser.parse_args()

    emulator = os.environ.get("FIRESTORE_EMULATOR_HOST")
    print(f"Firestore project : {db.project}")
    print(f"Emulator host     : {emulator or '(none — PROD)'}")
    print(f"Mode              : {'COMMIT (will delete)' if args.commit else 'DRY RUN (read-only)'}")
    print(f"Strip config flds : {args.strip_config}")
    print("-" * 60)

    grand_total = 0
    for collection, doc_id in TARGETS:
        print(f"\n{collection}/{doc_id}")
        parent = db.collection(collection).document(doc_id)

        for sub in INVESTMENT_SUBCOLLECTIONS:
            docs = list(parent.collection(sub).stream())
            print(f"  {sub}: {len(docs)} doc(s)")
            grand_total += len(docs)
            if args.commit and docs:
                commit_in_batches(docs, _delete)
                print(f"    -> deleted {len(docs)} doc(s)")

        snap = parent.get()
        data = snap.to_dict() or {}
        present = [f for f in CONFIG_FIELDS if f in data]
        if present:
            print(f"  parent-doc fields present: {', '.join(present)}")
            if args.strip_config:
                if args.commit:
                    parent.update({f: firestore.DELETE_FIELD for f in present})
                    print(f"    -> removed fields: {', '.join(present)}")
                else:
                    print(f"    (would remove: {', '.join(present)})")
            else:
                print("    (left in place — pass --strip-config to remove)")

    print("-" * 60)
    if args.commit:
        print(f"Done. Deleted {grand_total} investment doc(s).")
    else:
        print(f"Dry run complete. {grand_total} investment doc(s) would be deleted.")
        print("Re-run with --commit (and optionally --strip-config) to apply.")


if __name__ == "__main__":
    main()
