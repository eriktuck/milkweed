"""One-off migration: rename the CSP label `savings` → `shrinking`.

Background
----------
The CSP label `savings` (the top-level group containing the sinking-fund
categories gifts/vacations/home_improvements) is being renamed to `shrinking`
to disambiguate it from the Monarch "Savings" category, which represents money
moved into long-term investment accounts and maps to the `investments` label.

What this changes
-----------------
For every doc in the `users` and `households` collections:
  * csp_labels  — values equal to "savings" become "shrinking".
                  (The KEY "savings", whose value is "investments", is left
                  untouched — that is the Monarch Savings → investments path.)
  * cat_order   — the display header "Savings" becomes "Shrinking".
                  (The lowercase "savings" csp key, which sits under the
                  Investments header, is left untouched.)

For every transactions subcollection under those docs:
  * documents with csp_label == "savings" are updated to "shrinking".
    (The `csp` field — e.g. gifts/vacations/home_improvements, or the
    investments "savings" key — is never touched.)

What this deliberately does NOT change
--------------------------------------
  * csp_from_group / csp_from_category (their "savings" value is the csp KEY
    that maps to the `investments` label).
  * group_names (the literal Monarch group name "Savings").

Targets whichever Firestore the environment points at: the emulator when
FIRESTORE_EMULATOR_HOST is set, otherwise production. Dry-run by default;
pass --apply to write.

Usage
-----
    # Dry run (read-only) — prints exactly what would change
    uv run python -m scripts.migrate_savings_to_shrinking

    # Apply against whatever Firestore the env points at
    uv run python -m scripts.migrate_savings_to_shrinking --apply
"""

from __future__ import annotations

import argparse
import os

from core.services.firebase import db, commit_in_batches

OLD_LABEL = "savings"
NEW_LABEL = "shrinking"
OLD_HEADER = "Savings"
NEW_HEADER = "Shrinking"

CONFIG_COLLECTIONS = ("users", "households")


def transform_config(data: dict) -> tuple[dict, list[str]]:
    """Return (fields_to_write, change_descriptions) for one config doc.

    Pure function — no Firestore access — so it can be tested offline against a
    downloaded config. Returns an empty dict when nothing needs to change.
    """
    changes: list[str] = []
    updates: dict = {}

    csp_labels = data.get("csp_labels")
    if isinstance(csp_labels, dict):
        new_labels = {
            k: (NEW_LABEL if v == OLD_LABEL else v) for k, v in csp_labels.items()
        }
        if new_labels != csp_labels:
            flipped = [k for k, v in csp_labels.items() if v == OLD_LABEL]
            changes.append(f"csp_labels: {len(flipped)} keys → '{NEW_LABEL}' ({', '.join(flipped)})")
            updates["csp_labels"] = new_labels

    cat_order = data.get("cat_order")
    if isinstance(cat_order, list) and OLD_HEADER in cat_order:
        new_order = [NEW_HEADER if c == OLD_HEADER else c for c in cat_order]
        changes.append(f"cat_order: header '{OLD_HEADER}' → '{NEW_HEADER}'")
        updates["cat_order"] = new_order

    return updates, changes


def migrate_configs(apply: bool) -> None:
    for collection in CONFIG_COLLECTIONS:
        for doc in db.collection(collection).stream():
            updates, changes = transform_config(doc.to_dict() or {})
            if not changes:
                continue
            print(f"  [{collection}/{doc.id}]")
            for c in changes:
                print(f"      - {c}")
            if apply:
                doc.reference.set(updates, merge=True)


def migrate_transactions(apply: bool) -> None:
    for collection in CONFIG_COLLECTIONS:
        for doc in db.collection(collection).stream():
            txn_ref = doc.reference.collection("transactions")
            stale = list(txn_ref.where("csp_label", "==", OLD_LABEL).stream())
            if not stale:
                continue
            print(f"  [{collection}/{doc.id}/transactions] {len(stale)} docs → csp_label '{NEW_LABEL}'")
            if apply:
                commit_in_batches(
                    stale,
                    lambda batch, d: batch.update(d.reference, {"csp_label": NEW_LABEL}),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes. Without this flag the script is read-only (dry run).",
    )
    args = parser.parse_args()

    target = os.environ.get("FIRESTORE_EMULATOR_HOST")
    target_desc = f"EMULATOR ({target})" if target else "PRODUCTION"
    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"Migration savings → shrinking | target: {target_desc} | mode: {mode}\n")

    print("Config docs:")
    migrate_configs(args.apply)
    print("\nTransactions:")
    migrate_transactions(args.apply)

    if not args.apply:
        print("\nDry run complete. Re-run with --apply to write these changes.")
    else:
        print("\nMigration applied.")


if __name__ == "__main__":
    main()
