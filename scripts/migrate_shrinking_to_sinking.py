"""One-off migration: rename the CSP label `shrinking` → `sinking`.

Background
----------
The CSP label for the sinking-fund group (gifts/vacations/home_improvements —
money set aside for near/midterm goals) was previously renamed `savings` →
`shrinking` (see migrate_savings_to_shrinking.py). "Shrinking" was a misnomer;
the correct term for these set-aside funds is "sinking funds", so the label is
being renamed `shrinking` → `sinking`.

What this changes
-----------------
For every doc in the `users` and `households` collections:
  * csp_labels  — values equal to "shrinking" become "sinking".
  * cat_order   — the display header "Shrinking" becomes "Sinking".

For every transactions subcollection under those docs:
  * documents with csp_label == "shrinking" are updated to "sinking".
    (The `csp` field — e.g. gifts/vacations/home_improvements — is never touched.)

What this deliberately does NOT change
--------------------------------------
  * csp_from_group / csp_from_category (these map Monarch group/category names to
    csp KEYS, not labels — they never held the value "shrinking").
  * group_names (the literal Monarch group names).
  * the `savings` csp key / `investments` label path (untouched).

Targets whichever Firestore the environment points at: the emulator when
FIRESTORE_EMULATOR_HOST is set, otherwise production. Dry-run by default;
pass --apply to write.

Usage
-----
    # Dry run (read-only) — prints exactly what would change
    uv run python -m scripts.migrate_shrinking_to_sinking

    # Apply against whatever Firestore the env points at
    uv run python -m scripts.migrate_shrinking_to_sinking --apply
"""

from __future__ import annotations

import argparse
import os

from core.services.firebase import db, commit_in_batches

OLD_LABEL = "shrinking"
NEW_LABEL = "sinking"
OLD_HEADER = "Shrinking"
NEW_HEADER = "Sinking"

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
    print(f"Migration shrinking → sinking | target: {target_desc} | mode: {mode}\n")

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
