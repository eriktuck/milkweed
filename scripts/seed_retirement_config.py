"""Optional one-time migration: seed the `retirement` config block.

Background
----------
The Retirement page (spec-retirement.md, Phase 2) adds a `retirement` map to
each user/household config (see core/models/session.RetirementConfig). Every
field is Optional and the service (core/services/retirement.resolve_assumptions)
falls back to DEFAULTS, so **read-compatibility needs no migration** — a config
without the block behaves as if the user accepted all defaults.

This script is therefore *optional*. It backfills the default-able assumptions
(retirement/death/claim ages, phase boundaries, inflation, return, withdrawal
rate) so the values are materialized in Firestore and visible/editable, rather
than living only as code defaults. It deliberately does **not** set
`birth_year` — that cannot be defaulted (it drives the RMD start age, 73 vs 75)
and must be entered by the user on the page.

What this changes
-----------------
For every doc in `users` and `households` that has **no** `retirement` block
(or one missing default-able keys), writes a `retirement` map seeded from
core.services.retirement.DEFAULTS. Existing user-set values are never
overwritten (per-key merge). `birth_year` and `phase_factors` are left untouched.

Targets whichever Firestore the environment points at: the emulator when
FIRESTORE_EMULATOR_HOST is set, otherwise production. Dry-run by default; pass
--apply to write. Per project convention, verify against the emulator before any
prod apply (feedback-migration-workflow).

Usage
-----
    # Dry run (read-only) — prints exactly what would change
    uv run python -m scripts.seed_retirement_config

    # Apply against whatever Firestore the env points at (emulator first!)
    uv run python -m scripts.seed_retirement_config --apply
"""

from __future__ import annotations

import argparse
import os

from core.services.firebase import db
from core.services.retirement import DEFAULTS

CONFIG_COLLECTIONS = ("users", "households")

# Only the retirement-scenario knobs are seeded here. Demographics
# (retirement_age / death_age / claim_age / birth date) are owned by the Profile
# page and live on the top-level config, so they are NOT written under `retirement`.
# phase_factors / account_balances are also excluded (per-user, not defaulted).
SEED_KEYS = ("slow_go_age", "no_go_age", "real_return", "withdrawal_rate")


def retirement_seed_update(data: dict) -> tuple[dict, list[str]]:
    """Return (fields_to_write, change_descriptions) for one config doc.

    Pure function — no Firestore access — so it can be tested offline against a
    downloaded config. Returns an empty dict when the block is already complete.
    Existing user values win (per-key merge); only missing keys are filled.
    """
    existing = data.get("retirement") or {}
    if not isinstance(existing, dict):
        existing = {}

    filled = dict(existing)
    added: list[str] = []
    for key in SEED_KEYS:
        if existing.get(key) is None:
            filled[key] = DEFAULTS[key]
            added.append(f"{key}={DEFAULTS[key]}")

    if not added:
        return {}, []
    return {"retirement": filled}, [f"retirement: seed {', '.join(added)}"]


def migrate(apply: bool) -> None:
    for collection in CONFIG_COLLECTIONS:
        for doc in db.collection(collection).stream():
            updates, changes = retirement_seed_update(doc.to_dict() or {})
            if not changes:
                continue
            print(f"  [{collection}/{doc.id}]")
            for c in changes:
                print(f"      - {c}")
            if apply:
                # merge=True so we never clobber sibling config fields.
                doc.reference.set(updates, merge=True)


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
    print(f"Seed retirement config | target: {target_desc} | mode: {mode}\n")

    migrate(args.apply)

    if not args.apply:
        print("\nDry run complete. Re-run with --apply to write these changes.")
    else:
        print("\nMigration applied.")


if __name__ == "__main__":
    main()
