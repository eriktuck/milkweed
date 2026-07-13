"""Transaction pipeline: raw Monarch dicts → processed, owner-attributed frames.

Pipeline order: convert_raw_transactions_to_dataframe → preprocess_transactions
→ process_and_attribute_transactions (which fans out per owner via
filter_owner_transactions + categorize_transactions).

Moved verbatim from core/utils/functions.py (Phase 3 of the core/
restructuring); pinned by tests/test_transactions.py.
"""
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def convert_raw_transactions_to_dataframe(txn_raw: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(txn_raw)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    return df


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the transactions DataFrame."""
    df["category_name"] = df["category"].apply(
        lambda x: x.get("name", "") if isinstance(x, dict) else "")

    df["account_name"] = df["account"].apply(
        lambda x: x.get("displayName", "") if isinstance(x, dict) else "")

    df.drop(columns=["category", "account", "merchant"], inplace=True)

    return df


def filter_owner_transactions(df: pd.DataFrame, user_config: dict):
    """Split transactions by owner based on account ownership in config.

    Accounts explicitly marked ``include == False`` in
    ``transaction_account_settings`` are dropped (their transactions are not
    saved). Accounts with no settings entry are included, so configs written
    before the control plane behave exactly as before.
    """
    accounts = user_config.get("accounts", [])
    name = user_config.get("name", "unknown")

    settings = user_config.get("transaction_account_settings") or {}
    included_accounts = [
        acct for acct in accounts
        if settings.get(acct, {}).get("include", True)
    ]

    # Filter rows
    filt = df["account_name"].isin(included_accounts)
    subset = df.loc[filt].copy()

    # Tag owner
    subset["account_owner"] = name

    if subset.empty:
        print(f"No transactions for user {name}")

    return subset


def categorize_transactions(df: pd.DataFrame, user_config: dict):
    """Categorize transactions according to the user's config and drop excluded categories"""
    category_names = user_config["cat_names"]
    csp_from_group = user_config["csp_from_group"]
    csp_from_category = user_config["csp_from_category"]
    csp_labels = user_config["csp_labels"]
    drop_cats = user_config["drop_cats"]

    # Assign labels
    df["category_group"] = df["category_name"].map(category_names)
    df = df.assign(
        csp_from_group=df["category_group"].map(csp_from_group),
        csp_from_category=df["category_name"].map(csp_from_category),
        csp=lambda x: x["csp_from_group"].fillna(x["csp_from_category"]).fillna("guilt_free"),
        csp_label=lambda x: x["csp"].map(csp_labels)
    )

    # Drop transactions
    df = df.loc[~df["category_name"].isin(drop_cats) & ~df["hideFromReports"]]

    return df


def filter_and_process_owner_transactions(df: pd.DataFrame, user_config: dict) -> pd.DataFrame:
    return categorize_transactions(
        filter_owner_transactions(df, user_config),
        user_config
    )


def check_processed(new_txn_df, processed):
    """Helper function to print deleted transactions."""
    # Combine processed and attributed transactions
    new_combined_txns = pd.concat(
        [entry['transactions'] for entry in processed],
        ignore_index=True
    )

    # Check dropped transactions (should only be from drop_cats)
    combined_ids = set(new_combined_txns["id"])
    new_ids = set(new_txn_df["id"])

    dropped = new_ids - combined_ids
    dropped_txns = new_txn_df[new_txn_df["id"].isin(dropped)]
    if len(dropped) > 0:
        print("Dropped transactions:")
        print(dropped_txns)

    return dropped_txns


def process_and_attribute_transactions(
        new_txn_df: pd.DataFrame,
        config,
        save_dropped: Optional[bool] = False
    ) -> list[dict[str, Any]]:
    user_configs = config.get_user_configs()

    processed = []
    for uid, user_config in user_configs.items():

        # Determine whether it's a user or household config
        if "members" in user_config:
            kind = "households"
        else:
            kind = "users"

        name = user_config["name"]

        # Attribute and process this owner's transactions
        df_owner = filter_and_process_owner_transactions(new_txn_df, user_config)

        # Store structured result
        processed.append({
            "owner_name": name,
            "kind": kind,
            "uid": uid,
            "transactions": df_owner
        })

    dropped_txn = check_processed(new_txn_df, processed)
    if save_dropped:
        fpath = Path("data") / 'dropped_txn.pkl'
        fpath.parent.mkdir(parents=True, exist_ok=True)

        dropped_txn.to_pickle(fpath)

    return processed
