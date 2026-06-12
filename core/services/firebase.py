import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
from typing import Optional
import pandas as pd
import pytz
from datetime import datetime as dt

from core.utils.functions import (
    convert_raw_transactions_to_dataframe,
    preprocess_transactions,
    process_and_attribute_transactions
)

# Prevent multiple initializations
if not firebase_admin._apps:
    # 1. Try to get the JSON string from Environment (Cloud Run secret variable)
    firebase_credentials_json = os.environ.get("FIREBASE_CREDENTIALS")
    
    # 2. Try to get the path to a key file (Local dev or Mounted Secret)
    # We check the specific variable or fall back to your local default
    service_account_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "secrets/firebase-service-account")

    if firebase_credentials_json:
        # Scenario A: Credentials passed as a raw JSON string (Env Var)
        try:
            cred_dict = json.loads(firebase_credentials_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            print("Firebase initialized via FIREBASE_CREDENTIALS env var.")
        except json.JSONDecodeError:
            print("Error: FIREBASE_CREDENTIALS contains invalid JSON.")
            raise

    elif os.path.exists(service_account_path):
        # Scenario B: Credentials found in a file (Local Dev or Mounted Volume)
        try:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
            print(f"Firebase initialized via file at {service_account_path}.")
        except Exception as e:
            print(f"Error loading credentials file: {e}")
            raise

    else:
        # Scenario C: Application Default Credentials (ADC)
        # This is the "Magic" Cloud Run mode. If no keys are provided, 
        # it uses the identity of the Cloud Run instance itself.
        print("No explicit credentials found. Attempting Application Default Credentials (ADC)...")
        try:
            firebase_admin.initialize_app()
            print("Firebase initialized via ADC.")
        except Exception as e:
            raise ValueError(f"Could not initialize Firebase. No keys found and ADC failed: {e}")

db = firestore.client()


def find_household_for_user(db, uid):
    query = (
        db.collection("households")
        .where("members", "array_contains", uid)
        .limit(1)
        .stream()
    )

    household_doc = next(query, None)
    if household_doc is None:
        print(f"No household found for user {uid}.")
        return None

    return household_doc.id


def get_user_config(
    collection_str: str, 
    doc_id: str
) -> Optional[dict]:
    """Retrieve user or household configuration from Firestore."""
    users_ref = db.collection(collection_str)
    
    user_doc = users_ref.document(doc_id).get()
    if not user_doc.exists:
        return None
    
    return user_doc.to_dict()


def fetch_all_transactions(uid: str, config) -> pd.DataFrame:
    """Fetch all transactions for the logged in member, household and household members"""
    # Initialize empty list of transactions
    all_txns = []

    # Helper function to stream into the transaction list
    def fetch_into(owner_kind: str, uid: str):
        """
        owner_kind = "users" or "households"
        owner_uid = Firestore doc id
        owner_name = name ot assign in df["account_owner"]
        """
        ref = db.collection(owner_kind).document(uid).collection("transactions")
        for doc in ref.stream():
            txn = doc.to_dict()
            txn["account_owner"] = uid
            all_txns.append(txn)
    
    # Get all user configs
    user_configs = config.get_user_configs()

    # Fetch logged in user's transactions
    fetch_into("users", uid)

    # Identify household config and household members
    household_id = None
    household_config = None
    member_configs = {}

    for cfg_id, cfg in user_configs.items():
        if cfg_id == uid:
            continue  # Already loaded logged in member's txns
        if "members" in cfg:
            household_id = cfg_id
            household_config = cfg
        else:
            member_configs[cfg_id] = cfg
    
    if household_config is not None:
        household_id = household_config["uid"]
        fetch_into("households", household_id)
    
    for member_uid, cfg in member_configs.items():
        fetch_into("users", member_uid)
    
    return pd.DataFrame(all_txns)


def commit_in_batches(items, write_fn, batch_size=400):
    """
    Generic Firestore batch commit helper.
    
    Parameters
    ----------
    items : iterable
        Items to write/delete. Could be records or IDs.
    write_fn : function(batch, item)
        A function defining how to write or delete the item.
        Must accept (batch, item).
    batch_size : int
        Number of writes per batch. Must be <= 500.
    """
    batch = db.batch()
    count = 0

    for item in items:
        write_fn(batch, item)
        count += 1

        if count >= batch_size:
            batch.commit()
            batch = db.batch()
            count = 0

    if count > 0:
        batch.commit()


def save_csp_snapshot_to_firestore(collection_str: str, uid: str, key: str, data: dict):
    """Write a CSP snapshot document under csp_snapshots/{key}.

    key is 'plan' (spending plan monthly averages) or 'net_worth'.
    """
    ref = (
        db.collection(collection_str)
        .document(uid)
        .collection("csp_snapshots")
        .document(key)
    )
    clean = {k: float(v) for k, v in data.items() if v is not None}
    ref.set(clean)


def save_budget_to_firestore(collection_str: str, uid: str, year: str | int, budget_data: dict):
    """Write/overwrite all monthly budget documents for one year.

    budget_data: {month (int or str): {category: value}}
    Documents are named '{year}-{month}' and fully replaced on each save.
    """
    ref = db.collection(collection_str).document(uid).collection("budgets")

    def _write(batch, item):
        month, categories = item
        doc_id = f"{year}-{int(month):02d}"
        clean = {k: float(v) for k, v in categories.items() if v is not None and v == v}
        batch.set(ref.document(doc_id), clean)

    commit_in_batches(budget_data.items(), _write)


def fetch_budget(owner_id: str, year: int, month: int) -> dict:
    """Fetch one month's budget for an owner: {csp_key: amount}.

    Budgets live under `{users|households}/{owner_id}/budgets/{YYYY-MM}` and are
    keyed by CSP key (the same keys as `csp_labels`). `owner_id` is unique to one
    collection, so we check households then users and return the first that exists
    (empty dict if neither has a budget for that month).
    """
    doc_id = f"{year}-{int(month):02d}"
    for kind in ("households", "users"):
        snap = (db.collection(kind).document(owner_id)
                .collection("budgets").document(doc_id).get())
        if snap.exists:
            return snap.to_dict() or {}
    return {}


def save_transaction_account_config(owner_docs: dict, assignments: dict):
    """Persist transaction-account ownership + per-account settings.

    Ownership is encoded by which config doc's ``accounts`` list contains an
    account's displayName; ``transaction_account_settings`` holds the per-account
    include flag and nickname. Both are rebuilt from ``assignments`` for every
    in-scope doc so each account belongs to exactly one owner.

    Parameters
    ----------
    owner_docs : dict[str, str]
        ``{uid: kind}`` for every config doc in scope, where ``kind`` is
        ``"users"`` or ``"households"``. These docs' ``accounts`` list and
        ``transaction_account_settings`` map are rewritten.
    assignments : dict[str, dict]
        Keyed by account ``displayName``::

            {"<displayName>": {"owner": <uid> | None,
                               "include": bool,
                               "nickname": str | None}}

        An account with ``owner is None`` is unassigned: it is removed from
        every doc and its transactions will not be attributed or saved.
    """
    for uid, kind in owner_docs.items():
        accounts = []
        settings = {}
        for name, a in assignments.items():
            if a.get("owner") != uid:
                continue
            accounts.append(name)
            entry = {"include": bool(a.get("include", True))}
            nickname = (a.get("nickname") or "").strip()
            if nickname:
                entry["nickname"] = nickname
            settings[name] = entry

        db.collection(kind).document(uid).set(
            {"accounts": accounts, "transaction_account_settings": settings},
            merge=True,
        )


# Profile fields persisted on users/{uid} (see core/models/session.py UserConfig
# and the Profile page). Only these keys are writable through save_user_profile.
PROFILE_FIELDS = (
    "birth_date",
    "coast_age",
    "retirement_age",
    "claim_age",
    "death_age",
    "income_growth_rate",
    "income_segments",
)


def save_user_profile(uid: str, payload: dict) -> None:
    """Persist Profile demographics + income to users/{uid} (merge).

    Only keys in ``PROFILE_FIELDS`` are written; everything else is ignored so a
    callback can hand over a wider dict without clobbering unrelated config. Keys
    mapping to ``None`` are skipped (left unchanged) rather than written as null.
    """
    data = {
        k: payload[k]
        for k in PROFILE_FIELDS
        if k in payload and payload[k] is not None
    }
    if data:
        db.collection("users").document(uid).set(data, merge=True)


def update_firestore_transactions(
        collection_str: str,
        uid: str,
        txn_df: pd.DataFrame,
        start_date: str,
        end_date: str
    ):
    # Format string dates
    utc = pytz.UTC
    start_date_dt = dt.fromisoformat(start_date).replace(tzinfo=utc)
    end_date_dt = dt.fromisoformat(end_date).replace(tzinfo=utc)

    txn_ref = db.collection(collection_str).document(uid).collection("transactions")

    # Delete old transactions in the date range
    old_docs = (
        txn_ref
        .where("date", ">=", start_date_dt)
        .where("date", "<=", end_date_dt)
        .stream()
    )

    def _delete(batch, doc):
        batch.delete(doc.reference)

    commit_in_batches(
        old_docs, 
        _delete
    )

    # Update new transactions
    records = txn_df.to_dict(orient='records')

    def _write(batch, txn):
        txn_id = str(txn["id"])
        batch.set(txn_ref.document(txn_id), txn)

    commit_in_batches(records, _write)


def delete_all_transactions(collection_str: str, uid: str):
    """
    Delete ALL transactions for a given owner (user or household).
    """
    txn_ref = db.collection(collection_str).document(uid).collection("transactions")

    def _delete(batch, doc):
        batch.delete(doc.reference)

    commit_in_batches(
        txn_ref.stream(), 
        _delete
    )


def sync_raw_transactions(
        raw_txns, 
        config, 
        start_date, 
        end_date, 
        delete_all=False
    ):
    new_df = convert_raw_transactions_to_dataframe(raw_txns)
    new_df = preprocess_transactions(new_df)

    processed = process_and_attribute_transactions(new_df, config)

    for entry in processed:
        if delete_all:
            delete_all_transactions(entry["kind"], entry["uid"])

        update_firestore_transactions(
            entry["kind"],
            entry["uid"],
            entry["transactions"],
            start_date,
            end_date
        )
    
    logged_in_uid = config.logged_in_uid
    all_transactions = fetch_all_transactions(logged_in_uid, config)
    
    return all_transactions
