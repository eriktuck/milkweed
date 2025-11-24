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
    firebase_credentials_json = os.environ.get("FIREBASE_CREDENTIALS")

    if firebase_credentials_json:
        # Load credentials from the environment variable (Cloud Run)
        try:
            cred = credentials.Certificate(json.loads(firebase_credentials_json))
            firebase_admin.initialize_app(cred)
            print("Firebase app initialized from environment variable.")
        except json.JSONDecodeError:
            raise ValueError("Error decoding Firebase credentials from environment variable.")
    else:
        # Load credentials from the local file (local development)
        try:
            cred = credentials.Certificate(
                os.path.join("secrets", "firebase-service-account")
            )
            firebase_admin.initialize_app(cred)
            print("Firebase app initialized from local file.")
        except FileNotFoundError:
            raise ValueError(
                "Firebase service account file not found at secrets/firebase-service-account"
            )
        except Exception as e:
            raise ValueError(f"Error loading Firebase credentials from file: {e}")

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
