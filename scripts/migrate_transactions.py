from firebase_admin import auth
import argparse
import pandas as pd

from core.models.session import SessionData
from core.services.firebase import (
    sync_raw_transactions
)
from scripts.process_transactions import load_transactions_from_pkl


def main(email, filename, delete_all=False):
    """
    Uploads processed transactions to Firestore.
    
    Note
    ----
    Expects raw transactions downloaded from the monarch API. 
    Run scripts.download_monarch before proceeding.
    """
    
    # Get user config
    uid = auth.get_user_by_email(email).uid
    config = SessionData.from_firestore(uid)

    # Read processed transactions
    txn_df = load_transactions_from_pkl(filename)

    start_date = txn_df['date'].min().strftime("%Y-%m-%d")
    end_date = txn_df['date'].max().strftime("%Y-%m-%d")
    
    txn_raw = txn_df.to_dict(orient="records")

    if delete_all:
        confirm = input("WARNING: This will delete ALL transactions for this user. Continue? (y/N): ")
        if confirm.lower() != "y":
            print("Operation cancelled.")
            return

    all_transactions = sync_raw_transactions(
        txn_raw, 
        config, 
        start_date, 
        end_date,
        delete_all=delete_all
    )

    return all_transactions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload local transactions to Firestore."
    )

    parser.add_argument(
        "--username",
        required=True,
        help="Monarch Money username or email address"
    )

    parser.add_argument(
        "--filename",
        required=True,
        help="Pickle file containing raw transactions"
    )

    parser.add_argument(
        "--delete-all",
        action="store_true",
        help="Delete ALL existing transactions in Firestore!!!"
    )

    args = parser.parse_args()

    main(args.username, args.filename, args.delete_all)
