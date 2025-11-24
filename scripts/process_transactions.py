import pandas as pd
from firebase_admin import auth
from pathlib import Path
import argparse

from core.utils.functions import (
    convert_raw_transactions_to_dataframe,
    preprocess_transactions,
    process_and_attribute_transactions
)
from core.services.firebase import (
    db, 
    find_household_for_user,
    get_user_config,
    update_firestore_transactions
)
from core.models.session import SessionData
from scripts.download_monarch import save_transactions_to_pkl

# Local transaction file
FILE_NAME = "e-transactions.pkl"

def load_transactions_from_pkl(filename: str = FILE_NAME) -> pd.DataFrame:
    """Load transactions from a CSV file into a DataFrame."""
    fpath = Path("data") / filename    
    df = pd.read_pickle(fpath)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    
    return df


def main(username: str , filename: str):
    # Get user transactions
    uid = auth.get_user_by_email(username).uid
    if not uid:
        raise ValueError(f"No UID found for email {username}.")
    
    config = SessionData.from_firestore(uid)

    new_txn_df = load_transactions_from_pkl()
    new_txn_df = preprocess_transactions(new_txn_df)
    
    processed = process_and_attribute_transactions(
                new_txn_df, config, save_dropped=True
            )
    
    new_combined_txns = pd.concat(
        [entry['transactions'] for entry in processed], 
        ignore_index=True
    )
    
    save_transactions_to_pkl(new_combined_txns, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process raw transactions from file and save to pkl."
    )

    parser.add_argument(
        "--username",
        required=True,
        help="Monarch Money username or email address"
    )

    parser.add_argument(
        "--filename",
        required=True,
        help="Output filename (e.g., transactions.pkl)"
    )

    args = parser.parse_args()

    main(args.username, args.filename)