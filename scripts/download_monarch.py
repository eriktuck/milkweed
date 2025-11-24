import pandas as pd
import monarchmoney
from monarchmoney import MonarchMoney
import asyncio
from pathlib import Path
from datetime import date, datetime
from typing import Any, Optional, List, Tuple
import argparse
import getpass

from core.services.monarch import (
    login_to_monarch,
    fetch_transactions_from_monarch
)


def save_transactions_to_pkl(
        transactions: dict, 
        fname: str
    ) -> pd.DataFrame:
    """Save transactions to a Pickle file in the ./data/ directory."""
    df = pd.DataFrame(transactions)

    # Ensure output directory exists
    fpath = Path("data") / fname
    fpath.parent.mkdir(parents=True, exist_ok=True)

    df.to_pickle(fpath)

    return df


async def main(
        username: str, 
        password: str, 
        start_date: str, 
        end_date: Optional[str] = None,
        filename: Optional[str] = None
    ):
    print(f"Using monarchmoney version {monarchmoney.__version__}")

    mm = await login_to_monarch(username, password)
    transactions = await fetch_transactions_from_monarch(mm, start_date=start_date, end_date=end_date)

    if transactions:
        save_transactions_to_pkl(transactions, filename)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download transactions from Monarch Money and save to pkl."
    )

    parser.add_argument(
        "--username",
        required=True,
        help="Monarch Money username or email address"
    )

    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date for transaction download (format: YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        required=False,
        default=None,
        help="End date for transaction download (format: YYYY-MM-DD, optional; defaults to today)"
    )

    parser.add_argument(
        "--filename",
        required=False,
        default="transactions.pkl",
        help="Output filename (e.g., transactions.pkl)"
    )

    args = parser.parse_args()

    # Prompt for password securely (not echoed in terminal)
    password = getpass.getpass(prompt="Monarch Money password: ")

    asyncio.run(
        main(
            args.username, 
            password, 
            args.start_date, 
            args.end_date, 
            args.filename
        )
    )