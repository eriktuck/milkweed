import pandas as pd
import monarchmoney
from monarchmoney import MonarchMoney
from pathlib import Path
from datetime import date, datetime
from typing import Any, Optional, List, Tuple
import pickle
import base64
from dotenv import load_dotenv
import os
import asyncio

def pickle_and_encode(obj):
    pickled = pickle.dumps(obj)
    encoded = base64.b64encode(pickled).decode('utf-8')
    return encoded


def decode_and_unpickle(encoded_str):
    decoded = base64.b64decode(encoded_str)
    obj = pickle.loads(decoded)
    return obj


async def login_to_monarch(email: str, password: str) -> MonarchMoney:
    """
    Login to Monarch Money account.

    Parameters
    ----------
    email : str
        The user's email address.
    password : str
        The user's password.

    Returns
    -------
    MonarchMoney
        An authenticated MonarchMoney instance.
    """
    mm = MonarchMoney()
    
    # Get device UUID
    env_path_str = os.getenv('ENV_PATH', './secrets/env-file')
    env_path = Path(env_path_str)
    load_dotenv(dotenv_path=env_path)

    device_uuid = os.environ.get('MILKWEED_DEVICE_UUID')
    mm._headers['Device-UUID'] = device_uuid

    await mm.login(
        email=email, 
        password=password, 
        use_saved_session=False, 
        save_session=False
    )
    return mm


def validate_date(date_str: str) -> date:
    """Validate and parse a date string in 'YYYY-MM-DD' format."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format: '{date_str}'. Expected 'YYYY-MM-DD'.")


def chunk_date_range(
        start_date: date, 
        end_date: date, 
        max_days: int = 365
    ) -> List[Tuple[date, date]]:
    """Chunk a date range into smaller ranges of up to max_days each."""
    chunks = []
    current_start = start_date
    while current_start <= end_date:
        current_end = min(current_start + pd.Timedelta(days=max_days - 1), end_date)
        chunks.append((current_start, current_end))
        current_start = current_end + pd.Timedelta(days=1)
    return chunks


async def fetch_transactions_from_monarch(
        mm: MonarchMoney,
        start_date: str, 
        end_date: Optional[str] = None,
        max_days: int = 365
    ) -> Optional[list[dict[str, Any]]]:
    """
    Download transactions from Monarch Money within the specified date range.

    Chunk any periods longer than max_days into smaller requests.

    Parameters
    ----------
    mm : MonarchMoney
        An authenticated MonarchMoney instance.
    start_date : str
        The start date for transaction download (format: 'YYYY-MM-DD').
    end_date : Optional[str], optional
        The end date for transaction download (format: 'YYYY-MM-DD'). 
        If not provided, defaults to today's date.
    max_days: int, optional
        Maximum number of days per chunk. Default is 365.
    
    Returns
    -------
    Optional[list[dict[str, Any]]]
        A list of transaction dictionaries, or None if no transactions were found
        or an error occurred. 
        Expected API return structure (only "results" is returned by the function):
        {
            "allTransactions": {
                "totalCount": int,
                "results": list[dict[str, Any]]
            },
            "transactionRules": list[dict[str, Any]]
        }

    Notes
    -----
    - The Monarch Money API will fail silently (returning no transactions) if the date range
    includes too many transactions, likely due to rate limiting or timeouts.
    """
    # Validate and normalize start_date
    start_date_obj = validate_date(start_date)

    # Use today’s date if end_date not provided
    if end_date is None:
        end_date_obj = date.today()
    else:
        end_date_obj = validate_date(end_date)

    # Ensure start_date <= end_date
    if start_date_obj > end_date_obj:
        raise ValueError(f"start_date ({start_date_obj}) cannot be after end_date ({end_date_obj}).")
    
    # Break into subranges
    chunks = chunk_date_range(start_date_obj, end_date_obj, max_days=max_days)
    print(f"Downloading {len(chunks)} chunks covering {start_date_obj} → {end_date_obj}")

    all_results: List[dict[str, Any]] = []

    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        chunk_start_str= chunk_start.strftime("%Y-%m-%d")
        chunk_end_str = chunk_end.strftime("%Y-%m-%d")
        print(f"  Chunk {i}: {chunk_start_str} → {chunk_end_str}")
        
        try:
            chunk_transactions = await mm.get_transactions(
                start_date=chunk_start_str, 
                end_date=chunk_end_str, 
                limit=None
            )

            if not chunk_transactions:
                print(f"  No transactions returned for chunk {i} (possible rate limit or timeout).")
                continue
            
            all_results.extend(chunk_transactions.get("allTransactions", {}).get("results", []))
    
        except Exception as e:
            print(f"  Error downloading chunk {i}: {str(e)}")
    
    print(f"Downloaded {len(chunks)} chunks. Total transactions: {len(all_results)}")

    return all_results
