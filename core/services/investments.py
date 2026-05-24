"""Vanguard CSV parser and Firestore helpers for investment data.

Two paths are used under users/{uid}/:
  investments/holdings               — single document, overwritten on each
                                       upload; holds the current holdings snapshot.
  investment_transactions/{doc_id}   — transaction history from the CSV's
                                       second section; upserted by deterministic
                                       ID so re-uploading the same export is
                                       idempotent.

Account type metadata (IRA, Roth IRA, Taxable, …) lives in UserConfig's
investment_accounts field and is not inferred from the CSV.
"""

import csv
import hashlib
import logging
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from pydantic import BaseModel
from firebase_admin import firestore

from core.services.firebase import db, commit_in_batches

_log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class HoldingRecord(BaseModel):
    account_number: str
    investment_name: str
    symbol: str
    shares: float
    share_price: float
    total_value: float


class TransactionRecord(BaseModel):
    account_number: str
    trade_date: str           # YYYY-MM-DD
    settlement_date: str      # YYYY-MM-DD
    transaction_type: str
    transaction_description: str
    investment_name: str
    symbol: str               # empty string when not applicable (e.g. cash)
    shares: float
    share_price: float
    principal_amount: float
    commissions_and_fees: float
    net_amount: float
    accrued_interest: float
    account_type: str         # Vanguard settlement type — e.g. "CASH"


class InvestmentSnapshot(BaseModel):
    snapshot_date: str        # YYYY-MM-DD — used as Firestore doc ID
    uploaded_at: str          # ISO datetime string
    holdings: list[HoldingRecord]


# ─────────────────────────────────────────────
# CSV Parser
# ─────────────────────────────────────────────

_HOLDINGS_HEADER = (
    "Account Number,Investment Name,Symbol,Shares,Share Price,Total Value"
)
_TRANSACTIONS_HEADER = (
    "Account Number,Trade Date,Settlement Date,Transaction Type,"
    "Transaction Description,Investment Name,Symbol,Shares,Share Price,"
    "Principal Amount,Commissions and Fees,Net Amount,Accrued Interest,Account Type"
)


def _parse_float(val: str) -> float:
    v = val.strip().replace(',', '')
    return float(v) if v and v != '-' else 0.0


def _strip(line: str) -> str:
    """Strip trailing comma and whitespace that Vanguard adds to every row."""
    return line.rstrip(',').strip()


def parse_vanguard_csv(
    content: str,
) -> tuple[list[HoldingRecord], list[TransactionRecord]]:
    """Parse a Vanguard portfolio CSV export.

    The file contains two sections separated by a blank line:
      Section 1 — current holdings snapshot (one row per position)
      Section 2 — transaction history

    Returns (holdings, transactions).
    Raises ValueError if the expected headers are not found.
    """
    lines = content.splitlines()

    holdings_start: Optional[int] = None
    transactions_start: Optional[int] = None

    for i, line in enumerate(lines):
        stripped = _strip(line)
        if stripped == _HOLDINGS_HEADER:
            holdings_start = i
        elif stripped == _TRANSACTIONS_HEADER:
            transactions_start = i

    if holdings_start is None or transactions_start is None:
        missing = []
        if holdings_start is None:
            missing.append("holdings header")
        if transactions_start is None:
            missing.append("transactions header")
        raise ValueError(
            f"Vanguard CSV is missing expected headers: {', '.join(missing)}. "
            "Ensure you are uploading an unmodified Vanguard OFX CSV export."
        )

    # ── Holdings section ─────────────────────────────────────────────────────
    holdings_lines = [
        _strip(l)
        for l in lines[holdings_start:transactions_start]
        if _strip(l)
    ]
    holdings_reader = csv.DictReader(holdings_lines)

    holdings: list[HoldingRecord] = []
    for row in holdings_reader:
        acct = (row.get("Account Number") or "").strip()
        if not acct:
            continue
        acct = acct[-4:]
        holdings.append(HoldingRecord(
            account_number=acct,
            investment_name=(row.get("Investment Name") or "").strip(),
            symbol=(row.get("Symbol") or "").strip(),
            shares=_parse_float(row.get("Shares", "")),
            share_price=_parse_float(row.get("Share Price", "")),
            total_value=_parse_float(row.get("Total Value", "")),
        ))

    # ── Transactions section ─────────────────────────────────────────────────
    txn_lines = [
        _strip(l)
        for l in lines[transactions_start:]
        if _strip(l)
    ]
    txn_reader = csv.DictReader(txn_lines)

    transactions: list[TransactionRecord] = []
    for row in txn_reader:
        acct = (row.get("Account Number") or "").strip()
        if not acct:
            continue
        acct = acct[-4:]
        transactions.append(TransactionRecord(
            account_number=acct,
            trade_date=(row.get("Trade Date") or "").strip(),
            settlement_date=(row.get("Settlement Date") or "").strip(),
            transaction_type=(row.get("Transaction Type") or "").strip(),
            transaction_description=(row.get("Transaction Description") or "").strip(),
            investment_name=(row.get("Investment Name") or "").strip(),
            symbol=(row.get("Symbol") or "").strip(),
            shares=_parse_float(row.get("Shares", "")),
            share_price=_parse_float(row.get("Share Price", "")),
            principal_amount=_parse_float(row.get("Principal Amount", "")),
            commissions_and_fees=_parse_float(row.get("Commissions and Fees", "")),
            net_amount=_parse_float(row.get("Net Amount", "")),
            accrued_interest=_parse_float(row.get("Accrued Interest", "")),
            account_type=(row.get("Account Type") or "").strip(),
        ))

    return holdings, transactions


def _transaction_doc_id(txn: TransactionRecord) -> str:
    """Deterministic Firestore document ID for a transaction row.

    Built from the fields that together uniquely identify a transaction:
    account, date, type, symbol, and net amount in cents. A short SHA-1
    suffix handles the rare edge case of two identical rows.
    """
    cents = str(round(txn.net_amount * 100))
    key = (
        f"{txn.account_number}_{txn.trade_date}_{txn.transaction_type}"
        f"_{txn.symbol or 'CASH'}_{cents}"
    )
    suffix = hashlib.sha1(key.encode()).hexdigest()[:8]
    return f"{key}_{suffix}"


# ─────────────────────────────────────────────
# Firestore helpers
# ─────────────────────────────────────────────

def upsert_investment_snapshot(
    uid: str,
    snapshot_date: str,
    holdings: list[HoldingRecord],
) -> None:
    """Overwrite the single holdings snapshot document for uid.

    snapshot_date is YYYY-MM-DD and records when the CSV was exported.
    The document at users/{uid}/investments/holdings is always overwritten
    in full — no per-date history is kept here.
    """
    ref = (
        db.collection("users")
          .document(uid)
          .collection("investments")
          .document("holdings")
    )
    ref.set({
        "snapshot_date": snapshot_date,
        "uploaded_at": datetime.utcnow().isoformat(),
        "holdings": [h.model_dump() for h in holdings],
    })


def upsert_investment_transactions(
    uid: str,
    transactions: list[TransactionRecord],
) -> None:
    """Upsert investment transactions using deterministic document IDs.

    Re-uploading the same Vanguard export will overwrite existing records
    with identical data — no duplicates are created.
    """
    txn_ref = (
        db.collection("users")
          .document(uid)
          .collection("investment_transactions")
    )

    def _write(batch, txn: TransactionRecord):
        doc_id = _transaction_doc_id(txn)
        batch.set(txn_ref.document(doc_id), txn.model_dump())

    commit_in_batches(transactions, _write)


def fetch_latest_holdings(uid: str) -> list[dict]:
    """Return the holdings list from the single snapshot document, or []."""
    ref = (
        db.collection("users")
          .document(uid)
          .collection("investments")
          .document("holdings")
    )
    doc = ref.get()
    return doc.to_dict().get("holdings", []) if doc.exists else []


def fetch_investment_transactions(
    uid: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[dict]:
    """Return investment transactions sorted newest-first.

    start_date / end_date are inclusive YYYY-MM-DD strings that filter on
    trade_date.
    """
    ref = (
        db.collection("users")
          .document(uid)
          .collection("investment_transactions")
    )
    query = ref
    if start_date:
        query = query.where("trade_date", ">=", start_date)
    if end_date:
        query = query.where("trade_date", "<=", end_date)
    query = query.order_by("trade_date", direction=firestore.Query.DESCENDING)
    return [doc.to_dict() for doc in query.stream()]


def process_and_upload_vanguard_csv(
    uid: str,
    content: str,
    snapshot_date: Optional[str] = None,
) -> tuple[int, int]:
    """Parse a Vanguard CSV and persist holdings + transactions to Firestore.

    snapshot_date defaults to today's date.  Returns (n_holdings, n_transactions).
    """
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

    holdings, transactions = parse_vanguard_csv(content)
    upsert_investment_snapshot(uid, snapshot_date, holdings)
    upsert_investment_transactions(uid, transactions)
    return len(holdings), len(transactions)


def save_investment_account_config(
    uid: str,
    types: dict[str, str],
    nicknames: dict[str, str],
) -> None:
    """Persist investment_accounts and investment_account_nicknames to users/{uid}."""
    ref = db.collection("users").document(uid)
    ref.set(
        {"investment_accounts": types, "investment_account_nicknames": nicknames},
        merge=True,
    )


def delete_investment_data(uid: str) -> tuple[int, int]:
    """Delete all investment data for uid from Firestore.

    Removes the holdings snapshot and investment_transactions subcollections,
    and clears the investment_accounts / investment_account_nicknames config
    fields (their keys are full account numbers).

    Returns (n_holdings_docs, n_transaction_docs) deleted.
    """
    user_ref = db.collection("users").document(uid)

    holdings_docs = list(user_ref.collection("investments").stream())
    txn_docs = list(user_ref.collection("investment_transactions").stream())

    def _delete(batch, doc):
        batch.delete(doc.reference)

    commit_in_batches(holdings_docs, _delete)
    commit_in_batches(txn_docs, _delete)

    if user_ref.get().exists:
        user_ref.update({
            "investment_accounts": firestore.DELETE_FIELD,
            "investment_account_nicknames": firestore.DELETE_FIELD,
        })

    return len(holdings_docs), len(txn_docs)


# ─────────────────────────────────────────────
# Asset class lookup
# ─────────────────────────────────────────────

ASSET_CLASS_MAP: dict[str, str] = {
    "VTSAX": "US Equity", "VEXAX": "US Equity", "VTI": "US Equity",
    "VFTAX": "US Equity", "VASVX": "US Equity", "VHCAX": "US Equity",
    "VHCOX": "US Equity", "ICLN": "US Equity",
    "VEU": "Intl Equity", "VSS": "Intl Equity",
    "VBTLX": "Bonds", "VBIRX": "Bonds",
    "VMFXX": "Cash", "VMRXX": "Cash",
}

MONEY_MARKET_SYMBOLS: frozenset[str] = frozenset({"VMFXX", "VMRXX"})

_CONTRIBUTION_TYPES: frozenset[str] = frozenset({"Buy"})


# ─────────────────────────────────────────────
# Analytics helpers
# ─────────────────────────────────────────────

def compute_ytd_contributions(transactions: list[dict], year: int) -> float:
    """Sum of cash invested via Buy transactions for the given year (excludes reinvestments)."""
    total = 0.0
    for txn in transactions:
        if txn.get("trade_date", "")[:4] == str(year):
            if txn.get("transaction_type") in _CONTRIBUTION_TYPES:
                total += abs(txn.get("net_amount", 0.0))
    return total


def reconstruct_portfolio_history(
    uid: str,
    investment_accounts: dict[str, str],
) -> pd.DataFrame:
    """Reconstruct daily portfolio value split by retirement vs non-retirement.

    Algorithm:
    1. Anchor on current holdings (known share counts today).
    2. For each (account, symbol), un-apply transaction share deltas backward
       to derive historical share counts at every date.
    3. Multiply daily share count × yfinance daily close price.
    4. Sum across all positions, splitting by account type.

    Symbols absent from current holdings (e.g. VHCOX after share-class
    conversion) are handled automatically: their current_count defaults to 0
    and the backward formula still yields the correct forward history.

    Returns DataFrame with DatetimeIndex and columns:
        retirement_value, taxable_value, total_value
    Returns an empty DataFrame (same columns) when no data is available.
    """
    _EMPTY = pd.DataFrame(
        columns=["retirement_value", "taxable_value", "total_value"]
    )
    _RETIREMENT_TYPES = frozenset({
        "IRA", "Roth IRA", "401k", "Roth 401k", "403b", "457b", "SEP IRA", "SIMPLE IRA",
    })

    holdings = fetch_latest_holdings(uid)
    transactions = fetch_investment_transactions(uid)

    if not holdings and not transactions:
        return _EMPTY

    # ── Current share counts per (account, symbol) ──────────────────────────
    current_shares: dict[tuple[str, str], float] = {}
    for h in holdings:
        sym = h.get("symbol", "")
        acct = h.get("account_number", "")
        if sym and acct:
            current_shares[(acct, sym)] = float(h.get("shares", 0.0))

    # ── Build share delta series per (account, symbol) ───────────────────────
    all_acct_syms: set[tuple[str, str]] = set(current_shares.keys())
    share_deltas: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    txn_dates: list[str] = []

    for txn in transactions:
        sym = txn.get("symbol", "")
        shares = float(txn.get("shares", 0.0))
        acct = txn.get("account_number", "")
        d = txn.get("trade_date", "")
        if d:
            txn_dates.append(d)
        if sym and acct and shares != 0:
            all_acct_syms.add((acct, sym))
            share_deltas[(acct, sym)][d] += shares

    if not txn_dates:
        return _EMPTY

    min_date = min(txn_dates)
    end_date = date.today()
    date_range = pd.date_range(min_date, end_date, freq="D")

    # ── Fetch closing prices from yfinance ───────────────────────────────────
    all_symbols: set[str] = {sym for (_, sym) in all_acct_syms}
    price_symbols = sorted(all_symbols - set(MONEY_MARKET_SYMBOLS))
    prices: dict[str, pd.Series] = {}

    if price_symbols:
        try:
            raw = yf.download(
                price_symbols,
                start=min_date,
                end=(end_date + timedelta(days=1)).isoformat(),
                progress=False,
                auto_adjust=True,
            )
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    close_df = raw["Close"]
                else:
                    close_df = raw[["Close"]].rename(
                        columns={"Close": price_symbols[0]}
                    )
                for sym in price_symbols:
                    if sym in close_df.columns:
                        s = close_df[sym].copy()
                        if getattr(s.index, "tz", None) is not None:
                            s.index = s.index.tz_localize(None)
                        prices[sym] = s.reindex(date_range).ffill()
        except Exception as exc:
            _log.warning("yfinance download failed: %s", exc)

    # Money market funds: stable $1 NAV
    for sym in MONEY_MARKET_SYMBOLS & all_symbols:
        prices[sym] = pd.Series(1.0, index=date_range)

    # ── Reconstruct daily values by account type ─────────────────────────────
    retirement_daily = pd.Series(0.0, index=date_range)
    taxable_daily = pd.Series(0.0, index=date_range)

    for (acct, sym) in all_acct_syms:
        if sym not in prices:
            continue

        price_s = prices[sym].fillna(0.0)
        current_count = current_shares.get((acct, sym), 0.0)
        deltas_dict = dict(share_deltas.get((acct, sym), {}))

        # Align transaction deltas to the date_range index
        full_deltas = pd.Series(0.0, index=date_range)
        for d_str, delta in deltas_dict.items():
            ts = pd.Timestamp(d_str)
            if ts in full_deltas.index:
                full_deltas[ts] += delta

        # share_count at close on date d = current_count − sum(deltas after d)
        # future_cumsum[d] = sum(deltas from d to end, inclusive)
        # shares gained strictly after d = future_cumsum[d] − full_deltas[d]
        future_cumsum = full_deltas.iloc[::-1].cumsum().iloc[::-1]
        share_counts = (current_count - future_cumsum + full_deltas).clip(lower=0)
        daily_value = share_counts * price_s

        acct_type = investment_accounts.get(acct, "Taxable")
        if acct_type in _RETIREMENT_TYPES:
            retirement_daily += daily_value
        else:
            taxable_daily += daily_value

    result = pd.DataFrame(
        {"retirement_value": retirement_daily, "taxable_value": taxable_daily},
        index=date_range,
    )
    result["total_value"] = result["retirement_value"] + result["taxable_value"]
    result.index.name = "date"
    return result
