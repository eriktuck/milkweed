"""Firestore adapters for investment data (holdings, transactions, cost basis).

The pure Vanguard CSV parsing, record models, and asset-class constants live in
core/domain/holdings.py (moved there in Phase 6 of the core/ restructuring);
this module keeps the Firestore CRUD, the thin fetch+domain compositions, and
the yfinance-backed portfolio-history reconstruction.

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

import hashlib
import logging
from collections import defaultdict
from datetime import datetime, date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from firebase_admin import firestore

from core.domain.holdings import (
    MONEY_MARKET_SYMBOLS,
    CostBasisRecord,
    HoldingRecord,
    TransactionRecord,
    buckets_from_holdings,
    gain_fraction_from_cost_basis,
    parse_vanguard_cost_basis_csv,
    parse_vanguard_csv,
)
from core.services.firebase import db, commit_in_batches

_log = logging.getLogger(__name__)


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


def current_portfolio_value(uid: str) -> float:
    """Total current portfolio value = Σ holdings.total_value for the user.

    Thin fetch+sum wrapper (moved from core/services/forecast.py, Phase 5).
    """
    return sum(h.get("total_value", 0.0) for h in fetch_latest_holdings(uid))


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


def upsert_cost_basis(
    uid: str,
    snapshot_date: str,
    records: list[CostBasisRecord],
) -> None:
    """Merge per-account cost-basis records into the single cost_basis document.

    Cost-basis reports are exported one account at a time, so each upload merges
    its accounts into users/{uid}/investments/cost_basis rather than overwriting
    — uploading account A then account B accumulates both. Re-uploading an
    account overwrites just that account's entry with the fresh figures.
    """
    ref = (
        db.collection("users")
          .document(uid)
          .collection("investments")
          .document("cost_basis")
    )
    existing = ref.get()
    accounts: dict = existing.to_dict().get("accounts", {}) if existing.exists else {}
    for rec in records:
        accounts[rec.account_number] = {
            "market_value": rec.market_value,
            "cost": rec.cost,
            "gain": rec.gain,
        }
    ref.set({
        "snapshot_date": snapshot_date,
        "uploaded_at": datetime.utcnow().isoformat(),
        "accounts": accounts,
    })


def fetch_cost_basis(uid: str) -> dict:
    """Return the {account: {market_value, cost, gain}} map, or {}."""
    ref = (
        db.collection("users")
          .document(uid)
          .collection("investments")
          .document("cost_basis")
    )
    doc = ref.get()
    return doc.to_dict().get("accounts", {}) if doc.exists else {}


def balances_by_tax_bucket(uid: str, investment_accounts: dict | None) -> dict:
    """Aggregate current holdings into {taxable, trad, roth, total}.

    Thin fetch+domain wrapper (moved from core/services/retirement.py, Phase 4):
    reads the holdings snapshot then delegates to the pure
    `core.domain.holdings.buckets_from_holdings` — kept at the edge so the
    projection math stays pure and testable.
    """
    return buckets_from_holdings(fetch_latest_holdings(uid), investment_accounts)


def taxable_gain_fraction(uid: str, investment_accounts: dict | None) -> float | None:
    """Derive the taxable gain fraction from the user's uploaded cost-basis report.

    Thin fetch+domain wrapper (moved from core/services/retirement.py, Phase 4)
    around the pure `core.domain.holdings.gain_fraction_from_cost_basis`.
    Returns None if the user has not uploaded a cost-basis CSV (or none of it is
    in taxable accounts), so the income pane falls back to the editable
    drawdown.TAXABLE_GAIN_FRACTION default.
    """
    return gain_fraction_from_cost_basis(fetch_cost_basis(uid), investment_accounts)


def process_and_upload_cost_basis_csv(
    uid: str,
    content: str,
    snapshot_date: Optional[str] = None,
) -> int:
    """Parse a Vanguard cost-basis CSV and merge it into Firestore.

    snapshot_date defaults to today's date. Returns the number of accounts in
    this upload.
    """
    if snapshot_date is None:
        snapshot_date = date.today().isoformat()

    records = parse_vanguard_cost_basis_csv(content)
    upsert_cost_basis(uid, snapshot_date, records)
    return len(records)


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

    Removes the investments subcollection (holdings + cost_basis snapshots) and
    the investment_transactions subcollection, and clears the
    investment_accounts / investment_account_nicknames config fields (their keys
    are full account numbers).

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
# Analytics helpers
# ─────────────────────────────────────────────

_CONTRIBUTION_TYPES: frozenset[str] = frozenset({"Buy"})

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
