"""Pure holdings logic: tax-bucket aggregation, cost-basis gain fraction,
asset-class classification, and Vanguard CSV parsing (text → records).

`buckets_from_holdings` / `gain_fraction_from_cost_basis` moved from
core/services/retirement.py in Phase 4 (pinned by tests/test_retirement_math.py);
ASSET_CLASS_MAP, MONEY_MARKET_SYMBOLS, the record models, and the Vanguard
parsers moved verbatim from core/services/investments.py in Phase 6. The
Firestore-reading wrappers stay in core/services/investments.py.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel

# Account-type labels (from investment_accounts) that are tax-deferred /
# tax-free, mirroring investments.reconstruct_portfolio_history.
TRAD_TYPES: frozenset[str] = frozenset(
    {"IRA", "401k", "403b", "457b", "SEP IRA", "SIMPLE IRA"}
)
ROTH_TYPES: frozenset[str] = frozenset({"Roth IRA", "Roth 401k"})


def buckets_from_holdings(holdings: list[dict], investment_accounts: dict | None) -> dict:
    """Aggregate a holdings snapshot into {taxable, trad, roth, total}.  (pure)

    Splits the holdings list by the account-type label in investment_accounts
    (last-4 → label). Unlabeled accounts default to taxable, matching
    reconstruct_portfolio_history. `holdings` is the list of holding dicts from
    the snapshot document (each with `account_number` and `total_value`).
    """
    accounts = investment_accounts or {}
    buckets = {"taxable": 0.0, "trad": 0.0, "roth": 0.0}
    for h in holdings:
        acct = h.get("account_number", "")
        value = float(h.get("total_value", 0.0))
        label = accounts.get(acct, "Taxable")
        if label in TRAD_TYPES:
            buckets["trad"] += value
        elif label in ROTH_TYPES:
            buckets["roth"] += value
        else:
            buckets["taxable"] += value
    buckets["total"] = sum(buckets.values())
    return buckets


def gain_fraction_from_cost_basis(
    cost_basis: dict, investment_accounts: dict | None
) -> float | None:
    """Realized-gain fraction implied by a cost-basis report.  (pure)

    (Σ market value − Σ cost) ÷ Σ market value over **taxable-labeled accounts
    only** — the share of a taxable sale that is long-term gain (taxed at LTCG);
    the rest is return of basis (tax-free). Tax-deferred accounts (TRAD_TYPES)
    are ordinary regardless of basis and Roth (ROTH_TYPES) is tax-free, so both
    are excluded. Unlabeled accounts default to taxable, matching
    buckets_from_holdings. Returns None when there is no taxable cost-basis data
    (the caller then keeps the manual drawdown.TAXABLE_GAIN_FRACTION default).

    `cost_basis` is the {account: {market_value, cost, gain}} map from
    fetch_cost_basis.
    """
    accounts = investment_accounts or {}
    tot_mv = tot_cost = 0.0
    for acct, vals in (cost_basis or {}).items():
        label = accounts.get(acct, "Taxable")
        if label in TRAD_TYPES or label in ROTH_TYPES:
            continue
        tot_mv += float(vals.get("market_value", 0.0))
        tot_cost += float(vals.get("cost", 0.0))
    if tot_mv <= 0:
        return None
    return (tot_mv - tot_cost) / tot_mv


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


# ─────────────────────────────────────────────
# Record models
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


class CostBasisRecord(BaseModel):
    """Per-account roll-up of a Vanguard *cost-basis* (unrealized gain/loss)
    export — distinct from the portfolio CSV. One record per account; lots are
    summed at parse time since the taxable-gain fraction only needs aggregate
    market value and cost.
    """
    account_number: str       # last 4 digits, matching HoldingRecord
    market_value: float
    cost: float
    gain: float               # market_value − cost


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


# ── Cost-basis (unrealized gain/loss) export ─────────────────────────────────────
#
# A separate Vanguard download from the portfolio CSV, exported **per account**
# ("Cost basis" → "Unrealized gains & losses" → Download). One row per tax lot.
# The header carries dynamic timestamps and a comma *inside* a quoted field name
# ("Market value as of 06/05/2026 04:00 PM, ET"), so it cannot be matched as a
# raw string like the portfolio headers — it must be CSV-parsed and located by
# field prefix. The first two lines are disclaimer text; the header row is the
# first whose leading cell is "Account".

_COST_BASIS_COST_COL = "Total cost"
_COST_BASIS_MV_PREFIX = "Market value"   # full header includes an export timestamp


def _is_cost_basis_header(row: list[str]) -> bool:
    return bool(row) and row[0].strip() == "Account" and any(
        c.strip() == _COST_BASIS_COST_COL for c in row
    )


def _is_portfolio_header(row: list[str]) -> bool:
    return bool(row) and row[0].strip() == "Account Number" and any(
        c.strip() == "Total Value" for c in row
    )


def detect_vanguard_csv_type(content: str) -> str:
    """Classify a Vanguard export as 'portfolio' | 'cost_basis' | 'unknown'.

    Lets a single upload affordance accept either the portfolio CSV (holdings +
    transactions) or a per-account cost-basis CSV and route accordingly.
    """
    for row in csv.reader(content.splitlines()):
        if _is_cost_basis_header(row):
            return "cost_basis"
        if _is_portfolio_header(row):
            return "portfolio"
    return "unknown"


def parse_vanguard_cost_basis_csv(content: str) -> list[CostBasisRecord]:
    """Parse a Vanguard cost-basis CSV, rolled up to one record per account.

    Sums each account's lots into total market value and total cost (gain is the
    difference). Returns one CostBasisRecord per account found.
    Raises ValueError if the cost-basis header is not present.
    """
    rows = list(csv.reader(content.splitlines()))

    header_idx = next((i for i, r in enumerate(rows) if _is_cost_basis_header(r)), None)
    if header_idx is None:
        raise ValueError(
            "Vanguard cost-basis CSV is missing its header row. Ensure you are "
            "uploading an unmodified 'Unrealized gains & losses' export."
        )

    header = [c.strip() for c in rows[header_idx]]
    acct_i = header.index("Account")
    cost_i = header.index(_COST_BASIS_COST_COL)
    mv_i = next(i for i, h in enumerate(header) if h.startswith(_COST_BASIS_MV_PREFIX))

    agg: dict[str, dict[str, float]] = defaultdict(lambda: {"market_value": 0.0, "cost": 0.0})
    for row in rows[header_idx + 1:]:
        if len(row) <= max(acct_i, cost_i, mv_i):
            continue
        acct = (row[acct_i] or "").strip()
        if not acct:
            continue
        acct = acct[-4:]
        agg[acct]["market_value"] += _parse_float(row[mv_i])
        agg[acct]["cost"] += _parse_float(row[cost_i])

    return [
        CostBasisRecord(
            account_number=acct,
            market_value=vals["market_value"],
            cost=vals["cost"],
            gain=vals["market_value"] - vals["cost"],
        )
        for acct, vals in agg.items()
    ]
