"""Pure holdings math: tax-bucket aggregation and cost-basis gain fraction.

Moved from core/services/retirement.py (Phase 4 of the core/ restructuring) —
`buckets_from_holdings` is the pure core of the old `balances_by_tax_bucket`
(the Firestore-reading wrapper now lives in core/services/investments.py);
`gain_fraction_from_cost_basis` moved verbatim. Pinned by
tests/test_retirement_math.py.

FUTURE HOME (Phase 6): ASSET_CLASS_MAP and the pure Vanguard CSV parsing from
core/services/investments.py.
"""

from __future__ import annotations

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
