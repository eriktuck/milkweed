"""Legacy re-export shim — the grab-bag was split in Phase 3 of the core/
restructuring:

- transaction pipeline      → core/domain/transactions.py
- budget / CSP-plan logic   → core/domain/budget.py
- Social Security helpers   → core/domain/social_security.py
- tax math                  → core/domain/taxes.py
- Plotly builders + table   → core/charts/reports.py

This module survives only because core/models/ (the frozen Engine A reference)
and the notebooks import it; it is deleted together with core/models/ once
those capabilities are ported. New code must import from the new homes.
"""
import pandas as pd

from core.charts.reports import (
    format_table,
    plot_csp_by_label,
    plot_report,
    plot_spending_trends,
)
from core.domain.budget import (
    build_budget_report,
    calc_proportions,
    generate_budget_from_csp,
    get_active_csp_plan,
    get_monthly_loading,
    order_budget,
    read_budget,
)
from core.domain.social_security import (
    _money_to_float,
    calculate_social_security_benefit,
    segments_to_annual_income,
    ssa_earnings_to_segments,
)
from core.domain.taxes import calculate_married_joint_tax
from core.domain.transactions import (
    categorize_transactions,
    check_processed,
    convert_raw_transactions_to_dataframe,
    filter_and_process_owner_transactions,
    filter_owner_transactions,
    preprocess_transactions,
    process_and_attribute_transactions,
)


def load_vanguard_cost_basis(csv_path: str) -> pd.DataFrame:
    """
    Load Vanguard cost basis CSV export and extract initial cost basis info.

    Parameters:
    - csv_path: str, path to Vanguard CSV export

    Returns:
    - DataFrame with columns: account, symbol, cost_basis

    Notebook-only helper (reads from disk, so it has no domain/ home); the
    app's Vanguard parsing lives in core/services/investments.py and moves to
    core/domain/holdings.py in Phase 6.
    """
    # Read CSV with appropriate options
    df = pd.read_csv(csv_path)

    # Clean up column names (remove extra whitespace)
    df.columns = df.columns.str.strip()

    # Convert cost values to numeric (strip $ and commas if needed)
    df['Total cost'] = pd.to_numeric(df['Total cost'], errors='coerce')

    # Standardize column names and output
    clean_df = df.rename(columns={
        'Account': 'account',
        'Symbol/CUSIP': 'symbol',
        'Total cost': 'cost_basis'
    })

    # Select and aggregate
    cost_basis_df = clean_df[['account', 'symbol', 'cost_basis']] \
        .groupby(['account', 'symbol'], as_index=False) \
        .sum()

    return cost_basis_df
