"""Budget logic: reading the per-month budget dict, CSP-plan resolution,
generating a rolling budget from a CSP plan, and the actual-vs-budget report.

Moved verbatim from core/utils/functions.py (Phase 3 of the core/
restructuring); pinned by tests/test_budget.py.
"""
from datetime import datetime as dt

import numpy as np
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta


def read_budget(config, user): # TODO read from dedicated session store in Dash?
    budget_dict = config['users'][user]['budget']

    budget = pd.DataFrame({
        (int(year), int(month)): values
        for year, months in budget_dict.items()
        for month, values in months.items()
    })

    return budget


def get_monthly_loading(budget: pd.DataFrame, category: str, period_months: list) -> list:
    """Normalize a category's budget values over period_months into weights summing to 1."""
    vals = []
    for year, month in period_months:
        col = (year, month)
        if col in budget.columns and category in budget.index:
            val = budget.loc[category, col]
            vals.append(0.0 if pd.isna(val) else float(val))
        else:
            vals.append(0.0)
    total = sum(vals)
    return [v / total for v in vals] if total > 0 else [1 / 12] * 12


def get_active_csp_plan(csp_plans: dict, as_of: str | None = None) -> dict:
    """Return the most recent plan with effective date <= as_of (ISO YYYY-MM-DD, defaults to today)."""
    if not csp_plans:
        return {}
    if as_of is None:
        as_of = dt.today().strftime('%Y-%m-%d')
    eligible = {k: v for k, v in csp_plans.items() if k <= as_of}
    return eligible[max(eligible)] if eligible else {}


def generate_budget_from_csp(
    csp_plan: dict,
    budget: pd.DataFrame,
    categories: list,
    window_start: dt,
) -> dict:
    """Generate a 12-month rolling budget from CSP monthly amounts.

    Uses the trailing 12-month budget shape (loading) to distribute CSP annual
    amounts across the forward 12-month window starting at window_start.

    Returns {str(year): {str(month): {category: amount}}}.
    """
    trailing_start = window_start.replace(day=1) - relativedelta(months=12)
    trailing_months = []
    d = trailing_start
    for _ in range(12):
        trailing_months.append((d.year, d.month))
        d += relativedelta(months=1)

    forward_months = []
    d = window_start.replace(day=1)
    for _ in range(12):
        forward_months.append((d.year, d.month))
        d += relativedelta(months=1)

    result = {}
    for category in categories:
        loading = get_monthly_loading(budget, category, trailing_months)
        annual = float(csp_plan.get(category, 0.0)) * 12
        for i, (year, month) in enumerate(forward_months):
            result.setdefault(str(year), {}).setdefault(str(month), {})[category] = round(annual * loading[i], 2)

    return result


def calc_proportions(df):
    # calculate overage
    filt = df['amount'] > df['budget']
    df.loc[filt, 'overage'] = (
        df.loc[filt, 'amount'] - df.loc[filt, 'budget']
    )
    df['overage'] = df['overage'].fillna(0)

    # calculate covered amount
    df['covered'] = df['amount'] - df['overage']

    # calculate remaining
    df['remaining'] = df['budget'] - df['amount']

    # calculate proportion and fill na with 100%
    df['proportion'] = df['covered'] / df['budget']
    df['proportion'] = df['proportion'].fillna(1)

    return df


def build_budget_report(transactions, budget, start_date, end_date, config, user):
    # Ensure start_date and end_date are timezone-aware in UTC
    utc = pytz.UTC
    start_date = start_date.replace(tzinfo=utc)
    end_date = end_date.replace(tzinfo=utc)
    # Subset transactions by date
    filt = (
        (transactions['date'] >= start_date) &
        (transactions['date'] <= end_date) &
        (transactions['account_owner'] == user)
    )
    transactions = transactions.loc[filt, :]

    # Sum spending for period
    spend = transactions.groupby(['csp', 'csp_label'])['amount'].sum().abs()
    spend = spend.reset_index()

    # Sum budget for period
    period_budget = budget.loc[:, (start_date.year, start_date.month):(end_date.year, end_date.month)]
    total_budget = period_budget.sum(axis=1)
    total_budget = total_budget[total_budget>0]
    total_budget.name = 'budget'

    # Merge spending with budget
    df = pd.merge(spend, total_budget, left_on='csp', right_index=True, how='outer')

    # NA values are actually 0
    df = df.fillna(0)

    # Calculate total spending
    sum_row = df.loc[df['csp_label'] != 'income'].sum(numeric_only=True)

    # Add the label for the new row (e.g., 'Total Spending') in a non-numeric column
    sum_row['csp'] = 'Total Spending'
    sum_row['csp_label'] = 'spending'

    # Append the sum row to the original DataFrame
    df = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)

    # Calculate total income
    sum_row = df.loc[df['csp_label'] == 'income'].sum(numeric_only=True)

    # Add the label for the new row (e.g., 'Total Spending') in a non-numeric column
    sum_row['csp'] = 'Total Income'
    sum_row['csp_label'] = 'income'

    # Append the sum row to the original DataFrame
    df = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)

    # Calculate proportions by category
    df = calc_proportions(df)

    # Add CSP categories and sort to order
    df = df.set_index('csp')
    csp_groups = ['Income', 'Fixed Costs', 'Investments', 'Sinking', 'Guilt Free']
    new_rows = pd.DataFrame(np.nan, index=csp_groups, columns=df.columns)
    df = pd.concat([df, new_rows])

    # Get category order
    cat_order = pd.DataFrame(config['users'][user]['cat_order'])
    cat_order = cat_order.reset_index()
    cat_order.columns = ['order', 'category']

    # Merge with category orders
    df_ordered = pd.merge(df, cat_order, left_index=True,
                            right_on='category', how='left')

    # Sort by category order
    df_ordered.sort_values('order', inplace=True, ascending=False)

    # # Total spending by group
    # groups = df.groupby('csp_label').sum(numeric_only=True).reset_index()
    # groups = groups.set_index('csp_label').sort_index(ascending=False)

    # # Calculate proportions by group
    # groups = calc_proportions(groups)

    return df_ordered


def order_budget(budget, config, user):
    cat_order = pd.DataFrame(config["users"][user]['cat_order'])
    cat_order = cat_order.reset_index()
    cat_order.columns = ['order', 'category']

    # Merge with category orders
    budget = pd.merge(budget, cat_order, left_index=True,
                            right_on='category', how='left')

    # Sort by category order
    budget.sort_values('order', inplace=True, ascending=True)

    budget = budget.set_index('category')
    budget.index.name = "category"

    # budget.loc["Total"] = budget.drop('income').sum(numeric_only=True)

    budget = budget.drop(columns=["order"])
    budget = budget.reset_index()

    return budget
