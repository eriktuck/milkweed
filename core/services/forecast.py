"""Forecast math for the Forecast page (v1).

v1 is a fixed-rate, single-individual CoastFIRE projection with a flat 4% safe
withdrawal rate in retirement. It deliberately avoids the heavier core/models
machinery (RetirementScenario, Portfolio/yfinance, income/tax Streams) — those
are reserved for v2 (per-holding/stochastic returns, tax-aware drawdown, SS,
RMDs). Everything here is pure and unit-testable; the only Firestore touch is
the existing investments helper.

Phase math (real, inflation-adjusted dollars):
    working    (current_age → coast):      V' = V·(1+r) + annual_contribution
    coast      (coast → retirement):       V' = V·(1+r)
    retirement (retirement → horizon):     V' = V·(1+r) − swr·V_at_retirement
"""

from __future__ import annotations

import pandas as pd

from core.services.investments import fetch_latest_holdings

# Safe withdrawal rate — fixed in v1.
SWR = 0.04

# CSP labels (see CLAUDE.md): fixed · investments · sinking · guilt-free · income.
#
# Portfolio contributions are the `investments` label (the Monarch "Savings" csp
# key — money moved into long-term investment accounts). The `sinking` label is
# sinking funds for near/midterm goals (gifts/vacations/home improvements); those
# are recurring living expenses that persist in retirement, so they count toward
# retirement spend, NOT the contribution lever.
CONTRIBUTION_LABELS: frozenset[str] = frozenset({"investments"})
RETIREMENT_SPEND_LABELS: frozenset[str] = frozenset({"fixed", "guilt-free", "sinking"})


# ── Data-derived inputs ────────────────────────────────────────────────────────

def current_portfolio_value(uid: str) -> float:
    """Total current portfolio value = Σ holdings.total_value for the user."""
    return sum(h.get("total_value", 0.0) for h in fetch_latest_holdings(uid))


def _sum_plan_for_labels(
    active_plan: dict | None,
    csp_labels: dict | None,
    labels: frozenset[str],
) -> float:
    """Sum monthly CSP-plan amounts whose category maps to one of `labels`."""
    if not active_plan or not csp_labels:
        return 0.0
    return sum(
        float(amount)
        for category, amount in active_plan.items()
        if csp_labels.get(category) in labels
    )


def default_monthly_contribution(
    active_plan: dict | None, csp_labels: dict | None
) -> float:
    """Default monthly portfolio contribution = Σ monthly CSP plan for the
    `investments` label (long-term money moved into investment accounts)."""
    return _sum_plan_for_labels(active_plan, csp_labels, CONTRIBUTION_LABELS)


def default_annual_retirement_spend(
    active_plan: dict | None, csp_labels: dict | None
) -> float:
    """Default annual retirement spend = 12 × Σ monthly recurring living expenses.

    Recurring expenses are the `fixed`, `guilt-free`, and `sinking` labels —
    the last being sinking funds (gifts/vacations/home improvements) that persist
    in retirement. Portfolio contributions (`investments`) stop at retirement and
    are excluded.
    """
    return 12.0 * _sum_plan_for_labels(active_plan, csp_labels, RETIREMENT_SPEND_LABELS)


def trailing_12mo_contribution(txn_df: pd.DataFrame | None) -> float:
    """Average monthly actual contribution over the trailing 12 months.

    Sums the magnitude of `investments`-labeled transactions in the 12 months
    preceding the most recent transaction date, divided by 12, so the hint
    reflects money actually invested. Returns 0.0 when no usable data.
    """
    if txn_df is None or len(txn_df) == 0:
        return 0.0
    if "csp_label" not in txn_df.columns or "amount" not in txn_df.columns or "date" not in txn_df.columns:
        return 0.0

    df = txn_df[["date", "csp_label", "amount"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return 0.0

    cutoff = df["date"].max() - pd.DateOffset(months=12)
    recent = df[(df["date"] > cutoff) & (df["csp_label"].isin(CONTRIBUTION_LABELS))]
    if recent.empty:
        return 0.0

    return float(recent["amount"].abs().sum()) / 12.0


# ── Core projection ────────────────────────────────────────────────────────────

def project_portfolio(
    start_value: float,
    current_age: int,
    coast_age: int,
    retirement_age: int,
    horizon_age: int,
    annual_contribution: float,
    real_return: float,
    swr: float = SWR,
) -> pd.DataFrame:
    """Year-by-year fixed-rate projection across the three phases.

    Returns a DataFrame indexed by `age` with columns:
        total, principal, growth, contribution, withdrawal,
        sustainable_income, phase

    `principal` is money put in (starting balance + cumulative contributions),
    held flat after the coast point and clipped not to exceed `total`;
    `growth` = total − principal. These feed both the stacked area and the
    donut so the two charts always agree. The retirement draw is a flat
    `swr × (value at retirement_age)`, applied from the year after retirement.
    """
    current_age = int(current_age)
    coast_age = int(coast_age)
    retirement_age = int(retirement_age)
    horizon_age = int(horizon_age)
    r = float(real_return)
    contrib = float(annual_contribution)

    total = float(start_value)
    principal = float(start_value)
    retirement_draw = 0.0

    rows = []
    for age in range(current_age, horizon_age + 1):
        if age > current_age:
            prev = age - 1
            year_contrib = contrib if prev < coast_age else 0.0
            year_withdrawal = retirement_draw if prev >= retirement_age else 0.0
            total = total * (1 + r) + year_contrib - year_withdrawal
            principal += year_contrib

        # Lock the retirement draw at the value reached entering retirement.
        if age == retirement_age:
            retirement_draw = swr * total

        if age < coast_age:
            phase = "working"
        elif age < retirement_age:
            phase = "coast"
        else:
            phase = "retirement"

        principal_disp = min(principal, total)
        rows.append({
            "age": age,
            "total": total,
            "principal": principal_disp,
            "growth": max(total - principal_disp, 0.0),
            "contribution": contrib if phase == "working" else 0.0,
            "withdrawal": retirement_draw if phase == "retirement" else 0.0,
            "sustainable_income": swr * total,
            "phase": phase,
        })

    return pd.DataFrame(rows).set_index("age")


def required_monthly_contribution(
    start_value: float,
    current_age: int,
    coast_age: int,
    coast_target: float,
    real_return: float,
) -> float:
    """Monthly contribution needed to reach `coast_target` exactly by `coast_age`.

    Closed-form inverse of `project_portfolio`'s working-phase recurrence
    (V' = V·(1+r) + C, applied once per year from current_age to coast_age):

        V_n = start·(1+r)^n + C·[(1+r)^n − 1]/r       (annual C, n contributions)

    Solved for C, then ÷12. Returns 0.0 when the starting balance already grows
    past the coast target on its own (no contributions needed), or when there is
    no working window (coast_age ≤ current_age).
    """
    n = int(coast_age) - int(current_age)
    if n <= 0:
        return 0.0

    r = float(real_return)
    grown_start = float(start_value) * ((1 + r) ** n)

    if r == 0:
        annual = (coast_target - grown_start) / n
    else:
        annuity_factor = (((1 + r) ** n) - 1) / r
        annual = (coast_target - grown_start) / annuity_factor

    return max(annual, 0.0) / 12.0


def forecast_summary(
    projection_df: pd.DataFrame,
    annual_spend: float,
    real_return: float,
    start_value: float,
    current_age: int,
    coast_age: int,
    retirement_age: int,
    swr: float = SWR,
) -> dict:
    """Headline numbers for the BANs.

    Distinguishes the two CoastFIRE quantities:
      retirement_goal  — nest egg needed AT retirement = annual_spend / swr
      coast_target     — amount needed AT coast_age so growth alone (no further
                         contributions) reaches retirement_goal by retirement_age
                         = retirement_goal / (1+r)^(retirement_age − coast_age)

    Returns:
        retirement_goal          — nest egg target at retirement
        coast_target             — target at coast age (the dashboard's focus)
        projected_at_coast       — projected balance at coast_age (given contribution)
        coast_surplus            — projected_at_coast − coast_target
        coast_point_age          — earliest age you could stop contributing and
                                   still hit the goal by retirement, or None
        required_monthly         — monthly contribution to hit coast_target by coast_age
        projected_at_retirement  — total at first retirement-phase age
        sustainable_income       — swr × projected_at_retirement
    """
    annual_spend = float(annual_spend)
    r = float(real_return)
    current_age = int(current_age)
    coast_age = int(coast_age)
    retirement_age = int(retirement_age)

    retirement_goal = annual_spend / swr if swr else 0.0
    years_coast_to_ret = max(retirement_age - coast_age, 0)
    coast_target = retirement_goal / ((1 + r) ** years_coast_to_ret)

    if coast_age in projection_df.index:
        projected_at_coast = float(projection_df.loc[coast_age, "total"])
    else:
        projected_at_coast = float(projection_df["total"].iloc[-1])

    ret_rows = projection_df[projection_df["phase"] == "retirement"]
    if not ret_rows.empty:
        projected_at_retirement = float(ret_rows["total"].iloc[0])
    else:
        projected_at_retirement = float(projection_df["total"].iloc[-1])

    # Earliest age you could stop contributing and still coast to the goal:
    # project that age's balance forward at r to retirement with no new money.
    coast_point_age = None
    for age, row in projection_df.iterrows():
        if age > retirement_age:
            break
        forward_value = row["total"] * ((1 + r) ** (retirement_age - age))
        if forward_value >= retirement_goal:
            coast_point_age = int(age)
            break

    required_monthly = required_monthly_contribution(
        start_value, current_age, coast_age, coast_target, r
    )

    return {
        "retirement_goal": retirement_goal,
        "coast_target": coast_target,
        "projected_at_coast": projected_at_coast,
        "coast_surplus": projected_at_coast - coast_target,
        "coast_point_age": coast_point_age,
        "required_monthly": required_monthly,
        "projected_at_retirement": projected_at_retirement,
        "sustainable_income": swr * projected_at_retirement,
    }
