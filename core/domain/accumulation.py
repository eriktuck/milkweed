"""Accumulation-side CoastFIRE math: coast target + required contribution.

`required_monthly_contribution` and `trailing_12mo_contribution` moved verbatim
from core/services/forecast.py (Phase 5 of the core/ restructuring — forecast.py
dissolved); `coast_target` is the one consolidation that phase made: the
`goal / (1+r)^(retirement − coast)` discount previously computed inline by both
`forecast.forecast_summary` and `scenario.build_scenario`. All pinned by
tests/test_lifetime.py.

The glide-path variants of both quantities (per-age rates instead of one flat
rate) live in core/domain/glide.py (`cumulative_growth`,
`required_monthly_glide`); `scenario.build_scenario` picks per the active
return model. Pure — no Firestore or network.
"""

from __future__ import annotations

import pandas as pd

from core.domain.assumptions import CONTRIBUTION_LABELS


def coast_target(
    retirement_goal: float,
    real_return: float,
    coast_age: int,
    retirement_age: int,
) -> float:
    """Amount needed AT coast_age so growth alone reaches the goal by retirement.

    Distinguishes the two CoastFIRE quantities:
      retirement_goal  — nest egg needed AT retirement (given, the
                         backward-from-spending PV)
      coast_target     — retirement_goal / (1+r)^(retirement_age − coast_age)

    A coast age at/after retirement discounts nothing (the target is the goal).
    """
    years = max(int(retirement_age) - int(coast_age), 0)
    growth = (1 + float(real_return)) ** years
    return float(retirement_goal) / growth if growth else float(retirement_goal)


def required_monthly_contribution(
    start_value: float,
    current_age: int,
    coast_age: int,
    coast_target: float,
    real_return: float,
) -> float:
    """Monthly contribution needed to reach `coast_target` exactly by `coast_age`.

    Closed-form inverse of the engine's working-phase recurrence
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
