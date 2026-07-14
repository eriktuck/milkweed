"""Retirement drawdown: backward-from-spending goal + tax-aware withdrawal math.

Moved verbatim from core/services/retirement.py (Phase 4 of the core/
restructuring); pinned by tests/test_retirement_math.py. The former private
helpers `_as_age_lookup` / `_allocate_withdrawals` are promoted public here
(`as_age_lookup`, `allocate_withdrawals`); core/domain/lifetime.py builds its
unified engine on them (the import edge points lifetime → drawdown, keeping
the graph a DAG).

The tax math is planning-grade and retirement-scoped (2025 MFJ constants in
core/domain/taxes.py) rather than reusing taxes.calculate_married_joint_tax
(2024, no LTCG) — that function backs the household models and changing it
would ripple. Flat-tax decision: brackets + LTCG + standard deduction; no
NIIT / SS-taxation / state. Social Security is treated as untaxed in v1 (a
documented simplification).

Spend is stored/returned as positive magnitudes here; **never abs() spend in
the UI — negate (* -1) for display** (see feedback-no-abs-on-spending).

FUTURE HOME: per-lot cost basis and the 59.5 IRA gate from
core/models/retirement.py (see docs/porting-backlog.md).
"""

from __future__ import annotations

import pandas as pd

from core.domain.assumptions import (
    DEFAULTS,
    PHASES,
    annual_spend_by_phase,
    phase_for_age,
    resolve_assumptions,
    resolve_phase_factors,
)
from core.domain.healthcare import healthcare_costs_by_age
from core.domain.social_security import resolve_pia, social_security_income

# Planning default: share of a taxable-account withdrawal that is realized
# long-term gain (taxed at LTCG); the rest is return of basis (tax-free). We do
# not track per-lot cost basis in v1, so this is an editable assumption.
TAXABLE_GAIN_FRACTION: float = 0.5


def as_age_lookup(src: pd.Series | dict | None) -> dict:
    """Normalize an income/healthcare-by-age input to a plain {age: float} dict."""
    if src is None:
        return {}
    if isinstance(src, pd.Series):
        return {int(k): float(v) for k, v in src.to_dict().items()}
    return {int(k): float(v) for k, v in src.items()}


def project_retirement(
    start_value: float,
    retirement_age: int,
    death_age: int,
    spend_by_phase: dict[str, float],
    slow_go_age: int,
    no_go_age: int,
    real_return: float,
    income_by_age: pd.Series | dict | None = None,
    healthcare_by_age: pd.Series | dict | None = None,
) -> pd.DataFrame:
    """Year-by-year drawdown from retirement_age to death_age.  (v1 light model)

    The backward-from-spending engine. Convention: each year the portfolio funds
    that year's net living cost up front, and the remainder grows one year at
    `real_return` (`total' = (total − withdrawal)·(1+r)`). `total` recorded for
    an age is the balance **entering** that age-year (before that year's draw).

    Net withdrawal = phase living spend + healthcare − income. A negative net
    (income > cost) is reinvested (grows the balance) rather than clamped.

    Returns a DataFrame indexed by `age` with columns:
        total, spend, healthcare, income, withdrawal, phase

    All flows are positive magnitudes (income is stored positive in its own
    column; it enters `withdrawal` as a subtraction). The chart negates
    spend/withdrawal for display (never abs()).
    """
    retirement_age = int(retirement_age)
    death_age = int(death_age)
    r = float(real_return)
    income = as_age_lookup(income_by_age)
    health = as_age_lookup(healthcare_by_age)

    total = float(start_value)
    rows = []
    for age in range(retirement_age, death_age + 1):
        phase = phase_for_age(age, slow_go_age, no_go_age)
        spend = float(spend_by_phase.get(phase, 0.0))
        hc = health.get(age, 0.0)
        inc = income.get(age, 0.0)
        withdrawal = spend + hc - inc

        rows.append({
            "age": age,
            "total": total,
            "spend": spend,
            "healthcare": hc,
            "income": inc,
            "withdrawal": withdrawal,
            "phase": phase,
        })
        total = (total - withdrawal) * (1 + r)

    return pd.DataFrame(rows).set_index("age")


def nest_egg_goal(
    projection_df: pd.DataFrame,
    real_return: float,
) -> float:
    """Portfolio needed AT retirement to exactly fund the projected draw stream.

    Present value (at the first/retirement age, discounted at `real_return`) of
    the per-year net withdrawals — the start_value at which the account funds
    every year's net cost and depletes to ~0 at death. This is the
    *backward-from-spending* answer, independent of current holdings.
    """
    r = float(real_return)
    ages = projection_df.index.tolist()
    if not ages:
        return 0.0
    base = ages[0]
    return float(sum(
        projection_df.loc[age, "withdrawal"] / ((1 + r) ** (age - base))
        for age in ages
    ))


def default_nest_egg_goal(
    user_cfg: dict | None,
    active_plan: dict | None,
    csp_labels: dict | None,
) -> float:
    """The from-config nest-egg goal — the default backward-from-spending PV.

    Single source of the *default* goal math. The Coast page tunes the same pipeline
    interactively (editable expense table, healthcare, SS) through
    `scenario.build_scenario`, which falls back to this default when a given input is
    not overridden. Retained as the no-interaction baseline; the reconciliation
    against `build_scenario`'s backward goal is pinned in `tests/test_scenario.py`.

    Mirrors the Coast assumption-module defaults:
      * spend — phase totals from the CSP plan (`annual_spend_by_phase`), healthcare
        keys excluded (modelled separately below).
      * healthcare — the module-default ACA-bridge → Medicare → LTC glide
        (`healthcare_costs_by_age` with its defaults; oop folded into those totals).
      * Social Security — Profile earnings history → PIA when present, else a
        CSP-`income` constant-income estimate; 0 when neither is available.

    Pure: callers pass the deserialized config + plan dicts (no Firestore). Returns
    a non-negative goal (0 when Social Security alone covers spending).
    """
    a = resolve_assumptions(user_cfg)
    retirement_age = int(a["retirement_age"])
    death_age = int(a["death_age"])
    slow_go_age = int(a["slow_go_age"])
    no_go_age = int(a["no_go_age"])
    claim_age = int(a["claim_age"])
    r = float(a["real_return"])

    # Spend: phase totals from the CSP plan (healthcare excluded inside the helper).
    phase_factors = resolve_phase_factors((user_cfg or {}).get("retirement"))
    by_phase = annual_spend_by_phase(active_plan, csp_labels, phase_factors)
    spend = {p: by_phase[p]["total"] for p in PHASES}

    # Healthcare: the module-default late-life glide (same defaults the page seeds).
    hc = healthcare_costs_by_age(retirement_age, death_age)

    # Social Security: prefer Profile earnings history; else a CSP-income estimate.
    pia = resolve_pia(user_cfg, active_plan, csp_labels)
    ss = social_security_income(pia, claim_age, retirement_age, death_age)

    stream = project_retirement(0.0, retirement_age, death_age, spend,
                                slow_go_age, no_go_age, r,
                                income_by_age=ss, healthcare_by_age=hc)
    return max(nest_egg_goal(stream, r), 0.0)


def retirement_summary(
    projection_df: pd.DataFrame,
    real_return: float,
    withdrawal_rate: float = DEFAULTS["withdrawal_rate"],
) -> dict:
    """Headline numbers for the BANs.

    Everything is derived from `projection_df`, which carries the per-age
    spend/healthcare/withdrawal/total columns; `real_return` is needed for the
    nest-egg present value.

    Returns:
        nest_egg_goal        — start value that funds the stream to death (PV)
        balance_at_death     — projected balance entering death_age
        avg_annual_spend     — mean annual gross living cost (spend + healthcare)
        first_year_drawdown  — net withdrawal in the first retirement year
        peak_drawdown        — largest single-year net withdrawal
        funded_through_age   — last age the balance stays ≥ 0 (or death_age)
        swr_sanity_goal      — first-year gross spend ÷ withdrawal_rate (4%-rule
                               cross-check on the backward-derived goal)
    """
    if projection_df.empty:
        return {
            "nest_egg_goal": 0.0, "balance_at_death": 0.0, "avg_annual_spend": 0.0,
            "first_year_drawdown": 0.0, "peak_drawdown": 0.0,
            "funded_through_age": None, "swr_sanity_goal": 0.0,
        }

    swr = float(withdrawal_rate)
    ages = projection_df.index.tolist()
    gross = projection_df["spend"] + projection_df["healthcare"]

    funded = projection_df[projection_df["total"] >= 0]
    funded_through_age = int(funded.index.max()) if not funded.empty else None

    first_year_drawdown = float(projection_df["withdrawal"].iloc[0])

    return {
        "nest_egg_goal": nest_egg_goal(projection_df, real_return),
        "balance_at_death": float(projection_df["total"].iloc[-1]),
        "avg_annual_spend": float(gross.mean()),
        "first_year_drawdown": first_year_drawdown,
        "peak_drawdown": float(projection_df["withdrawal"].max()),
        "funded_through_age": funded_through_age,
        "swr_sanity_goal": (float(gross.iloc[0]) / swr) if swr else 0.0,
    }


# ── Tax-aware withdrawal ordering (v1.5) ─────────────────────────────────────────
# The full tax-aware engine (simulate_lifetime + project_retirement_taxaware)
# lives in core/domain/lifetime.py; this is the per-year allocation rule it uses.

def allocate_withdrawals(target, taxable, trad, roth, rmd):
    """Fund `target` after RMD, in order taxable → trad → roth.

    The RMD is withdrawn from `trad` first (forced, taxed as ordinary); if it
    exceeds the cash needed, the excess is returned to be reinvested in taxable.
    Returns (w_taxable, w_trad, w_roth, excess_rmd, shortfall).
    """
    w_trad = min(rmd, trad)
    excess_rmd = max(w_trad - target, 0.0)
    remaining = max(target - w_trad, 0.0)

    w_taxable = min(remaining, taxable)
    remaining -= w_taxable

    extra_trad = min(remaining, trad - w_trad)
    w_trad += extra_trad
    remaining -= extra_trad

    w_roth = min(remaining, roth)
    remaining -= w_roth

    return w_taxable, w_trad, w_roth, excess_rmd, remaining
