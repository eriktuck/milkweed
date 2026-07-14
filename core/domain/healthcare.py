"""Healthcare cost glide: working → ACA bridge → Medicare → late-life LTC spike.

Moved verbatim from core/services/retirement.py (Phase 4 of the core/
restructuring); pinned by tests/test_retirement_math.py.

FUTURE HOME: the ACA→Medicare premium crossover model from
core/models/healthcare.py (see docs/porting-backlog.md).
"""

from __future__ import annotations

import pandas as pd

# Medicare standard Part B premium (annualized monthly figure).
MEDICARE_AGE: int = 65
MEDICARE_PART_B_MONTHLY_2025: float = 185.00
MEDICARE_PART_B_MONTHLY_2026: float = 202.90

# Healthcare glide defaults (annual, real $, per individual) — research §"Healthcare".
# These are *total* annual healthcare costs for each life stage, editable on the
# page. Medicare age 65 splits the pre-65 ACA bridge from Medicare + out-of-pocket.
HC_OOP_ANNUAL: float = 4_000.0      # Medigap / Part D / out-of-pocket on top of Part B
HC_ACA_ANNUAL: float = 11_000.0     # pre-65 ACA-bridge premium (unsubsidized, 60-something)

# Long-term-care spike (Genworth 2024 medians). Modelled as an explicit editable
# late-life event, not baked into routine spend (research recommendation). The
# `type` dropdown pre-populates the (still-editable) annual cost from these medians:
#   * in-home — homemaker / home-health-aide median (~44 hrs/week)
#   * nursing — private-room nursing-home median
LTC_COST_BY_TYPE: dict[str, float] = {
    "none": 0.0,
    "in_home": 75_000.0,
    "nursing": 128_000.0,
}
LTC_DEFAULT_TYPE: str = "nursing"
LTC_DEFAULT_ANNUAL: float = LTC_COST_BY_TYPE[LTC_DEFAULT_TYPE]
# LTC runs from this start age through the end of the planning horizon (death age).
LTC_DEFAULT_START_AGE: int = 85


def healthcare_costs_by_age(
    retirement_age: int,
    death_age: int,
    *,
    oop_annual: float = HC_OOP_ANNUAL,
    aca_annual: float = HC_ACA_ANNUAL,
    medicare_annual: float | None = None,
    medicare_age: int = MEDICARE_AGE,
    ltc_annual: float = LTC_DEFAULT_ANNUAL,
    ltc_start_age: int = LTC_DEFAULT_START_AGE,
    ltc_years: int | None = None,
) -> pd.Series:
    """Annual healthcare cost by age (real $).  (pure, age-indexed)

    Reproduces the working → ACA-bridge → Medicare → late-life glide of
    core.models.healthcare, but **indexed on age** (fixing that model's
    age-vs-year bug) and grounded in the research $:

      * before `medicare_age` (early retirement): ACA-bridge premium + baseline OOP
      * `medicare_age`+: Medicare Part B + baseline OOP
      * an editable **LTC spike** of `ltc_annual` from `ltc_start_age` onward —
        through the planning horizon (`death_age`) by default, or for a fixed
        `ltc_years` window when given (kept for callers that still pass one)

    This is the single source of the late-life "smile" upturn: the `medical` /
    `health_insurance` csp keys are excluded from the generic spend pool upstream
    (assumptions.HEALTHCARE_CSP_KEYS), so adding this series is additive, not
    double-counted. `medicare_annual` defaults to the standard Part B premium
    (×12). Returns a Series indexed by age (retirement_age … death_age).
    """
    medicare_annual = (MEDICARE_PART_B_MONTHLY_2025 * 12.0
                       if medicare_annual is None else float(medicare_annual))
    ltc_start = int(ltc_start_age)
    # LTC runs from its start age through the end of the planning horizon, unless a
    # fixed-length window is explicitly requested.
    ltc_end = (int(death_age) + 1 if ltc_years is None
               else ltc_start + max(int(ltc_years), 0))
    spike = float(ltc_annual or 0.0)

    out: dict[int, float] = {}
    for age in range(int(retirement_age), int(death_age) + 1):
        cost = float(oop_annual) + (medicare_annual if age >= int(medicare_age)
                                    else float(aca_annual))
        if spike > 0 and ltc_start <= age < ltc_end:
            cost += spike
        out[age] = cost
    return pd.Series(out)
