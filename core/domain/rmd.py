"""Required Minimum Distributions (SECURE 2.0 start ages + IRS Uniform Lifetime table).

Moved verbatim from core/services/retirement.py (Phase 4 of the core/
restructuring); pinned by tests/test_retirement_math.py.

FUTURE HOME: calculate_withdrawal_strategy (RMD-aware withdrawal ordering) from
core/models/retirement.py (see docs/porting-backlog.md once created).
"""

from __future__ import annotations

# RMDs — IRS Uniform Lifetime divisors (post-2022). Start age 73 (born 1951–59)
# or 75 (born ≥1960). Roth IRA exempt for the owner.
RMD_START_AGE_BY_BIRTH_YEAR: list[tuple[int, int]] = [
    (1950, 72),   # born ≤1950
    (1959, 73),   # born 1951–1959
    (9999, 75),   # born ≥1960
]
UNIFORM_LIFETIME_DIVISORS: dict[int, float] = {
    73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
    87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1,
    94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4,
}


def rmd_start_age(birth_year: int) -> int:
    """RMD start age from birth year (SECURE 2.0): 72 / 73 / 75."""
    for cutoff, age in RMD_START_AGE_BY_BIRTH_YEAR:
        if int(birth_year) <= cutoff:
            return age
    return 75


def rmd_divisor(age: int) -> float | None:
    """IRS Uniform Lifetime divisor for an age, or None if outside the table."""
    return UNIFORM_LIFETIME_DIVISORS.get(int(age))


def required_min_distribution(trad_balance: float, age: int, start_age: int) -> float:
    """Forced annual RMD from the tax-deferred balance (0 before `start_age`).

    RMD = prior-year-end balance ÷ Uniform-Lifetime divisor. Past the table's top
    age, fall back to the oldest divisor (conservative — slightly understates).
    Roth is exempt and never passed here.
    """
    if trad_balance <= 0 or int(age) < int(start_age):
        return 0.0
    div = rmd_divisor(age) or UNIFORM_LIFETIME_DIVISORS[max(UNIFORM_LIFETIME_DIVISORS)]
    return trad_balance / div
