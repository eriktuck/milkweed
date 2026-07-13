"""Tax math.

`calculate_married_joint_tax` moved verbatim from core/utils/functions.py
(Phase 3 of the core/ restructuring); `annual_tax` / `progressive_tax` and the
2025 MFJ constants moved verbatim from core/services/retirement.py (Phase 4,
`progressive_tax` promoted public). Pinned by tests/test_retirement_math.py.

The two tax functions deliberately coexist: `annual_tax` is the planning-grade
retirement calculation (2025 brackets + LTCG stacking + standard deduction);
`calculate_married_joint_tax` (2024, ordinary only) backs the frozen
core/models/ household models and is not edited. FUTURE HOME: FICA, CO state
tax, per-lot LTCG (see core/models/ + docs/porting-backlog.md once created).
"""

from __future__ import annotations

# ── 2025 MFJ constants (single source of truth for the tax year + rate tables) ──

TAX_YEAR: int = 2025

# Ordinary income brackets, Married-Filing-Jointly (rate, upper edge).
ORDINARY_BRACKETS_MFJ: list[tuple[float, float]] = [
    (0.10, 23_850),
    (0.12, 96_950),
    (0.22, 206_700),
    (0.24, 394_600),
    (0.32, 501_050),
    (0.35, 751_600),
    (0.37, float("inf")),
]

# Long-term capital-gains thresholds, MFJ (rate applies up to the upper edge).
LTCG_BRACKETS_MFJ: list[tuple[float, float]] = [
    (0.00, 96_700),
    (0.15, 600_050),
    (0.20, float("inf")),
]

STANDARD_DEDUCTION_MFJ: float = 31_500.0   # acts as the bottom "0% bracket"

# Net Investment Income Tax (statutory, never indexed).
NIIT_RATE: float = 0.038
NIIT_THRESHOLD_MFJ: float = 250_000.0


def progressive_tax(amount: float, brackets: list[tuple[float, float]]) -> float:
    """Tax on `amount` through a cumulative (rate, upper-edge) bracket table."""
    tax = 0.0
    lower = 0.0
    for rate, upper in brackets:
        if amount <= lower:
            break
        tax += (min(amount, upper) - lower) * rate
        lower = upper
    return tax


def annual_tax(ordinary_income: float, ltcg_gain: float = 0.0) -> float:
    """Federal tax on a year's ordinary income + long-term capital gain (MFJ).

    Ordinary income is taxed through the ordinary brackets after the standard
    deduction. The LTCG **stacks on top** of taxable ordinary income (its 0/15/20%
    band depends on total taxable income); any standard deduction unused by
    ordinary income shelters gain first.
    """
    ordinary_income = max(float(ordinary_income), 0.0)
    ltcg_gain = max(float(ltcg_gain), 0.0)
    std = STANDARD_DEDUCTION_MFJ

    ord_taxable = max(ordinary_income - std, 0.0)
    unused_deduction = max(std - ordinary_income, 0.0)
    gain_taxable = max(ltcg_gain - unused_deduction, 0.0)

    tax = progressive_tax(ord_taxable, ORDINARY_BRACKETS_MFJ)

    # LTCG occupies the band from ord_taxable upward.
    lower = ord_taxable
    remaining = gain_taxable
    for rate, upper in LTCG_BRACKETS_MFJ:
        if remaining <= 0:
            break
        if lower < upper:
            band = min(upper, lower + remaining) - lower
            tax += band * rate
            lower += band
            remaining -= band
    return tax


def calculate_married_joint_tax(income):
    """
    Calculate federal income tax for a married couple filing jointly in 2024.

    Parameters:
    income (float): The taxable income of the couple.

    Returns:
    float: The total federal income tax owed.
    """
    # 2024 Tax Brackets for Married Filing Jointly
    brackets = [
        (0, 23000, 0.10),
        (23000, 94300, 0.12),
        (94300, 201050, 0.22),
        (201050, 383900, 0.24),
        (383900, 487450, 0.32),
        (487450, 731200, 0.35),
        (731200, float('inf'), 0.37)
    ]

    tax_owed = 0
    for lower, upper, rate in brackets:
        if income > lower:
            taxable_amount = min(income, upper) - lower
            tax_owed += taxable_amount * rate
        else:
            break

    return round(tax_owed, 2)
