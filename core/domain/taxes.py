"""Tax math.

Moved verbatim from core/utils/functions.py (Phase 3 of the core/
restructuring); pinned by tests/test_retirement_math.py.

FUTURE HOME (Phase 4): annual_tax, progressive_tax and the 2025 MFJ brackets
from core/services/retirement.py. Later feature work: FICA, CO state tax,
per-lot LTCG (see core/models/ + docs/porting-backlog.md once created).
"""


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
