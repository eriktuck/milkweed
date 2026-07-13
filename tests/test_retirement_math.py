"""Characterization tests for the retirement math in core/domain/ (assumptions,
taxes, social_security, healthcare, rmd, drawdown, holdings — dissolved out of
core/services/retirement.py in Phase 4) + SS/tax helpers in core/utils/functions.py.

These pin the CURRENT behavior of main (2026-07 characterization epoch) as the
safety net for the core/domain/ restructuring. Every literal below was produced
by running the current code — do not "correct" an expectation, even where a
`# NOTE: possible bug` marks surprising output.

Goldens: tests/goldens/retmath_*.csv — generated once via
`uv run python -m tests.test_retirement_math` (see warning at bottom).
"""

from __future__ import annotations


import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from core.domain import assumptions, drawdown, healthcare, holdings, rmd
from core.domain import social_security
from core.domain import taxes
from core.utils import functions as fn

# ════════════════════════════════════════════════════════════════════════════
# Shared synthetic fixtures (module-level builders; no Firestore, no clock)
# ════════════════════════════════════════════════════════════════════════════

SPEND_BY_PHASE = {"go_go": 80_000.0, "slow_go": 68_000.0, "no_go": 56_000.0}

PLAN = {
    "mortgage": 2_000.0, "groceries": 800.0, "vacations": 500.0,
    "guilt_free": 600.0, "auto": 300.0, "pets": 100.0,
    "medical": 200.0, "health_insurance": 400.0,
    "savings": 1_000.0, "income": 9_000.0, "joint_contribution": 1_500.0,
    "brokerage": 700.0, "salary": 5_000.0, "mystery_key": 50.0,
}
LABELS = {
    "mortgage": "fixed", "groceries": "fixed", "vacations": "sinking",
    "guilt_free": "guilt-free", "auto": "fixed", "pets": "fixed",
    "medical": "fixed", "health_insurance": "fixed",
    "brokerage": "investments", "salary": "income", "income": "income",
    "mystery_key": "guilt-free",
}

SEGMENTS = [
    {"date": "2015-01-01", "amount": 60_000.0},
    {"date": "2018-07-01", "amount": 90_000.0},
    {"date": "2021-01-01", "amount": 120_000.0},
]

USER_CFG = {
    "birth_date": "1985-06-15",
    "retirement_age": 65, "death_age": 90, "claim_age": 67,
    "retirement": {"slow_go_age": 75, "no_go_age": 85, "real_return": 0.05},
}

SSA_CSV = (
    "Work Year,Taxed Social Security Earnings,Taxed Medicare Earnings\n"
    '2015,"$60,000","$60,000"\n'
    '2016,"$60,000","$60,000"\n'
    '2017,"$60,000","$60,000"\n'
    '2018,"$75,000","$75,000"\n'
    '2019,"$75,000","$75,000"\n'
    '2020,"$168,600","$200,000"\n'
    '2021,"$168,600","$200,000"\n'
    "2022,-,-\n"
    '2023,"","$0"\n'
)


def _ss_series() -> pd.Series:
    return social_security.social_security_income(40_000.0, 67, 65, 90)


def _hc_series() -> pd.Series:
    return healthcare.healthcare_costs_by_age(65, 90)


def _projection() -> pd.DataFrame:
    return drawdown.project_retirement(
        1_500_000.0, 65, 90, SPEND_BY_PHASE, 75, 85, 0.05,
        income_by_age=_ss_series(), healthcare_by_age=_hc_series(),
    )


def _taxaware() -> pd.DataFrame:
    return drawdown.project_retirement_taxaware(
        {"taxable": 600_000.0, "trad": 700_000.0, "roth": 200_000.0},
        65, 90, SPEND_BY_PHASE, 75, 85, 0.05,
        ss_by_age=_ss_series(), healthcare_by_age=_hc_series(),
        rmd_start=75, taxable_gain_fraction=0.5,
    )


approx = lambda v: pytest.approx(v, rel=1e-9)  # noqa: E731


def _read_golden(path, index_col: str) -> pd.DataFrame:
    """Read a golden CSV, normalizing the int64-vs-float64 round-trip.

    `%.17g` writes whole floats (e.g. 80000.0) without a decimal point, so
    read_csv infers int64; the values are identical — only the dtype needs
    normalizing back to float64 for a strict value comparison.
    """
    df = pd.read_csv(path, index_col=index_col)
    for col in df.columns:
        if df[col].dtype.kind == "i":
            df[col] = df[col].astype("float64")
    return df


# ════════════════════════════════════════════════════════════════════════════
# annual_tax / _progressive_tax
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("income, expected", [
    (0.0, 0.0),
    (31_500.0, 0.0),          # exactly the standard deduction
    (55_350.0, 2_385.0),      # deduction + top of 10% bracket
    (128_450.0, 11_157.0),    # deduction + top of 12% bracket
    (238_200.0, 35_302.0),    # deduction + top of 22% bracket
    (426_100.0, 80_398.0),    # deduction + top of 24% bracket
    (532_550.0, 114_462.0),   # deduction + top of 32% bracket
    (783_100.0, 202_154.5),   # deduction + top of 35% bracket
    (883_100.0, 239_154.5),   # 100k into the 37% bracket
    (100_000.0, 7_743.0),
    (300_000.0, 50_134.0),
])
def test_annual_tax_ordinary_boundaries(income, expected):
    """Ordinary-only MFJ tax at (and between) 2025 bracket edges."""
    assert taxes.annual_tax(income) == approx(expected)


@pytest.mark.parametrize("ordinary, ltcg, expected", [
    (0.0, 31_500.0, 0.0),      # deduction fully shelters the gain
    (0.0, 128_200.0, 0.0),     # deduction + the whole 0% LTCG band
    (0.0, 128_300.0, 15.0),    # first $100 into the 15% band
    (150_000.0, 0.0, 15_898.0),
    (150_000.0, 50_000.0, 23_398.0),   # gain stacks fully in the 15% band
    (100_000.0, 600_000.0, 96_935.5),  # gain spans into the 20% band
    (-5_000.0, -5_000.0, 0.0),         # negatives clamp to zero
])
def test_annual_tax_ltcg_stacking(ordinary, ltcg, expected):
    """LTCG stacks on top of taxable ordinary income."""
    assert taxes.annual_tax(ordinary, ltcg) == approx(expected)


def test_annual_tax_no_niit_above_threshold():
    """Income far above NIIT_THRESHOLD_MFJ adds NO 3.8% NIIT.

    # NOTE: possible bug — NIIT_RATE / NIIT_THRESHOLD_MFJ are defined in §3 but
    # annual_tax never applies them: 65_134 = ordinary 50_134 + 15% × 100_000.
    # (The module docstring calls this a documented Phase-0 simplification.)
    """
    assert taxes.annual_tax(300_000.0, 100_000.0) == approx(65_134.0)


@pytest.mark.parametrize("amount, expected", [
    (0.0, 0.0),
    (23_850.0, 2_385.0),      # exactly at the first bracket edge
    (23_851.0, 2_385.12),
    (96_950.0, 11_157.0),     # exactly at the second bracket edge
    (1_000_000.0, 294_062.5),
])
def test_progressive_tax_edges(amount, expected):
    """_progressive_tax through the 2025 MFJ ordinary table."""
    assert taxes.progressive_tax(amount, taxes.ORDINARY_BRACKETS_MFJ) == approx(expected)


# ════════════════════════════════════════════════════════════════════════════
# PIA pipeline (ported from scripts/verify_ss_estimator.py)
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("gross, emp, expected", [
    (100_000, "W2", 100_000.0),
    (100_000, "1099", 92_350.0),
    (100_000, "self-employed", 92_350.0),
    (100_000, "se", 92_350.0),
    (100_000, "mixed", 100_000.0),   # mixed treated as W2 (documented v1)
    (300_000, "W2", 176_100.0),      # OASDI wage-base cap
    (-10, "W2", 0.0),
    (None, "W2", 0.0),
])
def test_covered_earnings(gross, emp, expected):
    """W2 vs self-employed covered earnings, wage-base cap, clamps."""
    assert social_security.covered_earnings(gross, emp) == approx(expected)


@pytest.mark.parametrize("aime, expected", [
    (900, 810.0),               # below first bend point (90%)
    (1_226, 1_103.4),           # exactly at bend point 1
    (1_227, 1_103.72),          # $1 past bend point 1 (32%)
    (4_000, 1_991.0800000000002),
    (7_391, 3_076.2),           # exactly at bend point 2
    (9_000, 3_317.5499999999997),  # past bend point 2 (15%)
    (0, 0.0),
    (-5, 0.0),
])
def test_primary_insurance_amount_bend_points(aime, expected):
    """90/32/15 bend-point formula at and around both bend points."""
    assert social_security.primary_insurance_amount(aime) == approx(expected)


@pytest.mark.parametrize("gross, emp, years, expected", [
    (100_000, "W2", 35, 38_610.6),
    (100_000, "1099", 35, 37_463.1),
    (300_000, "W2", 35, 50_025.59999999999),   # capped at wage base
    (100_000, "W2", 20, 26_818.674285714285),  # short career averages $0 years
    (100_000, "W2", 50, 38_610.6),             # career_years clamps to 35
    (0, "W2", 35, 0.0),
])
def test_estimate_annual_pia_from_income(gross, emp, years, expected):
    """Constant-income FRA benefit estimate."""
    assert social_security.estimate_annual_pia_from_income(gross, emp, years) == approx(expected)


def test_annual_pia_from_earnings():
    """Real earnings series: top-35 AIME, None/zero filtering, empty → 0."""
    assert social_security.annual_pia_from_earnings([100_000.0] * 35) == approx(38_610.6)
    # 40 rising years — only the top 35 count
    rising = [50_000 + 2_000 * i for i in range(40)]
    assert social_security.annual_pia_from_earnings(rising) == approx(37_710.6)
    # None and 0 years are dropped before the top-35 sort
    assert social_security.annual_pia_from_earnings([None, 0, 100_000.0, 100_000.0]) == approx(
        5_142.857142857143)
    assert social_security.annual_pia_from_earnings([]) == 0.0


def test_social_security_income_claim_factors():
    """Claim-age factor scales the PIA; benefit starts at max(claim, retirement)."""
    ss = social_security.social_security_income(40_000.0, 70, 67, 75)
    assert float(ss.loc[69]) == 0.0
    assert float(ss.loc[70]) == approx(49_600.0)  # 1.24 × PIA
    assert float(ss.loc[75]) == approx(49_600.0)
    assert list(ss.index) == list(range(67, 76))

    # Claiming before retirement: payments still start at retirement age.
    early = social_security.social_security_income(40_000.0, 62, 67, 72)
    assert float(early.loc[67]) == approx(28_000.0)  # 0.70 × PIA from age 67

    # Unknown claim age (71) falls back to factor 1.0, paid from 71.
    odd = social_security.social_security_income(40_000.0, 71, 67, 73)
    assert float(odd.loc[70]) == 0.0
    assert float(odd.loc[71]) == approx(40_000.0)


# ════════════════════════════════════════════════════════════════════════════
# core/utils/functions.py — SS + tax legacy helpers
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("earnings, claim_age, expected", [
    ([100_000.0] * 35, 67, 3_134.18),
    ([100_000.0] * 35, 62, 2_193.93),
    ([100_000.0] * 35, 70, 3_886.38),
    ([100_000.0] * 20, 67, 2_204.73),  # <35 years: uses what exists, still ÷420
    ([20_000.0] * 35, 67, 1_214.25),
])
def test_calculate_social_security_benefit(earnings, claim_age, expected):
    """Legacy 2024 bend-point monthly benefit (rounded to cents)."""
    got = fn.calculate_social_security_benefit(pd.Series(earnings), claim_age)
    assert float(got) == approx(expected)


@pytest.mark.parametrize("bad_age", [61, 71])
def test_calculate_social_security_benefit_claim_age_range(bad_age):
    """Claim age outside 62–70 raises."""
    with pytest.raises(ValueError):
        fn.calculate_social_security_benefit(pd.Series([100_000.0] * 35), bad_age)


@pytest.mark.parametrize("income, expected", [
    (0, 0.0),
    (23_000, 2_300.0),         # exactly the 2024 10% bracket edge
    (100_000, 12_110.0),
    (500_000, 115_753.5),
    (1_000_000, 296_129.5),
])
def test_calculate_married_joint_tax_2024(income, expected):
    """Legacy 2024 MFJ table (no deduction, no LTCG)."""
    assert fn.calculate_married_joint_tax(income) == approx(expected)


def test_segments_to_annual_income():
    """Mid-year raise splits the year; monthly-grain integration."""
    ann = fn.segments_to_annual_income(SEGMENTS)
    assert ann.to_dict() == {
        2015: 60_000.0, 2016: 60_000.0, 2017: 60_000.0,
        2018: 75_000.0,  # 6 mo @ 60k + 6 mo @ 90k
        2019: 90_000.0, 2020: 90_000.0, 2021: 120_000.0,
    }


def test_segments_to_annual_income_projection_tail():
    """thru_year extends the last rate at growth_rate compounding."""
    ann = fn.segments_to_annual_income(SEGMENTS, thru_year=2024, growth_rate=0.03)
    assert ann.loc[2022] == approx(123_600.0)
    assert ann.loc[2023] == approx(127_308.0)
    assert ann.loc[2024] == approx(131_127.24)


def test_segments_to_annual_income_empty():
    """Empty segments → empty Series."""
    assert fn.segments_to_annual_income([]).empty


def test_ssa_earnings_to_segments_medicare_basis():
    """SSA CSV → segments: Jan-1 dates, $-string parsing, run-length de-dupe."""
    segs = fn.ssa_earnings_to_segments(SSA_CSV)
    assert segs == [
        {"date": "2015-01-01", "amount": 60_000.0},
        {"date": "2018-01-01", "amount": 75_000.0},
        {"date": "2020-01-01", "amount": 200_000.0},  # medicare column, uncapped
        {"date": "2023-01-01", "amount": 0.0},        # "$0" row kept ("-" rows dropped)
    ]


def test_ssa_earnings_to_segments_ss_basis():
    """basis='social_security' reads the capped column; '-'/'' rows drop."""
    segs = fn.ssa_earnings_to_segments(SSA_CSV, basis="social_security")
    assert segs == [
        {"date": "2015-01-01", "amount": 60_000.0},
        {"date": "2018-01-01", "amount": 75_000.0},
        {"date": "2020-01-01", "amount": 168_600.0},
    ]


def test_ssa_earnings_to_segments_errors():
    """Empty text and unusable CSVs raise ValueError."""
    with pytest.raises(ValueError):
        fn.ssa_earnings_to_segments("")
    with pytest.raises(ValueError):
        fn.ssa_earnings_to_segments("foo,bar\n1,2\n")
    with pytest.raises(ValueError):  # header ok, but no usable rows
        fn.ssa_earnings_to_segments(
            "Work Year,Taxed Medicare Earnings\nabc,-\n")


def test_ssa_roundtrip_to_pia():
    """Synthetic SSA CSV → segments → annual income → FRA PIA."""
    segs = fn.ssa_earnings_to_segments(SSA_CSV)
    ann = fn.segments_to_annual_income(segs)
    assert ann.to_dict() == {
        2015: 60_000.0, 2016: 60_000.0, 2017: 60_000.0,
        2018: 75_000.0, 2019: 75_000.0,
        2020: 200_000.0, 2021: 200_000.0, 2022: 200_000.0, 2023: 0.0,
    }
    assert social_security.annual_pia_from_earnings(ann) == approx(16_380.274285714288)


@pytest.mark.parametrize("text, expected", [
    ("$1,234", 1_234.0),
    ("1234.5", 1_234.5),
    (" $2,000.75 ", 2_000.75),
    ("$0", 0.0),
    (None, None),
    ("", None),
    ("-", None),
    ("abc", None),
])
def test_money_to_float(text, expected):
    """_money_to_float parses formatted money; junk → None."""
    assert fn._money_to_float(text) == expected


def test_load_vanguard_cost_basis(tmp_path):
    """Synthetic Vanguard export: rename, numeric coercion, account/symbol groupby."""
    csv_path = tmp_path / "costbasis.csv"
    csv_path.write_text(
        "Account,Symbol/CUSIP,Total cost\n"
        "12341370,VTI,1000.50\n"
        "12341370,VTI,499.50\n"      # duplicate lot — must aggregate
        "12341370,VXUS,2000.00\n"
        "56788191,VTI,300.25\n"
        "56788191,BND,not-a-number\n"  # coerced to NaN, sums to 0-contribution
    )
    df = fn.load_vanguard_cost_basis(str(csv_path))
    assert list(df.columns) == ["account", "symbol", "cost_basis"]
    rows = {(r.account, r.symbol): r.cost_basis for r in df.itertuples()}
    assert rows[(12341370, "VTI")] == approx(1_500.0)
    assert rows[(12341370, "VXUS")] == approx(2_000.0)
    assert rows[(56788191, "VTI")] == approx(300.25)
    assert rows[(56788191, "BND")] == approx(0.0)  # NaN sums to 0.0


# ════════════════════════════════════════════════════════════════════════════
# Healthcare glide (ported from scripts/verify_healthcare.py — actuals pinned)
# ════════════════════════════════════════════════════════════════════════════

PART_B = healthcare.MEDICARE_PART_B_MONTHLY_2025 * 12.0  # 2220.0


def test_healthcare_defaults_aca_to_medicare():
    """ACA bridge before 65, Medicare + OOP from 65; step DOWN at 65."""
    hc = healthcare.healthcare_costs_by_age(62, 90)
    assert hc.loc[62] == approx(15_000.0)   # OOP 4000 + ACA 11000
    assert hc.loc[64] == approx(15_000.0)
    assert hc.loc[65] == approx(6_220.0)    # OOP 4000 + Part B 2220
    assert hc.loc[64] > hc.loc[65]
    assert hc.loc[84] == approx(6_220.0)


def test_healthcare_default_ltc_runs_through_death():
    """Default LTC spike (nursing, 128k) runs 85 → death.

    # NOTE: possible bug — scripts/verify_healthcare.py asserts
    # hc.loc[90] == HC_OOP_ANNUAL + PART_B (6_220), which FAILS on current main:
    # the default ltc_annual is now 128_000 from age 85 through the horizon, so
    # the actual value is 134_220. Pinning the actual.
    """
    hc = healthcare.healthcare_costs_by_age(62, 90)
    assert hc.loc[85] == approx(134_220.0)
    assert hc.loc[90] == approx(134_220.0)


def test_healthcare_ltc_fixed_window():
    """Explicit ltc_years bounds the spike to [start, start+years).

    # NOTE: possible bug — verify_healthcare compares this 'spiked' run against a
    # default-args 'base' run assuming base has NO spike; on current main the base
    # already carries the default 128k LTC from 85, so those assertions fail.
    # Actual values pinned here instead.
    """
    spiked = healthcare.healthcare_costs_by_age(
        65, 95, ltc_annual=70_000, ltc_start_age=85, ltc_years=3)
    assert spiked.loc[84] == approx(6_220.0)
    assert spiked.loc[85] == approx(76_220.0)
    assert spiked.loc[87] == approx(76_220.0)
    assert spiked.loc[88] == approx(6_220.0)   # spike ends after 3 years
    assert spiked.loc[85] > spiked.loc[66]     # the smile upturn
    base = healthcare.healthcare_costs_by_age(65, 95)
    assert base.loc[85] == approx(134_220.0)   # default base DOES spike


def test_healthcare_no_ltc_and_overrides():
    """ltc_annual=0 disables the spike; medicare_annual override flows through."""
    hc = healthcare.healthcare_costs_by_age(62, 90, ltc_annual=0.0)
    assert hc.loc[85] == approx(6_220.0)
    assert hc.loc[90] == approx(6_220.0)
    hc26 = healthcare.healthcare_costs_by_age(
        63, 70, medicare_annual=healthcare.MEDICARE_PART_B_MONTHLY_2026 * 12,
        ltc_annual=0.0)
    assert hc26.loc[64] == approx(15_000.0)
    assert hc26.loc[65] == approx(6_434.8)


def test_healthcare_default_golden(goldens_dir):
    """Full default series (62→90) matches the committed golden."""
    actual = healthcare.healthcare_costs_by_age(62, 90).rename_axis("age").to_frame("cost")
    golden = _read_golden(goldens_dir / "retmath_healthcare_default_62_90.csv",
                          index_col="age")
    assert_frame_equal(actual, golden, rtol=1e-9)


# ════════════════════════════════════════════════════════════════════════════
# RMD
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("birth_year, expected", [
    (1949, 72), (1950, 72),   # ≤1950 boundary
    (1951, 73), (1959, 73),   # 1951–1959 boundary
    (1960, 75), (1985, 75),   # ≥1960
])
def test_rmd_start_age_boundaries(birth_year, expected):
    """SECURE 2.0 start ages at every birth-year boundary."""
    assert rmd.rmd_start_age(birth_year) == expected


@pytest.mark.parametrize("age, expected", [
    (50, None), (72, None),          # below the table (72 despite being a start age!)
    (73, 26.5), (74, 25.5),          # table start
    (99, 6.8), (100, 6.4),           # table end
    (101, None),                     # beyond the table
])
def test_rmd_divisor_table_edges(age, expected):
    """Uniform Lifetime divisor lookup at/beyond the table edges."""
    assert rmd.rmd_divisor(age) == expected


def test_required_min_distribution():
    """0 before start age; balance ÷ divisor at/after; fallback past the table."""
    assert rmd.required_min_distribution(1_000_000, 74, 75) == 0.0
    assert rmd.required_min_distribution(1_000_000, 75, 75) == approx(40_650.40650406504)
    assert rmd.required_min_distribution(1_000_000, 90, 75) == approx(81_967.21311475411)
    # Past the table top: falls back to the age-100 divisor (6.4).
    assert rmd.required_min_distribution(1_000_000, 101, 75) == approx(156_250.0)
    assert rmd.required_min_distribution(0, 80, 75) == 0.0
    assert rmd.required_min_distribution(-5, 80, 75) == 0.0


def test_required_min_distribution_start_age_72_below_table():
    """Start age 72 (born ≤1950) is BELOW the divisor table.

    # NOTE: possible bug — rmd_divisor(72) is None, so the `or` fallback picks
    # the OLDEST divisor (age 100, 6.4) and a 72-year-old's RMD computes as
    # 1_000_000 / 6.4 = 156_250 — wildly overstated vs the real age-72 divisor.
    """
    assert rmd.required_min_distribution(1_000_000, 72, 72) == approx(156_250.0)


# ════════════════════════════════════════════════════════════════════════════
# Config resolution / phases / spend
# ════════════════════════════════════════════════════════════════════════════

def test_resolve_assumptions_defaults():
    """None config → DEFAULTS + birth_year None."""
    assert assumptions.resolve_assumptions(None) == {
        "retirement_age": 65, "death_age": 90, "claim_age": 67,
        "slow_go_age": 75, "no_go_age": 85, "real_return": 0.05,
        "withdrawal_rate": 0.04, "birth_year": None,
    }


def test_resolve_assumptions_precedence():
    """Demographics from top level; knobs from `retirement`; nested demographics ignored."""
    cfg = {
        "birth_date": "1985-06-15",
        "retirement_age": 60, "death_age": 95, "claim_age": 70,
        "retirement": {"slow_go_age": 73, "no_go_age": 83, "real_return": 0.04,
                       "withdrawal_rate": 0.035,
                       "retirement_age": 99},  # ignored — Profile owns demographics
    }
    assert assumptions.resolve_assumptions(cfg) == {
        "retirement_age": 60, "death_age": 95, "claim_age": 70,
        "slow_go_age": 73, "no_go_age": 83, "real_return": 0.04,
        "withdrawal_rate": 0.035, "birth_year": 1985,
    }


def test_resolve_assumptions_none_values_fall_back():
    """Explicit None values fall back to DEFAULTS."""
    got = assumptions.resolve_assumptions(
        {"retirement_age": None, "retirement": {"real_return": None}})
    assert got["retirement_age"] == 65
    assert got["real_return"] == 0.05


def test_resolve_phase_factors():
    """Per-key per-phase merge; unknown keys seed from DEFAULT_PHASE_FACTOR."""
    merged = assumptions.resolve_phase_factors(
        {"phase_factors": {"mortgage": {"no_go": 0.0}, "new_key": {"slow_go": 0.5}}})
    assert merged["mortgage"] == {"slow_go": 0.95, "no_go": 0.0}
    assert merged["new_key"] == {"slow_go": 0.5, "no_go": 1.0}
    assert merged["groceries"] == {"slow_go": 0.90, "no_go": 0.75}  # untouched
    # No overrides → identical to the default table (but a fresh copy).
    plain = assumptions.resolve_phase_factors(None)
    assert plain == assumptions.PHASE_FACTORS
    assert plain is not assumptions.PHASE_FACTORS


@pytest.mark.parametrize("age, expected", [
    (64, "go_go"), (74, "go_go"),
    (75, "slow_go"), (84, "slow_go"),
    (85, "no_go"), (100, "no_go"),
])
def test_phase_for_age_boundaries(age, expected):
    """slow_go starts AT slow_go_age; no_go starts AT no_go_age."""
    assert assumptions.phase_for_age(age, 75, 85) == expected


def test_annual_spend_by_phase_totals():
    """Drops non-spend/healthcare/investments/income keys; scales per phase."""
    out = assumptions.annual_spend_by_phase(PLAN, LABELS)
    assert out["go_go"]["total"] == approx(52_200.0)
    assert out["slow_go"]["total"] == approx(45_420.0)
    assert out["no_go"]["total"] == approx(38_820.0)
    # Excluded: healthcare keys, savings/income/joint_contribution,
    # investments- and income-labeled keys.
    for excluded in ("medical", "health_insurance", "savings", "income",
                     "joint_contribution", "brokerage", "salary"):
        assert excluded not in out["go_go"]["by_key"]
    # An unlabeled/unknown key rides through flat (DEFAULT_PHASE_FACTOR).
    assert out["no_go"]["by_key"]["mystery_key"] == approx(600.0)


def test_annual_spend_by_phase_golden(goldens_dir):
    """Per-key annual dollars by phase match the committed golden."""
    out = assumptions.annual_spend_by_phase(PLAN, LABELS)
    actual = (pd.DataFrame({p: out[p]["by_key"] for p in assumptions.PHASES})
              .rename_axis("csp_key").sort_index())
    golden = _read_golden(goldens_dir / "retmath_spend_by_phase.csv",
                          index_col="csp_key")
    assert_frame_equal(actual, golden, rtol=1e-9)


def test_annual_spend_by_phase_empty():
    """Empty plan → zero totals for all three phases."""
    out = assumptions.annual_spend_by_phase(None, None)
    assert {p: out[p]["total"] for p in assumptions.PHASES} == {
        "go_go": 0, "slow_go": 0, "no_go": 0}


def test_household_expense_share():
    """Proportional to joint contributions; zero/negative clamped."""
    assert assumptions.household_expense_share({"a": 3_000.0, "b": 2_000.0}, "a") == approx(0.6)
    assert assumptions.household_expense_share({"a": 3_000.0}, "z") == 0.0
    assert assumptions.household_expense_share({"a": 0.0, "b": 0.0}, "a") == 0.0
    assert assumptions.household_expense_share({"a": -500.0, "b": 1_000.0}, "a") == 0.0


def test_merge_household_expenses():
    """Adds share × household amounts per key; transfer lines ride along."""
    merged = assumptions.merge_household_expenses(
        {"groceries": 200.0, "joint_contribution": 1_500.0},
        {"mortgage": 3_000.0, "groceries": 500.0, "income": 100.0},
        0.6,
    )
    assert merged == {
        "groceries": 500.0, "joint_contribution": 1_500.0,
        "mortgage": 1_800.0, "income": 60.0,
    }
    # share 0 → individual plan unchanged (copied).
    ind = {"groceries": 200.0}
    same = assumptions.merge_household_expenses(ind, {"mortgage": 3_000.0}, 0.0)
    assert same == {"groceries": 200.0}
    assert same is not ind


# ════════════════════════════════════════════════════════════════════════════
# Projection / goal / summary
# ════════════════════════════════════════════════════════════════════════════

def test_project_retirement_golden(goldens_dir):
    """Full v1-light drawdown frame matches the committed golden."""
    actual = _projection()
    golden = _read_golden(goldens_dir / "retmath_project_retirement.csv",
                          index_col="age")
    assert_frame_equal(actual, golden, rtol=1e-9)


def test_project_retirement_first_years():
    """Balance-entering convention and net-withdrawal composition."""
    proj = _projection()
    assert proj.loc[65, "total"] == approx(1_500_000.0)
    assert proj.loc[65, "withdrawal"] == approx(86_220.0)   # 80k + 6220 − 0
    assert proj.loc[66, "total"] == approx(1_484_469.0)     # (1.5M − 86220)·1.05
    assert proj.loc[67, "withdrawal"] == approx(46_220.0)   # SS 40k kicks in at 67
    assert proj.loc[65, "phase"] == "go_go"
    assert proj.loc[75, "phase"] == "slow_go"
    assert proj.loc[85, "phase"] == "no_go"


def test_nest_egg_goal():
    """PV of the withdrawal stream at the retirement age; empty df → 0."""
    assert drawdown.nest_egg_goal(_projection(), 0.05) == approx(924_904.2311928394)
    assert drawdown.nest_egg_goal(pd.DataFrame(), 0.05) == 0.0


def test_retirement_summary():
    """Headline BAN numbers for the standard projection."""
    summ = drawdown.retirement_summary(_projection(), 0.05)
    assert summ["nest_egg_goal"] == approx(924_904.2311928394)
    assert summ["balance_at_death"] == approx(2_097_698.39819046)
    assert summ["avg_annual_spend"] == approx(105_604.61538461539)
    assert summ["first_year_drawdown"] == approx(86_220.0)
    assert summ["peak_drawdown"] == approx(150_220.0)
    assert summ["funded_through_age"] == 90
    assert summ["swr_sanity_goal"] == approx(2_155_500.0)


def test_retirement_summary_empty():
    """Empty projection → zeroed summary with funded_through_age None."""
    assert drawdown.retirement_summary(pd.DataFrame(), 0.05) == {
        "nest_egg_goal": 0.0, "balance_at_death": 0.0, "avg_annual_spend": 0.0,
        "first_year_drawdown": 0.0, "peak_drawdown": 0.0,
        "funded_through_age": None, "swr_sanity_goal": 0.0,
    }


def test_project_retirement_taxaware_golden(goldens_dir):
    """Full tax-aware drawdown frame matches the committed golden."""
    actual = _taxaware()
    golden = _read_golden(goldens_dir / "retmath_taxaware.csv", index_col="age")
    assert_frame_equal(actual, golden, rtol=1e-9)


def test_project_retirement_taxaware_shape():
    """Column contract and endpoint balances."""
    df = _taxaware()
    assert list(df.columns) == [
        "total", "taxable", "trad", "roth", "spend", "healthcare", "income",
        "rmd", "withdrawal", "w_taxable", "w_trad", "w_roth", "ordinary_income",
        "ltcg_gain", "tax", "net_spend", "phase",
    ]
    assert df.loc[65, "total"] == approx(1_500_000.0)
    assert df.loc[90, "total"] == approx(2_028_631.5339715758)
    assert df.loc[74, "rmd"] == 0.0                    # before rmd_start
    assert float(df.loc[75, "rmd"]) > 0.0              # forced from 75


@pytest.mark.parametrize("args, expected", [
    # (target, taxable, trad, roth, rmd) → (w_taxable, w_trad, w_roth, excess, shortfall)
    ((50_000, 100_000, 200_000, 50_000, 0.0),
     (50_000.0, 0.0, 0.0, 0.0, 0.0)),                       # taxable covers
    ((50_000, 100_000, 200_000, 50_000, 80_000.0),
     (0.0, 80_000.0, 0.0, 30_000.0, 0.0)),                  # RMD exceeds target → excess
    ((150_000, 100_000, 200_000, 50_000, 20_000.0),
     (100_000, 50_000.0, 0.0, 0.0, 0.0)),                   # spills into trad past RMD
    ((400_000, 100_000, 200_000, 50_000, 0.0),
     (100_000, 200_000.0, 50_000, 0.0, 50_000.0)),          # exhausts all → shortfall
    ((0.0, 100_000, 200_000, 50_000, 30_000.0),
     (0.0, 30_000.0, 0.0, 30_000.0, 0.0)),                  # RMD only, all excess
])
def test_allocate_withdrawals(args, expected):
    """taxable → trad → roth ordering with forced-RMD handling (promoted in Phase 4)."""
    assert drawdown.allocate_withdrawals(*args) == expected


def test_default_contribution_allocation():
    """Tax-advantaged-first split: trad cap, then Roth cap, then taxable."""
    assert assumptions.default_contribution_allocation(50_000) == {
        "taxable": 19_500.0, "trad": 23_500.0, "roth": 7_000.0}
    assert assumptions.default_contribution_allocation(20_000) == {
        "taxable": 0.0, "trad": 20_000.0, "roth": 0.0}
    assert assumptions.default_contribution_allocation(25_000) == {
        "taxable": 0.0, "trad": 23_500.0, "roth": 1_500.0}
    assert assumptions.default_contribution_allocation(-5) == {
        "taxable": 0.0, "trad": 0.0, "roth": 0.0}


def test_project_balances_to_retirement():
    """Per-bucket V' = V·(1+r) + contribution over `years`; years ≤ 0 passthrough."""
    grown = assumptions.project_balances_to_retirement(
        {"taxable": 100_000.0, "trad": 200_000.0, "roth": 50_000.0},
        {"taxable": 5_000.0, "trad": 10_000.0, "roth": 7_000.0}, 10, 0.05)
    assert grown["taxable"] == approx(225_778.92535548835)
    assert grown["trad"] == approx(451_557.8507109767)
    assert grown["roth"] == approx(169_489.9790877139)
    assert grown["total"] == approx(846_826.755154179)
    flat = assumptions.project_balances_to_retirement(
        {"taxable": 100_000.0, "trad": 200_000.0, "roth": 50_000.0},
        {"taxable": 5_000.0}, 0, 0.05)
    assert flat == {"taxable": 100_000.0, "trad": 200_000.0, "roth": 50_000.0,
                    "total": 350_000.0}


# ════════════════════════════════════════════════════════════════════════════
# default_nest_egg_goal — one of the two duplicated PIA-resolution paths
# ════════════════════════════════════════════════════════════════════════════

def test_default_nest_egg_goal_with_income_segments():
    """Segments present → PIA via segments_to_annual_income + annual_pia_from_earnings.

    Internal PIA path: SEGMENTS → annual income 2015–2021 →
    annual_pia_from_earnings (7 covered years averaged over 35) → SS at claim 67.
    """
    goal = drawdown.default_nest_egg_goal(
        {**USER_CFG, "income_segments": SEGMENTS}, PLAN, LABELS)
    assert goal == approx(899_448.7611446389)


def test_default_nest_egg_goal_csp_income_fallback():
    """No segments → constant-income estimate from CSP `income`-labeled lines.

    PIA path: 12 × (9_000 + 5_000) = 168_000/yr → estimate_annual_pia_from_income
    (35-year constant career, wage-base capped) — a much larger SS offset than the
    segments path, hence the smaller goal.
    """
    goal = drawdown.default_nest_egg_goal(USER_CFG, PLAN, LABELS)
    assert goal == approx(436_821.6413845806)


def test_default_nest_egg_goal_no_income_no_ss():
    """Neither segments nor CSP income → PIA 0, goal is pure spend + healthcare PV."""
    goal = drawdown.default_nest_egg_goal(
        USER_CFG, {"mortgage": 2_000.0}, {"mortgage": "fixed"})
    assert goal == approx(702_449.1432860926)


def test_default_nest_egg_goal_all_defaults():
    """Everything None → DEFAULTS ages, zero spend, default healthcare glide only."""
    assert drawdown.default_nest_egg_goal(None, None, None) == approx(350_988.1696171528)


# ════════════════════════════════════════════════════════════════════════════
# Cost-basis math (pure part of scripts/verify_cost_basis.py)
# ════════════════════════════════════════════════════════════════════════════

COST_BASIS = {
    "1370": {"market_value": 100_000.0, "cost": 62_600.0, "gain": 37_400.0},
    "8191": {"market_value": 50_000.0, "cost": 23_400.0, "gain": 26_600.0},
    "5555": {"market_value": 80_000.0, "cost": 10_000.0, "gain": 70_000.0},
}


def test_gain_fraction_all_taxable():
    """Blended (ΣMV − Σcost)/ΣMV over taxable accounts."""
    accounts = {k: "Taxable" for k in COST_BASIS}
    assert holdings.gain_fraction_from_cost_basis(COST_BASIS, accounts) == approx(
        0.5826086956521739)


def test_gain_fraction_label_filtering():
    """TRAD_TYPES and ROTH_TYPES accounts are excluded from the blend."""
    accounts = {"1370": "Taxable", "8191": "IRA", "5555": "Roth IRA"}
    assert holdings.gain_fraction_from_cost_basis(COST_BASIS, accounts) == approx(0.374)


def test_gain_fraction_unlabeled_defaults_taxable():
    """No investment_accounts map → every account counts as taxable."""
    assert holdings.gain_fraction_from_cost_basis(COST_BASIS, None) == approx(
        0.5826086956521739)


def test_gain_fraction_none_cases():
    """All tax-deferred, or empty report → None (caller keeps the manual default)."""
    assert holdings.gain_fraction_from_cost_basis(
        COST_BASIS, {k: "IRA" for k in COST_BASIS}) is None
    assert holdings.gain_fraction_from_cost_basis({}, None) is None


# ════════════════════════════════════════════════════════════════════════════
# Golden generation — run ONCE per characterization epoch.
# WARNING: a refactor phase may NEVER regenerate these goldens; they pin the
# behavior of current main. `uv run python -m tests.test_retirement_math`
# ════════════════════════════════════════════════════════════════════════════

def _generate_goldens() -> None:
    from tests.conftest import GOLDENS_DIR

    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)

    _projection().to_csv(
        GOLDENS_DIR / "retmath_project_retirement.csv", float_format="%.17g")
    _taxaware().to_csv(
        GOLDENS_DIR / "retmath_taxaware.csv", float_format="%.17g")
    (healthcare.healthcare_costs_by_age(62, 90).rename_axis("age").to_frame("cost")
     .to_csv(GOLDENS_DIR / "retmath_healthcare_default_62_90.csv",
             float_format="%.17g"))
    out = assumptions.annual_spend_by_phase(PLAN, LABELS)
    (pd.DataFrame({p: out[p]["by_key"] for p in assumptions.PHASES})
     .rename_axis("csp_key").sort_index()
     .to_csv(GOLDENS_DIR / "retmath_spend_by_phase.csv", float_format="%.17g"))
    print("retmath_* goldens written to", GOLDENS_DIR)


if __name__ == "__main__":
    _generate_goldens()
