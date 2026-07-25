"""Characterization tests for core/domain/housing.py (SPEC-housing Phase 1).

Pins the behavior of the housing leaf module at its introduction (2026-07
characterization epoch). Nothing imports housing.py yet — Phase 1 is pure math
in isolation — so these tests are the module's only consumer, and the goldens
are its contract for the wiring phases that follow.

Goldens: tests/goldens/housing_*.csv — generated once via
`uv run python -m tests.test_housing` (see warning at bottom).
"""

from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from core.domain import assumptions, housing
from core.domain.config import (
    HousingConfig,
    Property,
    PropertySale,
    Residence,
    RetirementConfig,
)

approx = lambda v: pytest.approx(v, rel=1e-9)  # noqa: E731

# ── Fixtures: the author's case — rent an apartment, let one house to tenants ──
CURRENT_AGE = 62
END_AGE = 90

# The house: $650k today, $2,400/mo PITI of which $700 is escrow, 6.25% note,
# paid off midway through age 68 (hence the fractional payoff age).
HOUSE = dict(
    current_value=650_000.0,
    real_appreciation=0.01,
    monthly_payment=2_400.0,
    escrow_monthly=700.0,
    payoff_age=68.5,
    maintenance_pct=0.01,
    inflation=0.025,
)
NOTE_RATE = 0.0625
MONTHLY_PI = HOUSE["monthly_payment"] - HOUSE["escrow_monthly"]   # 1,700

# Let to tenants at $3,200/mo, standard vacancy + management haircut.
TENANTS = dict(monthly_rent=3_200.0, vacancy_pct=0.08, real_rent_growth=0.0)

# Where the household actually lives: a $2,100/mo apartment, rent flat in real terms.
APARTMENT = dict(residence_mode="rent", monthly_rent=2_100.0, real_rent_growth=0.0)


def _carry() -> pd.DataFrame:
    return housing.carrying_cost_breakdown(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, **HOUSE)


def _own_and_rent_out() -> pd.DataFrame:
    """The full own-and-rent-out scenario as one frame."""
    gross = housing.rental_gross_income_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, **TENANTS)
    net = housing.rental_net_income_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, **HOUSE, **TENANTS)
    cost = housing.housing_costs_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, **APARTMENT)
    equity = housing.equity_by_age(
        CURRENT_AGE, END_AGE, note_rate=NOTE_RATE,
        **{k: v for k, v in HOUSE.items() if k != "maintenance_pct"})
    return pd.DataFrame({
        "housing_cost": cost,
        "rental_gross": gross,
        "rental_net": net,
        "value": equity["value"],
        "balance": equity["balance"],
        "equity": equity["equity"],
    }).rename_axis("age")


def _read_golden(path, index_col: str) -> pd.DataFrame:
    """Read a golden CSV, normalizing the int64-vs-float64 round-trip.

    `%.17g` writes whole floats (e.g. 25200.0) without a decimal point, so
    read_csv infers int64; the values are identical — only the dtype needs
    normalizing back to float64 for a strict value comparison.
    """
    df = pd.read_csv(path, index_col=index_col)
    for col in df.columns:
        if df[col].dtype.kind == "i":
            df[col] = df[col].astype("float64")
    return df


# ════════════════════════════════════════════════════════════════════════════
# age_at_date
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("iso, birth_year, expected", [
    ("2047-01-01", 1985, 62.0),                 # Jan 1 is exactly year - birth_year
    ("2047-12-31", 1985, 62 + 364 / 365),       # last day of a non-leap year
    ("2048-07-01", 1985, 63 + 182 / 366),       # leap year — 366-day denominator
    ("2047-03-01T00:00:00", 1985, 62 + 59 / 365),   # datetime strings truncate to the date
])
def test_age_at_date(iso, birth_year, expected):
    assert housing.age_at_date(iso, birth_year) == approx(expected)


@pytest.mark.parametrize("iso, birth_year", [
    (None, 1985),          # no date
    ("2047-01-01", None),  # no birth year
    ("", 1985),
    ("not-a-date", 1985),
])
def test_age_at_date_underivable_is_none(iso, birth_year):
    assert housing.age_at_date(iso, birth_year) is None


def test_age_at_date_matches_resolve_assumptions_convention():
    """Jan 1 anchoring keeps this consistent with `current_age = year - birth_year`."""
    assert housing.age_at_date("2026-01-01", 1985) == approx(2026 - 1985)


# ════════════════════════════════════════════════════════════════════════════
# amortization_balance
# ════════════════════════════════════════════════════════════════════════════

def test_amortization_balance_zero_at_payoff():
    """Nothing left to discount once the last payment is made."""
    assert housing.amortization_balance(MONTHLY_PI, NOTE_RATE, 0) == 0.0
    assert housing.amortization_balance(MONTHLY_PI, NOTE_RATE, -12) == 0.0


def test_amortization_balance_no_loan_is_zero():
    assert housing.amortization_balance(0.0, NOTE_RATE, 240) == 0.0


def test_amortization_balance_zero_rate_is_payment_times_months():
    """A 0% note is just the remaining payments, undiscounted."""
    assert housing.amortization_balance(MONTHLY_PI, 0.0, 240) == approx(MONTHLY_PI * 240)


def test_amortization_balance_reconciles_with_the_payment_schedule():
    """The PV formula must equal the month-by-month amortization it stands in for.

    Walking the recurrence `balance = balance*(1+i) - payment` from the PV-derived
    opening balance must land exactly on zero at the final payment — that is what
    makes payment + rate + payoff date sufficient to describe the whole loan.
    """
    months = 78                      # 62 → 68.5
    opening = housing.amortization_balance(MONTHLY_PI, NOTE_RATE, months)
    i = NOTE_RATE / 12.0

    balance = opening
    interest_paid = 0.0
    for _ in range(months):
        interest = balance * i
        interest_paid += interest
        balance = balance + interest - MONTHLY_PI

    assert balance == pytest.approx(0.0, abs=1e-6)
    # Payments = principal + interest, by construction.
    assert MONTHLY_PI * months == approx(opening + interest_paid)


def test_amortization_balance_declines_monotonically():
    balances = [housing.amortization_balance(MONTHLY_PI, NOTE_RATE, n)
                for n in range(78, -1, -1)]
    assert all(b > nxt for b, nxt in zip(balances, balances[1:]))


# ════════════════════════════════════════════════════════════════════════════
# value_by_age / equity_by_age
# ════════════════════════════════════════════════════════════════════════════

def test_value_by_age_compounds_from_current_age():
    v = housing.value_by_age(62, 90, 650_000.0, real_appreciation=0.01)
    assert list(v.index) == list(range(62, 91))       # inclusive of end_age
    assert v.loc[62] == approx(650_000.0)             # anchored, not grown
    assert v.loc[72] == approx(650_000.0 * 1.01 ** 10)


def test_equity_by_age_balance_deflated_to_todays_dollars():
    """Nominal debt, real value — the balance must be deflated to match."""
    eq = housing.equity_by_age(CURRENT_AGE, END_AGE, note_rate=NOTE_RATE,
                               **{k: v for k, v in HOUSE.items()
                                  if k != "maintenance_pct"})
    nominal_65 = housing.amortization_balance(MONTHLY_PI, NOTE_RATE, (68.5 - 65) * 12)
    assert eq.loc[65, "balance"] == approx(nominal_65 * 1.025 ** -3)
    assert eq.loc[65, "equity"] == approx(eq.loc[65, "value"] - eq.loc[65, "balance"])


def test_equity_by_age_is_all_value_after_payoff():
    eq = housing.equity_by_age(CURRENT_AGE, END_AGE, note_rate=NOTE_RATE,
                               **{k: v for k, v in HOUSE.items()
                                  if k != "maintenance_pct"})
    assert eq.loc[69, "balance"] == 0.0
    assert eq.loc[69, "equity"] == approx(eq.loc[69, "value"])


# ════════════════════════════════════════════════════════════════════════════
# carrying_cost_breakdown — the phase gate: real decay + the payoff cliff
# ════════════════════════════════════════════════════════════════════════════

def test_carrying_cost_pi_decays_in_real_terms():
    """A nominally fixed payment shrinks by exactly one year of inflation."""
    df = _carry()
    for age in range(62, 67):        # full-payment years only
        assert df.loc[age + 1, "pi"] / df.loc[age, "pi"] == approx(1 / 1.025)


def test_carrying_cost_payoff_year_is_prorated():
    """Payoff at 68.5 means half a year of P&I in the year of age 68."""
    df = _carry()
    full_68 = MONTHLY_PI * 12.0 * 1.025 ** -(68 - 62)
    assert df.loc[68, "pi"] == approx(full_68 * 0.5)
    assert df.loc[67, "pi"] == approx(MONTHLY_PI * 12.0 * 1.025 ** -(67 - 62))


def test_carrying_cost_cliff_lands_on_a_nonzero_floor():
    """P&I ends; escrow and maintenance do not. This is the whole module."""
    df = _carry()
    assert df.loc[69, "pi"] == 0.0
    assert df.loc[90, "pi"] == 0.0
    floor_69 = df.loc[69, "escrow"] + df.loc[69, "maintenance"]
    assert df.loc[69, "total"] == approx(floor_69)
    assert floor_69 > 0.0
    # The drop from 68 to 69 is exactly the (prorated) P&I that ended.
    assert df.loc[68, "total"] - df.loc[69, "total"] == approx(
        df.loc[68, "pi"] - (df.loc[69, "maintenance"] - df.loc[68, "maintenance"]))


def test_carrying_cost_escrow_is_flat_real_and_survives_payoff():
    df = _carry()
    assert df["escrow"].tolist() == approx([700.0 * 12] * len(df))


def test_carrying_cost_maintenance_grows_with_the_home():
    df = _carry()
    assert df.loc[72, "maintenance"] == approx(0.01 * 650_000.0 * 1.01 ** 10)


def test_carrying_cost_no_payoff_age_runs_throughout():
    """An unmodelled payoff date means the payment does not end within the horizon."""
    df = housing.carrying_cost_breakdown(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE,
        **{**HOUSE, "payoff_age": None})
    assert df.loc[90, "pi"] > 0.0


def test_carrying_cost_by_age_is_the_total_column():
    assert housing.carrying_cost_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE,
        **HOUSE).equals(_carry()["total"])


def test_carrying_cost_golden(goldens_dir):
    """Itemized carry, 62→90, payoff at 68.5 — the P&I decay and cliff in one file."""
    actual = _carry().rename_axis("age")
    golden = _read_golden(goldens_dir / "housing_primary_carry_62_90.csv",
                          index_col="age")
    assert_frame_equal(actual, golden, rtol=1e-9)


# ════════════════════════════════════════════════════════════════════════════
# Rental income — net, because the carry offsets the rent (Schedule E)
# ════════════════════════════════════════════════════════════════════════════

def test_rental_gross_applies_the_vacancy_haircut():
    g = housing.rental_gross_income_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, **TENANTS)
    assert g.loc[62] == approx(3_200.0 * 12 * 0.92)
    assert g.loc[90] == approx(3_200.0 * 12 * 0.92)      # 0% real growth → flat


def test_rental_gross_real_growth_can_be_negative():
    """Rent control is a negative real rate, not a special case."""
    g = housing.rental_gross_income_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE,
        **{**TENANTS, "real_rent_growth": -0.01})
    assert g.loc[72] == approx(3_200.0 * 12 * 0.92 * 0.99 ** 10)


def test_rental_net_is_gross_less_that_propertys_carry():
    net = housing.rental_net_income_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, **HOUSE, **TENANTS)
    gross = housing.rental_gross_income_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, **TENANTS)
    carry = housing.carrying_cost_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, **HOUSE)
    assert net.equals(gross - carry)


def test_rental_net_may_be_negative_and_is_not_clamped():
    """A carry-heavy rental is a real loss; clamping it would hide the tax effect."""
    net = housing.rental_net_income_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE,
        **HOUSE, **{**TENANTS, "monthly_rent": 1_000.0})
    assert net.loc[62] < 0.0


def test_rental_net_improves_at_payoff():
    net = housing.rental_net_income_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, **HOUSE, **TENANTS)
    assert net.loc[69] > net.loc[68] > net.loc[67]


# ════════════════════════════════════════════════════════════════════════════
# housing_costs_by_age — where you live, only
# ════════════════════════════════════════════════════════════════════════════

def test_housing_costs_renting():
    c = housing.housing_costs_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, **APARTMENT)
    assert list(c.index) == list(range(62, 91))
    assert c.loc[62] == approx(2_100.0 * 12)
    assert c.loc[90] == approx(2_100.0 * 12)             # 0% real growth → flat


def test_housing_costs_renting_applies_real_growth():
    c = housing.housing_costs_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE,
        **{**APARTMENT, "real_rent_growth": 0.005})
    assert c.loc[72] == approx(2_100.0 * 12 * 1.005 ** 10)


def test_housing_costs_owning_is_the_primary_carry():
    c = housing.housing_costs_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE,
        residence_mode="own", primary=HOUSE)
    assert c.equals(_carry()["total"])


def test_housing_costs_excludes_a_rented_out_property():
    """Renting your home while letting a house out: the house's carry is NOT here.

    It nets against its rent in `rental_net_income_by_age` instead — the whole
    reason the two channels are separate.
    """
    c = housing.housing_costs_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, **APARTMENT)
    assert c.loc[62] == approx(2_100.0 * 12)             # rent only, no carry


def test_housing_costs_owning_with_no_property_is_zero():
    c = housing.housing_costs_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, residence_mode="own")
    assert (c == 0.0).all()
    assert list(c.index) == list(range(62, 91))


def test_own_and_rent_out_golden(goldens_dir):
    """The author's case end to end: rent an apartment, let one house to tenants."""
    actual = _own_and_rent_out()
    golden = _read_golden(goldens_dir / "housing_own_and_rent_out_62_90.csv",
                          index_col="age")
    assert_frame_equal(actual, golden, rtol=1e-9)


# ════════════════════════════════════════════════════════════════════════════
# sale_proceeds — the itemization is the point
# ════════════════════════════════════════════════════════════════════════════

SALE = dict(
    current_age=CURRENT_AGE,
    current_value=650_000.0,
    real_appreciation=0.01,
    purchase_price=310_000.0,
    purchase_age=34.0,
    monthly_payment=2_400.0,
    escrow_monthly=700.0,
    note_rate=NOTE_RATE,
    payoff_age=68.5,
    inflation=0.025,
)


def test_sale_net_is_the_sum_of_its_parts():
    s = housing.sale_proceeds(70, occupancy="primary", **SALE)
    assert s["net"] == approx(
        s["gross"] - s["selling_costs"] - s["mortgage_payoff"]
        - s["capital_gains_tax"] - s["recapture_tax"])


def test_sale_after_payoff_has_no_mortgage_to_clear():
    s = housing.sale_proceeds(70, occupancy="primary", **SALE)
    assert s["mortgage_payoff"] == 0.0


def test_sale_before_payoff_clears_the_deflated_balance():
    s = housing.sale_proceeds(65, occupancy="primary", **SALE)
    nominal = housing.amortization_balance(MONTHLY_PI, NOTE_RATE, (68.5 - 65) * 12)
    assert s["mortgage_payoff"] == approx(nominal * 1.025 ** -3)


def test_sale_primary_takes_section_121_and_no_depreciation():
    s = housing.sale_proceeds(70, occupancy="primary", **SALE)
    assert s["depreciation"] == 0.0
    assert s["recapture_amount"] == 0.0
    assert s["recapture_tax"] == 0.0
    assert s["exclusion"] == approx(min(s["gain"], 500_000.0))
    # A gain comfortably under $500k is fully sheltered.
    assert s["gain"] < 500_000.0
    assert s["taxable_gain"] == 0.0
    assert s["capital_gains_tax"] == 0.0


def test_sale_primary_gain_above_the_exclusion_is_taxed():
    s = housing.sale_proceeds(70, occupancy="primary",
                              **{**SALE, "current_value": 1_400_000.0})
    assert s["exclusion"] == approx(500_000.0)
    assert s["taxable_gain"] == approx(s["gain"] - 500_000.0)
    assert s["capital_gains_tax"] > 0.0


def test_sale_rental_gets_no_exclusion_and_pays_recapture():
    """18 years of ownership: straight-line depreciation, no §121, 25% recapture."""
    s = housing.sale_proceeds(70, occupancy="rental", **{**SALE, "purchase_age": 52.0})
    depreciable = 310_000.0 * (1 - 0.20)          # land is not depreciable
    assert s["depreciation"] == approx(depreciable * 18.0 / 27.5)
    assert s["exclusion"] == 0.0
    assert s["recapture_tax"] == approx(s["recapture_amount"] * 0.25)
    assert s["recapture_tax"] > 10_000.0          # five figures — not optional detail
    assert s["adjusted_basis"] == approx(310_000.0 - s["depreciation"])


def test_sale_rental_depreciation_capped_at_the_depreciable_basis():
    """27.5 years of straight-line exhausts the building basis; it cannot go past."""
    s = housing.sale_proceeds(70, occupancy="rental", **SALE)  # 36 years owned
    assert s["depreciation"] == approx(310_000.0 * (1 - 0.20))


def test_sale_rental_costs_materially_more_than_primary():
    """Omitting the rental regime makes a planned sale look better than it is."""
    primary = housing.sale_proceeds(70, occupancy="primary", **SALE)
    rental = housing.sale_proceeds(70, occupancy="rental", **SALE)
    assert primary["net"] - rental["net"] > 50_000.0


def test_sale_capital_gains_tax_stacks_on_other_income():
    """The gain is taxed at the margin of that year's return, not a flat rate."""
    low = housing.sale_proceeds(70, occupancy="primary", other_ordinary_income=0.0,
                                **{**SALE, "current_value": 1_400_000.0})
    high = housing.sale_proceeds(70, occupancy="primary",
                                 other_ordinary_income=300_000.0,
                                 **{**SALE, "current_value": 1_400_000.0})
    assert high["capital_gains_tax"] > low["capital_gains_tax"]


def test_sale_selling_costs_are_a_share_of_gross():
    s = housing.sale_proceeds(70, occupancy="primary", **SALE)
    assert s["selling_costs"] == approx(s["gross"] * 0.07)
    assert s["gross"] == approx(650_000.0 * 1.01 ** 8)


# ════════════════════════════════════════════════════════════════════════════
# derive_real_appreciation — display only
# ════════════════════════════════════════════════════════════════════════════

CPI = pd.Series({y: 0.025 for y in range(2019, 2026)})


def test_derive_real_appreciation_deflates_by_realized_cpi():
    """$310k in 2019 → $650k in 2026, net of seven years of 2.5% CPI."""
    r = housing.derive_real_appreciation(310_000.0, 650_000.0, 2019, 2026, CPI)
    assert r == approx((650_000.0 / 310_000.0) ** (1 / 7) / 1.025 - 1)


def test_derive_real_appreciation_without_cpi_is_the_nominal_cagr():
    r = housing.derive_real_appreciation(310_000.0, 650_000.0, 2019, 2026, None)
    assert r == approx((650_000.0 / 310_000.0) ** (1 / 7) - 1)


def test_derive_real_appreciation_far_exceeds_the_forward_default():
    """The reason it is a display figure and not the forward assumption."""
    r = housing.derive_real_appreciation(310_000.0, 650_000.0, 2019, 2026, CPI)
    assert r > 5 * housing.LONG_RUN_REAL_APPRECIATION


@pytest.mark.parametrize("price, value, y0, y1", [
    (0.0, 650_000.0, 2019, 2026),        # no purchase price
    (310_000.0, 0.0, 2019, 2026),        # no current value
    (310_000.0, 650_000.0, 2026, 2026),  # bought this year
    (310_000.0, 650_000.0, 2027, 2026),  # inverted period
])
def test_derive_real_appreciation_underivable_is_none(price, value, y0, y1):
    assert housing.derive_real_appreciation(price, value, y0, y1, CPI) is None


def test_derive_real_appreciation_ignores_cpi_years_outside_the_hold():
    """Only CPI actually realized during ownership deflates the return."""
    wide = pd.Series({y: 0.025 for y in range(1990, 2030)})
    assert (housing.derive_real_appreciation(310_000.0, 650_000.0, 2019, 2026, wide)
            == approx(housing.derive_real_appreciation(
                310_000.0, 650_000.0, 2019, 2026, CPI)))


# ════════════════════════════════════════════════════════════════════════════
# Config models (SPEC-housing Phase 2) — absent → inactive, no migration
# ════════════════════════════════════════════════════════════════════════════

def test_retirement_config_without_housing_is_valid():
    """A config predating the housing block must load untouched."""
    cfg = RetirementConfig(slow_go_age=75, no_go_age=85, real_return=0.05)
    assert cfg.housing is None


def test_housing_config_all_fields_optional():
    """Every field optional means an empty block is legal — no migration needed."""
    cfg = HousingConfig()
    assert cfg.residence is None and cfg.properties is None and cfg.inflation is None


def test_housing_config_round_trips_through_json():
    """Firestore/dcc.Store persistence: nested models must survive a JSON hop."""
    cfg = RetirementConfig(housing=HousingConfig(
        residence=Residence(mode="rent", monthly_rent=2_100.0, real_rent_growth=0.0),
        properties=[Property(
            label="Elm St", occupancy="rental",
            purchase_price=310_000.0, purchase_date="2019-04-01",
            current_value=650_000.0, real_appreciation=0.01,
            monthly_payment=2_400.0, escrow_monthly=700.0,
            note_rate=0.0625, payoff_date="2047-07-01",
            monthly_rent_income=3_200.0, vacancy_pct=0.08,
            sale=PropertySale(age=70, disposition="rent", selling_cost_pct=0.07),
        )],
        inflation=0.025,
    ))
    revived = RetirementConfig.model_validate_json(cfg.model_dump_json())
    assert revived == cfg
    assert revived.housing.properties[0].sale.disposition == "rent"
    assert revived.housing.residence.mode == "rent"


def test_property_field_names_match_the_engine_kwargs():
    """Fields mirror `core.domain.housing` kwargs so a property can be splatted in.

    This is the healthcare-override pattern; if the names drift, Phase 3's
    resolution silently loses inputs.
    """
    prop = Property(current_value=650_000.0, real_appreciation=0.01,
                    monthly_payment=2_400.0, escrow_monthly=700.0,
                    maintenance_pct=0.01)
    kwargs = {k: v for k, v in prop.model_dump().items() if v is not None}
    carry = housing.carrying_cost_by_age(
        CURRENT_AGE, END_AGE, current_age=CURRENT_AGE, payoff_age=68.5,
        inflation=0.025, **kwargs)
    assert carry.equals(_carry()["total"])


def test_housing_csp_keys_are_the_two_the_module_takes_over():
    """`bills_utilities` stays in the pool; `taxes` means tax *preparation*."""
    assert assumptions.HOUSING_CSP_KEYS == frozenset({"mortgage", "home_improvements"})
    assert not (assumptions.HOUSING_CSP_KEYS & assumptions.HEALTHCARE_CSP_KEYS)
    assert "bills_utilities" not in assumptions.HOUSING_CSP_KEYS


def test_housing_keys_keep_their_phase_factors_for_unconfigured_users():
    """They are excluded only when a housing block exists — the goldens depend
    on the no-config path still applying the 0.95/0.90 taper."""
    for key in assumptions.HOUSING_CSP_KEYS:
        assert assumptions.PHASE_FACTORS[key] == {"slow_go": 0.95, "no_go": 0.90}


def test_housing_exclusion_drops_only_those_keys_from_the_spend_pool():
    plan = {"mortgage": 2_000.0, "home_improvements": 200.0,
            "bills_utilities": 250.0, "groceries": 600.0}
    labels = {k: "fixed" for k in plan}

    default = assumptions.annual_spend_by_phase(plan, labels)
    gated = assumptions.annual_spend_by_phase(
        plan, labels,
        exclude_keys=assumptions.HEALTHCARE_CSP_KEYS | assumptions.HOUSING_CSP_KEYS)

    assert set(default["go_go"]["by_key"]) == set(plan)
    assert set(gated["go_go"]["by_key"]) == {"bills_utilities", "groceries"}
    # The taper still applies to what stays behind.
    assert gated["slow_go"]["by_key"]["bills_utilities"] == approx(250.0 * 12 * 0.95)


# ════════════════════════════════════════════════════════════════════════════
# Golden generation — run ONCE per characterization epoch.
# WARNING: a refactor phase may NEVER regenerate these goldens; they pin the
# behavior of current main. `uv run python -m tests.test_housing`
# ════════════════════════════════════════════════════════════════════════════

def _generate_goldens() -> None:
    from tests.conftest import GOLDENS_DIR

    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in [
        ("housing_primary_carry_62_90.csv", _carry().rename_axis("age")),
        ("housing_own_and_rent_out_62_90.csv", _own_and_rent_out()),
    ]:
        path = GOLDENS_DIR / name
        if path.exists():
            raise SystemExit(f"REFUSING to overwrite existing golden: {path}")
        df.to_csv(path, float_format="%.17g")
        print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    _generate_goldens()
