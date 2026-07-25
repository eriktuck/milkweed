"""Housing: a cost stream that ends on a date, an asset that compounds, and a sale.

Unlike every other retirement expense, housing is three things at once — a cost
(with a *cliff* at mortgage payoff, not a smooth glide), an asset-and-liability
(value compounding, balance amortizing, neither in `buckets`), and an event (a
sale, converting both into a one-time portfolio inflow). New math, not a port
from core/models/. Pinned by tests/test_housing.py.

**Real-dollar frame.** The engine runs entirely in real dollars (see
assumptions.DEFAULTS — there is deliberately no general inflation knob). Housing
breaks that frame in exactly one place: a mortgage payment is *nominally* fixed,
so it shrinks in real terms every year. `inflation` below is the narrow,
purpose-specific carve-out the DEFAULTS docstring anticipates — it deflates the
fixed P&I stream and the outstanding loan balance, and **nothing else**. Every
other rate here (appreciation, rent growth) is a *real* rate, expressed as growth
above inflation.

Ownership of csp keys: `mortgage` and `home_improvements` are excluded from the
generic spend pool upstream so this module can supply them instead (see
assumptions.HOUSING_CSP_KEYS). `bills_utilities` stays in the pool — you pay
utilities either way — as does `taxes`, which means tax *preparation*; property
tax lives inside escrow here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import datetime as dt

import pandas as pd

from core.domain.assumptions import HEALTHCARE_CSP_KEYS, HOUSING_CSP_KEYS
from core.domain.taxes import annual_tax

# ── Asset & carrying-cost defaults ───────────────────────────────────────────
# US housing has run ≈ +1%/yr above inflation over the long run (Shiller). A
# figure derived from a recent purchase (see `derive_real_appreciation`) is a
# good *display* and a bad forward default — a 2019 buyer's ~8%/yr real does not
# extrapolate 30 years. Show the derived number; default the forward one to this.
LONG_RUN_REAL_APPRECIATION: float = 0.01
MAINTENANCE_PCT: float = 0.01       # annual upkeep, % of *current* value (rule of thumb)
LAND_FRACTION: float = 0.20         # non-depreciable share of purchase price

# Deflates the fixed nominal P&I stream and the loan balance ONLY — see the
# module docstring. Nothing else in the engine reads an inflation rate.
DEFAULT_INFLATION: float = 0.025

# ── Rental defaults ──────────────────────────────────────────────────────────
VACANCY_PCT: float = 0.08           # vacancy + management haircut on gross rent
# Rent growth is expressed as growth *above* inflation, so 0.0 means "rent keeps
# pace with inflation". A raw "3% rent growth" input would double-count the
# real-dollar frame. Rent control is a negative real rate.
DEFAULT_REAL_RENT_GROWTH: float = 0.0

# ── Sale defaults (statutory rates + market conventions) ─────────────────────
SELLING_COST_PCT: float = 0.07              # agent commission + closing costs
DEPRECIATION_LIFE_YEARS: float = 27.5       # residential rental, straight-line (IRC §168)
SECTION_121_EXCLUSION_MFJ: float = 500_000.0  # primary-residence gain exclusion, MFJ
RECAPTURE_RATE: float = 0.25                # unrecaptured §1250 gain (statutory)

OCCUPANCIES: tuple[str, ...] = ("primary", "rental")
RESIDENCE_MODES: tuple[str, ...] = ("own", "rent")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Dates → ages
# ═══════════════════════════════════════════════════════════════════════════

def age_at_date(iso_date: str, birth_year: int) -> float | None:
    """Fractional age on an ISO date, given a birth *year*.  (pure)

    The engine indexes everything on age, but config stores `purchase_date` /
    `payoff_date` as ISO "YYYY-MM-DD" strings. This is the one translation, kept
    here so the series functions stay purely numeric.

    Anchored on January 1 to match `assumptions.resolve_assumptions`, which
    derives current age as `year - birth_year` — so Jan 1 of year Y is exactly
    age `Y - birth_year`. Returns None when either input is missing/unparseable.
    """
    if not iso_date or not birth_year:
        return None
    try:
        d = dt.fromisoformat(str(iso_date)[:10]).date()
    except ValueError:
        return None
    year_start = date(d.year, 1, 1)
    year_days = (date(d.year + 1, 1, 1) - year_start).days
    return (d.year - int(birth_year)) + (d - year_start).days / year_days


# ═══════════════════════════════════════════════════════════════════════════
# 2. The loan
# ═══════════════════════════════════════════════════════════════════════════

def amortization_balance(monthly_pi: float, note_rate: float,
                         months_remaining: float) -> float:
    """Outstanding principal as the PV of the remaining P&I payments.  (nominal $)

    A fixed-rate loan's balance at any point *is* the present value of what is
    still owed, discounted at the note rate — so payment + rate + payoff date
    yield the whole amortization schedule with no extra input (no original
    principal, no origination date). Zero at and after payoff, by construction.
    """
    p = float(monthly_pi or 0.0)
    n = float(months_remaining or 0.0)
    if p <= 0.0 or n <= 0.0:
        return 0.0
    i = float(note_rate or 0.0) / 12.0
    if i == 0.0:
        return p * n
    return p * (1.0 - (1.0 + i) ** -n) / i


def _months_remaining(age: float, payoff_age: float | None) -> float:
    """Months of P&I still due at `age`. Zero once past payoff."""
    if payoff_age is None:
        return 0.0
    return max((float(payoff_age) - float(age)) * 12.0, 0.0)


def _pi_year_fraction(age: int, payoff_age: float | None) -> float:
    """Share of the year at `age` during which P&I is still due (0…1).

    A full year while the loan runs, a prorated final year, then zero — so a
    mid-year payoff puts the cliff in the right place. `payoff_age is None`
    means the loan is not modelled as ending within the horizon, so the payment
    runs throughout.
    """
    if payoff_age is None:
        return 1.0
    return min(max(float(payoff_age) - float(age), 0.0), 1.0)


def _deflator(age: float, current_age: float, inflation: float) -> float:
    """Today's-dollar value of one nominal dollar `age - current_age` years out."""
    return (1.0 + float(inflation)) ** -(float(age) - float(current_age))


# ═══════════════════════════════════════════════════════════════════════════
# 3. The asset
# ═══════════════════════════════════════════════════════════════════════════

def value_by_age(current_age: int, end_age: int, current_value: float, *,
                 real_appreciation: float = LONG_RUN_REAL_APPRECIATION) -> pd.Series:
    """Home value by age (real $).  (pure, age-indexed)

    `current_value` is observed at `current_age` and compounds at the *real*
    appreciation rate from there. Returns a Series indexed by age
    (current_age … end_age, inclusive).
    """
    v = float(current_value or 0.0)
    g = float(real_appreciation or 0.0)
    start = int(current_age)
    out = {age: v * (1.0 + g) ** (age - start)
           for age in range(start, int(end_age) + 1)}
    return pd.Series(out, dtype="float64")


def equity_by_age(current_age: int, end_age: int, *,
                  current_value: float = 0.0,
                  real_appreciation: float = LONG_RUN_REAL_APPRECIATION,
                  monthly_payment: float = 0.0,
                  escrow_monthly: float = 0.0,
                  note_rate: float = 0.0,
                  payoff_age: float | None = None,
                  inflation: float = DEFAULT_INFLATION) -> pd.DataFrame:
    """Home value, loan balance and equity by age (real $).  (pure, age-indexed)

    The balance is amortized in *nominal* dollars (that is what a fixed-rate note
    is) and then deflated to today's dollars, so `equity = value - balance` stays
    in one frame. `monthly_payment` is the total PITI; only the P&I half
    (`monthly_payment - escrow_monthly`) amortizes.

    Columns: value, balance, equity. Indexed by age (current_age … end_age).
    """
    start = int(current_age)
    monthly_pi = max(float(monthly_payment or 0.0) - float(escrow_monthly or 0.0), 0.0)
    value = value_by_age(start, end_age, current_value,
                         real_appreciation=real_appreciation)

    balance = pd.Series(
        {age: amortization_balance(monthly_pi, note_rate,
                                   _months_remaining(age, payoff_age))
              * _deflator(age, start, inflation)
         for age in value.index},
        dtype="float64",
    )
    return pd.DataFrame({"value": value, "balance": balance,
                         "equity": value - balance})


# ═══════════════════════════════════════════════════════════════════════════
# 4. Carrying cost
# ═══════════════════════════════════════════════════════════════════════════

def carrying_cost_breakdown(start_age: int, end_age: int, *,
                            current_age: int,
                            current_value: float = 0.0,
                            real_appreciation: float = LONG_RUN_REAL_APPRECIATION,
                            monthly_payment: float = 0.0,
                            escrow_monthly: float = 0.0,
                            payoff_age: float | None = None,
                            maintenance_pct: float = MAINTENANCE_PCT,
                            inflation: float = DEFAULT_INFLATION) -> pd.DataFrame:
    """Annual cost of owning one property, itemized (real $).  (pure, age-indexed)

    The three components behave completely differently, which is the whole reason
    housing cannot be a single tapering line in the generic spend pool:

      * **pi** — principal & interest. Nominally fixed, so it *shrinks* in real
        terms every year, and it **ends** at `payoff_age` (prorated in the payoff
        year). This is the cliff.
      * **escrow** — property tax + insurance. Flat in real terms (tax tracks
        assessed value, insurance tracks replacement cost) and it **continues
        after payoff**. This is the floor beneath the cliff.
      * **maintenance** — a share of *current* value, so it grows with the home.

    `monthly_payment` is the total payment; `escrow_monthly` is the part of it
    that is escrow, so P&I is the difference. Getting that split right is what
    makes the cliff land at the correct height. Columns: pi, escrow, maintenance,
    total. Indexed by age (start_age … end_age, inclusive).
    """
    anchor = int(current_age)
    monthly_pi = max(float(monthly_payment or 0.0) - float(escrow_monthly or 0.0), 0.0)
    escrow_annual = float(escrow_monthly or 0.0) * 12.0
    m_pct = float(maintenance_pct or 0.0)
    v0 = float(current_value or 0.0)
    g = float(real_appreciation or 0.0)

    ages = list(range(int(start_age), int(end_age) + 1))
    pi = [monthly_pi * 12.0 * _deflator(a, anchor, inflation)
          * _pi_year_fraction(a, payoff_age) for a in ages]
    escrow = [escrow_annual for _ in ages]
    maintenance = [m_pct * v0 * (1.0 + g) ** (a - anchor) for a in ages]

    df = pd.DataFrame({"pi": pi, "escrow": escrow, "maintenance": maintenance},
                      index=pd.Index(ages), dtype="float64")
    df["total"] = df["pi"] + df["escrow"] + df["maintenance"]
    return df


def carrying_cost_by_age(start_age: int, end_age: int, *,
                         current_age: int,
                         current_value: float = 0.0,
                         real_appreciation: float = LONG_RUN_REAL_APPRECIATION,
                         monthly_payment: float = 0.0,
                         escrow_monthly: float = 0.0,
                         payoff_age: float | None = None,
                         maintenance_pct: float = MAINTENANCE_PCT,
                         inflation: float = DEFAULT_INFLATION) -> pd.Series:
    """Total annual cost of owning one property (real $).  (pure, age-indexed)

    The `total` column of `carrying_cost_breakdown` — see it for the parameters
    and for what each component does. This is the Series the engine consumes;
    the breakdown is what the module chart draws.
    """
    return carrying_cost_breakdown(
        start_age, end_age, current_age=current_age,
        current_value=current_value, real_appreciation=real_appreciation,
        monthly_payment=monthly_payment, escrow_monthly=escrow_monthly,
        payoff_age=payoff_age, maintenance_pct=maintenance_pct,
        inflation=inflation,
    )["total"]


# ═══════════════════════════════════════════════════════════════════════════
# 5. Rental income
# ═══════════════════════════════════════════════════════════════════════════

def rental_gross_income_by_age(start_age: int, end_age: int, *,
                               current_age: int,
                               monthly_rent: float = 0.0,
                               vacancy_pct: float = VACANCY_PCT,
                               real_rent_growth: float = DEFAULT_REAL_RENT_GROWTH,
                               ) -> pd.Series:
    """Collected rent by age, after the vacancy & management haircut (real $).

    `real_rent_growth` is growth *above* inflation (see the module docstring), so
    the default 0.0 means rent keeps pace. Returns a Series indexed by age
    (start_age … end_age, inclusive).  (pure, age-indexed)
    """
    anchor = int(current_age)
    annual = float(monthly_rent or 0.0) * 12.0 * (1.0 - float(vacancy_pct or 0.0))
    g = float(real_rent_growth or 0.0)
    out = {age: annual * (1.0 + g) ** (age - anchor)
           for age in range(int(start_age), int(end_age) + 1)}
    return pd.Series(out, dtype="float64")


def rental_net_income_by_age(start_age: int, end_age: int, *,
                             current_age: int,
                             current_value: float = 0.0,
                             real_appreciation: float = LONG_RUN_REAL_APPRECIATION,
                             monthly_payment: float = 0.0,
                             escrow_monthly: float = 0.0,
                             payoff_age: float | None = None,
                             maintenance_pct: float = MAINTENANCE_PCT,
                             inflation: float = DEFAULT_INFLATION,
                             monthly_rent: float = 0.0,
                             vacancy_pct: float = VACANCY_PCT,
                             real_rent_growth: float = DEFAULT_REAL_RENT_GROWTH,
                             ) -> pd.Series:
    """Net rental income by age — collected rent less that property's carry.

    A rental is not a business entity here, it is a **tax regime**: its carrying
    costs offset its rent (Schedule E), so only the net is ordinary income. This
    is why a rental's carry never appears in `housing_costs_by_age` — netting it
    here is what makes the downstream tax right.

    Goes **negative** when the carry exceeds the rent; that is a real result and
    must not be clamped. Returns a Series indexed by age (start_age … end_age).
    (pure, age-indexed)
    """
    gross = rental_gross_income_by_age(
        start_age, end_age, current_age=current_age, monthly_rent=monthly_rent,
        vacancy_pct=vacancy_pct, real_rent_growth=real_rent_growth,
    )
    carry = carrying_cost_by_age(
        start_age, end_age, current_age=current_age,
        current_value=current_value, real_appreciation=real_appreciation,
        monthly_payment=monthly_payment, escrow_monthly=escrow_monthly,
        payoff_age=payoff_age, maintenance_pct=maintenance_pct,
        inflation=inflation,
    )
    return gross - carry


# ═══════════════════════════════════════════════════════════════════════════
# 6. The app entry — where you live, as a cost
# ═══════════════════════════════════════════════════════════════════════════

def housing_costs_by_age(start_age: int, end_age: int, *,
                         current_age: int,
                         residence_mode: str = "own",
                         monthly_rent: float = 0.0,
                         real_rent_growth: float = DEFAULT_REAL_RENT_GROWTH,
                         primary: dict | None = None) -> pd.Series:
    """Annual cost of where you live (real $).  (pure, age-indexed)

    "Where you live" and "what you own" are independent facts — someone can own a
    rental and rent their own home. This series is only the first: rent paid, or
    the primary residence's carrying cost. Properties let to tenants never appear
    here; their economics run through `rental_net_income_by_age` instead.

      * `residence_mode == "rent"` — monthly rent × 12, growing at the *real*
        rent growth rate (0 by default; negative models rent control).
      * `residence_mode == "own"` — `carrying_cost_by_age(**primary)`, where
        `primary` is the kwargs of the property whose occupancy is primary.

    Additive, not double-counted: the `mortgage` and `home_improvements` csp keys
    are excluded from the generic spend pool upstream (assumptions.HOUSING_CSP_KEYS)
    whenever a housing config is present, exactly as healthcare does with
    `medical` / `health_insurance`. Returns a Series indexed by age
    (start_age … end_age, inclusive).
    """
    if str(residence_mode) == "rent":
        anchor = int(current_age)
        annual = float(monthly_rent or 0.0) * 12.0
        g = float(real_rent_growth or 0.0)
        out = {age: annual * (1.0 + g) ** (age - anchor)
               for age in range(int(start_age), int(end_age) + 1)}
        return pd.Series(out, dtype="float64")

    if not primary:
        return pd.Series({age: 0.0 for age in
                          range(int(start_age), int(end_age) + 1)}, dtype="float64")
    return carrying_cost_by_age(start_age, end_age, current_age=current_age, **primary)


# ═══════════════════════════════════════════════════════════════════════════
# 7. The event — a sale
# ═══════════════════════════════════════════════════════════════════════════

def sale_proceeds(sale_age: float, *,
                  current_age: int,
                  occupancy: str = "primary",
                  current_value: float = 0.0,
                  real_appreciation: float = LONG_RUN_REAL_APPRECIATION,
                  purchase_price: float = 0.0,
                  purchase_age: float | None = None,
                  land_fraction: float = LAND_FRACTION,
                  monthly_payment: float = 0.0,
                  escrow_monthly: float = 0.0,
                  note_rate: float = 0.0,
                  payoff_age: float | None = None,
                  inflation: float = DEFAULT_INFLATION,
                  selling_cost_pct: float = SELLING_COST_PCT,
                  other_ordinary_income: float = 0.0) -> dict:
    """Itemized net proceeds of selling a property at `sale_age` (real $).  (pure)

    The itemization is the point — recapture in particular surprises people, and
    a sale that omits it looks materially better than it is:

        gross  −  selling costs  −  mortgage payoff
               −  capital-gains tax on (gain − §121 exclusion − recapture)
               −  25% recapture on accumulated depreciation   =  net

    Occupancy selects the tax regime. A **primary** residence gets the §121
    exclusion ($500k MFJ) and takes no depreciation. A **rental** gets no
    exclusion, and 27.5-year straight-line depreciation on the building share of
    basis comes back as unrecaptured §1250 gain at 25%.

    Capital-gains tax is the *marginal* effect of the gain on that year's return
    (`taxes.annual_tax` with and without it), so it stacks on other income
    correctly rather than assuming a flat rate. MFJ throughout, matching
    `core/domain/taxes.py`.

    NOTE (v1): tax basis is nominal historical cost while the sale value is in
    today's dollars, so the taxable gain — and hence the tax — is understated
    relative to a fully nominal calculation. Per-improvement basis tracking is
    out of scope (see the spec's open questions).

    Returns {gross, selling_costs, mortgage_payoff, depreciation, adjusted_basis,
    gain, exclusion, recapture_amount, recapture_tax, taxable_gain,
    capital_gains_tax, net}.
    """
    anchor = int(current_age)
    age = float(sale_age)
    is_rental = str(occupancy) == "rental"

    gross = float(current_value or 0.0) * (1.0 + float(real_appreciation or 0.0)) ** (age - anchor)
    selling_costs = gross * float(selling_cost_pct or 0.0)

    monthly_pi = max(float(monthly_payment or 0.0) - float(escrow_monthly or 0.0), 0.0)
    mortgage_payoff = (amortization_balance(monthly_pi, note_rate,
                                            _months_remaining(age, payoff_age))
                       * _deflator(age, anchor, inflation))

    basis = float(purchase_price or 0.0)
    depreciation = 0.0
    if is_rental and purchase_age is not None:
        years_owned = max(age - float(purchase_age), 0.0)
        depreciable = basis * (1.0 - float(land_fraction or 0.0))
        depreciation = min(depreciable * years_owned / DEPRECIATION_LIFE_YEARS, depreciable)

    adjusted_basis = basis - depreciation
    gain = max(gross - selling_costs - adjusted_basis, 0.0)

    exclusion = 0.0 if is_rental else min(gain, SECTION_121_EXCLUSION_MFJ)
    # Unrecaptured §1250 gain: the lesser of depreciation taken and the gain.
    recapture_amount = min(depreciation, gain)
    recapture_tax = recapture_amount * RECAPTURE_RATE
    taxable_gain = max(gain - exclusion - recapture_amount, 0.0)

    other = float(other_ordinary_income or 0.0)
    capital_gains_tax = annual_tax(other, taxable_gain) - annual_tax(other, 0.0)

    net = gross - selling_costs - mortgage_payoff - capital_gains_tax - recapture_tax
    return {
        "gross": gross,
        "selling_costs": selling_costs,
        "mortgage_payoff": mortgage_payoff,
        "depreciation": depreciation,
        "adjusted_basis": adjusted_basis,
        "gain": gain,
        "exclusion": exclusion,
        "recapture_amount": recapture_amount,
        "recapture_tax": recapture_tax,
        "taxable_gain": taxable_gain,
        "capital_gains_tax": capital_gains_tax,
        "net": net,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. Derived appreciation (display only)
# ═══════════════════════════════════════════════════════════════════════════

def derive_real_appreciation(purchase_price: float, current_value: float,
                             purchase_year: int, as_of_year: int,
                             inflation: pd.Series | None = None) -> float | None:
    """Realized real appreciation CAGR from purchase to today.  (pure)

    `inflation` is an annual CPI-rate Series indexed by calendar year — the same
    market-data seam `core/domain/returns.py` uses, supplied by
    `core/services/market_data.py` at the page edge. Nominal growth is deflated
    by *realized* CPI over the holding period, using the precise geometric form
    `(1+nominal)/(1+inflation)-1` per that module's convention. With no inflation
    series the result is simply the nominal CAGR.

    **Display only.** This is context for the user ("your home has appreciated
    7.9%/yr above inflation since 2019 — well above the long-run ~1%"), not a
    forward default: a short recent hold extrapolated over 30 years is fantasy.
    The forward assumption defaults to `LONG_RUN_REAL_APPRECIATION`.

    Returns None when the period is empty or the purchase price is unusable.
    """
    p0 = float(purchase_price or 0.0)
    p1 = float(current_value or 0.0)
    n = int(as_of_year) - int(purchase_year)
    if p0 <= 0.0 or p1 <= 0.0 or n <= 0:
        return None

    cum_inflation = 1.0
    if inflation is not None and not inflation.empty:
        for year in range(int(purchase_year), int(as_of_year)):
            if year in inflation.index:
                cum_inflation *= 1.0 + float(inflation.loc[year])

    return ((p1 / p0) / cum_inflation) ** (1.0 / n) - 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 9. Config → series (the wiring edge)
# ═══════════════════════════════════════════════════════════════════════════
#
# Everything above is numeric and age-indexed. This section is the single edge
# where a stored `HousingConfig` becomes those numbers: ISO dates become ages
# (`age_at_date`), a residence/property list fans out into "where you live" vs
# "what you own", and a planned sale becomes a splice plus a lump sum. Keeping
# it here rather than in `scenario.py` honours that module's promise to
# orchestrate rather than introduce math, and lets the splice be tested against
# plain dicts instead of a whole Scenario.

# Fields that make a property block mean something. A `Property` with only a
# label (or none at all) is not a configuration — see `is_configured`.
_MEANINGFUL_PROPERTY_FIELDS: tuple[str, ...] = (
    "current_value", "purchase_price", "monthly_payment", "escrow_monthly",
    "monthly_rent_income",
)


@dataclass(frozen=True)
class ResolvedHousing:
    """Everything `build_scenario` needs from a `HousingConfig`, as series.

    Three channels, deliberately separate because the engine treats them
    differently (and because netting them would get the tax wrong):

      * `costs_by_age` — where you live, a positive cost, rides the spend channel.
      * `rental_net_income_by_age` — rent less that property's carry, **taxed**
        ordinary income. May be negative; that is a real result, not an error.
      * `inflows_by_age` — a sale, an event rather than a series, landing in the
        taxable bucket.

    `equity` and `sale` are for display (the Phase 4 chart and the Phase 5
    breakdown table); nothing in the engine reads them.
    """
    costs_by_age: pd.Series              # where you live — the spend channel
    rental_net_income_by_age: pd.Series  # the taxed ordinary channel; may be < 0
    inflows_by_age: dict[int, float]     # {sale age: net proceeds to the portfolio}
    equity: pd.DataFrame                 # value / balance / equity, all properties
    sale: dict | None                    # itemized proceeds + age/label/disposition


def is_configured(housing_cfg: dict | None) -> bool:
    """True when the housing block actually says something.  (pure)

    The trap this exists for: `HousingConfig()` dumps to a dict of all-Nones,
    which is truthy. Gating on truthiness would drop `mortgage` and
    `home_improvements` from the generic spend pool *and* hand back a zero cost
    stream — silently cutting the nest-egg goal for a user who configured
    nothing. A block counts only if it names where you live or describes at
    least one property in economic terms.
    """
    cfg = housing_cfg or {}
    if (cfg.get("residence") or {}).get("mode"):
        return True
    return any(
        any(prop.get(f) is not None for f in _MEANINGFUL_PROPERTY_FIELDS)
        for prop in (cfg.get("properties") or [])
        if prop
    )


def spend_exclude_keys(housing_cfg: dict | None) -> frozenset[str]:
    """csp keys the generic spend pool must drop for this user.  (pure)

    Healthcare always; housing only when a block is configured — a user without
    housing keeps the `mortgage` / `home_improvements` rows and their 0.95/0.90
    PHASE_FACTORS taper, exactly as before this module existed. This is the one
    expression of that rule; `scenario.build_scenario` and the Coast page's
    expense-table seed both call it so the two cannot drift.
    """
    if is_configured(housing_cfg):
        return HEALTHCARE_CSP_KEYS | HOUSING_CSP_KEYS
    return HEALTHCARE_CSP_KEYS


def _opt(value, default: float) -> float:
    """Config value with a code-side default, treating 0.0 as a real answer."""
    return default if value is None else float(value)


def _occupancy(prop: dict) -> str:
    """A property with no stated occupancy is the home you live in.

    Matches `sale_proceeds`' own default and the overwhelmingly common case of
    one house; a rental is the deliberate, explicit choice because it selects a
    whole tax regime.
    """
    return str(prop.get("occupancy") or "primary")


def _carry_kwargs(prop: dict, *, birth_year: int | None, inflation: float) -> dict:
    """The `carrying_cost_by_age` kwargs for one property (no `note_rate`)."""
    return dict(
        current_value=float(prop.get("current_value") or 0.0),
        real_appreciation=_opt(prop.get("real_appreciation"), LONG_RUN_REAL_APPRECIATION),
        monthly_payment=float(prop.get("monthly_payment") or 0.0),
        escrow_monthly=float(prop.get("escrow_monthly") or 0.0),
        payoff_age=age_at_date(prop.get("payoff_date"), birth_year),
        maintenance_pct=_opt(prop.get("maintenance_pct"), MAINTENANCE_PCT),
        inflation=inflation,
    )


def _rent_kwargs(prop: dict) -> dict:
    """The rental-income kwargs for one property let to tenants."""
    return dict(
        monthly_rent=float(prop.get("monthly_rent_income") or 0.0),
        vacancy_pct=_opt(prop.get("vacancy_pct"), VACANCY_PCT),
        real_rent_growth=_opt(prop.get("real_rent_growth"), DEFAULT_REAL_RENT_GROWTH),
    )


def _new_home_kwargs(prop: dict, value_at_sale: float, new_price: float | None, *,
                     inflation: float) -> dict:
    """Carrying cost of the home bought with the sale proceeds.  (v1 rule)

    `PropertySale` carries only a price, so the rest is inferred. The purchase is
    **cash** — the spec's `proceeds − new price` spends the whole price, so there
    is no new loan and no P&I. Escrow scales with price (property tax is ad
    valorem; insurance tracks replacement cost) and this household's maintenance
    rate and appreciation assumption carry over. A missing price is a
    like-for-like move.

    Assumes: no mortgage on the new home, escrow linear in price, the same real
    appreciation, no buy-side closing costs, and no fresh §121 clock (which only
    matters for a second sale, not modelled in v1).
    """
    price = float(new_price or 0.0) or float(value_at_sale)
    ratio = (price / value_at_sale) if value_at_sale else 1.0
    return dict(
        current_value=price,
        real_appreciation=_opt(prop.get("real_appreciation"), LONG_RUN_REAL_APPRECIATION),
        monthly_payment=0.0,
        escrow_monthly=float(prop.get("escrow_monthly") or 0.0) * ratio,
        payoff_age=None,
        maintenance_pct=_opt(prop.get("maintenance_pct"), MAINTENANCE_PCT),
        inflation=inflation,
    )


def _splice(pre: pd.Series, post: pd.Series, sale_age: int | None) -> pd.Series:
    """`pre` before `sale_age`, `post` from `sale_age` on (shared index)."""
    if sale_age is None:
        return pre
    return pre.where(pre.index < int(sale_age), post)


def _zero_from(series_or_frame, sale_age: int | None):
    """Zero every row from `sale_age` on — the property is no longer owned."""
    if sale_age is None:
        return series_or_frame
    keep = pd.Series(series_or_frame.index < int(sale_age),
                     index=series_or_frame.index)
    if isinstance(series_or_frame, pd.DataFrame):
        return series_or_frame.where(keep, 0.0, axis=0)
    return series_or_frame.where(keep, 0.0)


def _sale_age(prop: dict, current_age: int) -> int | None:
    """The planned sale age, or None if there isn't one we can model.

    A sale in the past is ignored rather than back-dated: the proceeds are
    already in the portfolio the caller passed in.
    """
    sale = prop.get("sale") or {}
    age = sale.get("age")
    if age is None:
        return None
    age = int(age)
    return age if age >= int(current_age) else None


def resolve_housing(
    housing_cfg: dict | None,
    *,
    current_age: int,
    retirement_age: int,
    death_age: int,
    birth_year: int | None,
    other_ordinary_income_at_sale: float = 0.0,
) -> ResolvedHousing | None:
    """`HousingConfig` → the cost stream, the rental channel, the sale.  (pure)

    Returns **None** when the block says nothing (`is_configured`). That None is
    the single gate `build_scenario` reads: no resolution → no csp exclusion →
    byte-identical results for every user without housing.

    Two windows, deliberately different. The cost and rental series span
    `retirement_age … death_age` — before retirement the property's economics are
    already in the CSP plan (carrying costs are budgeted, rent carries the
    `income` label, and the contribution lever is net of both), so adding them
    would double-count. `equity` spans `current_age … death_age` because the
    module's chart draws the whole path.

    A sale has four independent effects, and separating them is what stops
    selling a rental from silently changing where you live:

      1. net proceeds (less a new home's price, if buying) land in `inflows_by_age`;
      2. a rental's net income stops;
      3. the cost stream switches — but only if the sold property is the one you
         live in (`occupancy == "primary"` *and* `residence.mode == "own"`);
      4. that property leaves the equity roll-up.

    `other_ordinary_income_at_sale` stacks the capital gain on the household's
    other ordinary income for that year, so the gains tax is marginal rather than
    flat. Any rental income the household still collects that year is added
    internally. Passing the true figure needs the forward walk's trad draw, which
    depends on the sale — a two-pass fixed point, deferred; v1 callers pass 0.0.
    """
    if not is_configured(housing_cfg):
        return None

    cfg = housing_cfg or {}
    anchor = int(current_age)
    start, end = int(retirement_age), int(death_age)
    inflation = _opt(cfg.get("inflation"), DEFAULT_INFLATION)

    residence = cfg.get("residence") or {}
    mode = str(residence.get("mode") or "own")
    properties = [p for p in (cfg.get("properties") or []) if p]

    cost_index = pd.Index(range(start, end + 1))
    # Named because `equity` is the one result that leaves here as a DataFrame —
    # the module chart and its golden both index it by age.
    equity_index = pd.Index(range(anchor, end + 1), name="age")

    # ── 2 & 4: per-property rental income and equity, each ending at its own sale
    rental_net = pd.Series(0.0, index=cost_index, dtype="float64")
    equity = pd.DataFrame(0.0, index=equity_index,
                          columns=["value", "balance", "equity"], dtype="float64")

    for prop in properties:
        sold_at = _sale_age(prop, anchor)
        carry = _carry_kwargs(prop, birth_year=birth_year, inflation=inflation)

        if _occupancy(prop) == "rental":
            net = rental_net_income_by_age(start, end, current_age=anchor,
                                           **carry, **_rent_kwargs(prop))
            rental_net = rental_net.add(_zero_from(net, sold_at), fill_value=0.0)

        eq = equity_by_age(anchor, end, note_rate=float(prop.get("note_rate") or 0.0),
                           **{k: v for k, v in carry.items() if k != "maintenance_pct"})
        equity = equity.add(_zero_from(eq, sold_at), fill_value=0.0)

    # ── 1: the sale itself. Gains stack on the ordinary income the household
    # still has that year — which is why this runs after the rental netting.
    inflows: dict[int, float] = {}
    sale_detail: dict | None = None

    for prop in properties:
        sold_at = _sale_age(prop, anchor)
        if sold_at is None:
            continue
        sale = prop.get("sale") or {}
        other = (float(other_ordinary_income_at_sale)
                 + max(float(rental_net.get(sold_at, 0.0)), 0.0))
        proceeds = sale_proceeds(
            sold_at, current_age=anchor, occupancy=_occupancy(prop),
            current_value=float(prop.get("current_value") or 0.0),
            real_appreciation=_opt(prop.get("real_appreciation"),
                                   LONG_RUN_REAL_APPRECIATION),
            purchase_price=float(prop.get("purchase_price") or 0.0),
            purchase_age=age_at_date(prop.get("purchase_date"), birth_year),
            land_fraction=_opt(prop.get("land_fraction"), LAND_FRACTION),
            monthly_payment=float(prop.get("monthly_payment") or 0.0),
            escrow_monthly=float(prop.get("escrow_monthly") or 0.0),
            note_rate=float(prop.get("note_rate") or 0.0),
            payoff_age=age_at_date(prop.get("payoff_date"), birth_year),
            inflation=inflation,
            selling_cost_pct=_opt(sale.get("selling_cost_pct"), SELLING_COST_PCT),
            other_ordinary_income=other,
        )
        # Buying replaces part (or more) of the proceeds. A price above the
        # proceeds is a NEGATIVE inflow — a real portfolio draw, deliberately not
        # clamped: hiding it would hide a purchase the household cannot afford.
        new_price = (float(sale.get("new_home_price") or 0.0)
                     if str(sale.get("disposition")) == "buy" else 0.0)
        inflows[sold_at] = inflows.get(sold_at, 0.0) + proceeds["net"] - new_price

        detail = {**proceeds, "age": sold_at, "label": prop.get("label"),
                  "disposition": sale.get("disposition"),
                  "new_home_price": new_price or None}
        if sale_detail is None or _occupancy(prop) == "primary":
            sale_detail = detail

    # ── 3: where you live, and the switch if you sell it ──────────────────────
    primary = next((p for p in properties if _occupancy(p) == "primary"), None)
    costs = housing_costs_by_age(
        start, end, current_age=anchor, residence_mode=mode,
        monthly_rent=float(residence.get("monthly_rent") or 0.0),
        real_rent_growth=_opt(residence.get("real_rent_growth"),
                              DEFAULT_REAL_RENT_GROWTH),
        primary=(_carry_kwargs(primary, birth_year=birth_year, inflation=inflation)
                 if primary else None),
    )

    if mode == "own" and primary is not None:
        sold_at = _sale_age(primary, anchor)
        if sold_at is not None:
            costs = _splice(costs, _post_sale_costs(
                primary, sold_at, start, end,
                residence=residence, anchor=anchor, inflation=inflation,
            ), sold_at)

    return ResolvedHousing(
        costs_by_age=costs,
        rental_net_income_by_age=rental_net,
        inflows_by_age=inflows,
        equity=equity,
        sale=sale_detail,
    )


def _post_sale_costs(prop: dict, sale_age: int, start: int, end: int, *,
                     residence: dict, anchor: int, inflation: float) -> pd.Series:
    """Where you live *after* selling the home you lived in.  (pure, age-indexed)

    Both dispositions quote their inputs in today's dollars and apply them from
    the sale age on, so the series is anchored at `sale_age`, not at the current
    age — which is exactly why `carrying_cost_by_age` keeps its anchor separate
    from its window. Post-move rent falls back to `Residence.monthly_rent` when
    the sale doesn't name its own.
    """
    sale = prop.get("sale") or {}
    if str(sale.get("disposition")) == "buy":
        value_at_sale = (float(prop.get("current_value") or 0.0)
                         * (1.0 + _opt(prop.get("real_appreciation"),
                                       LONG_RUN_REAL_APPRECIATION)) ** (sale_age - anchor))
        return carrying_cost_by_age(
            start, end, current_age=sale_age,
            **_new_home_kwargs(prop, value_at_sale, sale.get("new_home_price"),
                               inflation=inflation))

    rent = sale.get("monthly_rent")
    growth = sale.get("real_rent_growth")
    return housing_costs_by_age(
        start, end, current_age=sale_age, residence_mode="rent",
        monthly_rent=float((rent if rent is not None
                            else residence.get("monthly_rent")) or 0.0),
        real_rent_growth=_opt(growth if growth is not None
                              else residence.get("real_rent_growth"),
                              DEFAULT_REAL_RENT_GROWTH),
    )
