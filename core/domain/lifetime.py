"""Unified lifetime projection engine.

One timeline cut at retirement age — accumulation works **forward** through
working + coast to a nest-egg number; the drawdown works from that number to
death. Historically the split Forecast/Retirement pages each carried their own
copy of the recurrence; this module is the single bucket-level (taxable / trad /
roth) walk from `current_age` to `death_age` that both halves — and the future
tax-allocation feature — read. Moved verbatim from core/services/projection.py
(Phase 5 of the core/ restructuring); pinned by tests/test_lifetime.py.

Three internal phases over the same bucket state, switching the return at the
retirement seam (returns are phase-scoped — `r_accum` ≠ `r_retire`, mirroring a
real glide toward less-risky assets in retirement):

    working    (age < coast):       each bucket  V·(1+r_accum) + allocation[bucket]
    coast      (coast ≤ age < ret):  each bucket  V·(1+r_accum)            (no new money)
    retirement (age ≥ ret):          allocate → RMD → grow at r_retire

The two thin slices of the engine live here too: `project_retirement_taxaware`
(retirement-only window, the legacy Retirement-page frame) and
`project_balances_to_retirement` (accumulation-only window, the at-retirement
bucket projector) — both byte-identical to the loops they replaced.

Pure — no Firestore or network. See .planning/SPEC-rationalize-pages.md for the
full design.
"""

from __future__ import annotations

import pandas as pd

from core.domain.assumptions import phase_for_age
from core.domain.drawdown import (
    TAXABLE_GAIN_FRACTION,
    allocate_withdrawals,
    as_age_lookup,
)
from core.domain.rmd import required_min_distribution
from core.domain.taxes import annual_tax

# Column order for the unified frame. Accumulation rows zero-fill the drawdown
# columns; retirement rows zero-fill `contribution`. `life_phase` is the timeline
# phase (working/coast/retirement); `spend_phase` is the go_go/slow_go/no_go
# bucket (only meaningful in retirement, "" during accumulation).
_BUCKETS = ("taxable", "trad", "roth")
_DRAW_COLS = (
    "spend", "healthcare", "income", "rmd", "withdrawal",
    "w_taxable", "w_trad", "w_roth", "ordinary_income", "ltcg_gain",
    "tax", "net_spend",
)


def simulate_lifetime(
    buckets: dict,
    allocation: dict,
    current_age: int,
    coast_age: int,
    retirement_age: int,
    death_age: int,
    spend_by_phase: dict,
    slow_go_age: int,
    no_go_age: int,
    r_accum: float | dict | pd.Series,
    r_retire: float,
    ss_by_age: pd.Series | dict | None = None,
    healthcare_by_age: pd.Series | dict | None = None,
    rmd_start: int = 75,
    gain_fraction: float = TAXABLE_GAIN_FRACTION,
    inflows_by_age: pd.Series | dict | None = None,
    ordinary_income_by_age: pd.Series | dict | None = None,
) -> pd.DataFrame:
    """Year-by-year bucket-level projection from current_age to death_age.

    `buckets` / `allocation` are {taxable, trad, roth} maps — current balances and
    the annual contribution split. `r_accum` is the accumulation real return: a
    scalar (flat rate) or an `{age: rate}` map / Series for a target-date glide path
    (.planning SPEC delta) — each accumulation year grows at its own rate; missing
    ages fall back to the path's terminal (oldest) rate. Contributions are applied
    while `age < coast_age` and grow at `r_accum`; from `coast_age` to
    `retirement_age` balances grow with
    no new money; from `retirement_age` the tax-aware drawdown runs at `r_retire`,
    funding each year's net living need (phase spend + healthcare − Social Security)
    after tax, withdrawing taxable → trad → roth with RMDs forced from trad at
    `rmd_start`.

    Two optional channels carry money that is neither a contribution nor a
    portfolio draw. Both are generic — housing is only their first consumer:

      * `inflows_by_age` — one-time lump sums (a home sale, an inheritance, a
        business sale) landing in **taxable** just before that year's growth.
        They need their own hook because `cash_need` is clamped at zero, which
        would silently clip a $500k inflow; and they work in both the
        accumulation and drawdown legs. Any tax on the event is the caller's to
        net out — what arrives here is proceeds.
      * `ordinary_income_by_age` — recurring income that is **taxed** (net rental
        income; eventually the Social Security taxation fix). Unlike `ss_by_age`,
        which rides an untaxed channel as a documented v1 simplification, this
        offsets the living need *and* enters the tax fixed-point alongside the
        trad draw. A negative value (a rental running at a loss) raises the cash
        need and shelters ordinary income, which `annual_tax` floors at zero.

    Returns a DataFrame indexed by `age` with the bucket balances
    (taxable/trad/roth/total), `contribution`, the drawdown flow columns
    (spend, healthcare, income, rmd, withdrawal, w_taxable, w_trad, w_roth,
    ordinary_income, ltcg_gain, tax, net_spend), and two phase columns
    (`life_phase`, `spend_phase`). `total` for an age is the balance **entering**
    that age-year, before that year's contribution or draw. All flows are positive
    magnitudes; the UI negates draws for display (never abs()).
    """
    current_age = int(current_age)
    coast_age = int(coast_age)
    retirement_age = int(retirement_age)
    death_age = int(death_age)
    # Accumulation return: scalar (flat) or {age: rate} glide path. For a glide,
    # missing ages fall back to the terminal (oldest) rate — the most de-risked.
    if isinstance(r_accum, (int, float)):
        _ra_const = float(r_accum)
        ra_at = lambda age: _ra_const  # noqa: E731
    else:
        _ra_map = as_age_lookup(r_accum)
        _ra_fallback = _ra_map[max(_ra_map)] if _ra_map else 0.0
        ra_at = lambda age: _ra_map.get(age, _ra_fallback)  # noqa: E731
    rr = float(r_retire)
    ss = as_age_lookup(ss_by_age)
    health = as_age_lookup(healthcare_by_age)
    inflow = as_age_lookup(inflows_by_age)
    ord_income = as_age_lookup(ordinary_income_by_age)
    gain_frac = float(gain_fraction)

    taxable = float(buckets.get("taxable", 0.0))
    trad = float(buckets.get("trad", 0.0))
    roth = float(buckets.get("roth", 0.0))
    alloc = {b: float(allocation.get(b, 0.0)) for b in _BUCKETS}

    rows: list[dict] = []

    def _blank_draw() -> dict:
        return {c: 0.0 for c in _DRAW_COLS}

    # ── Accumulation: working + coast (ages current_age … retirement_age − 1) ──────
    for age in range(current_age, retirement_age):
        contributing = age < coast_age
        life_phase = "working" if contributing else "coast"
        year_alloc = alloc if contributing else {b: 0.0 for b in _BUCKETS}

        row = {
            "age": age,
            "total": taxable + trad + roth,
            "taxable": taxable, "trad": trad, "roth": roth,
            "contribution": sum(year_alloc.values()),
            "life_phase": life_phase, "spend_phase": "",
            **_blank_draw(),
        }
        rows.append(row)

        ra = ra_at(age)
        taxable = (taxable + inflow.get(age, 0.0)) * (1 + ra) + year_alloc["taxable"]
        trad = trad * (1 + ra) + year_alloc["trad"]
        roth = roth * (1 + ra) + year_alloc["roth"]

    # ── Retirement: tax-aware drawdown (ages retirement_age … death_age) ───────────
    for age in range(retirement_age, death_age + 1):
        spend_phase = phase_for_age(age, slow_go_age, no_go_age)
        spend = float(spend_by_phase.get(spend_phase, 0.0))
        hc = health.get(age, 0.0)
        income = ss.get(age, 0.0)
        other_ord = ord_income.get(age, 0.0)
        total_before = taxable + trad + roth

        # SS is an untaxed offset (v1); `other_ord` is spendable *and* taxed below.
        cash_need = max(spend + hc - income - other_ord, 0.0)
        rmd = required_min_distribution(trad, age, rmd_start)

        # Fixed-point on tax: the draw must fund cash_need + the tax it triggers.
        tax = 0.0
        w_taxable = w_trad = w_roth = excess_rmd = 0.0
        for _ in range(8):
            target = cash_need + tax
            w_taxable, w_trad, w_roth, excess_rmd, _short = allocate_withdrawals(
                target, taxable, trad, roth, rmd)
            # All trad draws are ordinary, plus any taxed outside income.
            ordinary_income = w_trad + other_ord
            ltcg_gain = w_taxable * gain_frac
            new_tax = annual_tax(ordinary_income, ltcg_gain)
            if abs(new_tax - tax) < 1.0:
                tax = new_tax
                break
            tax = new_tax

        withdrawal = w_taxable + w_trad + w_roth
        # After-tax living cash: draws plus both income channels.
        net_spend = withdrawal - tax - excess_rmd + income + other_ord

        rows.append({
            "age": age, "total": total_before,
            "taxable": taxable, "trad": trad, "roth": roth,
            "contribution": 0.0,
            "spend": spend, "healthcare": hc, "income": income, "rmd": rmd,
            "withdrawal": withdrawal, "w_taxable": w_taxable, "w_trad": w_trad,
            "w_roth": w_roth, "ordinary_income": ordinary_income,
            "ltcg_gain": ltcg_gain, "tax": tax, "net_spend": net_spend,
            "life_phase": "retirement", "spend_phase": spend_phase,
        })

        # Apply draws; reinvest any forced-RMD excess and lump-sum inflows into
        # taxable; then grow.
        taxable = max(taxable - w_taxable + excess_rmd, 0.0) + inflow.get(age, 0.0)
        trad = max(trad - w_trad, 0.0)
        roth = max(roth - w_roth, 0.0)
        taxable *= (1 + rr)
        trad *= (1 + rr)
        roth *= (1 + rr)

    return pd.DataFrame(rows).set_index("age")


# ── Thin slices of the engine (retirement-only / accumulation-only windows) ──────

def project_retirement_taxaware(
    balances: dict,
    retirement_age: int,
    death_age: int,
    spend_by_phase: dict[str, float],
    slow_go_age: int,
    no_go_age: int,
    real_return: float,
    ss_by_age: pd.Series | dict | None = None,
    healthcare_by_age: pd.Series | dict | None = None,
    rmd_start: int = 75,
    taxable_gain_fraction: float = TAXABLE_GAIN_FRACTION,
) -> pd.DataFrame:
    """Year-by-year tax-aware drawdown across taxable / trad / roth buckets.

    Each year: cover the net living need (phase spend + healthcare − Social
    Security) with after-tax cash, withdrawing taxable → trad → roth, RMDs forced
    from trad at `rmd_start`. Taxes (ordinary on trad, LTCG on the taxable gain
    fraction, Roth tax-free) are solved with a short fixed-point loop since the
    draw must cover spending *and* the tax it generates. Surviving balances grow
    at `real_return`.

    Returns a DataFrame indexed by `age` with columns:
        total, taxable, trad, roth, spend, healthcare, income (SS), rmd,
        withdrawal, w_taxable, w_trad, w_roth, ordinary_income, ltcg_gain,
        tax, net_spend, phase

    `total` is the portfolio entering each age-year (before that year's draw).
    All flows are positive magnitudes; the UI negates draws for display.

    Thin wrapper over the shared `simulate_lifetime` engine (retirement slice):
    with no accumulation window (current_age = coast_age = retirement_age) the
    engine runs only the tax-aware drawdown, byte-identical to the original loop.
    The engine's `spend_phase` column is the `phase` returned here.
    """
    df = simulate_lifetime(
        buckets=balances, allocation={},
        current_age=retirement_age, coast_age=retirement_age,
        retirement_age=retirement_age, death_age=death_age,
        spend_by_phase=spend_by_phase, slow_go_age=slow_go_age, no_go_age=no_go_age,
        r_accum=real_return, r_retire=real_return,
        ss_by_age=ss_by_age, healthcare_by_age=healthcare_by_age,
        rmd_start=rmd_start, gain_fraction=taxable_gain_fraction,
    )
    out = df.rename(columns={"spend_phase": "phase"})
    return out[[
        "total", "taxable", "trad", "roth", "spend", "healthcare", "income",
        "rmd", "withdrawal", "w_taxable", "w_trad", "w_roth", "ordinary_income",
        "ltcg_gain", "tax", "net_spend", "phase",
    ]]


def project_balances_to_retirement(
    current_balances: dict,
    annual_allocation: dict,
    years: int,
    real_return: float,
) -> dict:
    """Grow each tax bucket from today to retirement.  (pure)

    Compounds every bucket forward `years` years, adding that bucket's annual
    contribution each year, using the working-phase recurrence
    `V' = V·(1+r) + contribution` — so the aggregate of the three buckets
    reconciles with the legacy Forecast projection given the same start, total
    contribution, and real return (pinned by tests/test_lifetime.py).

    `current_balances` / `annual_allocation` are {taxable, trad, roth} dicts.
    Returns {taxable, trad, roth, total} projected at retirement. With years ≤ 0
    (already at/after retirement) the projection is just the current balances.

    Thin wrapper over the shared `simulate_lifetime` engine (accumulation slice):
    runs an all-working window of `years` and reads the balances entering
    retirement. The recurrence is identical, so the result is unchanged.
    """
    n = max(int(years), 0)
    df = simulate_lifetime(
        buckets=current_balances, allocation=annual_allocation,
        current_age=0, coast_age=n, retirement_age=n, death_age=n,
        spend_by_phase={}, slow_go_age=n, no_go_age=n,
        r_accum=real_return, r_retire=real_return,
    )
    row = df.loc[n]
    out = {b: float(row[b]) for b in ("taxable", "trad", "roth")}
    out["total"] = out["taxable"] + out["trad"] + out["roth"]
    return out
