"""Unified lifetime projection engine (shared by Forecast & Retirement).

The two pages are one timeline cut at retirement age — Forecast works **forward**
through accumulation (working + coast) to a nest-egg number; Retirement works the
drawdown **backward/forward** from that number to death. Historically each page
carried its own copy of the recurrence; this module stitches the two halves that
already agreed into a single bucket-level (taxable / trad / roth) walk from
`current_age` to `death_age`, so both pages — and the future tax-allocation
feature — read one engine.

Three internal phases over the same bucket state, switching the return at the
retirement seam (returns are phase-scoped — `r_accum` ≠ `r_retire`, mirroring a
real glide toward less-risky assets in retirement):

    working    (age < coast):       each bucket  V·(1+r_accum) + allocation[bucket]
    coast      (coast ≤ age < ret):  each bucket  V·(1+r_accum)            (no new money)
    retirement (age ≥ ret):          allocate → RMD → grow at r_retire
                                      (the tax-aware draw lifted from
                                       retirement.project_retirement_taxaware)

Pure — no Firestore or network. The accumulation recurrence is identical to
forecast.project_portfolio's working phase, so the aggregate of the three buckets
at retirement reconciles with the Forecast page (gated by
scripts/verify_lifetime_engine.py). The retirement-phase loop body is the same
math as retirement.project_retirement_taxaware, so that function becomes a thin
wrapper with byte-identical output.

See .planning/SPEC-rationalize-pages.md for the full design.
"""

from __future__ import annotations

import pandas as pd

from core.services.retirement import (
    TAXABLE_GAIN_FRACTION,
    _allocate_withdrawals,
    _as_age_lookup,
    annual_tax,
    phase_for_age,
    required_min_distribution,
)

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
    r_accum: float,
    r_retire: float,
    ss_by_age: pd.Series | dict | None = None,
    healthcare_by_age: pd.Series | dict | None = None,
    rmd_start: int = 75,
    gain_fraction: float = TAXABLE_GAIN_FRACTION,
) -> pd.DataFrame:
    """Year-by-year bucket-level projection from current_age to death_age.

    `buckets` / `allocation` are {taxable, trad, roth} maps — current balances and
    the annual contribution split. Contributions are applied while `age < coast_age`
    and grow at `r_accum`; from `coast_age` to `retirement_age` balances grow with
    no new money; from `retirement_age` the tax-aware drawdown runs at `r_retire`,
    funding each year's net living need (phase spend + healthcare − Social Security)
    after tax, withdrawing taxable → trad → roth with RMDs forced from trad at
    `rmd_start`.

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
    ra = float(r_accum)
    rr = float(r_retire)
    ss = _as_age_lookup(ss_by_age)
    health = _as_age_lookup(healthcare_by_age)
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

        taxable = taxable * (1 + ra) + year_alloc["taxable"]
        trad = trad * (1 + ra) + year_alloc["trad"]
        roth = roth * (1 + ra) + year_alloc["roth"]

    # ── Retirement: tax-aware drawdown (ages retirement_age … death_age) ───────────
    for age in range(retirement_age, death_age + 1):
        spend_phase = phase_for_age(age, slow_go_age, no_go_age)
        spend = float(spend_by_phase.get(spend_phase, 0.0))
        hc = health.get(age, 0.0)
        income = ss.get(age, 0.0)
        total_before = taxable + trad + roth

        cash_need = max(spend + hc - income, 0.0)   # SS untaxed offset (v1)
        rmd = required_min_distribution(trad, age, rmd_start)

        # Fixed-point on tax: the draw must fund cash_need + the tax it triggers.
        tax = 0.0
        w_taxable = w_trad = w_roth = excess_rmd = 0.0
        for _ in range(8):
            target = cash_need + tax
            w_taxable, w_trad, w_roth, excess_rmd, _short = _allocate_withdrawals(
                target, taxable, trad, roth, rmd)
            ordinary_income = w_trad                 # all trad draws are ordinary
            ltcg_gain = w_taxable * gain_frac
            new_tax = annual_tax(ordinary_income, ltcg_gain)
            if abs(new_tax - tax) < 1.0:
                tax = new_tax
                break
            tax = new_tax

        withdrawal = w_taxable + w_trad + w_roth
        net_spend = withdrawal - tax - excess_rmd + income   # after-tax living cash

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

        # Apply draws; reinvest any forced-RMD excess into taxable; then grow.
        taxable = max(taxable - w_taxable + excess_rmd, 0.0)
        trad = max(trad - w_trad, 0.0)
        roth = max(roth - w_roth, 0.0)
        taxable *= (1 + rr)
        trad *= (1 + rr)
        roth *= (1 + rr)

    return pd.DataFrame(rows).set_index("age")
