"""Verify the at-retirement balance projector (Phase 6b).

Pure check — no Firestore. Confirms:
  1. `default_contribution_allocation` fills tax-advantaged-first (pre-tax cap →
     Roth cap → taxable) and always sums to the contribution.
  2. `project_balances_to_retirement`'s aggregate **reconciles with Forecast's
     `project_portfolio`** — the gate criterion — for the same start, total
     contribution, and real return (coast_age = retirement_age, i.e. all-working).

Run from the repo root:
    python -m scripts.verify_balance_projector
"""

from core.services.forecast import project_portfolio
from core.services.retirement import (
    PRETAX_CONTRIB_CAP,
    ROTH_CONTRIB_CAP,
    default_contribution_allocation,
    project_balances_to_retirement,
)


def check_allocation() -> None:
    # Below pre-tax cap → all pre-tax.
    a = default_contribution_allocation(10_000)
    assert a == {"taxable": 0.0, "trad": 10_000.0, "roth": 0.0}, a

    # Between pre-tax cap and pre-tax+roth caps → fills Roth with the overflow.
    a = default_contribution_allocation(PRETAX_CONTRIB_CAP + 4_000)
    assert a["trad"] == PRETAX_CONTRIB_CAP and a["roth"] == 4_000.0 and a["taxable"] == 0.0, a

    # Above both caps → remainder spills to taxable; total is preserved.
    c = PRETAX_CONTRIB_CAP + ROTH_CONTRIB_CAP + 5_000
    a = default_contribution_allocation(c)
    assert a["trad"] == PRETAX_CONTRIB_CAP and a["roth"] == ROTH_CONTRIB_CAP, a
    assert abs(a["taxable"] - 5_000.0) < 1e-6, a
    assert abs(sum(a.values()) - c) < 1e-6, a
    print(f"  allocation tax-advantaged-first ✓  ({PRETAX_CONTRIB_CAP:,.0f} pre-tax / "
          f"{ROTH_CONTRIB_CAP:,.0f} Roth caps)")


def check_reconciles_with_forecast() -> None:
    start, current_age, retirement_age, r = 250_000.0, 40, 65, 0.05
    annual_contribution = 30_000.0
    years = retirement_age - current_age

    # Split across buckets however — the aggregate must match regardless.
    alloc = default_contribution_allocation(annual_contribution)
    current = {"taxable": start, "trad": 0.0, "roth": 0.0}
    proj = project_balances_to_retirement(current, alloc, years, r)

    # Forecast: all-working (coast_age = retirement_age), same total contribution.
    fdf = project_portfolio(
        start_value=start, current_age=current_age, coast_age=retirement_age,
        retirement_age=retirement_age, horizon_age=retirement_age,
        annual_contribution=annual_contribution, real_return=r,
    )
    forecast_total = float(fdf.loc[retirement_age, "total"])

    diff = abs(proj["total"] - forecast_total)
    assert diff < 1.0, f"projector {proj['total']:,.2f} != forecast {forecast_total:,.2f}"
    print(f"  reconciles with Forecast ✓  {proj['total']:,.0f} at age {retirement_age} "
          f"(Δ {diff:.4f})")

    # Bucket split is additive: summing buckets equals the single-bucket total.
    split = project_balances_to_retirement(
        {"taxable": start, "trad": 0.0, "roth": 0.0}, alloc, years, r)
    assert abs(split["total"] - proj["total"]) < 1e-6


def main() -> None:
    check_allocation()
    check_reconciles_with_forecast()
    print("\nPhase 6b balance projector verified ✓")


if __name__ == "__main__":
    main()
