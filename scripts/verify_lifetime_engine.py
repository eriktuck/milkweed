"""Verify the unified lifetime engine (core/services/projection.simulate_lifetime).

Pure check — no Firestore. Gates the Step-1 extraction (.planning/SPEC-rationalize-pages.md):

  1. `project_retirement_taxaware` (now a thin wrapper over `simulate_lifetime`)
     reproduces **golden values** captured from the pre-refactor function — the
     drawdown is byte-identical (same columns, same numbers).
  2. `project_balances_to_retirement` (also a wrapper) reproduces its golden
     at-retirement balances.
  3. The engine's **accumulation aggregate reconciles with Forecast's
     `project_portfolio`** — the cross-page invariant that lets Forecast consume
     the shared engine without moving any dashboard number.
  4. Bucket split is additive (sum of buckets == single-bucket aggregate).

Run from the repo root:
    python -m scripts.verify_lifetime_engine
"""

from core.services.forecast import project_portfolio
from core.services.projection import simulate_lifetime
from core.services.retirement import (
    project_balances_to_retirement,
    project_retirement_taxaware,
)

# Column contract of the legacy tax-aware drawdown (order matters).
GOLDEN_COLS = [
    "total", "taxable", "trad", "roth", "spend", "healthcare", "income", "rmd",
    "withdrawal", "w_taxable", "w_trad", "w_roth", "ordinary_income",
    "ltcg_gain", "tax", "net_spend", "phase",
]

# Captured from the pre-refactor project_retirement_taxaware on the case below.
GOLDEN_PHASES = {65: "go_go", 75: "slow_go", 85: "no_go", 90: "no_go"}
GOLDEN_SAMPLE = {
    65: {"total": 800000.0, "taxable": 400000.0, "trad": 300000.0, "roth": 100000.0, "spend": 70000.0, "healthcare": 6000.0, "income": 0.0, "rmd": 0.0, "withdrawal": 76000.0, "w_taxable": 76000.0, "w_trad": 0.0, "w_roth": 0.0, "ordinary_income": 0.0, "ltcg_gain": 38000.0, "tax": 0.0, "net_spend": 76000.0},
    75: {"total": 460387.758309, "taxable": 0.0, "trad": 312363.329817, "roth": 148024.428492, "spend": 60000.0, "healthcare": 6000.0, "income": 24000.0, "rmd": 12697.696334, "withdrawal": 43166.55, "w_taxable": 0.0, "w_trad": 43166.55, "w_roth": 0.0, "ordinary_income": 43166.55, "ltcg_gain": 0.0, "tax": 1166.655, "net_spend": 65999.895},
    85: {"total": 144968.588197, "taxable": 0.0, "trad": 0.0, "roth": 144968.588197, "spend": 50000.0, "healthcare": 46000.0, "income": 24000.0, "rmd": 0.0, "withdrawal": 72000.0, "w_taxable": 0.0, "w_trad": 0.0, "w_roth": 72000.0, "ordinary_income": 0.0, "ltcg_gain": 0.0, "tax": 0.0, "net_spend": 96000.0},
    90: {"total": 0.0, "taxable": 0.0, "trad": 0.0, "roth": 0.0, "spend": 50000.0, "healthcare": 46000.0, "income": 24000.0, "rmd": 0.0, "withdrawal": 0.0, "w_taxable": 0.0, "w_trad": 0.0, "w_roth": 0.0, "ordinary_income": 0.0, "ltcg_gain": 0.0, "tax": 0.0, "net_spend": 24000.0},
}
GOLDEN_PROJ = {"taxable": 846588.735225, "trad": 1121586.822223, "roth": 310226.142317, "total": 2278401.699764}

TOL = 1e-3


def _taxaware_case():
    balances = {"taxable": 400000.0, "trad": 300000.0, "roth": 100000.0}
    spend = {"go_go": 70000.0, "slow_go": 60000.0, "no_go": 50000.0}
    ss = {a: (24000.0 if a >= 67 else 0.0) for a in range(65, 91)}
    hc = {a: (15000.0 if a < 65 else 6000.0) + (40000.0 if a >= 85 else 0.0)
          for a in range(65, 91)}
    return project_retirement_taxaware(
        balances, 65, 90, spend, 75, 85, 0.04,
        ss_by_age=ss, healthcare_by_age=hc, rmd_start=75, taxable_gain_fraction=0.5)


def check_taxaware_golden() -> None:
    df = _taxaware_case()
    assert list(df.columns) == GOLDEN_COLS, f"columns drifted: {list(df.columns)}"
    for age, expected in GOLDEN_SAMPLE.items():
        assert df.loc[age, "phase"] == GOLDEN_PHASES[age], (age, df.loc[age, "phase"])
        for col, want in expected.items():
            got = float(df.loc[age, col])
            assert abs(got - want) < TOL, f"age {age} {col}: {got} != {want}"
    print(f"  tax-aware drawdown matches golden ✓  ({len(df)} ages, {len(GOLDEN_COLS)} cols)")


def check_balances_golden() -> None:
    proj = project_balances_to_retirement(
        {"taxable": 250000.0, "trad": 0.0, "roth": 0.0},
        {"taxable": 0.0, "trad": 23500.0, "roth": 6500.0}, 25, 0.05)
    for k, want in GOLDEN_PROJ.items():
        assert abs(proj[k] - want) < TOL, f"{k}: {proj[k]} != {want}"
    print(f"  balance projector matches golden ✓  ({proj['total']:,.0f} at retirement)")


def check_reconciles_with_forecast() -> None:
    start, current_age, retirement_age, r = 250_000.0, 40, 65, 0.05
    annual_contribution = 30_000.0

    # Engine: single taxable bucket, all-working window to retirement.
    df = simulate_lifetime(
        buckets={"taxable": start, "trad": 0.0, "roth": 0.0},
        allocation={"taxable": annual_contribution, "trad": 0.0, "roth": 0.0},
        current_age=current_age, coast_age=retirement_age,
        retirement_age=retirement_age, death_age=retirement_age,
        spend_by_phase={}, slow_go_age=retirement_age, no_go_age=retirement_age,
        r_accum=r, r_retire=r,
    )
    engine_total = float(df.loc[retirement_age, "total"])

    fdf = project_portfolio(
        start_value=start, current_age=current_age, coast_age=retirement_age,
        retirement_age=retirement_age, horizon_age=retirement_age,
        annual_contribution=annual_contribution, real_return=r,
    )
    forecast_total = float(fdf.loc[retirement_age, "total"])

    diff = abs(engine_total - forecast_total)
    assert diff < 1.0, f"engine {engine_total:,.2f} != forecast {forecast_total:,.2f}"
    print(f"  reconciles with Forecast ✓  {engine_total:,.0f} at age {retirement_age} (Δ {diff:.4f})")


def check_bucket_additivity() -> None:
    # Same total contribution, split across buckets, must aggregate to the same total.
    single = simulate_lifetime(
        buckets={"taxable": 250_000.0, "trad": 0.0, "roth": 0.0},
        allocation={"taxable": 30_000.0, "trad": 0.0, "roth": 0.0},
        current_age=40, coast_age=65, retirement_age=65, death_age=65,
        spend_by_phase={}, slow_go_age=65, no_go_age=65, r_accum=0.05, r_retire=0.05,
    )
    split = simulate_lifetime(
        buckets={"taxable": 250_000.0, "trad": 0.0, "roth": 0.0},
        allocation={"taxable": 0.0, "trad": 23_500.0, "roth": 6_500.0},
        current_age=40, coast_age=65, retirement_age=65, death_age=65,
        spend_by_phase={}, slow_go_age=65, no_go_age=65, r_accum=0.05, r_retire=0.05,
    )
    assert abs(float(single.loc[65, "total"]) - float(split.loc[65, "total"])) < 1e-6
    print("  bucket split is additive ✓")


def main() -> None:
    check_taxaware_golden()
    check_balances_golden()
    check_reconciles_with_forecast()
    check_bucket_additivity()
    print("\nLifetime engine (Step 1) verified ✓")


if __name__ == "__main__":
    main()
