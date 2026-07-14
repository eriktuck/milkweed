"""Verify the unified Coast scenario builder (core/domain/scenario.build_scenario).

Pure check — no Firestore. Gates Phase 1 of .planning/SPEC-scenario-planning.md:
the consolidated `build_scenario` must reproduce the numbers the legacy split
pages produced, so collapsing Forecast + Retirement onto one screen moves no
dashboard value.

Reconciliation, for one sample config:
  1. Backward goal == `drawdown.default_nest_egg_goal` (legacy Retirement default).
  2. Live overrides move the numbers in the expected direction.

(The former project_portfolio / forecast_summary reconciliation checks were
retired with those functions in Phase 5 — the equivalence is pinned by
tests/test_lifetime.py against the frozen forecast_project_portfolio.csv
golden and its constants.)

Run from the repo root:
    python -m scripts.verify_scenario
"""

from core.domain.drawdown import default_nest_egg_goal
from core.domain.scenario import build_scenario

TOL = 1e-3

# A minimal but complete config + latest CSP plan (monthly $ by csp key).
CFG = {
    "birth_date": "1984-01-01",
    "coast_age": 50,
    "retirement_age": 65,
    "claim_age": 67,
    "death_age": 90,
    "retirement": {"real_return": 0.05},
}
PLAN = {
    "mortgage": 2_000.0, "groceries": 700.0, "bills_utilities": 400.0,
    "auto": 300.0, "guilt_free": 900.0, "vacations": 400.0,
    "income": 8_000.0, "savings": 1_500.0,
}
LABELS = {
    "mortgage": "fixed", "groceries": "fixed", "bills_utilities": "fixed",
    "auto": "fixed", "guilt_free": "guilt-free", "vacations": "sinking",
    "income": "income", "savings": "investments",
}
BUCKETS = {"taxable": 300_000.0, "trad": 150_000.0, "roth": 50_000.0, "total": 500_000.0}
AS_OF_YEAR = 2026   # deterministic current_age (= 42 for a 1984 birth)


def _scenario():
    # No live overrides → build_scenario resolves everything from CFG, matching
    # how the legacy default pipeline resolved the same config.
    return build_scenario(CFG, PLAN, LABELS, BUCKETS, as_of_year=AS_OF_YEAR)


def check_goal_matches_legacy_default() -> None:
    s = _scenario()
    legacy = default_nest_egg_goal(CFG, PLAN, LABELS)
    assert abs(s.nest_egg_goal - legacy) < TOL, f"{s.nest_egg_goal} != {legacy}"
    assert s.nest_egg_goal > 100_000.0, f"goal implausibly small: {s.nest_egg_goal}"
    print(f"  goal matches legacy default ✓  {s.nest_egg_goal:,.0f}")


def check_live_override_moves_numbers() -> None:
    # Sanity: retiring earlier shortens the drawdown horizon's discounting and
    # raises the goal, lifting the CoastFIRE target the user must hit.
    base = build_scenario(CFG, PLAN, LABELS, BUCKETS, as_of_year=AS_OF_YEAR,
                          retirement_age=65, coast_age=50)
    early = build_scenario(CFG, PLAN, LABELS, BUCKETS, as_of_year=AS_OF_YEAR,
                           retirement_age=58, coast_age=50)
    assert early.nest_egg_goal > base.nest_egg_goal, (early.nest_egg_goal, base.nest_egg_goal)
    assert early.coast_target > base.coast_target, (early.coast_target, base.coast_target)
    print(f"  live overrides move the numbers ✓  "
          f"goal {base.nest_egg_goal:,.0f} → {early.nest_egg_goal:,.0f}, "
          f"target {base.coast_target:,.0f} → {early.coast_target:,.0f} (retire 65→58)")


def main() -> None:
    check_goal_matches_legacy_default()
    check_live_override_moves_numbers()
    print("\nUnified scenario builder verified ✓")


if __name__ == "__main__":
    main()
