"""Verify the unified Coast scenario builder (core/services/scenario.build_scenario).

Pure check — no Firestore. Gates Phase 1 of .planning/SPEC-scenario-planning.md:
the consolidated `build_scenario` must reproduce the numbers the legacy split
pages produced, so collapsing Forecast + Retirement onto one screen moves no
dashboard value.

Reconciliation, for one sample config:
  1. Backward goal == `retirement.default_nest_egg_goal` (legacy Retirement default).
  2. Accumulation aggregate (projected at coast / retirement) == Forecast's
     `project_portfolio` totals at those ages.
  3. CoastFIRE target + required monthly == Forecast's `forecast_summary`
     (computed off project_portfolio + the goal).

Run from the repo root:
    python -m scripts.verify_scenario
"""

from core.services.forecast import forecast_summary, project_portfolio
from core.domain.drawdown import default_nest_egg_goal
from core.services.scenario import build_scenario

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


def check_accumulation_reconciles_with_forecast() -> None:
    s = _scenario()
    pp = project_portfolio(
        start_value=s.start_value, current_age=s.current_age, coast_age=s.coast_age,
        retirement_age=s.retirement_age, horizon_age=s.retirement_age,
        annual_contribution=s.annual_contribution, real_return=s.real_return,
    )
    at_coast = float(pp.loc[s.coast_age, "total"])
    at_ret = float(pp.loc[s.retirement_age, "total"])
    assert abs(s.projected_at_coast - at_coast) < 1.0, (s.projected_at_coast, at_coast)
    assert abs(s.projected_at_retirement - at_ret) < 1.0, (s.projected_at_retirement, at_ret)
    print(f"  accumulation reconciles with Forecast ✓  "
          f"coast {s.projected_at_coast:,.0f} / ret {s.projected_at_retirement:,.0f}")


def check_bans_match_forecast_summary() -> None:
    s = _scenario()
    pp = project_portfolio(
        start_value=s.start_value, current_age=s.current_age, coast_age=s.coast_age,
        retirement_age=s.retirement_age, horizon_age=s.retirement_age,
        annual_contribution=s.annual_contribution, real_return=s.real_return,
    )
    summary = forecast_summary(
        pp, s.nest_egg_goal, s.real_return, s.start_value,
        s.current_age, s.coast_age, s.retirement_age,
    )
    assert abs(s.coast_target - summary["coast_target"]) < TOL, s.coast_target
    assert abs(s.required_monthly - summary["required_monthly"]) < TOL, s.required_monthly
    assert abs(s.projected_at_coast - summary["projected_at_coast"]) < 1.0
    print(f"  BANs match forecast_summary ✓  target {s.coast_target:,.0f} · "
          f"required ${s.required_monthly:,.0f}/mo · on_track={s.on_track}")


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
    check_accumulation_reconciles_with_forecast()
    check_bans_match_forecast_summary()
    check_live_override_moves_numbers()
    print("\nUnified scenario builder verified ✓")


if __name__ == "__main__":
    main()
