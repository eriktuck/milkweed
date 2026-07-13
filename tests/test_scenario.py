"""Characterization tests for core.services.scenario.build_scenario.

Pins the CURRENT behavior of the Coast page's façade (and its `_resolve_pia`
helper) on current main, ahead of the core/domain/ restructuring. All expected
literals were observed by running the code as-is; nothing here asserts intended
behavior. Fixtures are lifted from scripts/verify_scenario.py (whose CASES are
ported into pytest below). Pure — no Firestore, no network, deterministic via
an explicit `as_of_year`.

Goldens (tests/goldens/scenario_*.csv) were generated ONCE from current main by
`_generate_goldens()` below; tests only read them.
"""

from pathlib import Path

import pandas as pd
import pytest

from core.domain.glide import STRATEGY_TERMINAL_EQUITY
from core.services.scenario import DEFAULT_COAST_AGE, _resolve_pia, build_scenario

GOLDENS = Path(__file__).parent / "goldens"

# ── base fixture (lifted from scripts/verify_scenario.py) ─────────────────────

BASE_CFG = {
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
AS_OF_YEAR = 2026  # deterministic current_age (= 42 for a 1984 birth)

# Two-leg glide params, the shape build_returns_payload produces.
GLIDE_PARAMS = {"w_equity0": 0.9, "r_equity": 0.065, "r_fixed": 0.02}

INCOME_SEGMENTS = [
    {"date": "2010-01-01", "amount": 60_000.0},
    {"date": "2018-01-01", "amount": 90_000.0},
]

approx = pytest.approx


def build(cfg=BASE_CFG, plan=PLAN, labels=LABELS, buckets=BUCKETS, **overrides):
    """build_scenario with the base fixture; as_of_year always pinned."""
    overrides.setdefault("as_of_year", AS_OF_YEAR)
    return build_scenario(cfg, plan, labels, buckets, **overrides)


def _assert_frame_matches_golden(actual: pd.DataFrame, path: Path) -> None:
    # keep_default_na=False: the phase columns contain empty strings ("" during
    # accumulation for spend_phase), which must not round-trip as NaN.
    golden = pd.read_csv(path, index_col="age", keep_default_na=False)
    golden.index = golden.index.astype(actual.index.dtype)
    for col in golden.columns:  # all-integer float columns round-trip as int64
        if actual[col].dtype.kind == "f":
            golden[col] = golden[col].astype("float64")
    pd.testing.assert_frame_equal(actual, golden, rtol=1e-9)


def _assert_series_matches_golden(actual: pd.Series, path: Path) -> None:
    golden = pd.read_csv(path, index_col="age")["value"].astype("float64")
    golden.index = golden.index.astype(actual.index.dtype)
    pd.testing.assert_series_equal(actual, golden, rtol=1e-9, check_names=False)


# ── base case: every scalar field of the returned Scenario ────────────────────

def test_base_resolved_input_scalars():
    """Base fixture resolves ages/return/contribution straight from config."""
    s = build()
    assert s.current_age == 42
    assert s.coast_age == 50
    assert s.retirement_age == 65
    assert s.claim_age == 67
    assert s.slow_go_age == 75
    assert s.no_go_age == 85
    assert s.death_age == 90
    assert s.real_return == approx(0.05, rel=1e-9)
    assert s.monthly_contribution == approx(1500.0, rel=1e-9)  # CSP `savings` line
    assert s.annual_contribution == approx(18000.0, rel=1e-9)
    assert s.start_value == approx(500000.0, rel=1e-9)
    assert s.return_strategy is None
    assert s.equity_glide is None


def test_base_headline_scalars():
    """Base fixture BANs (the numbers verify_scenario reconciles with legacy)."""
    s = build()
    assert s.pia_annual == approx(38010.6, rel=1e-9)
    assert s.nest_egg_goal == approx(636030.4303070052, rel=1e-9)
    assert s.coast_target == approx(305941.5118838266, rel=1e-9)
    assert s.projected_at_coast == approx(910611.6816585938, rel=1e-9)
    assert s.coast_surplus == approx(604670.1697747672, rel=1e-9)
    assert s.required_monthly == approx(0.0, abs=0.0)  # already coasting past target
    assert s.projected_at_retirement == approx(1893096.2855012254, rel=1e-9)
    assert s.on_track is True
    assert s.coast_surplus == approx(s.projected_at_coast - s.coast_target, rel=1e-9)


def test_base_dict_fields():
    """Allocation (tax-advantaged-first), spend-by-phase, and buckets echo."""
    s = build()
    assert s.allocation == {"taxable": 0.0, "trad": 18000.0, "roth": 0.0}
    assert s.spend_by_phase == approx(
        {"go_go": 56400.0, "slow_go": 48900.0, "no_go": 41760.0}, rel=1e-9)
    assert s.buckets == BUCKETS


def test_base_series_shapes_and_spot_values():
    """SS starts at the claim age; healthcare steps up at the LTC start age."""
    s = build()
    assert list(s.ss_by_age.index) == list(range(65, 91))
    assert float(s.ss_by_age.loc[66]) == 0.0
    assert float(s.ss_by_age.loc[67]) == approx(38010.6, rel=1e-9)
    assert float(s.healthcare_by_age.loc[65]) == approx(6220.0, rel=1e-9)
    assert float(s.healthcare_by_age.loc[84]) == approx(6220.0, rel=1e-9)
    assert float(s.healthcare_by_age.loc[85]) == approx(134220.0, rel=1e-9)  # + LTC


def test_base_lifetime_golden():
    """Full base lifetime walk (age 42→90, all 19 columns) vs golden."""
    s = build()
    assert s.lifetime.shape == (49, 19)
    _assert_frame_matches_golden(s.lifetime, GOLDENS / "scenario_base_lifetime.csv")


def test_base_ss_by_age_golden():
    s = build()
    _assert_series_matches_golden(s.ss_by_age, GOLDENS / "scenario_base_ss_by_age.csv")


def test_base_healthcare_by_age_golden():
    s = build()
    _assert_series_matches_golden(
        s.healthcare_by_age, GOLDENS / "scenario_base_healthcare_by_age.csv")


# ── glide strategies (every STRATEGY_TERMINAL_EQUITY key) ─────────────────────

GLIDE_EXPECTED = {
    # strategy: (real_return/effective CAGR, nest_egg_goal, coast_target,
    #            projected_at_coast, projected_at_retirement, equity@64)
    "aggressive": (0.05619246277125978, 624415.355685144, 281175.4784216679,
                   968953.0301079716, 2151784.9078920335, 0.708695652173913),
    "moderate": (0.051878494380984685, 699558.6591345704, 342539.6655715094,
                 959572.9581073495, 1959707.559109627, 0.5173913043478261),
    "conservative": (0.04755801509092494, 789535.5822010076, 420593.9800416259,
                     950270.012489871, 1783839.1017510404, 0.326086956521739),
}


def test_glide_expected_covers_all_strategies():
    """The parametrization below exercises every available glide strategy."""
    assert set(GLIDE_EXPECTED) == set(STRATEGY_TERMINAL_EQUITY)


@pytest.mark.parametrize("strategy", sorted(STRATEGY_TERMINAL_EQUITY))
def test_glide_strategy(strategy):
    """Each glide preset: effective CAGR, BANs, and the equity path endpoints."""
    eff, goal, target, at_coast, at_ret, w64 = GLIDE_EXPECTED[strategy]
    s = build(return_strategy=strategy, glide_params=GLIDE_PARAMS)
    assert s.return_strategy == strategy
    assert s.real_return == approx(eff, rel=1e-9)
    assert s.nest_egg_goal == approx(goal, rel=1e-9)
    assert s.coast_target == approx(target, rel=1e-9)
    assert s.projected_at_coast == approx(at_coast, rel=1e-9)
    assert s.projected_at_retirement == approx(at_ret, rel=1e-9)
    assert len(s.equity_glide) == 23  # ages 42..64 (accumulation years)
    assert s.equity_glide[42] == approx(0.9, rel=1e-9)
    assert s.equity_glide[64] == approx(w64, rel=1e-9)


def test_glide_moderate_lifetime_golden():
    """Non-default variant golden: moderate glide lifetime walk."""
    s = build(return_strategy="moderate", glide_params=GLIDE_PARAMS)
    assert s.equity_glide[50] == approx(0.7608695652173914, rel=1e-9)
    _assert_frame_matches_golden(
        s.lifetime, GOLDENS / "scenario_glide_moderate_lifetime.csv")


def test_glide_falls_back_to_flat_without_full_params():
    """Missing two-leg params → flat-rate path, strategy nulled out."""
    s = build(return_strategy="aggressive", glide_params={"w_equity0": 0.9})
    assert s.return_strategy is None
    assert s.equity_glide is None
    assert s.real_return == approx(0.05, rel=1e-9)


def test_glide_downward_only_clamp():
    """w0 at/below the terminal weight holds flat (never buys equity)."""
    s = build(return_strategy="aggressive",
              glide_params={"w_equity0": 0.5, "r_equity": 0.065, "r_fixed": 0.02})
    assert s.equity_glide[42] == approx(0.5, rel=1e-9)
    assert s.equity_glide[64] == approx(0.5, rel=1e-9)
    assert s.real_return == approx(0.04249999999999998, rel=1e-9)


# ── live-input / assumption overrides (one per parameter) ─────────────────────

def test_override_monthly_contribution():
    s = build(monthly_contribution=2_500.0)
    assert s.monthly_contribution == approx(2500.0, rel=1e-9)
    assert s.annual_contribution == approx(30000.0, rel=1e-9)
    assert s.projected_at_coast == approx(1025200.9881679688, rel=1e-9)
    assert s.nest_egg_goal == approx(636030.4303070052, rel=1e-9)  # goal unmoved


def test_override_coast_age():
    s = build(coast_age=55)
    assert s.coast_age == 55
    assert s.coast_target == approx(390467.5108207026, rel=1e-9)
    assert s.projected_at_coast == approx(1261658.2623979829, rel=1e-9)


def test_override_retirement_age():
    """Retiring earlier raises the goal (the verify_scenario sanity case)."""
    s = build(retirement_age=60)
    assert s.retirement_age == 60
    assert s.nest_egg_goal == approx(822927.3508129242, rel=1e-9)
    assert s.coast_target == approx(505206.00736524, rel=1e-9)
    base = build()
    early = build(retirement_age=58)
    assert early.nest_egg_goal > base.nest_egg_goal
    assert early.coast_target > base.coast_target


def test_override_claim_age():
    s = build(claim_age=70)
    assert s.claim_age == 70
    assert s.nest_egg_goal == approx(638389.0251706865, rel=1e-9)
    assert float(s.ss_by_age.loc[69]) == 0.0
    assert float(s.ss_by_age.loc[70]) == approx(47133.144, rel=1e-9)  # 1.24 × PIA


def test_override_slow_go_age():
    s = build(slow_go_age=78)
    assert s.slow_go_age == 78
    assert s.nest_egg_goal == approx(649196.1550130408, rel=1e-9)


def test_override_no_go_age():
    s = build(no_go_age=88)
    assert s.no_go_age == 88
    assert s.nest_egg_goal == approx(643725.0777778132, rel=1e-9)


def test_override_death_age():
    s = build(death_age=95)
    assert s.death_age == 95
    assert int(s.lifetime.index.max()) == 95
    assert list(s.ss_by_age.index) == list(range(65, 96))
    assert s.nest_egg_goal == approx(812425.1996049583, rel=1e-9)


def test_override_real_return():
    s = build(real_return=0.03)
    assert s.real_return == approx(0.03, rel=1e-9)
    assert s.nest_egg_goal == approx(829226.4911530453, rel=1e-9)
    assert s.coast_target == approx(532248.9304444406, rel=1e-9)
    assert s.projected_at_coast == approx(793447.0895263777, rel=1e-9)


def test_override_allocation():
    """Explicit allocation replaces the tax-advantaged-first default."""
    alloc = {"taxable": 6_000.0, "trad": 12_000.0, "roth": 0.0}
    s = build(allocation=alloc)
    assert s.allocation == alloc
    assert float(s.lifetime.loc[65, "taxable"]) == approx(1040568.5959561169, rel=1e-9)
    assert float(s.lifetime.loc[65, "trad"]) == approx(698951.5017492183, rel=1e-9)


def test_override_pia_annual():
    s = build(pia_annual=30_000.0)
    assert s.pia_annual == approx(30000.0, rel=1e-9)
    assert s.nest_egg_goal == approx(741302.2397906204, rel=1e-9)


def test_override_gain_fraction():
    """Gain fraction scales the realized-LTCG share of taxable withdrawals."""
    base = build()
    s = build(gain_fraction=0.8)
    assert float(base.lifetime.loc[65, "ltcg_gain"]) == approx(31310.0, rel=1e-9)
    assert float(s.lifetime.loc[65, "ltcg_gain"]) == approx(50096.0, rel=1e-9)
    # Headline BANs are accumulation/goal-side and do not move.
    assert s.nest_egg_goal == approx(base.nest_egg_goal, rel=1e-9)


def test_override_healthcare():
    s = build(healthcare_overrides={"aca_annual": 20_000.0, "ltc_annual": 0.0})
    # retirement_age (65) == medicare_age, so every year is Medicare + OOP; the
    # ACA figure never applies and killing LTC flattens the whole series.
    assert set(s.healthcare_by_age.round(9)) == {6220.0}
    assert s.nest_egg_goal == approx(378926.59589065064, rel=1e-9)


def test_override_spend_by_phase():
    spend = {"go_go": 60_000.0, "slow_go": 50_000.0, "no_go": 40_000.0}
    s = build(spend_by_phase=spend)
    assert s.spend_by_phase == approx(spend, rel=1e-9)
    assert s.nest_egg_goal == approx(667158.659718524, rel=1e-9)


def test_override_as_of_year():
    """as_of_year drives current_age (and thus the whole forward walk)."""
    s = build(as_of_year=2030)
    assert s.current_age == 46
    assert int(s.lifetime.index.min()) == 46
    assert s.projected_at_coast == approx(685335.375, rel=1e-9)


# ── fallback paths ─────────────────────────────────────────────────────────────

def test_no_birth_date_fallback():
    """No birth_date → current_age = max(retirement_age − 25, 18) = 40."""
    cfg = {k: v for k, v in BASE_CFG.items() if k != "birth_date"}
    s = build(cfg=cfg)
    assert s.current_age == 40
    assert int(s.lifetime.index.min()) == 40
    assert s.projected_at_coast == approx(1040849.3790286, rel=1e-9)
    assert s.nest_egg_goal == approx(636030.4303070052, rel=1e-9)  # goal age-blind


def test_none_config_uses_all_defaults():
    """cfg=None → DEFAULTS + DEFAULT_COAST_AGE + the no-birth-date age fallback."""
    s = build(cfg=None)
    assert DEFAULT_COAST_AGE == 50
    assert (s.current_age, s.coast_age, s.retirement_age, s.claim_age,
            s.slow_go_age, s.no_go_age, s.death_age) == (40, 50, 65, 67, 75, 85, 90)
    assert s.real_return == approx(0.05, rel=1e-9)
    assert s.nest_egg_goal == approx(636030.4303070052, rel=1e-9)


def test_required_monthly_nonzero_when_behind():
    """A small portfolio flips on_track and produces a real required monthly."""
    low = {"taxable": 10_000.0, "trad": 5_000.0, "roth": 0.0, "total": 15_000.0}
    s = build(buckets=low)
    assert s.start_value == approx(15000.0, rel=1e-9)
    assert s.on_track is False
    assert s.required_monthly == approx(2476.4935653378197, rel=1e-9)
    assert s.coast_surplus == approx(-111895.72046292815, rel=1e-9)


def test_start_value_summed_when_no_total_key():
    """Buckets without `total` → start_value is the three-bucket sum."""
    s = build(buckets={"taxable": 100_000.0, "trad": 50_000.0, "roth": 25_000.0})
    assert s.start_value == approx(175000.0, rel=1e-9)


# ── PIA resolution: both build_scenario branches + _resolve_pia directly ──────

def test_pia_from_retirement_config():
    """Persisted RetirementConfig.pia_annual wins over any estimate."""
    cfg = dict(BASE_CFG, retirement={"real_return": 0.05, "pia_annual": 24_000.0})
    s = build(cfg=cfg)
    assert s.pia_annual == approx(24000.0, rel=1e-9)
    assert s.nest_egg_goal == approx(820151.6214726031, rel=1e-9)


def test_pia_estimated_from_csp_income():
    """No config PIA, no segments → PIA estimated from the CSP `income` line."""
    s = build()
    # $8k/mo → $96k/yr → AIME $8k/mo → bend-point PIA × 12 = $38,010.60
    assert s.pia_annual == approx(38010.6, rel=1e-9)


def test_pia_from_income_segments():
    """Profile income segments → earnings-history PIA (the accurate branch)."""
    cfg = dict(BASE_CFG, income_segments=INCOME_SEGMENTS)
    s = build(cfg=cfg)
    assert s.pia_annual == approx(13744.388571428572, rel=1e-9)
    assert s.nest_egg_goal == approx(954926.3914581919, rel=1e-9)


def test_resolve_pia_direct():
    """_resolve_pia: segments branch, CSP-income branch, and the zero fallbacks."""
    seg_cfg = dict(BASE_CFG, income_segments=INCOME_SEGMENTS)
    assert _resolve_pia(BASE_CFG, PLAN, LABELS) == approx(38010.6, rel=1e-9)
    assert _resolve_pia(seg_cfg, PLAN, LABELS) == approx(13744.388571428572, rel=1e-9)
    assert _resolve_pia(seg_cfg, None, None) == approx(13744.388571428572, rel=1e-9)
    assert _resolve_pia(None, None, None) == 0.0
    assert _resolve_pia(BASE_CFG, PLAN, {}) == 0.0  # no key labeled `income`


# ── golden generation (run ONCE per characterization epoch) ───────────────────

def _generate_goldens() -> None:
    """Write the scenario_* goldens from current behavior.

    WARNING: run once per characterization epoch, on known-good main, then
    commit. A refactor may NEVER regenerate these — if a refactor changes the
    numbers, the refactor is wrong.
    """
    GOLDENS.mkdir(exist_ok=True)
    base = build()
    base.lifetime.to_csv(GOLDENS / "scenario_base_lifetime.csv", float_format="%.17g")
    base.ss_by_age.rename("value").to_csv(
        GOLDENS / "scenario_base_ss_by_age.csv", float_format="%.17g",
        index_label="age")
    base.healthcare_by_age.rename("value").to_csv(
        GOLDENS / "scenario_base_healthcare_by_age.csv", float_format="%.17g",
        index_label="age")
    glide = build(return_strategy="moderate", glide_params=GLIDE_PARAMS)
    glide.lifetime.to_csv(
        GOLDENS / "scenario_glide_moderate_lifetime.csv", float_format="%.17g")
    print(f"goldens written to {GOLDENS}")


if __name__ == "__main__":
    _generate_goldens()
