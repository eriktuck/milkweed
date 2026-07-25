"""Characterization tests for core.domain.scenario.build_scenario.

Pins the CURRENT behavior of the Coast page's façade (and the shared
`social_security.resolve_pia` helper) on current main. All expected
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
from core.domain.scenario import DEFAULT_COAST_AGE, build_scenario
from core.domain.social_security import resolve_pia

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


# ── PIA resolution: both build_scenario branches + resolve_pia directly ───────

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
    """resolve_pia: segments branch, CSP-income branch, and the zero fallbacks."""
    seg_cfg = dict(BASE_CFG, income_segments=INCOME_SEGMENTS)
    assert resolve_pia(BASE_CFG, PLAN, LABELS) == approx(38010.6, rel=1e-9)
    assert resolve_pia(seg_cfg, PLAN, LABELS) == approx(13744.388571428572, rel=1e-9)
    assert resolve_pia(seg_cfg, None, None) == approx(13744.388571428572, rel=1e-9)
    assert resolve_pia(None, None, None) == 0.0
    assert resolve_pia(BASE_CFG, PLAN, {}) == 0.0  # no key labeled `income`


# ═════════════════════════════════════════════════════════════════════════════
# Housing wiring (.planning/SPEC-housing.md, Phase 3)
# ═════════════════════════════════════════════════════════════════════════════
#
# The base fixture's household, given a house: bought 2012, $2,400/mo PITI of
# which $700 is escrow, 6.25% note. Birth year 1984, so a payoff date of
# 2050-01-01 is exactly age 66 — inside the retirement window, where the cliff
# can move the goal.

HOUSE = {
    "occupancy": "primary", "label": "Elm St",
    "purchase_price": 310_000.0, "purchase_date": "2012-04-01",
    "current_value": 650_000.0, "real_appreciation": 0.01,
    "monthly_payment": 2_400.0, "escrow_monthly": 700.0,
    "note_rate": 0.0625, "payoff_date": "2050-01-01", "maintenance_pct": 0.01,
}


def housing_cfg(*, properties=None, **house_over) -> dict:
    """A configured housing block: own the house, no rental, no sale by default."""
    return {
        "residence": {"mode": "own"},
        "properties": (properties if properties is not None
                       else [{**HOUSE, **house_over}]),
        "inflation": 0.025,
    }


def with_housing(block) -> dict:
    return {**BASE_CFG, "retirement": {**BASE_CFG["retirement"], "housing": block}}


# ── the gate: absent config must change absolutely nothing ────────────────────

def test_absent_housing_config_leaves_every_new_field_none():
    """The hard constraint. `None` on all four fields IS the no-housing signal."""
    s = build()
    assert s.housing_by_age is None
    assert s.rental_net_income_by_age is None
    assert s.home_equity is None
    assert s.sale_proceeds is None


@pytest.mark.parametrize("block", [
    None,
    {},
    {"residence": None, "properties": None, "inflation": None},  # HousingConfig()
    {"properties": []},
])
def test_an_inactive_housing_block_is_still_no_housing(block):
    """`HousingConfig()` serialises to a truthy dict of Nones. Gating on
    truthiness would drop `mortgage` from the spend pool and supply a zero cost
    stream — quietly cutting the goal for a user who configured nothing."""
    s = build(cfg=with_housing(block))
    assert s.housing_by_age is None
    assert s.spend_by_phase["go_go"] == approx(56_400.0, rel=1e-9)   # mortgage still in
    assert s.nest_egg_goal == approx(636030.4303070052, rel=1e-9)


# ── the csp exclusion ─────────────────────────────────────────────────────────

def test_a_configured_block_takes_the_housing_keys_out_of_the_spend_pool():
    """`mortgage` leaves the generic pool at go-go value; the module supplies the
    cost instead. (`home_improvements` is not in this fixture's plan.)"""
    s = build(cfg=with_housing(housing_cfg()))
    assert s.spend_by_phase["go_go"] == approx(56_400.0 - 12 * 2_000.0, rel=1e-9)
    assert s.housing_by_age is not None
    assert list(s.housing_by_age.index) == list(range(65, 91))


def test_the_exclusion_does_not_touch_an_explicit_spend_override():
    """When the page's expenses table drives the spend, it has already applied
    the same exclusion (pages/coast.py::_seed_expense_rows) — build_scenario must
    not subtract it twice."""
    override = {"go_go": 40_000.0, "slow_go": 35_000.0, "no_go": 30_000.0}
    s = build(cfg=with_housing(housing_cfg()), spend_by_phase=override)
    assert s.spend_by_phase == approx(override, rel=1e-9)


# ── Gate (a): the goal FALLS when the mortgage ends before death ──────────────

def test_the_nest_egg_goal_falls_when_the_payoff_precedes_death():
    """The spec's headline. Both scenarios are housing-configured, so the csp
    exclusion is identical and the terminating P&I is the only variable."""
    payoff = build(cfg=with_housing(housing_cfg(payoff_date="2050-01-01")))  # age 66
    never = build(cfg=with_housing(housing_cfg(payoff_date=None)))

    assert payoff.nest_egg_goal < never.nest_egg_goal
    # The cliff itself: P&I is gone the year after payoff, onto the escrow +
    # maintenance floor that outlives the loan.
    assert float(payoff.housing_by_age.loc[66]) < float(payoff.housing_by_age.loc[65])
    assert float(payoff.housing_by_age.loc[90]) > 0.0


def test_the_goal_delta_is_exactly_the_pv_of_the_pi_that_ended():
    """The backward engine is linear in the cost stream (no clamp anywhere), so
    this identity — not just the inequality — is what proves the wiring."""
    payoff = build(cfg=with_housing(housing_cfg(payoff_date="2050-01-01")))
    never = build(cfg=with_housing(housing_cfg(payoff_date=None)))

    pv = sum((float(never.housing_by_age.loc[a]) - float(payoff.housing_by_age.loc[a]))
             / 1.05 ** (a - 65) for a in range(65, 91))
    assert never.nest_egg_goal - payoff.nest_egg_goal == approx(pv, rel=1e-9)


def test_housing_costs_land_in_the_lifetime_spend_column():
    """No new column — that is what keeps the lifetime goldens and the
    `shape == (49, 19)` assertion above in place."""
    s = build(cfg=with_housing(housing_cfg()))
    base = build()
    assert s.lifetime.shape == base.lifetime.shape
    assert list(s.lifetime.columns) == list(base.lifetime.columns)
    assert float(s.lifetime.loc[70, "spend"]) == approx(
        s.spend_by_phase["go_go"] + float(s.housing_by_age.loc[70]), rel=1e-9)


# ── Gate (b): a sale STEPS the portfolio up ──────────────────────────────────

def test_a_pre_retirement_sale_steps_the_projected_balance_up():
    """`projected_at_retirement` is the balance *entering* retirement, so the
    sale has to land before it. Ages 55→64 are ten growth steps."""
    sold = build(cfg=with_housing(housing_cfg(
        sale={"age": 55, "disposition": "rent", "monthly_rent": 2_100.0})))
    kept = build(cfg=with_housing(housing_cfg()))

    assert sold.projected_at_retirement > kept.projected_at_retirement
    net = sold.sale_proceeds["net"]
    assert (sold.projected_at_retirement - kept.projected_at_retirement
            == approx(net * 1.05 ** 10, rel=1e-9))


def test_a_post_retirement_sale_lands_in_the_walk_and_lowers_the_goal():
    """The other side of the seam: the inflow reaches the portfolio the year
    after the sale, and rides the unclamped `income_by_age` on the backward leg
    so the goal drops too.

    The step is not *exactly* `net · 1.05` here, and that is correct: selling the
    home you live in also switches that year's cost stream to rent, which changes
    the draw. The unmixed version is the test below.
    """
    sold = build(cfg=with_housing(housing_cfg(
        sale={"age": 70, "disposition": "rent", "monthly_rent": 2_100.0})))
    kept = build(cfg=with_housing(housing_cfg()))

    assert sold.projected_at_retirement == approx(kept.projected_at_retirement, rel=1e-9)
    assert float(sold.lifetime.loc[70, "total"]) == approx(
        float(kept.lifetime.loc[70, "total"]), rel=1e-9)     # nothing before the event
    assert (float(sold.lifetime.loc[71, "total"])
            > float(kept.lifetime.loc[71, "total"]) + 0.9 * sold.sale_proceeds["net"])
    assert sold.nest_egg_goal < kept.nest_egg_goal


# A property that is neither where you live nor let to anyone: no carry, no rent.
# Selling it moves exactly one thing — the lump sum — so the plumbing can be
# pinned to the cent without the cost-stream switch mixed in.
VACANT_LOT = {
    "occupancy": "rental", "label": "Lot 7", "current_value": 400_000.0,
    "purchase_price": 200_000.0, "purchase_date": "2010-01-01",
    "real_appreciation": 0.01, "monthly_payment": 0.0, "escrow_monthly": 0.0,
    "maintenance_pct": 0.0, "monthly_rent_income": 0.0, "payoff_date": None,
}


def test_the_inflow_alone_lands_in_taxable_before_that_years_growth():
    sold = build(cfg=with_housing(housing_cfg(
        properties=[HOUSE, {**VACANT_LOT, "sale": {"age": 70, "disposition": "rent"}}])))
    kept = build(cfg=with_housing(housing_cfg(properties=[HOUSE, VACANT_LOT])))

    assert float(sold.lifetime.loc[70, "total"]) == approx(
        float(kept.lifetime.loc[70, "total"]), rel=1e-9)
    assert (float(sold.lifetime.loc[71, "total"]) - float(kept.lifetime.loc[71, "total"])
            == approx(sold.sale_proceeds["net"] * 1.05, rel=1e-9))


def test_the_sale_switches_the_cost_stream_as_well_as_paying_out():
    """Selling the home you live in and renting after is one event with two
    effects; modelling only the proceeds would leave you paying a carry on a
    house you no longer own."""
    sold = build(cfg=with_housing(housing_cfg(
        sale={"age": 70, "disposition": "rent", "monthly_rent": 2_100.0})))
    assert float(sold.housing_by_age.loc[70]) == approx(25_200.0, rel=1e-9)
    assert float(sold.home_equity.loc[70, "equity"]) == 0.0
    assert sold.sale_proceeds["age"] == 70
    assert sold.sale_proceeds["exclusion"] > 0.0     # §121, it is a primary


# ── the rental channel ───────────────────────────────────────────────────────

def test_net_rental_income_is_taxed_and_never_appears_as_a_housing_cost():
    """The settled sign convention: a rental's carry offsets its rent (Schedule
    E) rather than joining where-you-live, so the tax is right."""
    s = build(cfg=with_housing(housing_cfg(properties=[
        HOUSE,
        {**HOUSE, "occupancy": "rental", "label": "Oak St",
         "monthly_payment": 0.0, "escrow_monthly": 350.0, "payoff_date": None,
         "monthly_rent_income": 3_200.0, "vacancy_pct": 0.08},
    ])))
    kept = build(cfg=with_housing(housing_cfg()))

    # Where you live is unchanged by letting a second house to tenants...
    assert s.housing_by_age.equals(kept.housing_by_age)
    # ...but its net income offsets the living need AND is taxed.
    assert float(s.rental_net_income_by_age.loc[70]) > 0.0
    assert float(s.lifetime.loc[70, "ordinary_income"]) > float(
        kept.lifetime.loc[70, "ordinary_income"])
    assert float(s.lifetime.loc[70, "income"]) == approx(
        float(kept.lifetime.loc[70, "income"]), rel=1e-9)     # `income` stays SS-only
    assert s.nest_egg_goal < kept.nest_egg_goal


# ═════════════════════════════════════════════════════════════════════════════
# Housing UI seams (.planning/SPEC-housing.md, Phase 4)
# ═════════════════════════════════════════════════════════════════════════════
#
# `housing_overrides` is the Coast module's live-edit channel: the form's state
# has to move the BANs before it is saved, exactly as the other three assumption
# modules do. `housing_breakdown` is the itemization its cost chart draws.

def test_housing_overrides_beats_the_stored_block():
    """The live form wins over whatever is persisted — otherwise every edit would
    need a save + reload round-trip to show up."""
    stored = build(cfg=with_housing(housing_cfg(payoff_date=None)))
    live = build(cfg=with_housing(housing_cfg(payoff_date=None)),
                 housing_overrides=housing_cfg(payoff_date="2050-01-01"))

    assert live.nest_egg_goal < stored.nest_egg_goal
    assert float(live.housing_by_age.loc[66]) < float(live.housing_by_age.loc[65])


def test_housing_overrides_none_falls_back_to_config():
    """None means "the form hasn't reported yet", not "no housing" — the first
    paint must already reflect a saved configuration."""
    stored = build(cfg=with_housing(housing_cfg()))
    explicit = build(cfg=with_housing(housing_cfg()), housing_overrides=None)
    assert explicit.nest_egg_goal == approx(stored.nest_egg_goal, rel=1e-9)
    assert explicit.housing_by_age.equals(stored.housing_by_age)


def test_an_untouched_form_overrides_to_nothing():
    """The shape `pages/coast.py::_housing_from_form` emits before the user has
    touched anything: every field None, no property. It must be as inert as no
    block at all, or simply opening the module would cut the goal."""
    untouched = {
        "residence": {"mode": None, "monthly_rent": None, "real_rent_growth": 0.0},
        "properties": [],
        "inflation": 0.025,
    }
    s = build(housing_overrides=untouched)
    base = build()
    assert s.housing_by_age is None
    assert s.housing_breakdown is None
    assert s.spend_by_phase["go_go"] == approx(base.spend_by_phase["go_go"], rel=1e-9)
    assert s.nest_egg_goal == approx(base.nest_egg_goal, rel=1e-9)


def test_an_override_can_turn_housing_off_again():
    """Clearing the form has to un-exclude `mortgage`, not strand the user with a
    zero cost stream and a permanently cut goal."""
    off = build(cfg=with_housing(housing_cfg()),
                housing_overrides={"residence": {"mode": None}, "properties": []})
    base = build()
    assert off.housing_by_age is None
    assert off.nest_egg_goal == approx(base.nest_egg_goal, rel=1e-9)


@pytest.mark.parametrize("block", [
    pytest.param(housing_cfg(), id="own"),
    pytest.param({"residence": {"mode": "rent", "monthly_rent": 2_100.0},
                  "properties": [], "inflation": 0.025}, id="rent"),
    pytest.param(housing_cfg(properties=[
        {**HOUSE, "sale": {"age": 72, "disposition": "rent", "monthly_rent": 2_100.0}},
    ]), id="own-then-sell"),
])
def test_the_breakdown_total_is_the_cost_stream(block):
    """The invariant that keeps the module's chart honest: what the engine spends
    and what the chart itemizes are the same numbers, splice included.

    The chart frame is *wider* — it spans current_age…death_age so the payoff
    cliff is visible on the same axis as the portfolio chart — so the invariant
    is over its retirement-window slice."""
    s = build(cfg=with_housing(block))
    bd = s.housing_breakdown

    assert bd.index[0] == s.current_age          # chart starts today...
    assert s.housing_by_age.index[0] == s.retirement_age   # ...engine at retirement
    assert bd["total"].loc[s.retirement_age:].equals(s.housing_by_age)
    assert bd["rental_net"].loc[s.retirement_age:].equals(s.rental_net_income_by_age)
    assert bd["total"].sum() == approx(
        bd[["pi", "escrow", "maintenance", "rent"]].sum().sum(), rel=1e-9)
    assert bd["net"].equals(bd["total"] - bd["rental_net"])


def test_the_breakdown_shows_the_cliff_landing_on_a_floor():
    """P&I ends at payoff; escrow and maintenance do not. This is the whole
    picture the cost chart exists to draw."""
    s = build(cfg=with_housing(housing_cfg(payoff_date="2050-01-01")))   # age 66
    bd = s.housing_breakdown

    assert float(bd.loc[65, "pi"]) > 0.0
    assert float(bd.loc[67, "pi"]) == 0.0
    assert float(bd.loc[90, "escrow"]) > 0.0
    assert float(bd.loc[90, "maintenance"]) > 0.0
    assert float(bd.loc[90, "rent"]) == 0.0
    # Renting is the mirror image: only the rent column carries anything.
    r = build(cfg=with_housing({"residence": {"mode": "rent", "monthly_rent": 2_100.0},
                                "properties": [], "inflation": 0.025}))
    assert float(r.housing_breakdown.loc[70, "rent"]) == approx(25_200.0, rel=1e-9)
    assert r.housing_breakdown[["pi", "escrow", "maintenance"]].abs().sum().sum() == 0.0


def test_the_net_position_answers_rent_here_and_let_that_out():
    """Rent an apartment while letting a house to tenants — the case the two
    channels exist for. Neither series alone says whether the household is ahead;
    their difference does, and that difference is what the module charts."""
    block = {
        "residence": {"mode": "rent", "monthly_rent": 2_100.0},
        "properties": [{**HOUSE, "occupancy": "rental",
                        "monthly_rent_income": 4_100.0, "vacancy_pct": 0.08}],
        "inflation": 0.025,
    }
    s = build(cfg=with_housing(block))

    cost = s.housing_breakdown["total"]
    assert float(cost.loc[70]) == approx(2_100.0 * 12, rel=1e-9)   # rent only
    # The let house's carry never lands in the cost stream — it nets against its
    # own rent instead, which is what makes the tax right.
    assert float(s.housing_breakdown.loc[70, "pi"]) == 0.0

    net = cost - s.rental_net_income_by_age
    assert float(net.loc[70]) < 0.0        # the rental more than covers the flat
    assert float(net.loc[70]) == approx(
        float(cost.loc[70]) - float(s.rental_net_income_by_age.loc[70]), rel=1e-9)


# ── goldens ──────────────────────────────────────────────────────────────────

def _housing_scenario():
    """Own the house (payoff at 66) and let a second to tenants; sell the house
    at 72 and rent afterwards — every Phase 3 channel active at once."""
    return build(cfg=with_housing(housing_cfg(properties=[
        {**HOUSE, "sale": {"age": 72, "disposition": "rent",
                           "monthly_rent": 2_100.0}},
        {**HOUSE, "occupancy": "rental", "label": "Oak St",
         "current_value": 400_000.0, "purchase_price": 200_000.0,
         "purchase_date": "2010-01-01", "monthly_payment": 0.0,
         "escrow_monthly": 350.0, "note_rate": 0.0, "payoff_date": None,
         "monthly_rent_income": 3_200.0, "vacancy_pct": 0.08},
    ])))


def test_housing_lifetime_golden():
    _assert_frame_matches_golden(_housing_scenario().lifetime,
                                 GOLDENS / "scenario_housing_lifetime.csv")


def test_housing_by_age_golden():
    _assert_series_matches_golden(_housing_scenario().housing_by_age,
                                  GOLDENS / "scenario_housing_by_age.csv")


def test_housing_equity_golden():
    _assert_frame_matches_golden(_housing_scenario().home_equity,
                                 GOLDENS / "scenario_housing_equity.csv")


# ── golden generation (run ONCE per characterization epoch) ───────────────────

def _generate_goldens() -> None:
    """Write the scenario_* goldens from current behavior.

    WARNING: run once per characterization epoch, on known-good main, then
    commit. A refactor may NEVER regenerate these — if a refactor changes the
    numbers, the refactor is wrong. Existing files are skipped rather than
    overwritten, so a later phase can add goldens here safely.
    """
    GOLDENS.mkdir(exist_ok=True)
    base = build()
    glide = build(return_strategy="moderate", glide_params=GLIDE_PARAMS)
    hsg = _housing_scenario()

    frames = [
        ("scenario_base_lifetime.csv", base.lifetime),
        ("scenario_glide_moderate_lifetime.csv", glide.lifetime),
        ("scenario_housing_lifetime.csv", hsg.lifetime),
        ("scenario_housing_equity.csv", hsg.home_equity.rename_axis("age")),
    ]
    series = [
        ("scenario_base_ss_by_age.csv", base.ss_by_age),
        ("scenario_base_healthcare_by_age.csv", base.healthcare_by_age),
        ("scenario_housing_by_age.csv", hsg.housing_by_age),
    ]

    for name, df in frames:
        path = GOLDENS / name
        if path.exists():
            print(f"skipping existing golden (never overwritten): {path}")
            continue
        df.to_csv(path, float_format="%.17g", index_label="age")
        print(f"wrote {path} ({len(df)} rows)")

    for name, s in series:
        path = GOLDENS / name
        if path.exists():
            print(f"skipping existing golden (never overwritten): {path}")
            continue
        s.rename("value").to_csv(path, float_format="%.17g", index_label="age")
        print(f"wrote {path} ({len(s)} rows)")


if __name__ == "__main__":
    _generate_goldens()
