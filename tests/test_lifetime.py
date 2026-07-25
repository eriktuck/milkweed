"""Characterization tests: the unified lifetime engine + accumulation pure math.

Pins the CURRENT behavior of:
  * core/domain/lifetime.py::simulate_lifetime (the shared engine) and
    project_balances_to_retirement (its accumulation-only wrapper)
  * core/domain/accumulation.py (coast_target, required_monthly_contribution,
    trailing_12mo_contribution)
  * core/domain/assumptions.py defaults (default_monthly_contribution,
    default_annual_retirement_spend, default_contribution_allocation)

forecast.py's project_portfolio / forecast_summary were DELETED in Phase 5 on
the proof that simulate_lifetime reproduces them: the frozen golden
forecast_project_portfolio.csv and the pinned constants in the reconciliation
tests below are that proof, kept live against the engine.

Cases ported from scripts/verify_lifetime_engine.py and
scripts/verify_balance_projector.py. All inputs are synthetic; no Firestore,
no network, no wall-clock dependence (trailing_12mo_contribution anchors on
the max transaction date, not today — see its tests).

Goldens live in tests/goldens/ (lifetime_*.csv, forecast_*.csv). They were
generated ONCE from current main via `uv run python -m tests.test_lifetime`;
a refactor may NEVER regenerate them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from core.domain.accumulation import (
    coast_target,
    required_monthly_contribution,
    trailing_12mo_contribution,
)
from core.domain.assumptions import (
    default_annual_retirement_spend,
    default_contribution_allocation,
    default_monthly_contribution,
)
from core.domain.lifetime import (
    project_balances_to_retirement,
    simulate_lifetime,
)

# Column contract of the unified frame (order matters — pinned).
LIFETIME_COLS = [
    "total", "taxable", "trad", "roth", "contribution", "life_phase",
    "spend_phase", "spend", "healthcare", "income", "rmd", "withdrawal",
    "w_taxable", "w_trad", "w_roth", "ordinary_income", "ltcg_gain",
    "tax", "net_spend",
]

# ── Fixture builders (synthetic, deterministic) ────────────────────────────────

def _accum_only_case(r_accum=0.05, **kw):
    """Accumulation-only walk: working (40–54) + coast (55–64), one zero-spend
    retirement row at 65 (death_age == retirement_age)."""
    return simulate_lifetime(
        buckets={"taxable": 100_000.0, "trad": 50_000.0, "roth": 25_000.0},
        allocation={"taxable": 6_000.0, "trad": 23_500.0, "roth": 6_500.0},
        current_age=40, coast_age=55, retirement_age=65, death_age=65,
        spend_by_phase={}, slow_go_age=65, no_go_age=65,
        r_accum=r_accum, r_retire=0.05, **kw,
    )


def _drawdown_only_case(**kw):
    """Drawdown-only walk — the verify_lifetime_engine tax-aware case, run
    directly through the engine (no accumulation window)."""
    ss = {a: (24_000.0 if a >= 67 else 0.0) for a in range(65, 91)}
    hc = {a: (15_000.0 if a < 65 else 6_000.0) + (40_000.0 if a >= 85 else 0.0)
          for a in range(65, 91)}
    return simulate_lifetime(
        buckets={"taxable": 400_000.0, "trad": 300_000.0, "roth": 100_000.0},
        allocation={},
        current_age=65, coast_age=65, retirement_age=65, death_age=90,
        spend_by_phase={"go_go": 70_000.0, "slow_go": 60_000.0, "no_go": 50_000.0},
        slow_go_age=75, no_go_age=85, r_accum=0.04, r_retire=0.04,
        ss_by_age=ss, healthcare_by_age=hc, rmd_start=75, gain_fraction=0.5, **kw,
    )


def _full_walk_case(**kw):
    """Full walk: working (50–59) → coast (60–66) → retirement (67–92),
    crossing rmd_start=75 with a live trad balance."""
    ss = {a: (30_000.0 if a >= 70 else 0.0) for a in range(67, 93)}
    hc = {a: 8_000.0 for a in range(67, 93)}
    return simulate_lifetime(
        buckets={"taxable": 200_000.0, "trad": 400_000.0, "roth": 100_000.0},
        allocation={"taxable": 5_000.0, "trad": 23_500.0, "roth": 6_500.0},
        current_age=50, coast_age=60, retirement_age=67, death_age=92,
        spend_by_phase={"go_go": 70_000.0, "slow_go": 60_000.0, "no_go": 50_000.0},
        slow_go_age=75, no_go_age=85, r_accum=0.05, r_retire=0.04,
        ss_by_age=ss, healthcare_by_age=hc, rmd_start=75, gain_fraction=0.5, **kw,
    )


def _glide_path():
    """Dense age→rate glide: 7% at 40 declining 0.1 pp/yr to 4.6% at 64."""
    return {age: 0.07 - 0.001 * (age - 40) for age in range(40, 65)}


def _csp_plan_and_labels():
    plan = {"mortgage": 2_000.0, "groceries": 600.0, "guilt_free": 800.0,
            "vacations": 300.0, "income": 7_000.0, "savings": 1_500.0}
    labels = {"mortgage": "fixed", "groceries": "fixed",
              "guilt_free": "guilt-free", "vacations": "sinking",
              "income": "income", "savings": "investments"}
    return plan, labels


_STRING_COLS = frozenset({"life_phase", "spend_phase", "phase"})


def _read_golden(goldens_dir: Path, name: str) -> pd.DataFrame:
    # keep_default_na=False so accumulation rows' spend_phase round-trips as ""
    # (not NaN). Columns whose values are all whole numbers round-trip as
    # int64, so coerce every numeric column back to float64 — the comparison
    # stays strict on values.
    golden = pd.read_csv(goldens_dir / name, index_col="age",
                         keep_default_na=False)
    for col in golden.columns:
        if col not in _STRING_COLS:
            golden[col] = golden[col].astype("float64")
    return golden


def _assert_matches_golden(df: pd.DataFrame, goldens_dir: Path, name: str) -> None:
    golden = _read_golden(goldens_dir, name)
    pdt.assert_frame_equal(df, golden, rtol=1e-9)


# ── simulate_lifetime: frame contract + goldens ────────────────────────────────

def test_lifetime_column_contract():
    """The unified frame's column order is pinned."""
    df = _accum_only_case()
    assert list(df.columns) == LIFETIME_COLS
    assert df.index.name == "age"


def test_lifetime_accum_only_golden(goldens_dir):
    """Accumulation-only walk matches its golden frame."""
    _assert_matches_golden(_accum_only_case(), goldens_dir, "lifetime_accum_only.csv")


def test_lifetime_accum_only_pins():
    """Working/coast structure and at-65 balances of the accumulation walk."""
    df = _accum_only_case()
    assert float(df.loc[55, "total"]) == pytest.approx(1140640.720573174, rel=1e-9)
    assert float(df.loc[65, "total"]) == pytest.approx(1857983.540825193, rel=1e-9)
    assert float(df.loc[65, "taxable"]) == pytest.approx(549530.7317845721, rel=1e-9)
    assert float(df.loc[65, "trad"]) == pytest.approx(995324.0946822835, rel=1e-9)
    assert float(df.loc[65, "roth"]) == pytest.approx(313128.71435833746, rel=1e-9)

    assert (df.loc[40:54, "life_phase"] == "working").all()
    assert (df.loc[55:64, "life_phase"] == "coast").all()
    assert df.loc[65, "life_phase"] == "retirement"
    # spend_phase is "" during accumulation; the zero-spend retirement row is
    # classified by phase_for_age (age 65 ≥ no_go_age 65 → no_go).
    assert (df.loc[40:64, "spend_phase"] == "").all()
    assert df.loc[65, "spend_phase"] == "no_go"

    assert (df.loc[40:54, "contribution"] == 36_000.0).all()
    assert (df.loc[55:, "contribution"] == 0.0).all()
    # Drawdown flow columns are zero-filled through accumulation.
    for col in ("spend", "healthcare", "income", "rmd", "withdrawal", "tax",
                "net_spend"):
        assert (df.loc[40:64, col] == 0.0).all()


def test_lifetime_drawdown_only_golden(goldens_dir):
    """Drawdown-only walk matches its golden frame."""
    _assert_matches_golden(_drawdown_only_case(), goldens_dir,
                           "lifetime_drawdown_only.csv")


@pytest.mark.parametrize("age, expected", [
    (65, {"total": 800000.0, "taxable": 400000.0, "trad": 300000.0,
          "roth": 100000.0, "rmd": 0.0, "withdrawal": 76000.0,
          "w_taxable": 76000.0, "w_trad": 0.0, "w_roth": 0.0,
          "ordinary_income": 0.0, "ltcg_gain": 38000.0, "tax": 0.0,
          "net_spend": 76000.0}),
    (75, {"total": 460387.75830891856, "taxable": 0.0,
          "trad": 312363.3298170841, "roth": 148024.42849183446,
          "rmd": 12697.696334027807, "withdrawal": 43166.55,
          "w_taxable": 0.0, "w_trad": 43166.55, "w_roth": 0.0,
          "ordinary_income": 43166.55, "ltcg_gain": 0.0,
          "tax": 1166.6550000000004, "net_spend": 65999.895}),
    (85, {"total": 144968.58819745423, "taxable": 0.0, "trad": 0.0,
          "roth": 144968.58819745423, "rmd": 0.0, "withdrawal": 72000.0,
          "w_taxable": 0.0, "w_trad": 0.0, "w_roth": 72000.0,
          "ordinary_income": 0.0, "ltcg_gain": 0.0, "tax": 0.0,
          "net_spend": 96000.0}),
    # Portfolio fully depleted: withdrawal 0, but net_spend still records the
    # Social Security income riding through (24000) — actual behavior.
    (90, {"total": 0.0, "taxable": 0.0, "trad": 0.0, "roth": 0.0,
          "rmd": 0.0, "withdrawal": 0.0, "w_taxable": 0.0, "w_trad": 0.0,
          "w_roth": 0.0, "ordinary_income": 0.0, "ltcg_gain": 0.0,
          "tax": 0.0, "net_spend": 24000.0}),
])
def test_lifetime_drawdown_sample_ages(age, expected):
    """Spot pins at the verify_lifetime_engine golden ages (full precision)."""
    df = _drawdown_only_case()
    phases = {65: "go_go", 75: "slow_go", 85: "no_go", 90: "no_go"}
    assert df.loc[age, "spend_phase"] == phases[age]
    assert df.loc[age, "life_phase"] == "retirement"
    for col, want in expected.items():
        assert float(df.loc[age, col]) == pytest.approx(want, rel=1e-9, abs=1e-9)


def test_lifetime_full_walk_golden(goldens_dir):
    """Full working→coast→retirement walk matches its golden frame."""
    _assert_matches_golden(_full_walk_case(), goldens_dir, "lifetime_full_walk.csv")


def test_lifetime_full_walk_rmd_seam():
    """RMDs are zero before rmd_start and equal trad/divisor at rmd_start."""
    df = _full_walk_case()
    assert (df.loc[67:74, "rmd"] == 0.0).all()
    assert float(df.loc[75, "rmd"]) == pytest.approx(74142.99258133734, rel=1e-9)
    # divisor at 75 is 24.6 (IRS Uniform Lifetime); rmd = trad entering 75 / 24.6
    assert float(df.loc[75, "rmd"]) == pytest.approx(
        float(df.loc[75, "trad"]) / 24.6, rel=1e-12)
    assert float(df.loc[67, "total"]) == pytest.approx(2223855.3490620716, rel=1e-9)
    assert float(df.loc[92, "total"]) == pytest.approx(3776420.3775999807, rel=1e-9)
    assert (df.loc[50:59, "life_phase"] == "working").all()
    assert (df.loc[60:66, "life_phase"] == "coast").all()
    assert (df.loc[67:, "life_phase"] == "retirement").all()


# ── simulate_lifetime: rate forms (scalar vs age-lookup) ──────────────────────

def test_lifetime_glide_golden(goldens_dir):
    """Accumulation with a dense {age: rate} glide matches its golden frame."""
    df = simulate_lifetime(
        buckets={"taxable": 100_000.0, "trad": 50_000.0, "roth": 25_000.0},
        allocation={"taxable": 6_000.0, "trad": 23_500.0, "roth": 6_500.0},
        current_age=40, coast_age=55, retirement_age=65, death_age=65,
        spend_by_phase={}, slow_go_age=65, no_go_age=65,
        r_accum=_glide_path(), r_retire=0.04,
    )
    _assert_matches_golden(df, goldens_dir, "lifetime_glide_path.csv")
    assert float(df.loc[65, "total"]) == pytest.approx(2094272.8353330004, rel=1e-9)


def test_lifetime_glide_series_equals_dict():
    """A pd.Series glide is interchangeable with the equivalent dict."""
    kwargs = dict(
        buckets={"taxable": 100_000.0, "trad": 50_000.0, "roth": 25_000.0},
        allocation={"taxable": 6_000.0, "trad": 23_500.0, "roth": 6_500.0},
        current_age=40, coast_age=55, retirement_age=65, death_age=65,
        spend_by_phase={}, slow_go_age=65, no_go_age=65, r_retire=0.04,
    )
    from_dict = simulate_lifetime(r_accum=_glide_path(), **kwargs)
    from_series = simulate_lifetime(r_accum=pd.Series(_glide_path()), **kwargs)
    pdt.assert_frame_equal(from_dict, from_series)


def test_lifetime_constant_glide_equals_scalar():
    """A flat {age: r} map reproduces the scalar-rate walk exactly."""
    flat = {age: 0.05 for age in range(40, 65)}
    pdt.assert_frame_equal(_accum_only_case(r_accum=0.05),
                           _accum_only_case(r_accum=flat))


def test_lifetime_sparse_glide_falls_back_to_terminal_rate():
    """Ages missing from a sparse glide use the rate at the map's OLDEST age
    (terminal/most de-risked), even for ages between listed entries."""
    df = simulate_lifetime(
        buckets={"taxable": 100_000.0, "trad": 0.0, "roth": 0.0},
        allocation={"taxable": 10_000.0, "trad": 0.0, "roth": 0.0},
        current_age=40, coast_age=43, retirement_age=43, death_age=43,
        spend_by_phase={}, slow_go_age=43, no_go_age=43,
        r_accum={40: 0.07, 45: 0.06, 50: 0.05}, r_retire=0.05,
    )
    # age 40 at 7%; ages 41–42 (missing) at the max-key rate 5% (not 6%).
    expected = ((100_000 * 1.07 + 10_000) * 1.05 + 10_000) * 1.05 + 10_000
    assert float(df.loc[43, "total"]) == pytest.approx(expected, rel=1e-12)
    assert float(df.loc[43, "total"]) == pytest.approx(149_492.5, rel=1e-12)


def test_lifetime_empty_glide_map_means_zero_rate():
    """An empty {} glide degrades to a 0.0 rate (contributions only)."""
    df = simulate_lifetime(
        buckets={"taxable": 100_000.0, "trad": 0.0, "roth": 0.0},
        allocation={"taxable": 1_000.0, "trad": 0.0, "roth": 0.0},
        current_age=40, coast_age=42, retirement_age=42, death_age=42,
        spend_by_phase={}, slow_go_age=42, no_go_age=42,
        r_accum={}, r_retire=0.05,
    )
    assert float(df.loc[42, "total"]) == 102_000.0


# ── simulate_lifetime: inflows + taxed ordinary income (SPEC-housing Phase 2) ──

@pytest.mark.parametrize("case", [_accum_only_case, _drawdown_only_case,
                                  _full_walk_case])
@pytest.mark.parametrize("payload", [
    {"inflows_by_age": None, "ordinary_income_by_age": None},           # explicit None
    {"inflows_by_age": {}, "ordinary_income_by_age": {}},               # empty map
    {"inflows_by_age": {a: 0.0 for a in range(40, 93)},                 # explicit zeros
     "ordinary_income_by_age": {a: 0.0 for a in range(40, 93)}},
])
def test_new_channels_are_no_ops_when_unset(case, payload):
    """The hard constraint: absent housing config must change nothing, anywhere.

    None, an empty map and a map of zeros must all reproduce the baseline frame
    exactly, in every leg — that is what keeps tests/goldens/*.csv from moving.
    """
    pdt.assert_frame_equal(case(**payload), case())


def test_inflow_lands_in_taxable_before_growth_during_accumulation():
    """A lump sum at 45 is in taxable and compounding by 46."""
    base = _accum_only_case()
    df = _accum_only_case(inflows_by_age={45: 200_000.0})
    # Untouched through age 45 (the row records the balance *entering* the year).
    assert float(df.loc[45, "taxable"]) == pytest.approx(
        float(base.loc[45, "taxable"]), rel=1e-12)
    # Pre-growth: the inflow earns that year's return.
    assert float(df.loc[46, "taxable"]) - float(base.loc[46, "taxable"]) == (
        pytest.approx(200_000.0 * 1.05, rel=1e-12))


def test_inflow_is_not_clipped_by_the_cash_need_clamp():
    """The reason inflows need their own hook rather than riding `income_by_age`.

    `cash_need = max(spend + hc − income, 0)` would clip a large lump sum to
    zero; routed through `inflows_by_age` the whole sum reaches the portfolio.
    """
    base = _drawdown_only_case()
    df = _drawdown_only_case(inflows_by_age={70: 500_000.0})
    # Nothing before the event; the full step-up lands the very next year.
    assert float(df.loc[70, "total"]) == pytest.approx(
        float(base.loc[70, "total"]), rel=1e-12)
    assert float(df.loc[71, "total"]) - float(base.loc[71, "total"]) == (
        pytest.approx(500_000.0 * 1.04, rel=1e-9))


def test_ordinary_income_offsets_spending_and_is_taxed():
    """Taxed outside income both funds the year and enters the fixed-point."""
    base = _drawdown_only_case()
    df = _drawdown_only_case(
        ordinary_income_by_age={a: 12_000.0 for a in range(65, 91)})
    # It offsets the living need, so less is drawn from the portfolio...
    assert float(df.loc[70, "withdrawal"]) < float(base.loc[70, "withdrawal"])
    # ...and unlike ss_by_age it shows up as taxable ordinary income.
    assert float(df.loc[70, "ordinary_income"]) == pytest.approx(
        float(df.loc[70, "w_trad"]) + 12_000.0, rel=1e-12)
    assert float(df.loc[70, "income"]) == 24_000.0    # `income` stays SS-only


def test_ordinary_income_is_taxed_unlike_social_security():
    """The same dollars through the taxed channel leave a smaller portfolio."""
    common = dict(
        buckets={"taxable": 400_000.0, "trad": 300_000.0, "roth": 100_000.0},
        allocation={},
        current_age=65, coast_age=65, retirement_age=65, death_age=90,
        spend_by_phase={"go_go": 70_000.0, "slow_go": 60_000.0, "no_go": 50_000.0},
        slow_go_age=75, no_go_age=85, r_accum=0.04, r_retire=0.04,
        healthcare_by_age={a: 6_000.0 for a in range(65, 91)},
        rmd_start=75, gain_fraction=0.5,
    )
    stream = {a: 40_000.0 for a in range(65, 91)}
    untaxed = simulate_lifetime(ss_by_age=stream, **common)
    taxed = simulate_lifetime(ordinary_income_by_age=stream, **common)
    assert float(taxed.loc[90, "total"]) < float(untaxed.loc[90, "total"])


def test_negative_ordinary_income_raises_the_cash_need():
    """A rental running at a loss is a real cost, and must not be clamped away."""
    base = _drawdown_only_case()
    loss = _drawdown_only_case(
        ordinary_income_by_age={a: -10_000.0 for a in range(65, 91)})
    assert float(loss.loc[70, "withdrawal"]) > float(base.loc[70, "withdrawal"])


def test_both_channels_accept_series_as_well_as_dicts():
    ages = range(65, 91)
    payload = dict(inflows_by_age={a: 1_000.0 for a in ages},
                   ordinary_income_by_age={a: 5_000.0 for a in ages})
    as_series = {k: pd.Series(v) for k, v in payload.items()}
    pdt.assert_frame_equal(_drawdown_only_case(**payload),
                           _drawdown_only_case(**as_series))


# ── simulate_lifetime: edge cases + invariants ────────────────────────────────

def test_lifetime_zero_balance_buckets():
    """All-zero buckets in drawdown: nothing to withdraw, balances stay 0,
    net_spend records only the income stream."""
    df = simulate_lifetime(
        buckets={"taxable": 0.0, "trad": 0.0, "roth": 0.0}, allocation={},
        current_age=70, coast_age=70, retirement_age=70, death_age=72,
        spend_by_phase={"go_go": 50_000.0, "slow_go": 50_000.0, "no_go": 50_000.0},
        slow_go_age=75, no_go_age=85, r_accum=0.04, r_retire=0.04,
        ss_by_age={70: 20_000.0, 71: 20_000.0, 72: 20_000.0}, rmd_start=75,
    )
    assert (df["total"] == 0.0).all()
    assert (df[["taxable", "trad", "roth"]] == 0.0).all().all()
    assert (df["withdrawal"] == 0.0).all()
    assert (df["tax"] == 0.0).all()
    assert (df["net_spend"] == 20_000.0).all()   # income only; spend unfunded
    assert (df["spend"] == 50_000.0).all()


def test_lifetime_bucket_split_is_additive():
    """Same total contribution split across buckets aggregates identically
    (verify_lifetime_engine.check_bucket_additivity)."""
    common = dict(
        buckets={"taxable": 250_000.0, "trad": 0.0, "roth": 0.0},
        current_age=40, coast_age=65, retirement_age=65, death_age=65,
        spend_by_phase={}, slow_go_age=65, no_go_age=65,
        r_accum=0.05, r_retire=0.05,
    )
    single = simulate_lifetime(
        allocation={"taxable": 30_000.0, "trad": 0.0, "roth": 0.0}, **common)
    split = simulate_lifetime(
        allocation={"taxable": 0.0, "trad": 23_500.0, "roth": 6_500.0}, **common)
    assert float(single.loc[65, "total"]) == pytest.approx(
        float(split.loc[65, "total"]), rel=1e-12)


def test_lifetime_reconciles_with_project_portfolio():
    """The engine's accumulation aggregate equals the value the deleted
    forecast.project_portfolio produced for the same start/contribution/rate,
    pinned as a constant before the deletion — the Phase 5 invariant."""
    engine = simulate_lifetime(
        buckets={"taxable": 250_000.0, "trad": 0.0, "roth": 0.0},
        allocation={"taxable": 30_000.0, "trad": 0.0, "roth": 0.0},
        current_age=40, coast_age=65, retirement_age=65, death_age=65,
        spend_by_phase={}, slow_go_age=65, no_go_age=65,
        r_accum=0.05, r_retire=0.05,
    )
    assert float(engine.loc[65, "total"]) == pytest.approx(
        2278401.699764478, rel=1e-9)


# ── The deleted forecast.project_portfolio, pinned via its frozen golden ──────

def test_lifetime_reproduces_project_portfolio_golden(goldens_dir):
    """simulate_lifetime reproduces the accumulation totals of the frozen
    project_portfolio golden (250k start, work 40–49, coast 50–64, 30k/yr at
    5% real) — the deletion proof, kept live against the engine. Only the
    shared columns (total, contribution) over the shared ages (40–65) compare;
    the golden's drawdown rows pin the deleted SWR draw model, not the engine."""
    engine = simulate_lifetime(
        buckets={"taxable": 250_000.0, "trad": 0.0, "roth": 0.0},
        allocation={"taxable": 30_000.0, "trad": 0.0, "roth": 0.0},
        current_age=40, coast_age=50, retirement_age=65, death_age=65,
        spend_by_phase={}, slow_go_age=65, no_go_age=65,
        r_accum=0.05, r_retire=0.05,
    )
    golden = _read_golden(goldens_dir, "forecast_project_portfolio.csv")
    for age in range(40, 66):
        assert float(engine.loc[age, "total"]) == pytest.approx(
            float(golden.loc[age, "total"]), rel=1e-9)
        assert float(engine.loc[age, "contribution"]) == pytest.approx(
            float(golden.loc[age, "contribution"]), rel=1e-9)


# ── accumulation.coast_target (ex-forecast_summary math) ──────────────────────

def test_coast_target_pins():
    """PV of the goal discounted from retirement back to the coast age — the
    coast_target values the deleted forecast_summary produced, pinned before
    its deletion."""
    assert coast_target(2_000_000.0, 0.05, 50, 65) == pytest.approx(
        962034.1961819399, rel=1e-9)
    assert coast_target(1_000_000.0, 0.05, 50, 65) == pytest.approx(
        481017.09809096996, rel=1e-9)


def test_coast_target_no_growth_window():
    """coast_age ≥ retirement_age → no growth years, target == goal."""
    assert coast_target(2_000_000.0, 0.05, 65, 65) == 2_000_000.0
    assert coast_target(2_000_000.0, 0.05, 70, 65) == 2_000_000.0


def test_required_monthly_to_coast_target():
    """required_monthly against coast_target reproduces the required_monthly
    values the deleted forecast_summary produced (250k start, work 40–49)."""
    assert required_monthly_contribution(
        250_000.0, 40, 50, coast_target(2_000_000.0, 0.05, 50, 65), 0.05,
    ) == pytest.approx(3675.831344026307, rel=1e-9)
    assert required_monthly_contribution(
        250_000.0, 40, 50, coast_target(1_000_000.0, 0.05, 50, 65), 0.05,
    ) == pytest.approx(488.9096827896473, rel=1e-9)


# ── accumulation.required_monthly_contribution ────────────────────────────────

@pytest.mark.parametrize("start, cur, coast, target, r, expected", [
    (250_000.0, 40, 50, 1_000_000.0, 0.05, 3927.369268674372),
    (100_000.0, 40, 50, 200_000.0, 0.0, 833.3333333333334),   # r == 0 branch
    (1_000_000.0, 40, 50, 1_000_000.0, 0.05, 0.0),            # already grows past
    (250_000.0, 50, 50, 1_000_000.0, 0.05, 0.0),              # no working window
])
def test_required_monthly_contribution(start, cur, coast, target, r, expected):
    """Closed-form inverse of the working-phase recurrence, with its edges."""
    got = required_monthly_contribution(start, cur, coast, target, r)
    assert got == pytest.approx(expected, rel=1e-9, abs=1e-12)


# ── assumptions defaults from the CSP plan ────────────────────────────────────

def test_default_monthly_contribution():
    """Sum of `investments`-labeled monthly plan lines; None inputs → 0."""
    plan, labels = _csp_plan_and_labels()
    assert default_monthly_contribution(plan, labels) == pytest.approx(1_500.0)
    assert default_monthly_contribution(None, labels) == 0.0
    assert default_monthly_contribution(plan, None) == 0.0


def test_default_annual_retirement_spend():
    """12 × Σ fixed + guilt-free + sinking lines; income/investments excluded."""
    plan, labels = _csp_plan_and_labels()
    assert default_annual_retirement_spend(plan, labels) == pytest.approx(44_400.0)
    assert default_annual_retirement_spend(None, labels) == 0.0
    assert default_annual_retirement_spend({}, {}) == 0.0


# ── accumulation.trailing_12mo_contribution ───────────────────────────────────
# Anchored on the MAX transaction date (not wall clock), so fixed synthetic
# dates are deterministic.

def test_trailing_12mo_contribution():
    """Sums |amount| of investments txns in the 12 months before the max date;
    a txn exactly 12 months before the max is EXCLUDED (strict > cutoff)."""
    txn = pd.DataFrame({
        "date": ["2025-06-15", "2024-07-01", "2024-06-15", "2025-01-10",
                 "2025-03-01"],
        "csp_label": ["investments", "investments", "investments",
                      "investments", "fixed"],
        # mixed signs: magnitudes are summed (abs), so the +2400 counts too.
        "amount": [-1_000.0, -500.0, -9_999.0, 2_400.0, -50.0],
    })
    # cutoff = 2025-06-15 − 12 months = 2024-06-15; the −9999 txn on the cutoff
    # date is excluded; (1000 + 500 + 2400) / 12 = 325.
    assert trailing_12mo_contribution(txn) == pytest.approx(325.0, rel=1e-9)


@pytest.mark.parametrize("txn", [
    None,
    pd.DataFrame(columns=["date", "csp_label", "amount"]),
    pd.DataFrame({"csp_label": ["investments"], "amount": [-100.0]}),   # no date col
    pd.DataFrame({"date": ["not-a-date"], "csp_label": ["investments"],
                  "amount": [-100.0]}),                                 # unparseable
    pd.DataFrame({"date": ["2025-01-01"], "csp_label": ["fixed"],
                  "amount": [-100.0]}),                                 # no investments
])
def test_trailing_12mo_contribution_degenerate_inputs(txn):
    """Missing/empty/unusable data returns 0.0."""
    assert trailing_12mo_contribution(txn) == 0.0


# ── retirement: contribution allocation + balance projector ──────────────────
# (from scripts/verify_balance_projector.py — these feed the lifetime walk)

@pytest.mark.parametrize("contribution, expected", [
    (10_000.0, {"taxable": 0.0, "trad": 10_000.0, "roth": 0.0}),
    (27_500.0, {"taxable": 0.0, "trad": 23_500.0, "roth": 4_000.0}),
    (35_500.0, {"taxable": 5_000.0, "trad": 23_500.0, "roth": 7_000.0}),
])
def test_default_contribution_allocation(contribution, expected):
    """Tax-advantaged-first fill: pre-tax cap (23.5k) → Roth cap (7k) → taxable."""
    alloc = default_contribution_allocation(contribution)
    assert alloc == expected
    assert sum(alloc.values()) == pytest.approx(contribution, rel=1e-12)


def test_project_balances_to_retirement_golden_pins():
    """The verify_balance_projector golden case at full precision."""
    proj = project_balances_to_retirement(
        {"taxable": 250_000.0, "trad": 0.0, "roth": 0.0},
        {"taxable": 0.0, "trad": 23_500.0, "roth": 6_500.0}, 25, 0.05)
    assert proj["taxable"] == pytest.approx(846588.7352248469, rel=1e-9)
    assert proj["trad"] == pytest.approx(1121586.8222227118, rel=1e-9)
    assert proj["roth"] == pytest.approx(310226.1423169202, rel=1e-9)
    assert proj["total"] == pytest.approx(2278401.699764479, rel=1e-9)


@pytest.mark.parametrize("years", [0, -3])
def test_project_balances_zero_or_negative_years(years):
    """years ≤ 0 returns the current balances unchanged (no contribution)."""
    proj = project_balances_to_retirement(
        {"taxable": 1.0, "trad": 2.0, "roth": 3.0}, {"taxable": 99.0},
        years, 0.05)
    assert proj == {"taxable": 1.0, "trad": 2.0, "roth": 3.0, "total": 6.0}


def test_project_balances_reconciles_with_project_portfolio():
    """Bucket projector aggregate == the pinned total the deleted
    forecast.project_portfolio produced for the same inputs
    (verify_balance_projector gate criterion, frozen as a constant)."""
    alloc = default_contribution_allocation(30_000.0)
    proj = project_balances_to_retirement(
        {"taxable": 250_000.0, "trad": 0.0, "roth": 0.0}, alloc, 25, 0.05)
    assert proj["total"] == pytest.approx(2278401.699764478, rel=1e-9)


# ── Golden generation (run ONCE per characterization epoch) ───────────────────
# WARNING: goldens pin current-main behavior. A refactor may NEVER regenerate
# them — if this block is being re-run, something is wrong with the process.

def _generate_goldens() -> None:
    from tests.conftest import GOLDENS_DIR

    GOLDENS_DIR.mkdir(exist_ok=True)
    frames = {
        "lifetime_accum_only.csv": _accum_only_case(),
        "lifetime_drawdown_only.csv": _drawdown_only_case(),
        "lifetime_full_walk.csv": _full_walk_case(),
        "lifetime_glide_path.csv": simulate_lifetime(
            buckets={"taxable": 100_000.0, "trad": 50_000.0, "roth": 25_000.0},
            allocation={"taxable": 6_000.0, "trad": 23_500.0, "roth": 6_500.0},
            current_age=40, coast_age=55, retirement_age=65, death_age=65,
            spend_by_phase={}, slow_go_age=65, no_go_age=65,
            r_accum=_glide_path(), r_retire=0.04,
        ),
        # forecast_project_portfolio.csv is NOT regenerable: it pins the
        # DELETED forecast.project_portfolio (written before Phase 5 removed
        # it) and survives as the frozen legacy pin the engine must reproduce.
    }
    for name, df in frames.items():
        path = GOLDENS_DIR / name
        if path.exists():
            raise SystemExit(f"REFUSING to overwrite existing golden: {path}")
        df.to_csv(path, float_format="%.17g")
        print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    _generate_goldens()
