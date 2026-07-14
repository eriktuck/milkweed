"""Characterization tests for the deterministic parts of core/domain/returns.py.

No network, no Firebase: `core.domain.returns` is pure — market data arrives as
arguments (`inflation: pd.Series`, `returns_for: Callable`), so the tests pass
synthetic per-ticker series through the seam directly. (Before Phase 6 this
file stubbed `core.services.investments` and monkeypatched the module-level
yfinance fetcher; the injection seam replaced both.)

Never exercised here (network-bound by design, now in
core/services/market_data.py): `inflation_series`, the real
`nominal_annual_returns`, and `build_returns_payload`.

Bootstrap Monte Carlo functions are pinned on INVARIANTS only (shape, N_SIMS,
seeded determinism, percentile monotonicity) — never exact golden paths.

Goldens live in tests/goldens/returns_*.csv — generated ONCE from current main
via ``uv run python -m tests.test_returns``. A refactor may NEVER regenerate
them; tests only read.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import core.domain.returns as R

GOLDENS_DIR = Path(__file__).parent / "goldens"


# ── Synthetic market data ────────────────────────────────────────────────────────

def _series(start: int, end: int, base: float, amp: float) -> pd.Series:
    """Deterministic annual nominal returns: base ± amp·{−2..2}% cycling by year."""
    return pd.Series(
        {y: base + amp * (((y * 7) % 5) - 2) / 100.0 for y in range(start, end + 1)},
        dtype="float64")


FAKE_NOMINAL: dict[str, pd.Series] = {
    "VTSAX": _series(1998, 2023, 0.08, 3.0),
    "VEU": _series(2008, 2023, 0.06, 2.0),
    "VBTLX": _series(2002, 2023, 0.03, 1.0),
    "ZZZTOP": _series(2019, 2023, 0.12, 4.0),  # 5 obs < MIN_YEARS_HISTORY → proxy
}

FAKE_INFLATION = pd.Series(
    {y: 0.02 + ((y % 4) - 1.5) * 0.004 for y in range(1995, 2024)}, dtype="float64")

HOLDINGS = [
    {"symbol": "VTSAX", "total_value": 60000.0},
    {"symbol": "VEU", "total_value": 20000.0},
    {"symbol": "VBTLX", "total_value": 10000.0},
    {"symbol": "VMFXX", "total_value": 5000.0},
    {"symbol": "ZZZTOP", "total_value": 5000.0},   # unknown symbol → "Other"
    {"symbol": "VTI", "total_value": 0.0},          # zero value → dropped
    {"symbol": "VSS", "total_value": -100.0},       # negative value → dropped
]

EQ_POOL = [0.10, -0.02, 0.06, 0.15, -0.08]
FX_POOL = [0.03, 0.01, 0.02, 0.00, 0.04]
BOOT_POOL = [0.08, -0.12, 0.15, 0.03, -0.05, 0.22, 0.07, -0.18, 0.11, 0.04, 0.09, -0.02]


def fake_nominal(ticker: str) -> pd.Series:
    """The injected `returns_for` seam: synthetic per-ticker nominal series."""
    return FAKE_NOMINAL.get(ticker, pd.Series(dtype="float64")).copy()


# ── portfolio_weights ────────────────────────────────────────────────────────────

def test_portfolio_weights_literal():
    """Classes weighted by value; unknown symbol → Other; ≤0 values dropped."""
    assert R.portfolio_weights(HOLDINGS) == pytest.approx({
        "US Equity": 0.6, "Intl Equity": 0.2, "Bonds": 0.1,
        "Cash": 0.05, "Other": 0.05,
    }, rel=1e-9)


@pytest.mark.parametrize("holdings", [
    [],
    [{"symbol": "VTSAX", "total_value": 0.0}],
    [{"symbol": "VTSAX", "total_value": -5.0}, {"symbol": "VEU"}],
])
def test_portfolio_weights_empty(holdings):
    """Empty / zero-value portfolios yield {}."""
    assert R.portfolio_weights(holdings) == {}


# ── weighted_average_real_returns ────────────────────────────────────────────────

def _hand_class_series() -> dict[str, pd.Series]:
    return {
        "US Equity": pd.Series({2000: 0.10, 2001: -0.04, 2002: 0.08, 2003: 0.12},
                               dtype="float64"),
        "Bonds": pd.Series({2002: 0.02, 2003: 0.03, 2004: 0.01}, dtype="float64"),
    }


def test_weighted_average_real_returns_golden():
    """Per-year weighted mean, renormalising over the classes present that year."""
    out = R.weighted_average_real_returns(
        _hand_class_series(), {"US Equity": 0.75, "Bonds": 0.25})
    golden = pd.read_csv(GOLDENS_DIR / "returns_weighted_avg.csv", index_col=0)
    golden = golden["value"].astype("float64")
    golden.index = golden.index.astype(out.index.dtype)
    golden.index.name = out.index.name
    golden.name = out.name
    pd.testing.assert_series_equal(out, golden, rtol=1e-9)


def test_weighted_average_single_class_years_pass_through():
    """A year covered by only one class returns that class's value unscaled."""
    out = R.weighted_average_real_returns(
        _hand_class_series(), {"US Equity": 0.75, "Bonds": 0.25})
    assert out.loc[2000] == pytest.approx(0.10, rel=1e-9)   # US only
    assert out.loc[2004] == pytest.approx(0.01, rel=1e-9)   # Bonds only
    assert out.loc[2002] == pytest.approx(0.065, rel=1e-9)  # 0.75·0.08 + 0.25·0.02


def test_weighted_average_empty_inputs():
    """No class series, or all-zero weights, yield an empty float Series."""
    assert R.weighted_average_real_returns({}, {"US Equity": 1.0}).empty
    out = R.weighted_average_real_returns(_hand_class_series(), {"Cash": 1.0})
    assert out.empty and out.dtype == "float64"


# ── suggested_real_return ────────────────────────────────────────────────────────

def test_suggested_real_return_geometric_mean():
    """CAGR of the per-year real returns, not median/arithmetic mean."""
    got = R.suggested_real_return(pd.Series([0.10, -0.05, 0.07]))
    assert got == pytest.approx(0.03792671274950532, rel=1e-9)


def test_suggested_real_return_nan_filtered():
    """NaN years are dropped before compounding."""
    got = R.suggested_real_return(pd.Series([0.10, np.nan, 0.07]))
    assert got == pytest.approx(0.08489630840924156, rel=1e-9)


@pytest.mark.parametrize("series", [None, pd.Series(dtype="float64"),
                                    pd.Series([np.nan, np.nan])])
def test_suggested_real_return_empty_is_zero(series):
    assert R.suggested_real_return(series) == 0.0


# ── percentile_bands ─────────────────────────────────────────────────────────────

def test_percentile_bands_literal():
    """Per-year percentiles across the sims axis (linear interpolation)."""
    sims = np.arange(1.0, 16.0).reshape(3, 5)  # 3 years × 5 sims
    assert R.percentile_bands(sims) == pytest.approx({
        10: [1.4, 6.4, 11.4],
        25: [2.0, 7.0, 12.0],
        50: [3.0, 8.0, 13.0],
        75: [4.0, 9.0, 14.0],
        90: [4.6, 9.6, 14.6],
    }, rel=1e-9)


def test_percentile_bands_empty():
    assert R.percentile_bands(np.zeros((0, 0))) == {p: [] for p in (10, 25, 50, 75, 90)}


# ── glide_return_params / glide_return_pools (injected market data) ─────────────

def test_glide_return_params_literals():
    """Two-leg glide params from the synthetic market: equity weight + leg CAGRs."""
    got = R.glide_return_params(HOLDINGS, FAKE_INFLATION, fake_nominal)
    # US 0.6 + Intl 0.2 + Other 0.05 (ZZZTOP has short history → VTSAX proxy)
    assert got["w_equity0"] == pytest.approx(0.85, rel=1e-9)
    assert got["r_equity"] == pytest.approx(0.05390076134518762, rel=1e-9)
    assert got["r_fixed"] == pytest.approx(0.009801991277713862, rel=1e-9)


def test_glide_return_params_none_without_portfolio():
    assert R.glide_return_params([], FAKE_INFLATION, fake_nominal) is None
    assert R.glide_return_params([{"symbol": "VTSAX", "total_value": 0.0}],
                                 FAKE_INFLATION, fake_nominal) is None


def test_glide_return_pools_golden():
    """Aligned equity/fixed real-return pools on the common (sorted) year index."""
    pools = R.glide_return_pools(HOLDINGS, FAKE_INFLATION, fake_nominal)
    # Equity leg spans 1998–2023 (VTSAX), bonds 2002–2023 → 22 common years.
    assert pools["years"] == list(range(2002, 2024))
    actual = pd.DataFrame(
        {"equity_pool": pools["equity_pool"], "fixed_pool": pools["fixed_pool"]},
        index=pd.Index(pools["years"], name="year"))
    golden = pd.read_csv(GOLDENS_DIR / "returns_glide_pools.csv", index_col=0)
    golden = golden.astype("float64")
    golden.index = golden.index.astype(actual.index.dtype)
    golden.index.name = actual.index.name
    pd.testing.assert_frame_equal(actual, golden, rtol=1e-9)


# ── glide_cagr_projection ────────────────────────────────────────────────────────

def test_glide_cagr_projection_literals():
    """Geometric central path with a ±z·SE band that tightens with horizon."""
    got = R.glide_cagr_projection(EQ_POOL, FX_POOL, [0.9, 0.7, 0.5],
                                  r_equity=0.05, r_fixed=0.02)
    assert got["expected"] == pytest.approx(
        [0.04700000000000001, 0.04399568964627436, 0.04098847249482913], rel=1e-9)
    assert got[25] == pytest.approx(
        [-0.0025679949376611333, 0.012952564488681088, 0.018632757014591567], rel=1e-9)
    assert got[75] == pytest.approx(
        [0.09656799493766115, 0.07503881480386762, 0.06334418797506669], rel=1e-9)


def test_glide_cagr_projection_empty():
    """Empty pool or empty weights yield empty per-key lists."""
    assert R.glide_cagr_projection([], FX_POOL, [0.9], 0.05, 0.02) == \
        {"expected": [], 25: [], 75: []}
    assert R.glide_cagr_projection(EQ_POOL, FX_POOL, [], 0.05, 0.02) == \
        {"expected": [], 25: [], 75: []}


# ── glide_annual_return_bands ────────────────────────────────────────────────────

def test_glide_annual_return_bands_literals():
    """Percentiles of the blended historical distribution per glide weight."""
    got = R.glide_annual_return_bands(EQ_POOL, FX_POOL, [1.0, 0.5])
    assert got == pytest.approx({
        10: [-0.056, -0.014],
        25: [-0.02, -0.005],
        50: [0.06, 0.04],
        75: [0.1, 0.065],
        90: [0.13, 0.071],
    }, rel=1e-9)


def test_glide_annual_return_bands_nan_pairs_dropped():
    """A year with NaN in either leg is dropped from the blend pool."""
    got = R.glide_annual_return_bands([0.10, float("nan")], [0.02, 0.03], [1.0])
    assert got == pytest.approx({p: [0.1] for p in (10, 25, 50, 75, 90)}, rel=1e-9)


def test_glide_annual_return_bands_empty_pool():
    assert R.glide_annual_return_bands([], [], [1.0]) == \
        {p: [] for p in (10, 25, 50, 75, 90)}


# ── bootstrap_projection — invariants only (never exact golden paths) ───────────

def test_bootstrap_projection_shape_and_determinism():
    """(horizon, N_SIMS) output, identical across calls under the fixed seed."""
    a = R.bootstrap_projection(BOOT_POOL, 10_000.0, [1_200.0] * 10, 10)
    b = R.bootstrap_projection(BOOT_POOL, 10_000.0, [1_200.0] * 10, 10)
    assert a.shape == (10, R.N_SIMS)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()
    # A different seed produces a different path set.
    c = R.bootstrap_projection(BOOT_POOL, 10_000.0, [1_200.0] * 10, 10, seed=7)
    assert not np.array_equal(a, c)


def test_bootstrap_projection_percentiles_monotonic():
    """p10 ≤ p25 ≤ p50 ≤ p75 ≤ p90 for every simulated year."""
    sims = R.bootstrap_projection(BOOT_POOL, 10_000.0, [1_200.0] * 10, 10)
    bands = R.percentile_bands(sims)
    for lo, hi in zip((10, 25, 50, 75), (25, 50, 75, 90)):
        assert all(a <= b + 1e-12 for a, b in zip(bands[lo], bands[hi]))


def test_bootstrap_projection_first_year_bounded_by_pool():
    """Year-1 values stay inside the extremes the pool allows."""
    sims = R.bootstrap_projection(BOOT_POOL, 10_000.0, [1_200.0] * 10, 10)
    lo = 10_000.0 * (1 + min(BOOT_POOL)) + 1_200.0
    hi = 10_000.0 * (1 + max(BOOT_POOL)) + 1_200.0
    assert sims[0].min() >= lo - 1e-9
    assert sims[0].max() <= hi + 1e-9


def test_bootstrap_projection_zero_return_pool_exact():
    """A degenerate {0.0} pool makes the recurrence exactly deterministic."""
    sims = R.bootstrap_projection([0.0], 1_000.0, [100.0, 100.0, 100.0], 3)
    assert np.array_equal(sims, np.array([[1_100.0], [1_200.0], [1_300.0]])
                          .repeat(R.N_SIMS, axis=1))


def test_bootstrap_projection_contributions_padded_with_zero():
    """A short contributions list is zero-padded to the horizon."""
    sims = R.bootstrap_projection([0.0], 1_000.0, [100.0], 3)
    assert np.array_equal(sims, np.array([[1_100.0], [1_100.0], [1_100.0]])
                          .repeat(R.N_SIMS, axis=1))


@pytest.mark.parametrize("pool,horizon,expected_shape", [
    ([], 5, (5, R.N_SIMS)),                 # empty pool → zero matrix
    ([np.nan, np.nan], 4, (4, R.N_SIMS)),   # all-NaN pool → zero matrix
    (BOOT_POOL, 0, (0, R.N_SIMS)),          # zero horizon
    (BOOT_POOL, -3, (0, R.N_SIMS)),         # negative horizon clamps to 0 rows
])
def test_bootstrap_projection_edge_cases(pool, horizon, expected_shape):
    sims = R.bootstrap_projection(pool, 10_000.0, [0.0], horizon)
    assert sims.shape == expected_shape
    assert (sims == 0.0).all()


# ── bootstrap_glide_projection — invariants only ─────────────────────────────────

_GLIDE_W = list(np.linspace(1.0, 0.4, 10))


def test_bootstrap_glide_shape_determinism_monotonic():
    a = R.bootstrap_glide_projection(BOOT_POOL, FX_POOL * 3, _GLIDE_W,
                                     10_000.0, [1_200.0] * 10, 10)
    b = R.bootstrap_glide_projection(BOOT_POOL, FX_POOL * 3, _GLIDE_W,
                                     10_000.0, [1_200.0] * 10, 10)
    assert a.shape == (10, R.N_SIMS)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()
    bands = R.percentile_bands(a)
    for lo, hi in zip((10, 25, 50, 75), (25, 50, 75, 90)):
        assert all(x <= y + 1e-12 for x, y in zip(bands[lo], bands[hi]))


def test_bootstrap_glide_single_pair_exact_and_weight_padding():
    """Single-year pools collapse the sampling; weights pad with their last value."""
    sims = R.bootstrap_glide_projection(
        [0.10], [0.0], [1.0, 0.5], 1_000.0, [], 4)
    # r = [0.10, 0.05, 0.05, 0.05] → 1100, 1155, 1212.75, 1273.3875 in every sim
    expected = np.array([[1_100.0], [1_155.0], [1_212.75], [1_273.3875]])
    assert sims.shape == (4, R.N_SIMS)
    np.testing.assert_allclose(sims, expected.repeat(R.N_SIMS, axis=1), rtol=1e-9)


def test_bootstrap_glide_nan_pairs_dropped():
    """A (sim-year) pair with NaN in either leg never enters the sample pool."""
    sims = R.bootstrap_glide_projection(
        [0.10, np.nan], [0.0, 0.0], [1.0], 1_000.0, [], 2)
    expected = np.array([[1_100.0], [1_210.0]])
    np.testing.assert_allclose(sims, expected.repeat(R.N_SIMS, axis=1), rtol=1e-9)


def test_bootstrap_glide_empty_weights_default_to_all_equity():
    """No weights → w=1 everywhere (pure equity leg)."""
    sims = R.bootstrap_glide_projection([0.10], [0.0], [], 1_000.0, [], 2)
    expected = np.array([[1_100.0], [1_210.0]])
    np.testing.assert_allclose(sims, expected.repeat(R.N_SIMS, axis=1), rtol=1e-9)


@pytest.mark.parametrize("eq,fx,horizon,expected_shape", [
    ([], [0.02], 5, (5, R.N_SIMS)),                    # one empty leg
    ([np.nan], [0.02], 3, (3, R.N_SIMS)),              # NaN-only after pairing
    ([0.05], [0.02], 0, (0, R.N_SIMS)),                # zero horizon
])
def test_bootstrap_glide_edge_cases(eq, fx, horizon, expected_shape):
    sims = R.bootstrap_glide_projection(eq, fx, [1.0], 1_000.0, [], horizon)
    assert sims.shape == expected_shape
    assert (sims == 0.0).all()


# ── Golden generation ────────────────────────────────────────────────────────────
# Run ONCE per characterization epoch: `uv run python -m tests.test_returns`.
# A refactor may NEVER regenerate these goldens — they pin pre-refactor behavior.

def _generate_goldens() -> None:
    out = GOLDENS_DIR

    wa = R.weighted_average_real_returns(
        _hand_class_series(), {"US Equity": 0.75, "Bonds": 0.25})
    wa.to_frame("value").to_csv(out / "returns_weighted_avg.csv",
                                float_format="%.17g", index_label="year")
    print("wrote returns_weighted_avg.csv")

    pools = R.glide_return_pools(HOLDINGS, FAKE_INFLATION, fake_nominal)
    frame = pd.DataFrame(
        {"equity_pool": pools["equity_pool"], "fixed_pool": pools["fixed_pool"]},
        index=pd.Index(pools["years"], name="year"))
    frame.to_csv(out / "returns_glide_pools.csv", float_format="%.17g")
    print("wrote returns_glide_pools.csv")


if __name__ == "__main__":
    _generate_goldens()
