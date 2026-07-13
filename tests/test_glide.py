"""Characterization tests for core/domain/glide.py (Coast rate-of-return glide).

Pins the CURRENT behavior of main (2026-07 characterization epoch) as the safety
net for the core/domain/ restructuring. All literals were produced by running the
current code. Everything is scalar/small-dict — no goldens needed.
"""

from __future__ import annotations


import pytest

from core.domain import glide

approx = lambda v: pytest.approx(v, rel=1e-9)  # noqa: E731

# Standard fixture: glide from 90% equity at 40 down to 50% at 45,
# with 7% equity / 2% fixed real returns.
W0, W_TERM = 0.9, 0.5
R_EQ, R_FX = 0.07, 0.02
RATES = glide.accumulation_rates(W0, W_TERM, R_EQ, R_FX, 40, 45)


def test_strategy_constants():
    """Preset terminal weights and default strategy."""
    assert glide.STRATEGY_TERMINAL_EQUITY == {
        "aggressive": 0.70, "moderate": 0.50, "conservative": 0.30}
    assert glide.DEFAULT_STRATEGY == "moderate"
    assert set(glide.STRATEGY_LABELS) == set(glide.STRATEGY_TERMINAL_EQUITY)


# ── equity_weight_path ────────────────────────────────────────────────────────

def test_equity_weight_path_linear_glide():
    """Linear from w0 toward w_term; terminal weight NOT reached (year before ret)."""
    path = glide.equity_weight_path(0.9, 0.5, 40, 45)
    assert list(path) == [40, 41, 42, 43, 44]
    assert path[40] == approx(0.9)
    assert path[41] == approx(0.8200000000000001)
    assert path[42] == approx(0.74)
    assert path[43] == approx(0.66)
    assert path[44] == approx(0.58)  # last working year — one step short of 0.5


def test_equity_weight_path_downward_only_clamp():
    """w0 at or below the terminal → hold flat at w0 (never buy equity)."""
    assert glide.equity_weight_path(0.4, 0.5, 40, 43) == {40: 0.4, 41: 0.4, 42: 0.4}
    assert glide.equity_weight_path(0.5, 0.5, 40, 43) == {40: 0.5, 41: 0.5, 42: 0.5}


@pytest.mark.parametrize("current, ret_age", [(45, 45), (50, 45)])
def test_equity_weight_path_no_window(current, ret_age):
    """start == end (or past it) → empty path."""
    assert glide.equity_weight_path(0.9, 0.5, current, ret_age) == {}


@pytest.mark.parametrize("strategy, w_term", sorted(glide.STRATEGY_TERMINAL_EQUITY.items()))
def test_equity_weight_path_per_strategy_endpoints(strategy, w_term):
    """From 90% equity, every preset glides down; first step is w0 exactly."""
    path = glide.equity_weight_path(0.9, w_term, 40, 50)
    assert path[40] == approx(0.9)
    # Linear: age 49 is one step short of terminal → w0 + (w_term−w0)·9/10.
    assert path[49] == approx(0.9 + (w_term - 0.9) * 0.9)


# ── accumulation_rates ────────────────────────────────────────────────────────

def test_accumulation_rates_blend():
    """r(age) = w·r_eq + (1−w)·r_fx for each glide year."""
    assert list(RATES) == [40, 41, 42, 43, 44]
    assert RATES[40] == approx(0.06500000000000002)
    assert RATES[41] == approx(0.061000000000000006)
    assert RATES[42] == approx(0.05700000000000001)
    assert RATES[43] == approx(0.053000000000000005)
    assert RATES[44] == approx(0.049)


def test_accumulation_rates_empty_window():
    """No accumulation years → empty rates map."""
    assert glide.accumulation_rates(0.9, 0.5, R_EQ, R_FX, 45, 45) == {}


# ── retirement_rate ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("strategy, expected", [
    ("aggressive", 0.055),
    ("moderate", 0.045000000000000005),
    ("conservative", 0.035),
])
def test_retirement_rate_per_strategy(strategy, expected):
    """Drawdown rate is the blend at the terminal (held) allocation."""
    w_term = glide.STRATEGY_TERMINAL_EQUITY[strategy]
    assert glide.retirement_rate(w_term, R_EQ, R_FX) == approx(expected)


def test_retirement_rate_all_equity_and_all_fixed():
    """Degenerate weights collapse to the single-leg rate."""
    assert glide.retirement_rate(1.0, R_EQ, R_FX) == approx(0.07)
    assert glide.retirement_rate(0.0, R_EQ, R_FX) == approx(0.02)


# ── effective_cagr ────────────────────────────────────────────────────────────

def test_effective_cagr():
    """Geometric mean of the glide rates; empty → 0; single passes through."""
    assert glide.effective_cagr(RATES) == approx(0.0569848625591578)
    assert glide.effective_cagr({}) == 0.0
    assert glide.effective_cagr({40: 0.05}) == approx(0.050000000000000044)


# ── cumulative_growth ─────────────────────────────────────────────────────────

def test_cumulative_growth():
    """∏(1+r) over [start, end); missing ages contribute factor 1.0."""
    assert glide.cumulative_growth(RATES, 40, 45) == approx(1.3193008382039848)
    assert glide.cumulative_growth(RATES, 42, 45) == approx(1.1675590289999997)
    # Ages 45–49 are missing from RATES → treated as 0% return.
    assert glide.cumulative_growth(RATES, 40, 50) == approx(1.3193008382039848)
    assert glide.cumulative_growth(RATES, 45, 45) == 1.0    # empty range
    assert glide.cumulative_growth({}, 40, 45) == 1.0       # empty rates


# ── required_monthly_glide ────────────────────────────────────────────────────

def test_required_monthly_glide():
    """Glide-aware inverse of the working recurrence, ÷12."""
    monthly = glide.required_monthly_glide(100_000.0, 40, 45, 300_000.0, RATES)
    assert monthly == approx(2_519.0624403502316)


def test_required_monthly_glide_recurrence_consistency():
    """Applying the recurrence with the returned C reaches the target exactly."""
    monthly = glide.required_monthly_glide(100_000.0, 40, 45, 300_000.0, RATES)
    balance = 100_000.0
    for k in range(5):
        balance = balance * (1 + RATES[40 + k]) + monthly * 12
    assert balance == approx(300_000.0)


def test_required_monthly_glide_no_window():
    """coast_age ≤ current_age → 0."""
    assert glide.required_monthly_glide(100_000.0, 45, 45, 300_000.0, RATES) == 0.0
    assert glide.required_monthly_glide(100_000.0, 50, 45, 300_000.0, RATES) == 0.0


def test_required_monthly_glide_already_coasting():
    """Start value already grows past the target → clamped to 0."""
    assert glide.required_monthly_glide(1_000_000.0, 40, 45, 300_000.0, RATES) == 0.0


@pytest.mark.parametrize("strategy", sorted(glide.STRATEGY_TERMINAL_EQUITY))
def test_required_monthly_glide_per_strategy_consistency(strategy):
    """For every preset, the recurrence with the returned C lands on the target."""
    w_term = glide.STRATEGY_TERMINAL_EQUITY[strategy]
    rates = glide.accumulation_rates(0.9, w_term, R_EQ, R_FX, 40, 55)
    monthly = glide.required_monthly_glide(50_000.0, 40, 55, 800_000.0, rates)
    balance = 50_000.0
    for k in range(15):
        balance = balance * (1 + rates[40 + k]) + monthly * 12
    assert balance == approx(800_000.0)
