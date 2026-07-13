"""Characterization tests for core/domain/trends.py (Trends page data helpers).

Pins the CURRENT behavior of the six public aggregation functions against a
single synthetic transactions DataFrame that mirrors the shape of the
`transaction-data-store`: the fixture is round-tripped through
``to_json(date_format='iso', orient='split')`` / ``read_json`` exactly as
`components/sidebar.py` writes it and `pages/trends.py` reads it, so dtypes
(tz-naive datetime64 `date`, float64 `amount`) match real usage.

Frequencies match the page's smoothing options ("YE", "ME", "W", "D") and the
date window is passed as ISO strings, as the DatePickerRange does.

Goldens live in tests/goldens/trends_*.csv — generated ONCE from current main
via ``uv run python -m tests.test_trends``. A refactor may NEVER regenerate
them; tests only read.
"""

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

import core.domain.trends as T  # noqa: E402

GOLDENS_DIR = Path(__file__).parent / "goldens"

# ── Fixture: one synthetic transactions frame ───────────────────────────────────
# 3 owners x 18 months (2024-01 .. 2025-06) x all five CSP labels, several
# categories per label, paycheck-keyword income rows, plus one refund row.

OWNERS = ("erik", "partner", "household")
OWNER_MULT = {"erik": 1.0, "partner": 0.8, "household": 1.5}
N_MONTHS = 18  # 2024-01 .. 2025-06

# (csp_label, csp, category_name, merchant, day-of-month, base signed amount)
_ROW_SPEC = [
    ("income", "paychecks", "Paycheck", "Acme Corp Payroll", 5, 5000.0),
    ("income", "paychecks", "Paycheck", "Acme Corp Payroll", 20, 5000.0),
    ("income", "other_income", "Freelance Income", "Upwork", 12, 500.0),
    ("fixed", "housing", "Rent", "Skyline Apartments", 1, -1800.0),
    ("fixed", "food", "Groceries", "Safeway", 8, -400.0),
    ("fixed", "utilities", "Utilities", "City Power", 15, -120.0),
    ("investments", "savings", "Transfer to Vanguard", "Vanguard", 2, -1000.0),
    ("sinking", "vacation", "Vacation", "Expedia", 18, -200.0),
    ("sinking", "gifts", "Gifts", "Etsy", 22, -50.0),
    ("guilt-free", "dining", "Restaurants", "Bistro", 9, -150.0),
    ("guilt-free", "dining", "Coffee Shops", "Blue Bottle", 3, -60.0),
    ("guilt-free", "dining", "Bars", "Dive Bar", 27, -40.0),
    ("guilt-free", "fun", "Apps", "App Store", 6, -15.0),
    ("guilt-free", "transport", "Taxi & Ride Shares", "Lyft", 11, -25.0),
    ("guilt-free", "fun", "Entertainment", "Cinema", 14, -80.0),
    ("guilt-free", "shopping", "Shopping", "Target", 21, -120.0),
]

START, END = "2024-01-01", "2025-06-30"

# Fixed siderail items, copied from pages/trends.py::SNAPSHOT_ITEMS ("Public
# Transit" has no fixture rows, pinning the zero/flat/NaN path).
SNAPSHOT_ITEMS = [
    ("Coffee Shops", ["Coffee Shops"]),
    ("Taxi & Ride Shares", ["Taxi & Ride Shares"]),
    ("Apps", ["Apps"]),
    ("Public Transit", ["Public Transit"]),
    ("Total Food Cost", [c for cats in T.DEFAULT_FOOD_DINING_PARTS.values() for c in cats]),
]


def _build_transactions() -> pd.DataFrame:
    """Deterministic synthetic frame, JSON round-tripped like the real store."""
    months = pd.period_range("2024-01", periods=N_MONTHS, freq="M")
    rows = []
    for m, period in enumerate(months):
        for owner in OWNERS:
            mult = OWNER_MULT[owner]
            for label, csp, cat, merchant, day, base in _ROW_SPEC:
                rows.append({
                    "date": pd.Timestamp(period.year, period.month, day),
                    "account_owner": owner,
                    "amount": base * mult * (1 + 0.01 * m),
                    "csp": csp,
                    "csp_label": label,
                    "category_name": cat,
                    "merchant": merchant,
                })
    # One refund: a credit into an expense category that exceeds that month's
    # spend, so the Apps net for erik in 2025-02 is positive (reads negative
    # after negation — the "net-credit month stays negative" convention).
    rows.append({
        "date": pd.Timestamp(2025, 2, 16), "account_owner": "erik",
        "amount": 50.0, "csp": "fun", "csp_label": "guilt-free",
        "category_name": "Apps", "merchant": "App Store refund",
    })
    df = pd.DataFrame(rows)
    # Mirror the store round-trip (components/sidebar.py line 366).
    payload = df.to_json(date_format="iso", orient="split")
    return pd.read_json(StringIO(payload), orient="split")


@pytest.fixture(scope="module")
def txns() -> pd.DataFrame:
    return _build_transactions()


# ── Golden helpers ───────────────────────────────────────────────────────────────

def _assert_time_frame_matches(actual: pd.DataFrame, name: str) -> None:
    """Compare a period-indexed frame against its golden, strict on values."""
    golden = pd.read_csv(GOLDENS_DIR / name, index_col=0).astype("float64")
    golden.index = pd.to_datetime(golden.index, utc=actual.index.tz is not None)
    golden.index = pd.DatetimeIndex(golden.index).as_unit("ns")
    golden.index.name = actual.index.name
    golden.columns.name = actual.columns.name
    normalized = actual.copy()
    normalized.index = pd.DatetimeIndex(normalized.index).as_unit("ns")
    pd.testing.assert_frame_equal(normalized, golden, rtol=1e-9, check_freq=False)


def _assert_series_matches(actual: pd.Series, name: str, col: str = "amount") -> None:
    _assert_time_frame_matches(actual.to_frame(col), name)


# ── income_by_period ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("freq", ["ME", "YE"])
def test_income_by_period_golden(txns, freq):
    """Sum of income rows per period, no negate/abs, across all owners."""
    out = T.income_by_period(txns, freq)
    _assert_series_matches(out, f"trends_income_by_period_{freq}.csv")


def test_income_by_period_empty_when_no_income(txns):
    """A frame with no income rows yields an empty float Series."""
    out = T.income_by_period(txns[txns["csp_label"] != "income"], "ME")
    assert out.empty and out.dtype == float


# ── csp_shares_over_time (Q1) ────────────────────────────────────────────────────

@pytest.mark.parametrize("freq", ["ME", "YE", "W"])
def test_csp_shares_over_time_golden(txns, freq):
    """CSP bucket pivot for one owner at each page smoothing with a golden."""
    pivot, _ = T.csp_shares_over_time(txns, "erik", START, END, freq)
    _assert_time_frame_matches(pivot, f"trends_csp_shares_{freq}.csv")


def test_csp_shares_income_series_golden(txns):
    """The income leg of the (pivot, income) tuple is Σ income per period."""
    _, income = T.csp_shares_over_time(txns, "erik", START, END, "ME")
    _assert_series_matches(income, "trends_csp_shares_ME_income.csv")


def test_csp_shares_as_percent_golden(txns):
    """as_percent divides each bucket by that period's income."""
    pivot, income = T.csp_shares_over_time(
        txns, "erik", START, END, "ME", as_percent=True)
    _assert_time_frame_matches(pivot, "trends_csp_shares_ME_pct.csv")
    # income stays in dollars for the 100% reference line
    _assert_series_matches(income, "trends_csp_shares_ME_income.csv")


def test_csp_shares_column_order(txns):
    """Columns follow SPENDING_LABEL_ORDER, never include income."""
    pivot, _ = T.csp_shares_over_time(txns, "erik", START, END, "ME")
    assert list(pivot.columns) == list(T.SPENDING_LABEL_ORDER)


def test_csp_shares_empty_for_unknown_owner(txns):
    """An owner with no transactions yields an empty pivot and empty income."""
    pivot, income = T.csp_shares_over_time(txns, "nobody", START, END, "ME")
    assert pivot.empty
    assert income.empty


def test_spend_totals_invariant_across_freqs(txns):
    """Total spend per bucket is the same under every page smoothing option."""
    totals = {}
    for freq in ("ME", "YE", "W", "D"):
        pivot, _ = T.csp_shares_over_time(txns, "erik", START, END, freq)
        totals[freq] = pivot.sum()
    for freq in ("YE", "W", "D"):
        pd.testing.assert_series_equal(totals[freq], totals["ME"], rtol=1e-9)


# ── composition_over_time (Q4) ───────────────────────────────────────────────────

def test_composition_over_time_golden(txns):
    """Default guilt-free breakdown: top-5 categories + an Other rollup."""
    out = T.composition_over_time(txns, "erik", START, END, "ME")
    assert T.OTHER_LABEL in out.columns  # 7 guilt-free categories > top_n=5
    _assert_time_frame_matches(out, "trends_composition_ME.csv")


def test_composition_top2_golden(txns):
    """top_n=2 folds five categories into Other."""
    out = T.composition_over_time(txns, "erik", START, END, "ME", top_n=2)
    _assert_time_frame_matches(out, "trends_composition_ME_top2.csv")


def test_composition_group_col_csp(txns):
    """group_col='csp' matches the drill-down grouping (coarser keys)."""
    out = T.composition_over_time(txns, "erik", START, END, "ME", group_col="csp")
    assert set(out.columns) == {"dining", "fun", "transport", "shopping"}


def test_composition_empty_when_label_absent(txns):
    """A bucket with no spend in scope yields an empty frame."""
    out = T.composition_over_time(txns, "erik", START, END, "ME",
                                  csp_label="no-such-bucket")
    assert out.empty


def test_refund_month_reads_negative(txns):
    """2025-02 Apps credit (+50) exceeds spend, so the negated value is negative."""
    out = T.composition_over_time(txns, "erik", START, END, "ME", top_n=10)
    feb = out.loc["2025-02-28", "Apps"]
    # base spend -15 * (1 + 0.01*13) = -16.95; refund +50 → net +33.05 → negated
    assert feb == pytest.approx(-33.05, rel=1e-9)


# ── composite_over_time (Q2) ─────────────────────────────────────────────────────

@pytest.mark.parametrize("freq", ["ME", "YE"])
def test_composite_over_time_golden(txns, freq):
    """Default Food & Dining composite: Groceries + broken-out dining + Combined."""
    out = T.composite_over_time(txns, "erik", START, END, freq)
    # Dining out is broken out; only the fixture's present subcats appear, in
    # DEFAULT_FOOD_DINING_PARTS listed order.
    assert list(out.columns) == [
        "Groceries", "Restaurants", "Bars", "Coffee Shops", T.COMBINED_LABEL]
    _assert_time_frame_matches(out, f"trends_composite_{freq}.csv")


def test_composite_custom_parts_no_breakout(txns):
    """A custom single part with no breakout gives one summed column + total."""
    out = T.composite_over_time(
        txns, "erik", START, END, "ME",
        parts={"Coffee": ["Coffee Shops"]}, breakout=frozenset())
    assert list(out.columns) == ["Coffee", T.COMBINED_LABEL]
    pd.testing.assert_series_equal(
        out["Coffee"], out[T.COMBINED_LABEL], rtol=1e-9, check_names=False)


def test_composite_empty_when_no_categories_match(txns):
    out = T.composite_over_time(
        txns, "erik", START, END, "ME", parts={"X": ["No Such Category"]})
    assert out.empty


# ── income_split_over_time (Q3) ──────────────────────────────────────────────────

@pytest.mark.parametrize("freq", ["ME", "YE"])
def test_income_split_over_time_golden(txns, freq):
    """Paycheck-keyword split: Paychecks / Other sources / Total income."""
    out = T.income_split_over_time(txns, "erik", START, END, freq)
    assert list(out.columns) == ["Paychecks", "Other sources", T.TOTAL_INCOME_LABEL]
    _assert_time_frame_matches(out, f"trends_income_split_{freq}.csv")


def test_income_split_explicit_categories(txns):
    """Explicit paycheck_categories overrides the keyword heuristic."""
    out = T.income_split_over_time(
        txns, "erik", START, END, "ME",
        paycheck_categories=["Freelance Income"])
    default = T.income_split_over_time(txns, "erik", START, END, "ME")
    # The split swaps but the total is unchanged.
    pd.testing.assert_series_equal(
        out["Paychecks"], default["Other sources"], rtol=1e-9, check_names=False)
    pd.testing.assert_series_equal(
        out[T.TOTAL_INCOME_LABEL], default[T.TOTAL_INCOME_LABEL], rtol=1e-9)


def test_income_split_empty_when_no_income(txns):
    out = T.income_split_over_time(
        txns[txns["csp_label"] != "income"], "erik", START, END, "ME")
    assert out.empty


# ── category_snapshot (siderail) ─────────────────────────────────────────────────
# The window anchors on min(as_of, latest transaction) — NOT wall-clock now —
# so a fixed as_of makes it fully deterministic and golden-able.

def _snapshot(txns, baseline):
    return T.category_snapshot(txns, "erik", items=SNAPSHOT_ITEMS,
                               as_of=END, baseline=baseline)


def _assert_snapshot_matches(actual: pd.DataFrame, name: str) -> None:
    golden = pd.read_csv(GOLDENS_DIR / name, index_col=0)
    golden.index.name = actual.index.name
    pd.testing.assert_frame_equal(actual, golden, rtol=1e-9)


@pytest.mark.parametrize("baseline", ["trailing_12mo", "last_year"])
def test_category_snapshot_golden(txns, baseline):
    """Rolling 30-day spend vs baseline for the fixed rail items."""
    out = _snapshot(txns, baseline)
    assert list(out.columns) == ["current", "baseline", "delta", "pct_change", "direction"]
    suffix = "trailing" if baseline == "trailing_12mo" else "lastyear"
    _assert_snapshot_matches(out, f"trends_snapshot_{suffix}.csv")


def test_category_snapshot_budget_falls_back_to_trailing(txns):
    """baseline='budget' currently produces the trailing_12mo numbers."""
    pd.testing.assert_frame_equal(
        _snapshot(txns, "budget"), _snapshot(txns, "trailing_12mo"), rtol=1e-9)


def test_category_snapshot_absent_item_is_flat_nan(txns):
    """An item with no matching categories: 0 spend, flat, NaN pct_change."""
    out = _snapshot(txns, "trailing_12mo")
    row = out.loc["Public Transit"]
    assert row["current"] == 0.0 and row["baseline"] == 0.0
    assert row["direction"] == "flat"
    assert pd.isna(row["pct_change"])


def test_category_snapshot_clamps_as_of_to_latest_txn(txns):
    """An as_of past the last synced transaction anchors on the latest txn."""
    far_future = T.category_snapshot(
        txns, "erik", items=SNAPSHOT_ITEMS, as_of="2030-01-01")
    default = T.category_snapshot(txns, "erik", items=SNAPSHOT_ITEMS)
    pd.testing.assert_frame_equal(far_future, default, rtol=1e-9)


def test_category_snapshot_empty_for_unknown_owner(txns):
    out = T.category_snapshot(txns, "nobody", items=SNAPSHOT_ITEMS)
    assert out.empty
    assert list(out.columns) == ["current", "baseline", "delta", "pct_change", "direction"]
    assert out.index.name == "label"


# ── Golden generation ────────────────────────────────────────────────────────────
# Run ONCE per characterization epoch: `uv run python -m tests.test_trends`.
# A refactor may NEVER regenerate these goldens — they pin pre-refactor behavior.

def _generate_goldens() -> None:
    df = _build_transactions()
    out = GOLDENS_DIR

    def write(frame: pd.DataFrame, name: str) -> None:
        frame.to_csv(out / name, float_format="%.17g")
        print(f"wrote {name}")

    for freq in ("ME", "YE"):
        write(T.income_by_period(df, freq).to_frame("amount"),
              f"trends_income_by_period_{freq}.csv")
    for freq in ("ME", "YE", "W"):
        pivot, _ = T.csp_shares_over_time(df, "erik", START, END, freq)
        write(pivot, f"trends_csp_shares_{freq}.csv")
    _, income = T.csp_shares_over_time(df, "erik", START, END, "ME")
    write(income.to_frame("amount"), "trends_csp_shares_ME_income.csv")
    pct, _ = T.csp_shares_over_time(df, "erik", START, END, "ME", as_percent=True)
    write(pct, "trends_csp_shares_ME_pct.csv")
    write(T.composition_over_time(df, "erik", START, END, "ME"),
          "trends_composition_ME.csv")
    write(T.composition_over_time(df, "erik", START, END, "ME", top_n=2),
          "trends_composition_ME_top2.csv")
    for freq in ("ME", "YE"):
        write(T.composite_over_time(df, "erik", START, END, freq),
              f"trends_composite_{freq}.csv")
        write(T.income_split_over_time(df, "erik", START, END, freq),
              f"trends_income_split_{freq}.csv")
    write(T.category_snapshot(df, "erik", items=SNAPSHOT_ITEMS,
                              as_of=END, baseline="trailing_12mo"),
          "trends_snapshot_trailing.csv")
    write(T.category_snapshot(df, "erik", items=SNAPSHOT_ITEMS,
                              as_of=END, baseline="last_year"),
          "trends_snapshot_lastyear.csv")


if __name__ == "__main__":
    _generate_goldens()
