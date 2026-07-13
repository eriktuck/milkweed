"""Characterization tests for the budget logic in core/utils/functions.py.

Pins the CURRENT behavior of: read_budget (pure dict -> frame, no Firestore),
get_active_csp_plan, get_monthly_loading, generate_budget_from_csp,
calc_proportions, build_budget_report, order_budget.

All data is synthetic; no Firebase, no network. Goldens live in
tests/goldens/ with the ``budget_`` prefix and are generated ONCE (see
_generate_goldens at the bottom) — a refactor may NEVER regenerate them.
"""

from datetime import datetime as dt

import pandas as pd
import pytest

from core.utils.functions import (
    read_budget,
    get_monthly_loading,
    get_active_csp_plan,
    generate_budget_from_csp,
    calc_proportions,
    build_budget_report,
    order_budget,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic budget dict / csp plans / processed transactions
# ---------------------------------------------------------------------------


def make_budget_dict():
    """Full 2024 budget + Jan 2025; categories are csp keys."""
    months_2024 = {}
    for m in range(1, 13):
        months_2024[str(m)] = {
            "fixed": 2000.0,
            "groceries": 400.0 + (50.0 if m == 12 else 0.0),  # December bump
            "savings": 500.0,
            "guilt_free": 300.0,
            "paycheck": 6000.0,
            "vacation": 2400.0 if m == 7 else 0.0,  # loaded entirely in July
        }
    return {
        "2024": months_2024,
        "2025": {
            "1": {
                "fixed": 2100.0,
                "groceries": 420.0,
                "savings": 500.0,
                "guilt_free": 310.0,
                "paycheck": 6100.0,
                "vacation": 200.0,
            },
        },
    }


def make_config():
    """Config dict shaped like SessionData.data: users keyed by use-case value."""
    return {
        "users": {
            "erik": {
                "name": "erik",
                "budget": make_budget_dict(),
                "cat_order": [
                    "Income", "Total Income", "paycheck",
                    "Fixed Costs", "fixed", "groceries",
                    "Investments", "savings",
                    "Sinking", "vacation",
                    "Guilt Free", "guilt_free",
                    "Total Spending",
                ],
            }
        }
    }


CSP_PLANS = {
    "2024-01-01": {"fixed": 1900.0},
    "2024-07-15": {"fixed": 2000.0},
    "2025-01-01": {"fixed": 2100.0},
}


def make_transactions():
    """Small processed-transactions frame (mirrors the txn_ pipeline output
    shape: date/amount/csp/csp_label/account_owner), spending negative."""
    rows = [
        ("t1", "2025-01-02", -1500.0, "fixed", "fixed"),
        ("t2", "2025-01-03", 4000.0, "paycheck", "income"),
        ("t3", "2025-01-05", -120.5, "groceries", "fixed"),
        ("t4", "2025-01-06", -500.0, "savings", "investments"),
        ("t6", "2025-01-08", -45.25, "guilt_free", "guilt-free"),
        ("t12", "2025-01-20", -2500.0, "fixed", "fixed"),   # blows the fixed budget
        ("t13", "2025-02-02", -80.0, "groceries", "fixed"),  # outside report window
    ]
    df = pd.DataFrame(rows, columns=["id", "date", "amount", "csp", "csp_label"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["account_owner"] = "erik"
    return df


TRAILING_2024 = [(2024, m) for m in range(1, 13)]


# ---------------------------------------------------------------------------
# read_budget  (pure dict transform — does NOT hit Firestore)
# ---------------------------------------------------------------------------

def test_read_budget_builds_multiindex_frame():
    """read_budget pivots {year: {month: {cat: val}}} into (year, month) columns."""
    budget = read_budget(make_config(), "erik")
    assert isinstance(budget.columns, pd.MultiIndex)
    assert list(budget.columns) == [(2024, m) for m in range(1, 13)] + [(2025, 1)]
    assert list(budget.index) == [
        "fixed", "groceries", "savings", "guilt_free", "paycheck", "vacation"]
    assert budget.loc["groceries", (2024, 12)] == pytest.approx(450.0, rel=1e-9)
    assert budget.loc["vacation", (2024, 7)] == pytest.approx(2400.0, rel=1e-9)
    assert budget.loc["fixed", (2025, 1)] == pytest.approx(2100.0, rel=1e-9)


# ---------------------------------------------------------------------------
# get_active_csp_plan
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("as_of,expected_fixed", [
    ("2025-01-01", 2100.0),  # exact-date match
    ("2024-12-31", 2000.0),  # between plans -> most recent earlier plan
    ("2024-07-15", 2000.0),  # boundary is inclusive (k <= as_of)
    ("2099-01-01", 2100.0),  # far future -> latest plan
])
def test_get_active_csp_plan_selects_by_as_of(as_of, expected_fixed):
    """Most recent plan with effective date <= as_of wins."""
    assert get_active_csp_plan(CSP_PLANS, as_of) == {"fixed": expected_fixed}


def test_get_active_csp_plan_empty_cases():
    """No plans, or as_of before all plans, returns {}."""
    assert get_active_csp_plan({}, "2025-01-01") == {}
    assert get_active_csp_plan(CSP_PLANS, "2023-01-01") == {}


def test_get_active_csp_plan_default_as_of_is_today():
    """With as_of=None an epoch-dated plan is always eligible (today >= 1970)."""
    assert get_active_csp_plan({"1970-01-01": {"fixed": 1.0}}) == {"fixed": 1.0}


# ---------------------------------------------------------------------------
# get_monthly_loading
# ---------------------------------------------------------------------------

def test_get_monthly_loading_normalizes_trailing_shape():
    """Weights follow the budget shape and sum to 1."""
    budget = read_budget(make_config(), "erik")
    loading = get_monthly_loading(budget, "groceries", TRAILING_2024)
    assert loading[:11] == [pytest.approx(0.08247422680412371, rel=1e-9)] * 11
    assert loading[11] == pytest.approx(0.09278350515463918, rel=1e-9)
    assert sum(loading) == pytest.approx(1.0, rel=1e-9)


def test_get_monthly_loading_one_hot_category():
    """A category budgeted in a single month loads entirely into that month."""
    budget = read_budget(make_config(), "erik")
    loading = get_monthly_loading(budget, "vacation", TRAILING_2024)
    assert loading == [0.0] * 6 + [1.0] + [0.0] * 5


def test_get_monthly_loading_missing_category_is_uniform():
    """A category absent from the budget falls back to a uniform 1/12 loading."""
    budget = read_budget(make_config(), "erik")
    assert get_monthly_loading(budget, "not_a_category", TRAILING_2024) == [1 / 12] * 12


def test_get_monthly_loading_zero_total_returns_twelve_weights():
    """Zero-total fallback always returns 12 weights, even for shorter periods."""
    # NOTE: possible bug — the fallback is hard-coded to [1/12]*12 regardless
    # of len(period_months); a 6-month period with no budget yields 12 weights.
    budget = read_budget(make_config(), "erik")
    six_months = [(2024, m) for m in range(1, 7)]
    loading = get_monthly_loading(budget, "vacation", six_months)  # all zeros Jan-Jun
    assert loading == [1 / 12] * 12
    assert len(loading) == 12


# ---------------------------------------------------------------------------
# generate_budget_from_csp
# ---------------------------------------------------------------------------

def _generated_to_frame(generated):
    """Nested {year: {month: {cat: amt}}} -> frame (rows=categories, cols=YYYY-MM)."""
    return pd.DataFrame({
        f"{year}-{int(month):02d}": cats
        for year, months in generated.items()
        for month, cats in months.items()
    })


def _generate_budget():
    budget = read_budget(make_config(), "erik")
    csp_plan = {"fixed": 2000.0, "groceries": 425.0, "vacation": 200.0, "newcat": 100.0}
    categories = ["fixed", "groceries", "vacation", "newcat"]
    return generate_budget_from_csp(csp_plan, budget, categories, dt(2025, 1, 1))


def test_generate_budget_from_csp_golden(goldens_dir):
    """Forward 12 months distributed by the trailing-12-month loading."""
    frame = _generated_to_frame(_generate_budget())
    golden = pd.read_csv(goldens_dir / "budget_generated_from_csp.csv", index_col=0)
    pd.testing.assert_frame_equal(frame, golden, rtol=1e-9, check_dtype=False)


def test_generate_budget_from_csp_structure_and_rounding():
    """String year/month keys; per-month rounding can drift off the exact annual."""
    generated = _generate_budget()
    assert set(generated) == {"2025"}
    assert set(generated["2025"]) == {str(m) for m in range(1, 13)}
    frame = _generated_to_frame(generated)
    # flat loading distributes exactly
    assert frame.loc["fixed"].sum() == pytest.approx(24000.0, rel=1e-9)
    # NOTE: rounding each month to 2dp drifts the annual total by $0.02
    # (425*12 = 5100 but the December-heavy loading rounds to 5100.02)
    assert frame.loc["groceries"].sum() == pytest.approx(5100.02, rel=1e-9)
    # one-hot loading concentrates the whole annual amount in July
    assert frame.loc["vacation", "2025-07"] == pytest.approx(2400.0, rel=1e-9)
    assert frame.loc["vacation"].sum() == pytest.approx(2400.0, rel=1e-9)
    # category absent from the trailing budget gets the uniform fallback
    assert frame.loc["newcat"].eq(100.0).all()


# ---------------------------------------------------------------------------
# calc_proportions
# ---------------------------------------------------------------------------

def test_calc_proportions_pins_columns():
    """overage/covered/remaining/proportion, with 0/0 proportion filled to 1."""
    df = pd.DataFrame({
        "amount": [50.0, 100.0, 0.0, 75.0],
        "budget": [100.0, 80.0, 0.0, 0.0],
    })
    out = calc_proportions(df)
    assert list(out["overage"]) == [0.0, 20.0, 0.0, 75.0]
    assert list(out["covered"]) == [50.0, 80.0, 0.0, 0.0]
    assert list(out["remaining"]) == [50.0, -20.0, 0.0, -75.0]
    # over budget caps proportion at 1; 0/0 -> NaN -> filled with 1
    assert out["proportion"].tolist() == pytest.approx(
        [0.5, 1.0, 1.0, 1.0], rel=1e-9)


# ---------------------------------------------------------------------------
# build_budget_report
# ---------------------------------------------------------------------------

def _build_report():
    budget = read_budget(make_config(), "erik")
    return build_budget_report(
        make_transactions(), budget,
        dt(2025, 1, 1), dt(2025, 1, 31),
        make_config(), "erik",
    )


def test_build_budget_report_golden(goldens_dir):
    """Full report frame: spend vs budget merge, totals, proportions, order."""
    report = _build_report().reset_index(drop=True)
    # csp_label is normalized to str for the CSV round-trip: budget-only rows
    # get literal int 0 (see test below), CSP group header rows get NaN.
    report["csp_label"] = report["csp_label"].astype(str)
    golden = pd.read_csv(goldens_dir / "budget_report.csv")
    golden["csp_label"] = golden["csp_label"].astype(str)
    pd.testing.assert_frame_equal(report, golden, rtol=1e-9, check_dtype=False)


def test_build_budget_report_totals_and_window():
    """Feb transaction excluded; totals sum non-income rows; spend is abs of sum."""
    report = _build_report().set_index("category")
    # fixed spend = |-1500 + -2500| = 4000 (t12 included, Feb t13 excluded)
    assert report.loc["fixed", "amount"] == pytest.approx(4000.0, rel=1e-9)
    assert report.loc["groceries", "amount"] == pytest.approx(120.5, rel=1e-9)
    assert report.loc["Total Spending", "amount"] == pytest.approx(4665.75, rel=1e-9)
    assert report.loc["Total Spending", "budget"] == pytest.approx(3530.0, rel=1e-9)
    assert report.loc["Total Income", "amount"] == pytest.approx(4000.0, rel=1e-9)
    assert report.loc["Total Income", "budget"] == pytest.approx(6100.0, rel=1e-9)


def test_build_budget_report_sorted_by_cat_order_descending():
    """Rows come back in reverse cat_order (sort_values order, ascending=False)."""
    report = _build_report()
    assert list(report["category"]) == list(reversed(make_config()["users"]["erik"]["cat_order"]))
    assert list(report["order"]) == list(range(12, -1, -1))


def test_build_budget_report_budget_only_row_label_is_zero():
    """A budgeted category with no spending gets csp_label 0 (int), not NaN."""
    # NOTE: possible bug — df.fillna(0) runs on the whole merged frame, so the
    # missing csp_label for budget-only rows ('vacation') becomes integer 0
    # rather than a label string.
    report = _build_report().set_index("category")
    assert report.loc["vacation", "csp_label"] == 0
    assert report.loc["vacation", "amount"] == pytest.approx(0.0, rel=1e-9)
    assert report.loc["vacation", "proportion"] == pytest.approx(0.0, rel=1e-9)


# ---------------------------------------------------------------------------
# order_budget
# ---------------------------------------------------------------------------

def test_order_budget_sorts_by_cat_order():
    """order_budget reorders rows by cat_order and emits a category column."""
    budget = read_budget(make_config(), "erik")
    small = budget.loc[:, [(2025, 1)]].copy()
    small.columns = ["Jan"]
    ordered = order_budget(small, make_config(), "erik")
    assert list(ordered.columns) == ["category", "Jan"]
    assert list(ordered["category"]) == [
        "paycheck", "fixed", "groceries", "savings", "vacation", "guilt_free"]
    assert ordered["Jan"].tolist() == pytest.approx(
        [6100.0, 2100.0, 420.0, 500.0, 200.0, 310.0], rel=1e-9)
    assert list(ordered.index) == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Golden generation — run ONCE per characterization epoch:
#   uv run python -m tests.test_budget
# A refactor may NEVER regenerate these goldens; they pin current-main behavior.
# ---------------------------------------------------------------------------

def _generate_goldens():
    from pathlib import Path

    goldens = Path(__file__).parent / "goldens"
    goldens.mkdir(exist_ok=True)

    frame = _generated_to_frame(_generate_budget())
    frame.to_csv(goldens / "budget_generated_from_csp.csv", float_format="%.17g")

    report = _build_report().reset_index(drop=True)
    report["csp_label"] = report["csp_label"].astype(str)
    report.to_csv(goldens / "budget_report.csv", index=False, float_format="%.17g")

    print(f"Wrote goldens to {goldens}")


if __name__ == "__main__":
    _generate_goldens()
