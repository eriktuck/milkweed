"""Characterization tests for the transaction pipeline in core/utils/functions.py.

Pins the CURRENT behavior of:
convert_raw_transactions_to_dataframe -> preprocess_transactions ->
filter_owner_transactions / categorize_transactions /
filter_and_process_owner_transactions -> process_and_attribute_transactions,
plus check_processed.

All data is synthetic; no Firebase, no network. Goldens live in
tests/goldens/ with the ``txn_`` prefix and are generated ONCE (see
_generate_goldens at the bottom) — a refactor may NEVER regenerate them.
"""

import copy

import pandas as pd
import pytest

from core.utils.functions import (
    convert_raw_transactions_to_dataframe,
    preprocess_transactions,
    filter_owner_transactions,
    categorize_transactions,
    filter_and_process_owner_transactions,
    process_and_attribute_transactions,
    check_processed,
)

# ---------------------------------------------------------------------------
# Fixtures: hand-crafted raw Monarch transactions + synthetic configs
# ---------------------------------------------------------------------------


def _txn(i, date, amount, cat, acct, merchant="Merchant", hide=False, notes=None):
    """One raw Monarch transaction dict (nested category/account/merchant)."""
    return {
        "id": f"t{i}",
        "date": date,
        "amount": amount,
        "pending": False,
        "hideFromReports": hide,
        "plaidName": f"PLAID {cat.upper()}" if cat else "PLAID RAW",
        "notes": notes,
        "category": {"id": f"cat_{i}", "name": cat} if cat is not None else None,
        "account": {"id": f"acct_{i}", "displayName": acct},
        "merchant": {"id": f"m_{i}", "name": merchant},
    }


def make_raw_txns():
    """Eleven raw transactions covering every attribution/categorization path."""
    return [
        # erik, group-mapped (Housing -> fixed)
        _txn(1, "2025-01-02", -1500.0, "Rent", "Erik Checking", "Landlord"),
        # erik, income (Income -> paycheck -> 'income' label)
        _txn(2, "2025-01-03", 4000.0, "Paycheck", "Erik Checking", "Employer",
             notes="january pay"),
        # erik, category-mapped (Groceries -> groceries; group Food unmapped)
        _txn(3, "2025-01-05", -120.5, "Groceries", "Erik Checking", "Grocer"),
        # erik, account ABSENT from transaction_account_settings -> still kept
        _txn(4, "2025-01-06", -500.0, "Vanguard Transfer", "Erik Savings", "Vanguard"),
        # erik, account marked include=False -> dropped by filter_owner_transactions
        _txn(5, "2025-01-07", -60.0, "Restaurants", "Erik Credit Card", "Bistro"),
        # erik, unmapped category AND group -> guilt_free fallback
        _txn(6, "2025-01-08", -45.25, "Restaurants", "Erik Checking", "Taqueria"),
        # erik, category in drop_cats -> dropped by categorize_transactions
        _txn(7, "2025-01-09", -300.0, "Credit Card Payment", "Erik Checking", "Card Co"),
        # erik, hideFromReports=True -> dropped by categorize_transactions
        _txn(8, "2025-01-10", -30.0, "Gym", "Erik Checking", "Gym Co", hide=True),
        # household (joint) transactions
        _txn(9, "2025-01-11", -2200.0, "Mortgage", "Joint Checking", "Bank"),
        _txn(10, "2025-01-12", -150.0, "Utilities", "Joint Checking", "Power Co"),
        # account owned by nobody -> never attributed
        _txn(11, "2025-01-13", -10.0, "Groceries", "Unknown Account", "Corner Store"),
    ]


ERIK_CONFIG = {
    "name": "erik",
    "accounts": ["Erik Checking", "Erik Savings", "Erik Credit Card"],
    "transaction_account_settings": {
        "Erik Checking": {"include": True, "nickname": "Checking"},
        "Erik Credit Card": {"include": False, "nickname": "CC"},
        # "Erik Savings" intentionally absent -> included by default
    },
    "cat_names": {
        "Rent": "Housing",
        "Paycheck": "Income",
        "Groceries": "Food",
        "Restaurants": "Food",
        "Vanguard Transfer": "Transfers",
        "Credit Card Payment": "Transfers",
        "Gym": "Health",
    },
    "csp_from_group": {"Housing": "fixed", "Income": "paycheck"},
    "csp_from_category": {"Groceries": "groceries", "Vanguard Transfer": "savings"},
    "csp_labels": {
        "fixed": "fixed",
        "paycheck": "income",
        "groceries": "fixed",
        "savings": "investments",
        "guilt_free": "guilt-free",
    },
    "drop_cats": ["Credit Card Payment"],
}

HOUSEHOLD_CONFIG = {
    "name": "the andersons",
    "members": ["erik-uid", "amy-uid"],  # presence of 'members' => kind 'households'
    "accounts": ["Joint Checking"],
    "transaction_account_settings": None,  # exercises the `or {}` default path
    "cat_names": {"Utilities": "Bills", "Mortgage": "Housing"},
    "csp_from_group": {"Housing": "fixed", "Bills": "fixed"},
    "csp_from_category": {},
    "csp_labels": {"fixed": "fixed", "guilt_free": "guilt-free"},
    "drop_cats": [],
}


class StubConfig:
    """Minimal stand-in for SessionData: only get_user_configs is consumed."""

    def __init__(self, users):
        self._users = users

    def get_user_configs(self):
        return self._users


def make_config():
    return StubConfig({
        "erik-uid": copy.deepcopy(ERIK_CONFIG),
        "hh-uid": copy.deepcopy(HOUSEHOLD_CONFIG),
    })


def make_preprocessed():
    """Raw -> converted -> preprocessed frame (the pipeline's shared input)."""
    return preprocess_transactions(
        convert_raw_transactions_to_dataframe(make_raw_txns())
    )


# ---------------------------------------------------------------------------
# Golden helpers
# ---------------------------------------------------------------------------

def _read_golden(path):
    golden = pd.read_csv(path)
    if "date" in golden.columns:
        golden["date"] = pd.to_datetime(golden["date"], utc=True)
    return golden


def _assert_matches_golden(df, path):
    """Strict on values; index is normalized (positional labels are asserted
    separately) and dtypes are relaxed for the CSV round-trip (all-NaN object
    columns come back float64, None comes back NaN)."""
    actual = df.reset_index(drop=True).copy()
    # Normalize None -> NaN in object columns (CSV round-trip yields NaN).
    for col in actual.columns:
        if actual[col].dtype == object:
            actual[col] = actual[col].where(actual[col].notna())
    golden = _read_golden(path)
    pd.testing.assert_frame_equal(actual, golden, rtol=1e-9, check_dtype=False)


# ---------------------------------------------------------------------------
# convert / preprocess
# ---------------------------------------------------------------------------

def test_convert_raw_transactions_parses_dates_utc():
    """convert_raw_transactions_to_dataframe keeps raw columns and parses date to UTC."""
    df = convert_raw_transactions_to_dataframe(make_raw_txns())
    assert df.shape == (11, 10)
    assert str(df["date"].dtype) == "datetime64[ns, UTC]"
    assert list(df["id"]) == [f"t{i}" for i in range(1, 12)]
    # nested dicts are untouched at this stage
    assert df.loc[0, "category"] == {"id": "cat_1", "name": "Rent"}


def test_preprocess_transactions_golden(goldens_dir):
    """preprocess flattens category/account and drops the nested columns."""
    pre = make_preprocessed()
    assert set(pre.columns) == {
        "id", "date", "amount", "pending", "hideFromReports",
        "plaidName", "notes", "category_name", "account_name",
    }
    _assert_matches_golden(pre, goldens_dir / "txn_preprocessed.csv")


def test_preprocess_transactions_mutates_input_in_place():
    """preprocess_transactions drops columns inplace on the frame it is given."""
    df = convert_raw_transactions_to_dataframe(make_raw_txns())
    out = preprocess_transactions(df)
    assert out is df
    assert "category" not in df.columns


def test_preprocess_handles_non_dict_category():
    """A None category/merchant flattens to empty-string category_name."""
    df = convert_raw_transactions_to_dataframe([
        {"id": "e1", "date": "2025-02-01", "amount": -5.0, "pending": False,
         "hideFromReports": False, "plaidName": "X", "notes": None,
         "category": None, "account": {"displayName": "Erik Checking"},
         "merchant": None},
    ])
    pre = preprocess_transactions(df)
    assert pre.loc[0, "category_name"] == ""
    assert pre.loc[0, "account_name"] == "Erik Checking"


# ---------------------------------------------------------------------------
# filter_owner_transactions
# ---------------------------------------------------------------------------

def test_filter_owner_respects_include_false():
    """include=False accounts are dropped; absent-from-settings accounts kept."""
    subset = filter_owner_transactions(make_preprocessed(), ERIK_CONFIG)
    # Erik Credit Card (t5) excluded; Erik Savings (t4) kept despite no settings entry
    assert sorted(subset["id"]) == ["t1", "t2", "t3", "t4", "t6", "t7", "t8"]
    assert (subset["account_owner"] == "erik").all()
    # original positional index labels are preserved (no reset)
    assert list(subset.index) == [0, 1, 2, 3, 5, 6, 7]


def test_filter_owner_none_settings_includes_all_accounts():
    """transaction_account_settings=None behaves as 'include everything'."""
    subset = filter_owner_transactions(make_preprocessed(), HOUSEHOLD_CONFIG)
    assert sorted(subset["id"]) == ["t10", "t9"]
    assert (subset["account_owner"] == "the andersons").all()


def test_filter_owner_no_matching_accounts_returns_empty():
    """A config with no owned accounts yields an empty, tagged frame."""
    config = {"name": "nobody", "accounts": ["Ghost Account"]}
    subset = filter_owner_transactions(make_preprocessed(), config)
    assert subset.empty
    assert "account_owner" in subset.columns


# ---------------------------------------------------------------------------
# categorize_transactions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("txn_id,csp,csp_label", [
    ("t1", "fixed", "fixed"),            # via csp_from_group (Housing)
    ("t2", "paycheck", "income"),        # via csp_from_group (Income)
    ("t3", "groceries", "fixed"),        # via csp_from_category (group unmapped)
    ("t4", "savings", "investments"),    # via csp_from_category
    ("t6", "guilt_free", "guilt-free"),  # unmapped -> guilt_free fallback
])
def test_categorize_transactions_csp_mapping(txn_id, csp, csp_label):
    """Group mapping first, category mapping second, guilt_free fallback last."""
    out = filter_and_process_owner_transactions(make_preprocessed(), ERIK_CONFIG)
    row = out.loc[out["id"] == txn_id].iloc[0]
    assert row["csp"] == csp
    assert row["csp_label"] == csp_label


def test_categorize_group_mapping_beats_category_mapping():
    """When both group and category map, csp_from_group wins (fillna order)."""
    config = copy.deepcopy(ERIK_CONFIG)
    config["csp_from_category"]["Rent"] = "should_not_win"
    out = filter_and_process_owner_transactions(make_preprocessed(), config)
    assert out.loc[out["id"] == "t1", "csp"].iloc[0] == "fixed"


def test_categorize_drops_drop_cats_and_hidden():
    """drop_cats categories and hideFromReports rows are removed."""
    out = filter_and_process_owner_transactions(make_preprocessed(), ERIK_CONFIG)
    assert "t7" not in set(out["id"])  # Credit Card Payment in drop_cats
    assert "t8" not in set(out["id"])  # hideFromReports=True


def test_categorize_unmapped_category_group_is_nan():
    """A category missing from cat_names leaves category_group as NaN."""
    filtered = filter_owner_transactions(make_preprocessed(), HOUSEHOLD_CONFIG)
    config = copy.deepcopy(HOUSEHOLD_CONFIG)
    del config["cat_names"]["Utilities"]
    out = categorize_transactions(filtered, config)
    row = out.loc[out["id"] == "t10"].iloc[0]
    assert pd.isna(row["category_group"])
    assert row["csp"] == "guilt_free"
    assert row["csp_label"] == "guilt-free"


# ---------------------------------------------------------------------------
# filter_and_process_owner_transactions (full per-owner pipeline) — goldens
# ---------------------------------------------------------------------------

def test_filter_and_process_erik_golden(goldens_dir):
    """Full per-owner pipeline output for the individual user config."""
    out = filter_and_process_owner_transactions(make_preprocessed(), ERIK_CONFIG)
    assert list(out.index) == [0, 1, 2, 3, 5]  # original labels survive
    _assert_matches_golden(out, goldens_dir / "txn_erik_final.csv")


def test_filter_and_process_household_golden(goldens_dir):
    """Full per-owner pipeline output for the household (joint) config."""
    out = filter_and_process_owner_transactions(make_preprocessed(), HOUSEHOLD_CONFIG)
    assert list(out.index) == [8, 9]
    _assert_matches_golden(out, goldens_dir / "txn_household_final.csv")


# ---------------------------------------------------------------------------
# process_and_attribute_transactions / check_processed
# ---------------------------------------------------------------------------

def test_process_and_attribute_structure(goldens_dir):
    """One entry per config, kind derived from the 'members' key, uid preserved."""
    processed = process_and_attribute_transactions(make_preprocessed(), make_config())
    meta = [(e["owner_name"], e["kind"], e["uid"]) for e in processed]
    assert meta == [
        ("erik", "users", "erik-uid"),
        ("the andersons", "households", "hh-uid"),
    ]
    _assert_matches_golden(
        processed[0]["transactions"], goldens_dir / "txn_erik_final.csv")
    _assert_matches_golden(
        processed[1]["transactions"], goldens_dir / "txn_household_final.csv")


def test_check_processed_returns_dropped_transactions():
    """Dropped = excluded account + drop_cats + hidden + unowned account."""
    pre = make_preprocessed()
    processed = process_and_attribute_transactions(pre, make_config())
    dropped = check_processed(pre, processed)
    assert sorted(dropped["id"]) == ["t11", "t5", "t7", "t8"]


def test_check_processed_empty_when_nothing_dropped():
    """When every transaction is attributed, the dropped frame is empty."""
    pre = make_preprocessed()
    keep = pre.loc[pre["id"].isin(["t1", "t2"])].copy()
    processed = process_and_attribute_transactions(keep, make_config())
    dropped = check_processed(keep, processed)
    assert dropped.empty


# ---------------------------------------------------------------------------
# Golden generation — run ONCE per characterization epoch:
#   uv run python -m tests.test_transactions
# A refactor may NEVER regenerate these goldens; they pin current-main behavior.
# ---------------------------------------------------------------------------

def _generate_goldens():
    from pathlib import Path

    goldens = Path(__file__).parent / "goldens"
    goldens.mkdir(exist_ok=True)

    pre = make_preprocessed()
    pre.reset_index(drop=True).to_csv(
        goldens / "txn_preprocessed.csv", index=False, float_format="%.17g")

    erik = filter_and_process_owner_transactions(make_preprocessed(), ERIK_CONFIG)
    erik.reset_index(drop=True).to_csv(
        goldens / "txn_erik_final.csv", index=False, float_format="%.17g")

    hh = filter_and_process_owner_transactions(make_preprocessed(), HOUSEHOLD_CONFIG)
    hh.reset_index(drop=True).to_csv(
        goldens / "txn_household_final.csv", index=False, float_format="%.17g")

    print(f"Wrote goldens to {goldens}")


if __name__ == "__main__":
    _generate_goldens()
