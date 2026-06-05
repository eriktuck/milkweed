"""Panel data helpers for the Trends page (Concept C, Phase 4.1).

Pure, unit-testable aggregation over the processed transactions DataFrame (the
same shape as the `transaction-data-store`: columns include `date`,
`account_owner`, `amount`, `csp`, `csp_label`, `category_name`, ...). No
Firestore coupling — the only external dependency is the injected DataFrame,
plus an optional injected mapping for the Budget baseline. The Plotly figure
builders (Phase 4.2) consume these frames; the page callbacks (4.3) wire them.

Column conventions (see core/utils/functions.py::categorize_transactions):
    csp_label ∈ {income, fixed, investments, sinking, guilt-free}  — the bucket
    csp                                                            — finer CSP key
    category_name                                                  — Monarch category
    amount                                                         — signed (negative = expense)

Sign convention: expenses are stored negative and income positive. Spending is
reported by **negating** the summed amount (``* -1``) so expenses read as a
positive spend, and a month whose credits exceed its debits correctly stays
**negative** (a net refund). We never ``.abs()`` spending — that would misreport
such a month as positive. Income is summed as-is (already positive) and is
identified by ``csp_label == 'income'`` rather than by amount sign.
"""

from __future__ import annotations

from datetime import datetime as dt

import pandas as pd
import pytz

# ── Conventions / defaults ──────────────────────────────────────────────────────

INCOME_LABEL = "income"

# Default Food & Dining composite (Q2). Maps a display "part" to the Monarch
# category_names it sums. Hardcoded in v1; the page may later source this from a
# saved-composite config. Category names must match the user's Monarch categories;
# `composite_over_time` ignores any that are absent, so listing variants is safe.
# "Dining out" lists the current split categories (Bars, Alcohol) plus the legacy
# combined "Bars & Alcohol" that some users still use, and the broader eating-out
# categories (Fast Food, Food Delivery, Coffee Shops).
DEFAULT_FOOD_DINING_PARTS: dict[str, list[str]] = {
    "Groceries": ["Groceries"],
    "Dining out": [
        "Restaurants", "Bars", "Alcohol", "Bars & Alcohol",
        "Fast Food", "Food Delivery", "Coffee Shops",
    ],
}

# Parts to break out into one column per subcategory (vs. a single summed column).
# Dining out is broken out by default so its subcategories show individually, while
# Groceries stays a single series — the figure shades each broken-out subcategory.
DEFAULT_FOOD_DINING_BREAKOUT: frozenset[str] = frozenset({"Dining out"})

# Heuristic for splitting income into paychecks vs other (Q3) when the caller
# doesn't supply an explicit paycheck-category list.
DEFAULT_PAYCHECK_KEYWORDS: tuple[str, ...] = ("paycheck", "payroll", "salary", "wage")

DEFAULT_TOP_N = 5
OTHER_LABEL = "Other"
COMBINED_LABEL = "Combined"
TOTAL_INCOME_LABEL = "Total income"

# Movers/overages defaults — the "what counts as an overage" knobs the spec left
# open. Tunable; chosen to suppress small-dollar noise.
DEFAULT_MIN_AMOUNT = 50.0       # ignore categories smaller than this in both periods
DEFAULT_OVERAGE_ABS = 200.0     # an overage must be at least this many $ over baseline
DEFAULT_OVERAGE_RATIO = 1.5     # ...and at least this multiple of the baseline


# ── Internal helpers ────────────────────────────────────────────────────────────

_UTC = pytz.UTC


def _to_utc(d) -> pd.Timestamp:
    """Normalize a date (str / datetime / Timestamp) to a UTC pandas Timestamp."""
    if isinstance(d, str):
        d = dt.fromisoformat(d)
    ts = pd.Timestamp(d)
    return ts.tz_localize(_UTC) if ts.tzinfo is None else ts.tz_convert(_UTC)


def _scope(
    df: pd.DataFrame,
    owner: str | None,
    start_date=None,
    end_date=None,
) -> pd.DataFrame:
    """Owner + date-range filter, with `date` coerced to UTC datetime.

    Mirrors the filtering in ``plot_spending_trends``: `owner` is matched
    opaquely against `account_owner` (whatever the page's `use-case` value is).
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    out = out.dropna(subset=["date"])

    mask = pd.Series(True, index=out.index)
    if owner is not None and "account_owner" in out.columns:
        mask &= out["account_owner"] == owner
    if start_date is not None:
        mask &= out["date"] >= _to_utc(start_date)
    if end_date is not None:
        mask &= out["date"] <= _to_utc(end_date)
    return out.loc[mask]


def _over_time(
    df: pd.DataFrame, group_col: str, freq: str, *, negate: bool = True
) -> pd.DataFrame:
    """Pivot to period × group_col of summed amounts.

    index = period start (Grouper `freq`), columns = distinct `group_col`
    values, values = Σ amount. With `negate=True` (the default, for spending) the
    summed amount is multiplied by −1 so expenses read as positive spend and a
    net-credit month stays negative — never abs'd. Pass `negate=False` for income
    (already positive). The shared time-bucketing primitive behind Q1/Q2/Q4.
    """
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby([pd.Grouper(key="date", freq=freq), group_col])["amount"]
        .sum()
        .reset_index()
    )
    pivot = grouped.pivot_table(
        index="date", columns=group_col, values="amount", aggfunc="sum"
    ).fillna(0.0)
    if negate:
        pivot = pivot.mul(-1)
    pivot.columns.name = None
    return pivot


def income_by_period(df: pd.DataFrame, freq: str) -> pd.Series:
    """Σ income per period (for the Q1 reference line / % normalization).

    Income is stored positive, so it is summed as-is (no negate, no abs); a
    net-negative income month would honestly read negative.
    """
    inc = df[df["csp_label"] == INCOME_LABEL]
    if inc.empty:
        return pd.Series(dtype=float)
    return inc.groupby(pd.Grouper(key="date", freq=freq))["amount"].sum()


# ── Q1 — CSP buckets over time (optionally as a share of income) ─────────────────

SPENDING_LABEL_ORDER: tuple[str, ...] = ("fixed", "investments", "sinking", "guilt-free")


def csp_shares_over_time(
    df: pd.DataFrame,
    owner: str | None,
    start_date,
    end_date,
    freq: str,
    *,
    as_percent: bool = False,
    labels: tuple[str, ...] = SPENDING_LABEL_ORDER,
) -> tuple[pd.DataFrame, pd.Series]:
    """CSP buckets over time + the income series (Q1).

    Returns ``(pivot, income)`` where `pivot` has one column per CSP `label`
    (ordered, spending shown positive via negation) and `income` is Σ income per
    period. With `as_percent=True`, each bucket is divided by that period's income
    (the share of income), leaving `income` unchanged for the 100% reference line.
    """
    scoped = _scope(df, owner, start_date, end_date)
    income = income_by_period(scoped, freq)

    spend = scoped[scoped["csp_label"].isin(labels)]
    pivot = _over_time(spend, "csp_label", freq)
    if not pivot.empty:
        pivot = pivot.reindex(columns=[l for l in labels if l in pivot.columns])

    if as_percent and not pivot.empty:
        ref = income.reindex(pivot.index).replace(0, pd.NA)
        pivot = pivot.div(ref, axis=0).fillna(0.0)

    return pivot, income


# ── Q4 — composition within a bucket (top-N + Other) ─────────────────────────────

def composition_over_time(
    df: pd.DataFrame,
    owner: str | None,
    start_date,
    end_date,
    freq: str,
    *,
    csp_label: str = "guilt-free",
    top_n: int = DEFAULT_TOP_N,
    group_col: str = "category_name",
) -> pd.DataFrame:
    """Breakdown of one CSP bucket over time, top-N categories + an "Other" rollup.

    Powers Q4 ("within guilt-free, which categories dominate?"). Ranks the
    bucket's `group_col` values by total spend over the window, keeps the top-N
    as their own columns (ordered by total, descending), and folds the long tail
    into a single ``Other`` column. Returns an empty frame when the bucket has no
    spend in scope.

    `group_col` defaults to `category_name` (finer, per the spec's "which
    categories"); pass `csp` to match the existing drill-down's grouping.
    """
    scoped = _scope(df, owner, start_date, end_date)
    scoped = scoped[scoped["csp_label"] == csp_label]
    pivot = _over_time(scoped, group_col, freq)
    if pivot.empty:
        return pivot

    totals = pivot.sum().sort_values(ascending=False)
    top = list(totals.index[:top_n])
    rest = [c for c in totals.index if c not in top]

    out = pivot[top].copy()
    if rest:
        out[OTHER_LABEL] = pivot[rest].sum(axis=1)
    return out


# ── Q2 — cross-CSP composite (Food & Dining) ─────────────────────────────────────

def composite_over_time(
    df: pd.DataFrame,
    owner: str | None,
    start_date,
    end_date,
    freq: str,
    parts: dict[str, list[str]] | None = None,
    *,
    breakout: frozenset[str] | set[str] | None = None,
    group_col: str = "category_name",
    total_label: str = COMBINED_LABEL,
) -> pd.DataFrame:
    """A composite metric that may span CSP buckets (Q2: Food & Dining).

    `parts` maps a display part → the `group_col` values it sums (default
    ``DEFAULT_FOOD_DINING_PARTS``). A part listed in `breakout` (default
    ``DEFAULT_FOOD_DINING_BREAKOUT`` = Dining out) is expanded into one column per
    present subcategory, in listed order; other parts become a single summed
    column named by the part. A ``total_label`` column sums everything. Because
    parts are addressed by category, the composite can pull Groceries (fixed) and
    the dining subcategories (guilt-free) into one frame. Returns an empty frame
    when none of the part categories appear in scope.
    """
    if parts is None:
        parts = DEFAULT_FOOD_DINING_PARTS
    if breakout is None:
        breakout = DEFAULT_FOOD_DINING_BREAKOUT

    scoped = _scope(df, owner, start_date, end_date)
    all_cats = [c for cats in parts.values() for c in cats]
    scoped = scoped[scoped[group_col].isin(all_cats)]
    if scoped.empty:
        return pd.DataFrame()

    def _monthly(sub: pd.DataFrame) -> pd.Series:
        return sub.groupby(pd.Grouper(key="date", freq=freq))["amount"].sum().mul(-1)

    series = {}
    present = set(scoped[group_col].unique())
    for part, cats in parts.items():
        if part in breakout:
            for cat in cats:  # one column per present subcategory, in listed order
                if cat in present:
                    series[cat] = _monthly(scoped[scoped[group_col] == cat])
        else:
            series[part] = _monthly(scoped[scoped[group_col].isin(cats)])

    out = pd.concat(series, axis=1).fillna(0.0)
    out[total_label] = out.sum(axis=1)
    return out


# ── Q3 — income split (paychecks vs other) ───────────────────────────────────────

def income_split_over_time(
    df: pd.DataFrame,
    owner: str | None,
    start_date,
    end_date,
    freq: str,
    *,
    paycheck_categories: list[str] | None = None,
    group_col: str = "category_name",
) -> pd.DataFrame:
    """Income over time, split Paychecks vs Other sources, with a total.

    Income is ``csp_label == 'income'``. When `paycheck_categories` is not given,
    categories whose name contains a ``DEFAULT_PAYCHECK_KEYWORDS`` token are
    treated as paychecks. Columns: ``Paychecks``, ``Other sources``,
    ``Total income``. Empty frame when there's no income in scope.
    """
    scoped = _scope(df, owner, start_date, end_date)
    scoped = scoped[scoped["csp_label"] == INCOME_LABEL]
    if scoped.empty:
        return pd.DataFrame()

    if paycheck_categories is None:
        cats = scoped[group_col].dropna().unique()
        paycheck_categories = [
            c for c in cats
            if any(tok in str(c).lower() for tok in DEFAULT_PAYCHECK_KEYWORDS)
        ]

    # Income is stored positive; sum as-is (no negate / no abs).
    is_paycheck = scoped[group_col].isin(paycheck_categories)
    pay = (
        scoped[is_paycheck].groupby(pd.Grouper(key="date", freq=freq))["amount"].sum()
    )
    other = (
        scoped[~is_paycheck].groupby(pd.Grouper(key="date", freq=freq))["amount"].sum()
    )
    out = pd.concat({"Paychecks": pay, "Other sources": other}, axis=1).fillna(0.0)
    out[TOTAL_INCOME_LABEL] = out.sum(axis=1)
    return out


# ── Movers rail — current period vs baseline ─────────────────────────────────────

def resolve_current_period(df: pd.DataFrame, owner: str | None, as_of=None) -> pd.Period | None:
    """The 'current month' for movers: the latest spending month present in the
    data that is ≤ `as_of` (default: the owner's last transaction). This is robust
    to recency gaps — a date-range end in an empty current calendar month falls
    back to the last month that actually has spend, so comparisons stay meaningful.
    Returns None when the owner has no spending."""
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], utc=True, errors="coerce")
    work = work.dropna(subset=["date"])
    if owner is not None and "account_owner" in work.columns:
        work = work[work["account_owner"] == owner]
    work = work[work["csp_label"] != INCOME_LABEL]
    if work.empty:
        return None
    anchor = _to_utc(as_of) if as_of is not None else work["date"].max()
    cand = anchor.tz_localize(None).to_period("M")
    ym = work["date"].dt.tz_localize(None).dt.to_period("M")
    avail = ym[ym <= cand]
    return avail.max() if not avail.empty else cand


def category_movers(
    df: pd.DataFrame,
    owner: str | None,
    *,
    as_of=None,
    baseline: str = "trailing_12mo",
    level: str = "category_name",
    baseline_amounts: dict[str, float] | None = None,
    min_amount: float = DEFAULT_MIN_AMOUNT,
) -> pd.DataFrame:
    """Per-category spend this month vs a baseline — the data behind the rail.

    Spending only (income excluded). The current period is the calendar month of
    `as_of` (default: the latest transaction date for the owner). The baseline is
    one of:

      * ``trailing_12mo`` — mean monthly spend over the 12 months *before* the
        current month (Σ over those months ÷ 12);
      * ``last_year``     — spend in the same calendar month one year earlier;
      * ``budget``        — the injected ``baseline_amounts`` map (category →
        monthly budget); categories absent from the map get a 0 baseline.

    Returns a DataFrame indexed by `level` with columns
    ``current, baseline, delta, pct_change, direction`` (``direction`` ∈
    {up, down, flat}), sorted by `delta` descending. Rows where neither the
    current nor baseline magnitude reaches `min_amount` are dropped to suppress
    noise. Empty frame when there's no spend in the current month.

    Note: the baseline window deliberately reaches *before* any chart date-range
    start — a 12-month baseline needs 12 months of history. Pass the full
    owner-scoped DataFrame; `as_of` (the page's end-of-range) anchors "now".
    The current month may be partial; that caveat is left to display/tuning.
    """
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], utc=True, errors="coerce")
    work = work.dropna(subset=["date"])
    if owner is not None and "account_owner" in work.columns:
        work = work[work["account_owner"] == owner]
    work = work[work["csp_label"] != INCOME_LABEL]
    if work.empty:
        return _empty_movers(level)

    cur_period = resolve_current_period(df, owner, as_of)
    if cur_period is None:
        return _empty_movers(level)
    # Drop tz before period conversion — month bucketing is tz-agnostic and the
    # conversion otherwise emits a "dropping timezone information" warning.
    work["ym"] = work["date"].dt.tz_localize(None).dt.to_period("M")

    # Spending = negated signed sum (expenses → positive; net-credit stays negative).
    current = work[work["ym"] == cur_period].groupby(level)["amount"].sum().mul(-1)
    if current.empty and baseline != "budget":
        return _empty_movers(level)

    if baseline == "budget":
        # Injected monthly budget magnitudes (expected positive); used as-is.
        base = pd.Series(baseline_amounts or {}, dtype=float)
    elif baseline == "last_year":
        base = (
            work[work["ym"] == (cur_period - 12)]
            .groupby(level)["amount"].sum().mul(-1)
        )
    else:  # trailing_12mo
        prior = work[(work["ym"] < cur_period) & (work["ym"] >= cur_period - 12)]
        base = prior.groupby(level)["amount"].sum().mul(-1) / 12.0

    out = pd.DataFrame({"current": current, "baseline": base}).fillna(0.0)
    out = out[out[["current", "baseline"]].max(axis=1) >= min_amount]
    if out.empty:
        return _empty_movers(level)

    out["delta"] = out["current"] - out["baseline"]
    out["pct_change"] = out.apply(
        lambda r: (r["delta"] / r["baseline"]) if r["baseline"] > 0 else float("nan"),
        axis=1,
    )
    out["direction"] = out["delta"].apply(
        lambda d: "up" if d > 0 else ("down" if d < 0 else "flat")
    )
    out.index.name = level
    return out.sort_values("delta", ascending=False)


def _empty_movers(level: str) -> pd.DataFrame:
    out = pd.DataFrame(
        columns=["current", "baseline", "delta", "pct_change", "direction"]
    )
    out.index.name = level
    return out


def top_movers(movers: pd.DataFrame, n: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a `category_movers` frame into the top-n risers and top-n fallers."""
    if movers.empty:
        return movers, movers
    up = movers[movers["delta"] > 0].nlargest(n, "delta")
    down = movers[movers["delta"] < 0].nsmallest(n, "delta")
    return up, down


def flag_overages(
    movers: pd.DataFrame,
    *,
    abs_threshold: float = DEFAULT_OVERAGE_ABS,
    ratio_threshold: float = DEFAULT_OVERAGE_RATIO,
) -> pd.DataFrame:
    """Categories materially above baseline this month — the rail's ⚠ section.

    An overage is both ≥ `abs_threshold` dollars over baseline *and* ≥
    `ratio_threshold` × the baseline (so a category that merely doubled from $20
    to $40 doesn't qualify). Sorted by `delta` descending.
    """
    if movers.empty:
        return movers
    over = movers[
        (movers["delta"] >= abs_threshold)
        & (movers["current"] >= movers["baseline"] * ratio_threshold)
    ]
    return over.sort_values("delta", ascending=False)
