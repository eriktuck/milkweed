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

SNAPSHOT_WINDOW_DAYS = 30        # rolling window for the siderail category snapshot


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


# ── Snapshot rail — rolling recent spend vs baseline ─────────────────────────────

_SNAPSHOT_COLUMNS = ["current", "baseline", "delta", "pct_change", "direction"]


def category_snapshot(
    df: pd.DataFrame,
    owner: str | None,
    *,
    items: list[tuple[str, list[str]]],
    as_of=None,
    baseline: str = "trailing_12mo",
    window_days: int = SNAPSHOT_WINDOW_DAYS,
) -> pd.DataFrame:
    """Rolling recent spend per item vs a baseline — the data behind the rail.

    Each item is a ``(label, category_names)`` pair; spend is summed across that
    item's `category_names`. Spending only (income excluded). ``current`` is the
    negated signed sum over the last `window_days` (expenses → positive; a net
    credit stays negative — never ``.abs()``). The window ends at `as_of`
    (default: the owner's latest transaction date). The baseline is one of:

      * ``trailing_12mo`` — mean monthly spend over the trailing 12 months
        (Σ over the 365 days ending at the anchor ÷ 12);
      * ``last_year``     — spend in the same `window_days` window 365 days back;
      * ``budget``        — falls back to ``trailing_12mo`` (the fixed snapshot
        categories have no per-category budgets).

    Returns a DataFrame indexed by item label (input order preserved) with
    columns ``current, baseline, delta, pct_change, direction`` (``direction`` ∈
    {up, down, flat}). ``pct_change`` is NaN when the baseline is ≤ 0. Empty
    frame (with those columns) when there's no usable data.
    """
    empty = pd.DataFrame(columns=_SNAPSHOT_COLUMNS)
    empty.index.name = "label"

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], utc=True, errors="coerce")
    work = work.dropna(subset=["date"])
    if owner is not None and "account_owner" in work.columns:
        work = work[work["account_owner"] == owner]
    work = work[work["csp_label"] != INCOME_LABEL]
    if work.empty:
        return empty

    # Anchor "now" at the latest transaction at-or-before `as_of`. The date picker
    # defaults its end to today, which may be past the last synced transaction —
    # clamping keeps the rolling window over data that actually exists.
    latest = work["date"].max()
    if pd.isna(latest):
        return empty
    anchor = min(_to_utc(as_of), latest) if as_of is not None else latest

    window = pd.Timedelta(days=window_days)
    year = pd.Timedelta(days=365)

    cur_win = work[(work["date"] > anchor - window) & (work["date"] <= anchor)]
    if baseline == "last_year":
        base_anchor = anchor - year
        base_win = work[(work["date"] > base_anchor - window) & (work["date"] <= base_anchor)]
        base_divisor = 1.0
    else:  # trailing_12mo (and budget, which has no per-category budgets)
        base_win = work[(work["date"] > anchor - year) & (work["date"] <= anchor)]
        base_divisor = 12.0

    def _spend(frame, cats):
        return float(frame[frame["category_name"].isin(cats)]["amount"].sum()) * -1

    rows = []
    for label, cats in items:
        current = _spend(cur_win, cats)
        base = _spend(base_win, cats) / base_divisor
        delta = current - base
        pct = (delta / base) if base > 0 else float("nan")
        rows.append({
            "current": current, "baseline": base, "delta": delta,
            "pct_change": pct,
            "direction": "up" if delta > 0 else ("down" if delta < 0 else "flat"),
        })

    out = pd.DataFrame(rows, columns=_SNAPSHOT_COLUMNS,
                       index=pd.Index([label for label, _ in items], name="label"))
    return out
