"""Social Security helpers: income-segment integration, SSA earnings-record CSV
parsing, and the legacy 2024-bend-point benefit estimator.

Moved verbatim from core/utils/functions.py (Phase 3 of the core/
restructuring); pinned by tests/test_retirement_math.py.

FUTURE HOME (Phase 4): the PIA pipeline from core/services/retirement.py
(covered_earnings, primary_insurance_amount, annual_pia_from_earnings,
social_security_income) and a consolidated resolve_pia (Phase 5).
"""
import csv
import io

import numpy as np
import pandas as pd


def calculate_social_security_benefit(earnings: pd.Series, claim_age: int) -> float:
    """
    Calculate the monthly Social Security benefit given a series of 35 years of earnings
    and the age at which benefits are first taken.

    Parameters:
    earnings (pd.Series): A series of earnings values.
    claim_age (int): The age at which benefits are first taken.

    Returns:
    float: Estimated monthly Social Security benefit.
    """
    earnings = np.array(earnings)

    if len(earnings) < 35:
        print("⚠️ Warning: Earnings contains less than 35 years of data.")
        highest_earnings = np.sort(earnings)[::-1][:len(earnings)]  # Use what we have
    else:
        highest_earnings = np.sort(earnings)[::-1][:35]  # Take top 35 values


    # Ensure claim age is within valid range (62-70)
    if claim_age < 62 or claim_age > 70:
        raise ValueError("Claim age must be between 62 and 70.")

    # Social Security Bend Points for 2024
    bend_point_1 = 1174  # 90% applies below this
    bend_point_2 = 7078  # 32% applies between this range

    # Compute AIME (Average Indexed Monthly Earnings)
    aime = highest_earnings.sum() / (35 * 12)

    # Compute PIA (Primary Insurance Amount)
    if aime <= bend_point_1:
        pia = 0.9 * aime
    elif aime <= bend_point_2:
        pia = 0.9 * bend_point_1 + 0.32 * (aime - bend_point_1)
    else:
        pia = 0.9 * bend_point_1 + 0.32 * (bend_point_2 - bend_point_1) + 0.15 * (aime - bend_point_2)

    # Adjust for claiming age
    delayed_retirement_factors = {
        62: 0.7,  # 30% reduction
        63: 0.75,
        64: 0.80,
        65: 0.866,
        66: 0.933,
        67: 1.0,  # Full PIA at FRA
        68: 1.08,
        69: 1.16,
        70: 1.24  # 8% increase per year after FRA
    }

    # Apply the adjustment factor
    benefit = pia * delayed_retirement_factors[claim_age]

    return round(benefit, 2)


def segments_to_annual_income(
    segments: list[dict],
    thru_year: int | None = None,
    growth_rate: float = 0.03,
) -> pd.Series:
    """Convert forward-filled income segments to an annual gross-income Series.

    Each segment is ``{"date": "YYYY-MM-DD", "amount": <annual gross income>}``:
    ``amount`` is the annual income *rate* in effect from ``date``, held flat
    (forward-filled) until the next segment's date. The rate is integrated at
    monthly grain so a segment starting mid-year contributes only its active
    months (e.g. a raise on 2024-07-01 splits 2024 between the old and new rate).

    Parameters
    ----------
    segments : list[dict]
        Income segments. Order-independent; empty → empty Series.
    thru_year : int | None
        If given and greater than the last segment's year, the series is extended
        through ``thru_year`` by growing the last segment's rate at ``growth_rate``
        per year (projection). If None, the series stops at the last segment year.
    growth_rate : float
        Real annual income growth applied to the projected tail (default 0.03).

    Returns
    -------
    pd.Series
        Gross income per calendar year, indexed by integer year (sorted).
    """
    if not segments:
        return pd.Series(dtype=float)

    seg = sorted(segments, key=lambda s: s["date"])
    dates = pd.to_datetime([s["date"] for s in seg])
    amounts = [float(s["amount"]) for s in seg]
    last_seg_year = int(dates[-1].year)

    # Forward-fill the annual rate at monthly grain from the first segment month
    # through December of the last segment's year, then integrate by year.
    months = pd.date_range(
        dates[0].to_period("M").to_timestamp(),
        pd.Timestamp(year=last_seg_year, month=12, day=1),
        freq="MS",
    )
    rate = pd.Series(index=months, dtype=float)
    for d, a in zip(dates, amounts):
        rate.loc[rate.index >= d.to_period("M").to_timestamp()] = a

    annual = (rate / 12.0).groupby(rate.index.year).sum()

    # Project beyond the last segment year by compounding the last rate.
    if thru_year is not None and thru_year > last_seg_year:
        base = amounts[-1]
        for i, year in enumerate(range(last_seg_year + 1, thru_year + 1), start=1):
            annual.loc[year] = base * ((1 + growth_rate) ** i)

    annual.index = annual.index.astype(int)
    return annual.sort_index()


def _money_to_float(text: str) -> float | None:
    """Parse a possibly-formatted money string ('$1,234', '1234.5') to float."""
    if text is None:
        return None
    cleaned = str(text).replace("$", "").replace(",", "").strip()
    if cleaned in ("", "-"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def ssa_earnings_to_segments(csv_text: str, basis: str = "medicare") -> list[dict]:
    """Convert an SSA *Earnings Record* CSV to forward-filled income segments.

    The Social Security Administration publishes a per-year earnings history with
    columns ``Work Year``, ``Taxed Social Security Earnings`` and ``Taxed Medicare
    Earnings`` (dollar values may carry ``$`` and thousands separators). Because the
    data is yearly, each year becomes a segment dated **Jan 1 (1/1/YYYY)**: this
    preserves the ``income_segments`` data model and lets the user later layer in
    mid-year raises without disturbing the imported history. Consecutive years with
    identical earnings collapse to a single segment, since the forward-fill carries
    the value forward (so only *changes* are stored).

    ``basis`` selects which column feeds the segment ``amount``:
    - ``"medicare"`` (default) — Medicare earnings are uncapped, so they are the
      truer gross-income figure. The retirement SS model re-caps internally
      (``covered_earnings``), so capping here would only lose information.
    - ``"social_security"`` — earnings already capped at the OASDI wage base.

    Returns segments sorted ascending by date:
    ``[{"date": "YYYY-01-01", "amount": float}, ...]``. Raises ``ValueError`` if the
    CSV has no recognisable year/earnings columns or no usable rows.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("The CSV is empty or has no header row.")

    # Locate columns by fuzzy header match (tolerant of spacing/case).
    norm = {name: name.strip().lower() for name in reader.fieldnames}
    year_col = ss_col = medicare_col = None
    for name, low in norm.items():
        if "year" in low:
            year_col = name
        elif "medicare" in low:
            medicare_col = name
        elif "social security" in low or low == "social_security":
            ss_col = name
    amount_col = medicare_col if basis == "medicare" else ss_col
    if amount_col is None:  # fall back to whichever earnings column exists
        amount_col = ss_col or medicare_col
    if year_col is None or amount_col is None:
        raise ValueError(
            "Could not find 'Work Year' and an earnings column in the CSV."
        )

    rows = []
    for row in reader:
        try:
            year = int(str(row[year_col]).strip())
        except (TypeError, ValueError):
            continue
        amount = _money_to_float(row.get(amount_col))
        if amount is None:
            continue
        rows.append((year, amount))

    if not rows:
        raise ValueError("No usable year/earnings rows were found in the CSV.")

    rows.sort(key=lambda r: r[0])  # ascending by year

    # Forward-fill de-dupe: keep a year only if its amount differs from the
    # previously kept year's amount.
    segments: list[dict] = []
    prev_amount = None
    for year, amount in rows:
        if amount == prev_amount:
            continue
        segments.append({"date": f"{year:04d}-01-01", "amount": amount})
        prev_amount = amount
    return segments
