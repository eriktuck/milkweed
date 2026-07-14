"""Social Security: the 2025 PIA pipeline, income-segment integration, SSA
earnings-record CSV parsing, and the legacy 2024-bend-point benefit estimator.

The segment/CSV helpers and `calculate_social_security_benefit` moved verbatim
from core/utils/functions.py (Phase 3 of the core/ restructuring); the PIA
pipeline (covered_earnings → AIME → primary_insurance_amount →
social_security_income, 2025 constants) moved verbatim from
core/services/retirement.py (Phase 4). Pinned by tests/test_retirement_math.py.

`resolve_pia` (Phase 5) is the single config→PIA resolution path — it replaced
the duplicated logic in scenario's `_resolve_pia` and inside
drawdown.default_nest_egg_goal. Later feature work: pad_earnings_to_35 (see
docs/porting-backlog.md once created).
"""
import csv
import io

import numpy as np
import pandas as pd

# ── 2025 benefit-formula constants — research §"Social Security" ─────────────────
#
# Monthly AIME bend points and the 90/32/15 replacement rates; AIME averages the
# top 35 wage-indexed years (÷ 420 months). The OASDI taxable maximum caps annual
# covered earnings; self-employment (1099) covered earnings = net × 0.9235.
SS_WAGE_BASE: float = 176_100.0                       # 2025 OASDI taxable maximum
SS_BEND_POINTS: tuple[float, float] = (1_226.0, 7_391.0)   # monthly AIME
SS_REPLACEMENT_RATES: tuple[float, float, float] = (0.90, 0.32, 0.15)
SE_COVERED_FRACTION: float = 0.9235                   # net SE income → covered earnings
AIME_YEARS: int = 35                                  # SS averages the top 35 years

# Social Security claim-age PIA factors (FRA 67). 62→70.
SS_PIA_FACTOR_BY_CLAIM_AGE: dict[int, float] = {
    62: 0.70, 63: 0.75, 64: 0.80, 65: 0.8667, 66: 0.9333,
    67: 1.00, 68: 1.08, 69: 1.16, 70: 1.24,
}


def covered_earnings(gross_income: float, employment_type: str = "W2") -> float:
    """Annual Social-Security-covered earnings from gross income.

    W2: gross wages, capped at the OASDI taxable maximum. 1099 (self-employed):
    net SE income × 0.9235, then capped. 'mixed': treated as W2 (the 0.9235
    haircut only applies to a self-employment portion we don't separately
    capture) — a documented v1 simplification.
    """
    g = max(float(gross_income or 0.0), 0.0)
    if str(employment_type).lower() in {"1099", "self-employed", "se"}:
        g *= SE_COVERED_FRACTION
    return min(g, SS_WAGE_BASE)


def primary_insurance_amount(aime_monthly: float) -> float:
    """Monthly PIA from AIME via the bend-point formula (90% / 32% / 15%)."""
    b1, b2 = SS_BEND_POINTS
    r1, r2, r3 = SS_REPLACEMENT_RATES
    a = max(float(aime_monthly), 0.0)
    pia = r1 * min(a, b1)
    if a > b1:
        pia += r2 * (min(a, b2) - b1)
    if a > b2:
        pia += r3 * (a - b2)
    return pia


def estimate_annual_pia_from_income(
    gross_income: float,
    employment_type: str = "W2",
    career_years: int = AIME_YEARS,
) -> float:
    """Annual Social Security benefit **at FRA (67)** estimated from income.

    Assumes a roughly constant `gross_income` earned over `career_years`; years
    short of 35 average in $0 (AIME always divides by 35), so a shorter career
    lowers the benefit. Pipeline: covered earnings → AIME (top-35 monthly
    average) → PIA → ×12. The **claim-age factor is applied downstream** by
    `social_security_income`, so this returns the FRA figure that seeds the
    editable `SS benefit at FRA` input (the SSA-statement override).
    """
    earnings = covered_earnings(gross_income, employment_type)
    years = min(max(int(career_years), 0), AIME_YEARS)
    aime = earnings * years / (AIME_YEARS * 12)
    return primary_insurance_amount(aime) * 12.0


def annual_pia_from_earnings(earnings, employment_type: str = "W2") -> float:
    """Annual Social Security benefit **at FRA (67)** from a real per-year earnings
    series (e.g. the Profile income history via segments_to_annual_income).

    The accurate counterpart to `estimate_annual_pia_from_income`: instead of
    assuming a constant income, it uses actual annual earnings. Each year is capped
    at the OASDI wage base via `covered_earnings`; the top 35 form the AIME (always
    divided by 35×12, so a short career averages in $0). Returns the FRA figure; the
    claim-age factor is applied downstream by `social_security_income`.

    `earnings` is any iterable of annual gross amounts (pd.Series or list).
    """
    covered = [covered_earnings(v, employment_type) for v in earnings
               if v is not None and float(v) > 0]
    if not covered:
        return 0.0
    top = sorted(covered, reverse=True)[:AIME_YEARS]
    aime = sum(top) / (AIME_YEARS * 12)
    return primary_insurance_amount(aime) * 12.0


def social_security_income(
    pia_annual: float,
    claim_age: int,
    retirement_age: int,
    death_age: int,
) -> pd.Series:
    """Annual Social Security income by age (0 before the later of claim age and
    retirement age), with the claim-age factor applied to the PIA.

    `pia_annual` is the annual benefit at Full Retirement Age (67). The full
    earnings-based PIA derivation (`calculate_social_security_benefit`) is left
    for when working-years earnings are captured; here the user supplies their
    SSA estimate and the claim-age slider scales it.
    """
    factor = SS_PIA_FACTOR_BY_CLAIM_AGE.get(int(claim_age), 1.0)
    benefit = float(pia_annual or 0.0) * factor
    start = max(int(claim_age), int(retirement_age))
    ages = range(int(retirement_age), int(death_age) + 1)
    return pd.Series({age: (benefit if age >= start else 0.0) for age in ages})


def resolve_pia(
    user_cfg: dict | None,
    active_plan: dict | None,
    csp_labels: dict | None,
) -> float:
    """Annual Social Security benefit at FRA, preferring real earnings history.

    The single config→PIA resolution path (consolidated in Phase 5 from
    scenario's `_resolve_pia` and drawdown.default_nest_egg_goal): use the
    Profile income segments via `annual_pia_from_earnings` when present, else
    estimate from the CSP `income` lines assuming a constant career, else 0 —
    so the slider-driven goal and the config-default goal always share the
    identical Social Security basis.
    """
    segments = (user_cfg or {}).get("income_segments")
    if segments:
        return annual_pia_from_earnings(segments_to_annual_income(segments))

    labels = csp_labels or {}
    annual_income = 12.0 * sum(
        float(amt)
        for key, amt in (active_plan or {}).items()
        if labels.get(key) == "income"
    )
    return estimate_annual_pia_from_income(annual_income) if annual_income else 0.0


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
