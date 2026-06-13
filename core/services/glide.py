"""Target-date-style return glide path for the Coast page (.planning SPEC delta:
"Rate-of-return glide path", Phase 5a).

Rate of return is the most sensitive input in a multi-decade accumulation, and a
single constant rate misrepresents reality: a portfolio that is aggressively in
equities today de-risks toward retirement (target-date-fund style), so the return
in any given year is a function of that year's asset allocation. This module makes
the **allocation glide path** the primitive and derives the per-year return from
it; the headline "rate" becomes a derived, displayed effective CAGR.

Everything here is pure, in real (inflation-adjusted) terms, and geometric. Two
asset legs only — equity vs. fixed — with the user's *current* equity weight as the
glide's start (`w0`, supplied from the live portfolio) gliding linearly to a
preset terminal weight at retirement:

    Aggressive 70% · Moderate 50% · Conservative 30%  (terminal equity)

**Downward-only clamp:** the glide never *buys* equity to reach a higher terminal
weight. If today's allocation is already at or below a preset's terminal, the path
holds flat at `w0` (you don't add risk over time to hit a target-date number).

`build_scenario` consumes these: per-age accumulation rates feed the forward walk,
the retirement-leg rate discounts the goal and drives drawdown, and the cumulative
glide growth replaces the constant-rate closed forms for the CoastFIRE target and
the required-monthly contribution.
"""

from __future__ import annotations

# Terminal equity weight (fraction) at retirement for each preset. The whole
# strategy choice reduces to "where, and how fast, do I glide down from where I am
# today" — these are the landing points.
STRATEGY_TERMINAL_EQUITY: dict[str, float] = {
    "aggressive": 0.70,
    "moderate": 0.50,
    "conservative": 0.30,
}

STRATEGY_LABELS: dict[str, str] = {
    "aggressive": "Aggressive",
    "moderate": "Moderate",
    "conservative": "Conservative",
}

DEFAULT_STRATEGY = "moderate"


def equity_weight_path(
    w0: float, w_term: float, current_age: int, retirement_age: int
) -> dict[int, float]:
    """Equity weight held during each accumulation year, `current_age … ret−1`.

    Linear from `w0` (this year) to `w_term` at the retirement boundary, so the
    return earned during year `age` uses that year's allocation. Downward-only: if
    `w0 <= w_term` the path holds flat at `w0` (never add equity to reach a higher
    terminal weight). Empty when there is no accumulation window.
    """
    current_age = int(current_age)
    retirement_age = int(retirement_age)
    n = retirement_age - current_age
    if n <= 0:
        return {}
    w0 = float(w0)
    w_term = float(w_term)
    if w0 <= w_term:
        return {current_age + k: w0 for k in range(n)}
    return {current_age + k: w0 + (w_term - w0) * (k / n) for k in range(n)}


def accumulation_rates(
    w0: float,
    w_term: float,
    r_equity: float,
    r_fixed: float,
    current_age: int,
    retirement_age: int,
) -> dict[int, float]:
    """Per-age real return during accumulation: the glide-weighted equity/fixed blend.

    `r(age) = w(age)·r_equity + (1 − w(age))·r_fixed` for each accumulation year.
    """
    r_equity = float(r_equity)
    r_fixed = float(r_fixed)
    return {
        age: w * r_equity + (1.0 - w) * r_fixed
        for age, w in equity_weight_path(w0, w_term, current_age, retirement_age).items()
    }


def retirement_rate(w_term: float, r_equity: float, r_fixed: float) -> float:
    """Drawdown-leg real return: the blend at the terminal (held) allocation.

    Used to discount the nest-egg goal and to grow balances in retirement — at
    retirement you are at `w_term` and you hold it. (A single rate can't represent
    sequence-of-returns risk in drawdown; the MC fan is the honest downside view.)
    """
    return float(w_term) * float(r_equity) + (1.0 - float(w_term)) * float(r_fixed)


def effective_cagr(rates: dict[int, float]) -> float:
    """Geometric mean real return across the accumulation years (the readout rate).

    The constant-equivalent return that produces the same growth as the glide; this
    is what the card displays in place of a user-entered rate. 0.0 for an empty map.
    """
    values = list(rates.values())
    if not values:
        return 0.0
    growth = 1.0
    for r in values:
        growth *= 1.0 + r
    return growth ** (1.0 / len(values)) - 1.0


def cumulative_growth(rates: dict[int, float], start_age: int, end_age: int) -> float:
    """∏(1 + r(age)) over `age ∈ [start_age, end_age)` — the glide growth factor.

    Replaces the constant-rate `(1+r)^n` for the CoastFIRE target: the amount needed
    at coast so growth alone reaches the goal is `goal / cumulative_growth(coast→ret)`.
    """
    g = 1.0
    for age in range(int(start_age), int(end_age)):
        g *= 1.0 + float(rates.get(age, 0.0))
    return g


def required_monthly_glide(
    start_value: float,
    current_age: int,
    coast_age: int,
    coast_target: float,
    rates: dict[int, float],
) -> float:
    """Monthly contribution to reach `coast_target` exactly by `coast_age`, glide-aware.

    Glide-path inverse of the working-phase recurrence `Bₖ₊₁ = Bₖ·(1+rₖ) + C`
    (constant annual `C`, one contribution per working year). With per-year growth
    factors `gₖ = 1 + r(current_age+k)`:

        B_n = start·∏gₖ + C·Σₖ ∏_{j>k} gⱼ        ⇒  C = (target − start·∏g) / Σ

    Solved for `C`, ÷12, clamped ≥ 0 (no negative contributions when the starting
    balance already coasts past the target). Returns 0 when there is no working
    window. Mirrors `forecast.required_monthly_contribution`, which is the constant-
    rate special case.
    """
    n = int(coast_age) - int(current_age)
    if n <= 0:
        return 0.0

    g = [1.0 + float(rates.get(int(current_age) + k, 0.0)) for k in range(n)]

    grown_start = float(start_value)
    for f in g:
        grown_start *= f

    # Annuity sum: a contribution made in working year k grows by the product of the
    # growth factors of every *later* working year (empty product = 1 for the last).
    annuity = 0.0
    for k in range(n):
        prod = 1.0
        for j in range(k + 1, n):
            prod *= g[j]
        annuity += prod
    if annuity <= 0:
        return 0.0

    annual = (float(coast_target) - grown_start) / annuity
    return max(annual, 0.0) / 12.0
