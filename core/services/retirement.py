"""Retirement math for the Retirement page (v1 light model).

This is the *backward-from-spending* counterpart to core/services/forecast.py:
the Forecast page works **forward** to a nest-egg number; this page works
**backward** from realistic, time-varying retirement spending (go-go / slow-go /
no-go) to the same number, then projects the drawdown forward to death.

Like forecast.py, everything here is **pure and unit-testable** — no Firestore
or network access inside the math. The only data-loading helper
(`balances_by_tax_bucket`) reads the existing holdings snapshot and is kept at
the edge, mirroring forecast.py's `current_portfolio_value`.

Maturity ladder (spec §6): this module is the **v1 light** engine —
`project_portfolio`-style drawdown with time-varying spend and simple income
offsets. The full tax-aware `RetirementScenario` engine (RMDs, capital-gains
ordering, SS claim-age) is wired in later phases; its entry points are stubbed
here so the panes can be built against a stable interface.

Phase map (which phase fills each stub):
    Phase 2 (now)  — constants, crosswalk, phase classification, spend-by-phase
    Phase 4        — project_retirement, retirement_summary  (drawdown chart + BANs)
    Phase 6        — social_security_income, drawdown taxes, RMDs  (income pane)
    Phase 7        — healthcare_costs_by_age via HealthCare  (late-life / LTC)

Conventions (CLAUDE.md + project feedback memories):
  * CSP labels: fixed · investments · sinking · guilt-free · income.
  * Spend is stored/returned as positive magnitudes here; **never abs() spend in
    the UI — negate (* -1) for display** (see feedback-no-abs-on-spending).
  * Multipliers apply at the **csp-key** level (not the 5 labels, not a new txn
    mapping) — see inventory.md §4.
  * Healthcare (`medical`, `health_insurance` keys) is **excluded** from the
    generic spend pool and modelled by the dedicated HealthCare model instead,
    to avoid double-counting the rising late-life "smile" (inventory.md §7).
"""

from __future__ import annotations

import pandas as pd

from core.services.investments import fetch_cost_basis, fetch_latest_holdings

# ════════════════════════════════════════════════════════════════════════════
# 1. Defaults  (overridable per-user via UserConfig.retirement / RetirementConfig)
# ════════════════════════════════════════════════════════════════════════════

DEFAULTS: dict = {
    "retirement_age": 65,
    "death_age": 90,        # planning horizon (matches Forecast _HORIZON_AGE)
    "claim_age": 67,        # Social Security Full Retirement Age (born ≥1960)
    "slow_go_age": 75,      # go-go ends / slow-go begins
    "no_go_age": 85,        # slow-go ends / no-go begins
    "real_return": 0.05,    # inflation-adjusted; the model runs entirely in real $
    "withdrawal_rate": 0.04,  # SWR — a sanity check on the backward-derived goal
}
# Note: there is deliberately NO general inflation assumption. The whole model is
# in real (inflation-adjusted) dollars — `real_return` already nets out inflation
# for portfolio growth, and spend is held real (the phase multipliers are real),
# so a general inflation knob would only double-count. Any future need is narrow
# and purpose-specific (nominal tax-bracket/IRMAA indexing drift in Phase 6;
# healthcare *excess* inflation in Phase 7) — add a dedicated rate there, not a
# top-level inflation field. Mirrors the Forecast page, which also has none.


# ════════════════════════════════════════════════════════════════════════════
# 2. Spending-phase crosswalk  (research §"Expenses pane", retirement-notes.md)
# ════════════════════════════════════════════════════════════════════════════
#
# Editable DEFAULT table: csp-key → per-phase multiplier off the go-go baseline
# (go-go is always 1.0 — it is seeded directly from the user's latest CSP plan).
# Seeded from the research's per-category decline factors, mapped onto the app's
# actual csp keys (see emulator config / csp_from_category):
#
#     research bucket   multiplier (slow / no)   csp keys
#     ───────────────   ──────────────────────   ─────────────────────────────
#     housing (sticky)  0.95 / 0.90              mortgage, bills_utilities,
#                                                home_other, home_improvements,
#                                                television, phone, taxes
#     transport         0.85 / 0.70              auto
#     food              0.90 / 0.75              groceries
#     discretionary     0.70 / 0.45              guilt_free, vacations, airbnb,
#                                                gifts, education
#     personal/other    0.90 / 0.80              pets, therapy
#     healthcare        (special-cased)          medical, health_insurance
#                       — EXCLUDED from this pool; see HEALTHCARE_CSP_KEYS and
#                         the dedicated HealthCare model (Phase 7).
#
# The roll-up (~85% / ~75% overall) reproduces Blanchett's "retirement spending
# smile". Users edit any line or scale a whole phase; overrides live in
# RetirementConfig.phase_factors and take precedence over this table.

# Standard spending-phase keys (order matters for display).
PHASES: tuple[str, ...] = ("go_go", "slow_go", "no_go")

# csp keys excluded from the generic spend pool (modelled by HealthCare instead).
HEALTHCARE_CSP_KEYS: frozenset[str] = frozenset({"medical", "health_insurance"})

# csp_key → {slow_go, no_go} multiplier (go_go implicitly 1.0).
PHASE_FACTORS: dict[str, dict[str, float]] = {
    # Housing-like (sticky)
    "mortgage":          {"slow_go": 0.95, "no_go": 0.90},
    "bills_utilities":   {"slow_go": 0.95, "no_go": 0.90},
    "home_other":        {"slow_go": 0.95, "no_go": 0.90},
    "home_improvements": {"slow_go": 0.95, "no_go": 0.90},
    "television":        {"slow_go": 0.95, "no_go": 0.90},
    "phone":             {"slow_go": 0.95, "no_go": 0.90},
    "taxes":             {"slow_go": 0.95, "no_go": 0.90},
    "airbnb":            {"slow_go": 0.95, "no_go": 0.90},
    # Transport
    "auto":              {"slow_go": 0.85, "no_go": 0.70},
    # Food
    "groceries":         {"slow_go": 0.90, "no_go": 0.75},
    # Discretionary / travel (steep decline)
    "guilt_free":        {"slow_go": 0.70, "no_go": 0.45},
    "vacations":         {"slow_go": 0.70, "no_go": 0.45},
    "gifts":             {"slow_go": 0.70, "no_go": 0.45},
    "education":         {"slow_go": 0.70, "no_go": 0.45},
    # Personal / other
    "pets":              {"slow_go": 0.90, "no_go": 0.80},
    "therapy":           {"slow_go": 0.90, "no_go": 0.80},
}

# Fallback for any csp key not in the table above (e.g. a new key, or `income`/
# `joint_contribution`/`savings` which are not living expenses and are filtered
# out upstream): hold flat across phases.
DEFAULT_PHASE_FACTOR: dict[str, float] = {"slow_go": 1.0, "no_go": 1.0}

# csp keys that are NOT retirement living expenses (contributions / income /
# transfers) and are dropped before applying multipliers.
NON_SPEND_CSP_KEYS: frozenset[str] = frozenset(
    {"savings", "income", "joint_contribution"}
)


# ════════════════════════════════════════════════════════════════════════════
# 3. Tax / benefit constants  (centralized here per inventory.md §6; 2025 MFJ)
# ════════════════════════════════════════════════════════════════════════════
#
# Single source of truth for the tax year and rate tables. The Phase 6 income
# pane consumes these instead of the scattered/hardcoded values in
# core/utils/functions.py and the notebooks. Values sourced from
# retirement-notes.md (IRS/SSA/CMS, 2025 unless noted).

TAX_YEAR: int = 2025

# Ordinary income brackets, Married-Filing-Jointly (rate, upper edge).
ORDINARY_BRACKETS_MFJ: list[tuple[float, float]] = [
    (0.10, 23_850),
    (0.12, 96_950),
    (0.22, 206_700),
    (0.24, 394_600),
    (0.32, 501_050),
    (0.35, 751_600),
    (0.37, float("inf")),
]

# Long-term capital-gains thresholds, MFJ (rate applies up to the upper edge).
LTCG_BRACKETS_MFJ: list[tuple[float, float]] = [
    (0.00, 96_700),
    (0.15, 600_050),
    (0.20, float("inf")),
]

STANDARD_DEDUCTION_MFJ: float = 31_500.0   # acts as the bottom "0% bracket"

# Net Investment Income Tax (statutory, never indexed).
NIIT_RATE: float = 0.038
NIIT_THRESHOLD_MFJ: float = 250_000.0

# RMDs — IRS Uniform Lifetime divisors (post-2022). Start age 73 (born 1951–59)
# or 75 (born ≥1960). Roth IRA exempt for the owner.
RMD_START_AGE_BY_BIRTH_YEAR: list[tuple[int, int]] = [
    (1950, 72),   # born ≤1950
    (1959, 73),   # born 1951–1959
    (9999, 75),   # born ≥1960
]
UNIFORM_LIFETIME_DIVISORS: dict[int, float] = {
    73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
    80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2,
    87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1,
    94: 9.5, 95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4,
}

# Social Security claim-age PIA factors (FRA 67). 62→70.
SS_PIA_FACTOR_BY_CLAIM_AGE: dict[int, float] = {
    62: 0.70, 63: 0.75, 64: 0.80, 65: 0.8667, 66: 0.9333,
    67: 1.00, 68: 1.08, 69: 1.16, 70: 1.24,
}

# Social Security benefit formula (PIA from earnings) — research §"Social Security".
# Monthly AIME bend points and the 90/32/15 replacement rates; AIME averages the
# top 35 wage-indexed years (÷ 420 months). The OASDI taxable maximum caps annual
# covered earnings; self-employment (1099) covered earnings = net × 0.9235.
SS_WAGE_BASE: float = 176_100.0                       # 2025 OASDI taxable maximum
SS_BEND_POINTS: tuple[float, float] = (1_226.0, 7_391.0)   # monthly AIME
SS_REPLACEMENT_RATES: tuple[float, float, float] = (0.90, 0.32, 0.15)
SE_COVERED_FRACTION: float = 0.9235                   # net SE income → covered earnings
AIME_YEARS: int = 35                                  # SS averages the top 35 years

# Medicare standard Part B premium (annualized monthly figure).
MEDICARE_PART_B_MONTHLY_2025: float = 185.00
MEDICARE_PART_B_MONTHLY_2026: float = 202.90

# Healthcare glide defaults (annual, real $, per individual) — research §"Healthcare".
# These are *total* annual healthcare costs for each life stage, editable on the
# page. Medicare age 65 splits the pre-65 ACA bridge from Medicare + out-of-pocket.
MEDICARE_AGE: int = 65
HC_OOP_ANNUAL: float = 4_000.0      # Medigap / Part D / out-of-pocket on top of Part B
HC_ACA_ANNUAL: float = 11_000.0     # pre-65 ACA-bridge premium (unsubsidized, 60-something)
# Long-term-care spike (Genworth 2024 medians ~$71k assisted living → ~$128k nursing
# home). OFF by default — modelled as an explicit editable late-life event, not baked
# into routine spend (research recommendation).
LTC_DEFAULT_ANNUAL: float = 0.0
LTC_DEFAULT_START_AGE: int = 85
LTC_DEFAULT_YEARS: int = 3

# Planning default: share of a taxable-account withdrawal that is realized
# long-term gain (taxed at LTCG); the rest is return of basis (tax-free). We do
# not track per-lot cost basis in v1, so this is an editable assumption.
TAXABLE_GAIN_FRACTION: float = 0.5

# Annual contribution caps (2025) used only to seed the **tax-advantaged-first**
# allocation default in the balance projector (Phase 6b). v1 simplification:
# pre-tax cap = the 401(k)/403(b) employee elective-deferral limit; Roth cap =
# the IRA limit. Catch-up contributions, employer match, and the combined
# 401(k) limit are ignored — the split is just a starting point the user edits.
PRETAX_CONTRIB_CAP: float = 23_500.0   # 2025 401(k)/403(b) employee deferral
ROTH_CONTRIB_CAP: float = 7_000.0      # 2025 IRA contribution limit


# ════════════════════════════════════════════════════════════════════════════
# 4. Config resolution  (pure — merge a RetirementConfig dict over DEFAULTS)
# ════════════════════════════════════════════════════════════════════════════

def resolve_assumptions(user_cfg: dict | None) -> dict:
    """Resolve a user's retirement assumptions from their config.

    Demographics — `retirement_age`, `death_age`, `claim_age`, and `birth_year`
    (derived from `birth_date`) — come from the **top-level Profile fields**
    (UserConfig.birth_date + ages); Profile is the single source of truth. Scenario
    knobs — `slow_go_age`, `no_go_age`, `real_return`, `withdrawal_rate` — come from
    `UserConfig.retirement`. Anything None/absent falls back to DEFAULTS, so a user
    who has touched neither page gets a complete, sensible set. Returns a flat dict
    with every key in DEFAULTS plus `birth_year` (None if no birth date saved).

    `user_cfg` is the full deserialized user config dict from config-store.
    Pure: callers pass the dict; no Firestore access.
    """
    cfg = user_cfg or {}
    ret = cfg.get("retirement") or {}

    def pick(key, src):
        v = src.get(key)
        return v if v is not None else DEFAULTS[key]

    resolved = {
        # Demographics — owned by Profile (top-level fields)
        "retirement_age": pick("retirement_age", cfg),
        "death_age": pick("death_age", cfg),
        "claim_age": pick("claim_age", cfg),
        # Scenario knobs — owned by the Retirement page (UserConfig.retirement)
        "slow_go_age": pick("slow_go_age", ret),
        "no_go_age": pick("no_go_age", ret),
        "real_return": pick("real_return", ret),
        "withdrawal_rate": pick("withdrawal_rate", ret),
    }
    birth_date = cfg.get("birth_date")
    resolved["birth_year"] = int(str(birth_date)[:4]) if birth_date else None
    return resolved


def resolve_phase_factors(retirement_cfg: dict | None) -> dict[str, dict[str, float]]:
    """User overrides (RetirementConfig.phase_factors) layered over PHASE_FACTORS.

    Per-key, per-phase merge so a user can override a single line (e.g. set
    housing no_go to 0 after the mortgage is paid off) without restating the
    whole table.
    """
    merged = {k: dict(v) for k, v in PHASE_FACTORS.items()}
    overrides = (retirement_cfg or {}).get("phase_factors") or {}
    for key, factors in overrides.items():
        merged.setdefault(key, dict(DEFAULT_PHASE_FACTOR)).update(factors or {})
    return merged


# ════════════════════════════════════════════════════════════════════════════
# 5. Spending phases  (Phase 2 deliverable — implemented)
# ════════════════════════════════════════════════════════════════════════════

def phase_for_age(age: int, slow_go_age: int, no_go_age: int) -> str:
    """Classify an age into 'go_go' | 'slow_go' | 'no_go'.

    go-go  = age < slow_go_age
    slow-go = slow_go_age ≤ age < no_go_age
    no-go  = age ≥ no_go_age
    """
    if age < int(slow_go_age):
        return "go_go"
    if age < int(no_go_age):
        return "slow_go"
    return "no_go"


def _factor_for(key: str, phase: str, phase_factors: dict[str, dict[str, float]]) -> float:
    """Multiplier for one csp key in one phase (go_go is always 1.0)."""
    if phase == "go_go":
        return 1.0
    return phase_factors.get(key, DEFAULT_PHASE_FACTOR).get(phase, 1.0)


def annual_spend_by_phase(
    active_plan: dict | None,
    csp_labels: dict | None,
    phase_factors: dict[str, dict[str, float]] | None = None,
    *,
    exclude_keys: frozenset[str] = HEALTHCARE_CSP_KEYS,
) -> dict[str, dict]:
    """Seed go-go from the latest CSP plan, then derive slow-go / no-go.

    The plan is keyed at the **csp-key** level (mortgage, groceries, vacations…)
    with monthly amounts. For each spend key (dropping NON_SPEND_CSP_KEYS and
    `exclude_keys`, i.e. healthcare) we multiply the go-go baseline by the
    per-phase factor.

    Returns:
        {
          "go_go":  {"total": <annual $>, "by_key": {key: annual $}},
          "slow_go": {...},
          "no_go":  {...},
        }

    Magnitudes are positive (annual dollars). The Expenses pane negates for
    display. Healthcare keys are intentionally absent — they are added back by
    the HealthCare model in the projection (Phase 7), not scaled here.
    """
    phase_factors = phase_factors if phase_factors is not None else PHASE_FACTORS
    plan = active_plan or {}
    labels = csp_labels or {}

    # Living-expense keys only: drop contributions/income/transfers, healthcare,
    # and any key whose CSP label is `investments` or `income`.
    drop = set(NON_SPEND_CSP_KEYS) | set(exclude_keys)
    spend_keys = {
        key: float(amount)
        for key, amount in plan.items()
        if key not in drop
        and labels.get(key) not in {"investments", "income"}
    }

    out: dict[str, dict] = {}
    for phase in PHASES:
        by_key = {
            key: monthly * 12.0 * _factor_for(key, phase, phase_factors)
            for key, monthly in spend_keys.items()
        }
        out[phase] = {"total": sum(by_key.values()), "by_key": by_key}
    return out


# ── Household shared expenses → per-individual share (only individuals "retire") ──
#
# Shared household costs are paid from the joint account, so they never appear in a
# member's own CSP plan — the member's plan records only the *transfer* that funds
# them, as a `joint_contribution` line. To plan one individual's retirement we fold
# in their proportional share of the real household expenses, distributed by how
# much each member contributes (their `joint_contribution`), not an even split: the
# person who funds 60% of the joint account bears 60% of the shared costs.

def household_expense_share(member_contributions: dict[str, float], uid: str) -> float:
    """Fraction of shared household expenses borne by member `uid`.

    = the member's joint contribution ÷ the sum of all members' joint
    contributions. Contributions come from each member's *latest* CSP plan — a
    single scalar projected forward (the most recent plan is the representative
    one for a forward-looking projection), deliberately not a time average. A
    member who contributes nothing bears no share (0.0); a zero household total
    likewise yields 0.0. Pure.
    """
    total = sum(max(float(v), 0.0) for v in member_contributions.values())
    if total <= 0:
        return 0.0
    return max(float(member_contributions.get(uid, 0.0)), 0.0) / total


def merge_household_expenses(
    individual_plan: dict | None,
    household_plan: dict | None,
    share: float,
) -> dict:
    """Fold an individual's `share` of the household CSP plan into their own plan.

    Returns a new monthly {csp_key: amount} map: the individual's own plan plus
    `share` × the household amount for each key. The individual's
    `joint_contribution` line (the transfer that *funded* the share) and the
    household's own income/contribution lines ride along unchanged — downstream
    `annual_spend_by_phase` drops them — so replacing the transfer with the real
    expense share does not double-count. With no household (share 0 or empty plan)
    the individual's plan is returned unchanged. Pure.

    Linearity note: because phase multipliers are per-key, running
    `annual_spend_by_phase` on this merged plan equals scaling the household's
    spend-by-phase by `share` and adding it to the individual's — they share the
    same per-key factor — so merging here keeps one editable row per category
    without changing the math.
    """
    merged = {k: float(v) for k, v in (individual_plan or {}).items()}
    if share:
        for key, amount in (household_plan or {}).items():
            merged[key] = merged.get(key, 0.0) + share * float(amount)
    return merged


# ════════════════════════════════════════════════════════════════════════════
# 6. Data-derived inputs  (edge — reads the holdings snapshot)
# ════════════════════════════════════════════════════════════════════════════

# Account-type labels (from investment_accounts) that are tax-deferred /
# tax-free, mirroring investments.reconstruct_portfolio_history.
TRAD_TYPES: frozenset[str] = frozenset(
    {"IRA", "401k", "403b", "457b", "SEP IRA", "SIMPLE IRA"}
)
ROTH_TYPES: frozenset[str] = frozenset({"Roth IRA", "Roth 401k"})


def balances_by_tax_bucket(uid: str, investment_accounts: dict | None) -> dict:
    """Aggregate current holdings into {taxable, trad, roth, total}.

    Splits the holdings snapshot by the account-type label in
    investment_accounts (last-4 → label). Unlabeled accounts default to taxable,
    matching reconstruct_portfolio_history. This is the v1 input the full
    tax-aware engine (Phase 6) needs; the light drawdown only needs `total`
    (equivalent to forecast.current_portfolio_value).

    Not pure (reads Firestore via fetch_latest_holdings) — kept thin and at the
    edge so the projection math stays pure and testable.
    """
    accounts = investment_accounts or {}
    buckets = {"taxable": 0.0, "trad": 0.0, "roth": 0.0}
    for h in fetch_latest_holdings(uid):
        acct = h.get("account_number", "")
        value = float(h.get("total_value", 0.0))
        label = accounts.get(acct, "Taxable")
        if label in TRAD_TYPES:
            buckets["trad"] += value
        elif label in ROTH_TYPES:
            buckets["roth"] += value
        else:
            buckets["taxable"] += value
    buckets["total"] = sum(buckets.values())
    return buckets


def gain_fraction_from_cost_basis(
    cost_basis: dict, investment_accounts: dict | None
) -> float | None:
    """Realized-gain fraction implied by a cost-basis report.  (pure)

    (Σ market value − Σ cost) ÷ Σ market value over **taxable-labeled accounts
    only** — the share of a taxable sale that is long-term gain (taxed at LTCG);
    the rest is return of basis (tax-free). Tax-deferred accounts (TRAD_TYPES)
    are ordinary regardless of basis and Roth (ROTH_TYPES) is tax-free, so both
    are excluded. Unlabeled accounts default to taxable, matching
    balances_by_tax_bucket. Returns None when there is no taxable cost-basis data
    (the caller then keeps the manual TAXABLE_GAIN_FRACTION default).

    `cost_basis` is the {account: {market_value, cost, gain}} map from
    fetch_cost_basis.
    """
    accounts = investment_accounts or {}
    tot_mv = tot_cost = 0.0
    for acct, vals in (cost_basis or {}).items():
        label = accounts.get(acct, "Taxable")
        if label in TRAD_TYPES or label in ROTH_TYPES:
            continue
        tot_mv += float(vals.get("market_value", 0.0))
        tot_cost += float(vals.get("cost", 0.0))
    if tot_mv <= 0:
        return None
    return (tot_mv - tot_cost) / tot_mv


def taxable_gain_fraction(uid: str, investment_accounts: dict | None) -> float | None:
    """Derive the taxable gain fraction from the user's uploaded cost-basis report.

    Edge wrapper around the pure gain_fraction_from_cost_basis: reads the
    cost_basis snapshot (Firestore) then computes. Returns None if the user has
    not uploaded a cost-basis CSV (or none of it is in taxable accounts), so the
    income pane falls back to the editable TAXABLE_GAIN_FRACTION default. Kept
    thin and at the edge, mirroring balances_by_tax_bucket.
    """
    return gain_fraction_from_cost_basis(fetch_cost_basis(uid), investment_accounts)


# ── At-retirement balance projector  (Phase 6b — pure) ────────────────────────────

def default_contribution_allocation(
    annual_contribution: float,
    pretax_cap: float = PRETAX_CONTRIB_CAP,
    roth_cap: float = ROTH_CONTRIB_CAP,
) -> dict:
    """Split an annual contribution across buckets **tax-advantaged-first**.

    Fill pre-tax (trad) up to its cap, then Roth up to its cap, then the
    remainder to taxable. Returns {taxable, trad, roth} annual dollar amounts
    summing to `annual_contribution`. This is just the editable default the
    projector seeds; the user can re-weight any bucket.
    """
    c = max(float(annual_contribution), 0.0)
    trad = min(c, float(pretax_cap))
    roth = min(c - trad, float(roth_cap))
    taxable = max(c - trad - roth, 0.0)
    return {"taxable": taxable, "trad": trad, "roth": roth}


def project_balances_to_retirement(
    current_balances: dict,
    annual_allocation: dict,
    years: int,
    real_return: float,
) -> dict:
    """Grow each tax bucket from today to retirement.  (pure)

    Compounds every bucket forward `years` years, adding that bucket's annual
    contribution each year, using the **same working-phase recurrence as
    Forecast's `project_portfolio`** — `V' = V·(1+r) + contribution` — so the
    aggregate of the three buckets reconciles with the Forecast page given the
    same start, total contribution, and real return.

    `current_balances` / `annual_allocation` are {taxable, trad, roth} dicts.
    Returns {taxable, trad, roth, total} projected at retirement. With years ≤ 0
    (already at/after retirement) the projection is just the current balances.
    """
    r = float(real_return)
    n = max(int(years), 0)
    out: dict = {}
    for bucket in ("taxable", "trad", "roth"):
        v = float(current_balances.get(bucket, 0.0))
        a = float(annual_allocation.get(bucket, 0.0))
        for _ in range(n):
            v = v * (1 + r) + a
        out[bucket] = v
    out["total"] = out["taxable"] + out["trad"] + out["roth"]
    return out


# ════════════════════════════════════════════════════════════════════════════
# 7. Core projection + summary  (Phase 4 — STUBBED; interface frozen here)
# ════════════════════════════════════════════════════════════════════════════

def _as_age_lookup(src: pd.Series | dict | None) -> dict:
    """Normalize an income/healthcare-by-age input to a plain {age: float} dict."""
    if src is None:
        return {}
    if isinstance(src, pd.Series):
        return {int(k): float(v) for k, v in src.to_dict().items()}
    return {int(k): float(v) for k, v in src.items()}


def project_retirement(
    start_value: float,
    retirement_age: int,
    death_age: int,
    spend_by_phase: dict[str, float],
    slow_go_age: int,
    no_go_age: int,
    real_return: float,
    income_by_age: pd.Series | dict | None = None,
    healthcare_by_age: pd.Series | dict | None = None,
) -> pd.DataFrame:
    """Year-by-year drawdown from retirement_age to death_age.  (v1 light model)

    The backward-from-spending engine. Convention: each year the portfolio funds
    that year's net living cost up front, and the remainder grows one year at
    `real_return` (`total' = (total − withdrawal)·(1+r)`). `total` recorded for
    an age is the balance **entering** that age-year (before that year's draw).

    Net withdrawal = phase living spend + healthcare − income. In v1 (Phase 4)
    `income_by_age` and `healthcare_by_age` are absent (Social Security lands in
    Phase 6, the HealthCare model in Phase 7); the signature already accepts them
    so later phases plug in without changing callers. A negative net (income >
    cost) is reinvested (grows the balance) rather than clamped.

    Returns a DataFrame indexed by `age` with columns:
        total, spend, healthcare, income, withdrawal, phase

    All flows are positive magnitudes (income is stored positive in its own
    column; it enters `withdrawal` as a subtraction). The chart negates
    spend/withdrawal for display (never abs()).
    """
    retirement_age = int(retirement_age)
    death_age = int(death_age)
    r = float(real_return)
    income = _as_age_lookup(income_by_age)
    health = _as_age_lookup(healthcare_by_age)

    total = float(start_value)
    rows = []
    for age in range(retirement_age, death_age + 1):
        phase = phase_for_age(age, slow_go_age, no_go_age)
        spend = float(spend_by_phase.get(phase, 0.0))
        hc = health.get(age, 0.0)
        inc = income.get(age, 0.0)
        withdrawal = spend + hc - inc

        rows.append({
            "age": age,
            "total": total,
            "spend": spend,
            "healthcare": hc,
            "income": inc,
            "withdrawal": withdrawal,
            "phase": phase,
        })
        total = (total - withdrawal) * (1 + r)

    return pd.DataFrame(rows).set_index("age")


def nest_egg_goal(
    projection_df: pd.DataFrame,
    real_return: float,
) -> float:
    """Portfolio needed AT retirement to exactly fund the projected draw stream.

    Present value (at the first/retirement age, discounted at `real_return`) of
    the per-year net withdrawals — the start_value at which the account funds
    every year's net cost and depletes to ~0 at death. This is the page's
    *backward-from-spending* answer, independent of current holdings.
    """
    r = float(real_return)
    ages = projection_df.index.tolist()
    if not ages:
        return 0.0
    base = ages[0]
    return float(sum(
        projection_df.loc[age, "withdrawal"] / ((1 + r) ** (age - base))
        for age in ages
    ))


def retirement_summary(
    projection_df: pd.DataFrame,
    real_return: float,
    withdrawal_rate: float = DEFAULTS["withdrawal_rate"],
) -> dict:
    """Headline numbers for the BANs.

    NOTE (Phase 4 refinement of the Phase-2 stub): the signature takes
    `real_return` (needed for the nest-egg present value) instead of
    `spend_by_phase`; everything else is derived from `projection_df`, which
    already carries the per-age spend/healthcare/withdrawal/total columns.

    Returns:
        nest_egg_goal        — start value that funds the stream to death (PV)
        balance_at_death     — projected balance entering death_age
        avg_annual_spend     — mean annual gross living cost (spend + healthcare)
        first_year_drawdown  — net withdrawal in the first retirement year
        peak_drawdown        — largest single-year net withdrawal
        funded_through_age   — last age the balance stays ≥ 0 (or death_age)
        swr_sanity_goal      — first-year gross spend ÷ withdrawal_rate (4%-rule
                               cross-check on the backward-derived goal)
    """
    if projection_df.empty:
        return {
            "nest_egg_goal": 0.0, "balance_at_death": 0.0, "avg_annual_spend": 0.0,
            "first_year_drawdown": 0.0, "peak_drawdown": 0.0,
            "funded_through_age": None, "swr_sanity_goal": 0.0,
        }

    swr = float(withdrawal_rate)
    ages = projection_df.index.tolist()
    gross = projection_df["spend"] + projection_df["healthcare"]

    funded = projection_df[projection_df["total"] >= 0]
    funded_through_age = int(funded.index.max()) if not funded.empty else None

    first_year_drawdown = float(projection_df["withdrawal"].iloc[0])

    return {
        "nest_egg_goal": nest_egg_goal(projection_df, real_return),
        "balance_at_death": float(projection_df["total"].iloc[-1]),
        "avg_annual_spend": float(gross.mean()),
        "first_year_drawdown": first_year_drawdown,
        "peak_drawdown": float(projection_df["withdrawal"].max()),
        "funded_through_age": funded_through_age,
        "swr_sanity_goal": (float(gross.iloc[0]) / swr) if swr else 0.0,
    }


# ════════════════════════════════════════════════════════════════════════════
# 8. Income — Social Security, RMDs, taxes, tax-aware drawdown  (Phase 6)
# ════════════════════════════════════════════════════════════════════════════
#
# Planning-grade and retirement-scoped: the tax math lives here (using the §3
# constants) rather than editing core.utils.functions.calculate_married_joint_tax
# (2024, no LTCG) — that function backs the household models (household.py) and
# changing it would ripple. Flat-tax decision (Phase 0): brackets + LTCG +
# standard deduction; no NIIT / SS-taxation / state. Social Security is treated
# as untaxed in v1 (a documented simplification).

def rmd_start_age(birth_year: int) -> int:
    """RMD start age from birth year (SECURE 2.0): 72 / 73 / 75."""
    for cutoff, age in RMD_START_AGE_BY_BIRTH_YEAR:
        if int(birth_year) <= cutoff:
            return age
    return 75


def rmd_divisor(age: int) -> float | None:
    """IRS Uniform Lifetime divisor for an age, or None if outside the table."""
    return UNIFORM_LIFETIME_DIVISORS.get(int(age))


def required_min_distribution(trad_balance: float, age: int, start_age: int) -> float:
    """Forced annual RMD from the tax-deferred balance (0 before `start_age`).

    RMD = prior-year-end balance ÷ Uniform-Lifetime divisor. Past the table's top
    age, fall back to the oldest divisor (conservative — slightly understates).
    Roth is exempt and never passed here.
    """
    if trad_balance <= 0 or int(age) < int(start_age):
        return 0.0
    div = rmd_divisor(age) or UNIFORM_LIFETIME_DIVISORS[max(UNIFORM_LIFETIME_DIVISORS)]
    return trad_balance / div


# ── Social Security from income (Phase 6c — derive the PIA, pure) ─────────────────

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


# ── Flat planning-grade taxes (2025 MFJ; §3 constants) ───────────────────────────

def _progressive_tax(amount: float, brackets: list[tuple[float, float]]) -> float:
    """Tax on `amount` through a cumulative (rate, upper-edge) bracket table."""
    tax = 0.0
    lower = 0.0
    for rate, upper in brackets:
        if amount <= lower:
            break
        tax += (min(amount, upper) - lower) * rate
        lower = upper
    return tax


def annual_tax(ordinary_income: float, ltcg_gain: float = 0.0) -> float:
    """Federal tax on a year's ordinary income + long-term capital gain (MFJ).

    Ordinary income is taxed through the ordinary brackets after the standard
    deduction. The LTCG **stacks on top** of taxable ordinary income (its 0/15/20%
    band depends on total taxable income); any standard deduction unused by
    ordinary income shelters gain first.
    """
    ordinary_income = max(float(ordinary_income), 0.0)
    ltcg_gain = max(float(ltcg_gain), 0.0)
    std = STANDARD_DEDUCTION_MFJ

    ord_taxable = max(ordinary_income - std, 0.0)
    unused_deduction = max(std - ordinary_income, 0.0)
    gain_taxable = max(ltcg_gain - unused_deduction, 0.0)

    tax = _progressive_tax(ord_taxable, ORDINARY_BRACKETS_MFJ)

    # LTCG occupies the band from ord_taxable upward.
    lower = ord_taxable
    remaining = gain_taxable
    for rate, upper in LTCG_BRACKETS_MFJ:
        if remaining <= 0:
            break
        if lower < upper:
            band = min(upper, lower + remaining) - lower
            tax += band * rate
            lower += band
            remaining -= band
    return tax


# ── Tax-aware drawdown engine (v1.5) ─────────────────────────────────────────────

def _allocate_withdrawals(target, taxable, trad, roth, rmd):
    """Fund `target` after RMD, in order taxable → trad → roth.

    The RMD is withdrawn from `trad` first (forced, taxed as ordinary); if it
    exceeds the cash needed, the excess is returned to be reinvested in taxable.
    Returns (w_taxable, w_trad, w_roth, excess_rmd, shortfall).
    """
    w_trad = min(rmd, trad)
    excess_rmd = max(w_trad - target, 0.0)
    remaining = max(target - w_trad, 0.0)

    w_taxable = min(remaining, taxable)
    remaining -= w_taxable

    extra_trad = min(remaining, trad - w_trad)
    w_trad += extra_trad
    remaining -= extra_trad

    w_roth = min(remaining, roth)
    remaining -= w_roth

    return w_taxable, w_trad, w_roth, excess_rmd, remaining


def project_retirement_taxaware(
    balances: dict,
    retirement_age: int,
    death_age: int,
    spend_by_phase: dict[str, float],
    slow_go_age: int,
    no_go_age: int,
    real_return: float,
    ss_by_age: pd.Series | dict | None = None,
    healthcare_by_age: pd.Series | dict | None = None,
    rmd_start: int = 75,
    taxable_gain_fraction: float = TAXABLE_GAIN_FRACTION,
) -> pd.DataFrame:
    """Year-by-year tax-aware drawdown across taxable / trad / roth buckets.

    Each year: cover the net living need (phase spend + healthcare − Social
    Security) with after-tax cash, withdrawing taxable → trad → roth, RMDs forced
    from trad at `rmd_start`. Taxes (ordinary on trad, LTCG on the taxable gain
    fraction, Roth tax-free) are solved with a short fixed-point loop since the
    draw must cover spending *and* the tax it generates. Surviving balances grow
    at `real_return`.

    Returns a DataFrame indexed by `age` with columns:
        total, taxable, trad, roth, spend, healthcare, income (SS), rmd,
        withdrawal, w_taxable, w_trad, w_roth, ordinary_income, ltcg_gain,
        tax, net_spend, phase

    `total` is the portfolio entering each age-year (before that year's draw).
    All flows are positive magnitudes; the UI negates draws for display.
    """
    r = float(real_return)
    ss = _as_age_lookup(ss_by_age)
    health = _as_age_lookup(healthcare_by_age)
    gain_frac = float(taxable_gain_fraction)

    taxable = float(balances.get("taxable", 0.0))
    trad = float(balances.get("trad", 0.0))
    roth = float(balances.get("roth", 0.0))

    rows = []
    for age in range(int(retirement_age), int(death_age) + 1):
        phase = phase_for_age(age, slow_go_age, no_go_age)
        spend = float(spend_by_phase.get(phase, 0.0))
        hc = health.get(age, 0.0)
        income = ss.get(age, 0.0)
        total_before = taxable + trad + roth

        cash_need = max(spend + hc - income, 0.0)   # SS untaxed offset (v1)
        rmd = required_min_distribution(trad, age, rmd_start)

        # Fixed-point on tax: the draw must fund cash_need + the tax it triggers.
        tax = 0.0
        w_taxable = w_trad = w_roth = excess_rmd = 0.0
        for _ in range(8):
            target = cash_need + tax
            w_taxable, w_trad, w_roth, excess_rmd, _short = _allocate_withdrawals(
                target, taxable, trad, roth, rmd)
            ordinary_income = w_trad                 # all trad draws are ordinary
            ltcg_gain = w_taxable * gain_frac
            new_tax = annual_tax(ordinary_income, ltcg_gain)
            if abs(new_tax - tax) < 1.0:
                tax = new_tax
                break
            tax = new_tax

        withdrawal = w_taxable + w_trad + w_roth
        net_spend = withdrawal - tax - excess_rmd + income   # after-tax living cash

        rows.append({
            "age": age, "total": total_before,
            "taxable": taxable, "trad": trad, "roth": roth,
            "spend": spend, "healthcare": hc, "income": income, "rmd": rmd,
            "withdrawal": withdrawal, "w_taxable": w_taxable, "w_trad": w_trad,
            "w_roth": w_roth, "ordinary_income": ordinary_income,
            "ltcg_gain": ltcg_gain, "tax": tax, "net_spend": net_spend,
            "phase": phase,
        })

        # Apply draws; reinvest any forced-RMD excess into taxable; then grow.
        taxable = max(taxable - w_taxable + excess_rmd, 0.0)
        trad = max(trad - w_trad, 0.0)
        roth = max(roth - w_roth, 0.0)
        taxable *= (1 + r)
        trad *= (1 + r)
        roth *= (1 + r)

    return pd.DataFrame(rows).set_index("age")


def healthcare_costs_by_age(
    retirement_age: int,
    death_age: int,
    *,
    oop_annual: float = HC_OOP_ANNUAL,
    aca_annual: float = HC_ACA_ANNUAL,
    medicare_annual: float | None = None,
    medicare_age: int = MEDICARE_AGE,
    ltc_annual: float = LTC_DEFAULT_ANNUAL,
    ltc_start_age: int = LTC_DEFAULT_START_AGE,
    ltc_years: int = LTC_DEFAULT_YEARS,
) -> pd.Series:
    """Annual healthcare cost by age (real $).  (Phase 7 — pure, age-indexed)

    Reproduces the working → ACA-bridge → Medicare → late-life glide of
    core.models.healthcare, but **indexed on age** (fixing that model's
    age-vs-year bug, inventory.md §6) and grounded in the research $:

      * before `medicare_age` (early retirement): ACA-bridge premium + baseline OOP
      * `medicare_age`+: Medicare Part B + baseline OOP
      * an editable **LTC spike** of `ltc_annual` for `ltc_years` from `ltc_start_age`

    This is the single source of the late-life "smile" upturn: the `medical` /
    `health_insurance` csp keys are excluded from the generic spend pool upstream
    (HEALTHCARE_CSP_KEYS), so adding this series is additive, not double-counted.
    `medicare_annual` defaults to the standard Part B premium (×12). Returns a
    Series indexed by age (retirement_age … death_age).
    """
    medicare_annual = (MEDICARE_PART_B_MONTHLY_2025 * 12.0
                       if medicare_annual is None else float(medicare_annual))
    ltc_start = int(ltc_start_age)
    ltc_end = ltc_start + max(int(ltc_years), 0)
    spike = float(ltc_annual or 0.0)

    out: dict[int, float] = {}
    for age in range(int(retirement_age), int(death_age) + 1):
        cost = float(oop_annual) + (medicare_annual if age >= int(medicare_age)
                                    else float(aca_annual))
        if spike > 0 and ltc_start <= age < ltc_end:
            cost += spike
        out[age] = cost
    return pd.Series(out)
