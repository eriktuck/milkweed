"""Retirement-planning defaults and config resolution.

Moved verbatim from core/services/retirement.py (Phase 4 of the core/
restructuring); pinned by tests/test_retirement_math.py.

Owns the DEFAULTS table, the spending-phase crosswalk (go-go / slow-go / no-go
multipliers), config resolution (`resolve_assumptions`, `resolve_phase_factors`),
the CSP-plan → spend-by-phase derivation, the household expense-share fold-in,
and the at-retirement balance projector defaults. Pure — callers pass
deserialized config dicts; no Firestore access.

Conventions (CLAUDE.md + project feedback memories):
  * CSP labels: fixed · investments · sinking · guilt-free · income.
  * Spend is stored/returned as positive magnitudes here; **never abs() spend in
    the UI — negate (* -1) for display** (see feedback-no-abs-on-spending).
  * Multipliers apply at the **csp-key** level (not the 5 labels, not a new txn
    mapping).
  * Healthcare (`medical`, `health_insurance` keys) is **excluded** from the
    generic spend pool and modelled by core/domain/healthcare.py instead, to
    avoid double-counting the rising late-life "smile".
"""

from __future__ import annotations

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
# and purpose-specific (nominal tax-bracket/IRMAA indexing drift; healthcare
# *excess* inflation) — add a dedicated rate there, not a top-level inflation
# field.


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
#                         core/domain/healthcare.py.
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

# Annual contribution caps (2025) used only to seed the **tax-advantaged-first**
# allocation default in the balance projector. v1 simplification: pre-tax cap =
# the 401(k)/403(b) employee elective-deferral limit; Roth cap = the IRA limit.
# Catch-up contributions, employer match, and the combined 401(k) limit are
# ignored — the split is just a starting point the user edits.
PRETAX_CONTRIB_CAP: float = 23_500.0   # 2025 401(k)/403(b) employee deferral
ROTH_CONTRIB_CAP: float = 7_000.0      # 2025 IRA contribution limit


# ════════════════════════════════════════════════════════════════════════════
# 3. Config resolution  (pure — merge a RetirementConfig dict over DEFAULTS)
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
# 4. Spending phases
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
    the HealthCare model in the projection, not scaled here.
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


# ── At-retirement balance projector  (pure) ──────────────────────────────────────

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

    Thin wrapper over the shared `simulate_lifetime` engine (accumulation slice):
    runs an all-working window of `years` and reads the balances entering
    retirement. The recurrence is identical, so the result is unchanged.
    (`simulate_lifetime` lives in core/services/projection.py until Phase 5
    moves it to core/domain/lifetime.py — imported lazily so importing this
    module never pulls in core.services.)
    """
    from core.services.projection import simulate_lifetime

    n = max(int(years), 0)
    df = simulate_lifetime(
        buckets=current_balances, allocation=annual_allocation,
        current_age=0, coast_age=n, retirement_age=n, death_age=n,
        spend_by_phase={}, slow_go_age=n, no_go_age=n,
        r_accum=real_return, r_retire=real_return,
    )
    row = df.loc[n]
    out = {b: float(row[b]) for b in ("taxable", "trad", "roth")}
    out["total"] = out["taxable"] + out["trad"] + out["roth"]
    return out
