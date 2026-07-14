"""Pydantic config models — the schema for per-user / per-household Firestore
documents and the ``config-store`` ``dcc.Store`` JSON.

Pure data models only (pydantic, no I/O). The Firestore loader that populates
them lives in ``core/services/session.py`` (``SessionData``).
"""

from typing import Dict, List
from pydantic import BaseModel


class RetirementConfig(BaseModel):
    """Retirement-page *scenario* assumptions, persisted under UserConfig.retirement.

    Demographics (birth date, retirement / claim / death ages) and income history
    are owned by the **Profile** page and stored on the top-level UserConfig fields
    (`birth_date`, `retirement_age`, `claim_age`, `death_age`, `income_segments`).
    Profile is the single source of truth; core/domain/assumptions.resolve_assumptions
    reads demographics from there. This block holds only the knobs specific to the
    retirement projection.

    Every field is optional with a code-side default in core/domain/assumptions.py
    (DEFAULTS), so a config predating this block behaves as if the user accepted all
    defaults — no Firestore migration required (mirrors how
    transaction_account_settings was introduced).
    """
    # Spending-phase boundaries (ages). go-go = [retirement_age, slow_go_age),
    # slow-go = [slow_go_age, no_go_age), no-go = [no_go_age, death_age].
    slow_go_age: int | None = None         # default 75
    no_go_age: int | None = None           # default 85

    # Economic assumptions ---------------------------------------------------
    # No general inflation field by design: the model is in real (inflation-
    # adjusted) dollars, so `real_return` already nets out inflation and a general
    # inflation knob would double-count. Any future need is purpose-specific
    # (nominal tax-bracket indexing / healthcare excess inflation) and gets its
    # own dedicated field then.
    real_return: float | None = None       # expected real return; default 0.05
    withdrawal_rate: float | None = None    # SWR sanity-check rate; default 0.04

    # Return glide-path strategy (aggressive | moderate | conservative), the Coast
    # rate-of-return card's primary control (.planning SPEC delta, Phase 5a). When
    # set with live portfolio holdings, scenario.build_scenario derives a target-date
    # glide (current equity weight → preset terminal) and ignores `real_return`;
    # `real_return` survives as the displayed effective CAGR. Absent → flat rate.
    return_strategy: str | None = None

    # Per-csp-key phase multiplier overrides. Shape:
    #   {csp_key: {"slow_go": 0.9, "no_go": 0.7}}  (go-go is always 1.0)
    # Absent keys fall back to PHASE_FACTORS in the service (seeded from research).
    # This is the user-editable crosswalk; the Expenses pane (Phase 5) writes it.
    phase_factors: Dict[str, Dict[str, float]] | None = None

    # Current balances by tax bucket (taxable / trad / roth). Optional override;
    # when None the service derives buckets from the holdings snapshot +
    # investment_accounts (see retirement.balances_by_tax_bucket). Used by the
    # full tax-aware engine in Phase 6.
    account_balances: Dict[str, float] | None = None

    # ── Coast page assumption-module edits (.planning/SPEC-scenario-planning.md) ──
    # All optional → engine defaults apply (no migration; see the class
    # docstring). The Coast page's collapsible assumption cards persist here via
    # save_retirement_config, and core/domain/scenario.build_scenario reads them
    # as the per-user defaults for each live input.

    # Healthcare glide overrides (annual real $, per individual). Keys mirror
    # retirement.healthcare_costs_by_age kwargs: oop_annual, aca_annual,
    # medicare_annual, ltc_annual, ltc_start_age — plus ltc_type (none/in_home/
    # nursing) which the UI maps to ltc_annual via LTC_COST_BY_TYPE. Absent keys
    # fall back to the HC_* constants.
    healthcare: Dict[str, float] | None = None
    # Long-term-care type selection (none | in_home | nursing); seeds ltc_annual.
    ltc_type: str | None = None

    # Social Security benefit at FRA (annual $), overriding the earnings/income-
    # derived estimate (e.g. typed from an SSA statement). Absent → derived.
    pia_annual: float | None = None

    # Realized long-term-gain fraction of a taxable withdrawal. Absent → derived
    # from the cost-basis report, else the TAXABLE_GAIN_FRACTION default.
    gain_fraction: float | None = None

    # Annual contribution split across buckets {taxable, trad, roth}. Absent → the
    # tax-advantaged-first default (retirement.default_contribution_allocation).
    contribution_allocation: Dict[str, float] | None = None

    # Edited per-category retirement expenses (monthly real $), keyed by csp_key:
    #   {csp_key: {"go_go": 3200, "slow_go": 2900, "no_go": 2700}}
    # Persisted by the Coast Expenses module so edits survive reloads; absent → the
    # table re-seeds from the latest CSP plan × research phase_factors. build_scenario
    # consumes the phase totals via its spend_by_phase override (the page passes them).
    expenses: Dict[str, Dict[str, float]] | None = None


class UserConfig(BaseModel):
    accounts: List[str]
    cat_names: Dict[str, str]
    cat_order: List[str]
    csp_from_category: Dict[str, str]
    csp_from_group: Dict[str, str]
    csp_labels: Dict[str, str]
    drop_cats: List[str]
    group_names: Dict[str, str]
    name: str
    uid: str | None = None
    budget: dict | None = None
    csp_plan: dict | None = None  # legacy — superseded by csp_plans
    csp_plans: dict | None = None
    net_worth: dict | None = None
    # Maps last-4 Vanguard account number → account type label.
    # Labels are user-defined, e.g. "IRA", "Roth IRA", "Brokerage", "403b".
    # Drives the retirement vs non-retirement split on the Investments page.
    investment_accounts: Dict[str, str] | None = None
    # Maps last-4 account number → user-chosen nickname, e.g. "Work 403b".
    investment_account_nicknames: Dict[str, str] | None = None
    # Per transaction-account settings, keyed by Monarch account displayName
    # (the same string stored in `accounts` and on each transaction's
    # `account_name`). Each value is {"include": bool, "nickname": str}.
    # `include == False` drops that account's transactions instead of saving
    # them. Absent key → included (backward compatible with pre-control-plane
    # configs that have no settings map).
    transaction_account_settings: Dict[str, dict] | None = None
    # Retirement-page planning assumptions (spec-retirement.md, Phase 2).
    # Absent → service defaults apply; no migration needed (see RetirementConfig).
    retirement: RetirementConfig | None = None
    # ── User Profile (retirement modelling background data) ──
    # Source of truth for demographics + income history, edited on the Profile page
    # (pages/profile.py) and consumed by the Retirement page via
    # core/domain/assumptions.resolve_assumptions. All optional; absent → functional
    # defaults supplied by the model/UI layer.
    birth_date: str | None = None              # ISO "YYYY-MM-DD"; birth_year = year part
    coast_age: int | None = None               # default 50; must be < retirement_age
    retirement_age: int | None = None          # default 67
    claim_age: int | None = None               # default 70; Social Security claim, 62–70
    death_age: int | None = None               # default 90; planning horizon
    income_growth_rate: float | None = None    # real annual income growth; default 0.03
    # Gross-income history as forward-filled segments. Each entry is
    # {"date": "YYYY-MM-DD", "amount": <annual gross income>}: the amount is the
    # annual rate in effect from `date` until the next segment. A future-dated
    # segment models a planned raise. See core.utils.functions.segments_to_annual_income.
    income_segments: List[dict] | None = None


class HouseholdConfig(UserConfig):
    members: List[str]


class SessionDataModel(BaseModel):
    """Validated structure for session data."""
    users: Dict[str, UserConfig | HouseholdConfig]
