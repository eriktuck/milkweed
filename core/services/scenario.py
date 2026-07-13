"""Unified scenario builder for the Coast page.

The Coast page replaces the split Forecast (accumulation) + Retirement (drawdown)
pages with one screen (.planning/SPEC-scenario-planning.md). This module is the
single entry point its callback calls: given a user's config + the two live inputs
(milestone ages and the monthly contribution), `build_scenario` composes the
existing pure helpers into one result object carrying the full lifetime walk and
every headline number (the four BANs).

It does **not** introduce new math. It orchestrates:
  * backward-from-spending goal — `drawdown.project_retirement` → `nest_egg_goal`
    (the same pipeline as `drawdown.default_nest_egg_goal`, but driven by the
    *live* slider ages and tuned spend/healthcare/SS instead of only config);
  * forward lifetime walk — `projection.simulate_lifetime` (accumulation → coast →
    tax-aware drawdown), whose accumulation aggregate reconciles with the old
    Forecast `project_portfolio` (see scripts/verify_lifetime_engine.py);
  * the CoastFIRE quantities — `coast_target = goal / (1+r)^(ret−coast)` and the
    closed-form `forecast.required_monthly_contribution`, identical to the old
    `forecast.forecast_summary`.

Because every number is produced by a helper the legacy pages already used,
`build_scenario` reconciles with both pages by construction — gated by
scripts/verify_scenario.py.

Pure: callers pass deserialized config dicts and the current `buckets` (read from
the holdings snapshot at the page edge, mirroring how the old pages called
`investments.balances_by_tax_bucket`). No Firestore or network here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from core.services.forecast import (
    default_monthly_contribution,
    required_monthly_contribution,
)
from core.domain.glide import (
    STRATEGY_TERMINAL_EQUITY,
    accumulation_rates,
    cumulative_growth,
    effective_cagr,
    equity_weight_path,
    required_monthly_glide,
    retirement_rate,
)
from core.services.projection import simulate_lifetime
from core.domain.assumptions import (
    DEFAULTS,
    PHASES,
    annual_spend_by_phase,
    default_contribution_allocation,
    resolve_assumptions,
    resolve_phase_factors,
)
from core.domain.drawdown import (
    TAXABLE_GAIN_FRACTION,
    nest_egg_goal,
    project_retirement,
)
from core.domain.healthcare import healthcare_costs_by_age
from core.domain.rmd import rmd_start_age
from core.domain.social_security import (
    annual_pia_from_earnings,
    estimate_annual_pia_from_income,
    social_security_income,
)

# Coast age lives on the Profile/UserConfig top-level (default 50); resolve_assumptions
# deliberately omits it (it is an accumulation-only milestone), so we read it directly.
DEFAULT_COAST_AGE: int = 50


@dataclass
class Scenario:
    """Everything the Coast page renders, from one `build_scenario` call.

    Resolved inputs are echoed back so the page can hydrate its controls from the
    same source of truth it just computed against (no second resolution path).
    `lifetime` is the full current_age→death_age bucket-level walk for the chart;
    the scalar fields are the four BANs plus supporting context.
    """

    # ── resolved inputs (the slider + contribution, after config/override merge) ──
    current_age: int
    coast_age: int
    retirement_age: int
    claim_age: int
    slow_go_age: int
    no_go_age: int
    death_age: int
    real_return: float
    monthly_contribution: float
    annual_contribution: float
    allocation: dict
    buckets: dict
    start_value: float

    # ── derived assumption series (consumed by the chart + assumption modules) ──
    spend_by_phase: dict          # {go_go, slow_go, no_go} → annual $ (healthcare excluded)
    pia_annual: float             # Social Security benefit at FRA (67)
    ss_by_age: pd.Series          # annual SS income by age
    healthcare_by_age: pd.Series  # annual healthcare cost by age

    # ── the lifetime walk (chart) ──
    lifetime: pd.DataFrame

    # ── headline numbers (BANs) ──
    nest_egg_goal: float          # backward-from-spending PV at retirement
    coast_target: float           # needed at coast_age so growth alone reaches the goal
    projected_at_coast: float     # forward walk balance entering coast_age
    coast_surplus: float          # projected_at_coast − coast_target
    required_monthly: float       # monthly contribution to hit coast_target by coast_age
    projected_at_retirement: float

    # ── return glide path (.planning SPEC delta, Phase 5a) ──
    # `real_return` above carries the effective (constant-equivalent) accumulation
    # CAGR when a glide is active, so existing readers see one rate either way.
    return_strategy: str | None = None       # aggressive|moderate|conservative, else None (flat rate)
    equity_glide: dict | None = None         # {age: equity weight} for the glide-path viz, else None

    on_track: bool = field(init=False)

    def __post_init__(self) -> None:
        self.on_track = self.projected_at_coast >= self.coast_target


def _resolve_pia(
    user_cfg: dict | None,
    active_plan: dict | None,
    csp_labels: dict | None,
) -> float:
    """Annual Social Security benefit at FRA, preferring real earnings history.

    Mirrors `drawdown.default_nest_egg_goal`: use the Profile income segments via
    `annual_pia_from_earnings` when present, else estimate from the CSP `income`
    line, else 0. Kept here so the slider-driven goal uses the identical SS basis
    the config-default goal does.
    """
    segments = (user_cfg or {}).get("income_segments")
    if segments:
        from core.domain.social_security import segments_to_annual_income

        return annual_pia_from_earnings(segments_to_annual_income(segments))

    labels = csp_labels or {}
    annual_income = 12.0 * sum(
        float(amt)
        for key, amt in (active_plan or {}).items()
        if labels.get(key) == "income"
    )
    return estimate_annual_pia_from_income(annual_income) if annual_income else 0.0


def build_scenario(
    user_cfg: dict | None,
    active_plan: dict | None,
    csp_labels: dict | None,
    buckets: dict,
    *,
    # live inputs (None → resolved from config / plan defaults)
    monthly_contribution: float | None = None,
    coast_age: int | None = None,
    retirement_age: int | None = None,
    claim_age: int | None = None,
    slow_go_age: int | None = None,
    no_go_age: int | None = None,
    death_age: int | None = None,
    real_return: float | None = None,
    # return glide path (.planning SPEC delta): strategy + the holdings-derived
    # two-leg params {w_equity0, r_equity, r_fixed} (from build_returns_payload).
    # When both resolve, the glide drives the return and `real_return` is ignored.
    return_strategy: str | None = None,
    glide_params: dict | None = None,
    # assumption-module overrides (None → the user's persisted RetirementConfig
    # value, else the service default)
    allocation: dict | None = None,
    pia_annual: float | None = None,
    gain_fraction: float | None = None,
    healthcare_overrides: dict | None = None,
    spend_by_phase: dict | None = None,
    as_of_year: int | None = None,
) -> Scenario:
    """Build the full Coast scenario from config + the two live inputs.

    `buckets` is the current {taxable, trad, roth, total} balance map (the page
    reads it from the holdings snapshot via `investments.balances_by_tax_bucket`).
    Live inputs and assumption overrides fall back to the user's config / plan
    defaults when None, so a fresh page load with no interaction renders a complete,
    sensible scenario. Pure — see module docstring.
    """
    cfg = user_cfg or {}
    ret_cfg = cfg.get("retirement") or {}
    a = resolve_assumptions(cfg)

    # ── resolve the milestone ages + return (live override → config → DEFAULTS) ──
    coast_age = int(coast_age if coast_age is not None
                    else cfg.get("coast_age") or DEFAULT_COAST_AGE)
    retirement_age = int(retirement_age if retirement_age is not None else a["retirement_age"])
    claim_age = int(claim_age if claim_age is not None else a["claim_age"])
    slow_go_age = int(slow_go_age if slow_go_age is not None else a["slow_go_age"])
    no_go_age = int(no_go_age if no_go_age is not None else a["no_go_age"])
    death_age = int(death_age if death_age is not None else a["death_age"])

    # current age from birth year (today's year by default); 40 when no birth date.
    year = int(as_of_year) if as_of_year is not None else date.today().year
    birth_year = a["birth_year"]
    current_age = (year - int(birth_year)) if birth_year else max(retirement_age - 25, 18)

    # ── return model: glide path (strategy + holdings legs) or flat rate ──────────
    # A resolved strategy + the two-leg params (w_equity0 / r_equity / r_fixed from
    # build_returns_payload) make the return a target-date glide: per-age accumulation
    # rates, a terminal (held) rate for drawdown + the goal discount, and an effective
    # CAGR echoed as `real_return`. Otherwise the flat rate (live override → config →
    # DEFAULTS) drives everything, exactly as before (verify_scenario reconciles this).
    strategy = return_strategy if return_strategy is not None else ret_cfg.get("return_strategy")
    gp = glide_params or {}
    use_glide = (
        strategy in STRATEGY_TERMINAL_EQUITY
        and gp.get("w_equity0") is not None
        and gp.get("r_equity") is not None
        and gp.get("r_fixed") is not None
    )
    if use_glide:
        w0 = float(gp["w_equity0"])
        w_term = STRATEGY_TERMINAL_EQUITY[strategy]
        r_eq, r_fx = float(gp["r_equity"]), float(gp["r_fixed"])
        accum_rates = accumulation_rates(w0, w_term, r_eq, r_fx, current_age, retirement_age)
        equity_glide = equity_weight_path(w0, w_term, current_age, retirement_age)
        r_retire = retirement_rate(w_term, r_eq, r_fx)
        r_accum_arg = accum_rates
        eff_return = effective_cagr(accum_rates) if accum_rates else r_retire
    else:
        strategy = None
        accum_rates = None
        equity_glide = None
        r_retire = float(real_return if real_return is not None else a["real_return"])
        r_accum_arg = r_retire
        eff_return = r_retire

    # ── contribution + allocation ──
    monthly = float(monthly_contribution if monthly_contribution is not None
                    else default_monthly_contribution(active_plan, csp_labels))
    annual = monthly * 12.0
    # allocation: explicit override → persisted RetirementConfig → tax-advantaged-first
    alloc = (allocation if allocation is not None
             else ret_cfg.get("contribution_allocation")
             or default_contribution_allocation(annual))

    # ── spend / healthcare / Social Security (the backward inputs) ──
    # The Coast Expenses module passes edited per-phase annual totals; otherwise the
    # spend is derived from the CSP plan × research phase factors (the default path,
    # identical to default_nest_egg_goal).
    if spend_by_phase is not None:
        spend = {p: float(spend_by_phase.get(p, 0.0)) for p in PHASES}
    else:
        phase_factors = resolve_phase_factors(ret_cfg)
        by_phase = annual_spend_by_phase(active_plan, csp_labels, phase_factors)
        spend = {p: by_phase[p]["total"] for p in PHASES}

    hc_overrides = healthcare_overrides if healthcare_overrides is not None else (
        ret_cfg.get("healthcare") or {})
    hc = healthcare_costs_by_age(retirement_age, death_age, **hc_overrides)

    if pia_annual is None:
        pia_annual = ret_cfg.get("pia_annual")
    pia = pia_annual if pia_annual is not None else _resolve_pia(cfg, active_plan, csp_labels)
    ss = social_security_income(pia, claim_age, retirement_age, death_age)

    if gain_fraction is None:
        gf = ret_cfg.get("gain_fraction")
        gain_fraction = float(gf) if gf is not None else TAXABLE_GAIN_FRACTION

    # ── backward-from-spending goal (same pipeline as default_nest_egg_goal) ──
    # Discount at the retirement-leg return (the held terminal allocation) — for a
    # flat rate that is just the single rate, unchanged.
    stream = project_retirement(
        0.0, retirement_age, death_age, spend, slow_go_age, no_go_age, r_retire,
        income_by_age=ss, healthcare_by_age=hc,
    )
    goal = max(nest_egg_goal(stream, r_retire), 0.0)

    # ── forward lifetime walk (chart + projected balances) ──
    rmd_start = rmd_start_age(birth_year) if birth_year else 75
    lifetime = simulate_lifetime(
        buckets=buckets, allocation=alloc,
        current_age=current_age, coast_age=coast_age,
        retirement_age=retirement_age, death_age=death_age,
        spend_by_phase=spend, slow_go_age=slow_go_age, no_go_age=no_go_age,
        r_accum=r_accum_arg, r_retire=r_retire,
        ss_by_age=ss, healthcare_by_age=hc,
        rmd_start=rmd_start, gain_fraction=gain_fraction,
    )

    # ── CoastFIRE quantities (flat rate: identical to forecast.forecast_summary;
    # glide: the cumulative glide growth replaces the constant-rate closed forms) ──
    years_coast_to_ret = max(retirement_age - coast_age, 0)
    coast_growth = (cumulative_growth(accum_rates, coast_age, retirement_age)
                    if use_glide else (1 + r_retire) ** years_coast_to_ret)
    coast_target = goal / coast_growth if coast_growth else goal

    def _at(age: int) -> float:
        return (float(lifetime.loc[age, "total"]) if age in lifetime.index
                else float(lifetime["total"].iloc[-1]))

    projected_at_coast = _at(coast_age)
    projected_at_retirement = _at(retirement_age)
    start_value = float(buckets.get("total", sum(
        float(buckets.get(b, 0.0)) for b in ("taxable", "trad", "roth"))))
    req_monthly = (required_monthly_glide(start_value, current_age, coast_age,
                                          coast_target, accum_rates) if use_glide
                   else required_monthly_contribution(start_value, current_age,
                                                      coast_age, coast_target, r_retire))

    return Scenario(
        current_age=current_age, coast_age=coast_age, retirement_age=retirement_age,
        claim_age=claim_age, slow_go_age=slow_go_age, no_go_age=no_go_age,
        death_age=death_age, real_return=eff_return,
        monthly_contribution=monthly, annual_contribution=annual,
        allocation=dict(alloc), buckets=dict(buckets), start_value=start_value,
        spend_by_phase=spend, pia_annual=float(pia), ss_by_age=ss, healthcare_by_age=hc,
        lifetime=lifetime,
        nest_egg_goal=goal, coast_target=coast_target,
        projected_at_coast=projected_at_coast,
        coast_surplus=projected_at_coast - coast_target,
        required_monthly=req_monthly,
        projected_at_retirement=projected_at_retirement,
        return_strategy=strategy,
        equity_glide={int(k): float(v) for k, v in equity_glide.items()} if equity_glide else None,
    )
