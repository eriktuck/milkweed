"""Coast page — the unified scenario-planning screen.

Replaces the split Forecast (accumulation) + Retirement (drawdown) pages with one
cockpit: a milestone slider + monthly-contribution input drive the four BANs and a
single accumulation→coast→drawdown chart. See .planning/SPEC-scenario-planning.md
(Phase 2 — primary screen; the background assumption modules land in Phase 3).

Every number comes from one call into `core.domain.scenario.build_scenario`,
which resolves all assumptions from `config-store` (UserConfig + RetirementConfig)
plus the two live inputs — no `retirement-goal-store` handshake, no hidden age
stores. This module is layout + callbacks only.

Scoped to the user selected in the global "Viewing as" (`use-case`) dropdown,
mirroring Forecast/Retirement's single-individual scope (household is a later
concern per the spec's open questions).
"""

import json
from datetime import date

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html
from flask import session

from core.domain.budget import get_active_csp_plan
from core.domain.housing import (
    DEFAULT_INFLATION,
    DEFAULT_REAL_RENT_GROWTH,
    LONG_RUN_REAL_APPRECIATION,
    MAINTENANCE_PCT,
    VACANCY_PCT,
    derive_real_appreciation,
    is_configured,
    spend_exclude_keys,
)
from core.domain.social_security import segments_to_annual_income
from core.domain.assumptions import (
    DEFAULT_PHASE_FACTOR,
    PHASES,
    annual_spend_by_phase,
    default_contribution_allocation,
    default_monthly_contribution,
    household_expense_share,
    merge_household_expenses,
    phase_for_age,
    resolve_assumptions,
    resolve_phase_factors,
)
from core.domain.drawdown import TAXABLE_GAIN_FRACTION
from core.domain.healthcare import (
    HC_ACA_ANNUAL,
    HC_OOP_ANNUAL,
    LTC_DEFAULT_ANNUAL,
    LTC_DEFAULT_START_AGE,
    LTC_DEFAULT_TYPE,
    MEDICARE_PART_B_MONTHLY_2025,
)
from core.domain.rmd import rmd_start_age
from core.domain.social_security import (
    annual_pia_from_earnings,
    estimate_annual_pia_from_income,
)
from core.services.firebase import save_retirement_config
from core.services.investments import (
    balances_by_tax_bucket,
    fetch_latest_holdings,
    taxable_gain_fraction,
)
from core.domain.returns import BOND_PROXY, glide_cagr_projection
from core.services.market_data import build_returns_payload
from core.domain.glide import (
    DEFAULT_STRATEGY,
    STRATEGY_LABELS,
    STRATEGY_TERMINAL_EQUITY,
    equity_weight_path,
)
from core.domain.scenario import DEFAULT_COAST_AGE, build_scenario

dash.register_page(__name__, path="/coast")

# ── Chart palette (shared with Forecast/Retirement) ──────────────────────────────
_C_PRINCIPAL = "#5b7ec9"   # blue — starting balance + contributions
_C_GROWTH = "#4ade80"      # green — market growth
_C_TARGET = "#f59e0b"      # amber — nest-egg goal / CoastFIRE target
_C_TOTAL = "#444444"       # total value line
_C_WORKING = "#3b82f6"
_C_TRAD = "#8b5cf6"        # violet — tax-deferred
_C_ROTH = "#14b8a6"        # teal — Roth

# Housing module. The asset reads as portfolio-blue, the debt as a muted grey
# beneath it, and the gap between them (equity) picks up the growth green.
_C_HOUSE_VALUE = "#5b7ec9"
_C_HOUSE_DEBT = "#9aa5b1"
# Cost components, in stacking order: the permanent floor first, the terminating
# P&I on top of it, so the payoff cliff visibly lands on something.
_HOUSING_COST_TRACES = (
    ("escrow", "Escrow (tax + insurance)", "#b9c9e8", "rgba(185,201,232,0.75)"),
    ("maintenance", "Maintenance", "#d6b38a", "rgba(214,179,138,0.65)"),
    ("pi", "Principal & interest", _C_HOUSE_VALUE, "rgba(91,126,201,0.55)"),
    ("rent", "Rent", "#7fb3f5", "rgba(127,179,245,0.55)"),
)

# Faint per-asset-class lines on the rate-of-return chart (equities, then bonds);
# colours are muted so the bold CAGR line leads. Cash is omitted — the scenario forces
# it to a flat 0% real, so it would just trace the zero reference line.
_CLASS_ORDER = ["US Equity", "Intl Equity", "Other", "Bonds"]
_CLASS_LINE_COLORS = {
    "US Equity": "#9bb4e0",    # muted blue
    "Intl Equity": "#86c5b9",  # muted teal
    "Other": "#c3aed6",        # muted violet
    "Bonds": "#d6b38a",        # muted tan
}

# Fallback ages (used only before a Profile exists; Profile is the truth).
_DEFAULT_CURRENT_AGE = 40
_DEFAULT_DEATH_AGE = 90

_CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="#444"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# The milestone slider's five handles, in the order build_scenario expects them.
# dcc.RangeSlider enforces this monotonic ordering (allowCross=False), which is
# exactly the natural milestone constraint: coast ≤ retire ≤ claim ≤ slow ≤ no-go.
_MILESTONES = [
    ("Coast", "coast_age"),
    ("Retire", "retirement_age"),
    ("SS", "claim_age"),
    ("Slow-go", "slow_go_age"),
    ("No-go", "no_go_age"),
]


# ── Formatting helpers (mirror pages/forecast.py) ────────────────────────────────

def _money(v) -> str:
    """Abbreviated currency: $1.50M / $548K / $900."""
    v = float(v or 0)
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:,.0f}K"
    return f"${v:,.0f}"


def _selected_uid(use_case):
    """The user the page is scoped to: the 'Viewing as' selection, else the
    logged-in user (mirrors Forecast/Retirement's single-individual scope)."""
    return use_case or session.get("user_id")


def _user_cfg(config_data, uid):
    if not config_data or not isinstance(config_data, str) or not uid:
        return None
    config = json.loads(config_data)
    return config.get("users", {}).get(uid)


def _active_plan(user_cfg) -> dict:
    """A user's latest CSP plan (monthly {csp_key: amount}), with the legacy
    fallback the other pages use."""
    csp_plans = (user_cfg or {}).get("csp_plans") or {}
    return get_active_csp_plan(csp_plans) or (user_cfg or {}).get("csp_plan") or {}


def _current_age(user_cfg) -> int:
    """Current whole-years age from the saved birth date, else the fallback."""
    a = resolve_assumptions(user_cfg or {})
    by = a.get("birth_year")
    return (date.today().year - int(by)) if by else _DEFAULT_CURRENT_AGE


def _money_input(id_, **kw):
    """$-prefixed numeric input (mirrors pages/retirement.py)."""
    return dbc.InputGroup([dbc.InputGroupText("$"),
                           dbc.Input(id=id_, type="number", size="sm", **kw)], size="sm")


def _pct_input(id_, **kw):
    """%-suffixed numeric input (the shape already inlined for co-gain-frac and
    the phase-scale inputs; the Housing module needs seven of them)."""
    return dbc.InputGroup([dbc.Input(id=id_, type="number", size="sm", **kw),
                           dbc.InputGroupText("%")], size="sm")


def _bucket_inputs(prefix, step):
    """Stacked Taxable / Tax-deferred / Roth money inputs sharing an id prefix."""
    return dbc.Row([
        dbc.Col([dbc.Label("Taxable", className="small text-muted mb-1"),
                 _money_input(f"{prefix}-taxable", value=0, min=0, step=step)], width=12),
        dbc.Col([dbc.Label("Tax-deferred", className="small text-muted mb-1"),
                 _money_input(f"{prefix}-trad", value=0, min=0, step=step)], width=12),
        dbc.Col([dbc.Label("Roth", className="small text-muted mb-1"),
                 _money_input(f"{prefix}-roth", value=0, min=0, step=step)], width=12),
    ], className="g-2")


# ── Expenses + healthcare constants/helpers (ported from pages/retirement.py) ─────

# Healthcare input defaults (totals per life stage = stage premium + baseline OOP).
_HC_ACA_DEFAULT = int(HC_ACA_ANNUAL + HC_OOP_ANNUAL)                            # pre-65
_HC_MEDICARE_DEFAULT = int(MEDICARE_PART_B_MONTHLY_2025 * 12 + HC_OOP_ANNUAL)   # 65+

_LTC_TYPE_OPTIONS = [
    {"label": "None", "value": "none"},
    {"label": "In-home care", "value": "in_home"},
    {"label": "Nursing home", "value": "nursing"},
]

# CSP label → display header, in canonical CSP order (mirrors pages/retirement.py).
_EXPENSE_GROUPS = [
    ("fixed", "Fixed Costs"),
    ("investments", "Investments"),
    ("sinking", "Sinking"),
    ("guilt-free", "Guilt Free"),
]
_EXPENSE_GROUP_LABEL = dict(_EXPENSE_GROUPS)
_EXPENSE_GROUP_RANK = {label: i for i, (label, _) in enumerate(_EXPENSE_GROUPS)}

_EXPENSE_COLUMNS = [
    {"name": "CSP category", "id": "category", "editable": False},
    {"name": "Go-go $/mo", "id": "go_go", "type": "numeric", "editable": True},
    {"name": "Slow-go $/mo", "id": "slow_go", "type": "numeric", "editable": True},
    {"name": "No-go $/mo", "id": "no_go", "type": "numeric", "editable": True},
    {"name": "factor", "id": "factor", "editable": False},
]


def _pretty(key: str) -> str:
    """csp-key → display label (e.g. 'home_other' → 'Home Other')."""
    return key.replace("_", " ").title()


def _num(v) -> float:
    """Coerce an editable DataTable cell (may be '' / str / None) to float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _spend_from_table(rows) -> dict:
    """Annual go-go/slow-go/no-go living spend = 12 × Σ each phase column."""
    rows = rows or []
    return {ph: 12.0 * sum(_num(r.get(ph)) for r in rows if r.get("key")) for ph in PHASES}


def _group_header_row(label: str) -> dict:
    """A non-numeric divider row that titles a CSP group (blank key → skipped by
    the edit/sum helpers)."""
    return {"key": None, "category": _EXPENSE_GROUP_LABEL[label],
            "go_go": "", "slow_go": "", "no_go": "", "factor": ""}


def _household_for(config, uid):
    """(household_cfg, member_uids) for the household containing `uid`, else
    (None, [uid]). The household is the config-store `users` entry with a `members`
    list (ported from pages/retirement.py)."""
    for cfg in (config.get("users") or {}).values():
        members = cfg.get("members")
        if members and uid in members:
            return cfg, members
    return None, [uid]


def _effective_expense_plan(config, uid):
    """The individual's retirement expense plan with their share of household shared
    costs folded in (ported from pages/retirement.py).

    Returns (plan, csp_labels, share): `plan` is the monthly {csp_key: amount} map
    (own plan + share × household plan per key), `csp_labels` merges household labels
    under the individual's, and `share` is this member's household-expense fraction
    (0.0 when not in a household). Only individuals retire, so household shared costs
    are attributed per member by joint-contribution share; downstream
    `annual_spend_by_phase` drops the funding transfer so nothing double-counts.
    """
    users = config.get("users") or {}
    user_cfg = users.get(uid) or {}
    labels = dict(user_cfg.get("csp_labels") or {})
    plan = _active_plan(user_cfg)

    household_cfg, member_ids = _household_for(config, uid)
    if household_cfg is None:
        return plan, labels, 0.0

    contribs = {
        m: float(_active_plan(users.get(m, {})).get("joint_contribution", 0.0))
        for m in member_ids if m in users
    }
    share = household_expense_share(contribs, uid)
    plan = merge_household_expenses(plan, _active_plan(household_cfg), share)
    labels = {**(household_cfg.get("csp_labels") or {}), **labels}
    return plan, labels, share


def _seed_expense_rows(config, uid, housing_cfg=None) -> list[dict]:
    """One editable monthly row per living-expense csp key, seeded from the CSP plan
    (go-go) × research multipliers (slow-go/no-go), with saved edits applied on top.

    `housing_cfg` is the *live* Housing-module form state (None → the persisted
    block). It has to be threaded in because these rows are fed straight back to
    `build_scenario` as `spend_by_phase=`, which bypasses its own gate: without it,
    switching Housing on would double-count the mortgage — once here, once from the
    module — until the user saved and reloaded.

    The plan folds in this individual's share of household shared expenses (see
    `_effective_expense_plan`) — only individuals retire, so household costs are
    attributed per member by joint-contribution share. The row set, ordering and
    per-row factor hint come from `annual_spend_by_phase`; any persisted
    `RetirementConfig.expenses` override the cell values so edits survive reloads.
    Rows are grouped by CSP label (Fixed → Investments → Sinking → Guilt Free) then
    the user's `cat_order`, with a divider row per group.
    """
    user_cfg = (config.get("users") or {}).get(uid) or {}
    active_plan, csp_labels, _share = _effective_expense_plan(config, uid)
    factors = resolve_phase_factors(user_cfg.get("retirement"))
    # These rows are fed straight back into build_scenario as `spend_by_phase=`,
    # which bypasses its own gating — so the exclusion has to happen here too, via
    # the same helper. Without it a housing-configured user double-counts the
    # mortgage: once in this table, once from the Housing module.
    housing = (housing_cfg if housing_cfg is not None
               else (user_cfg.get("retirement") or {}).get("housing"))
    by_phase = annual_spend_by_phase(
        active_plan, csp_labels, factors,
        exclude_keys=spend_exclude_keys(housing))
    saved = (user_cfg.get("retirement") or {}).get("expenses") or {}

    rows = []
    for key in by_phase["go_go"]["by_key"]:
        f = factors.get(key, DEFAULT_PHASE_FACTOR)
        row = {
            "key": key,
            "category": _pretty(key),
            "go_go": round(by_phase["go_go"]["by_key"][key] / 12),
            "slow_go": round(by_phase["slow_go"]["by_key"][key] / 12),
            "no_go": round(by_phase["no_go"]["by_key"][key] / 12),
            "factor": f"{f.get('slow_go', 1.0):.0%}/{f.get('no_go', 1.0):.0%}",
        }
        if key in saved:  # persisted edits win over the CSP re-seed
            for ph in PHASES:
                if saved[key].get(ph) is not None:
                    row[ph] = round(_num(saved[key][ph]))
        rows.append(row)

    order_index = {name: i for i, name in enumerate((user_cfg or {}).get("cat_order") or [])}
    rows.sort(key=lambda r: (
        _EXPENSE_GROUP_RANK.get(csp_labels.get(r["key"]), len(_EXPENSE_GROUPS)),
        order_index.get(r["key"], len(order_index)),
        r["category"],
    ))

    out: list[dict] = []
    current = object()
    for r in rows:
        label = csp_labels.get(r["key"])
        if label != current:
            current = label
            if label in _EXPENSE_GROUP_LABEL and label != "guilt-free":
                out.append(_group_header_row(label))
        out.append(r)
    return out


# ── Housing module helpers (.planning/SPEC-housing.md, Phase 4) ──────────────────

# Percent-valued form fields, shown as whole percents and stored as fractions.
_HSG_PCT_DEFAULTS = {
    "rent_growth": DEFAULT_REAL_RENT_GROWTH,
    "prop_rent_growth": DEFAULT_REAL_RENT_GROWTH,
    "appreciation": LONG_RUN_REAL_APPRECIATION,
    "maintenance": MAINTENANCE_PCT,
    "vacancy": VACANCY_PCT,
    "inflation": DEFAULT_INFLATION,
}


def _blank(v):
    """A cleared numeric input → None, so it round-trips as "unset" rather than 0.

    Deliberately not `_num`: zero is a real answer (a paid-off house has no
    payment) and must be distinguishable from an empty box.
    """
    return None if v in (None, "") else float(v)


def _from_pct(v):
    """Whole-percent input → fraction, preserving the blank/None distinction."""
    v = _blank(v)
    return None if v is None else v / 100.0


def _to_pct(v, default: float):
    """Stored fraction (or its code-side default) → whole percent for the form."""
    return round((default if v is None else float(v)) * 100.0, 4)


def _housing_from_form(mode, rent, rent_growth, has_prop, occupancy, label,
                       payment, escrow, note_rate, payoff,
                       price, purchase_date, value, appreciation, maintenance,
                       rent_income, vacancy, prop_rent_growth, inflation) -> dict:
    """The Housing form as a `HousingConfig`-shaped dict.  (pure)

    Always returns the complete block, including explicit `None`s — that is what
    makes a cleared field actually clear in Firestore, whose `merge=True`
    deep-merges nested maps (see RETIREMENT_FIELDS in core/services/firebase.py).
    Emptiness is not decided here: `housing.is_configured` is the single gate, so
    an untouched form resolves to nothing and leaves the scenario byte-identical.

    A property is written as a one-element list; multi-property is a later UI
    iteration, not a config-shape change.
    """
    prop = {
        "label": label or None,
        "occupancy": occupancy or "primary",
        "purchase_price": _blank(price),
        "purchase_date": purchase_date or None,
        "current_value": _blank(value),
        "real_appreciation": _from_pct(appreciation),
        "land_fraction": None,
        "monthly_payment": _blank(payment),
        "escrow_monthly": _blank(escrow),
        "note_rate": _from_pct(note_rate),
        "payoff_date": payoff or None,
        "maintenance_pct": _from_pct(maintenance),
        "monthly_rent_income": _blank(rent_income) if occupancy == "rental" else None,
        "vacancy_pct": _from_pct(vacancy) if occupancy == "rental" else None,
        "real_rent_growth": (_from_pct(prop_rent_growth) if occupancy == "rental"
                             else None),
        "sale": None,   # Phase 5 owns the sale inputs
    }
    return {
        "residence": {
            "mode": mode or None,
            "monthly_rent": _blank(rent),
            "real_rent_growth": _from_pct(rent_growth),
        },
        "properties": [prop] if has_prop else [],
        "inflation": _from_pct(inflation),
    }


def _housing_block(user_cfg) -> dict:
    """The persisted housing block for a user (never None)."""
    return ((user_cfg or {}).get("retirement") or {}).get("housing") or {}


def _first_property(block: dict) -> dict:
    props = [p for p in (block.get("properties") or []) if p]
    return props[0] if props else {}


def _housing_summary(s, block: dict) -> str:
    """The module's collapsed one-line summary.

    Reads the resolved scenario rather than the form, so it states what the
    numbers above are actually doing. The sale clause the spec sketches
    ("sell at 70 → +$512k") arrives with the Phase 5 sale inputs.
    """
    if s.housing_by_age is None or s.housing_by_age.empty:
        return "not modelled — the mortgage stays in your expenses"

    res = block.get("residence") or {}
    prop = _first_property(block)
    mode = res.get("mode")
    if mode == "own" and not prop:
        return "owning, but no home entered — housing costs are $0"
    if mode == "rent" and not res.get("monthly_rent"):
        return "renting, but no rent entered — housing costs are $0"

    # Monthly figures stay unabbreviated — "$2,100/mo" reads as a real payment,
    # "$2K/mo" does not. Balances keep `_money`'s abbreviation.
    per_mo = lambda v: f"${v:,.0f}/mo"  # noqa: E731

    # Read from the chart frame's first row — TODAY, not retirement. The summary
    # should agree with the left edge of the chart the user is looking at.
    now = s.housing_breakdown.iloc[0]
    bits = [f"{'Rent' if mode == 'rent' else 'Own'} {per_mo(float(now['total']) / 12)}"]

    if str(prop.get("occupancy")) == "rental":
        bits.append("1 rental")
    if s.home_equity is not None and not s.home_equity.empty:
        equity = float(s.home_equity["equity"].iloc[0])
        if equity:
            bits.append(f"{_money(equity)} equity")

    # The headline for anyone who owns a rental: what the two net to. "Ahead"
    # means what you own more than pays for where you live.
    if round(float(now["rental_net"])):
        net = float(now["net"]) / 12
        bits.append(f"{per_mo(abs(net))} {'net' if net > 0 else 'ahead'}")
    return " · ".join(bits)


def _healthcare_overrides(aca, medicare, ltc, ltc_age) -> dict:
    """The healthcare_costs_by_age kwargs build_scenario expects. oop_annual=0 because
    the per-stage totals already fold out-of-pocket in (mirrors retirement.recompute)."""
    return {
        "oop_annual": 0.0,
        "aca_annual": _num(aca),
        "medicare_annual": _num(medicare),
        "ltc_annual": _num(ltc),
        "ltc_start_age": int(ltc_age or LTC_DEFAULT_START_AGE),
    }


# ── Compact BAN (cockpit-sized; lighter than the dbc.Card BANs elsewhere) ─────────

def _ban(label: str, value: str, sub: str = "", value_class: str = ""):
    return html.Div([
        html.Div(label, className="ban-label"),
        html.P(value, className=f"ban-value {value_class}".strip()),
        html.P(sub, className="ban-sub") if sub else None,
    ])


def _module(module_id: str, icon: str, title: str, summary_id: str, body):
    """A collapsible assumption row (the below-the-fold accordion pattern).

    Header shows the title + a live one-line summary (filled via `summary_id`) and a
    chevron; clicking the header toggles the body. Mirrors mockups/coast-hybrid.html.
    The chart/BANs above stay visible and re-run when the body's inputs change.
    """
    head = html.Div([
        html.Div([
            html.I(className=f"fas {icon} me-2 text-muted"),
            html.Span(title, className="title"),
            html.Span(id=summary_id, className="sub ms-2"),
        ]),
        html.I(className="fas fa-chevron-down text-muted", id=f"{module_id}-chevron"),
    ], id=f"{module_id}-head", className="module-head", n_clicks=0)
    collapse = dbc.Collapse(html.Div(body, className="module-body"),
                            id=f"{module_id}-collapse", is_open=False)
    return dbc.Card([head, collapse], className="module-card shadow-sm mb-2")


def _module_toggle(module_id: str):
    """Register the open/close + chevron-flip callback for a `_module` row."""
    @callback(
        Output(f"{module_id}-collapse", "is_open"),
        Output(f"{module_id}-chevron", "className"),
        Input(f"{module_id}-head", "n_clicks"),
        prevent_initial_call=True,
    )
    def _toggle(n):
        is_open = bool(n) and n % 2 == 1
        chevron = "fas fa-chevron-up text-muted" if is_open else "fas fa-chevron-down text-muted"
        return is_open, chevron


def _milestone_chips(coast, retire, claim, slow, no_go):
    """Live label→age summary under the slider (the slider only shows raw values)."""
    pairs = list(zip([lbl for lbl, _ in _MILESTONES],
                     [coast, retire, claim, slow, no_go]))
    return [
        html.Span([html.Span(f"{lbl} ", className="text-muted"),
                   html.B(f"{int(v)}")], className="me-3")
        for lbl, v in pairs
    ]


# ── Cockpit: the two live inputs (left) + the four BANs (right) ───────────────────

_cockpit = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([
        html.Div([html.I(className="fas fa-sliders me-2 text-muted"),
                  "Milestones — drag to run a scenario"], className="ban-label mb-2"),
        dcc.RangeSlider(
            id="co-milestones",
            min=_DEFAULT_CURRENT_AGE, max=_DEFAULT_DEATH_AGE, step=1,
            value=[50, 65, 67, 75, 85],
            allowCross=False, pushable=0,
            marks={a: str(a) for a in range(40, 91, 5)},
            tooltip={"placement": "top", "always_visible": True},
        ),
        html.Div(id="co-milestone-chips", className="small mt-1 mb-3"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Monthly contribution", className="ban-label mb-1"),
                dbc.InputGroup([
                    dbc.InputGroupText("$"),
                    dbc.Input(id="co-monthly", type="number", min=0, step=50, size="sm"),
                ], size="sm"),
            ], width="auto"),
            dbc.Col(html.Small(id="co-contribution-hint", className="assist-note"),
                    className="d-flex align-items-end pb-1"),
        ], className="g-2", style={"maxWidth": "420px"}),
    ], lg=7),

    dbc.Col(dbc.Row([
        dbc.Col(html.Div(id="co-ban-goal"), width=6),
        dbc.Col(html.Div(id="co-ban-coast-target"), width=6),
        dbc.Col(html.Div(id="co-ban-coast-status"), width=6, className="mt-3"),
        dbc.Col(html.Div(id="co-ban-required"), width=6, className="mt-3"),
    ], className="g-3 text-center"), lg=5),
], className="g-4 align-items-center")), className="mb-3 shadow-sm")


# ── Primary chart (always visible) ────────────────────────────────────────────────

_chart_card = dbc.Card(dbc.CardBody([
    html.Div([
        html.P("Portfolio value over time", className="ban-label mb-0 d-inline"),
        dbc.RadioItems(
            id="co-color-mode", value="pg", inline=True,
            options=[{"label": "Principal/Growth", "value": "pg"},
                     {"label": "Tax bucket", "value": "bucket"}],
            className="small float-end",
            inputClassName="me-1", labelClassName="me-2",
        ),
    ], className="mb-2"),
    dcc.Loading(dcc.Graph(id="co-projection-chart",
                          config={"displayModeBar": False}), type="circle"),
]), className="mb-4 shadow-sm")


# ── Assumption module 1: Rate of return ──────────────────────────────────────────

_STRATEGY_OPTIONS = [
    {"label": f"{STRATEGY_LABELS[k]} ({int(w * 100)}% equity at retirement)", "value": k}
    for k, w in STRATEGY_TERMINAL_EQUITY.items()
]

_rate_module = _module(
    "co-rr", "fa-percent", "Rate of return", "co-rr-summary",
    body=[
        html.P(
            "Your return isn't one number — it follows your asset mix, which de-risks "
            "toward retirement like a target-date fund. Pick how much equity risk to "
            "carry: aggressive aims for a bigger nest egg, conservative locks in what "
            "you have. The glide starts from your portfolio's current equity weight. "
            "Markets are stochastic; the simulation shows a range of outcomes, not a "
            "guarantee.",
            className="text-muted small mb-3",
        ),
        dbc.Row([
            dbc.Col(dbc.Label("Strategy", className="ban-label mb-0"), width="auto",
                    className="d-flex align-items-center"),
            dbc.Col(dbc.RadioItems(
                id="co-strategy", options=_STRATEGY_OPTIONS, value=DEFAULT_STRATEGY,
                className="small", inputClassName="me-1", labelClassName="me-3 mb-0",
                inline=True,
            ), className="d-flex align-items-center"),
        ], className="g-2 mb-2"),
        html.P("The faint lines are each leg's actual annual real return — equity and "
               "bonds swing wildly year to year. The bold line is your portfolio's "
               "average return (CAGR): smooth in the past, projected forward under the "
               "chosen glide. Any single year is a coin toss, but over the long run the "
               "average converges — and the shaded band shows how a more conservative "
               "glide tightens that long-run outcome.",
               className="text-muted small mb-2"),
        dcc.Loading(dcc.Graph(id="co-fan-chart",
                              config={"displayModeBar": False}), type="circle"),
        html.P(id="co-fan-caption", className="text-muted small mt-2 mb-0"),
        html.Div([
            dbc.Button("Save strategy", id="co-rr-save", color="primary", size="sm",
                       outline=True),
            html.Span(id="co-rr-save-status", className="small ms-2"),
        ], className="mt-2"),
    ],
)


# ── Assumption module 2: Retirement expenses + healthcare ────────────────────────

_expenses_table = dash_table.DataTable(
    id="co-exp-table",
    columns=_EXPENSE_COLUMNS,
    data=[],
    editable=True,
    cell_selectable=True,
    style_as_list_view=True,
    style_table={"overflowX": "auto"},
    style_cell={"fontSize": "13px", "padding": "6px 10px", "fontFamily": "inherit"},
    style_header={"fontWeight": "600", "fontSize": "11px", "textTransform": "uppercase",
                  "color": "#888", "border": "none", "borderBottom": "1px solid #dee2e6"},
    style_cell_conditional=[
        {"if": {"column_id": "category"}, "textAlign": "left", "minWidth": "180px"},
        {"if": {"column_id": "factor"}, "textAlign": "right", "color": "#aaa",
         "fontSize": "11px"},
        {"if": {"column_id": "go_go"}, "textAlign": "right"},
        {"if": {"column_id": "slow_go"}, "textAlign": "right"},
        {"if": {"column_id": "no_go"}, "textAlign": "right"},
    ],
    style_data_conditional=[
        {"if": {"column_id": "go_go"}, "color": "#3f78c4"},
        {"if": {"column_id": "slow_go"}, "color": "#3a9d5d"},
        {"if": {"column_id": "no_go"}, "color": "#b9791b"},
        *[
            {"if": {"filter_query": f'{{category}} = "{name}"'},
             "fontWeight": "700", "textTransform": "uppercase", "fontSize": "11px",
             "color": "#555", "backgroundColor": "#f5f5f5"}
            for _, name in _EXPENSE_GROUPS
        ],
    ],
)

_expenses_module = _module(
    "co-exp", "fa-receipt", "Retirement expenses & healthcare", "co-exp-summary",
    body=[
        html.P([
            "Go-go is seeded from your latest CSP plan; slow-go and no-go apply research "
            "multipliers — edit any cell, or rescale a whole phase below. Healthcare is "
            "modelled separately (it rises late), so ",
            html.B("medical & health insurance are excluded here"), ".",
        ], className="text-muted small mb-3"),
        _expenses_table,
        html.P(id="co-exp-note", className="text-muted small mt-2 mb-1"),
        dbc.Row([
            dbc.Col(html.Span("Rescale a whole phase from go-go (keeps the research shape):",
                              className="small text-muted"), width="auto",
                    className="d-flex align-items-center"),
            dbc.Col([
                dbc.Label("Slow-go", className="small text-muted mb-1"),
                dbc.InputGroup([
                    dbc.Input(id="co-exp-slow-scale", type="number", value=100, min=0,
                              max=200, step=5, size="sm", style={"width": "70px"}),
                    dbc.InputGroupText("%")], size="sm"),
            ], width="auto"),
            dbc.Col([
                dbc.Label("No-go", className="small text-muted mb-1"),
                dbc.InputGroup([
                    dbc.Input(id="co-exp-no-scale", type="number", value=100, min=0,
                              max=200, step=5, size="sm", style={"width": "70px"}),
                    dbc.InputGroupText("%")], size="sm"),
            ], width="auto"),
            dbc.Col(dbc.Button("Apply", id="co-exp-apply-scale", color="secondary",
                               size="sm", outline=True),
                    width="auto", className="d-flex align-items-end"),
            dbc.Col(html.Span(id="co-exp-totals", className="small fw-bold"),
                    className="d-flex align-items-end"),
        ], className="g-3 align-items-end"),

        html.Hr(className="my-3"),
        html.Div([html.B("Healthcare — its own expense "),
                  dbc.Badge("rises late", color="warning", className="align-middle")],
                 className="mb-2"),
        dbc.Row([
            dbc.Col([dbc.Label("ACA bridge (pre-65), $/yr", className="small text-muted mb-1"),
                     _money_input("co-hc-aca", value=_HC_ACA_DEFAULT, min=0, step=500)], lg=3),
            dbc.Col([dbc.Label("Medicare + OOP (65+), $/yr", className="small text-muted mb-1"),
                     _money_input("co-hc-medicare", value=_HC_MEDICARE_DEFAULT, min=0, step=500)], lg=3),
            dbc.Col([dbc.Label("Long-term care", className="small text-muted mb-1"),
                     dbc.Select(id="co-hc-ltc-type", size="sm", options=_LTC_TYPE_OPTIONS,
                                value=LTC_DEFAULT_TYPE)], lg=2),
            dbc.Col([dbc.Label("LTC cost, $/yr", className="small text-muted mb-1"),
                     _money_input("co-hc-ltc", value=int(LTC_DEFAULT_ANNUAL), min=0, step=5000)], lg=2),
            dbc.Col([dbc.Label("LTC start age", className="small text-muted mb-1"),
                     dbc.Input(id="co-hc-ltc-age", type="number", value=LTC_DEFAULT_START_AGE,
                               min=65, max=105, step=1, size="sm")], lg=2),
        ], className="g-2 mb-1"),
        html.Small("Defaults are research medians (Medicare Part B + out-of-pocket; ACA "
                   "unsubsidized). Genworth 2024 LTC medians: in-home ~$75k, nursing "
                   "~$128k — type the cost for your chosen care type, or set type to "
                   "\"None\" to drop the LTC spike.", className="text-muted d-block mb-3"),

        html.P("Expenses over retirement — living spend, healthcare, and the total",
               className="text-muted small mb-2"),
        dcc.Loading(dcc.Graph(id="co-glide-chart", config={"displayModeBar": False}),
                    type="circle"),
        html.P(id="co-glide-note", className="text-muted small mt-2 mb-0"),

        html.Div([
            dbc.Button("Save expenses & healthcare", id="co-exp-save", color="primary",
                       size="sm", outline=True),
            html.Span(id="co-exp-save-status", className="small ms-2"),
        ], className="mt-3"),
    ],
)


# ── Assumption module 3: Housing & real property ─────────────────────────────────
#
# Placed between expenses and income: it is a spending assumption, and it took the
# `mortgage` / `home_improvements` rows out of the table directly above it.

def _hsg_field(label, control, hint=None, width=None):
    """A labelled Housing input with an optional muted hint beneath."""
    return dbc.Col([
        dbc.Label(label, className="small text-muted mb-1"),
        control,
        html.Small(hint, className="text-muted d-block mt-1") if hint else None,
    ], **({"lg": width} if width else {}))


_housing_module = _module(
    "co-hsg", "fa-house", "Housing & real property", "co-hsg-summary",
    body=[
        html.P([
            "Housing is the one expense that ", html.B("ends on a known date"),
            " — and the one that is also an asset. Describe where you live and what "
            "you own, and the mortgage comes out of the expense table above and is "
            "modelled properly here: principal & interest stop at payoff, escrow and "
            "maintenance carry on forever. Everything is in today's dollars.",
        ], className="text-muted small mb-3"),

        # ── Where you live ───────────────────────────────────────────────────────
        html.Div([html.B("Where you live"),
                  html.Span(" — a pure cost", className="text-muted small ms-1")],
                 className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.RadioItems(
                id="co-hsg-mode", value=None, inline=True,
                options=[{"label": "I own my home", "value": "own"},
                         {"label": "I rent", "value": "rent"}],
                className="small", inputClassName="me-1", labelClassName="me-3 mb-0",
            ), width="auto", className="d-flex align-items-center"),
        ], className="g-2 mb-2"),
        dbc.Row([
            _hsg_field("Rent, $/mo", _money_input("co-hsg-rent", min=0, step=50),
                       "What you pay to live where you live. Required — this is "
                       "your entire housing cost when you rent.", width=3),
            _hsg_field("Rent growth above inflation", _pct_input(
                "co-hsg-rent-growth", step=0.25),
                "0% means rent keeps pace with inflation; negative models rent control.",
                width=4),
        ], id="co-hsg-rent-row", className="g-3 mb-2"),
        # Leaving this blank silently zeroes the entire cost stream while still
        # pulling `mortgage` out of the expense table — the one way this module
        # can quietly understate the nest-egg goal. Say it loudly.
        html.Div(id="co-hsg-rent-warning", className="mb-3"),

        html.Hr(className="my-3"),

        # ── What you own ─────────────────────────────────────────────────────────
        html.Div([
            html.Div([html.B("What you own"),
                      html.Span(" — an asset, a loan, and a cost stream",
                                className="text-muted small ms-1")]),
            dbc.Switch(id="co-hsg-has-prop", value=False, className="mb-0",
                       label="I own a property"),
        ], className="d-flex justify-content-between align-items-center mb-2"),

        html.Div([
            dbc.Row([
                _hsg_field("What it is", dbc.Select(
                    id="co-hsg-occupancy", size="sm", value="primary",
                    options=[{"label": "The home I live in", "value": "primary"},
                             {"label": "Let to tenants (rental)", "value": "rental"}]),
                    "A rental is a tax regime, not a business: its carrying costs "
                    "offset its rent as ordinary income.", width=4),
                _hsg_field("Nickname (optional)", dbc.Input(
                    id="co-hsg-label", type="text", size="sm",
                    placeholder="e.g. Elm St"), width=3),
            ], className="g-3 mb-3"),

            html.P("The loan", className="small fw-bold mb-2"),
            dbc.Row([
                _hsg_field("Total monthly payment, $/mo",
                           _money_input("co-hsg-payment", min=0, step=50),
                           "Usually PITI — the whole thing you pay the servicer.",
                           width=3),
                _hsg_field("…of which escrow, $/mo",
                           _money_input("co-hsg-escrow", min=0, step=25), width=3),
                _hsg_field("Note rate", _pct_input("co-hsg-note-rate", min=0, step=0.125),
                           width=2),
                _hsg_field("Paid off on", dbc.Input(id="co-hsg-payoff", type="date",
                                                    size="sm"), width=3),
            ], className="g-3 mb-1"),
            dbc.Alert([html.I(className="fas fa-circle-info me-2"),
                       html.B("Escrow keeps being paid after payoff"),
                       " — only the principal & interest ends. Property tax tracks "
                       "assessed value and insurance tracks replacement cost, so both "
                       "carry on forever. Splitting the payment correctly is what puts "
                       "the payoff cliff at the right height."],
                      color="light", className="small py-2 mb-3"),

            html.P("The asset", className="small fw-bold mb-2"),
            dbc.Row([
                _hsg_field("Current value, $",
                           _money_input("co-hsg-value", min=0, step=5000), width=3),
                _hsg_field("Purchase price, $",
                           _money_input("co-hsg-price", min=0, step=5000), width=3),
                _hsg_field("Purchased on", dbc.Input(id="co-hsg-purchase-date",
                                                     type="date", size="sm"), width=3),
            ], className="g-3 mb-2"),
            dbc.Row([
                _hsg_field("Appreciation above inflation",
                           _pct_input("co-hsg-appreciation", step=0.25), width=3),
                dbc.Col(html.Small(id="co-hsg-appr-hint", className="text-muted"),
                        className="d-flex align-items-center"),
            ], className="g-3 mb-2"),
            dbc.Row([
                _hsg_field("Maintenance, % of value/yr",
                           _pct_input("co-hsg-maintenance", min=0, step=0.25),
                           "Replaces your home-improvements budget line, which is "
                           "excluded from the expense table above. Goes to zero if "
                           "you rent.", width=4),
            ], className="g-3 mb-3"),

            html.Div(id="co-hsg-rental-block", children=[
                html.P("Rented out", className="small fw-bold mb-2"),
                dbc.Row([
                    _hsg_field("Rent collected, $/mo",
                               _money_input("co-hsg-rent-income", min=0, step=50),
                               width=3),
                    _hsg_field("Vacancy & management",
                               _pct_input("co-hsg-vacancy", min=0, step=1), width=3),
                    _hsg_field("Rent growth above inflation",
                               _pct_input("co-hsg-prop-rent-growth", step=0.25),
                               width=3),
                ], className="g-3 mb-2"),
                html.Small("Rent less this property's carrying costs is ordinary "
                           "income (Schedule E) and is taxed as such. It can go "
                           "negative in a carry-heavy year — that is a real result.",
                           className="text-muted d-block mb-3"),
            ]),
        ], id="co-hsg-property-block"),

        dbc.Row([
            _hsg_field("Inflation (deflates the fixed payment only)",
                       _pct_input("co-hsg-inflation", min=0, step=0.25),
                       "The model works in today's dollars, so a nominally fixed "
                       "principal & interest payment shrinks in real terms every "
                       "year. This is the only place inflation is read.", width=4),
        ], className="g-3 mb-3"),

        html.Hr(className="my-3"),

        # ── The charts — the whole argument for the module in two images ─────────
        html.P("What you own — home value, the loan beneath it, and the equity gap",
               className="text-muted small mb-2"),
        dcc.Loading(dcc.Graph(id="co-hsg-equity-chart",
                              config={"displayModeBar": False}), type="circle"),
        html.P([
            "What it costs — what you pay to live somewhere above the line, what a "
            "property you let out pays you below it, and the net in bold. ",
            html.B("Where the bold line drops below zero, what you own is paying for "
                   "where you live."),
            " Principal & interest falls away at payoff onto the escrow and "
            "maintenance floor that never ends.",
        ], className="text-muted small mt-3 mb-2"),
        dcc.Loading(dcc.Graph(id="co-hsg-cost-chart",
                              config={"displayModeBar": False}), type="circle"),

        html.Div([
            dbc.Button("Save housing", id="co-hsg-save", color="primary", size="sm",
                       outline=True),
            html.Span(id="co-hsg-save-status", className="small ms-2"),
        ], className="mt-3"),
    ],
)


# ── Assumption module 4: Income & tax planning ───────────────────────────────────

_income_module = _module(
    "co-inc", "fa-landmark", "Income & tax planning", "co-inc-summary",
    body=[
        html.P([
            "Social Security and account withdrawals fund the spending above. The lifetime "
            "walk withdraws taxable → tax-deferred → Roth with ", html.B("RMDs"),
            " forced from the tax-deferred balance, and flat planning-grade taxes "
            "(ordinary + long-term capital gains).",
        ], className="text-muted small mb-3"),
        dbc.Row([
            dbc.Col([
                html.P("Current balances", className="small fw-bold mb-2"),
                _bucket_inputs("co-bal", 1000),
                html.Small(id="co-bal-hint", className="text-muted"),
            ], lg=4),
            dbc.Col([
                html.P("Annual contribution split", className="small fw-bold mb-2"),
                _bucket_inputs("co-alloc", 500),
                html.Small(id="co-alloc-hint", className="text-muted"),
            ], lg=4),
            dbc.Col([
                html.P("Social Security & taxes", className="small fw-bold mb-2"),
                dbc.Row([
                    dbc.Col([dbc.Label("Current gross income, $/yr", className="small text-muted mb-1"),
                             _money_input("co-ss-income", value=0, min=0, step=1000),
                             html.Small(id="co-ss-income-hint", className="text-warning")], width=12),
                ], className="g-2"),
                dbc.Row([
                    dbc.Col([dbc.Label("Employment", className="small text-muted mb-1"),
                             dbc.Select(id="co-ss-emptype", size="sm",
                                        options=[{"label": "W-2", "value": "W2"},
                                                 {"label": "1099 (self-emp.)", "value": "1099"},
                                                 {"label": "Mixed", "value": "mixed"}],
                                        value="W2")], width=6),
                    dbc.Col([dbc.Label("Career yrs", className="small text-muted mb-1"),
                             dbc.Input(id="co-ss-career", type="number", value=35, min=1,
                                       max=50, step=1, size="sm")], width=6),
                ], className="g-2 mt-1"),
                dbc.Row([
                    dbc.Col([dbc.Label("SS benefit at FRA (67), $/yr", className="small text-muted mb-1"),
                             _money_input("co-ss-pia", value=0, min=0, step=500),
                             html.Small(id="co-ss-hint", className="text-muted")], width=6),
                    dbc.Col([dbc.Label("Taxable gain %", className="small text-muted mb-1"),
                             dbc.InputGroup([
                                 dbc.Input(id="co-gain-frac", type="number",
                                           value=int(TAXABLE_GAIN_FRACTION * 100),
                                           min=0, max=100, step=5, size="sm"),
                                 dbc.InputGroupText("%")], size="sm"),
                             html.Small(id="co-gain-hint", className="text-muted")], width=6),
                ], className="g-2 mt-1"),
                html.Small("Claim age is the SS handle on the milestone slider above.",
                           className="text-muted d-block mt-1"),
            ], lg=4),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col(html.Div(id="co-income-summary"), lg=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Annual cash flow — Social Security floor vs. portfolio draw",
                       className="text-muted small mb-2"),
                dcc.Loading(dcc.Graph(id="co-cashflow-chart",
                                      config={"displayModeBar": False}), type="circle"),
                html.P("Social Security is the guaranteed floor; the portfolio fills the gap. "
                       "Forced RMDs can spike taxable draws late (the excess is reinvested, "
                       "not spent).", className="text-muted small mt-2 mb-0"),
            ]), color="light"), lg=8),
        ], className="g-3"),

        html.Div([
            dbc.Button("Save income & tax", id="co-inc-save", color="primary",
                       size="sm", outline=True),
            html.Span(id="co-inc-save-status", className="small ms-2"),
        ], className="mt-3"),
    ],
)


# ── Page layout ──────────────────────────────────────────────────────────────────

layout = html.Div(dbc.Container([
    # Network-bound returns analytics (yfinance/CPI), computed once per visit and
    # kept off the recompute hot path (mirrors Forecast's fc-returns-store).
    dcc.Store(id="co-returns-store", storage_type="memory"),
    # The Housing module's form, collected into one HousingConfig-shaped block so
    # `recompute` and `seed_expenses` take a single Input instead of eighteen.
    dcc.Store(id="co-housing-store", storage_type="memory"),
    # What `seed_housing` last hydrated the form from, so an unrelated write to
    # `config-store` cannot silently revert the user's unsaved Housing edits.
    dcc.Store(id="co-hsg-seeded-store", storage_type="memory"),
    # Just `is_configured(housing)` — the expenses table's only stake in Housing.
    dcc.Store(id="co-hsg-active-store", storage_type="memory"),
    dbc.Row([
        dbc.Col(html.H1("Coast"), width="auto"),
        dbc.Col(html.Span(id="co-meta", className="text-muted small"),
                className="d-flex align-items-center"),
    ], className="pt-3 pb-2", align="center"),
    html.P(
        "One screen for the whole scenario: drag the milestone ages and set your "
        "monthly contribution to watch the nest-egg goal, the CoastFIRE target, and "
        "the contribution you need move together.",
        className="text-muted small mb-3",
    ),
    _cockpit,
    _chart_card,

    html.H6("Assumptions — set & forget", className="text-muted text-uppercase mb-2",
            style={"fontSize": ".72rem", "letterSpacing": ".1em"}),
    _rate_module,
    _expenses_module,
    _housing_module,
    _income_module,
    html.P([html.I(className="fas fa-circle-info me-1"),
            "Editing any assumption re-runs the scenario above — the chart and BANs "
            "update live."], className="assist-note mt-2"),
], fluid=False))


# ── Seed the slider + contribution from config (Profile is the truth) ────────────

_module_toggle("co-rr")
_module_toggle("co-exp")
_module_toggle("co-hsg")
_module_toggle("co-inc")


@callback(
    Output("co-milestones", "min"),
    Output("co-milestones", "max"),
    Output("co-milestones", "marks"),
    Output("co-milestones", "value"),
    Output("co-monthly", "value"),
    Output("co-contribution-hint", "children"),
    Output("co-meta", "children"),
    Output("co-strategy", "value"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def seed_inputs(config_data, use_case):
    uid = _selected_uid(use_case)
    user_cfg = _user_cfg(config_data, uid)
    if user_cfg is None:
        raise dash.exceptions.PreventUpdate

    a = resolve_assumptions(user_cfg)
    current_age = _current_age(user_cfg)
    death_age = int(a.get("death_age") or _DEFAULT_DEATH_AGE)
    coast_age = int(user_cfg.get("coast_age") or DEFAULT_COAST_AGE)

    # Slider handles, in milestone order (the slider keeps them monotonic).
    value = [
        coast_age,
        int(a["retirement_age"]),
        int(a["claim_age"]),
        int(a["slow_go_age"]),
        int(a["no_go_age"]),
    ]
    lo, hi = min(current_age, value[0]), max(death_age, value[-1])
    marks = {x: str(x) for x in range(lo - lo % 5, hi + 1, 5)}

    active_plan = _active_plan(user_cfg)
    csp_labels = user_cfg.get("csp_labels") or {}
    monthly = default_monthly_contribution(active_plan, csp_labels)
    hint = ("from your CSP investments plan" if monthly
            else "no CSP plan — enter a monthly amount")

    has_birth = bool(a.get("birth_year"))
    meta = (f"Age {current_age} · planning to {death_age}" if has_birth
            else f"Set your birth date on the Profile page (assuming age {current_age})")

    strategy = (user_cfg.get("retirement") or {}).get("return_strategy") or DEFAULT_STRATEGY

    return lo, hi, marks, value, round(monthly, 2), hint, meta, strategy


# ── Module 2 seeds: expenses table + healthcare inputs from config ────────────────

@callback(
    Output("co-exp-table", "data"),
    Input("config-store", "data"),
    Input("use-case", "value"),
    Input("co-hsg-active-store", "data"),
    State("co-housing-store", "data"),
)
def seed_expenses(config_data, use_case, _active, housing):
    """Re-seed when Housing switches on or off, because that decides whether
    `mortgage` / `home_improvements` belong in this table at all (see
    `_seed_expense_rows`).

    Gated on the on/off flip, not on the block itself: every keystroke in the
    Housing form rewrites `co-housing-store`, and re-seeding on each one would
    discard unsaved cell edits here for no reason — only the exclusion set matters,
    and it only changes when `is_configured` flips. The block still comes in as
    State so `spend_exclude_keys` stays the single expression of that rule."""
    uid = _selected_uid(use_case)
    if not config_data or not isinstance(config_data, str) or not uid:
        raise dash.exceptions.PreventUpdate
    config = json.loads(config_data)
    if uid not in (config.get("users") or {}):
        raise dash.exceptions.PreventUpdate
    return _seed_expense_rows(config, uid, housing)


@callback(
    Output("co-hc-aca", "value"),
    Output("co-hc-medicare", "value"),
    Output("co-hc-ltc-type", "value"),
    Output("co-hc-ltc", "value"),
    Output("co-hc-ltc-age", "value"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def seed_healthcare(config_data, use_case):
    """Hydrate the healthcare card from persisted RetirementConfig.healthcare/ltc_type,
    falling back to the research-median defaults."""
    user_cfg = _user_cfg(config_data, _selected_uid(use_case))
    if user_cfg is None:
        raise dash.exceptions.PreventUpdate
    hc = ((user_cfg.get("retirement") or {}).get("healthcare")) or {}
    ltc_type = (user_cfg.get("retirement") or {}).get("ltc_type") or LTC_DEFAULT_TYPE
    return (
        int(hc.get("aca_annual", _HC_ACA_DEFAULT)),
        int(hc.get("medicare_annual", _HC_MEDICARE_DEFAULT)),
        ltc_type,
        int(hc.get("ltc_annual", LTC_DEFAULT_ANNUAL)),
        int(hc.get("ltc_start_age", LTC_DEFAULT_START_AGE)),
    )


@callback(
    Output("co-exp-table", "data", allow_duplicate=True),
    Input("co-exp-apply-scale", "n_clicks"),
    State("co-exp-table", "data"),
    State("co-exp-slow-scale", "value"),
    State("co-exp-no-scale", "value"),
    State("config-store", "data"),
    State("use-case", "value"),
    prevent_initial_call=True,
)
def apply_scale(_n, rows, slow_pct, no_pct, config_data, use_case):
    """Recompute slow-go/no-go = go-go × per-category research factor × phase-scale%
    (honours manual go-go edits; composes with the per-category shape)."""
    user_cfg = _user_cfg(config_data, _selected_uid(use_case))
    if user_cfg is None or not rows:
        raise dash.exceptions.PreventUpdate
    factors = resolve_phase_factors((user_cfg or {}).get("retirement"))
    _pct = lambda v: 1.0 if v in (None, "") else _num(v) / 100.0
    slow_s, no_s = _pct(slow_pct), _pct(no_pct)
    for row in rows:
        if not row.get("key"):
            continue
        gg = _num(row.get("go_go"))
        f = factors.get(row.get("key"), DEFAULT_PHASE_FACTOR)
        row["slow_go"] = round(gg * f.get("slow_go", 1.0) * slow_s)
        row["no_go"] = round(gg * f.get("no_go", 1.0) * no_s)
    return rows


@callback(
    Output("config-store", "data", allow_duplicate=True),
    Output("co-exp-save-status", "children"),
    Input("co-exp-save", "n_clicks"),
    State("co-exp-table", "data"),
    State("co-hc-aca", "value"),
    State("co-hc-medicare", "value"),
    State("co-hc-ltc-type", "value"),
    State("co-hc-ltc", "value"),
    State("co-hc-ltc-age", "value"),
    State("config-store", "data"),
    State("use-case", "value"),
    prevent_initial_call=True,
)
def save_expenses(n, rows, hc_aca, hc_medicare, ltc_type, hc_ltc, hc_ltc_age,
                  config_data, use_case):
    """Persist edited per-category expenses + healthcare to RetirementConfig and mirror
    into config-store so the in-session view matches Firestore without a reload."""
    if not n:
        raise dash.exceptions.PreventUpdate
    uid = _selected_uid(use_case)
    if not uid or not config_data:
        raise dash.exceptions.PreventUpdate

    expenses = {
        r["key"]: {ph: _num(r.get(ph)) for ph in PHASES}
        for r in (rows or []) if r.get("key")
    }
    healthcare = _healthcare_overrides(hc_aca, hc_medicare, hc_ltc, hc_ltc_age)
    payload = {"expenses": expenses, "healthcare": healthcare,
               "ltc_type": ltc_type or LTC_DEFAULT_TYPE}
    save_retirement_config(uid, payload)

    cfg = json.loads(config_data)
    ret_cfg = cfg.setdefault("users", {}).setdefault(uid, {}).setdefault("retirement", {})
    ret_cfg.update(payload)
    return json.dumps(cfg), dbc.Badge("Saved", color="success", className="ms-1")


# ── Module 3 (Housing): seed, reveal, collect, hint, save ────────────────────────

_HSG_FORM_INPUTS = (
    "co-hsg-mode", "co-hsg-rent", "co-hsg-rent-growth",
    "co-hsg-has-prop", "co-hsg-occupancy", "co-hsg-label",
    "co-hsg-payment", "co-hsg-escrow", "co-hsg-note-rate", "co-hsg-payoff",
    "co-hsg-price", "co-hsg-purchase-date", "co-hsg-value",
    "co-hsg-appreciation", "co-hsg-maintenance",
    "co-hsg-rent-income", "co-hsg-vacancy", "co-hsg-prop-rent-growth",
    "co-hsg-inflation",
)


@callback(
    *[Output(i, "value") for i in _HSG_FORM_INPUTS],
    Output("co-hsg-seeded-store", "data"),
    Input("config-store", "data"),
    Input("use-case", "value"),
    State("co-hsg-seeded-store", "data"),
)
def seed_housing(config_data, use_case, seeded):
    """Hydrate the Housing form from the persisted `RetirementConfig.housing` block.

    Absent → every field blank except the research/long-run defaults, and crucially
    `mode` unselected: `housing.is_configured` reads an unset mode plus an empty
    property list as "says nothing", which is what keeps an untouched form inert and
    the scenario byte-identical to before this module existed.

    **Only re-seeds when the stored block actually changes.** `config-store` is
    rewritten by every save on this page (expenses, income, rate) and by five other
    pages; re-hydrating on each of those would blow away whatever the user has typed
    here but not yet saved — silently, and then persist the reverted value on their
    next Save. `co-hsg-seeded-store` remembers what was last seeded so an unrelated
    config write is a no-op. Switching "Viewing as" still re-seeds, because the uid
    is part of the comparison.
    """
    uid = _selected_uid(use_case)
    user_cfg = _user_cfg(config_data, uid)
    if user_cfg is None:
        raise dash.exceptions.PreventUpdate

    block = _housing_block(user_cfg)
    if seeded and seeded.get("uid") == uid and seeded.get("block") == block:
        raise dash.exceptions.PreventUpdate
    res = block.get("residence") or {}
    prop = _first_property(block)
    return (
        res.get("mode"),
        res.get("monthly_rent"),
        _to_pct(res.get("real_rent_growth"), _HSG_PCT_DEFAULTS["rent_growth"]),
        bool(prop),
        prop.get("occupancy") or "primary",
        prop.get("label"),
        prop.get("monthly_payment"),
        prop.get("escrow_monthly"),
        _to_pct(prop.get("note_rate"), 0.0),
        prop.get("payoff_date"),
        prop.get("purchase_price"),
        prop.get("purchase_date"),
        prop.get("current_value"),
        _to_pct(prop.get("real_appreciation"), _HSG_PCT_DEFAULTS["appreciation"]),
        _to_pct(prop.get("maintenance_pct"), _HSG_PCT_DEFAULTS["maintenance"]),
        prop.get("monthly_rent_income"),
        _to_pct(prop.get("vacancy_pct"), _HSG_PCT_DEFAULTS["vacancy"]),
        _to_pct(prop.get("real_rent_growth"), _HSG_PCT_DEFAULTS["prop_rent_growth"]),
        _to_pct(block.get("inflation"), _HSG_PCT_DEFAULTS["inflation"]),
        {"uid": uid, "block": block},
    )


@callback(
    Output("co-hsg-rent-row", "style"),
    Output("co-hsg-property-block", "style"),
    Output("co-hsg-rental-block", "style"),
    Input("co-hsg-mode", "value"),
    Input("co-hsg-has-prop", "value"),
    Input("co-hsg-occupancy", "value"),
)
def toggle_housing_sections(mode, has_prop, occupancy):
    """Reveal only the inputs that apply: rent fields when renting, the property
    block when one is owned, the tenant fields when it is let."""
    hide = {"display": "none"}
    return (
        None if mode == "rent" else hide,
        None if has_prop else hide,
        None if occupancy == "rental" else hide,
    )


@callback(
    Output("co-hsg-active-store", "data"),
    Input("co-housing-store", "data"),
    State("co-hsg-active-store", "data"),
)
def housing_active(block, prev):
    """Whether the Housing module is switched on — the only thing about it the
    expenses table cares about. Held apart from `co-housing-store` and short-
    circuited when unchanged so a keystroke doesn't re-seed that table."""
    active = is_configured(block)
    if active == prev:
        raise dash.exceptions.PreventUpdate
    return active


@callback(
    Output("co-hsg-rent-warning", "children"),
    Input("co-hsg-mode", "value"),
    Input("co-hsg-rent", "value"),
)
def warn_missing_rent(mode, rent):
    """A renter with no rent amount is the module's one silent-failure mode: the
    cost stream is $0 while `mortgage` has already left the expense table, so the
    nest-egg goal is understated with nothing on screen to explain it."""
    if mode != "rent" or rent not in (None, ""):
        return None
    return dbc.Alert([
        html.I(className="fas fa-triangle-exclamation me-2"),
        html.B("Enter your rent above."),
        " Until you do, your housing cost is $0 — and because the Housing module "
        "is on, the mortgage row has already left the expense table, so the "
        "nest-egg goal above is understated.",
    ], color="warning", className="small py-2 mb-0")


@callback(
    Output("co-housing-store", "data"),
    *[Input(i, "value") for i in _HSG_FORM_INPUTS],
)
def collect_housing(*values):
    """The whole form as one HousingConfig-shaped block, parked in a store.

    Keeps `recompute` and `seed_expenses` to a single Input each rather than
    eighteen, mirroring how `co-returns-store` carries the returns payload.
    """
    return _housing_from_form(*values)


@callback(
    Output("co-hsg-appr-hint", "children"),
    Input("co-hsg-price", "value"),
    Input("co-hsg-value", "value"),
    Input("co-hsg-purchase-date", "value"),
    Input("co-returns-store", "data"),
)
def housing_appreciation_hint(price, value, purchase_date, returns_store):
    """What this home has *actually* done, as context beneath the forward assumption.

    Display only, never a default: a 2019 purchase yields ~8%/yr real, and
    extrapolating that for 30 years is fantasy — hence the long-run ~1% default in
    the input itself. CPI comes from the returns payload's `inflation` map (the
    market-data seam); with no holdings that map is empty and the figure degrades
    to a *nominal* CAGR, which the hint says explicitly.
    """
    if not price or not value or not purchase_date:
        return "Long-run US housing runs about 1%/yr above inflation."
    try:
        purchase_year = int(str(purchase_date)[:4])
    except (TypeError, ValueError):
        return ""

    cpi = {int(k): float(v) for k, v in ((returns_store or {}).get("inflation") or {}).items()}
    series = pd.Series(cpi, dtype="float64") if cpi else None
    rate = derive_real_appreciation(_num(price), _num(value), purchase_year,
                                    date.today().year, series)
    if rate is None:
        return "Long-run US housing runs about 1%/yr above inflation."

    frame = "above inflation" if cpi else "nominal (no CPI data — upload holdings)"
    verdict = ("well above" if rate > 0.03 else "below" if rate < 0.005 else "close to")
    return (f"Yours has appreciated {rate:.1%}/yr {frame} since {purchase_year} — "
            f"{verdict} the long-run ~1%. Shown as context; the forward assumption "
            f"defaults to the long-run rate.")


@callback(
    Output("config-store", "data", allow_duplicate=True),
    Output("co-hsg-save-status", "children"),
    Input("co-hsg-save", "n_clicks"),
    State("co-housing-store", "data"),
    State("config-store", "data"),
    State("use-case", "value"),
    prevent_initial_call=True,
)
def save_housing(n, block, config_data, use_case):
    """Persist the Housing block to RetirementConfig and mirror into config-store.

    The whole block is written every time, explicit `None`s included: Firestore's
    `merge=True` deep-merges nested maps, so a partial write would leave cleared
    fields behind at their old values.
    """
    if not n:
        raise dash.exceptions.PreventUpdate
    uid = _selected_uid(use_case)
    if not uid or not config_data or block is None:
        raise dash.exceptions.PreventUpdate

    save_retirement_config(uid, {"housing": block})

    cfg = json.loads(config_data)
    ret_cfg = cfg.setdefault("users", {}).setdefault(uid, {}).setdefault("retirement", {})
    ret_cfg["housing"] = block
    return json.dumps(cfg), dbc.Badge("Saved", color="success", className="ms-1")


# ── Module 4 seeds: balances, allocation, Social Security income, gain % ──────────

@callback(
    Output("co-bal-taxable", "value"),
    Output("co-bal-trad", "value"),
    Output("co-bal-roth", "value"),
    Output("co-bal-hint", "children"),
    Output("co-ss-income", "value"),
    Output("co-ss-income-hint", "children"),
    Output("co-gain-frac", "value"),
    Output("co-gain-hint", "children"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def seed_income(config_data, use_case):
    """Seed current balances (from holdings), the gross-income line (from the CSP
    income plan), and the taxable-gain % (from the cost-basis report) — all editable
    overrides (ported from pages/retirement.py seed_income)."""
    uid = _selected_uid(use_case)
    user_cfg = _user_cfg(config_data, uid)
    if user_cfg is None:
        raise dash.exceptions.PreventUpdate
    accounts = user_cfg.get("investment_accounts")

    buckets = balances_by_tax_bucket(uid, accounts)
    bal_hint = (f"auto-filled from your holdings ({_money(buckets['total'])} total) — editable"
                if buckets["total"] > 0 else "no holdings found — enter your current balances")

    active_plan = _active_plan(user_cfg)
    csp_labels = user_cfg.get("csp_labels") or {}
    annual_income = 12.0 * sum(float(a) for cat, a in active_plan.items()
                               if csp_labels.get(cat) == "income")
    income_hint = (f"seeded from your CSP income ({_money(annual_income)}/yr) — likely net; "
                   "enter gross for accuracy" if annual_income > 0
                   else "enter your current gross annual income to estimate Social Security")

    derived = taxable_gain_fraction(uid, accounts)
    if derived is not None:
        gain_pct = round(derived * 100)
        gain_hint = f"derived from your cost-basis report ({gain_pct}% gain) — editable"
    else:
        gain_pct = int(TAXABLE_GAIN_FRACTION * 100)
        gain_hint = "default — upload your Vanguard cost-basis CSV to derive this"

    return (round(buckets["taxable"]), round(buckets["trad"]), round(buckets["roth"]), bal_hint,
            round(annual_income), income_hint, gain_pct, gain_hint)


@callback(
    Output("co-alloc-taxable", "value"),
    Output("co-alloc-trad", "value"),
    Output("co-alloc-roth", "value"),
    Output("co-alloc-hint", "children"),
    Input("config-store", "data"),
    Input("use-case", "value"),
    Input("co-monthly", "value"),
)
def seed_allocation(config_data, use_case, monthly):
    """Split the annual contribution tax-advantaged-first across buckets, preferring a
    persisted RetirementConfig.contribution_allocation. Re-seeds when the monthly
    contribution changes so the split tracks it; each bucket stays editable."""
    user_cfg = _user_cfg(config_data, _selected_uid(use_case))
    if user_cfg is None:
        raise dash.exceptions.PreventUpdate
    annual = float(monthly or 0) * 12.0
    saved = (user_cfg.get("retirement") or {}).get("contribution_allocation")
    alloc = saved if saved else default_contribution_allocation(annual)
    hint = ("tax-advantaged-first split of your contribution — editable" if annual > 0
            else "set a monthly contribution above to split")
    return round(alloc.get("taxable", 0)), round(alloc.get("trad", 0)), round(alloc.get("roth", 0)), hint


@callback(
    Output("co-ss-pia", "value"),
    Output("co-ss-hint", "children"),
    Input("co-ss-income", "value"),
    Input("co-ss-emptype", "value"),
    Input("co-ss-career", "value"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def estimate_ss(gross_income, employment_type, career_years, config_data, use_case):
    """Estimate the SS benefit at FRA, preferring the Profile income history
    (income_segments → top-35 PIA), else the constant-income estimate. The field
    stays an editable SSA-statement override (ported from pages/retirement.py)."""
    emptype = employment_type or "W2"
    user_cfg = _user_cfg(config_data, _selected_uid(use_case)) or {}
    segments = user_cfg.get("income_segments")

    if segments:
        earnings = segments_to_annual_income(segments)
        pia = annual_pia_from_earnings(earnings, emptype)
        n = int((earnings > 0).sum())
        hint = (f"estimated from your Profile income history ({n} year{'' if n == 1 else 's'}) — "
                "edit on the Profile page, or type your SSA figure to override")
        return round(pia), hint

    income = _num(gross_income)
    years = int(career_years) if career_years else 35
    pia = estimate_annual_pia_from_income(income, emptype, years)
    if income > 0:
        kind = {"1099": "1099 self-employment", "mixed": "mixed"}.get(emptype, "W-2")
        hint = (f"estimated from {_money(income)}/yr {kind} over {years} yrs — add income "
                "history on the Profile page, or type your SSA figure to override")
    else:
        hint = "enter gross income, add Profile income history, or type your SSA figure"
    return round(pia), hint


@callback(
    Output("config-store", "data", allow_duplicate=True),
    Output("co-inc-save-status", "children"),
    Input("co-inc-save", "n_clicks"),
    State("co-alloc-taxable", "value"),
    State("co-alloc-trad", "value"),
    State("co-alloc-roth", "value"),
    State("co-ss-pia", "value"),
    State("co-gain-frac", "value"),
    State("config-store", "data"),
    State("use-case", "value"),
    prevent_initial_call=True,
)
def save_income(n, al_taxable, al_trad, al_roth, ss_pia, gain_frac, config_data, use_case):
    """Persist the contribution allocation, SS benefit (PIA) and taxable-gain fraction
    to RetirementConfig and mirror into config-store."""
    if not n:
        raise dash.exceptions.PreventUpdate
    uid = _selected_uid(use_case)
    if not uid or not config_data:
        raise dash.exceptions.PreventUpdate

    payload = {
        "contribution_allocation": {"taxable": _num(al_taxable), "trad": _num(al_trad),
                                    "roth": _num(al_roth)},
        "pia_annual": _num(ss_pia),
        "gain_fraction": (_num(gain_frac) or 0.0) / 100.0,
    }
    save_retirement_config(uid, payload)

    cfg = json.loads(config_data)
    ret_cfg = cfg.setdefault("users", {}).setdefault(uid, {}).setdefault("retirement", {})
    ret_cfg.update(payload)
    return json.dumps(cfg), dbc.Badge("Saved", color="success", className="ms-1")


# ── Recompute: BANs + chart from build_scenario ──────────────────────────────────

@callback(
    Output("co-milestone-chips", "children"),
    Output("co-ban-goal", "children"),
    Output("co-ban-coast-target", "children"),
    Output("co-ban-coast-status", "children"),
    Output("co-ban-required", "children"),
    Output("co-projection-chart", "figure"),
    Output("co-rr-summary", "children"),
    Output("co-glide-chart", "figure"),
    Output("co-glide-note", "children"),
    Output("co-exp-totals", "children"),
    Output("co-exp-note", "children"),
    Output("co-exp-summary", "children"),
    Output("co-hsg-summary", "children"),
    Output("co-hsg-equity-chart", "figure"),
    Output("co-hsg-cost-chart", "figure"),
    Output("co-cashflow-chart", "figure"),
    Output("co-income-summary", "children"),
    Output("co-inc-summary", "children"),
    Input("co-milestones", "value"),
    Input("co-monthly", "value"),
    Input("co-color-mode", "value"),
    Input("co-strategy", "value"),
    Input("co-returns-store", "data"),
    Input("co-exp-table", "data"),
    Input("co-hc-aca", "value"),
    Input("co-hc-medicare", "value"),
    Input("co-hc-ltc", "value"),
    Input("co-hc-ltc-age", "value"),
    Input("co-bal-taxable", "value"),
    Input("co-bal-trad", "value"),
    Input("co-bal-roth", "value"),
    Input("co-alloc-taxable", "value"),
    Input("co-alloc-trad", "value"),
    Input("co-alloc-roth", "value"),
    Input("co-ss-pia", "value"),
    Input("co-gain-frac", "value"),
    Input("co-housing-store", "data"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def recompute(milestones, monthly, color_mode, strategy, returns_store, exp_rows,
              hc_aca, hc_medicare, hc_ltc, hc_ltc_age,
              bal_taxable, bal_trad, bal_roth, al_taxable, al_trad, al_roth,
              ss_pia, gain_frac, housing, config_data, use_case):
    uid = _selected_uid(use_case)
    user_cfg = _user_cfg(config_data, uid)
    if user_cfg is None or not milestones or len(milestones) != 5:
        raise dash.exceptions.PreventUpdate

    coast_age, retirement_age, claim_age, slow_go_age, no_go_age = (int(v) for v in milestones)
    chips = _milestone_chips(coast_age, retirement_age, claim_age, slow_go_age, no_go_age)

    active_plan = _active_plan(user_cfg)
    csp_labels = user_cfg.get("csp_labels") or {}

    # Current balances: the Income module's editable inputs, falling back to the
    # holdings snapshot before they seed (so the page never flashes a $0 portfolio).
    bal = {"taxable": _num(bal_taxable), "trad": _num(bal_trad), "roth": _num(bal_roth)}
    if sum(bal.values()) > 0:
        buckets = {**bal, "total": sum(bal.values())}
    else:
        buckets = balances_by_tax_bucket(uid, user_cfg.get("investment_accounts"))

    # Return glide path from the rate-of-return module: the chosen strategy + the
    # holdings-derived two-leg params (parked in co-returns-store off the hot path).
    # No holdings/params → build_scenario falls back to the config flat rate.
    glide_params = (returns_store or {}).get("glide")

    # Live expenses + healthcare overrides from the Expenses module. The table drives
    # the spend stream; None (table not seeded yet) → build_scenario's CSP-derived
    # default so the page still renders on first paint.
    spend_override = _spend_from_table(exp_rows) if exp_rows else None
    hc_overrides = _healthcare_overrides(hc_aca, hc_medicare, hc_ltc, hc_ltc_age)

    # Income & tax overrides: contribution split, SS benefit, taxable-gain fraction.
    alloc = {"taxable": _num(al_taxable), "trad": _num(al_trad), "roth": _num(al_roth)}
    alloc_override = alloc if sum(alloc.values()) > 0 else None
    pia_override = _num(ss_pia) if ss_pia is not None else None
    gain_override = (_num(gain_frac) / 100.0) if gain_frac is not None else None

    # Single resolution path: build_scenario merges the live inputs over config.
    # death_age stays config-resolved (the slider owns the 5 milestone handles).
    s = build_scenario(
        user_cfg, active_plan, csp_labels, buckets,
        monthly_contribution=float(monthly or 0),
        coast_age=coast_age, retirement_age=retirement_age, claim_age=claim_age,
        slow_go_age=slow_go_age, no_go_age=no_go_age,
        return_strategy=strategy, glide_params=glide_params,
        spend_by_phase=spend_override, healthcare_overrides=hc_overrides,
        allocation=alloc_override, pia_annual=pia_override, gain_fraction=gain_override,
        # Live Housing form; None (store not filled yet) → the persisted block, so
        # the first paint already reflects a saved configuration.
        housing_overrides=housing,
    )
    if s.return_strategy and s.equity_glide:
        ages = sorted(int(a) for a in s.equity_glide.keys())
        w0 = float(s.equity_glide[ages[0]]) * 100.0
        w_term = STRATEGY_TERMINAL_EQUITY[s.return_strategy] * 100.0
        path = (f"holds {w0:.0f}% equity" if w0 <= w_term + 1e-9
                else f"{w0:.0f}%→{w_term:.0f}% equity")
        rr_summary = (f"{STRATEGY_LABELS[s.return_strategy]} glide · {path} · "
                      f"{s.real_return * 100:.1f}% real")
    elif s.return_strategy:
        rr_summary = f"{STRATEGY_LABELS[s.return_strategy]} glide · {s.real_return * 100:.1f}% real"
    else:
        rr_summary = f"{s.real_return * 100:.1f}% real"

    # ── BANs: goal → coast target → projected-at-coast → required monthly ─────────
    ban_goal = _ban("Nest-egg goal", _money(s.nest_egg_goal))
    ban_target = _ban("CoastFIRE target", _money(s.coast_target),
                      f"needed at age {coast_age}")

    if s.coast_surplus >= 0:
        status_sub, status_cls = f"+{_money(s.coast_surplus)} over target", "good"
    else:
        status_sub, status_cls = f"{_money(-s.coast_surplus)} short", "warn"
    ban_status = _ban("Projected at coast", _money(s.projected_at_coast),
                      status_sub, value_class=status_cls)

    req = s.required_monthly
    if req <= 0:
        req_sub, req_cls = "on track — $0 needed", "good"
    else:
        delta = req - s.monthly_contribution
        req_cls = "good" if delta <= 0 else "warn"
        req_sub = (f"contributing {_money(s.monthly_contribution)}/mo" if delta <= 0
                   else f"+{_money(delta)}/mo to get on track")
    ban_required = _ban("Required monthly", f"${req:,.0f}/mo", req_sub, value_class=req_cls)

    fig = _projection_figure(s, color_mode)

    # ── Expenses module outputs (glide smile + summaries) ────────────────────────
    glide = _glide_figure(s.spend_by_phase, s.healthcare_by_age, s.housing_by_age,
                          retirement_age, slow_go_age, no_go_age, s.death_age)
    spend = s.spend_by_phase
    totals = (f"Annual: go-go {_money(spend['go_go'])} · slow-go {_money(spend['slow_go'])} "
              f"· no-go {_money(spend['no_go'])}")
    n_rows = sum(1 for r in (exp_rows or []) if r.get("key"))
    # Note the folded-in share of household shared expenses (see _effective_expense_plan)
    # so the seeded totals aren't a mystery for a member of a household.
    share = _effective_expense_plan(json.loads(config_data), uid)[2] if config_data else 0.0
    share_bit = f" (incl. your {share:.0%} share of household expenses)" if share else ""
    # Housing takes the mortgage / home-improvement rows out of this table when it
    # is configured; say so, or the missing rows read as a bug.
    housing_bit = (" Mortgage and home improvements are modelled in Housing below."
                   if s.housing_by_age is not None else "")
    note = (f"{n_rows} living-expense categories from your CSP plan{share_bit}, stepping down "
            "at each phase boundary. Healthcare is modelled separately below."
            + housing_bit
            ) if n_rows else "No CSP plan found — add one on the CSP page to seed these expenses."
    health = {int(k): float(v) for k, v in
              (s.healthcare_by_age.to_dict() if s.healthcare_by_age is not None else {}).items()}
    hc_peak = max(health.values()) if health else 0.0
    housing_note = ""
    if s.housing_by_age is not None and len(s.housing_by_age):
        housing_note = (f" Housing starts at {_money(float(s.housing_by_age.iloc[0]))} "
                        "and is modelled on its own (below), because it ends on a "
                        "known date rather than tapering.")
    glide_note = (f"Living spend steps down from {_money(max(spend.values()) if spend else 0)} "
                  f"while healthcare climbs to {_money(hc_peak)} late — their sum is the "
                  "spending smile that drives the nest-egg goal above." + housing_note)
    ltc_bit = (f" · LTC from {int(hc_ltc_age)}" if _num(hc_ltc) > 0 else "")
    exp_summary = f"{_money(spend['go_go'])}/yr go-go{ltc_bit}"

    # ── Housing module outputs ───────────────────────────────────────────────────
    hsg_summary = _housing_summary(s, housing or _housing_block(user_cfg))
    hsg_equity = _housing_equity_figure(s.home_equity)
    hsg_cost = _housing_cost_figure(s.housing_breakdown, retirement_age)

    # ── Income & tax module outputs (cash-flow chart + summary) ──────────────────
    ret_df = s.lifetime[s.lifetime["life_phase"] == "retirement"]
    cashflow = _cashflow_figure(ret_df, retirement_age, slow_go_age, no_go_age, s.death_age)
    birth_year = resolve_assumptions(user_cfg).get("birth_year")
    rmd_start = rmd_start_age(birth_year) if birth_year else 75
    income_summary = _income_summary_card(ret_df, s.ss_by_age, claim_age, rmd_start)
    ss_monthly = (float(s.ss_by_age.max()) / 12.0) if len(s.ss_by_age) else 0.0
    inc_summary = f"SS {_money(ss_monthly)}/mo at {claim_age}"

    return (chips, ban_goal, ban_target, ban_status, ban_required, fig, rr_summary,
            glide, glide_note, totals, note, exp_summary,
            hsg_summary, hsg_equity, hsg_cost,
            cashflow, income_summary, inc_summary)


# ── Figure ───────────────────────────────────────────────────────────────────────

def _principal_growth(lifetime, start_value):
    """Principal (start + cumulative contributions, clipped to total) and growth
    (total − principal) across the whole lifetime — same split Forecast's chart
    and donut use, so the accumulation half reads identically.
    """
    total = lifetime["total"]
    cum_contrib = lifetime["contribution"].fillna(0.0).cumsum().shift(1).fillna(0.0)
    principal = (float(start_value) + cum_contrib).clip(upper=total)
    growth = (total - principal).clip(lower=0.0)
    return principal, growth


def _projection_figure(s, mode) -> go.Figure:
    lifetime = s.lifetime
    ages = lifetime.index.tolist()
    fig = go.Figure()

    if mode == "bucket":
        for col, name, color, fill in (
            ("taxable", "Taxable", _C_PRINCIPAL, "rgba(91,126,201,0.45)"),
            ("trad", "Tax-deferred", _C_TRAD, "rgba(139,92,246,0.35)"),
            ("roth", "Roth", _C_ROTH, "rgba(20,184,166,0.35)"),
        ):
            fig.add_trace(go.Scatter(
                x=ages, y=lifetime[col], name=name, mode="lines",
                line=dict(width=0, color=color), stackgroup="v", fillcolor=fill,
                hovertemplate=f"Age %{{x}}<br>{name} %{{y:$,.0f}}<extra></extra>",
            ))
    else:
        principal, growth = _principal_growth(lifetime, s.start_value)
        fig.add_trace(go.Scatter(
            x=ages, y=principal, name="Principal", mode="lines",
            line=dict(width=0, color=_C_PRINCIPAL), stackgroup="v",
            fillcolor="rgba(91,126,201,0.45)",
            hovertemplate="Age %{x}<br>Principal %{y:$,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=ages, y=growth, name="Growth", mode="lines",
            line=dict(width=0, color=_C_GROWTH), stackgroup="v",
            fillcolor="rgba(74,222,128,0.35)",
            hovertemplate="Age %{x}<br>Growth %{y:$,.0f}<extra></extra>",
        ))

    # Total value line on top of the stack.
    fig.add_trace(go.Scatter(
        x=ages, y=lifetime["total"], name="Total value", mode="lines",
        line=dict(width=2.5, color=_C_TOTAL),
        hovertemplate="Age %{x}<br>Total %{y:$,.0f}<extra></extra>",
    ))

    # Phase shading: working → coast → retirement.
    fig.add_vrect(x0=ages[0], x1=s.coast_age, fillcolor=_C_WORKING, opacity=0.06,
                  layer="below", line_width=0, annotation_text="Working",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color=_C_WORKING))
    fig.add_vrect(x0=s.coast_age, x1=s.retirement_age, fillcolor=_C_GROWTH, opacity=0.06,
                  layer="below", line_width=0, annotation_text="Coast",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color="#3a9d5d"))
    fig.add_vrect(x0=s.retirement_age, x1=ages[-1], fillcolor=_C_TARGET, opacity=0.05,
                  layer="below", line_width=0, annotation_text="Retirement",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color="#b8770a"))

    # Nest-egg goal — horizontal target line (the number needed AT retirement).
    fig.add_hline(y=s.nest_egg_goal, line_dash="dash", line_color=_C_TARGET, line_width=1.5,
                  annotation_text=f"Nest-egg goal {_money(s.nest_egg_goal)}",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color=_C_TARGET))

    # CoastFIRE target marker at coast age (what you need to stop contributing).
    fig.add_trace(go.Scatter(
        x=[s.coast_age], y=[s.coast_target], name="CoastFIRE target",
        mode="markers+text",
        marker=dict(size=11, color=_C_TARGET, symbol="diamond",
                    line=dict(width=1, color="white")),
        text=[f"Coast target {_money(s.coast_target)}"], textposition="top center",
        textfont=dict(size=10, color="#b9791b"),
        hovertemplate=f"Coast target {_money(s.coast_target)} at age {s.coast_age}<extra></extra>",
    ))

    fig.update_layout(
        **_CHART_LAYOUT, height=380, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Age", showgrid=False),
        yaxis=dict(title=None, tickprefix="$", tickformat="~s", showgrid=True,
                   gridcolor="#eee", zeroline=False),
        hovermode="x unified",
    )
    return fig


def _glide_figure(spend, hc, housing, retirement_age, slow_go_age, no_go_age,
                  death_age) -> go.Figure:
    """Annual expenses over retirement: living spend (step-down across the three
    phases), healthcare on its own line (the late-life rise), housing when the
    module is configured (the payoff cliff), and their sum — the rising "smile"
    that drives the nest-egg goal (ported from pages/retirement.py).

    Housing has to be in the total: once the module is on, `mortgage` and
    `home_improvements` are out of the living-spend pool, so a total without it
    under-reports the smile by the whole housing amount."""
    ages = list(range(int(retirement_age), int(death_age) + 1))
    living = [spend.get(phase_for_age(a, slow_go_age, no_go_age), 0.0) for a in ages]
    health = {int(k): float(v) for k, v in (hc.to_dict() if hc is not None else {}).items()}
    hc_y = [health.get(a, 0.0) for a in ages]
    house = {int(k): float(v) for k, v in
             (housing.to_dict() if housing is not None else {}).items()}
    hsg_y = [house.get(a, 0.0) for a in ages]
    total = [lv + h + hs for lv, h, hs in zip(living, hc_y, hsg_y)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ages, y=living, mode="lines", name="Living spend",
        line=dict(width=2.2, color="#7fb3f5", shape="hv"), fill="tozeroy",
        fillcolor="rgba(127,179,245,0.12)",
        hovertemplate="Age %{x}<br>living spend %{y:$,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=hc_y, mode="lines", name="Healthcare",
        line=dict(width=2.2, color=_C_TARGET, shape="hv"),
        hovertemplate="Age %{x}<br>healthcare %{y:$,.0f}<extra></extra>",
    ))
    if housing is not None:
        fig.add_trace(go.Scatter(
            x=ages, y=hsg_y, mode="lines", name="Housing",
            line=dict(width=2.2, color=_C_HOUSE_VALUE, shape="hv"),
            hovertemplate="Age %{x}<br>housing %{y:$,.0f}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=ages, y=total, mode="lines", name="Total expenses",
        line=dict(width=2.8, color="#444", shape="hv"),
        hovertemplate="Age %{x}<br>total %{y:$,.0f}<extra></extra>",
    ))
    fig.add_vline(x=slow_go_age, line_width=1, line_dash="dot", line_color="#cbd5e1")
    fig.add_vline(x=no_go_age, line_width=1, line_dash="dot", line_color="#cbd5e1")
    fig.update_layout(
        **_CHART_LAYOUT, height=300, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Age", showgrid=False),
        yaxis=dict(title=None, tickprefix="$", tickformat="~s", showgrid=True,
                   gridcolor="#eee", zeroline=False, rangemode="tozero"),
        hovermode="x unified", showlegend=True,
    )
    return fig


# ── Module 3 (Housing) figures ───────────────────────────────────────────────────

def _housing_equity_figure(eq) -> go.Figure:
    """Home value, the loan beneath it, and the equity gap between them.

    `eq` is `Scenario.home_equity` (value / balance / equity by age) or None. The
    stack is debt-then-equity so the two shaded bands sum to the value line — the
    balance amortizes to zero and the whole band becomes equity, which is the
    picture the module exists to show. Neither band is spendable, which is why it
    lives here and not on the portfolio chart above.
    """
    if eq is None or eq.empty:
        return _detail_empty("Describe a property above to see its equity build.")

    ages = eq.index.tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ages, y=eq["balance"], name="Mortgage balance", mode="lines",
        line=dict(width=0, color=_C_HOUSE_DEBT), stackgroup="h",
        fillcolor="rgba(154,165,177,0.45)",
        hovertemplate="Age %{x}<br>balance %{y:$,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=eq["equity"], name="Equity", mode="lines",
        line=dict(width=0, color=_C_GROWTH), stackgroup="h",
        fillcolor="rgba(74,222,128,0.35)",
        hovertemplate="Age %{x}<br>equity %{y:$,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=eq["value"], name="Home value", mode="lines",
        line=dict(width=2.5, color=_C_HOUSE_VALUE),
        hovertemplate="Age %{x}<br>value %{y:$,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **_CHART_LAYOUT, height=300, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Age", showgrid=False),
        yaxis=dict(title=None, tickprefix="$", tickformat="~s", showgrid=True,
                   gridcolor="#eee", zeroline=False, rangemode="tozero"),
        hovermode="x unified",
    )
    return fig


def _housing_cost_figure(breakdown, retirement_age=None) -> go.Figure:
    """Annual housing cost above the line, rental income below it, net in bold.

    Three things at once, which is what housing is:

      * **Above zero** — the components of *where you live*, stacked:
        `Scenario.housing_breakdown`'s escrow + maintenance floor beneath the
        principal & interest that terminates at payoff, or a single rent band.
        Drawing the total alone would show the cliff but not what it lands on,
        which is the single most misunderstood thing about a mortgage. Empty
        components are skipped so the legend stays honest.
      * **Below zero** — net rental income as a credit, because it is money that
        offsets the bill above even though the engine carries it on a separate
        (taxed) channel. Plotting it downward is what lets the two be compared
        instead of merely co-displayed.
      * **The bold line** — the net of the two. Where it crosses below zero, what
        you own is paying for where you live. For the own-a-rental-and-rent-your-
        home case that crossing IS the answer, and nothing else on the page
        states it.

    Net rental income may itself be negative in a carry-heavy year; it is not
    clamped — per the module's sign convention that is a real result, and it
    simply pushes the credit band above the line where it belongs.

    Spans current age → death, matching the portfolio chart above, so the payoff
    cliff is visible in context rather than appearing mid-story at retirement.
    Only the part from `retirement_age` on feeds the nest-egg goal (before then
    these costs are already inside your CSP plan), so that boundary is marked
    rather than left implicit.
    """
    if breakdown is None or breakdown.empty:
        return _detail_empty("Set where you live above to see your housing costs.")

    ages = breakdown.index.tolist()
    rental_net = breakdown.get("rental_net")
    has_rental = rental_net is not None and bool(rental_net.abs().sum())
    fig = go.Figure()

    for col, name, color, fill in _HOUSING_COST_TRACES:
        if col not in breakdown.columns or not breakdown[col].abs().sum():
            continue
        fig.add_trace(go.Scatter(
            x=ages, y=breakdown[col], name=name, mode="lines",
            line=dict(width=0, color=color, shape="hv"), stackgroup="c",
            fillcolor=fill,
            hovertemplate=f"Age %{{x}}<br>{name.lower()} %{{y:$,.0f}}<extra></extra>",
        ))

    if has_rental:
        fig.add_trace(go.Scatter(
            x=ages, y=-rental_net, name="Net rental income (credit)", mode="lines",
            line=dict(width=0, color=_C_GROWTH, shape="hv"), fill="tozeroy",
            fillcolor="rgba(74,222,128,0.30)",
            customdata=rental_net.tolist(),
            hovertemplate="Age %{x}<br>net rental income %{customdata:$,.0f}"
                          "<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=ages, y=breakdown["net"],
        name="Net housing cost" if has_rental else "Total housing cost",
        mode="lines", line=dict(width=2.8, color="#444", shape="hv"),
        hovertemplate="Age %{x}<br>net housing %{y:$,.0f}<extra></extra>",
    ))
    if has_rental:
        # The gross cost stays visible as a faint reference, so the credit is
        # legible as a reduction of something rather than a free-floating band.
        fig.add_trace(go.Scatter(
            x=ages, y=breakdown["total"], name="Cost of where you live",
            mode="lines", line=dict(width=1.4, color="#999", dash="dot", shape="hv"),
            hovertemplate="Age %{x}<br>where you live %{y:$,.0f}<extra></extra>",
        ))
        fig.add_hline(y=0, line_color="#bbb", line_width=1)

    # Before retirement these costs are already inside the CSP plan, so they are
    # shown for context but do not drive the goal. Say so on the chart.
    if retirement_age is not None and ages and ages[0] < int(retirement_age):
        fig.add_vrect(
            x0=ages[0], x1=int(retirement_age), fillcolor="#94a3b8", opacity=0.07,
            layer="below", line_width=0, annotation_text="already in your budget",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#7a869a"),
        )
        fig.add_vline(x=int(retirement_age), line_width=1, line_dash="dot",
                      line_color="#cbd5e1")

    fig.update_layout(
        **_CHART_LAYOUT, height=300, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Age", showgrid=False),
        yaxis=dict(title=None, tickprefix="$", tickformat="~s", showgrid=True,
                   gridcolor="#eee", zeroline=False),
        hovermode="x unified",
    )
    return fig


def _cashflow_figure(df, retirement_age, slow_go_age, no_go_age, death_age) -> go.Figure:
    """Annual cash flow over the (always tax-aware) drawdown: Social Security floor
    (green) + portfolio draw stacked on top (blue), with the tax line (amber).
    `df` is the retirement-phase slice of the lifetime walk (ported from
    pages/retirement.py)."""
    ages = df.index.tolist()
    ss = df["income"] if "income" in df.columns else [0] * len(ages)
    # Portfolio cash funding living = net draw above taxes/reinvested RMD.
    port = (df["net_spend"] - df["income"]).clip(lower=0) if "net_spend" in df.columns \
        else df.get("withdrawal", [0] * len(ages))
    tax = df["tax"] if "tax" in df.columns else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ages, y=ss, mode="lines", name="Social Security",
        line=dict(width=0, color=_C_GROWTH), stackgroup="cf",
        fillcolor="rgba(74,222,128,0.35)",
        hovertemplate="Age %{x}<br>Social Security %{y:$,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=port, mode="lines", name="Portfolio draw",
        line=dict(width=0, color="#7fb3f5"), stackgroup="cf",
        fillcolor="rgba(127,179,245,0.40)",
        hovertemplate="Age %{x}<br>Portfolio %{y:$,.0f}<extra></extra>",
    ))
    if tax is not None:
        fig.add_trace(go.Scatter(
            x=ages, y=tax, mode="lines", name="Tax",
            line=dict(width=1.6, color=_C_TARGET, dash="dot"),
            hovertemplate="Age %{x}<br>Tax %{y:$,.0f}<extra></extra>",
        ))
    for b in (slow_go_age, no_go_age):
        fig.add_vline(x=b, line_width=1, line_dash="dot", line_color="#cbd5e1")
    fig.update_layout(
        **_CHART_LAYOUT, height=260, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Age", showgrid=False),
        yaxis=dict(title=None, tickprefix="$", tickformat="~s", showgrid=True,
                   gridcolor="#eee", zeroline=False, rangemode="tozero"),
        hovermode="x unified",
    )
    return fig


def _income_summary_card(df, ss, claim_age, rmd_start):
    """Small stat block summarizing the income/tax picture over the drawdown
    (ported from pages/retirement.py; the Coast walk is always tax-aware)."""
    if df is None or df.empty:
        return dbc.Alert("Set your milestone ages to project the drawdown.",
                         color="light", className="small mb-0")
    lifetime_tax = float(df["tax"].sum())
    ss_annual = float(ss.max()) if len(ss) else 0.0
    first = df.iloc[0]
    first_draw = float(first["withdrawal"])
    eff = (float(first["tax"]) / first_draw) if first_draw else 0.0

    def _row(label, value, cls=""):
        return html.Div([
            html.Span(label, className="text-muted small"),
            html.Span(value, className=f"small fw-bold float-end {cls}"),
        ], className="mb-1")

    return dbc.Card(dbc.CardBody([
        html.P("Income & tax summary", className="small fw-bold mb-2"),
        _row(f"SS benefit (claim {claim_age})", f"{_money(ss_annual)}/yr", "text-success"),
        _row("RMDs begin", f"age {rmd_start}"),
        _row("First-year tax", f"{_money(first['tax'])} ({eff:.0%})", "text-warning"),
        _row("Lifetime taxes", _money(lifetime_tax), "text-warning"),
    ]), className="h-100")


# ── Module 1 (Rate of return): returns analytics + suggested rate + save ──────────

@callback(
    Output("co-returns-store", "data"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def compute_returns_data(config_data, use_case):
    """Network-bound yfinance/CPI return analytics, run ONCE per visit and parked in
    its own store — the recompute hot path never touches the network (mirrors
    Forecast's compute_returns_data)."""
    uid = _selected_uid(use_case)
    if not uid:
        raise dash.exceptions.PreventUpdate
    return build_returns_payload(fetch_latest_holdings(uid))


@callback(
    Output("co-fan-chart", "figure"),
    Output("co-fan-caption", "children"),
    Input("co-returns-store", "data"),
    Input("co-milestones", "value"),
    Input("co-strategy", "value"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def render_returns_chart(store, milestones, strategy, config_data, use_case):
    """Rate-of-return story on one calendar-year timeline:

      • Faint lines — each leg's *actual* annual real return (equity proxy + bond
        proxy), historical only. They swing wildly: any single year is a coin toss.
      • Bold line — the *consequence*: the portfolio's average real return (CAGR).
        Past = trailing CAGR of today's mix; future = the cumulative CAGR you'll realise
        gliding down to the chosen terminal allocation.
      • Shaded band — the long-run uncertainty around the *future* CAGR (25–75 standard-
        error), which tightens with horizon and with a more conservative glide.

    The forward annual spread is deliberately omitted — single-year resampled returns
    are too optimistic to anchor a plan on; only the converging average is shown forward.
    Portfolio value and contributions don't affect the *rate*, so they don't appear."""
    if not store or not store.get("ok"):
        return _detail_empty((store or {}).get("message", "No data") if store else "Loading…"), ""

    uid = _selected_uid(use_case)
    user_cfg = _user_cfg(config_data, uid)
    if user_cfg is None or not milestones or len(milestones) != 5:
        return _detail_empty(""), ""

    current_age = _current_age(user_cfg)
    retirement_age = int(milestones[1])
    horizon_years = max(retirement_age - current_age, 0)

    weighted_avg = {int(k): float(v) for k, v in (store.get("weighted_avg") or {}).items()}
    glide = store.get("glide")
    eq_pool, fx_pool = store.get("equity_pool"), store.get("fixed_pool")

    # Faint background: each asset class's *actual* annual real return — the real
    # inputs to the trailing CAGR. Kept as faint, unlabelled background texture (no
    # legend entry) — only the short class name shows on hover.
    class_real = {c: s for c, s in (store.get("class_real_returns") or {}).items()
                  if c != "Cash"}
    class_labels = store.get("class_labels") or {}
    ordered = ([c for c in _CLASS_ORDER if c in class_real]
               + [c for c in class_real if c not in _CLASS_ORDER])
    class_lines = []
    for cls in ordered:
        pts = {int(y): float(r) for y, r in class_real[cls].items()}
        if not pts:
            continue
        yrs = sorted(pts)
        class_lines.append({
            "name": cls,
            "years": yrs,
            "rets": [pts[y] for y in yrs],
            "color": _CLASS_LINE_COLORS.get(cls, "#bbbbbb"),
        })

    use_glide = bool(glide and eq_pool and fx_pool and horizon_years > 0
                     and strategy in STRATEGY_TERMINAL_EQUITY)

    weights = cagr_proj = None
    if use_glide:
        w0, w_term = glide["w_equity0"], STRATEGY_TERMINAL_EQUITY[strategy]
        wpath = equity_weight_path(w0, w_term, current_age, retirement_age)
        weights = [wpath.get(current_age + t, w_term) for t in range(horizon_years)]
        cagr_proj = glide_cagr_projection(eq_pool, fx_pool, weights,
                                          glide["r_equity"], glide["r_fixed"])
        path = (f"holds {w0 * 100:.0f}% equity" if w0 <= w_term
                else f"{w0 * 100:.0f}% → {w_term * 100:.0f}% equity")
        # Fixed leg uses your actual bonds when held, the proxy otherwise.
        fixed_label = "/".join(class_labels.get("Bonds") or [BOND_PROXY])
        caption = (
            f"Faint lines are each asset class's actual annual real return from your "
            f"holdings — the real inputs to the bold CAGR (geometric mean). Past the "
            f"\"today\" line the model collapses to a two-leg equity-vs-bond mix "
            f"(bonds = {fixed_label}) and projects the CAGR forward along your "
            f"{STRATEGY_LABELS[strategy].lower()} glide ({path}). The band is the 25th–75th "
            f"standard-error interval, tightening as the average spans more years and the "
            f"mix de-risks toward bonds."
        )
    else:
        caption = ("Add holdings on the Investments page (and pick a strategy) to "
                   "project this CAGR forward along a glide.")

    fig = _returns_figure(
        weighted_avg, class_lines, weights, cagr_proj,
        current_age, retirement_age, date.today().year,
    )
    return fig, caption


@callback(
    Output("config-store", "data", allow_duplicate=True),
    Output("co-rr-save-status", "children"),
    Input("co-rr-save", "n_clicks"),
    State("co-strategy", "value"),
    State("config-store", "data"),
    State("use-case", "value"),
    prevent_initial_call=True,
)
def save_rate(n, strategy, config_data, use_case):
    """Persist the return-glide strategy to RetirementConfig and mirror it into
    config-store so the in-session view matches Firestore without a reload."""
    if not n:
        raise dash.exceptions.PreventUpdate
    uid = _selected_uid(use_case)
    if not uid or not config_data or strategy not in STRATEGY_TERMINAL_EQUITY:
        raise dash.exceptions.PreventUpdate

    save_retirement_config(uid, {"return_strategy": strategy})

    cfg = json.loads(config_data)
    ret_cfg = cfg.setdefault("users", {}).setdefault(uid, {}).setdefault("retirement", {})
    ret_cfg["return_strategy"] = strategy
    return json.dumps(cfg), dbc.Badge("Saved", color="success", className="ms-1")


# ── Module 1 figure builders (ported from forecast.py) ────────────────────────────

def _detail_empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_CHART_LAYOUT, height=280, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[{"text": message, "xref": "paper", "yref": "paper",
                      "x": 0.5, "y": 0.5, "showarrow": False,
                      "font": {"color": "#999", "size": 13}}] if message else [],
    )
    return fig


def _trailing_cagr(weighted_avg: dict[int, float], window: int | None = None,
                   min_years: int = 5) -> tuple[list[int], list[float]]:
    """Trailing geometric-mean (CAGR) of the portfolio's real annual returns, ending at
    each historical year. With `window=None` (the default) it expands over all available
    history, so the rightmost point spans the same `n_years` the forward projection
    resamples; pass `window` to cap it to a rolling N-year mean. Needs at least
    `min_years` to plot a point. Returns (years, cagr)."""
    years = sorted(weighted_avg)
    rets = [weighted_avg[y] for y in years]
    span = window or len(years)
    out_years: list[int] = []
    out_cagr: list[float] = []
    for i in range(len(years)):
        w = rets[max(0, i - span + 1): i + 1]
        if len(w) < min_years:
            continue
        growth = 1.0
        for r in w:
            growth *= (1.0 + r)
        out_years.append(years[i])
        out_cagr.append(growth ** (1.0 / len(w)) - 1.0)
    return out_years, out_cagr


# SE band fill around the future CAGR.
_FILL_INNER = "rgba(91,126,201,0.28)"   # 25–75 standard-error band


def _returns_figure(weighted_avg, class_lines, weights, cagr_proj,
                    current_age, retirement_age, current_year) -> go.Figure:
    """Single panel on a calendar-year x-axis (see render_returns_chart).

    Faint per-leg annual returns (`class_lines`) give the wild year-to-year texture;
    the bold CAGR line (trailing past + projected future) is the emphasised signal; the
    shaded band is the future-CAGR standard-error interval. `weights`/`cagr_proj` are
    None when there's no glide to project — then only the historical halves render."""
    fig = go.Figure()

    # ── Faint background: each class's actual annual real return (historical only) ─
    # Unlabelled texture — kept out of the legend; the class name shows only on hover.
    for cl in class_lines:
        fig.add_trace(go.Scatter(
            x=cl["years"], y=cl["rets"], mode="lines", name=cl["name"],
            line=dict(width=1, color=cl["color"]), opacity=0.7, showlegend=False,
            hovertemplate=f"{cl['name']} %{{y:.1%}}<extra></extra>",
        ))

    horizon = len(weights) if weights else 0
    fut_years = [current_year + 1 + t for t in range(horizon)]

    # ── Future CAGR standard-error band (drawn before the lines so it sits behind) ─
    cagr_years, cagr_vals = _trailing_cagr(weighted_avg)
    if cagr_proj and cagr_proj.get("expected") and horizon:
        anchor = cagr_vals[-1] if cagr_vals else cagr_proj["expected"][0]
        xc = [current_year] + fut_years
        fig.add_trace(go.Scatter(x=xc, y=[anchor] + cagr_proj[75], mode="lines",
                                 line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xc, y=[anchor] + cagr_proj[25], mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor=_FILL_INNER, name="CAGR 25–75 (long-run)",
                                 hoverinfo="skip"))

    # ── Emphasised: trailing CAGR (past) → projected CAGR (future), one bold story ─
    if cagr_years:
        fig.add_trace(go.Scatter(
            x=cagr_years, y=cagr_vals, mode="lines", name="Average return (CAGR)",
            line=dict(width=3, color=_C_TOTAL),
            hovertemplate="trailing CAGR %{y:.1%}<extra></extra>",
        ))
    if cagr_proj and cagr_proj.get("expected") and horizon:
        anchor = cagr_vals[-1] if cagr_vals else cagr_proj["expected"][0]
        fig.add_trace(go.Scatter(
            x=[current_year] + fut_years, y=[anchor] + cagr_proj["expected"],
            mode="lines", name="Projected CAGR",
            line=dict(width=3, color=_C_WORKING),
            hovertemplate="projected CAGR %{y:.1%}<extra></extra>",
        ))
        # End-of-line readout: the small CAGR scale makes strategy differences subtle,
        # so pin the projected value to the terminal point (mirrors the card header).
        # An annotation, not a data point, so it never reads as an extra trace/marker.
        end_cagr = cagr_proj["expected"][-1]
        fig.add_annotation(
            x=fut_years[-1], y=end_cagr, text=f"{end_cagr:.1%}",
            showarrow=False, xanchor="left", yanchor="middle", xshift=6,
            font=dict(size=12, color=_C_WORKING),
        )

    # Zero reference + the "today" divider.
    fig.add_hline(y=0, line_color="#e3e3e3", line_width=1)
    if horizon:
        fig.add_vline(x=current_year, line_color="#bbb", line_width=1, line_dash="dot")
        fig.add_annotation(x=current_year, yref="paper", y=1.0, text="today",
                           showarrow=False, font=dict(size=10, color="#999"),
                           xanchor="left", yanchor="bottom")

    fig.update_layout(
        **_CHART_LAYOUT, height=420, margin=dict(l=10, r=48, t=30, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".0%", showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_xaxes(showgrid=False, title="Year")
    return fig
