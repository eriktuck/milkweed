"""Retirement page — v1 scaffold (Concept A "Single-Scroll Narrative").

The inverse of Forecast: works **backward** from realistic, time-varying
retirement spending (go-go / slow-go / no-go) to the nest-egg goal, then shows
that goal drawing down from retirement to death. See
.planning/spec-retirement.md and .planning/phase-3-design.md (Concept A chosen).

Phase 4 built the top of the scroll: the assumptions header + drawdown chart +
BANs, using the light v1 model in core/services/retirement.py. Phase 5 adds the
editable go-go/slow-go/no-go **Expenses** calculator below, whose edits flow into
the projection. Income and Risk remain placeholders (Phases 6–7; Income gets
Concept C's annual cash-flow chart).

Math lives in core/services/retirement.py; this module is layout + callbacks.
Scoped to the user selected in the global "Viewing as" (`use-case`) dropdown,
mirroring the Forecast page's single-individual scope.
"""

import datetime
import json

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html
from flask import session

import core.utils.functions as functions
from core.services.forecast import default_monthly_contribution
from core.services.retirement import (
    DEFAULT_PHASE_FACTOR,
    PHASES,
    TAXABLE_GAIN_FRACTION,
    annual_spend_by_phase,
    balances_by_tax_bucket,
    default_contribution_allocation,
    estimate_annual_pia_from_income,
    nest_egg_goal,
    phase_for_age,
    project_balances_to_retirement,
    project_retirement,
    project_retirement_taxaware,
    resolve_assumptions,
    resolve_phase_factors,
    retirement_summary,
    rmd_start_age,
    social_security_income,
    taxable_gain_fraction,
)

dash.register_page(__name__, path="/retirement")

# ── Phase colors (match the chosen mockup, retirement-a.html) ────────────────────
_C_GOGO = "#7fb3f5"      # blue — go-go (spend peaks)
_C_SLOWGO = "#4ade80"    # green — slow-go (tapering)
_C_NOGO = "#f59e0b"      # amber — no-go (healthcare rises)
_C_TOTAL = "#e0e0e0"     # nest-egg balance line
_C_PRINCIPAL = "#5b7ec9"

_CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="#444"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


# ── Formatting helpers (mirror pages/forecast.py) ────────────────────────────────

def _money(v) -> str:
    """Abbreviated currency: $1.50M / $548K / $900."""
    v = float(v or 0)
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:,.0f}K"
    return f"${v:,.0f}"


def _ban_card(label: str, value: str, subtitle, value_class: str = "") -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.P(label, className="text-muted small mb-1"),
        html.H3(value, className=f"mb-0 fw-bold {value_class}"),
        html.P(subtitle, className="text-muted small mt-1 mb-0"),
    ]))


def _num_input(id_, value, **kw):
    return dbc.Input(id=id_, type="number", value=value, size="sm", **kw)


# ── Assumptions header ───────────────────────────────────────────────────────────

_input_bar = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([
        dbc.Label("Birth year", className="small text-muted mb-1"),
        _num_input("ret-birth-year", None, min=1920, max=2010, step=1,
                   placeholder="needed"),
        html.Small(id="ret-birth-hint", className="text-warning"),
    ], width="auto"),
    dbc.Col([
        dbc.Label("Retirement age", className="small text-muted mb-1"),
        _num_input("ret-retirement-age", 65, min=40, max=90, step=1),
    ], width="auto"),
    dbc.Col([
        dbc.Label("Death age", className="small text-muted mb-1"),
        _num_input("ret-death-age", 90, min=70, max=110, step=1),
    ], width="auto"),
    dbc.Col([
        dbc.Label("Slow-go starts", className="small text-muted mb-1"),
        _num_input("ret-slow-go-age", 75, min=60, max=100, step=1),
    ], width="auto"),
    dbc.Col([
        dbc.Label("No-go starts", className="small text-muted mb-1"),
        _num_input("ret-no-go-age", 85, min=65, max=105, step=1),
    ], width="auto"),
    dbc.Col([
        dbc.Label("Claim age (SS)", className="small text-muted mb-1"),
        _num_input("ret-claim-age", 67, min=62, max=70, step=1),
    ], width="auto"),
    dbc.Col([
        dbc.Label("Real return", className="small text-muted mb-1"),
        dbc.InputGroup([
            _num_input("ret-real-return", 5.0, min=0, max=12, step=0.5),
            dbc.InputGroupText("%"),
        ], size="sm"),
        html.Small("inflation-adjusted", className="text-muted"),
    ], width="auto"),
], className="g-3 align-items-end")), className="mb-3")


# ── Placeholder section (Expenses / Income / Risk filled in Phases 5–7) ──────────

def _placeholder(title: str, phase: str, body: str) -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.Div([
            html.H4(title, className="d-inline mb-0"),
            dbc.Badge(phase, color="secondary", className="ms-2 align-middle"),
        ], className="mb-2"),
        html.P(body, className="text-muted small mb-0"),
    ]), className="mb-3", style={"borderStyle": "dashed"})


# ── Expenses calculator (Phase 5) ────────────────────────────────────────────────

# csp-key column metadata (last column holds the per-row factor hint, read-only).
_EXPENSE_COLUMNS = [
    {"name": "CSP category", "id": "category", "editable": False},
    {"name": "Go-go $/mo", "id": "go_go", "type": "numeric", "editable": True},
    {"name": "Slow-go $/mo", "id": "slow_go", "type": "numeric", "editable": True},
    {"name": "No-go $/mo", "id": "no_go", "type": "numeric", "editable": True},
    {"name": "factor", "id": "factor", "editable": False},
]

_expenses_table = dash_table.DataTable(
    id="ret-expenses-table",
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
    ],
)

_expenses_section = dbc.Card(dbc.CardBody([
    html.Div([
        html.H4("Retirement expenses — go-go / slow-go / no-go", className="d-inline mb-0"),
        dbc.Badge("editable", color="success", className="ms-2 align-middle"),
    ], className="mb-1"),
    html.P([
        "Go-go is seeded from your latest CSP plan at the category level. Slow-go and "
        "no-go are proposed with research multipliers — edit any cell, or rescale a whole "
        "phase below. Healthcare is modelled separately (it rises late), so ",
        html.B("medical & health insurance are excluded here."),
    ], className="text-muted small mb-3"),

    dbc.Row([
        dbc.Col(_expenses_table, lg=8),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Annual living-spend glide", className="text-muted small mb-2"),
            dcc.Graph(id="ret-glide-chart", config={"displayModeBar": False}),
            html.P(id="ret-expenses-note", className="text-muted small mt-2 mb-0"),
        ])), lg=4),
    ], className="g-3 align-items-start"),

    html.Hr(className="my-3"),
    dbc.Row([
        dbc.Col(html.Span("Rescale a whole phase from go-go (keeps the research shape):",
                          className="small text-muted"), width="auto",
                className="d-flex align-items-center"),
        dbc.Col([
            dbc.Label("Slow-go", className="small text-muted mb-1"),
            dbc.InputGroup([
                dbc.Input(id="ret-slow-scale", type="number", value=100, min=0, max=200,
                          step=5, size="sm", style={"width": "70px"}),
                dbc.InputGroupText("%"),
            ], size="sm"),
        ], width="auto"),
        dbc.Col([
            dbc.Label("No-go", className="small text-muted mb-1"),
            dbc.InputGroup([
                dbc.Input(id="ret-no-scale", type="number", value=100, min=0, max=200,
                          step=5, size="sm", style={"width": "70px"}),
                dbc.InputGroupText("%"),
            ], size="sm"),
        ], width="auto"),
        dbc.Col(dbc.Button("Apply", id="ret-apply-scale", color="secondary", size="sm",
                           outline=True), width="auto", className="d-flex align-items-end"),
        dbc.Col(html.Span(id="ret-expenses-totals", className="small fw-bold"),
                className="d-flex align-items-end"),
    ], className="g-3 align-items-end"),
]), className="mb-3")


# ── Income calculator (Phase 6) ──────────────────────────────────────────────────

def _money_input(id_, **kw):
    return dbc.InputGroup([dbc.InputGroupText("$"),
                           dbc.Input(id=id_, type="number", size="sm", **kw)], size="sm")


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


# Balance projector (Phase 6b): current holdings + planned contributions → the
# at-retirement balances that feed the drawdown. The three "Projected" inputs
# (ret-bal-*) are written by the projector callback and remain editable overrides.
_projector_card = dbc.Card(dbc.CardBody([
    html.P("Project your balances to retirement", className="small fw-bold mb-1"),
    html.P([
        "Your current holdings grow with planned contributions until retirement. "
        "Contributions default to your CSP ", html.B("investments"), " plan, split ",
        html.B("tax-advantaged-first"), " (pre-tax to its cap, then Roth, then taxable). "
        "The projected balances feed the drawdown — edit any field to override.",
    ], className="text-muted small mb-3"),
    dbc.Row([
        dbc.Col([
            html.P("Current balances", className="small fw-bold mb-2"),
            _bucket_inputs("ret-cur", 1000),
            html.Small(id="ret-cur-hint", className="text-muted"),
        ], lg=4),
        dbc.Col([
            html.P("Annual contribution", className="small fw-bold mb-2"),
            _bucket_inputs("ret-alloc", 500),
            html.Small(id="ret-alloc-hint", className="text-muted"),
        ], lg=4),
        dbc.Col([
            html.P("Projected at retirement", className="small fw-bold mb-2"),
            _bucket_inputs("ret-bal", 1000),
            html.Small(id="ret-proj-hint", className="text-muted"),
        ], lg=4),
    ], className="g-3"),
]), className="mb-3", color="light")

_income_section = dbc.Card(dbc.CardBody([
    html.Div([
        html.H4("Retirement income — what offsets the draw", className="d-inline mb-0"),
        dbc.Badge("tax-aware", color="info", className="ms-2 align-middle"),
    ], className="mb-1"),
    html.P([
        "Social Security and account withdrawals fund the spending above. Withdrawals "
        "follow taxable → tax-deferred → Roth, with ", html.B("RMDs"), " forced from the "
        "tax-deferred balance at your start age, and flat planning-grade taxes (ordinary + "
        "long-term capital gains). The drawdown projects from the at-retirement balances below.",
    ], className="text-muted small mb-3"),

    _projector_card,

    dbc.Row([
        # Social Security (estimated from income) + gain assumption
        dbc.Col([
            html.P("Social Security & taxes", className="small fw-bold mb-2"),
            # Estimate the benefit from income; the SSA figure is the override.
            dbc.Row([
                dbc.Col([dbc.Label("Current gross income, $/yr",
                                   className="small text-muted mb-1"),
                         _money_input("ret-ss-income", value=0, min=0, step=1000),
                         html.Small(id="ret-ss-income-hint", className="text-warning")], width=5),
                dbc.Col([dbc.Label("Employment", className="small text-muted mb-1"),
                         dbc.Select(id="ret-ss-emptype", size="sm",
                                    options=[{"label": "W-2", "value": "W2"},
                                             {"label": "1099 (self-emp.)", "value": "1099"},
                                             {"label": "Mixed", "value": "mixed"}],
                                    value="W2")], width=4),
                dbc.Col([dbc.Label("Career yrs", className="small text-muted mb-1"),
                         _num_input("ret-ss-career", 35, min=1, max=50, step=1)], width=3),
            ], className="g-2"),
            dbc.Row([
                dbc.Col([dbc.Label("SS benefit at FRA (67), $/yr",
                                   className="small text-muted mb-1"),
                         _money_input("ret-ss-pia", value=0, min=0, step=500),
                         html.Small(id="ret-ss-hint", className="text-muted")], width=6),
                dbc.Col([dbc.Label("Taxable gain %", className="small text-muted mb-1"),
                         dbc.InputGroup([
                             dbc.Input(id="ret-gain-frac", type="number",
                                       value=int(TAXABLE_GAIN_FRACTION * 100),
                                       min=0, max=100, step=5, size="sm"),
                             dbc.InputGroupText("%")], size="sm"),
                         html.Small("share of taxable sales that is gain",
                                    id="ret-gain-hint", className="text-muted")], width=6),
            ], className="g-2 mt-1"),
            html.Small("Claim age is set in the header above; it scales this benefit.",
                       className="text-muted d-block mt-1"),
        ], lg=8),
        # Income summary
        dbc.Col(html.Div(id="ret-income-summary"), lg=4),
    ], className="g-3 mb-3"),

    dbc.Card(dbc.CardBody([
        html.P("Annual cash flow — Social Security floor vs. portfolio draw",
               className="text-muted small mb-2"),
        dcc.Graph(id="ret-cashflow-chart", config={"displayModeBar": False}),
        html.P("Social Security is the guaranteed floor; the portfolio fills the gap. "
               "Forced RMDs can spike taxable draws late (the excess is reinvested, "
               "not spent). Healthcare's late-life rise is layered in the next phase.",
               className="text-muted small mt-2 mb-0"),
    ]), color="light"),
]), className="mb-3")


# ── Page layout ─────────────────────────────────────────────────────────────────

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Retirement"), width="auto"),
            dbc.Col(
                html.Span(id="ret-meta", className="text-muted small"),
                className="d-flex align-items-center",
            ),
        ], className="pt-3 pb-1", align="center"),
        html.P(
            "The inverse of Forecast: start from realistic, time-varying spending — "
            "go-go, slow-go, no-go — and read off the nest egg you need, then watch it "
            "draw down to your planning horizon.",
            className="text-muted small mb-3",
        ),

        _input_bar,

        # BANs: goal → avg spend → first-year draw → remaining at death
        dbc.Row([
            dbc.Col(id="ret-ban-goal", width=3),
            dbc.Col(id="ret-ban-avg-spend", width=3),
            dbc.Col(id="ret-ban-first-draw", width=3),
            dbc.Col(id="ret-ban-remaining", width=3),
        ], className="mb-3 g-3"),

        # Drawdown chart
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Nest egg from retirement to death — the drawdown",
                       className="text-muted small mb-2"),
                dcc.Loading(dcc.Graph(id="ret-drawdown-chart",
                                      config={"displayModeBar": False}), type="circle"),
                html.P(id="ret-insight", className="text-muted small mt-2 mb-0"),
            ])), width=12),
        ], className="mb-4"),

        # ── Expenses calculator (Phase 5) ────────────────────────────────────────
        _expenses_section,
        # ── Income calculator (Phase 6) ──────────────────────────────────────────
        _income_section,
        _placeholder(
            "Risk & late-life", "Phase 7",
            "Sequence-of-returns fan chart and probability of success, the HealthCare "
            "glide (Medicare/IRMAA), and an editable long-term-care spike.",
        ),
    ], fluid=False),
])


# ── Helpers shared by the callbacks ──────────────────────────────────────────────

def _selected_uid(use_case):
    """The user the page is scoped to: the 'Viewing as' selection, else the
    logged-in user (mirrors Forecast's single-individual scope)."""
    return use_case or session.get("user_id")


def _user_cfg(config_data, uid):
    if not config_data or not isinstance(config_data, str) or not uid:
        return None
    config = json.loads(config_data)
    return config.get("users", {}).get(uid)


def _pretty(key: str) -> str:
    """csp-key → display label (e.g. 'home_other' → 'Home Other')."""
    return key.replace("_", " ").title()


def _factors_for(user_cfg) -> dict:
    return resolve_phase_factors(user_cfg.get("retirement"))


def _seed_expense_rows(user_cfg) -> list[dict]:
    """One editable row per living-expense csp key, monthly, seeded from the CSP
    plan (go-go) and the research multipliers (slow-go/no-go).

    Reuses the service's filtering (drops contributions/income/healthcare) by
    going through annual_spend_by_phase, then divides back to monthly. The
    per-row `factor` hint and `key` (hidden) ride along in the data.
    """
    csp_labels = user_cfg.get("csp_labels") or {}
    csp_plans = user_cfg.get("csp_plans") or {}
    active_plan = functions.get_active_csp_plan(csp_plans) or user_cfg.get("csp_plan") or {}
    factors = _factors_for(user_cfg)
    by_phase = annual_spend_by_phase(active_plan, csp_labels, factors)

    rows = []
    for key, gg_annual in by_phase["go_go"]["by_key"].items():
        f = factors.get(key, DEFAULT_PHASE_FACTOR)
        rows.append({
            "key": key,
            "category": _pretty(key),
            "go_go": round(by_phase["go_go"]["by_key"][key] / 12),
            "slow_go": round(by_phase["slow_go"]["by_key"][key] / 12),
            "no_go": round(by_phase["no_go"]["by_key"][key] / 12),
            "factor": f"{f.get('slow_go', 1.0):.0%}/{f.get('no_go', 1.0):.0%}",
        })
    return sorted(rows, key=lambda r: r["go_go"], reverse=True)


def _num(v) -> float:
    """Coerce an editable DataTable cell (may be '' / str / None) to float."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _spend_from_table(rows) -> dict:
    """Annual go-go/slow-go/no-go living spend = 12 × Σ each phase column."""
    rows = rows or []
    return {ph: 12.0 * sum(_num(r.get(ph)) for r in rows) for ph in PHASES}


# ── Seed assumption inputs from the user's config ────────────────────────────────

@callback(
    Output("ret-birth-year", "value"),
    Output("ret-retirement-age", "value"),
    Output("ret-death-age", "value"),
    Output("ret-slow-go-age", "value"),
    Output("ret-no-go-age", "value"),
    Output("ret-claim-age", "value"),
    Output("ret-real-return", "value"),
    Output("ret-birth-hint", "children"),
    Output("ret-meta", "children"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def seed_assumptions(config_data, use_case):
    uid = _selected_uid(use_case)
    user_cfg = _user_cfg(config_data, uid)
    if user_cfg is None:
        raise dash.exceptions.PreventUpdate

    a = resolve_assumptions(user_cfg.get("retirement"))
    birth_hint = "" if a["birth_year"] else "needed — drives RMD age (set on the page)"
    meta = (
        f"Viewing as {user_cfg.get('name', uid).title()} · spending seeded from your "
        f"latest CSP plan"
    )
    return (
        a["birth_year"], a["retirement_age"], a["death_age"], a["slow_go_age"],
        a["no_go_age"], a["claim_age"], a["real_return"] * 100,
        birth_hint, meta,
    )


# ── Seed the editable expenses table from the CSP plan + multipliers ─────────────

@callback(
    Output("ret-expenses-table", "data"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def seed_expenses(config_data, use_case):
    user_cfg = _user_cfg(config_data, _selected_uid(use_case))
    if user_cfg is None:
        raise dash.exceptions.PreventUpdate
    return _seed_expense_rows(user_cfg)


# ── Rescale slow-go / no-go from go-go (keeps the research per-category shape) ────

@callback(
    Output("ret-expenses-table", "data", allow_duplicate=True),
    Input("ret-apply-scale", "n_clicks"),
    State("ret-expenses-table", "data"),
    State("ret-slow-scale", "value"),
    State("ret-no-scale", "value"),
    State("config-store", "data"),
    State("use-case", "value"),
    prevent_initial_call=True,
)
def apply_scale(_n, rows, slow_pct, no_pct, config_data, use_case):
    """Recompute slow-go/no-go = go-go × per-category factor × phase-scale%.

    Respects manual go-go edits (reads the current go-go cell) and reapplies the
    research multipliers, so 'scale a whole phase by a percentage' composes with
    the per-category shape rather than flattening it.
    """
    user_cfg = _user_cfg(config_data, _selected_uid(use_case))
    if user_cfg is None or not rows:
        raise dash.exceptions.PreventUpdate
    factors = _factors_for(user_cfg)
    # Only fall back to 100% when the field is cleared (None/'') — honour an
    # explicit 0% (Python treats 0 as falsy, so `or` would wrongly reset it).
    _pct = lambda v: 1.0 if v in (None, "") else _num(v) / 100.0
    slow_s, no_s = _pct(slow_pct), _pct(no_pct)
    for row in rows:
        gg = _num(row.get("go_go"))
        f = factors.get(row.get("key"), DEFAULT_PHASE_FACTOR)
        row["slow_go"] = round(gg * f.get("slow_go", 1.0) * slow_s)
        row["no_go"] = round(gg * f.get("no_go", 1.0) * no_s)
    return rows


# ── Seed current balances, contribution allocation, gain % from the user's data ──

@callback(
    Output("ret-cur-taxable", "value"),
    Output("ret-cur-trad", "value"),
    Output("ret-cur-roth", "value"),
    Output("ret-cur-hint", "children"),
    Output("ret-alloc-taxable", "value"),
    Output("ret-alloc-trad", "value"),
    Output("ret-alloc-roth", "value"),
    Output("ret-alloc-hint", "children"),
    Output("ret-ss-income", "value"),
    Output("ret-ss-income-hint", "children"),
    Output("ret-gain-frac", "value"),
    Output("ret-gain-hint", "children"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def seed_income(config_data, use_case):
    uid = _selected_uid(use_case)
    user_cfg = _user_cfg(config_data, uid)
    if user_cfg is None:
        raise dash.exceptions.PreventUpdate
    accounts = user_cfg.get("investment_accounts")

    # Current balances by tax bucket, from holdings.
    buckets = balances_by_tax_bucket(uid, accounts)
    if buckets["total"] > 0:
        cur_hint = f"auto-filled from your holdings ({_money(buckets['total'])} total) — editable"
    else:
        cur_hint = "no holdings found — enter your current balances"

    # Contribution allocation: default the annual CSP `investments` contribution,
    # split tax-advantaged-first across buckets. Each bucket stays editable.
    csp_labels = user_cfg.get("csp_labels") or {}
    csp_plans = user_cfg.get("csp_plans") or {}
    active_plan = functions.get_active_csp_plan(csp_plans) or user_cfg.get("csp_plan") or {}
    annual_contrib = default_monthly_contribution(active_plan, csp_labels) * 12.0
    alloc = default_contribution_allocation(annual_contrib)
    if annual_contrib > 0:
        alloc_hint = (f"tax-advantaged-first split of {_money(annual_contrib)}/yr from your "
                      "CSP investments plan — editable")
    else:
        alloc_hint = "no CSP investments contribution found — enter annual amounts"

    # Gross income for the SS estimate: seed from the CSP `income` line, flagged
    # because CSP income is net (take-home) while Social Security needs gross.
    annual_income = 12.0 * sum(
        float(amount) for cat, amount in active_plan.items()
        if csp_labels.get(cat) == "income"
    )
    if annual_income > 0:
        income_hint = (f"seeded from your CSP income ({_money(annual_income)}/yr) — likely "
                       "net; enter gross for accuracy")
    else:
        income_hint = "enter your current gross annual income to estimate Social Security"

    # Taxable gain %: derive from the uploaded cost-basis report when available,
    # else fall back to the editable default. Manual edits still override.
    derived = taxable_gain_fraction(uid, accounts)
    if derived is not None:
        gain_pct = round(derived * 100)
        gain_hint = f"derived from your cost-basis report ({gain_pct}% gain) — editable"
    else:
        gain_pct = int(TAXABLE_GAIN_FRACTION * 100)
        gain_hint = "default — upload your Vanguard cost-basis CSV to derive this"

    return (round(buckets["taxable"]), round(buckets["trad"]), round(buckets["roth"]), cur_hint,
            round(alloc["taxable"]), round(alloc["trad"]), round(alloc["roth"]), alloc_hint,
            round(annual_income), income_hint, gain_pct, gain_hint)


# ── Project current balances → at-retirement balances (Phase 6b) ─────────────────

@callback(
    Output("ret-bal-taxable", "value"),
    Output("ret-bal-trad", "value"),
    Output("ret-bal-roth", "value"),
    Output("ret-proj-hint", "children"),
    Input("ret-cur-taxable", "value"),
    Input("ret-cur-trad", "value"),
    Input("ret-cur-roth", "value"),
    Input("ret-alloc-taxable", "value"),
    Input("ret-alloc-trad", "value"),
    Input("ret-alloc-roth", "value"),
    Input("ret-retirement-age", "value"),
    Input("ret-birth-year", "value"),
    Input("ret-real-return", "value"),
)
def project_balances(cur_taxable, cur_trad, cur_roth, al_taxable, al_trad, al_roth,
                     retirement_age, birth_year, real_return):
    """Grow current balances to retirement and write the at-retirement balances
    (ret-bal-*) that feed the drawdown. Those remain editable as a manual override
    until any projector input changes and re-seeds them."""
    current = {"taxable": _num(cur_taxable), "trad": _num(cur_trad), "roth": _num(cur_roth)}
    alloc = {"taxable": _num(al_taxable), "trad": _num(al_trad), "roth": _num(al_roth)}
    r = (_num(real_return)) / 100.0

    years = 0
    if birth_year and retirement_age:
        current_age = datetime.date.today().year - int(birth_year)
        years = max(int(retirement_age) - current_age, 0)

    proj = project_balances_to_retirement(current, alloc, years, r)

    cur_total = sum(current.values())
    if years > 0:
        hint = (f"{_money(cur_total)} today → {_money(proj['total'])} in {years} yrs "
                f"at {r * 100:.1f}% real · editable")
    elif birth_year:
        hint = "already at/after retirement age — projected equals current · editable"
    else:
        hint = "enter birth year above to grow these to retirement · editable"

    return round(proj["taxable"]), round(proj["trad"]), round(proj["roth"]), hint


# ── Estimate SS benefit at FRA from income (Phase 6c) ────────────────────────────

@callback(
    Output("ret-ss-pia", "value"),
    Output("ret-ss-hint", "children"),
    Input("ret-ss-income", "value"),
    Input("ret-ss-emptype", "value"),
    Input("ret-ss-career", "value"),
)
def estimate_ss(gross_income, employment_type, career_years):
    """Estimate the SS benefit at FRA from gross income + employment type, and
    write it into the editable SS-benefit field. That field stays the manual
    override (an SSA-statement figure) until any of these inputs changes."""
    income = _num(gross_income)
    years = int(career_years) if career_years else 35
    pia = estimate_annual_pia_from_income(income, employment_type or "W2", years)
    if income > 0:
        kind = {"1099": "1099 self-employment", "mixed": "mixed"}.get(
            employment_type, "W-2")
        hint = (f"estimated from {_money(income)}/yr {kind} over {years} yrs — "
                "replace with your SSA statement figure to override")
    else:
        hint = "enter gross income above, or type your SSA figure here directly"
    return round(pia), hint


# ── Recompute: BANs + drawdown + glide + income cash-flow (table- & input-driven) ─

@callback(
    Output("ret-ban-goal", "children"),
    Output("ret-ban-avg-spend", "children"),
    Output("ret-ban-first-draw", "children"),
    Output("ret-ban-remaining", "children"),
    Output("ret-drawdown-chart", "figure"),
    Output("ret-insight", "children"),
    Output("ret-glide-chart", "figure"),
    Output("ret-expenses-totals", "children"),
    Output("ret-expenses-note", "children"),
    Output("ret-cashflow-chart", "figure"),
    Output("ret-income-summary", "children"),
    Input("ret-retirement-age", "value"),
    Input("ret-death-age", "value"),
    Input("ret-slow-go-age", "value"),
    Input("ret-no-go-age", "value"),
    Input("ret-real-return", "value"),
    Input("ret-claim-age", "value"),
    Input("ret-birth-year", "value"),
    Input("ret-expenses-table", "data"),
    Input("ret-bal-taxable", "value"),
    Input("ret-bal-trad", "value"),
    Input("ret-bal-roth", "value"),
    Input("ret-ss-pia", "value"),
    Input("ret-gain-frac", "value"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def recompute(retirement_age, death_age, slow_go_age, no_go_age, real_return,
              claim_age, birth_year, expenses_rows, bal_taxable, bal_trad, bal_roth,
              ss_pia, gain_frac, config_data, use_case):
    uid = _selected_uid(use_case)
    user_cfg = _user_cfg(config_data, uid)
    if user_cfg is None:
        raise dash.exceptions.PreventUpdate

    msg = _validation_message(retirement_age, death_age, slow_go_age, no_go_age)
    if msg:
        blank = _ban_card("—", "—", "")
        empty = _empty_figure(msg)
        return (blank, blank, blank, blank, empty, "", _empty_figure(""), "", "",
                _empty_figure(""), "")

    retirement_age = int(retirement_age)
    death_age = int(death_age)
    slow_go_age = int(slow_go_age)
    no_go_age = int(no_go_age)
    r = float(real_return or 0) / 100.0
    claim_age = int(claim_age or 67)

    # Spending comes from the editable table (the single source of truth).
    spend = _spend_from_table(expenses_rows)

    # Income: Social Security from the PIA estimate × claim-age factor.
    ss = social_security_income(_num(ss_pia), claim_age, retirement_age, death_age)

    # Backward nest-egg goal: PV of the net (spend − SS) stream (pre-tax, the
    # planning target). Independent of starting balances.
    stream = project_retirement(0.0, retirement_age, death_age, spend,
                                slow_go_age, no_go_age, r, income_by_age=ss)
    goal = nest_egg_goal(stream, r)

    # Drawdown line: if the user has actual balances, project them through the
    # tax-aware engine (RMDs + ordinary/LTCG taxes); otherwise fall back to the
    # light "exactly funded from the goal" view so the page is useful without them.
    balances = {"taxable": _num(bal_taxable), "trad": _num(bal_trad), "roth": _num(bal_roth)}
    start_total = sum(balances.values())
    rmd_start = rmd_start_age(birth_year) if birth_year else 75
    gf = (_num(gain_frac) or 50.0) / 100.0

    if start_total > 0:
        df = project_retirement_taxaware(
            balances, retirement_age, death_age, spend, slow_go_age, no_go_age, r,
            ss_by_age=ss, rmd_start=rmd_start, taxable_gain_fraction=gf)
        tax_aware = True
    else:
        df = project_retirement(goal, retirement_age, death_age, spend,
                                slow_go_age, no_go_age, r, income_by_age=ss)
        tax_aware = False
    s = retirement_summary(df, r)

    # ── BANs ─────────────────────────────────────────────────────────────────────
    # Goal can go ≤ 0 when Social Security alone covers spending — clamp for display.
    ss_covers = goal <= 0
    display_goal = max(goal, 0.0)
    ban_goal = _ban_card(
        "Nest-egg goal at retirement", _money(display_goal),
        ("Social Security alone covers your spending" if ss_covers
         else f"PV of spend − Social Security to age {death_age}"),
        value_class="text-primary",
    )
    ban_avg = _ban_card(
        "Avg annual spend", f"{_money(s['avg_annual_spend'])}/yr",
        f"go-go {_money(spend['go_go'])} → no-go {_money(spend['no_go'])}",
    )
    draw_sub = "incl. taxes & RMDs" if tax_aware else "peak draw (go-go years)"
    ban_first = _ban_card(
        "First-year withdrawal", f"{_money(s['first_year_drawdown'])}/yr",
        draw_sub, value_class="text-warning",
    )
    funded = s["funded_through_age"]
    funded_ok = funded is not None and funded >= death_age
    remaining_sub = (f"funded through {death_age} ✓" if funded_ok
                     else f"runs out at age {funded}" if funded else "underfunded")
    ban_remaining = _ban_card(
        "Remaining at death", _money(s["balance_at_death"]), remaining_sub,
        value_class="text-success" if funded_ok else "text-danger",
    )

    fig = _drawdown_figure(df, retirement_age, slow_go_age, no_go_age, death_age,
                           display_goal)

    # Teaching insight: the 4% rule vs. the backward goal.
    swr_goal = s["swr_sanity_goal"]
    over = swr_goal - display_goal
    if ss_covers:
        insight = (
            f"Social Security ({_money(float(ss.max()) if len(ss) else 0)}/yr) more than "
            f"covers your planned spending, so you need no dedicated nest egg to fund it — "
            f"a flat 4% rule would still have suggested {_money(swr_goal)}."
        )
    elif over > 0:
        insight = (
            f"A flat 4% rule on your go-go spend would target {_money(swr_goal)} — "
            f"about {_money(over)} more than the {_money(display_goal)} you actually need "
            f"once spending declines through slow-go and no-go, Social Security offsets the "
            f"draw, and the horizon is finite. Modeling these avoids over-saving."
        )
    else:
        insight = (
            "Spending stays high enough across all phases that the time-varying goal "
            "is close to the flat 4%-rule target."
        )

    # ── Expenses-section outputs ─────────────────────────────────────────────────
    glide = _glide_figure(spend, retirement_age, slow_go_age, no_go_age, death_age)
    totals = (
        f"Annual: go-go {_money(spend['go_go'])} · slow-go {_money(spend['slow_go'])} "
        f"· no-go {_money(spend['no_go'])}"
    )
    n_rows = len(expenses_rows or [])
    note = (
        f"{n_rows} living-expense categories from your CSP plan. The glide steps "
        "down at each phase boundary; healthcare's late-life rise (added in a "
        "later phase) is what turns this decline into the full spending smile."
    ) if n_rows else "No CSP plan found — add one on the CSP page to seed these expenses."

    # ── Income-section outputs ────────────────────────────────────────────────────
    cashflow = _cashflow_figure(df, retirement_age, slow_go_age, no_go_age, death_age,
                                tax_aware)
    income_summary = _income_summary_card(df, ss, claim_age, rmd_start, tax_aware,
                                          start_total)

    return (ban_goal, ban_avg, ban_first, ban_remaining, fig, insight,
            glide, totals, note, cashflow, income_summary)


# ── Figure builders ──────────────────────────────────────────────────────────────

def _validation_message(retirement_age, death_age, slow_go_age, no_go_age):
    if None in (retirement_age, death_age, slow_go_age, no_go_age):
        return "Enter retirement, death, and phase-boundary ages."
    if not (retirement_age <= slow_go_age <= no_go_age <= death_age):
        return "Ages must satisfy retirement ≤ slow-go ≤ no-go ≤ death."
    return None


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_CHART_LAYOUT, height=340, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[{"text": message, "xref": "paper", "yref": "paper",
                      "x": 0.5, "y": 0.5, "showarrow": False,
                      "font": {"color": "#999", "size": 13}}] if message else [],
    )
    return fig


def _drawdown_figure(df, retirement_age, slow_go_age, no_go_age, death_age, goal):
    ages = df.index.tolist()
    fig = go.Figure()

    # Faint principal-drawn fill under the balance line.
    fig.add_trace(go.Scatter(
        x=ages, y=df["total"], name="Nest-egg balance", mode="lines",
        line=dict(width=2.5, color="#444"), fill="tozeroy",
        fillcolor="rgba(91,126,201,0.16)",
        hovertemplate="Age %{x}<br>Balance %{y:$,.0f}<extra></extra>",
    ))

    # Phase background bands.
    fig.add_vrect(x0=ages[0], x1=slow_go_age, fillcolor=_C_GOGO, opacity=0.07,
                  layer="below", line_width=0, annotation_text="Go-go · spend peaks",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color="#3f78c4"))
    fig.add_vrect(x0=slow_go_age, x1=no_go_age, fillcolor=_C_SLOWGO, opacity=0.07,
                  layer="below", line_width=0, annotation_text="Slow-go · tapering",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color="#3a9d5d"))
    fig.add_vrect(x0=no_go_age, x1=ages[-1], fillcolor=_C_NOGO, opacity=0.07,
                  layer="below", line_width=0, annotation_text="No-go · healthcare ↑",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color="#b9791b"))

    # Goal marker at retirement.
    fig.add_trace(go.Scatter(
        x=[retirement_age], y=[goal], name="Nest-egg goal",
        mode="markers+text", marker=dict(size=11, color=_C_GOGO, symbol="diamond",
                                         line=dict(width=1, color="white")),
        text=[f"Goal {_money(goal)}"], textposition="top right",
        textfont=dict(size=10, color="#3f78c4"),
        hovertemplate=f"Goal {_money(goal)} at age {retirement_age}<extra></extra>",
    ))

    fig.update_layout(
        **_CHART_LAYOUT, height=340, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Age", showgrid=False),
        yaxis=dict(title=None, tickprefix="$", tickformat="~s", showgrid=True,
                   gridcolor="#eee", zeroline=True, zerolinecolor="#ddd"),
        hovermode="x unified", showlegend=False,
    )
    return fig


def _glide_figure(spend, retirement_age, slow_go_age, no_go_age, death_age):
    """Annual living-spend step-down across the three phases, vs. a flat-spend
    reference (what assuming constant go-go spend would cost). Spend values are
    positive plan magnitudes — no abs()/negation involved."""
    ages = list(range(retirement_age, death_age + 1))
    y = [spend.get(phase_for_age(a, slow_go_age, no_go_age), 0.0) for a in ages]
    flat = [spend.get("go_go", 0.0)] * len(ages)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ages, y=flat, mode="lines", name="Flat go-go",
        line=dict(width=1.2, color="#bbb", dash="dot"),
        hovertemplate="Age %{x}<br>flat %{y:$,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=y, mode="lines", name="Phased spend",
        line=dict(width=2.4, color=_C_GOGO, shape="hv"), fill="tozeroy",
        fillcolor="rgba(127,179,245,0.15)",
        hovertemplate="Age %{x}<br>spend %{y:$,.0f}<extra></extra>",
    ))
    fig.add_vline(x=slow_go_age, line_width=1, line_dash="dot", line_color="#cbd5e1")
    fig.add_vline(x=no_go_age, line_width=1, line_dash="dot", line_color="#cbd5e1")
    fig.update_layout(
        **_CHART_LAYOUT, height=240, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Age", showgrid=False),
        yaxis=dict(title=None, tickprefix="$", tickformat="~s", showgrid=True,
                   gridcolor="#eee", zeroline=False, rangemode="tozero"),
        hovermode="x unified", showlegend=False,
    )
    return fig


def _cashflow_figure(df, retirement_age, slow_go_age, no_go_age, death_age, tax_aware):
    """Annual cash flow: Social Security floor (green) + portfolio draw stacked on
    top (blue), with the tax line (amber) when tax-aware. Concept C's chart."""
    ages = df.index.tolist()
    ss = df["income"] if "income" in df.columns else [0] * len(ages)
    # Portfolio cash that actually funds living = net draw above taxes/reinvested
    # RMD. Use net_spend − SS so the blue band is the portfolio's living share.
    if "net_spend" in df.columns:
        port = (df["net_spend"] - df["income"]).clip(lower=0)
    else:
        port = df["withdrawal"]
    tax = df["tax"] if "tax" in df.columns else None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ages, y=ss, mode="lines", name="Social Security",
        line=dict(width=0, color=_C_SLOWGO), stackgroup="cf",
        fillcolor="rgba(74,222,128,0.35)",
        hovertemplate="Age %{x}<br>Social Security %{y:$,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=port, mode="lines", name="Portfolio draw",
        line=dict(width=0, color=_C_GOGO), stackgroup="cf",
        fillcolor="rgba(127,179,245,0.40)",
        hovertemplate="Age %{x}<br>Portfolio %{y:$,.0f}<extra></extra>",
    ))
    if tax is not None:
        fig.add_trace(go.Scatter(
            x=ages, y=tax, mode="lines", name="Tax",
            line=dict(width=1.6, color=_C_NOGO, dash="dot"),
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


def _income_summary_card(df, ss, claim_age, rmd_start, tax_aware, start_total):
    """Small stat block summarizing the income/tax picture."""
    if not tax_aware:
        return dbc.Alert(
            "Add current balances (or projected at-retirement balances) to model "
            "Social Security, withdrawal ordering, RMDs and taxes. The drawdown above "
            "currently shows the exactly-funded goal.", color="light", className="small mb-0")

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
        _row("SS benefit (claim " + str(claim_age) + ")", f"{_money(ss_annual)}/yr",
             "text-success"),
        _row("RMDs begin", f"age {rmd_start}"),
        _row("First-year tax", f"{_money(first['tax'])} ({eff:.0%})", "text-warning"),
        _row("Lifetime taxes", _money(lifetime_tax), "text-warning"),
    ]), className="h-100")
