"""Coast page — the unified scenario-planning screen.

Replaces the split Forecast (accumulation) + Retirement (drawdown) pages with one
cockpit: a milestone slider + monthly-contribution input drive the four BANs and a
single accumulation→coast→drawdown chart. See .planning/SPEC-scenario-planning.md
(Phase 2 — primary screen; the background assumption modules land in Phase 3).

Every number comes from one call into `core.services.scenario.build_scenario`,
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
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from flask import session

import core.utils.functions as functions
from core.services.firebase import save_retirement_config
from core.services.forecast import default_monthly_contribution
from core.services.investments import fetch_latest_holdings
from core.services.retirement import balances_by_tax_bucket, resolve_assumptions
from core.services.returns import (
    bootstrap_projection,
    build_returns_payload,
    percentile_bands,
)
from core.services.scenario import DEFAULT_COAST_AGE, build_scenario

dash.register_page(__name__, path="/coast")

# ── Chart palette (shared with Forecast/Retirement) ──────────────────────────────
_C_PRINCIPAL = "#5b7ec9"   # blue — starting balance + contributions
_C_GROWTH = "#4ade80"      # green — market growth
_C_TARGET = "#f59e0b"      # amber — nest-egg goal / CoastFIRE target
_C_TOTAL = "#444444"       # total value line
_C_WORKING = "#3b82f6"
_C_TRAD = "#8b5cf6"        # violet — tax-deferred
_C_ROTH = "#14b8a6"        # teal — Roth

# Asset-class line colors for the historical-returns chart (rate-of-return module).
_CLASS_COLORS = {
    "US Equity": "#3b82f6",
    "Intl Equity": "#8b5cf6",
    "Bonds": "#f59e0b",
    "Cash": "#9ca3af",
    "Other": "#14b8a6",
}
_C_INFLATION = "#ef4444"   # red — inflation line

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
    return functions.get_active_csp_plan(csp_plans) or (user_cfg or {}).get("csp_plan") or {}


def _current_age(user_cfg) -> int:
    """Current whole-years age from the saved birth date, else the fallback."""
    a = resolve_assumptions(user_cfg or {})
    by = a.get("birth_year")
    return (date.today().year - int(by)) if by else _DEFAULT_CURRENT_AGE


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

_rate_module = _module(
    "co-rr", "fa-percent", "Rate of return", "co-rr-summary",
    body=[
        html.P(
            "Historical, inflation-adjusted context for the return that drives the "
            "projection above — grounded in your own asset mix. Markets are stochastic; "
            "the simulation shows a range of likely outcomes, not a guarantee.",
            className="text-muted small mb-3",
        ),
        dbc.Row([
            dbc.Col([
                dbc.Label("Expected real return", className="ban-label mb-1"),
                dbc.InputGroup([
                    dbc.Input(id="co-real-return", type="number", value=5.0,
                              min=0, max=15, step=0.5, size="sm"),
                    dbc.InputGroupText("%"),
                ], size="sm"),
                dbc.Switch(id="co-use-suggested", label="Use suggested rate",
                           value=False, className="small mt-2"),
                html.Small(id="co-suggested-hint", className="text-muted"),
            ], lg=3),
            dbc.Col([
                html.P("Historical real (inflation-adjusted) returns by asset class, with inflation",
                       className="text-muted small mb-2"),
                dcc.Loading(dcc.Graph(id="co-class-returns-chart",
                                      config={"displayModeBar": False}), type="circle"),
            ], lg=5),
            dbc.Col([
                html.P("Likely outcomes — MCMC bootstrap to retirement",
                       className="text-muted small mb-2"),
                dcc.Loading(dcc.Graph(id="co-fan-chart",
                                      config={"displayModeBar": False}), type="circle"),
                html.P(id="co-fan-caption", className="text-muted small mt-2 mb-0"),
            ], lg=4),
        ], className="g-3"),
        html.Div([
            dbc.Button("Save rate", id="co-rr-save", color="primary", size="sm",
                       outline=True),
            html.Span(id="co-rr-save-status", className="small ms-2"),
        ], className="mt-2"),
    ],
)


# ── Page layout ──────────────────────────────────────────────────────────────────

layout = html.Div(dbc.Container([
    # Network-bound returns analytics (yfinance/CPI), computed once per visit and
    # kept off the recompute hot path (mirrors Forecast's fc-returns-store).
    dcc.Store(id="co-returns-store", storage_type="memory"),
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
    html.P([html.I(className="fas fa-circle-info me-1"),
            "Editing any assumption re-runs the scenario above — the chart and BANs "
            "update live."], className="assist-note mt-2"),
], fluid=False))


# ── Seed the slider + contribution from config (Profile is the truth) ────────────

_module_toggle("co-rr")


@callback(
    Output("co-milestones", "min"),
    Output("co-milestones", "max"),
    Output("co-milestones", "marks"),
    Output("co-milestones", "value"),
    Output("co-monthly", "value"),
    Output("co-contribution-hint", "children"),
    Output("co-meta", "children"),
    Output("co-real-return", "value"),
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

    real_return_pct = round(float(a["real_return"]) * 100.0, 1)

    return lo, hi, marks, value, round(monthly, 2), hint, meta, real_return_pct


# ── Recompute: BANs + chart from build_scenario ──────────────────────────────────

@callback(
    Output("co-milestone-chips", "children"),
    Output("co-ban-goal", "children"),
    Output("co-ban-coast-target", "children"),
    Output("co-ban-coast-status", "children"),
    Output("co-ban-required", "children"),
    Output("co-projection-chart", "figure"),
    Output("co-rr-summary", "children"),
    Input("co-milestones", "value"),
    Input("co-monthly", "value"),
    Input("co-color-mode", "value"),
    Input("co-real-return", "value"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def recompute(milestones, monthly, color_mode, real_return, config_data, use_case):
    uid = _selected_uid(use_case)
    user_cfg = _user_cfg(config_data, uid)
    if user_cfg is None or not milestones or len(milestones) != 5:
        raise dash.exceptions.PreventUpdate

    coast_age, retirement_age, claim_age, slow_go_age, no_go_age = (int(v) for v in milestones)
    chips = _milestone_chips(coast_age, retirement_age, claim_age, slow_go_age, no_go_age)

    active_plan = _active_plan(user_cfg)
    csp_labels = user_cfg.get("csp_labels") or {}
    buckets = balances_by_tax_bucket(uid, user_cfg.get("investment_accounts"))

    # Live real-return override from the rate-of-return module (None → config default).
    r_override = float(real_return) / 100.0 if real_return is not None else None

    # Single resolution path: build_scenario merges the live inputs over config.
    # death_age stays config-resolved (the slider owns the 5 milestone handles).
    s = build_scenario(
        user_cfg, active_plan, csp_labels, buckets,
        monthly_contribution=float(monthly or 0),
        coast_age=coast_age, retirement_age=retirement_age, claim_age=claim_age,
        slow_go_age=slow_go_age, no_go_age=no_go_age,
        real_return=r_override,
    )
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
    return chips, ban_goal, ban_target, ban_status, ban_required, fig, rr_summary


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
    Output("co-real-return", "value", allow_duplicate=True),
    Output("co-real-return", "disabled"),
    Output("co-suggested-hint", "children"),
    Input("co-use-suggested", "value"),
    Input("co-returns-store", "data"),
    prevent_initial_call=True,
)
def apply_suggested_rate(use_suggested, store):
    """Fill the real-return input with the holdings-suggested median when toggled on
    (and lock it); re-enable manual editing when off. No loop: co-real-return.value is
    not an Input here."""
    if use_suggested and store and store.get("ok"):
        pct = store.get("suggested_real_return_pct", 0.0)
        return pct, True, f"using suggested median ({pct:.1f}% real)"
    if use_suggested:
        return dash.no_update, False, "no suggestion available — add holdings"
    return dash.no_update, False, ""


@callback(
    Output("co-class-returns-chart", "figure"),
    Input("co-returns-store", "data"),
)
def render_class_returns_chart(store):
    if not store or not store.get("ok"):
        return _detail_empty((store or {}).get("message", "No data") if store else "Loading…")
    return _class_returns_figure(store)


@callback(
    Output("co-fan-chart", "figure"),
    Output("co-fan-caption", "children"),
    Input("co-returns-store", "data"),
    Input("co-milestones", "value"),
    Input("co-monthly", "value"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def render_fan_chart(store, milestones, monthly, config_data, use_case):
    """MCMC bootstrap of the accumulation climb to retirement (mirrors Forecast's
    fan), resampling the user's holdings' real annual returns."""
    if not store or not store.get("ok") or not store.get("real_return_pool"):
        return _detail_empty((store or {}).get("message", "No data") if store else "Loading…"), ""

    uid = _selected_uid(use_case)
    user_cfg = _user_cfg(config_data, uid)
    if user_cfg is None or not milestones or len(milestones) != 5:
        return _detail_empty(""), ""

    coast_age, retirement_age = int(milestones[0]), int(milestones[1])
    current_age = _current_age(user_cfg)
    horizon_years = max(retirement_age - current_age, 0)
    if horizon_years == 0:
        return _detail_empty("Retirement age must be after your current age."), ""

    buckets = balances_by_tax_bucket(uid, user_cfg.get("investment_accounts"))
    start_value = float(buckets.get("total", 0.0))
    annual_contribution = float(monthly or 0) * 12.0
    # Contribute during the working phase only (age < coast_age).
    contribs = [annual_contribution if (current_age + t) < coast_age else 0.0
                for t in range(horizon_years)]

    sims = bootstrap_projection(store["real_return_pool"], start_value, contribs, horizon_years)
    bands = percentile_bands(sims)
    ages = list(range(current_age + 1, current_age + 1 + horizon_years))

    fig = _fan_figure(bands, ages, start_value, current_age)
    n_years = store.get("n_years", 0)
    caption = (
        f"1,000 bootstrap simulations resampling {n_years} years of your portfolio's "
        f"real returns. Bands show the 10th–90th and 25th–75th (IQR) percentiles; the "
        f"line is the median outcome at age {retirement_age}."
    )
    return fig, caption


@callback(
    Output("config-store", "data", allow_duplicate=True),
    Output("co-rr-save-status", "children"),
    Input("co-rr-save", "n_clicks"),
    State("co-real-return", "value"),
    State("config-store", "data"),
    State("use-case", "value"),
    prevent_initial_call=True,
)
def save_rate(n, real_return, config_data, use_case):
    """Persist the real return to RetirementConfig (stored as a fraction) and mirror
    it into config-store so the in-session view matches Firestore without a reload."""
    if not n:
        raise dash.exceptions.PreventUpdate
    uid = _selected_uid(use_case)
    if not uid or not config_data or real_return is None:
        raise dash.exceptions.PreventUpdate

    r = float(real_return) / 100.0
    save_retirement_config(uid, {"real_return": r})

    cfg = json.loads(config_data)
    ret_cfg = cfg.setdefault("users", {}).setdefault(uid, {}).setdefault("retirement", {})
    ret_cfg["real_return"] = r
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


def _class_returns_figure(store) -> go.Figure:
    fig = go.Figure()
    for cls, series in store.get("class_real_returns", {}).items():
        years = sorted(int(y) for y in series.keys())
        if not years:
            continue
        fig.add_trace(go.Scatter(
            x=years, y=[series.get(str(y), series.get(y)) for y in years],
            name=cls, mode="lines",
            line=dict(width=1.5, color=_CLASS_COLORS.get(cls, "#888")),
            hovertemplate=f"{cls} · %{{x}}<br>%{{y:.1%}} real<extra></extra>",
        ))

    weighted = store.get("weighted_avg", {})
    if weighted:
        wyears = sorted(int(y) for y in weighted.keys())
        fig.add_trace(go.Scatter(
            x=wyears, y=[weighted.get(str(y), weighted.get(y)) for y in wyears],
            name="Portfolio weighted avg", mode="lines",
            line=dict(width=3, color=_C_TOTAL),
            hovertemplate="Weighted avg · %{x}<br>%{y:.1%} real<extra></extra>",
        ))

    inflation = store.get("inflation", {})
    if inflation:
        iyears = sorted(int(y) for y in inflation.keys())
        fig.add_trace(go.Scatter(
            x=iyears, y=[inflation.get(str(y), inflation.get(y)) for y in iyears],
            name="Inflation", mode="lines",
            line=dict(width=1.5, color=_C_INFLATION, dash="dot"),
            hovertemplate="Inflation · %{x}<br>%{y:.1%}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="#ccc", line_width=1)
    fig.update_layout(
        **_CHART_LAYOUT, height=280, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Year", showgrid=False),
        yaxis=dict(title=None, tickformat=".0%", showgrid=True, gridcolor="#eee", zeroline=False),
        hovermode="x unified",
    )
    return fig


def _fan_figure(bands, ages, start_value, current_age) -> go.Figure:
    x = [current_age] + ages

    def _band(p):
        return [start_value] + bands.get(p, [])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=_band(90), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=_band(10), mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(91,126,201,0.15)",
                             name="10th–90th pct", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=_band(75), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=_band(25), mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(74,222,128,0.30)",
                             name="25th–75th (IQR)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=_band(50), mode="lines",
                             line=dict(width=2.5, color=_C_TOTAL), name="Median",
                             hovertemplate="Age %{x}<br>Median %{y:$,.0f}<extra></extra>"))
    fig.update_layout(
        **_CHART_LAYOUT, height=280, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Age", showgrid=False),
        yaxis=dict(title=None, tickprefix="$", tickformat="~s", showgrid=True,
                   gridcolor="#eee", zeroline=False),
        hovermode="x unified",
    )
    return fig
