"""Forecast page — v1 (Concept B "Contribution Planner").

Fixed-rate, single-individual CoastFIRE projection with a flat 4% safe
withdrawal rate. See .planning/spec-forecast.md for the approved layout and the
v1 scope. Math lives in core/services/forecast.py; this module is layout +
callbacks only.
"""

import json

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from flask import session

import core.utils.functions as functions
from core.services.forecast import (
    SWR,
    current_portfolio_value,
    default_annual_retirement_spend,
    default_monthly_contribution,
    forecast_summary,
    project_portfolio,
    trailing_12mo_contribution,
)

dash.register_page(__name__, path="/forecast")

# ── Constants ──────────────────────────────────────────────────────────────────
_C_PRINCIPAL = "#5b7ec9"   # blue — starting balance + contributions
_C_GROWTH = "#4ade80"      # green — market growth
_C_TARGET = "#f59e0b"      # amber — CoastFIRE target / retirement
_C_TOTAL = "#444444"       # total value line
_C_WORKING = "#3b82f6"

_HORIZON_AGE = 90          # planning horizon (fixed in v1; Concept B has no input)

_CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="#444"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


# ── Formatting helpers ──────────────────────────────────────────────────────────

def _money(v: float) -> str:
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


# ── Input controls ──────────────────────────────────────────────────────────────

def _num_input(id_, value, **kw):
    return dbc.Input(id=id_, type="number", value=value, size="sm", **kw)


_input_bar = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([
        dbc.Label("Current age", className="small text-muted mb-1"),
        _num_input("fc-current-age", 40, min=18, max=100, step=1),
    ], width="auto"),
    dbc.Col([
        dbc.Label("Coast age", className="small text-muted mb-1"),
        _num_input("fc-coast-age", 50, min=18, max=100, step=1),
    ], width="auto"),
    dbc.Col([
        dbc.Label("Retirement age", className="small text-muted mb-1"),
        _num_input("fc-retirement-age", 65, min=40, max=100, step=1),
    ], width="auto"),
    dbc.Col([
        dbc.Label("Monthly contribution", className="small text-muted mb-1"),
        _num_input("fc-monthly-contribution", 0, min=0, step=50),
        html.Small(id="fc-contribution-hint", className="text-success"),
    ], width="auto"),
    dbc.Col([
        dbc.Label("Expected real return", className="small text-muted mb-1"),
        dbc.InputGroup([
            _num_input("fc-real-return", 5.0, min=0, max=15, step=0.5),
            dbc.InputGroupText("%"),
        ], size="sm"),
    ], width="auto"),
    dbc.Col([
        dbc.Label("Annual retirement spend", className="small text-muted mb-1"),
        _num_input("fc-annual-spend", 0, min=0, step=1000),
        html.Small(id="fc-spend-hint", className="text-success"),
    ], width="auto"),
    dbc.Col([
        dbc.Label("Withdrawal rate", className="small text-muted mb-1"),
        dbc.Input(value=f"{SWR:.0%}", size="sm", disabled=True, style={"width": "70px"}),
    ], width="auto"),
], className="g-3 align-items-end")), className="mb-3")


# ── Page layout ───────────────────────────────────────────────────────────────

layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Forecast"), width="auto"),
            dbc.Col(
                html.Span(id="fc-meta", className="text-muted small"),
                className="d-flex align-items-center",
            ),
        ], className="pt-3 pb-2", align="center"),

        _input_bar,

        # BANs (filled by callback): goal → coast target → projected → required
        dbc.Row([
            dbc.Col(id="fc-ban-goal", width=3),
            dbc.Col(id="fc-ban-coast-target", width=3),
            dbc.Col(id="fc-ban-coast-status", width=3),
            dbc.Col(id="fc-ban-required", width=3),
        ], className="mb-3 g-3"),

        # Charts
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Portfolio value over time — by phase", className="text-muted small mb-2"),
                dcc.Loading(dcc.Graph(id="fc-projection-chart", config={"displayModeBar": False}), type="circle"),
            ])), width=8),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Where your retirement value comes from", className="text-muted small mb-2"),
                dcc.Loading(dcc.Graph(id="fc-donut-chart", config={"displayModeBar": False}), type="circle"),
                html.P(id="fc-insight", className="text-muted small mt-2 mb-0"),
            ])), width=4),
        ], className="mb-4"),

    ], fluid=False),
])


# ── Defaults: seed contribution + spend from CSP plan & holdings ────────────────

@callback(
    Output("fc-monthly-contribution", "value"),
    Output("fc-annual-spend", "value"),
    Output("fc-contribution-hint", "children"),
    Output("fc-spend-hint", "children"),
    Output("fc-meta", "children"),
    Input("config-store", "data"),
    State("transaction-data-store", "data"),
)
def seed_defaults(config_data, txn_json):
    if not config_data or not isinstance(config_data, str):
        raise dash.exceptions.PreventUpdate

    uid = session.get("user_id")
    if not uid:
        raise dash.exceptions.PreventUpdate

    config = json.loads(config_data)
    user_cfg = config.get("users", {}).get(uid, {})
    csp_labels = user_cfg.get("csp_labels") or {}
    csp_plans = user_cfg.get("csp_plans") or {}
    active_plan = functions.get_active_csp_plan(csp_plans) or user_cfg.get("csp_plan") or {}

    monthly = default_monthly_contribution(active_plan, csp_labels)
    annual_spend = default_annual_retirement_spend(active_plan, csp_labels)

    # Trailing-12-month actual contribution hint (informational only).
    contribution_hint = ""
    if txn_json:
        try:
            import io
            import pandas as pd
            txn_df = pd.read_json(io.StringIO(txn_json), orient="split")
            actual = trailing_12mo_contribution(txn_df)
            if actual:
                contribution_hint = f"last 12 mo actual: ${actual:,.0f}/mo"
        except Exception:
            contribution_hint = ""

    spend_hint = "from CSP plan (fixed + guilt-free)" if annual_spend else "no CSP plan — enter manually"
    if not contribution_hint:
        contribution_hint = "from CSP plan (shrinking)" if monthly else "no CSP plan — enter manually"

    portfolio_value = current_portfolio_value(uid)
    meta = (
        f"Starting portfolio {_money(portfolio_value)} · contributions from CSP plan"
        if portfolio_value else
        "No holdings found — upload a Vanguard CSV on the Investments page"
    )

    return round(monthly, 2), round(annual_spend, 2), contribution_hint, spend_hint, meta


# ── Recompute: BANs + charts ─────────────────────────────────────────────────────

@callback(
    Output("fc-ban-goal", "children"),
    Output("fc-ban-coast-target", "children"),
    Output("fc-ban-coast-status", "children"),
    Output("fc-ban-required", "children"),
    Output("fc-projection-chart", "figure"),
    Output("fc-donut-chart", "figure"),
    Output("fc-insight", "children"),
    Input("fc-current-age", "value"),
    Input("fc-coast-age", "value"),
    Input("fc-retirement-age", "value"),
    Input("fc-monthly-contribution", "value"),
    Input("fc-real-return", "value"),
    Input("fc-annual-spend", "value"),
    Input("config-store", "data"),
)
def recompute(current_age, coast_age, retirement_age, monthly_contribution,
              real_return, annual_spend, config_data):
    uid = session.get("user_id")
    if not uid:
        raise dash.exceptions.PreventUpdate

    # Validate input ordering; show a friendly prompt rather than crashing.
    msg = _validation_message(current_age, coast_age, retirement_age)
    if msg:
        empty = _empty_figure(msg)
        blank = _ban_card("—", "—", "")
        return blank, blank, blank, blank, empty, _empty_figure(""), ""

    start_value = current_portfolio_value(uid)
    your_monthly = float(monthly_contribution or 0)
    annual_contribution = your_monthly * 12
    real_return = float(real_return or 0) / 100.0
    annual_spend = float(annual_spend or 0)
    current_age, coast_age, retirement_age = int(current_age), int(coast_age), int(retirement_age)

    df = project_portfolio(
        start_value=start_value,
        current_age=current_age,
        coast_age=coast_age,
        retirement_age=retirement_age,
        horizon_age=_HORIZON_AGE,
        annual_contribution=annual_contribution,
        real_return=real_return,
    )
    s = forecast_summary(
        df, annual_spend=annual_spend, real_return=real_return, start_value=start_value,
        current_age=current_age, coast_age=coast_age, retirement_age=retirement_age,
    )

    # ── BANs: goal → coast target → on-track → required lever ────────────────────
    ban_goal = _ban_card(
        "Retirement nest egg goal", _money(s["retirement_goal"]),
        f"{_money(annual_spend)}/yr ÷ {SWR:.0%} at age {retirement_age}",
    )

    ban_coast_target = _ban_card(
        "CoastFIRE target", _money(s["coast_target"]),
        f"needed at age {coast_age} · grows to goal by {retirement_age}",
        value_class="text-primary",
    )

    cs = s["coast_surplus"]
    if cs >= 0:
        if s["coast_point_age"] is not None:
            status_sub = f"on track · could coast at age {s['coast_point_age']}"
        else:
            status_sub = f"on track · +{_money(cs)} over coast target"
        status_cls = "text-success"
    else:
        status_sub = f"short by {_money(-cs)} at age {coast_age}"
        status_cls = "text-danger"
    ban_coast_status = _ban_card(
        "Projected at coast age", _money(s["projected_at_coast"]), status_sub,
        value_class=status_cls,
    )

    req = s["required_monthly"]
    if req <= 0:
        req_sub = "you're already on track — $0 needed"
        req_cls = "text-success"
    else:
        req_cls = "text-success" if your_monthly >= req else "text-danger"
        req_sub = f"you're contributing ${your_monthly:,.0f}/mo"
    ban_required = _ban_card(
        "Required contribution", f"${req:,.0f}/mo", req_sub, value_class=req_cls,
    )

    # ── Figures ─────────────────────────────────────────────────────────────────
    fig = _projection_figure(
        df, s["retirement_goal"], s["coast_target"], coast_age, retirement_age, real_return,
    )
    donut, insight = _donut_figure(df, retirement_age, s, real_return)

    return ban_goal, ban_coast_target, ban_coast_status, ban_required, fig, donut, insight


# ── Figure builders ──────────────────────────────────────────────────────────────

def _validation_message(current_age, coast_age, retirement_age):
    if None in (current_age, coast_age, retirement_age):
        return "Enter current, coast, and retirement ages."
    if not (current_age <= coast_age <= retirement_age <= _HORIZON_AGE):
        return f"Ages must satisfy current ≤ coast ≤ retirement ≤ {_HORIZON_AGE}."
    return None


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_CHART_LAYOUT, height=320, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[{"text": message, "xref": "paper", "yref": "paper",
                      "x": 0.5, "y": 0.5, "showarrow": False,
                      "font": {"color": "#999", "size": 13}}] if message else [],
    )
    return fig


def _projection_figure(df, retirement_goal, coast_target, coast_age,
                       retirement_age, real_return) -> go.Figure:
    ages = df.index.tolist()
    fig = go.Figure()

    # Stacked areas: principal (bottom) + growth (top).
    fig.add_trace(go.Scatter(
        x=ages, y=df["principal"], name="Principal", mode="lines",
        line=dict(width=0, color=_C_PRINCIPAL), stackgroup="v",
        fillcolor="rgba(91,126,201,0.45)",
        hovertemplate="Age %{x}<br>Principal %{y:$,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=df["growth"], name="Growth", mode="lines",
        line=dict(width=0, color=_C_GROWTH), stackgroup="v",
        fillcolor="rgba(74,222,128,0.35)",
        hovertemplate="Age %{x}<br>Growth %{y:$,.0f}<extra></extra>",
    ))
    # Total value line on top of the stack.
    fig.add_trace(go.Scatter(
        x=ages, y=df["total"], name="Total value", mode="lines",
        line=dict(width=2.5, color=_C_TOTAL),
        hovertemplate="Age %{x}<br>Total %{y:$,.0f}<extra></extra>",
    ))

    # Phase background bands.
    fig.add_vrect(x0=ages[0], x1=coast_age, fillcolor=_C_WORKING, opacity=0.06,
                  layer="below", line_width=0, annotation_text="Working",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color=_C_WORKING))
    fig.add_vrect(x0=coast_age, x1=retirement_age, fillcolor=_C_GROWTH, opacity=0.06,
                  layer="below", line_width=0, annotation_text="Coast",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color="#3a9d5d"))
    fig.add_vrect(x0=retirement_age, x1=ages[-1], fillcolor=_C_TARGET, opacity=0.06,
                  layer="below", line_width=0, annotation_text="Retirement · 4% draw",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color="#b9791b"))

    # Retirement nest egg goal — horizontal line (the target AT retirement).
    fig.add_hline(y=retirement_goal, line_dash="dash", line_color=_C_TARGET, line_width=1.5,
                  annotation_text=f"Retirement goal {_money(retirement_goal)}",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color=_C_TARGET))

    # Coast glide path: coast_target compounding at r from coast age to the goal.
    glide_ages = [a for a in ages if coast_age <= a <= retirement_age]
    glide_y = [coast_target * ((1 + real_return) ** (a - coast_age)) for a in glide_ages]
    if glide_ages:
        fig.add_trace(go.Scatter(
            x=glide_ages, y=glide_y, name="Coast glide path", mode="lines",
            line=dict(color="#b9791b", width=1.5, dash="dot"),
            hovertemplate="Age %{x}<br>on-track minimum %{y:$,.0f}<extra></extra>",
        ))

    # CoastFIRE target marker at coast age (what you need to STOP contributing).
    fig.add_trace(go.Scatter(
        x=[coast_age], y=[coast_target], name="CoastFIRE target",
        mode="markers+text", marker=dict(size=11, color=_C_TARGET, symbol="diamond",
                                         line=dict(width=1, color="white")),
        text=[f"Coast target {_money(coast_target)}"], textposition="top center",
        textfont=dict(size=10, color="#b9791b"),
        hovertemplate=f"Coast target {_money(coast_target)} at age {coast_age}<extra></extra>",
    ))

    fig.update_layout(
        **_CHART_LAYOUT, height=320, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Age", showgrid=False),
        yaxis=dict(title=None, tickprefix="$", tickformat="~s", showgrid=True,
                   gridcolor="#eee", zeroline=False),
        hovermode="x unified",
    )
    return fig


def _donut_figure(df, retirement_age, summary, real_return):
    ret_rows = df[df["phase"] == "retirement"]
    row = ret_rows.iloc[0] if not ret_rows.empty else df.iloc[-1]
    principal, growth, total = row["principal"], row["growth"], row["total"]

    fig = go.Figure(go.Pie(
        labels=["Principal", "Growth"],
        values=[principal, growth],
        hole=0.62,
        marker=dict(colors=[_C_PRINCIPAL, _C_GROWTH]),
        sort=False,
        direction="clockwise",
        textinfo="percent",
        hovertemplate="%{label}: %{value:$,.0f} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#444"),
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        annotations=[{"text": f"<b>{_money(total)}</b><br><span style='font-size:11px;color:#888'>at age {retirement_age}</span>",
                      "x": 0.5, "y": 0.5, "showarrow": False, "font": {"size": 16}}],
    )

    growth_pct = (growth / total) if total else 0
    insight = (
        f"{growth_pct:.0%} of your retirement nest egg is compounding growth — the heart "
        f"of the coast strategy. You only need {_money(summary['coast_target'])} by your "
        f"coast age; growth alone carries it to the {_money(summary['retirement_goal'])} "
        f"goal by {retirement_age}."
    )
    return fig, insight
