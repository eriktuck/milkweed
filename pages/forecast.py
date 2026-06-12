"""Forecast page — v1 (Concept B "Contribution Planner").

Fixed-rate, single-individual CoastFIRE projection with a flat 4% safe
withdrawal rate. See .planning/spec-forecast.md for the approved layout and the
v1 scope. Math lives in core/services/forecast.py; this module is layout +
callbacks only.
"""

import json
from datetime import date

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from flask import session

import core.utils.functions as functions
from core.services.forecast import (
    current_portfolio_value,
    default_monthly_contribution,
    forecast_summary,
    project_portfolio,
    trailing_12mo_contribution,
)
from core.services.investments import fetch_latest_holdings
from core.services.retirement import default_nest_egg_goal
from core.services.returns import (
    bootstrap_projection,
    build_returns_payload,
    percentile_bands,
)

dash.register_page(__name__, path="/forecast")

# ── Constants ──────────────────────────────────────────────────────────────────
_C_PRINCIPAL = "#5b7ec9"   # blue — starting balance + contributions
_C_GROWTH = "#4ade80"      # green — market growth
_C_TARGET = "#f59e0b"      # amber — CoastFIRE target / retirement
_C_TOTAL = "#444444"       # total value line
_C_WORKING = "#3b82f6"

# Asset-class line colors for the historical-returns chart.
_CLASS_COLORS = {
    "US Equity": "#3b82f6",
    "Intl Equity": "#8b5cf6",
    "Bonds": "#f59e0b",
    "Cash": "#9ca3af",
    "Other": "#14b8a6",
}
_C_INFLATION = "#ef4444"   # red — inflation line

_HORIZON_AGE = 90          # planning horizon (fixed in v1; Concept B has no input)

# Fallback ages used only when the user has no saved Profile value (the Profile
# page is the source of truth for demographics; see seed_defaults).
_DEFAULT_CURRENT_AGE = 40
_DEFAULT_COAST_AGE = 50
_DEFAULT_RETIREMENT_AGE = 65

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


def _age_from_birth_date(birth_date) -> int | None:
    """Current age (whole years) from an ISO birth date, or None if unparseable."""
    if not birth_date:
        return None
    try:
        b = date.fromisoformat(str(birth_date)[:10])
    except ValueError:
        return None
    today = date.today()
    return today.year - b.year - ((today.month, today.day) < (b.month, b.day))


def _resolve_goal(goal_store, uid, retirement_age, config_data) -> tuple[float, str]:
    """The nest-egg goal Forecast should plan to, and where it came from.

    Prefers the Retirement page's tuned goal from `retirement-goal-store` when it
    belongs to this user and was computed for the same retirement age (the
    stale-guard); otherwise falls back to `default_nest_egg_goal` from config so
    Forecast works standalone before Retirement is ever opened. Returns
    (goal, "tuned" | "default").
    """
    if (isinstance(goal_store, dict)
            and goal_store.get("uid") == uid
            and goal_store.get("nest_egg_goal") is not None
            and int(goal_store.get("retirement_age", -1)) == int(retirement_age)):
        return float(goal_store["nest_egg_goal"]), "tuned"

    config = json.loads(config_data) if isinstance(config_data, str) else (config_data or {})
    user_cfg = config.get("users", {}).get(uid, {})
    csp_labels = user_cfg.get("csp_labels") or {}
    csp_plans = user_cfg.get("csp_plans") or {}
    active_plan = functions.get_active_csp_plan(csp_plans) or user_cfg.get("csp_plan") or {}
    return default_nest_egg_goal(user_cfg, active_plan, csp_labels), "default"


def _ban_card(label: str, value: str, subtitle, value_class: str = "") -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.P(label, className="text-muted small mb-1"),
        html.H3(value, className=f"mb-0 fw-bold {value_class}"),
        html.P(subtitle, className="text-muted small mt-1 mb-0"),
    ]))


# ── Input controls ──────────────────────────────────────────────────────────────

def _num_input(id_, value, **kw):
    return dbc.Input(id=id_, type="number", value=value, size="sm", **kw)


# Current age (from birth date) and retirement age (from Profile) are derived, not
# entered here — they live in hidden stores (see layout) and are surfaced in fc-meta.
_input_bar = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([
        dbc.Label("Coast age", className="small text-muted mb-1"),
        _num_input("fc-coast-age", _DEFAULT_COAST_AGE, min=18, max=100, step=1),
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
        dbc.Switch(id="fc-use-suggested", label="Use suggested rate",
                   value=False, className="small mt-1"),
        html.Small(id="fc-suggested-hint", className="text-muted"),
    ], width="auto"),
], className="g-3 align-items-end")), className="mb-3")


# ── Page layout ───────────────────────────────────────────────────────────────

layout = html.Div([
    dcc.Store(id="fc-returns-store", storage_type="memory"),
    # Derived demographics (seed_defaults fills them from the saved Profile):
    # current age from birth date, retirement age from the Profile retirement age.
    # Hidden — not user inputs — but they drive the projection and validation.
    dcc.Store(id="fc-current-age", storage_type="memory", data=_DEFAULT_CURRENT_AGE),
    dcc.Store(id="fc-retirement-age", storage_type="memory", data=_DEFAULT_RETIREMENT_AGE),
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

        # ── Rate of return & inflation (educational detail) ──────────────────
        dbc.Row([
            dbc.Col([
                html.H4("Rate of return & inflation", className="mb-1"),
                html.P(
                    "Historical, inflation-adjusted context for the return rate above — "
                    "grounded in your own asset mix. Markets are stochastic; the simulation "
                    "shows a range of likely outcomes, not a guarantee.",
                    className="text-muted small mb-0",
                ),
            ]),
        ], className="pt-2 pb-2"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Historical real (inflation-adjusted) returns by asset class, with inflation",
                       className="text-muted small mb-2"),
                dcc.Loading(dcc.Graph(id="fc-class-returns-chart",
                                      config={"displayModeBar": False}), type="circle"),
            ])), xs=12, lg=6, className="mb-3 mb-lg-0"),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Likely outcomes — MCMC bootstrap", className="text-muted small mb-2"),
                dcc.Loading(dcc.Graph(id="fc-fan-chart",
                                      config={"displayModeBar": False}), type="circle"),
                html.P(id="fc-fan-caption", className="text-muted small mt-2 mb-0"),
            ])), xs=12, lg=6),
        ], className="mb-4"),

    ], fluid=False),
])


# ── Defaults: seed contribution + spend from CSP plan & holdings ────────────────

@callback(
    Output("fc-current-age", "data"),
    Output("fc-coast-age", "value"),
    Output("fc-retirement-age", "data"),
    Output("fc-monthly-contribution", "value"),
    Output("fc-contribution-hint", "children"),
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

    # Ages default from the saved Profile (the source of truth for demographics);
    # fall back to the page defaults when the user has no Profile value yet.
    current_age = _age_from_birth_date(user_cfg.get("birth_date")) or _DEFAULT_CURRENT_AGE
    coast_age = user_cfg.get("coast_age") or _DEFAULT_COAST_AGE
    retirement_age = user_cfg.get("retirement_age") or _DEFAULT_RETIREMENT_AGE

    csp_labels = user_cfg.get("csp_labels") or {}
    csp_plans = user_cfg.get("csp_plans") or {}
    active_plan = functions.get_active_csp_plan(csp_plans) or user_cfg.get("csp_plan") or {}

    monthly = default_monthly_contribution(active_plan, csp_labels)

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

    if not contribution_hint:
        contribution_hint = "from CSP plan (investments)" if monthly else "no CSP plan — enter manually"

    # Surface the derived demographics (no longer rail inputs) so they're visible.
    has_birth = bool(user_cfg.get("birth_date"))
    age_bit = (f"Age {current_age} · retiring at {retirement_age}" if has_birth
               else f"Set your birth date on the Profile page (assuming age {current_age})")
    portfolio_value = current_portfolio_value(uid)
    portfolio_bit = (f"starting portfolio {_money(portfolio_value)} · contributions from CSP plan"
                     if portfolio_value else
                     "no holdings — upload a Vanguard CSV on the Investments page")
    meta = f"{age_bit} · {portfolio_bit}"

    return (current_age, coast_age, retirement_age,
            round(monthly, 2), contribution_hint, meta)


# ── Recompute: BANs + charts ─────────────────────────────────────────────────────

@callback(
    Output("fc-ban-goal", "children"),
    Output("fc-ban-coast-target", "children"),
    Output("fc-ban-coast-status", "children"),
    Output("fc-ban-required", "children"),
    Output("fc-projection-chart", "figure"),
    Output("fc-donut-chart", "figure"),
    Output("fc-insight", "children"),
    Input("fc-current-age", "data"),
    Input("fc-coast-age", "value"),
    Input("fc-retirement-age", "data"),
    Input("fc-monthly-contribution", "value"),
    Input("fc-real-return", "value"),
    Input("config-store", "data"),
    Input("retirement-goal-store", "data"),
)
def recompute(current_age, coast_age, retirement_age, monthly_contribution,
              real_return, config_data, goal_store):
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
    current_age, coast_age, retirement_age = int(current_age), int(coast_age), int(retirement_age)

    # Nest-egg goal: the Retirement page's tuned value when it matches this view,
    # else the from-config default. Forecast no longer derives goal from spend/4%.
    goal, goal_source = _resolve_goal(goal_store, uid, retirement_age, config_data)

    # Accumulation only — truncate at retirement age. The drawdown (and whether the
    # nest egg lasts) is the Retirement page's job; Forecast plots the climb to it.
    df = project_portfolio(
        start_value=start_value,
        current_age=current_age,
        coast_age=coast_age,
        retirement_age=retirement_age,
        horizon_age=retirement_age,
        annual_contribution=annual_contribution,
        real_return=real_return,
    )
    s = forecast_summary(
        df, retirement_goal=goal, real_return=real_return, start_value=start_value,
        current_age=current_age, coast_age=coast_age, retirement_age=retirement_age,
    )

    # ── BANs: goal → coast target → on-track → required lever ────────────────────
    goal_sub = (
        f"from your Retirement plan · at age {retirement_age}"
        if goal_source == "tuned"
        else "estimated from your plan · refine on the Retirement page"
    )
    ban_goal = _ban_card(
        "Retirement nest egg goal", _money(s["retirement_goal"]), goal_sub,
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
    if current_age is None or retirement_age is None:
        return "Add your birth date and retirement age on the Profile page."
    if coast_age is None:
        return "Enter a coast age."
    if not (current_age <= coast_age <= retirement_age <= _HORIZON_AGE):
        return (f"Coast age must be between your current age ({current_age}) and "
                f"retirement age ({retirement_age}).")
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


# ── Rate of return & inflation: heavy compute (off the hot path) ─────────────────

@callback(
    Output("fc-returns-store", "data"),
    Input("config-store", "data"),
)
def compute_returns_data(config_data):
    """Run the network-bound yfinance/cpi/return analytics ONCE per page visit.

    Driven by config-store (fires on mount). Decoupled from `recompute` so the
    fast hot path never touches the network. Results land in fc-returns-store.
    """
    uid = session.get("user_id")
    if not uid:
        raise dash.exceptions.PreventUpdate
    holdings = fetch_latest_holdings(uid)
    return build_returns_payload(holdings)


# ── Suggested-rate opt-in ────────────────────────────────────────────────────────

@callback(
    Output("fc-real-return", "value", allow_duplicate=True),
    Output("fc-real-return", "disabled"),
    Output("fc-suggested-hint", "children"),
    Input("fc-use-suggested", "value"),
    Input("fc-returns-store", "data"),
    State("fc-real-return", "value"),
    prevent_initial_call=True,
)
def apply_suggested_rate(use_suggested, store, current_value):
    """Fill `fc-real-return` with the suggested median real return when toggled on.

    No loop: `fc-real-return.value` is State here (not Input), and no other
    callback writes it, so setting it cannot re-trigger this callback.
    """
    if use_suggested and store and store.get("ok"):
        pct = store.get("suggested_real_return_pct", 0.0)
        hint = f"using suggested median ({pct:.1f}% real)"
        return pct, True, hint
    if use_suggested:
        # Toggle on but no data yet — re-enable and explain.
        return dash.no_update, False, "no suggestion available — add holdings"
    # Toggle off: re-enable manual editing, keep the current value.
    return dash.no_update, False, ""


# ── Detail charts (read the store; no network) ───────────────────────────────────

@callback(
    Output("fc-class-returns-chart", "figure"),
    Input("fc-returns-store", "data"),
)
def render_asset_class_chart(store):
    if not store or not store.get("ok"):
        msg = (store or {}).get("message", "No data") if store else "Loading…"
        return _detail_empty(msg)
    return _class_returns_figure(store)


@callback(
    Output("fc-fan-chart", "figure"),
    Output("fc-fan-caption", "children"),
    Input("fc-returns-store", "data"),
    Input("fc-current-age", "data"),
    Input("fc-coast-age", "value"),
    Input("fc-retirement-age", "data"),
    Input("fc-monthly-contribution", "value"),
)
def render_fan_chart(store, current_age, coast_age, retirement_age, monthly_contribution):
    if not store or not store.get("ok") or not store.get("real_return_pool"):
        return _detail_empty((store or {}).get("message", "No data") if store else "Loading…"), ""

    msg = _validation_message(current_age, coast_age, retirement_age)
    if msg:
        return _detail_empty(msg), ""

    uid = session.get("user_id")
    start_value = current_portfolio_value(uid) if uid else 0.0
    current_age, coast_age, retirement_age = int(current_age), int(coast_age), int(retirement_age)
    annual_contribution = float(monthly_contribution or 0) * 12

    horizon_years = max(retirement_age - current_age, 0)
    if horizon_years == 0:
        return _detail_empty("Retirement age must be after current age."), ""

    # Contribute during the working phase only (age < coast_age).
    contribs = [annual_contribution if (current_age + t) < coast_age else 0.0
                for t in range(horizon_years)]

    sims = bootstrap_projection(
        store["real_return_pool"], start_value, contribs, horizon_years,
    )
    bands = percentile_bands(sims)
    ages = list(range(current_age + 1, current_age + 1 + horizon_years))

    fig = _fan_figure(bands, ages, start_value, current_age, retirement_age)
    n_years = store.get("n_years", 0)
    caption = (
        f"1,000 bootstrap simulations resampling {n_years} years of your portfolio's "
        f"real returns. Bands show the 10th–90th and 25th–75th (IQR) percentiles; "
        f"the line is the median outcome at age {retirement_age}."
    )
    return fig, caption


# ── Detail figure builders ───────────────────────────────────────────────────────

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
    class_returns = store.get("class_real_returns", {})
    for cls, series in class_returns.items():
        years = sorted(int(y) for y in series.keys())
        if not years:
            continue
        fig.add_trace(go.Scatter(
            x=years, y=[series[str(y)] if str(y) in series else series[y] for y in years],
            name=cls, mode="lines",
            line=dict(width=1.5, color=_CLASS_COLORS.get(cls, "#888")),
            hovertemplate=f"{cls} · %{{x}}<br>%{{y:.1%}} real<extra></extra>",
        ))

    weighted = store.get("weighted_avg", {})
    if weighted:
        wyears = sorted(int(y) for y in weighted.keys())
        fig.add_trace(go.Scatter(
            x=wyears, y=[weighted[str(y)] if str(y) in weighted else weighted[y] for y in wyears],
            name="Portfolio weighted avg", mode="lines",
            line=dict(width=3, color=_C_TOTAL),
            hovertemplate="Weighted avg · %{x}<br>%{y:.1%} real<extra></extra>",
        ))

    inflation = store.get("inflation", {})
    if inflation:
        iyears = sorted(int(y) for y in inflation.keys())
        fig.add_trace(go.Scatter(
            x=iyears, y=[inflation[str(y)] if str(y) in inflation else inflation[y] for y in iyears],
            name="Inflation", mode="lines",
            line=dict(width=1.5, color=_C_INFLATION, dash="dot"),
            hovertemplate="Inflation · %{x}<br>%{y:.1%}<extra></extra>",
        ))

    fig.add_hline(y=0, line_color="#ccc", line_width=1)
    fig.update_layout(
        **_CHART_LAYOUT, height=300, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Year", showgrid=False),
        yaxis=dict(title=None, tickformat=".0%", showgrid=True, gridcolor="#eee", zeroline=False),
        hovermode="x unified",
    )
    return fig


def _fan_figure(bands, ages, start_value, current_age, retirement_age) -> go.Figure:
    # Prepend the starting point at current_age so every band shares an origin.
    x = [current_age] + ages
    def _band(p):
        return [start_value] + bands.get(p, [])

    fig = go.Figure()
    # Outer band P10–P90 (light), then IQR P25–P75 (darker), then median line.
    fig.add_trace(go.Scatter(
        x=x, y=_band(90), mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=_band(10), mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(91,126,201,0.15)", name="10th–90th pct",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=_band(75), mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=_band(25), mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(74,222,128,0.30)", name="25th–75th (IQR)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=_band(50), mode="lines", line=dict(width=2.5, color=_C_TOTAL),
        name="Median", hovertemplate="Age %{x}<br>Median %{y:$,.0f}<extra></extra>",
    ))
    fig.update_layout(
        **_CHART_LAYOUT, height=300, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Age", showgrid=False),
        yaxis=dict(title=None, tickprefix="$", tickformat="~s", showgrid=True,
                   gridcolor="#eee", zeroline=False),
        hovermode="x unified",
    )
    return fig
