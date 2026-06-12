"""Profile page — demographics + gross-income-over-time (retirement model inputs).

Persists per-user background data that the retirement model needs: birth date,
the ages anchoring scenarios (coast/retirement/Social-Security-claim/plan-to),
and gross income as forward-filled segments ({date, annual amount}) plus a real
income-growth rate. The income timeline previews actual vs projected income
through the coast year. See .planning/SPEC-profile.md.
"""

import json
from datetime import date

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from flask import session

from core.services.firebase import save_user_profile
from core.utils.functions import segments_to_annual_income

dash.register_page(__name__, path="/profile")

# ── palette (matches forecast/investments) ──────────────────────────────────────
_C_ACTUAL = "#5b7ec9"      # realized income (solid)
_C_PROJECTED = "#4ade80"   # projected income (dashed)
_C_COAST = "#f59e0b"       # coast-year marker

# Functional defaults supplied when a user has no saved value yet.
_DEFAULTS = {
    "coast_age": 50,
    "retirement_age": 67,
    "claim_age": 70,
    "death_age": 90,
    "income_growth_rate": 0.03,
}


# ── config helpers ──────────────────────────────────────────────────────────────

def _parse_config(config_data):
    if not config_data:
        return {}
    try:
        return json.loads(config_data) if isinstance(config_data, str) else dict(config_data)
    except Exception:
        return {}


def _user_cfg(config_data):
    """Return (cfg, uid, this_user_config) for the logged-in user."""
    cfg = _parse_config(config_data)
    uid = session.get("user_id")
    return cfg, uid, (cfg.get("users", {}).get(uid, {}) if uid else {})


# ── small input helpers ─────────────────────────────────────────────────────────

def _age_input(id_, value=None, **kw):
    return dbc.Input(id=id_, type="number", value=value, size="sm",
                     style={"maxWidth": "110px"}, **kw)


def _labeled(label, control, width="auto"):
    return dbc.Col([dbc.Label(label, className="small text-muted mb-1"), control], width=width)


def _segment_row(i, seg):
    """One income segment: effective date + annual amount + remove button."""
    return dbc.Row([
        dbc.Col(
            dcc.DatePickerSingle(
                id={"type": "profile-seg-date", "index": i},
                date=seg.get("date"),
                display_format="YYYY-MM-DD",
                style={"borderWidth": 0},
            ),
            width="auto",
        ),
        dbc.Col(
            dbc.InputGroup([
                dbc.InputGroupText("$"),
                dbc.Input(
                    id={"type": "profile-seg-amount", "index": i},
                    type="number", value=seg.get("amount"), step=1000, size="sm",
                    style={"maxWidth": "160px"},
                ),
                dbc.InputGroupText("/yr"),
            ], size="sm"),
            width="auto",
        ),
        dbc.Col(
            dbc.Button(html.I(className="fas fa-xmark"),
                       id={"type": "profile-seg-remove", "index": i},
                       color="link", size="sm", className="text-muted"),
            width="auto", className="d-flex align-items-center",
        ),
    ], className="g-2 mb-2 align-items-center")


# ── timeline figure (shared by the live callback) ───────────────────────────────

def _income_timeline_figure(segments, growth, birth_year, coast_age):
    coast_year = birth_year + coast_age
    series = segments_to_annual_income(segments, thru_year=coast_year, growth_rate=growth)
    fig = go.Figure()
    if series.empty:
        return _blank_chart(fig)

    current_year = date.today().year
    solid = series[series.index <= current_year]
    dashed = series[series.index >= current_year]  # overlap one point so lines connect

    fig.add_trace(go.Scatter(
        x=solid.index, y=solid.values, mode="lines", name="Actual",
        line=dict(color=_C_ACTUAL, width=2.5),
        hovertemplate="%{x}: $%{y:,.0f}<extra>Actual</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dashed.index, y=dashed.values, mode="lines", name="Projected",
        line=dict(color=_C_PROJECTED, width=2.5, dash="dash"),
        hovertemplate="%{x}: $%{y:,.0f}<extra>Projected</extra>",
    ))
    fig.add_vline(x=coast_year, line=dict(color=_C_COAST, width=1.5, dash="dot"))
    fig.add_annotation(x=coast_year, yref="paper", y=1.0, text="Coast year",
                       showarrow=False, font=dict(color=_C_COAST, size=12),
                       xanchor="right", xshift=-4)

    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#444"),
        margin=dict(l=10, r=10, t=10, b=10), height=320,
        xaxis=dict(showgrid=False, dtick=2),
        yaxis=dict(showgrid=True, gridcolor="#eee", tickprefix="$", tickformat=",.0f",
                   rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _blank_chart(fig=None):
    fig = fig or go.Figure()
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white", height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text="Add an income period to see your timeline.",
                          showarrow=False, font=dict(color="#999"))],
    )
    return fig


# ── cards ───────────────────────────────────────────────────────────────────────

_demographics_card = dbc.Card([
    dbc.CardHeader([
        html.H5("Demographics", className="mb-1"),
        html.Div(
            "Your birth date and the ages that anchor retirement scenarios. These are "
            "defaults — you can still adjust them when scenario planning.",
            className="text-muted small",
        ),
    ]),
    dbc.CardBody(dbc.Row([
        _labeled("Birth date", dcc.DatePickerSingle(
            id="profile-birth-date", display_format="YYYY-MM-DD", style={"borderWidth": 0})),
        _labeled("Coast age", _age_input("profile-coast-age", _DEFAULTS["coast_age"],
                                         min=18, max=100, step=1)),
        _labeled("Retirement age", _age_input("profile-retirement-age",
                                              _DEFAULTS["retirement_age"], min=40, max=100, step=1)),
        _labeled("Social Security claim age", _age_input("profile-claim-age",
                                              _DEFAULTS["claim_age"], min=62, max=70, step=1)),
        _labeled("Plan-to age", _age_input("profile-death-age", _DEFAULTS["death_age"],
                                          min=70, max=120, step=1)),
    ], className="g-3 align-items-end")),
    dbc.CardFooter(dbc.Row([
        dbc.Col(html.Div(id="profile-demo-status")),
        dbc.Col(dbc.Button("Save changes", id="profile-demo-save", color="primary", size="sm"),
                width="auto"),
    ], align="center")),
], className="mb-4")


_income_card = dbc.Card([
    dbc.CardHeader([
        html.H5("Gross income", className="mb-1"),
        html.Div(
            "Your gross (pre-tax) income over time, used to estimate Social Security. "
            "Each row is an annual income level effective from its date and held until "
            "the next row. A future-dated row models an expected raise or new job.",
            className="text-muted small",
        ),
    ]),
    dbc.CardBody([
        dbc.Row(
            _labeled("Income growth (real, applied after the last row)",
                     dbc.InputGroup([
                         dbc.Input(id="profile-income-growth", type="number",
                                   value=round(_DEFAULTS["income_growth_rate"] * 100, 1),
                                   min=0, max=15, step=0.5, size="sm",
                                   style={"maxWidth": "90px"}),
                         dbc.InputGroupText("%"),
                     ], size="sm")),
            className="g-3 mb-3",
        ),
        html.Div([
            html.Span("Effective date", className="small text-muted me-5"),
            html.Span("Annual income", className="small text-muted"),
        ], className="mb-1"),
        html.Div(id="profile-segments-container"),
        dbc.Button([html.I(className="fas fa-plus me-1"), "Add income period"],
                   id="profile-seg-add", color="link", size="sm", className="ps-0"),
    ]),
    dbc.CardFooter(dbc.Row([
        dbc.Col(html.Div(id="profile-income-status")),
        dbc.Col(dbc.Button("Save changes", id="profile-income-save", color="primary", size="sm"),
                width="auto"),
    ], align="center")),
], className="mb-4")


_timeline_card = dbc.Card([
    dbc.CardHeader([
        html.H5("Income timeline", className="mb-1"),
        html.Div("Actual income (solid) and projected income (dashed) through your "
                 "coast year.", className="text-muted small"),
    ]),
    dbc.CardBody(dcc.Loading(
        dcc.Graph(id="profile-income-chart", figure=_blank_chart(),
                  config={"displayModeBar": False}),
        type="circle",
    )),
], className="mb-5")


layout = html.Div([
    dcc.Store(id="profile-segments-store", storage_type="memory"),
    dbc.Container([
        html.H1("Profile", className="pt-3"),
        html.P("Background details that power retirement planning.", className="text-muted"),
        _demographics_card,
        _income_card,
        _timeline_card,
    ]),
])


# ── callbacks ─────────────────────────────────────────────────────────────────────

@callback(
    Output("profile-birth-date", "date"),
    Output("profile-coast-age", "value"),
    Output("profile-retirement-age", "value"),
    Output("profile-claim-age", "value"),
    Output("profile-death-age", "value"),
    Output("profile-income-growth", "value"),
    Output("profile-segments-store", "data"),
    Input("config-store", "data"),
)
def hydrate(config_data):
    """Populate the form from the logged-in user's saved profile (or defaults)."""
    _, uid, c = _user_cfg(config_data)
    if not uid:
        raise PreventUpdate
    growth = c.get("income_growth_rate")
    if growth is None:
        growth = _DEFAULTS["income_growth_rate"]
    return (
        c.get("birth_date"),
        c.get("coast_age") or _DEFAULTS["coast_age"],
        c.get("retirement_age") or _DEFAULTS["retirement_age"],
        c.get("claim_age") or _DEFAULTS["claim_age"],
        c.get("death_age") or _DEFAULTS["death_age"],
        round(growth * 100, 1),
        c.get("income_segments") or [],
    )


@callback(
    Output("profile-segments-container", "children"),
    Input("profile-segments-store", "data"),
)
def render_segments(segments):
    return [_segment_row(i, s) for i, s in enumerate(segments or [])]


@callback(
    Output("profile-segments-store", "data", allow_duplicate=True),
    Input("profile-seg-add", "n_clicks"),
    Input({"type": "profile-seg-remove", "index": ALL}, "n_clicks"),
    State({"type": "profile-seg-date", "index": ALL}, "date"),
    State({"type": "profile-seg-amount", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def modify_segments(add_clicks, remove_clicks, dates, amounts):
    """Add a blank row or remove one, snapshotting current edits first."""
    trig = ctx.triggered_id
    current = [{"date": d, "amount": a} for d, a in zip(dates, amounts)]

    if trig == "profile-seg-add":
        current.append({"date": None, "amount": None})
    elif isinstance(trig, dict) and trig.get("type") == "profile-seg-remove":
        # Only act on a real click (re-render fires this input with all-None).
        if not any(remove_clicks):
            raise PreventUpdate
        idx = trig.get("index")
        if 0 <= idx < len(current):
            current.pop(idx)
    else:
        raise PreventUpdate
    return current


@callback(
    Output("profile-income-chart", "figure"),
    Input({"type": "profile-seg-date", "index": ALL}, "date"),
    Input({"type": "profile-seg-amount", "index": ALL}, "value"),
    Input("profile-income-growth", "value"),
    Input("profile-birth-date", "date"),
    Input("profile-coast-age", "value"),
)
def update_chart(dates, amounts, growth_pct, birth_date, coast_age):
    segs = [{"date": d, "amount": a} for d, a in zip(dates, amounts)
            if d and a not in (None, "")]
    if not segs or not birth_date:
        return _blank_chart()
    growth = (growth_pct or 0) / 100
    birth_year = int(str(birth_date)[:4])
    return _income_timeline_figure(segs, growth, birth_year,
                                   coast_age or _DEFAULTS["coast_age"])


@callback(
    Output("config-store", "data", allow_duplicate=True),
    Output("profile-demo-status", "children"),
    Input("profile-demo-save", "n_clicks"),
    State("profile-birth-date", "date"),
    State("profile-coast-age", "value"),
    State("profile-retirement-age", "value"),
    State("profile-claim-age", "value"),
    State("profile-death-age", "value"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def save_demographics(n, birth_date, coast, retire, claim, death, config_data):
    if not n:
        raise PreventUpdate
    cfg, uid, _ = _user_cfg(config_data)
    if not uid:
        raise PreventUpdate

    errors = []
    if coast is not None and retire is not None and coast >= retire:
        errors.append("Coast age must be less than retirement age.")
    if claim is not None and not (62 <= claim <= 70):
        errors.append("Social Security claim age must be between 62 and 70.")
    if not birth_date:
        errors.append("Birth date is required.")
    if errors:
        return no_update, dbc.Alert(" ".join(errors), color="danger", className="py-1 mb-0")

    payload = {
        "birth_date": str(birth_date)[:10],
        "coast_age": coast,
        "retirement_age": retire,
        "claim_age": claim,
        "death_age": death,
    }
    save_user_profile(uid, payload)
    cfg.setdefault("users", {}).setdefault(uid, {}).update(payload)
    return json.dumps(cfg), dbc.Alert("Saved.", color="success", className="py-1 mb-0")


@callback(
    Output("config-store", "data", allow_duplicate=True),
    Output("profile-income-status", "children"),
    Output("profile-segments-store", "data", allow_duplicate=True),
    Input("profile-income-save", "n_clicks"),
    State({"type": "profile-seg-date", "index": ALL}, "date"),
    State({"type": "profile-seg-amount", "index": ALL}, "value"),
    State("profile-income-growth", "value"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def save_income(n, dates, amounts, growth_pct, config_data):
    if not n:
        raise PreventUpdate
    cfg, uid, _ = _user_cfg(config_data)
    if not uid:
        raise PreventUpdate

    segments = []
    for d, a in zip(dates, amounts):
        if not d or a in (None, ""):
            continue
        segments.append({"date": str(d)[:10], "amount": float(a)})
    segments.sort(key=lambda s: s["date"])

    payload = {"income_segments": segments, "income_growth_rate": (growth_pct or 0) / 100}
    save_user_profile(uid, payload)
    cfg.setdefault("users", {}).setdefault(uid, {}).update(payload)

    status = dbc.Alert(f"Saved {len(segments)} income period(s).", color="success",
                       className="py-1 mb-0")
    return json.dumps(cfg), status, segments
