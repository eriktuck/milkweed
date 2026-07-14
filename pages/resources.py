"""Resources page — explainers for the modelling choices the app makes.

Standalone educational page (not part of the scenario hot path). Each topic is a
collapsible card. The first one, "What is CAGR?", makes the volatility-drag
argument visual: a bootstrap of the user's *current* portfolio over a fixed horizon
produces a right-skewed (log-normal) distribution of terminal values; compounding at
the **arithmetic mean** of the historical returns lands on the distribution's *mean*
(out in the optimistic right tail), while the **CAGR** (geometric mean) lands on its
*median* — a true 50/50. The punchline is the share of outcomes that fall *below* the
arithmetic-mean projection (always > 50%).
"""

import json

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from flask import session

from core.domain.returns import bootstrap_projection
from core.services.investments import fetch_latest_holdings
from core.services.market_data import build_returns_payload

dash.register_page(__name__, path="/resources")

# Fixed, contribution-free horizon for the CAGR illustration. Long enough that the
# log-normal skew (and the gap between mean and median) is unmistakable; pure
# compounding only, so the start balance is the sole driver of scale.
_CAGR_HORIZON = 30
_CAGR_NSIMS = 5000

_C_MEDIAN = "#3a9d5d"   # green — the CAGR / median (the honest middle)
_C_MEAN = "#b8770a"     # amber — the arithmetic mean (the optimistic tail)
_C_BARS = "#9bb4e0"     # muted blue histogram

_CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="#444"),
    showlegend=False,
)


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _selected_uid(use_case):
    return use_case or session.get("user_id")


def _money(v) -> str:
    """Abbreviated currency: $1.50M / $548K / $900 (mirrors pages/coast.py)."""
    v = float(v or 0)
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:,.0f}K"
    return f"${v:,.0f}"


def _collapsible_card(card_id: str, icon: str, title: str, body, is_open: bool = True):
    """A card whose header toggles a `dbc.Collapse` body (the page's one UI pattern).

    Header carries the title + a chevron; clicking toggles the body open/closed.
    Register the toggle with `_register_toggle(card_id)`.
    """
    head = html.Div([
        html.Div([
            html.I(className=f"fas {icon} me-2 text-muted"),
            html.Span(title, className="title"),
        ]),
        html.I(className=f"fas fa-chevron-{'up' if is_open else 'down'} text-muted",
               id=f"{card_id}-chevron"),
    ], id=f"{card_id}-head", className="module-head", n_clicks=0)
    collapse = dbc.Collapse(html.Div(body, className="module-body"),
                            id=f"{card_id}-collapse", is_open=is_open)
    return dbc.Card([head, collapse], className="module-card shadow-sm mb-2")


def _register_toggle(card_id: str):
    """Open/close + chevron-flip for a `_collapsible_card` (inverts current state, so
    it works whatever the card's initial `is_open`)."""
    @callback(
        Output(f"{card_id}-collapse", "is_open"),
        Output(f"{card_id}-chevron", "className"),
        Input(f"{card_id}-head", "n_clicks"),
        State(f"{card_id}-collapse", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle(_n, is_open):
        new = not is_open
        chevron = ("fas fa-chevron-up text-muted" if new
                   else "fas fa-chevron-down text-muted")
        return new, chevron


def _kde_density(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian kernel density of `samples` evaluated on `grid` (no scipy dependency).

    Silverman's rule-of-thumb bandwidth on the robust spread (min of SD and IQR/1.349)
    — smooth enough to read as a curve, narrow enough to keep the right-skew honest.
    """
    s = np.asarray(samples, dtype="float64")
    n = s.size
    if n == 0:
        return np.zeros_like(grid)
    std = float(s.std(ddof=1)) if n > 1 else 0.0
    q75, q25 = np.percentile(s, [75, 25])
    iqr = float(q75 - q25)
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    if sigma <= 0:
        sigma = std or 1.0
    h = 0.9 * sigma * n ** (-0.2)
    u = (grid[:, None] - s[None, :]) / h
    return np.exp(-0.5 * u * u).sum(axis=1) / (n * h * np.sqrt(2.0 * np.pi))


def _empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_CHART_LAYOUT, height=360, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[{"text": message, "xref": "paper", "yref": "paper",
                      "x": 0.5, "y": 0.5, "showarrow": False,
                      "font": {"color": "#999", "size": 13}}] if message else [],
    )
    return fig


# ── "What is CAGR?" card ───────────────────────────────────────────────────────────

_cagr_card = _collapsible_card(
    "res-cagr", "fa-percent", "What is CAGR?",
    body=[
        html.P([
            "We headline your portfolio's expected return as a ", html.B("CAGR"),
            " (compound annual growth rate, the geometric mean of past returns) rather "
            "than the simple ", html.B("arithmetic average"), " of the yearly returns. "
            "The reason is ", html.B("volatility drag"), ": a −40% year needs a +67% "
            "year just to break even, so a bumpy path always compounds to less than its "
            "average rate would suggest. The arithmetic mean ignores that and overstates "
            "the typical outcome.",
        ], className="text-muted small mb-2"),
        html.P([
            "The chart bootstraps your ", html.B("current portfolio"),
            f" forward {_CAGR_HORIZON} years — resampling whole historical years of "
            "real returns, no new contributions — over thousands of simulated paths. "
            "The spread of final values is right-skewed (a few lucky paths run away), so "
            "its ", html.B("mean sits well above its median"), ".",
        ], className="text-muted small mb-3"),
        dcc.Loading(dcc.Graph(id="res-cagr-chart",
                              config={"displayModeBar": False}), type="circle"),
        html.P(id="res-cagr-caption", className="text-muted small mt-2 mb-0"),
    ],
)


# ── Page layout ────────────────────────────────────────────────────────────────────

layout = html.Div(dbc.Container([
    # Network-bound returns analytics (yfinance/CPI + holdings), computed once per visit.
    dcc.Store(id="res-returns-store", storage_type="memory"),
    dbc.Row([
        dbc.Col(html.H1("Resources"), width="auto"),
    ], className="pt-3 pb-2", align="center"),
    html.P("Short explainers for the modelling choices behind the scenario planner.",
           className="text-muted small mb-3"),
    _cagr_card,
], fluid=False))


_register_toggle("res-cagr")


# ── Data + figure ──────────────────────────────────────────────────────────────────

@callback(
    Output("res-returns-store", "data"),
    Input("config-store", "data"),
    Input("use-case", "value"),
)
def compute_returns_data(_config_data, use_case):
    """Network-bound return pool + portfolio start value, run once per visit and parked
    in a store so the figure callback stays pure numpy (mirrors Coast's pattern)."""
    uid = _selected_uid(use_case)
    if not uid:
        raise dash.exceptions.PreventUpdate
    holdings = fetch_latest_holdings(uid)
    payload = build_returns_payload(holdings)
    start_value = sum(float(h.get("total_value", 0.0) or 0.0) for h in holdings)
    return {
        "ok": bool(payload.get("ok")) and start_value > 0,
        "pool": payload.get("real_return_pool", []),
        "n_years": payload.get("n_years", 0),
        "start_value": start_value,
    }


@callback(
    Output("res-cagr-chart", "figure"),
    Output("res-cagr-caption", "children"),
    Input("res-returns-store", "data"),
)
def render_cagr_chart(store):
    """Bootstrap distribution of terminal portfolio values with the median (≈ CAGR
    projection) and mean (≈ arithmetic-mean projection) marked. The gap between them —
    and the share of outcomes below the mean — is the volatility-drag story."""
    if not store or not store.get("ok"):
        return _empty("Upload your holdings on the Investments page to see this for "
                      "your own portfolio."), ""

    pool = np.asarray(store.get("pool") or [], dtype="float64")
    pool = pool[~np.isnan(pool)]
    start_value = float(store.get("start_value") or 0.0)
    n_years = int(store.get("n_years") or pool.size)
    if pool.size == 0 or start_value <= 0:
        return _empty("No historical return data available for these holdings."), ""

    geom = float(np.prod(1.0 + pool) ** (1.0 / pool.size) - 1.0)   # CAGR
    arith = float(np.mean(pool))                                   # arithmetic mean

    sims = bootstrap_projection(pool, start_value, [0.0], _CAGR_HORIZON,
                                n_sims=_CAGR_NSIMS)
    terminal = sims[-1]                                            # final-year values
    median = float(np.median(terminal))
    mean = float(np.mean(terminal))
    pct_below_mean = float(np.mean(terminal < mean)) * 100.0

    # Clip the long right tail for display so the body's bell shape stays legible; the
    # mean line still sits comfortably inside this window.
    x_hi = float(np.percentile(terminal, 99))

    # Smooth KDE curve over the displayed window instead of histogram bars.
    grid = np.linspace(0.0, x_hi, 256)
    density = _kde_density(terminal, grid)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid, y=density, mode="lines", fill="tozeroy",
        line=dict(width=2, color=_C_BARS),
        fillcolor="rgba(155,180,224,0.35)",
        hovertemplate="%{x:$,.0f}<extra></extra>",
    ))
    fig.add_vline(
        x=median, line_color=_C_MEDIAN, line_width=2.5,
        annotation_text=f"Median {_money(median)} · CAGR {geom:.1%}",
        annotation_position="top left",
        annotation_font=dict(size=11, color=_C_MEDIAN),
    )
    fig.add_vline(
        x=mean, line_color=_C_MEAN, line_width=2.5, line_dash="dash",
        annotation_text=f"Mean {_money(mean)} · avg return {arith:.1%}",
        annotation_position="top right",
        annotation_font=dict(size=11, color=_C_MEAN),
    )
    fig.update_layout(
        **_CHART_LAYOUT, height=380, margin=dict(l=10, r=10, t=40, b=10),
        bargap=0.02,
        xaxis=dict(title=f"Portfolio value after {_CAGR_HORIZON} years (today's dollars)",
                   range=[0, x_hi], tickprefix="$", tickformat="~s", showgrid=False),
        yaxis=dict(title="Likelihood of outcome", showgrid=False, zeroline=False,
                   showticklabels=False, rangemode="tozero"),
    )

    caption = (
        f"The curve is the distribution of {_CAGR_NSIMS:,} simulated paths for your "
        f"{_money(start_value)} portfolio over {_CAGR_HORIZON} years, drawing from "
        f"{n_years} years of history. Compounding at the {arith:.1%} arithmetic average "
        f"lands on the distribution's mean ({_money(mean)}) — but "
        f"{pct_below_mean:.0f}% of outcomes fall below it. The {geom:.1%} CAGR lands on "
        f"the median ({_money(median)}): half above, half below. That gap is volatility "
        f"drag, and it's why we plan on the CAGR."
    )
    return fig, caption
