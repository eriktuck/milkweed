"""Trends page — Concept C (Question Board + Movers rail).

Single-column board of purpose-built panels (Q1 CSP shares · Q2 Food & Dining
composite · Q3 income · Q4 guilt-free composition) plus a sticky "What changed"
rail. See .planning/spec-trends.md (Approved Layout — Concept C). Data helpers
live in core/services/trends.py; figure builders in core/utils/trends_charts.py;
this module is layout + callbacks only.

Owner is the sidebar `use-case` value (a uid, matching `account_owner` in the
store). The global category filter scopes the CSP-structured panels (Q1, Q4);
Q2 owns a hardcoded composite and Q3 owns an income-source toggle.
"""

from datetime import datetime as dt
from io import StringIO

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html
from dash.exceptions import PreventUpdate

from core.services import trends as T
from core.utils import trends_charts as C

dash.register_page(__name__, path="/trends")

_today = dt.today()
_default_start = f"{_today.year - 3}-01-01"

_GROUP_DISPLAY = {
    "income": "Income",
    "fixed": "Fixed Expenses",
    "investments": "Investments",
    "sinking": "Sinking",
    "guilt-free": "Guilt-free Spending",
}
_GROUP_ORDER = list(_GROUP_DISPLAY)

_BUCKET_LABELS = {
    "fixed": "Fixed", "investments": "Investments",
    "sinking": "Sinking", "guilt-free": "Guilt-free",
}


# ── Formatting helpers ──────────────────────────────────────────────────────────

def _money(v) -> str:
    v = float(v or 0)
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:,.2f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:,.1f}K"
    return f"${v:,.0f}"


def _delta_chip(cur, base, *, higher_is_bad=True):
    """A '▲ 11%' / '▼ 7%' span colored by whether the change is good or bad."""
    if cur is None or base is None or base == 0:
        return html.Span("– new", className="small text-muted")
    pct = (cur - base) / abs(base)
    if abs(pct) < 0.005:
        return html.Span("– flat", className="small text-muted")
    up = cur > base
    good = (not up) if higher_is_bad else up
    cls = "text-success" if good else "text-danger"
    return html.Span(f"{'▲' if up else '▼'} {abs(pct):.0%}", className=f"small {cls}")


def _series_delta(s: pd.Series, baseline: str):
    """(current, baseline) for a monthly series: last point vs last-year point or
    trailing-12 mean. `budget` falls back to trailing until per-category budgets wire up."""
    if s is None or len(s) == 0:
        return None, None
    cur = float(s.iloc[-1])
    if baseline == "last_year" and len(s) >= 13:
        return cur, float(s.iloc[-13])
    prior = s.iloc[-13:-1] if len(s) >= 13 else s.iloc[:-1]
    return cur, (float(prior.mean()) if len(prior) else None)


def _stat(children):
    return html.Div(children, className="small text-muted mb-2 d-flex flex-wrap gap-3")


def _stat_item(label, value, chip=None):
    bits = [html.Span(f"{label} ", className="text-muted"),
            html.Span(value, className="fw-semibold text-body")]
    if chip is not None:
        bits.append(html.Span(" "))
        bits.append(chip)
    return html.Span(bits)


# ── Panel + control scaffolding ──────────────────────────────────────────────────

def _graph(id_):
    return dcc.Loading(dcc.Graph(id=id_, config={"displayModeBar": False},
                                 style={"cursor": "pointer"}), type="circle")


def _panel(eyebrow, title, chart_id, stat_id, *, header_extra=None,
           back_id=None, breadcrumb_id=None):
    head = [html.Div(eyebrow, className="text-primary small text-uppercase fw-semibold"),
            html.Div(title, className="fw-semibold mb-1", style={"fontSize": "15px"})]
    if header_extra is not None:
        head.append(header_extra)
    body = list(head)
    if breadcrumb_id is not None:
        body.append(html.Div(id=breadcrumb_id, className="small mb-1"))
    if back_id is not None:
        body.append(dbc.Button("← Back", id=back_id, color="secondary", outline=True,
                                size="sm", class_name="mb-2", style={"display": "none"}))
    body.append(html.Div(id=stat_id))
    body.append(_graph(chart_id))
    return dbc.Card(dbc.CardBody(body), class_name="mb-3")


_controls = dbc.Card(dbc.CardBody(dbc.Row([
    dbc.Col([dbc.Label("Date range", class_name="small text-muted mb-1"),
             html.Div(dcc.DatePickerRange(
                 id="tr-date-picker", start_date=_default_start,
                 end_date=_today.strftime("%Y-%m-%d"), min_date_allowed="2020-01-01",
                 number_of_months_shown=2, persistence=True, updatemode="bothdates",
                 style={"borderWidth": 0}))], width="auto"),
    dbc.Col([dbc.Label("Smoothing", class_name="small text-muted mb-1"),
             dbc.RadioItems(id="tr-smoothing", inline=True, value="ME",
                            options=[{"label": "Annual", "value": "YE"},
                                     {"label": "Monthly", "value": "ME"},
                                     {"label": "Weekly", "value": "W"},
                                     {"label": "Daily", "value": "D"}])], width="auto"),
    dbc.Col([dbc.Label("Metric", class_name="small text-muted mb-1"),
             dbc.RadioItems(id="tr-display", inline=True, value="dollars",
                            options=[{"label": "$", "value": "dollars"},
                                     {"label": "% income", "value": "percent"}])], width="auto"),
    dbc.Col([dbc.Label("Baseline", class_name="small text-muted mb-1"),
             dbc.RadioItems(id="tr-baseline", inline=True, value="trailing_12mo",
                            options=[{"label": "Trailing 12-mo", "value": "trailing_12mo"},
                                     {"label": "Last year", "value": "last_year"},
                                     {"label": "Budget*", "value": "budget"}])], width="auto"),
], class_name="g-3 align-items-end")), class_name="mb-2")


_filter_panel = dbc.Row(dbc.Col([
    dbc.Button("Filter categories", id="tr-filter-btn", color="light", size="sm",
               class_name="border text-muted"),
    dbc.Collapse(dbc.Card(dbc.CardBody([
        dbc.Row(dbc.Col(dbc.ButtonGroup([
            dbc.Button("Select all", id="tr-filter-select-all", color="link", size="sm",
                       class_name="text-muted p-0 me-2"),
            dbc.Button("Deselect all", id="tr-filter-deselect-all", color="link", size="sm",
                       class_name="text-muted p-0"),
        ]), class_name="d-flex justify-content-end"), class_name="mb-1"),
        dbc.Row([
            dbc.Col([
                dbc.Button(f"[ ] {label}", id=f"tr-grp-{group}", color="link", size="sm",
                           class_name="fw-semibold small text-uppercase mb-1 p-0 "
                                      "text-decoration-none text-muted d-block text-start"),
                dbc.Checklist(id=f"tr-filter-{group}", options=[], value=[], class_name="small"),
            ], md=True, xs=6)
            for group, label in _GROUP_DISPLAY.items()
        ]),
    ]), class_name="shadow-sm mt-2"), id="tr-filter-collapse", is_open=False),
]), class_name="pb-3")


layout = html.Div([
    dcc.Store(id="tr-q1-drill", data=None),
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Trends"), width="auto"),
            dbc.Col(html.Span(id="tr-meta", className="text-muted small"),
                    className="d-flex align-items-center"),
        ], className="pt-3 pb-2", align="center"),
        _controls,
        _filter_panel,
        dbc.Row([
            dbc.Col([
                _panel("Q1 · Conscious Spending Plan",
                       "How are my CSP buckets shifting as a share of income?",
                       "tr-q1-chart", "tr-q1-stat",
                       back_id="tr-q1-back", breadcrumb_id="tr-q1-breadcrumb"),
                _panel("Q2 · Composite metric", "How much do I spend on food & dining?",
                       "tr-q2-chart", "tr-q2-stat"),
                _panel("Q3 · Income", "How has my income changed over time?",
                       "tr-q3-chart", "tr-q3-stat"),
                _panel("Q4 · Guilt-free composition",
                       "Within guilt-free, what dominates — and how is it shifting?",
                       "tr-q4-chart", "tr-q4-stat"),
            ], lg=8),
            dbc.Col(html.Div(dbc.Card(dbc.CardBody(html.Div(id="tr-rail")),
                                      class_name="mb-3"),
                             style={"position": "sticky", "top": "14px"}), lg=4),
        ], class_name="g-3"),
    ], fluid=True),
])


# ── Filter callbacks (group-toggle UX completion is Phase 4.4) ───────────────────

@callback(Output("tr-filter-collapse", "is_open"),
          Input("tr-filter-btn", "n_clicks"),
          State("tr-filter-collapse", "is_open"), prevent_initial_call=True)
def toggle_filter_panel(_n, is_open):
    return not is_open


@callback(
    *[Output(f"tr-filter-{g}", "value", allow_duplicate=True) for g in _GROUP_ORDER],
    Input("tr-filter-select-all", "n_clicks"),
    Input("tr-filter-deselect-all", "n_clicks"),
    *[State(f"tr-filter-{g}", "options") for g in _GROUP_ORDER],
    prevent_initial_call=True,
)
def bulk_select_filter(_a, _b, *options_per_group):
    if ctx.triggered_id == "tr-filter-select-all":
        return tuple([o["value"] for o in opts] for opts in options_per_group)
    return tuple([] for _ in _GROUP_ORDER)


@callback(
    *[Output(f"tr-filter-{g}", "value", allow_duplicate=True) for g in _GROUP_ORDER],
    *[Input(f"tr-grp-{g}", "n_clicks") for g in _GROUP_ORDER],
    *[State(f"tr-filter-{g}", "options") for g in _GROUP_ORDER],
    *[State(f"tr-filter-{g}", "value") for g in _GROUP_ORDER],
    prevent_initial_call=True,
)
def toggle_group(*args):
    """Per-group header click: flip the whole group. If every subcategory is
    already selected, deselect them all; otherwise select them all. Action-only
    (reads current state, writes the group's checklist) — group actions override
    individual picks, with no checkbox cycle."""
    n = len(_GROUP_ORDER)
    options, values = args[n:2 * n], args[2 * n:3 * n]
    out = [dash.no_update] * n
    trig = ctx.triggered_id
    if isinstance(trig, str) and trig.startswith("tr-grp-"):
        i = _GROUP_ORDER.index(trig[len("tr-grp-"):])
        opts = [o["value"] for o in (options[i] or [])]
        cur = set(values[i] or [])
        out[i] = [] if (opts and cur == set(opts)) else opts
    return tuple(out)


@callback(
    *[Output(f"tr-grp-{g}", "children") for g in _GROUP_ORDER],
    *[Input(f"tr-filter-{g}", "value") for g in _GROUP_ORDER],
    *[State(f"tr-filter-{g}", "options") for g in _GROUP_ORDER],
)
def sync_group_labels(*args):
    """One-way indicator (subs → header glyph): ☑ all · ◧ some · ☐ none. Safe —
    writes only the button label, which is no callback's input, so no cycle."""
    n = len(_GROUP_ORDER)
    values, options = args[:n], args[n:2 * n]
    out = []
    for i, g in enumerate(_GROUP_ORDER):
        opts = {o["value"] for o in (options[i] or [])}
        cur = set(values[i] or [])
        mark = "[✓]" if (opts and cur == opts) else ("[ ]" if not cur else "[–]")
        out.append(f"{mark} {_GROUP_DISPLAY[g]}")
    return tuple(out)


@callback(
    *[x for g in _GROUP_ORDER
      for x in (Output(f"tr-filter-{g}", "options"), Output(f"tr-filter-{g}", "value"))],
    Input("transaction-data-store", "data"),
    Input("use-case", "value"),
)
def populate_filter_options(store, use_case):
    if not store or not use_case:
        return tuple([] for _ in range(2 * len(_GROUP_ORDER)))
    df = pd.read_json(StringIO(store), orient="split")
    df = df[df["account_owner"] == use_case]
    out = []
    for group in _GROUP_ORDER:
        cats = sorted(df[df["csp_label"] == group]["category_name"].dropna().unique().tolist())
        out += [[{"label": c, "value": c} for c in cats], cats]
    return tuple(out)


@callback(Output("tr-q1-drill", "data"),
          Input("tr-q1-chart", "clickData"),
          Input("tr-q1-back", "n_clicks"),
          Input("use-case", "value"),
          State("tr-q1-drill", "data"), prevent_initial_call=True)
def update_q1_drill(click_data, _back, _uc, current):
    trig = ctx.triggered_id
    if trig in ("tr-q1-back", "use-case"):
        return None
    if trig == "tr-q1-chart" and click_data and current is None:
        label = click_data["points"][0].get("customdata")
        # only buckets (not the income reference line) are drillable
        if label in _BUCKET_LABELS:
            return label
    return dash.no_update


# ── Board ─────────────────────────────────────────────────────────────────────

@callback(
    Output("tr-q1-chart", "figure"), Output("tr-q1-stat", "children"),
    Output("tr-q1-back", "style"), Output("tr-q1-breadcrumb", "children"),
    Output("tr-q2-chart", "figure"), Output("tr-q2-stat", "children"),
    Output("tr-q3-chart", "figure"), Output("tr-q3-stat", "children"),
    Output("tr-q4-chart", "figure"), Output("tr-q4-stat", "children"),
    Output("tr-meta", "children"),
    Input("transaction-data-store", "data"),
    Input("use-case", "value"),
    Input("tr-date-picker", "start_date"), Input("tr-date-picker", "end_date"),
    Input("tr-smoothing", "value"), Input("tr-display", "value"),
    Input("tr-baseline", "value"), Input("tr-q1-drill", "data"),
    *[Input(f"tr-filter-{g}", "value") for g in _GROUP_ORDER],
)
def render_board(store, use_case, start, end, smoothing, display, baseline,
                 q1_drill, *filters):
    if not store or not use_case:
        raise PreventUpdate
    df = pd.read_json(StringIO(store), orient="split")
    as_percent = display == "percent"

    selected = [c for group in filters for c in (group or [])]
    fdf = df[df["category_name"].isin(selected)] if selected else df  # Q1/Q4 scope

    owner_df = df[df["account_owner"] == use_case]
    if owner_df.empty:
        empty = C.empty_figure(message="No transactions for this view")
        blank = _stat("")
        meta = "No transactions for this view"
        return (empty, blank, {"display": "none"}, "", empty, blank,
                empty, blank, empty, blank, meta)

    # ── Q1 — CSP shares (or drill-down into a bucket's categories) ───────────────
    if q1_drill in _BUCKET_LABELS:
        comp = T.composition_over_time(fdf, use_case, start, end, smoothing,
                                       csp_label=q1_drill, top_n=8)
        q1_fig = C.figure_composition(comp, smoothing=smoothing)
        q1_stat = _stat(f"{_BUCKET_LABELS[q1_drill]} broken down by category ($).")
        q1_back = {"display": "inline-block"}
        q1_crumb = html.Span([html.Span("All buckets ", className="text-primary"),
                              html.Span(f"› {_BUCKET_LABELS[q1_drill]}", className="text-muted")])
    else:
        piv, inc = T.csp_shares_over_time(fdf, use_case, start, end, smoothing,
                                          as_percent=as_percent)
        q1_fig = C.figure_csp_shares(piv, inc, as_percent=as_percent, smoothing=smoothing)
        items = []
        for key, lbl in _BUCKET_LABELS.items():
            if key in piv.columns:
                cur, base = _series_delta(piv[key], baseline)
                val = f"{cur:.0%}" if as_percent else _money(cur)
                items.append(_stat_item(lbl, val, _delta_chip(cur, base)))
        q1_stat = _stat(items)
        q1_back = {"display": "none"}
        q1_crumb = html.Span("Click a band to drill into its categories", className="text-muted")

    # ── Q2 — Food & Dining composite (dining broken out into shaded subcategories) ─
    co = T.composite_over_time(df, use_case, start, end, smoothing)
    if co.empty:
        q2_fig, q2_stat = C.empty_figure(message="No food & dining categories"), _stat("")
    else:
        total = co[T.COMBINED_LABEL]
        cur, base = _series_delta(total, baseline)
        q2_fig = C.figure_composite(co, smoothing=smoothing, baseline=base)
        # Stat headlines stay at the part level (Groceries vs Dining out) even
        # though the chart breaks dining into subcategories.
        parts = T.DEFAULT_FOOD_DINING_PARTS
        dining_cols = [c for c in co.columns if c in parts["Dining out"]]
        items = [_stat_item("Total/mo", _money(cur), _delta_chip(cur, base))]
        if "Groceries" in co.columns:
            items.append(_stat_item("Groceries", _money(co["Groceries"].iloc[-1])))
        if dining_cols:
            items.append(_stat_item("Dining out", _money(co[dining_cols].sum(axis=1).iloc[-1])))
        q2_stat = _stat(items)

    # ── Q3 — Income (Paychecks/Other are separate legend series — double-click the
    #         legend to isolate one, Plotly built-in; no radio needed) ─────────────
    isp = T.income_split_over_time(df, use_case, start, end, smoothing)
    if isp.empty:
        q3_fig, q3_stat = C.empty_figure(message="No income in range"), _stat("")
    else:
        q3_fig = C.figure_income(isp, smoothing=smoothing)
        cur, base = _series_delta(isp[T.TOTAL_INCOME_LABEL], baseline)
        q3_stat = _stat([
            _stat_item("Run-rate/mo", _money(cur), _delta_chip(cur, base, higher_is_bad=False)),
            _stat_item("Paychecks", _money(isp["Paychecks"].iloc[-1])),
            _stat_item("Other", _money(isp["Other sources"].iloc[-1])),
        ])

    # ── Q4 — Guilt-free composition ──────────────────────────────────────────────
    comp4 = T.composition_over_time(fdf, use_case, start, end, smoothing,
                                    csp_label="guilt-free", top_n=5)
    if comp4.empty:
        q4_fig, q4_stat = C.empty_figure(message="No guilt-free spending in range"), _stat("")
    else:
        q4_fig = C.figure_composition(comp4, smoothing=smoothing)
        totals = comp4.sum().sort_values(ascending=False)
        biggest = totals.index[0]
        q4_stat = _stat([_stat_item("Biggest", str(biggest)),
                         _stat_item("Categories shown", f"{len(comp4.columns)}")])

    meta = f"Viewing {len(owner_df):,} transactions"
    return (q1_fig, q1_stat, q1_back, q1_crumb, q2_fig, q2_stat,
            q3_fig, q3_stat, q4_fig, q4_stat, meta)


# ── Movers rail ───────────────────────────────────────────────────────────────

@callback(Output("tr-rail", "children"),
          Input("transaction-data-store", "data"),
          Input("use-case", "value"),
          Input("tr-date-picker", "end_date"),
          Input("tr-baseline", "value"))
def render_rail(store, use_case, end, baseline):
    if not store or not use_case:
        raise PreventUpdate
    df = pd.read_json(StringIO(store), orient="split")

    period = T.resolve_current_period(df, use_case, end)
    note = None
    effective = baseline
    level = "category_name"          # trailing/last-year compare at category level
    baseline_amounts = None
    is_budget = False

    if baseline == "budget":
        # Budgets are keyed by CSP key, so the budget baseline compares at csp level.
        from core.services.firebase import fetch_budget
        budget = {}
        if period is not None:
            budget = fetch_budget(use_case, period.year, period.month)
        budget = {k: v for k, v in budget.items() if k != "income"}  # income isn't spend
        if budget:
            level, baseline_amounts, is_budget = "csp", budget, True
        else:
            effective = "trailing_12mo"
            when = period.strftime("%b %Y") if period is not None else "this month"
            note = f"No budget found for {when} — showing trailing 12-mo."

    mv = T.category_movers(df, use_case, as_of=end, baseline=effective,
                           level=level, baseline_amounts=baseline_amounts)

    sub = f"vs {effective.replace('_', ' ')}"
    if period is not None:
        sub += f" · {period.strftime('%b %Y')}"
    header = [html.H5("What changed", className="mb-0"),
              html.Div(sub, className="small text-muted mb-2")]
    if note:
        header.append(html.Div(note, className="small fst-italic text-warning mb-2"))
    if mv.empty:
        return header + [html.Div("No movers in range.", className="small text-muted")]

    over = T.flag_overages(mv)
    up, down = T.top_movers(mv, 4)

    children = list(header)
    if not over.empty:
        children.append(html.Div("⚠ Over budget" if is_budget else "⚠ Temporary overages",
                                  className="small text-uppercase text-muted mt-3 mb-1"))
        usual = "budget" if is_budget else "usual"
        for cat, r in over.iterrows():
            children.append(html.Div(
                [html.Span(f"+{_money(r['delta'])} ", className="fw-bold text-danger"),
                 html.Span(f"{_pretty(cat, is_budget)} — {_money(r['current'])} "
                           f"vs {_money(r['baseline'])} {usual}", className="small")],
                className="p-2 mb-1 rounded",
                style={"background": "rgba(224,105,127,.08)",
                       "border": "1px solid rgba(224,105,127,.28)"}))

    up_title = "Most over budget ▲" if is_budget else "Biggest movers ▲"
    down_title = "Most under budget ▼" if is_budget else "Biggest movers ▼"
    children.append(_mover_section(up_title, up, up=True, pretty=is_budget))
    children.append(_mover_section(down_title, down, up=False, pretty=is_budget))
    return children


def _pretty(name, is_csp):
    """CSP keys (lowercase_underscore) → 'Title Case'; category names left as-is."""
    return str(name).replace("_", " ").title() if is_csp else str(name)


def _mover_section(title, frame, *, up, pretty=False):
    rows = [html.Div(title, className="small text-uppercase text-muted mt-3 mb-1")]
    if frame.empty:
        rows.append(html.Div("—", className="small text-muted"))
        return html.Div(rows)
    for cat, r in frame.iterrows():
        pct = r["pct_change"]
        pct_txt = f"{'▲' if up else '▼'}{abs(pct):.0%}" if pd.notna(pct) else "new"
        cls = "text-danger" if up else "text-success"
        rows.append(html.Div([
            html.Div([html.Div(_pretty(cat, pretty), className="text-body"),
                      html.Div(_money(r["current"]), className="small text-muted")],
                     className="flex-grow-1"),
            html.Span(pct_txt, className=f"fw-bold {cls}"),
        ], className="d-flex align-items-center gap-2 p-2 mb-1 rounded",
            style={"background": "var(--bs-tertiary-bg, #0f1629)",
                   "borderLeft": f"3px solid {'#e0697f' if up else '#3f9e86'}"}))
    return html.Div(rows)
