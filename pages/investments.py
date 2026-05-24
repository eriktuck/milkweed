import datetime
import json

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import dcc, html, callback, Input, Output, State, ALL
from flask import session

from core.services.investments import (
    ASSET_CLASS_MAP,
    compute_ytd_contributions,
    delete_investment_data,
    fetch_investment_transactions,
    fetch_latest_holdings,
    reconstruct_portfolio_history,
    save_investment_account_config,
)

dash.register_page(__name__, path="/investments")

# ── Color constants ───────────────────────────────────────────────────────────
_C_RETIREMENT = "#5b7ec9"
_C_TAXABLE = "#e2a03f"
_C_US = "#5b7ec9"
_C_INTL = "#4ade80"
_C_BONDS = "#f59e0b"
_C_CASH = "#888888"

_CHART_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(color="#444"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

_ALLOC_CLASSES = ["US Equity", "Intl Equity", "Bonds", "Cash"]
_ALLOC_COLORS = [_C_US, _C_INTL, _C_BONDS, _C_CASH]
_RETIREMENT_TYPES = frozenset({
    "IRA", "Roth IRA", "401k", "Roth 401k", "403b", "457b", "SEP IRA", "SIMPLE IRA",
})
_ACCOUNT_TYPES = [
    "IRA", "Roth IRA", "401k", "Roth 401k", "403b", "457b",
    "SEP IRA", "SIMPLE IRA", "Brokerage", "HSA", "529", "Other",
]


# ── Sub-component builders ────────────────────────────────────────────────────

def _ban_card(label: str, value: str, subtitle: str) -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.P(label, className="text-muted small mb-1"),
        html.H3(value, className="mb-0 fw-bold"),
        html.P(subtitle, className="text-muted small mt-1 mb-0"),
    ]))


def _alloc_chart(values: list[float]) -> go.Figure:
    total = sum(values) or 1
    pcts = [v / total for v in values]
    fig = go.Figure()
    for cls, pct, color in zip(_ALLOC_CLASSES, pcts, _ALLOC_COLORS):
        fig.add_trace(go.Bar(
            x=[pct], y=[""], name=cls, orientation="h",
            marker_color=color,
            text=f"{pct:.0%}",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=10, color="white"),
            hovertemplate=f"{cls}: {{%x:.1%}}<extra></extra>",
            showlegend=False,
        ))
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#444"),
        barmode="stack",
        height=50,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showgrid=False, visible=False, range=[0, 1]),
        yaxis=dict(showgrid=False, visible=False),
        showlegend=False,
    )
    return fig


def _alloc_card(title: str, total: float, alloc: dict) -> dbc.Card:
    values = [alloc.get(cls, 0.0) for cls in _ALLOC_CLASSES]
    return dbc.Card(dbc.CardBody([
        html.P(title, className="text-muted small mb-1"),
        html.H5(f"${total:,.0f}", className="mb-0 fw-bold"),
        html.Div(className="mt-2 mb-1", children=dcc.Graph(
            figure=_alloc_chart(values),
            config={"displayModeBar": False},
            style={"height": "50px"},
        )),
        html.Div([
            html.Span(
                cls, className="badge me-1",
                style={"backgroundColor": color, "fontSize": "10px"},
            )
            for cls, color in zip(_ALLOC_CLASSES, _ALLOC_COLORS)
        ]),
    ]))


# ── AG Grid definitions ───────────────────────────────────────────────────────

_HOLDINGS_COLS = [
    {"field": "symbol", "headerName": "Symbol", "width": 90},
    {"field": "investment_name", "headerName": "Name", "flex": 2},
    {"field": "account_number", "headerName": "Account", "width": 100},
    {"field": "nickname", "headerName": "Nickname", "width": 130},
    {"field": "account_type", "headerName": "Type", "width": 110},
    {"field": "shares", "headerName": "Shares", "width": 100,
     "valueFormatter": {"function": "params.value?.toLocaleString('en-US', {minimumFractionDigits:3})"}},
    {"field": "share_price", "headerName": "Price", "width": 90,
     "valueFormatter": {"function": "params.value != null ? '$' + params.value.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) : ''"}},
    {"field": "total_value", "headerName": "Value", "width": 110,
     "valueFormatter": {"function": "params.value != null ? '$' + params.value.toLocaleString('en-US', {minimumFractionDigits:0, maximumFractionDigits:0}) : ''"}},
    {"field": "pct_portfolio", "headerName": "% Portfolio", "width": 110,
     "valueFormatter": {"function": "params.value != null ? (params.value*100).toFixed(1) + '%' : ''"}},
]

_HOLDINGS_PLACEHOLDER = [
    {"symbol": "—", "investment_name": "No data uploaded yet.", "account_number": "—",
     "nickname": "", "account_type": "—", "shares": 0.0, "share_price": 0.0,
     "total_value": 0.0, "pct_portfolio": 0.0},
]

_TXN_COLS = [
    {"field": "trade_date", "headerName": "Date", "width": 110, "sort": "desc"},
    {"field": "transaction_type", "headerName": "Type", "width": 130},
    {"field": "transaction_description", "headerName": "Description", "flex": 2},
    {"field": "symbol", "headerName": "Symbol", "width": 90},
    {"field": "shares", "headerName": "Shares", "width": 90},
    {"field": "share_price", "headerName": "Price", "width": 90,
     "valueFormatter": {"function": "params.value ? '$' + params.value.toFixed(2) : ''"}},
    {"field": "net_amount", "headerName": "Amount", "width": 110,
     "valueFormatter": {"function": "params.value != null ? '$' + params.value.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) : ''"}},
    {"field": "account_number", "headerName": "Account", "width": 100},
]

_TXN_PLACEHOLDER = [
    {"trade_date": "—", "transaction_type": "—", "transaction_description": "No data uploaded yet.",
     "symbol": "", "shares": 0.0, "share_price": 0.0, "net_amount": 0.0, "account_number": "—"},
]

from components.investment_upload import investment_upload  # noqa: E402 (after module-level constants)

# ── Account categorization modal ──────────────────────────────────────────────

_account_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Investment Accounts")),
        dbc.ModalBody(html.Div(id="account-modal-body")),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="account-modal-cancel", color="secondary", className="me-2"),
            dbc.Button("Save", id="account-modal-save", color="primary"),
        ]),
    ],
    id="account-modal",
    size="lg",
    is_open=False,
)

# ── Delete-data confirmation modal ────────────────────────────────────────────

_delete_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Delete Investment Data")),
        dbc.ModalBody([
            html.P(
                "This permanently deletes all your investment holdings, "
                "transaction history, and account labels from the database."
            ),
            html.P("This cannot be undone.", className="fw-bold text-danger mb-0"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="delete-data-cancel", color="secondary", className="me-2"),
            dbc.Button("Delete everything", id="delete-data-confirm", color="danger"),
        ]),
    ],
    id="delete-data-modal",
    is_open=False,
)

# ── Page layout ───────────────────────────────────────────────────────────────

layout = html.Div([
    dcc.Store(id="accounts-for-modal", storage_type="memory"),
    _account_modal,
    _delete_modal,
    dbc.Container([

        # Header
        dbc.Row([
            dbc.Col(html.H1("Investments"), width="auto"),
            dbc.Col(
                dbc.Button(
                    [html.I(className="fas fa-cog me-1"), "Edit Accounts"],
                    id="edit-accounts-btn",
                    color="outline-secondary",
                    size="sm",
                ),
                width="auto",
                className="d-flex align-items-center",
            ),
            dbc.Col(
                dbc.Button(
                    [html.I(className="fas fa-trash me-1"), "Delete Investment Data"],
                    id="delete-data-btn",
                    color="outline-danger",
                    size="sm",
                ),
                width="auto",
                className="d-flex align-items-center",
            ),
            dbc.Col(investment_upload, width=4, className="ms-auto d-flex align-items-center"),
        ], className="pt-3 pb-2", align="center"),

        # Row 1 — BANs (filled by callback)
        dbc.Row([
            dbc.Col(id="investments-ban-total", width=4),
            dbc.Col(id="investments-ban-contributions", width=4),
            dbc.Col(id="investments-ban-coastfire", width=4),
        ], className="mb-3 g-3"),

        # Row 2 — Portfolio value over time
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Portfolio Value Over Time", className="text-muted small mb-2"),
                dcc.Loading(
                    dcc.Graph(
                        id="investments-history-chart",
                        config={"displayModeBar": False},
                    ),
                    type="circle",
                ),
            ])))
        ], className="mb-3"),

        # Row 3 — Allocation cards (filled by callback)
        dbc.Row([
            dbc.Col(id="investments-alloc-combined", width=4),
            dbc.Col(id="investments-alloc-retirement", width=4),
            dbc.Col(id="investments-alloc-taxable", width=4),
        ], className="mb-3 g-3"),

        # Row 4 — Holdings table
        dbc.Row([
            dbc.Col([
                html.P("Current Holdings", className="text-muted small mb-2"),
                dag.AgGrid(
                    id="investments-holdings-grid",
                    className="ag-theme-quartz",
                    columnDefs=_HOLDINGS_COLS,
                    rowData=_HOLDINGS_PLACEHOLDER,
                    defaultColDef={"sortable": True, "resizable": True, "filter": True},
                    dashGridOptions={"domLayout": "autoHeight"},
                    style={"width": "100%"},
                ),
            ])
        ], className="mb-3"),

        # Row 5 — Transactions table
        dbc.Row([
            dbc.Col([
                html.P("Recent Transactions", className="text-muted small mb-2"),
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id="investments-txn-type-filter",
                            placeholder="All transaction types",
                            multi=True,
                            clearable=True,
                        ),
                        width=4,
                    ),
                    dbc.Col(
                        dcc.Dropdown(
                            id="investments-txn-account-filter",
                            placeholder="All accounts",
                            multi=True,
                            clearable=True,
                        ),
                        width=4,
                    ),
                ], className="mb-2"),
                dag.AgGrid(
                    id="investments-transactions-grid",
                    className="ag-theme-quartz",
                    columnDefs=_TXN_COLS,
                    rowData=_TXN_PLACEHOLDER,
                    defaultColDef={"sortable": True, "resizable": True, "filter": True},
                    dashGridOptions={"domLayout": "autoHeight"},
                    style={"width": "100%"},
                ),
            ])
        ], className="mb-4"),

    ], fluid=False),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@callback(
    Output("investments-ban-total", "children"),
    Output("investments-ban-contributions", "children"),
    Output("investments-ban-coastfire", "children"),
    Output("investments-alloc-combined", "children"),
    Output("investments-alloc-retirement", "children"),
    Output("investments-alloc-taxable", "children"),
    Output("investments-holdings-grid", "rowData"),
    Input("config-store", "data"),
    Input("investments-data-version", "data"),
)
def update_portfolio_data(config_data, _version):
    if not config_data or not isinstance(config_data, str):
        raise dash.exceptions.PreventUpdate

    uid = session.get("user_id")
    if not uid:
        raise dash.exceptions.PreventUpdate

    config = json.loads(config_data)
    user_cfg = config.get("users", {}).get(uid, {})
    investment_accounts: dict[str, str] = user_cfg.get("investment_accounts") or {}
    investment_account_nicknames: dict[str, str] = user_cfg.get("investment_account_nicknames") or {}

    holdings = fetch_latest_holdings(uid)
    transactions = fetch_investment_transactions(uid)

    # ── Total portfolio value ────────────────────────────────────────────────
    total_value = sum(h.get("total_value", 0.0) for h in holdings)

    # ── YTD contributions ────────────────────────────────────────────────────
    ytd = compute_ytd_contributions(transactions, datetime.date.today().year)

    # ── CoastFIRE progress ───────────────────────────────────────────────────
    nw = user_cfg.get("net_worth") or {}
    coast_target = nw.get("coast_target")

    # ── Allocation by asset class and account type ───────────────────────────
    from collections import defaultdict as _dd
    alloc_combined: dict[str, float] = _dd(float)
    alloc_retirement: dict[str, float] = _dd(float)
    alloc_taxable: dict[str, float] = _dd(float)

    for h in holdings:
        sym = h.get("symbol", "")
        acct = h.get("account_number", "")
        value = h.get("total_value", 0.0)
        asset_class = ASSET_CLASS_MAP.get(sym, "Other")
        acct_type = investment_accounts.get(acct, "Taxable")

        alloc_combined[asset_class] += value
        if acct_type in _RETIREMENT_TYPES:
            alloc_retirement[asset_class] += value
        else:
            alloc_taxable[asset_class] += value

    ret_total = sum(alloc_retirement.values())
    tax_total = sum(alloc_taxable.values())

    # ── Build outputs ────────────────────────────────────────────────────────
    snapshot_label = f"as of {datetime.date.today().strftime('%b %d, %Y')}"
    ytd_label = f"Jan–{datetime.date.today().strftime('%b %Y')}"

    ban_total = _ban_card("Total Portfolio Value", f"${total_value:,.0f}", snapshot_label)
    ban_contributions = _ban_card("YTD Contributions", f"${ytd:,.0f}", ytd_label)

    if coast_target:
        pct = total_value / float(coast_target)
        ban_coastfire = _ban_card(
            "CoastFIRE Progress",
            f"{pct:.0%}",
            f"${total_value:,.0f} of ${float(coast_target):,.0f} target",
        )
    else:
        ban_coastfire = _ban_card(
            "CoastFIRE Progress",
            f"${total_value:,.0f}",
            "Add coast_target to net worth config",
        )

    card_combined = _alloc_card("Combined Portfolio", total_value, alloc_combined)
    card_retirement = _alloc_card("Retirement", ret_total, alloc_retirement)
    card_taxable = _alloc_card("Non-Retirement", tax_total, alloc_taxable)

    holdings_rows = [
        {
            "symbol": h.get("symbol", ""),
            "investment_name": h.get("investment_name", ""),
            "account_number": f"****{h.get('account_number', '')}",
            "nickname": investment_account_nicknames.get(h.get("account_number", ""), ""),
            "account_type": investment_accounts.get(h.get("account_number", ""), "—"),
            "shares": h.get("shares", 0.0),
            "share_price": h.get("share_price", 0.0),
            "total_value": h.get("total_value", 0.0),
            "pct_portfolio": h.get("total_value", 0.0) / total_value if total_value else 0.0,
        }
        for h in sorted(holdings, key=lambda x: -x.get("total_value", 0.0))
    ] or _HOLDINGS_PLACEHOLDER

    return (
        ban_total, ban_contributions, ban_coastfire,
        card_combined, card_retirement, card_taxable,
        holdings_rows,
    )


@callback(
    Output("investments-history-chart", "figure"),
    Input("config-store", "data"),
    Input("investments-data-version", "data"),
)
def update_history_chart(config_data, _version):
    if not config_data or not isinstance(config_data, str):
        raise dash.exceptions.PreventUpdate

    uid = session.get("user_id")
    if not uid:
        raise dash.exceptions.PreventUpdate

    config = json.loads(config_data)
    investment_accounts: dict[str, str] = (
        config.get("users", {}).get(uid, {}).get("investment_accounts") or {}
    )

    hist = reconstruct_portfolio_history(uid, investment_accounts)

    _empty_layout = dict(
        **_CHART_LAYOUT,
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
    )

    if hist.empty:
        fig = go.Figure()
        fig.update_layout(
            **_empty_layout,
            annotations=[{
                "text": "No investment data. Upload a Vanguard CSV to get started.",
                "xref": "paper", "yref": "paper",
                "x": 0.5, "y": 0.5, "showarrow": False,
                "font": {"color": "#888"},
            }],
        )
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["retirement_value"],
        name="Retirement",
        stackgroup="one",
        fillcolor=_C_RETIREMENT,
        line=dict(color=_C_RETIREMENT, width=0),
        mode="lines",
        hovertemplate="$%{y:,.0f}<extra>Retirement</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["taxable_value"],
        name="Non-Retirement",
        stackgroup="one",
        fillcolor=_C_TAXABLE,
        line=dict(color=_C_TAXABLE, width=0),
        mode="lines",
        hovertemplate="$%{y:,.0f}<extra>Non-Retirement</extra>",
    ))
    fig.update_layout(
        **_CHART_LAYOUT,
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#eee", tickprefix="$", tickformat=",.0f"),
    )
    return fig


@callback(
    Output("investments-transactions-grid", "rowData"),
    Output("investments-txn-type-filter", "options"),
    Output("investments-txn-account-filter", "options"),
    Input("config-store", "data"),
    Input("investments-data-version", "data"),
    Input("investments-txn-type-filter", "value"),
    Input("investments-txn-account-filter", "value"),
)
def update_transactions(config_data, _version, type_filter, account_filter):
    if not config_data or not isinstance(config_data, str):
        raise dash.exceptions.PreventUpdate

    uid = session.get("user_id")
    if not uid:
        raise dash.exceptions.PreventUpdate

    transactions = fetch_investment_transactions(uid)

    all_types = sorted({t.get("transaction_type", "") for t in transactions if t.get("transaction_type")})
    all_accounts = sorted({t.get("account_number", "") for t in transactions if t.get("account_number")})
    type_options = [{"label": v, "value": v} for v in all_types]
    account_options = [{"label": f"****{v}", "value": v} for v in all_accounts]

    filtered = transactions
    if type_filter:
        filtered = [t for t in filtered if t.get("transaction_type") in type_filter]
    if account_filter:
        filtered = [t for t in filtered if t.get("account_number") in account_filter]

    rows = filtered[:500] if filtered else _TXN_PLACEHOLDER
    return rows, type_options, account_options


# ── Account modal callbacks ───────────────────────────────────────────────────

@callback(
    Output("accounts-for-modal", "data", allow_duplicate=True),
    Input("investments-new-accounts", "data"),
    prevent_initial_call=True,
)
def open_modal_on_new_accounts(new_accounts):
    if not new_accounts:
        raise dash.exceptions.PreventUpdate
    return {"accounts": new_accounts, "mode": "new"}


@callback(
    Output("accounts-for-modal", "data", allow_duplicate=True),
    Input("edit-accounts-btn", "n_clicks"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def open_modal_on_edit(n_clicks, config_data):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    uid = session.get("user_id")
    if not uid:
        raise dash.exceptions.PreventUpdate

    holdings = fetch_latest_holdings(uid)
    accounts = sorted({h.get("account_number", "") for h in holdings if h.get("account_number")})
    if not accounts:
        raise dash.exceptions.PreventUpdate
    return {"accounts": accounts, "mode": "edit"}


@callback(
    Output("account-modal-body", "children"),
    Output("account-modal", "is_open"),
    Input("accounts-for-modal", "data"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def render_modal_body(modal_data, config_data):
    if not modal_data:
        raise dash.exceptions.PreventUpdate

    accounts: list[str] = modal_data.get("accounts", [])
    mode: str = modal_data.get("mode", "new")

    existing_types: dict[str, str] = {}
    existing_nicknames: dict[str, str] = {}
    if config_data:
        try:
            cfg = json.loads(config_data)
            uid = session.get("user_id") or ""
            user_cfg = cfg.get("users", {}).get(uid, {})
            existing_types = user_cfg.get("investment_accounts") or {}
            existing_nicknames = user_cfg.get("investment_account_nicknames") or {}
        except Exception:
            pass

    type_options = [{"label": t, "value": t} for t in _ACCOUNT_TYPES]
    header_msg = (
        "New accounts found. Please assign a type and optional nickname to each."
        if mode == "new"
        else "Edit account types and nicknames."
    )

    rows = [
        html.P(header_msg, className="text-muted small mb-3"),
        dbc.Row([
            dbc.Col(html.Strong("Account"), width=2),
            dbc.Col(html.Strong("Nickname"), width=4),
            dbc.Col(html.Strong("Type"), width=4),
        ], className="mb-2"),
    ]

    for acct in accounts:
        rows.append(
            dbc.Row([
                dbc.Col(
                    html.Span(f"****{acct}", className="font-monospace"),
                    width=2,
                    className="d-flex align-items-center",
                ),
                dbc.Col(
                    dbc.Input(
                        id={"type": "acct-nickname", "index": acct},
                        placeholder="e.g. Work 403b",
                        value=existing_nicknames.get(acct, ""),
                        size="sm",
                    ),
                    width=4,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id={"type": "acct-type", "index": acct},
                        options=type_options,
                        value=existing_types.get(acct),
                        placeholder="Select type…",
                        clearable=False,
                    ),
                    width=4,
                ),
            ], className="mb-2 align-items-center"),
        )

    return rows, True


@callback(
    Output("account-modal", "is_open", allow_duplicate=True),
    Output("config-store", "data", allow_duplicate=True),
    Input("account-modal-save", "n_clicks"),
    State({"type": "acct-type", "index": ALL}, "value"),
    State({"type": "acct-type", "index": ALL}, "id"),
    State({"type": "acct-nickname", "index": ALL}, "value"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def save_account_config(n_clicks, type_values, type_ids, nickname_values, config_data):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    uid = session.get("user_id")
    if not uid:
        raise dash.exceptions.PreventUpdate

    types: dict[str, str] = {}
    nicknames: dict[str, str] = {}
    for id_obj, acct_type, nickname in zip(type_ids, type_values, nickname_values):
        acct = id_obj["index"]
        if acct_type:
            types[acct] = acct_type
        if nickname:
            nicknames[acct] = nickname.strip()

    # Merge with existing config to preserve other accounts
    existing_config: dict = {}
    if config_data:
        try:
            existing_config = json.loads(config_data)
        except Exception:
            pass

    existing_types = existing_config.get("users", {}).get(uid, {}).get("investment_accounts") or {}
    existing_nicknames = existing_config.get("users", {}).get(uid, {}).get("investment_account_nicknames") or {}
    merged_types = {**existing_types, **types}
    merged_nicknames = {**existing_nicknames, **nicknames}

    save_investment_account_config(uid, merged_types, merged_nicknames)

    # Update config-store in-memory
    if "users" not in existing_config:
        existing_config["users"] = {}
    if uid not in existing_config["users"]:
        existing_config["users"][uid] = {}
    existing_config["users"][uid]["investment_accounts"] = merged_types
    existing_config["users"][uid]["investment_account_nicknames"] = merged_nicknames

    return False, json.dumps(existing_config)


@callback(
    Output("account-modal", "is_open", allow_duplicate=True),
    Input("account-modal-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_account_modal(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return False


# ── Delete-data callbacks ─────────────────────────────────────────────────────

@callback(
    Output("delete-data-modal", "is_open", allow_duplicate=True),
    Input("delete-data-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_delete_modal(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return True


@callback(
    Output("delete-data-modal", "is_open", allow_duplicate=True),
    Input("delete-data-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_delete_modal(n_clicks):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return False


@callback(
    Output("delete-data-modal", "is_open", allow_duplicate=True),
    Output("investments-data-version", "data", allow_duplicate=True),
    Output("config-store", "data", allow_duplicate=True),
    Input("delete-data-confirm", "n_clicks"),
    State("investments-data-version", "data"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def confirm_delete_data(n_clicks, current_version, config_data):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    uid = session.get("user_id")
    if not uid:
        raise dash.exceptions.PreventUpdate

    delete_investment_data(uid)

    # Clear investment config fields from the in-memory store so the UI matches.
    cfg = {}
    if config_data:
        try:
            cfg = json.loads(config_data)
        except Exception:
            cfg = {}
    user_cfg = cfg.get("users", {}).get(uid)
    if isinstance(user_cfg, dict):
        user_cfg.pop("investment_accounts", None)
        user_cfg.pop("investment_account_nicknames", None)

    return False, (current_version or 0) + 1, json.dumps(cfg)
