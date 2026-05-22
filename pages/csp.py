import dash
from dash import html, dcc, callback, Input, Output, State, ctx, Patch
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import json
import pandas as pd
import numpy as np
import os
from io import StringIO
import pytz

import dash_ag_grid as dag
import calendar

from core.utils import functions
from core.services.firebase import save_csp_snapshot_to_firestore

CSP_GROUPS = ['Income', 'Fixed Costs', 'Investments', 'Savings', 'Guilt Free']
HEADER_ROWS = CSP_GROUPS + ['Total']
CSP_DICT = {
    'income': 'Income',
    'fixed': 'Fixed Costs',
    'investments': 'Investments',
    'savings': 'Savings',
    'guilt-free': 'Guilt Free',
}

NET_WORTH_CATEGORIES = ["Assets", "Investments", "Savings", "Debt"]
NET_WORTH_TOTAL = "Total Net Worth"

# Special row IDs for the two joint_contribution rows
JC_FIXED_ID = "jc_fixed"
JC_INCOME_ID = "jc_income"

dash.register_page(__name__, path='/csp')

net_worth_grid = dag.AgGrid(
    id="net-worth-grid",
    className="ag-theme-quartz",
    defaultColDef={"editable": False, "sortable": False},
    style={"width": "100%", "height": "100%"},
    dashGridOptions={"domLayout": "autoHeight"},
)

grid = dag.AgGrid(
    id="csp-grid",
    className="ag-theme-quartz",
    defaultColDef={"editable": True, "sortable": False},
    style={"width": "100%", "height": "100%"},
    getRowId="params.data.id",
    dashGridOptions={
        "domLayout": "normal",
        'undoRedoCellEditing': True,
        'undoRedoCellEditingLimit': 12,
        'suppressMaintainUnsortedOrder': True,
        'tooltipShowDelay': 300,
        },
)


layout = html.Div([
    dcc.Store(id="csp-is-editing", data=False),
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1('Conscious Spending Plan'), width="auto"),
            dbc.Col(
                dbc.Select(
                    id="csp-source",
                    options=[
                        {"label": "From Saved", "value": "saved"},
                        {"label": "From Budget", "value": "budget"},
                        {"label": "From Actuals", "value": "actuals"},
                    ],
                    value="saved",
                    style={"width": "180px"},
                ),
                width="auto",
                className="d-flex align-items-center",
            ),
            dbc.Col(
                dbc.Select(
                    id="csp-snapshot-date",
                    options=[],
                    value=None,
                    style={"width": "160px"},
                ),
                id="csp-snapshot-date-col",
                width="auto",
                className="d-flex align-items-center",
                style={"display": "none"},
            ),
            dbc.Col(
                html.Div([
                    dbc.Button("Edit", id="csp-edit-btn", size="md", color="secondary", class_name="me-2"),
                    dbc.Button("Cancel", id="csp-cancel-btn", size="md", color="light", style={"display": "none"}),
                ]),
                width="auto",
                className="d-flex align-items-center",
            ),
        ], className="pt-3 pb-3 align-items-center"),
        html.H4("Net Worth", className="mb-2"),
        html.Div(net_worth_grid, id="net-worth-grid-container", className="mb-4"),
        html.H4("Spending Plan", className="mb-2"),
        html.Div(grid, id="csp-grid-container", style={"height": "calc(100vh - 200px)"}),
        html.Div(
            [
                dbc.Input(
                    id="csp-save-date",
                    type="date",
                    style={"width": "160px"},
                    className="me-2",
                ),
                dbc.Button(
                    "Save CSP",
                    id="save-csp",
                    size="md",
                    color="primary",
                ),
            ],
            id="save-csp-container",
            className="pt-3 pb-3 d-flex justify-content-end align-items-center",
            style={"display": "none"},
        ),
    ])
])


def _build_user_csp_series(user, monthly_avg_series, csp_labels_config):
    """Build per-user CSP series. Returns (frame, jc_value) where jc_value is the
    joint_contribution monthly amount for this user (extracted before the row is dropped)."""
    csp_labels_df = pd.DataFrame.from_dict(
        csp_labels_config, orient='index', columns=['csp_label']
    )
    frame = pd.merge(monthly_avg_series, csp_labels_df, left_index=True, right_index=True, how='left')

    jc_value = float(frame.loc['joint_contribution', user]) if 'joint_contribution' in frame.index else 0.0

    subtotals = frame.groupby('csp_label')[user].sum()
    subpcts = subtotals.div(subtotals.loc['income'])
    subpcts.index = subpcts.index.map(CSP_DICT)

    frame = pd.concat([frame, subpcts])
    frame = frame.drop(columns=['csp_label'], index=['joint_contribution'], errors='ignore')
    return frame, jc_value


def _process_users(config, users, monthly_avg_fn):
    """Two-pass helper: processes individuals first, then households.

    monthly_avg_fn(user) → pd.Series of monthly amounts.
    Household joint_contribution is overridden with sum of member contributions.
    Returns (frames dict, jc_values dict).
    """
    individual_users = [u for u in users if "members" not in config["users"][u]]
    household_users = [u for u in users if "members" in config["users"][u]]

    frames = {}
    jc_values = {}

    for user in individual_users:
        monthly_avg = monthly_avg_fn(user)
        frame, jc = _build_user_csp_series(user, monthly_avg, config["users"][user]['csp_labels'])
        frames[user] = frame
        jc_values[user] = jc

    for user in household_users:
        monthly_avg = monthly_avg_fn(user)
        member_ids = config["users"][user].get("members", [])
        member_jc = sum(jc_values.get(m, 0) for m in member_ids if m in jc_values)
        if 'joint_contribution' in monthly_avg.index:
            monthly_avg = monthly_avg.copy()
            monthly_avg.loc['joint_contribution'] = member_jc
        frame, jc = _build_user_csp_series(user, monthly_avg, config["users"][user]['csp_labels'])
        frames[user] = frame
        jc_values[user] = jc

    return frames, jc_values


def _csp_from_budget(config, users):
    def monthly_avg_fn(user):
        available_years = [int(y) for y in config["users"][user]['budget'].keys()]
        year = str(max(available_years))
        budget = pd.DataFrame(config["users"][user]['budget'][year])
        s = budget.sum(axis=1).div(12)
        s.name = user
        return s

    frames, jc_values = _process_users(config, users, monthly_avg_fn)
    return pd.concat([frames[u] for u in users], axis=1), jc_values


def _csp_from_actuals(config, users, transactions_json):
    transactions = pd.read_json(StringIO(transactions_json), orient='split')

    today = dt.today()
    end = today.replace(day=1) - relativedelta(days=1)
    start = end.replace(day=1) - relativedelta(months=2)

    utc = pytz.UTC
    start_utc = pd.Timestamp(start.year, start.month, start.day, tzinfo=utc)
    end_utc = pd.Timestamp(end.year, end.month, end.day, 23, 59, 59, tzinfo=utc)

    def monthly_avg_fn(user):
        csp_labels_config = config["users"][user]['csp_labels']
        filt = (
            (transactions["date"] >= start_utc) &
            (transactions["date"] <= end_utc) &
            (transactions["account_owner"] == user)
        )
        user_txns = transactions.loc[filt]
        base = pd.DataFrame.from_dict(csp_labels_config, orient='index', columns=['csp_label'])
        if not user_txns.empty:
            actuals = user_txns.groupby("csp")["amount"].sum().abs() / 3
        else:
            actuals = pd.Series(dtype=float)
        actuals.name = user
        s = base.join(actuals, how='left')[user].fillna(0)
        s.name = user
        return s

    frames, jc_values = _process_users(config, users, monthly_avg_fn)
    return pd.concat([frames[u] for u in users], axis=1), jc_values


def _csp_from_saved(config, users, snapshot_date=None):
    def monthly_avg_fn(user):
        csp_labels_config = config["users"][user]['csp_labels']
        base = pd.DataFrame.from_dict(csp_labels_config, orient='index', columns=['csp_label'])
        csp_plans = config["users"][user].get("csp_plans") or {}
        if snapshot_date and snapshot_date in csp_plans:
            active_plan = csp_plans[snapshot_date]
        else:
            active_plan = functions.get_active_csp_plan(csp_plans)
            # backward compat: fall back to legacy flat csp_plan for in-flight sessions
            if not active_plan:
                active_plan = config["users"][user].get("csp_plan") or {}
        saved = pd.Series(active_plan, dtype=float)
        saved.name = user
        s = base.join(saved, how='left')[user].fillna(0)
        s.name = user
        return s

    frames, jc_values = _process_users(config, users, monthly_avg_fn)
    return pd.concat([frames[u] for u in users], axis=1), jc_values


@callback(
    [Output("csp-grid", "rowData"),
     Output("csp-grid", "columnDefs"),
     Output("csp-grid", "getRowStyle"),
     Output("csp-grid-container", "style"),
     Output("net-worth-grid", "rowData"),
     Output("net-worth-grid", "columnDefs"),
     Output("net-worth-grid", "getRowStyle"),
     Output("net-worth-grid-container", "style")],
    Input('config-store', 'data'),
    Input('csp-source', 'value'),
    Input('csp-is-editing', 'data'),
    Input('csp-snapshot-date', 'value'),
    State('transaction-data-store', 'data'),
)
def populate_csp(config, source, is_editing, snapshot_date, transactions_json):
    config = json.loads(config)
    users = list(config["users"].keys())
    individual_users = [u for u in users if "members" not in config["users"][u]]
    household_users = [u for u in users if "members" in config["users"][u]]

    if source == "actuals":
        if not transactions_json:
            raise PreventUpdate
        csp, jc_values = _csp_from_actuals(config, users, transactions_json)
    elif source == "saved":
        csp, jc_values = _csp_from_saved(config, users, snapshot_date=snapshot_date)
    else:
        csp, jc_values = _csp_from_budget(config, users)

    csp = functions.order_budget(csp, config, users[-1])
    csp['id'] = csp.index
    csp['total'] = csp[users].sum(axis=1)

    csp_labels_df = pd.DataFrame.from_dict(
        config["users"][users[-1]]['csp_labels'], orient='index', columns=['csp_label']
    )
    csp = pd.merge(csp, csp_labels_df, left_on='category', right_index=True, how='left')
    subtotals = csp.groupby('csp_label')['total'].sum()
    subpcts = subtotals.div(subtotals.loc['income'])
    subpcts.index = subpcts.index.map(CSP_DICT)

    csp = csp.set_index('category')
    for index, value in subpcts.items():
        csp.loc[index, 'total'] = value
    csp = csp.reset_index()

    row_data = csp.to_dict("records")

    # Build and insert the two joint_contribution special rows
    hh_jc = sum(jc_values.get(u, 0) for u in individual_users)

    jc_fixed_row = {"category": "joint_contribution", "id": JC_FIXED_ID, "csp_label": "fixed"}
    for u in individual_users:
        jc_fixed_row[u] = jc_values.get(u, 0)
    for u in household_users:
        jc_fixed_row[u] = 0
    jc_fixed_row["total"] = 0
    jc_fixed_row["total_tooltip"] = (
        "joint contribution does not factor into Total Fixed Costs. "
        "It is a transfer from members to cover household shared expenses."
    )

    jc_income_row = {"category": "joint_contribution", "id": JC_INCOME_ID, "csp_label": "income"}
    for u in individual_users:
        jc_income_row[u] = 0
    for u in household_users:
        jc_income_row[u] = hh_jc
    jc_income_row["total"] = 0
    jc_income_row["total_tooltip"] = (
        "joint contribution does not factor into Total Income. "
        "It is a transfer from members to cover household shared expenses."
    )

    fc_idx = next((i for i, r in enumerate(row_data) if r.get("category") == "Fixed Costs"), None)
    if fc_idx is not None:
        # Insert jc_income as last item in Income group (right before Fixed Costs header)
        row_data.insert(fc_idx, jc_income_row)
        # Insert jc_fixed as first item in Fixed Costs group (Fixed Costs header is now at fc_idx+1)
        row_data.insert(fc_idx + 2, jc_fixed_row)

    user_columns = [col for col in csp.columns if col not in ["category", "csp_label", "id"]]

    def col_header(col):
        if col == "total":
            return "Total"
        return config["users"].get(col, {}).get("name", col).title()

    def editable_fn(col):
        if col == "total" or not is_editing:
            return False
        is_household_col = "members" in config["users"].get(col, {})
        if is_household_col:
            # Household: neither jc row is editable
            return {"function": f"!{HEADER_ROWS}.includes(params.data.category) && params.data.id !== '{JC_FIXED_ID}' && params.data.id !== '{JC_INCOME_ID}'"}
        else:
            # Individual: jc_fixed is editable, jc_income is not
            return {"function": f"!{HEADER_ROWS}.includes(params.data.category) && params.data.id !== '{JC_INCOME_ID}'"}

    columnDefs = [
        {"field": "category", "headerName": "Category", "editable": False, "width": 200},
    ] + [
        {
            "field": col,
            "headerName": col_header(col),
            "type": "number",
            "editable": editable_fn(col),
            "width": 210,
            "minWidth": 150,
            "resizable": True,
            "valueFormatter": {
                "function": f"{HEADER_ROWS}.includes(params.data.category) ? (params.value * 100).toFixed(0) + '%' : '$' + params.value.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}})"
            },
            'valueParser': {'function': 'Number(params.newValue)'},
            **({"tooltipField": "total_tooltip"} if col == "total" else {}),
        }
        for col in user_columns
    ] + [
        {"field": "csp_label", "editable": False, "hide": True},
    ]

    getRowStyle = {
        "styleConditions": [
            {
                "condition": f"{HEADER_ROWS}.includes(params.data.category)",
                "style": {"backgroundColor": "#4a5568", "color": "white", "fontWeight": "bold"},
            },
            {
                "condition": "params.rowIndex % 2 === 1",
                "style": {"backgroundColor": "#f4f6f8"},
            },
        ]}

    category_width = 200
    data_col_width = 210
    grid_width = category_width + len(user_columns) * data_col_width
    container_style = {"height": "calc(100vh - 200px)", "width": f"{grid_width}px", "margin": "0 auto"}

    dollar_formatter = {"function": "'$' + params.value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})"}
    nw_data_cols = [u for u in users] + ["total"]

    nw_rows = []
    for cat in NET_WORTH_CATEGORIES:
        row = {"category": cat}
        for u in users:
            saved_nw = config["users"][u].get("net_worth") or {}
            row[u] = saved_nw.get(cat, 0)
        row["total"] = sum(row[u] for u in users)
        nw_rows.append(row)

    total_row = {"category": NET_WORTH_TOTAL}
    for u in users:
        saved_nw = config["users"][u].get("net_worth") or {}
        total_row[u] = (
            saved_nw.get("Assets", 0)
            + saved_nw.get("Investments", 0)
            + saved_nw.get("Savings", 0)
            - saved_nw.get("Debt", 0)
        )
    total_row["total"] = sum(total_row[u] for u in users)
    nw_rows.append(total_row)

    nw_editable_fn = (
        {"function": f"params.data.category !== '{NET_WORTH_TOTAL}'"}
        if is_editing else False
    )

    nw_columnDefs = [
        {"field": "category", "headerName": "Category", "editable": False, "width": 200},
    ] + [
        {
            "field": col,
            "headerName": col_header(col),
            "type": "number",
            "editable": nw_editable_fn if col != "total" else False,
            "width": 210,
            "minWidth": 150,
            "resizable": True,
            "valueFormatter": dollar_formatter,
            "valueParser": {"function": "Number(params.newValue)"},
        }
        for col in nw_data_cols
    ]

    nw_getRowStyle = {
        "styleConditions": [
            {
                "condition": f"params.data.category === '{NET_WORTH_TOTAL}'",
                "style": {"backgroundColor": "#4a5568", "color": "white", "fontWeight": "bold"},
            },
            {
                "condition": "params.rowIndex % 2 === 1",
                "style": {"backgroundColor": "#f4f6f8"},
            },
        ]}

    nw_container_style = {"width": f"{grid_width}px", "margin": "0 auto"}

    return (
        row_data, columnDefs, getRowStyle, container_style,
        nw_rows, nw_columnDefs, nw_getRowStyle, nw_container_style,
    )


@callback(
    Output("csp-is-editing", "data"),
    Input("csp-edit-btn", "n_clicks"),
    Input("csp-cancel-btn", "n_clicks"),
    Input("save-csp", "n_clicks"),
    Input("csp-source", "value"),
    prevent_initial_call=True,
)
def toggle_edit_mode(edit_n, cancel_n, save_n, source):
    if ctx.triggered_id == "csp-edit-btn":
        return True
    return False


@callback(
    Output("csp-edit-btn", "style"),
    Output("csp-cancel-btn", "style"),
    Output("save-csp-container", "style"),
    Input("csp-is-editing", "data"),
)
def update_edit_controls(is_editing):
    if is_editing:
        return {"display": "none"}, {}, {"display": "flex"}
    return {}, {"display": "none"}, {"display": "none"}


@callback(
    Output("csp-snapshot-date", "options"),
    Output("csp-snapshot-date", "value"),
    Output("csp-snapshot-date-col", "style"),
    Input("config-store", "data"),
    Input("csp-source", "value"),
)
def update_snapshot_date_selector(config_json, source):
    if source != "saved":
        return [], None, {"display": "none"}
    config = json.loads(config_json)
    all_dates = set()
    for user_data in config["users"].values():
        all_dates.update((user_data.get("csp_plans") or {}).keys())
    sorted_dates = sorted(all_dates, reverse=True)
    options = [{"label": d, "value": d} for d in sorted_dates]
    value = sorted_dates[0] if sorted_dates else None
    return options, value, {}


@callback(
    Output("csp-save-date", "value"),
    Input("csp-is-editing", "data"),
    prevent_initial_call=True,
)
def initialize_save_date(is_editing):
    if not is_editing:
        raise PreventUpdate
    return dt.today().strftime('%Y-%m-%d')


@callback(
    Output("csp-grid", "rowData", allow_duplicate=True),
    Input("csp-grid", "cellValueChanged"),
    State("csp-grid", "rowData"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def update_csp_total(_, row_data, config_json):
    if not row_data:
        raise PreventUpdate
    config = json.loads(config_json)
    users = list(config["users"].keys())
    individual_users = [u for u in users if "members" not in config["users"][u]]
    household_users = [u for u in users if "members" in config["users"][u]]

    df = pd.DataFrame(row_data)
    non_user_cols = {"category", "csp_label", "id", "total"}
    user_cols = [c for c in df.columns if c not in non_user_cols]
    jc_mask = df["id"].isin([JC_FIXED_ID, JC_INCOME_ID])
    data_rows = ~df["category"].isin(HEADER_ROWS) & ~jc_mask
    df.loc[data_rows, "total"] = (
        df.loc[data_rows, user_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
    )

    # Keep household jc_income in sync with individual jc_fixed values
    jc_fixed_mask = df["id"] == JC_FIXED_ID
    jc_income_mask = df["id"] == JC_INCOME_ID
    if jc_fixed_mask.any() and jc_income_mask.any() and household_users:
        jc_fixed_row = df.loc[jc_fixed_mask].iloc[0]
        ind_vals = pd.to_numeric(
            pd.Series({u: jc_fixed_row.get(u, 0) for u in individual_users if u in df.columns}),
            errors='coerce'
        ).fillna(0)
        hh_jc = ind_vals.sum()
        for hh in household_users:
            if hh in df.columns:
                df.loc[jc_income_mask, hh] = hh_jc

    # jc rows never contribute to Total — it's a transfer, not income/expense
    df.loc[jc_mask, "total"] = 0

    return df.to_dict("records")


@callback(
    Output("net-worth-grid", "rowData", allow_duplicate=True),
    Input("net-worth-grid", "cellValueChanged"),
    State("net-worth-grid", "rowData"),
    prevent_initial_call=True,
)
def update_net_worth_total(_, row_data):
    if not row_data:
        raise PreventUpdate
    df = pd.DataFrame(row_data).set_index("category")
    all_cols = [c for c in df.columns]
    user_cols = [c for c in all_cols if c != "total"]
    df[all_cols] = df[all_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    for cat in NET_WORTH_CATEGORIES:
        if cat in df.index:
            df.loc[cat, "total"] = df.loc[cat, user_cols].sum()

    for col in all_cols:
        assets = df.loc["Assets", col] if "Assets" in df.index else 0
        investments = df.loc["Investments", col] if "Investments" in df.index else 0
        savings_val = df.loc["Savings", col] if "Savings" in df.index else 0
        debt = df.loc["Debt", col] if "Debt" in df.index else 0
        df.loc[NET_WORTH_TOTAL, col] = assets + investments + savings_val - debt

    return df.reset_index().to_dict("records")


@callback(
    Output("config-store", "data", allow_duplicate=True),
    Input("save-csp", "n_clicks"),
    State("csp-grid", "rowData"),
    State("net-worth-grid", "rowData"),
    State("config-store", "data"),
    State("csp-save-date", "value"),
    prevent_initial_call=True,
)
def save_csp(n, csp_row_data, nw_row_data, config_json, save_date):
    if n is None:
        raise PreventUpdate

    config = json.loads(config_json)
    users = list(config["users"].keys())

    csp_df = pd.DataFrame(csp_row_data)
    nw_df = pd.DataFrame(nw_row_data)

    # Exclude group headers and the computed jc_income row (it's derived, not stored)
    save_mask = ~csp_df["category"].isin(HEADER_ROWS) & (csp_df["id"] != JC_INCOME_ID)

    today_iso = save_date or dt.today().strftime('%Y-%m-%d')

    for user in users:
        collection_str = "households" if "members" in config["users"][user] else "users"
        uid = config["users"][user]["uid"]

        csp_plan = (
            csp_df[save_mask]
            .set_index("category")[user]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_dict()
        )
        save_csp_snapshot_to_firestore(collection_str, uid, today_iso, csp_plan)
        csp_plans = config["users"][user].get("csp_plans") or {}
        csp_plans[today_iso] = csp_plan
        config["users"][user]["csp_plans"] = csp_plans

        net_worth = (
            nw_df[nw_df["category"] != NET_WORTH_TOTAL]
            .set_index("category")[user]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_dict()
        )
        save_csp_snapshot_to_firestore(collection_str, uid, "net_worth", net_worth)
        config["users"][user]["net_worth"] = net_worth

    return json.dumps(config)
