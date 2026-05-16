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
        },
    # rowClassRules = {"bg-info": f"{header_rows}.includes(params.data.category)"},  # To use theme default
)


layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1('Conscious Spending Plan'), width="auto"),
            dbc.Col(
                dbc.Select(
                    id="csp-source",
                    options=[
                        {"label": "From Budget", "value": "budget"},
                        {"label": "Editable", "value": "editable"},
                        {"label": "From Actuals", "value": "actuals"},
                    ],
                    value="budget",
                    style={"width": "180px"},
                ),
                width="auto",
                className="d-flex align-items-center",
            ),
        ], className="pt-3 pb-3 align-items-center"),
        html.H4("Net Worth", className="mb-2"),
        html.Div(net_worth_grid, id="net-worth-grid-container", className="mb-4"),
        html.H4("Spending Plan", className="mb-2"),
        html.Div(grid, id="csp-grid-container", style={"height": "calc(100vh - 200px)"}),
        html.Div(
            dbc.Button(
                "Save CSP",
                id="save-csp",
                size="md",
                color="primary",
                class_name="me-md-2",
            ),
            id="save-csp-container",
            className="d-grid pt-3 pb-3 d-md-flex justify-content-md-end",
            style={"display": "none"},
        ),
    ])
])


def _build_user_csp_series(user, monthly_avg_series, csp_labels_config):
    """Attach csp_label, compute group % subtotals, drop joint_contribution."""
    csp_labels_df = pd.DataFrame.from_dict(
        csp_labels_config, orient='index', columns=['csp_label']
    )
    frame = pd.merge(monthly_avg_series, csp_labels_df, left_index=True, right_index=True, how='left')

    subtotals = frame.groupby('csp_label')[user].sum()
    subpcts = subtotals.div(subtotals.loc['income'])
    subpcts.index = subpcts.index.map(CSP_DICT)

    frame = pd.concat([frame, subpcts])
    frame = frame.drop(columns=['csp_label'], index=['joint_contribution'], errors='ignore')
    return frame


def _csp_from_budget(config, users):
    year = 2025
    user_frames = []
    for user in users:
        budget = pd.DataFrame(config["users"][user]['budget'][str(year)])
        monthly_avg = budget.sum(axis=1).div(12)
        monthly_avg.name = user
        frame = _build_user_csp_series(user, monthly_avg, config["users"][user]['csp_labels'])
        user_frames.append(frame)
    return pd.concat(user_frames, axis=1)


def _csp_from_actuals(config, users, transactions_json):
    transactions = pd.read_json(StringIO(transactions_json), orient='split')

    # Last 3 complete calendar months
    today = dt.today()
    end = today.replace(day=1) - relativedelta(days=1)      # last day of last month
    start = end.replace(day=1) - relativedelta(months=2)    # first day, 3 months ago

    utc = pytz.UTC
    start_utc = pd.Timestamp(start.year, start.month, start.day, tzinfo=utc)
    end_utc = pd.Timestamp(end.year, end.month, end.day, 23, 59, 59, tzinfo=utc)

    user_frames = []
    for user in users:
        csp_labels_config = config["users"][user]['csp_labels']

        filt = (
            (transactions["date"] >= start_utc) &
            (transactions["date"] <= end_utc) &
            (transactions["account_owner"] == user)
        )
        user_txns = transactions.loc[filt]

        # Start from the full configured csp category list so every row appears
        base = pd.DataFrame.from_dict(csp_labels_config, orient='index', columns=['csp_label'])

        if not user_txns.empty:
            actuals = user_txns.groupby("csp")["amount"].sum().abs() / 3
        else:
            actuals = pd.Series(dtype=float)
        actuals.name = user

        monthly_avg = base.join(actuals, how='left')[user].fillna(0)
        monthly_avg.name = user

        frame = _build_user_csp_series(user, monthly_avg, csp_labels_config)
        user_frames.append(frame)
    return pd.concat(user_frames, axis=1)


def _csp_from_editable(config, users):
    user_frames = []
    for user in users:
        csp_labels_config = config["users"][user]['csp_labels']
        base = pd.DataFrame.from_dict(csp_labels_config, orient='index', columns=['csp_label'])

        saved = pd.Series(config["users"][user].get("csp_plan") or {}, dtype=float)
        saved.name = user

        monthly_avg = base.join(saved, how='left')[user].fillna(0)
        monthly_avg.name = user

        frame = _build_user_csp_series(user, monthly_avg, csp_labels_config)
        user_frames.append(frame)
    return pd.concat(user_frames, axis=1)


@callback(
    [Output("csp-grid", "rowData"),
     Output("csp-grid", "columnDefs"),
     Output("csp-grid", "getRowStyle"),
     Output("csp-grid-container", "style"),
     Output("net-worth-grid", "rowData"),
     Output("net-worth-grid", "columnDefs"),
     Output("net-worth-grid", "getRowStyle"),
     Output("net-worth-grid-container", "style"),
     Output("save-csp-container", "style")],
    Input('config-store', 'data'),
    Input('csp-source', 'value'),
    State('transaction-data-store', 'data'),
)
def populate_csp(config, source, transactions_json):
    config = json.loads(config)
    users = list(config["users"].keys())

    if source == "actuals":
        if not transactions_json:
            raise PreventUpdate
        csp = _csp_from_actuals(config, users, transactions_json)
    elif source == "editable":
        csp = _csp_from_editable(config, users)
    else:
        csp = _csp_from_budget(config, users)

    # Order rows, add id and total column
    csp = functions.order_budget(csp, config, users[-1])
    csp['id'] = csp.index
    csp['total'] = csp[users].sum(axis=1)

    # Compute total column group % subtotals
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

    # Build grid outputs
    row_data = csp.to_dict("records")
    user_columns = [col for col in csp.columns if col not in ["category", "csp_label", "id"]]

    def col_header(col):
        if col == "total":
            return "Total"
        return config["users"].get(col, {}).get("name", col).title()

    columnDefs = [
        {"field": "category", "headerName": "Category", "editable": False, "width": 200},
    ] + [
        {
            "field": col,
            "headerName": col_header(col),
            "type": "number",
            "editable": False if col == "total" else {"function": f"!{HEADER_ROWS}.includes(params.data.category)"},
            "width": 210,
            "minWidth": 150,
            "resizable": True,
            "valueFormatter": {
                "function": f"{HEADER_ROWS}.includes(params.data.category) ? (params.value * 100).toFixed(0) + '%' : '$' + params.value.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}})"
            },
            'valueParser': {'function': 'Number(params.newValue)'}
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

    # Build net worth grid outputs
    dollar_formatter = {"function": "'$' + params.value.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})"}
    nw_data_cols = [u for u in users] + ["total"]

    nw_rows = []
    for cat in NET_WORTH_CATEGORIES:
        row = {"category": cat}
        if source == "editable":
            for u in users:
                saved_nw = config["users"][u].get("net_worth") or {}
                row[u] = saved_nw.get(cat, 0)
            row["total"] = sum(row[u] for u in users)
        else:
            row.update({col: 0 for col in nw_data_cols})
        nw_rows.append(row)

    # Total Net Worth = Assets + Investments + Savings - Debt
    total_row = {"category": NET_WORTH_TOTAL}
    for u in users:
        if source == "editable":
            saved_nw = config["users"][u].get("net_worth") or {}
            total_row[u] = (
                saved_nw.get("Assets", 0)
                + saved_nw.get("Investments", 0)
                + saved_nw.get("Savings", 0)
                - saved_nw.get("Debt", 0)
            )
        else:
            total_row[u] = 0
    total_row["total"] = sum(total_row[u] for u in users)
    nw_rows.append(total_row)

    nw_editable_fn = (
        {"function": f"params.data.category !== '{NET_WORTH_TOTAL}'"}
        if source == "editable"
        else False
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
    save_container_style = (
        {"display": "flex"} if source == "editable" else {"display": "none"}
    )

    return (
        row_data, columnDefs, getRowStyle, container_style,
        nw_rows, nw_columnDefs, nw_getRowStyle, nw_container_style,
        save_container_style,
    )


@callback(
    Output("csp-grid", "rowData", allow_duplicate=True),
    Input("csp-grid", "cellValueChanged"),
    State("csp-grid", "rowData"),
    prevent_initial_call=True,
)
def update_csp_total(_, row_data):
    if not row_data:
        raise PreventUpdate
    df = pd.DataFrame(row_data)
    non_user_cols = {"category", "csp_label", "id", "total"}
    user_cols = [c for c in df.columns if c not in non_user_cols]
    data_rows = ~df["category"].isin(HEADER_ROWS)
    df.loc[data_rows, "total"] = (
        df.loc[data_rows, user_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
    )
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

    # Recompute "total" column for each category row as sum of user columns
    for cat in NET_WORTH_CATEGORIES:
        if cat in df.index:
            df.loc[cat, "total"] = df.loc[cat, user_cols].sum()

    # Recompute Total Net Worth row for every column (user cols + total)
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
    prevent_initial_call=True,
)
def save_csp(n, csp_row_data, nw_row_data, config_json):
    if n is None:
        raise PreventUpdate

    config = json.loads(config_json)
    users = list(config["users"].keys())

    csp_df = pd.DataFrame(csp_row_data)
    nw_df = pd.DataFrame(nw_row_data)

    for user in users:
        collection_str = "households" if "members" in config["users"][user] else "users"
        uid = config["users"][user]["uid"]

        # Spending plan: category rows only (no group header rows)
        csp_plan = (
            csp_df[~csp_df["category"].isin(HEADER_ROWS)]
            .set_index("category")[user]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .to_dict()
        )
        save_csp_snapshot_to_firestore(collection_str, uid, "plan", csp_plan)
        config["users"][user]["csp_plan"] = csp_plan

        # Net worth: data rows only (no Total Net Worth row)
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
