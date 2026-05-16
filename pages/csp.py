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

CSP_GROUPS = ['Income', 'Fixed Costs', 'Investments', 'Savings', 'Guilt Free']
HEADER_ROWS = CSP_GROUPS + ['Total']
CSP_DICT = {
    'income': 'Income',
    'fixed': 'Fixed Costs',
    'investments': 'Investments',
    'savings': 'Savings',
    'guilt-free': 'Guilt Free',
}

dash.register_page(__name__, path='/csp')

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
                        {"label": "Editable", "value": "editable", "disabled": True},
                        {"label": "From Actuals", "value": "actuals"},
                    ],
                    value="budget",
                    style={"width": "180px"},
                ),
                width="auto",
                className="d-flex align-items-center",
            ),
        ], className="pt-3 pb-3 align-items-center"),
        html.Div(grid, id="csp-grid-container", style={"height": "calc(100vh - 200px)"}),
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


@callback(
    [Output("csp-grid", "rowData"),
     Output("csp-grid", "columnDefs"),
     Output("csp-grid", "getRowStyle"),
     Output("csp-grid-container", "style")],
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
            "editable": {"function": f"!{HEADER_ROWS}.includes(params.data.category)"},
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

    return row_data, columnDefs, getRowStyle, container_style
