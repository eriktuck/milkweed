import dash
from dash import html, dcc, callback, Input, Output, State, ctx, Patch
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from datetime import datetime as dt
import json
import pandas as pd
import numpy as np
import os
from io import StringIO

import dash_ag_grid as dag
import calendar

from core.utils import functions

CSP_GROUPS = ['Income', 'Fixed Costs', 'Investments', 'Savings', 'Guilt Free']
HEADER_ROWS = CSP_GROUPS + ['Total']

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
        dbc.Row(html.H1('Conscious Spending Plan'), className="pt-3 pb-3"),
        html.Div(grid, style={"height": "calc(100vh - 200px)"}),
    ])
])


@callback(
    [Output("csp-grid", "rowData"),
     Output("csp-grid", "columnDefs"),
     Output("csp-grid", "getRowStyle")],
    Input('config-store', 'data')
)
def populate_csp(config):
    year=2025
    config = json.loads(config)
    users = list(config["users"].keys())

    user_budgets = []
    for user in users:    
        budget = config["users"][user]['budget'][str(year)]
        budget = pd.DataFrame(budget)

        monthly_budget = budget.sum(axis=1).div(12)
        monthly_budget.name = user
        

        csp_labels = pd.DataFrame.from_dict(config["users"][user]['csp_labels'],
                                        orient='index', 
                                        columns=['csp_label'])

        monthly_budget = pd.merge(monthly_budget, csp_labels, left_index=True,
                                  right_index=True, how='left')
        
        subtotals = monthly_budget.groupby('csp_label').sum()
        subpcts = subtotals.div(subtotals.loc['income'])

        csp_dict = {'income': 'Income', 'fixed': 'Fixed Costs', 'investments': 'Investments', 'savings': 'Savings', 'guilt-free': 'Guilt Free'}
        subpcts.index = subpcts.index.map(csp_dict)

        monthly_budget = pd.concat([monthly_budget, subpcts])

        #TODO: rename csp_label and hide in report for easier aggregation on edit
        monthly_budget = monthly_budget.drop(columns=['csp_label'], index=['joint_contribution'])
        
        user_budgets.append(monthly_budget)

    csp = pd.concat(user_budgets, axis=1)     

    csp = functions.order_budget(csp, config, user)
    csp['id'] = csp.index

    csp['total'] = csp[users].sum(axis=1)
    csp_labels = pd.DataFrame.from_dict(config["users"][user]['csp_labels'],
                                        orient='index', 
                                        columns=['csp_label'])
    csp = pd.merge(csp, csp_labels, left_on='category',
                   right_index=True, how='left')
    subtotals = csp.groupby('csp_label')['total'].sum()
    subpcts = subtotals.div(subtotals.loc['income'])

    csp_dict = {'income': 'Income', 'fixed': 'Fixed Costs', 'investments': 'Investments', 'savings': 'Savings', 'guilt-free': 'Guilt Free'}
    subpcts.index = subpcts.index.map(csp_dict)

    csp = csp.set_index('category')

    for index, value in subpcts.items():
        # print(csp.loc[i, 'total'])
        csp.loc[index, 'total'] = value
    
    csp = csp.reset_index()
    
    # Convert DataFrame to rowData for Dash AG Grid
    row_data = csp.to_dict("records")
    user_columns = [col for col in csp.columns if col not in ["category", "csp_label", "id"]]

    columnDefs = [
    {"field": "category", "editable": False},
    ] + [
        {
            "field": col,
            "type": "number",
            "editable": {"function": f"!{HEADER_ROWS}.includes(params.data.category)"},
            "width": 210,
            "minWidth": 150,
            "resizable": True,
            "valueFormatter": {
                "function": f"{HEADER_ROWS}.includes(params.data.category) ? (params.value * 100).toFixed(0) + '%' : '$' + params.value.toFixed(2).toLocaleString()"
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

    return row_data, columnDefs, getRowStyle