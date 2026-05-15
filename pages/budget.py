import dash
from dash import html, dcc, callback, Input, Output, State, ctx, Patch, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import json
import pandas as pd
import numpy as np
import calendar
from io import StringIO

from core.utils import functions
from core.services.firebase import save_budget_to_firestore

dash.register_page(__name__, path='/budget')

CSP_GROUPS = ['Income', 'Fixed Costs', 'Investments', 'Savings', 'Guilt Free']
HEADER_ROWS = CSP_GROUPS + ['Total']
NON_BUDGETABLE = {'Total Income', 'Total Expenses', 'Total Spending', 'Unbudgeted'}

### UI COMPONENTS ###
year_dropdown = dcc.Dropdown(
    id='budget-year',
    clearable=False
)

remaining_to_budget = dbc.Button(
    id='assign-gf',
    size="md",
    color='bg-info',
    class_name='p-3',
    disabled=False,
)

grid = dag.AgGrid(
    id="my-grid",
    className="ag-theme-quartz",
    defaultColDef={"editable": True, "sortable": False},
    style={"width": "100%", "height": "100%"},
    getRowId="params.data.id",
    dashGridOptions={
        "domLayout": "normal",
        'undoRedoCellEditing': True,  # does not work with total row since update_total replaces rowData, even within a transaction
        'undoRedoCellEditingLimit': 12,
        'suppressMaintainUnsortedOrder': True,  # performance boost on editing 
    }
    # rowClassRules = {"bg-info": f"{header_rows}.includes(params.data.category)"},  # To use theme default
)

save_budget = dbc.Button(
    "Save Budget",
    id='save-budget',
    size="md",
    color="primary",
    class_name="me-md-2",
    disabled=False,
)

add_new_button = dbc.Button(
    "Add New",
    id="open-new-budget-modal",
    size="md",
    color="success",
)

new_budget_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Add New Budget")),
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label("Budget Year"),
                    dbc.Input(
                        id="new-budget-year",
                        type="number",
                        min=2000,
                        max=2100,
                        step=1,
                    ),
                ], width=3),
            ], className="mb-4"),
            html.Div(id="new-budget-warning", className="mb-3"),
            html.Div(
                id="new-budget-category-table",
                style={"maxHeight": "55vh", "overflowY": "auto"},
            ),
        ]),
        dbc.ModalFooter(
            dbc.Button(
                "Preview",
                id="preview-new-budget",
                color="primary",
            ),
        ),
    ],
    id="new-budget-modal",
    size="xl",
    is_open=False,
    scrollable=False,
)

### LAYOUT ###
layout = html.Div([
    new_budget_modal,
    dbc.Container([
        dbc.Row(
            [
                dbc.Col(html.H1('Budget'), width=8),
                dbc.Col(year_dropdown, width=2),
                dbc.Col(add_new_button, width=2, className="d-flex align-items-center"),
            ], className="pt-3 pb-3"),
        html.Div(remaining_to_budget, className="d-grid pb-3"),
        html.Div(grid, style={"height": "calc(100vh - 300px)"}),
        html.Div(
            [
                save_budget
            ],
            className="d-grid pt-3 pb-3 d-md-flex justify-content-md-end",
        )
    ])
])


### CALLBACKS ###
@callback(
    [Output("budget-year", "options"),
    Output("budget-year", "value")],
    Input("use-case", "value"),
    [State("budget-year", "value"),
     State('config-store', 'data')]
)
def initialize_budget_year(user, budget_year, config):
    config = json.loads(config)

    budget_dict = config["users"][user]['budget']

    budget_years = [year for year, months in budget_dict.items()]

    options=[
        {'label': str(year), 'value': str(year)}
        for year in budget_years
    ]

    if not budget_year:
        budget_year=str(budget_years[-1]) if budget_years else None

    return options, budget_year


@callback(
    [Output("my-grid", "rowData"),
     Output("my-grid", "columnDefs"),
     Output("my-grid", "getRowStyle")],
    Input("budget-year", "value"),
    State('config-store', 'data'),
    State('use-case', 'value')
)
def populate_budget(year, config, user):    
    year = int(year)
    config = json.loads(config)
    budget = functions.read_budget(config, user)
    
    budget = budget.loc[:, (year, 1):(year, 12)]

    budget.columns = [
        f"{calendar.month_abbr[int(month)]}" 
        for year, month in budget.columns
    ]

    csp_labels = pd.DataFrame.from_dict(config["users"][user]['csp_labels'],
                                        orient='index', 
                                        columns=['csp_label'])

    budget = pd.merge(budget, csp_labels, left_index=True,
                      right_index=True, how='left')

    new_rows = pd.DataFrame(np.nan, index=CSP_GROUPS, columns=budget.columns)
    budget = pd.concat([budget, new_rows])

    budget = functions.order_budget(budget, config, user)
    budget['id'] = budget.index
    month_columns = [col for col in budget.columns if col not in ["category", "csp_label", "id"]]

    # Convert DataFrame to rowData for Dash AG Grid
    row_data = budget.to_dict("records")
    
    columnDefs = [
    {"field": "category", "editable": False},
    ] + [
        {
            "field": col,
            "type": "number",
            "editable": {"function": f"!{HEADER_ROWS}.includes(params.data.category)"},
            "width": 100,
            "minWidth": 50,
            "resizable": True,
            "valueFormatter": {
                "function": "params.value && params.value !== 0 ? '$' + params.value.toLocaleString() : ''"
            },
            'valueParser': {'function': 'Number(params.newValue)'}
        }
        for col in month_columns
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


@callback(
    Output("my-grid", "dashGridOptions"),
    [Input("my-grid", "cellValueChanged"),
    Input("my-grid", "rowData")],
)
def pin_total_row(cell_value_changed, row_data):
    df = pd.DataFrame(row_data).set_index("category")
    month_columns = [col for col in df.columns if col not in ["category", "csp_label", "id"]]

    for col in month_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate total spending
    filt = df['csp_label'] != 'income'
    total_row = df.loc[filt, month_columns].sum().to_dict()

    grid_option_patch = Patch()
    grid_option_patch["pinnedBottomRowData"] = [{"category": "Total", **total_row}]
    return grid_option_patch


@callback(
    [Output("assign-gf", "children"),
     Output("assign-gf", "color"),
     Output("assign-gf", "disabled")],
    Input("my-grid", "cellValueChanged"),
    Input("my-grid", "rowData")
)
def update_total_button(cell_value_changed, row_data):
    df = pd.DataFrame(row_data).set_index('category')
    month_columns = [col for col in df.columns if col not in ["category", "csp_label", "id"]]

    filt = df['csp_label'] != 'income'
    total_spend = df.loc[filt, month_columns].sum().sum()

    total_income = df[df['csp_label'] == 'income'][month_columns].sum().sum()

    total_remaining = total_income - total_spend

    disabled=False
    if total_remaining > 0.12:
        text = f"${total_remaining:,.2f} Remaining! Click to assign to Guilt-Free spending."
        color = "primary"
    elif total_remaining < -0.12:
        text = f"${-total_remaining:,.2f} Over Budget! Click to subtract from Guilt-Free spending."
        color="danger"
    else:
        text = "Well Done! Every penny has a job."
        color="light"
        disabled=True

    return text, color, disabled


@callback(
    Output("my-grid", "rowData", allow_duplicate=True),
    Input("assign-gf", "n_clicks"),
    State("my-grid", "rowData"),
    prevent_initial_call=True
)
def assign_to_guilt_free(n, row_data):
    if n is None:
        return "Not clicked."
    else:
        df = pd.DataFrame(row_data).set_index("category")
        month_columns = [col for col in df.columns if col not in ["category", "csp_label", "id"]]
        
        filt = df['csp_label'] != 'income'
        total_spend = df.loc[filt, month_columns].sum().sum()
       
        total_income = df[df['csp_label'] == 'income'][month_columns].sum().sum()
       
        total_remaining = total_income - total_spend
        monthly_remaining = round(total_remaining / 12, 2)
        
        df.loc['guilt_free', month_columns] += monthly_remaining
        
        updated_row_data = df.reset_index().to_dict("records")

        return updated_row_data


@callback(
    Output('config-store', 'data', allow_duplicate=True),
    Input("save-budget", "n_clicks"),
    [State("my-grid", "rowData"),
     State('config-store', 'data'),
     State("budget-year", "value"),
     State("use-case", "value")],
    prevent_initial_call=True
)
def save_budget(n, row_data, config, budget_year, user):
    if n is None:
        raise PreventUpdate

    existing_config = json.loads(config)
    budget = pd.DataFrame(row_data).set_index('category')
    budget = budget.drop(columns=['csp_label', 'id'], index=CSP_GROUPS)

    month_mapping = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
    }

    budget = budget.reset_index().melt(
        id_vars="category", var_name="month", value_name="value"
    ).set_index('category')

    budget['month'] = budget['month'].map(month_mapping).astype(int)

    grouped = budget.groupby(['month', budget.index])['value'].sum()

    budget_by_month = {}
    for (month, category), value in grouped.items():
        budget_by_month.setdefault(month, {})[category] = value

    existing_config["users"][user]['budget'][budget_year] = budget_by_month

    collection_str = "households" if "members" in existing_config["users"][user] else "users"
    uid = existing_config["users"][user]["uid"]
    save_budget_to_firestore(collection_str, uid, budget_year, budget_by_month)

    return json.dumps(existing_config)


@callback(
    Output("new-budget-modal", "is_open"),
    Input("open-new-budget-modal", "n_clicks"),
    State("new-budget-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_new_budget_modal(n_clicks, is_open):
    return not is_open


@callback(
    Output("new-budget-year", "value"),
    Input("new-budget-modal", "is_open"),
    State("config-store", "data"),
    State("use-case", "value"),
    prevent_initial_call=True,
)
def initialize_new_budget_year(is_open, config, user):
    if not is_open:
        raise PreventUpdate
    config = json.loads(config)
    budget_dict = config["users"][user]['budget']
    years = [int(y) for y in budget_dict.keys()]
    return max(years) + 1 if years else 2025


@callback(
    Output("new-budget-category-table", "children"),
    Input("new-budget-modal", "is_open"),
    State("config-store", "data"),
    State("use-case", "value"),
    prevent_initial_call=True,
)
def populate_category_table(is_open, config, user):
    if not is_open:
        raise PreventUpdate

    config = json.loads(config)
    cat_order = config["users"][user]['cat_order']

    header_style = {
        "backgroundColor": "#4a5568",
        "color": "white",
        "fontWeight": "bold",
        "padding": "6px 12px",
    }

    rows = []
    for category in cat_order:
        if category in CSP_GROUPS:
            rows.append(html.Tr(
                html.Td(category, colSpan=3, style=header_style)
            ))
        elif category in NON_BUDGETABLE:
            continue
        else:
            rows.append(html.Tr([
                html.Td(category, style={"padding": "4px 12px", "verticalAlign": "middle"}),
                html.Td(
                    dbc.Select(
                        id={'type': 'new-budget-method', 'index': category},
                        options=[
                            {'label': 'Copy LY Actuals', 'value': 'copy_actuals'},
                            {'label': 'Copy LY Budget', 'value': 'copy_budget'},
                            {'label': 'Manual', 'value': 'manual'},
                        ],
                        value='copy_actuals',
                        size='sm',
                    ),
                    style={"padding": "4px 12px"},
                ),
                html.Td(
                    dbc.InputGroup([
                        dbc.Input(
                            id={'type': 'new-budget-pct', 'index': category},
                            type='number',
                            value=100,
                            min=0,
                            step=1,
                        ),
                        dbc.InputGroupText('%'),
                    ], size='sm'),
                    style={"padding": "4px 12px", "width": "130px"},
                ),
            ]))

    thead = html.Thead(
        html.Tr([
            html.Th("Category"),
            html.Th("Method"),
            html.Th("% Adj", style={"width": "130px"}),
        ]),
        style={"position": "sticky", "top": 0, "backgroundColor": "white", "zIndex": 1},
    )

    return dbc.Table(
        [thead, html.Tbody(rows)],
        bordered=True,
        hover=True,
        size='sm',
        className="mb-0",
    )


@callback(
    Output("new-budget-warning", "children"),
    Input("new-budget-modal", "is_open"),
    State("config-store", "data"),
    State("use-case", "value"),
    State("transaction-data-store", "data"),
    prevent_initial_call=True,
)
def show_unbudgeted_warning(is_open, config_json, user, transactions_data):
    if not is_open or not transactions_data:
        raise PreventUpdate

    config = json.loads(config_json)
    budget_years = [int(y) for y in config["users"][user]['budget'].keys()]
    ly = max(budget_years) if budget_years else None
    if ly is None:
        raise PreventUpdate

    df = pd.read_json(StringIO(transactions_data), orient='split')
    df['date'] = pd.to_datetime(df['date'])
    filt = (df['date'].dt.year == ly) & (df['account_owner'] == user) & (df['csp'] == 'guilt_free')
    unbudgeted_total = df.loc[filt, 'amount'].sum()

    if unbudgeted_total <= 0:
        raise PreventUpdate

    return dbc.Alert(
        f"${unbudgeted_total:,.0f} in unbudgeted (catch-all) spending in {ly}. "
        "Consider adding line items for recurring spend before previewing.",
        color="warning",
        className="mb-0",
    )


@callback(
    Output("new-budget-modal", "is_open", allow_duplicate=True),
    Output("budget-year", "options", allow_duplicate=True),
    Output("budget-year", "value", allow_duplicate=True),
    Output("config-store", "data", allow_duplicate=True),
    Input("preview-new-budget", "n_clicks"),
    State("new-budget-year", "value"),
    State({'type': 'new-budget-method', 'index': ALL}, 'value'),
    State({'type': 'new-budget-pct', 'index': ALL}, 'value'),
    State("config-store", "data"),
    State("use-case", "value"),
    State("transaction-data-store", "data"),
    State("budget-year", "options"),
    prevent_initial_call=True,
)
def preview_new_budget(n_clicks, new_year, methods, pcts, config_json, user, transactions_data, current_options):
    if n_clicks is None:
        raise PreventUpdate

    config = json.loads(config_json)
    new_year = int(new_year)
    ly = new_year - 1

    cat_order = config["users"][user]['cat_order']
    categories = [c for c in cat_order if c not in CSP_GROUPS and c not in NON_BUDGETABLE]

    method_map = dict(zip(categories, methods))
    pct_map = {cat: (pcts[i] or 100) / 100 for i, cat in enumerate(categories)}

    # LY budget — slice only existing columns to avoid KeyError
    budget = functions.read_budget(config, user)
    ly_cols = [(ly, m) for m in range(1, 13) if (ly, m) in budget.columns]
    ly_budget = budget[ly_cols] if ly_cols else pd.DataFrame(index=budget.index)

    # LY actuals — only parse transactions if any category uses copy_actuals
    ly_actuals = {}
    if 'copy_actuals' in methods and transactions_data:
        df = pd.read_json(StringIO(transactions_data), orient='split')
        df['date'] = pd.to_datetime(df['date'])
        filt = (df['date'].dt.year == ly) & (df['account_owner'] == user)
        ly_actuals = df.loc[filt].groupby('csp')['amount'].sum().abs().to_dict()

    new_budget_months = {m: {} for m in range(1, 13)}

    for category in categories:
        method = method_map.get(category, 'copy_actuals')
        pct = pct_map[category]

        if method == 'manual':
            continue

        # Retrieve LY monthly budget values for this category
        monthly_budgets = []
        for month in range(1, 13):
            col = (ly, month)
            if col in ly_budget.columns and category in ly_budget.index:
                val = ly_budget.loc[category, col]
                val = 0.0 if pd.isna(val) else float(val)
            else:
                val = 0.0
            monthly_budgets.append(val)

        if method == 'copy_budget':
            for month, val in enumerate(monthly_budgets, start=1):
                new_budget_months[month][category] = round(val * pct, 2)

        elif method == 'copy_actuals':
            total_budget = sum(monthly_budgets)
            if total_budget > 0:
                loading = [b / total_budget for b in monthly_budgets]
            else:
                loading = [1 / 12] * 12
            ly_total = float(ly_actuals.get(category, 0.0))
            for i, month in enumerate(range(1, 13)):
                new_budget_months[month][category] = round(ly_total * loading[i] * pct, 2)

    config["users"][user]['budget'][str(new_year)] = {
        str(month): values for month, values in new_budget_months.items()
    }

    existing_values = {opt['value'] for opt in current_options}
    if str(new_year) not in existing_values:
        new_options = current_options + [{'label': str(new_year), 'value': str(new_year)}]
    else:
        new_options = current_options

    return False, new_options, str(new_year), json.dumps(config)
