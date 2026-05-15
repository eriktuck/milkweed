import dash
from dash import html, dcc, callback, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from datetime import datetime as dt
from flask import session
import pandas as pd
from io import StringIO
from dateutil.relativedelta import relativedelta

from core.models.session import SessionData
from core.services.firebase import (
    fetch_all_transactions
)
from core.utils import functions

dash.register_page(__name__, path='/')

FILE_NAME = 'transformed-transactions.pkl'

### UI COMPONENTS ###
date_picker = dcc.DatePickerRange(
                id='date-picker-range',
                start_date='2025-01-01',
                min_date_allowed='2020-01-01',
                end_date=dt.today().strftime('%Y-%m-%d'),
                number_of_months_shown=2,
                persistence=True,
                updatemode='bothdates',
                style={'borderWidth': 0}  
                )


back_year = dbc.Button(
    "<<",
    id='back-year',
    color="light", 
    className="me-1"
)

forward_year = dbc.Button(
    ">>",
    id='forward-year',
    color="light", 
    className="me-1"
)


uploader = dcc.Upload(
    id='upload-transactions',
    children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files')
    ]),
    style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'background-color': 'white'
    },
    # Allow multiple files to be uploaded
    multiple=False
)

### LAYOUT ###
layout = html.Div([
    dbc.Container(
        [
            # First Row: Header and Controls
            dbc.Row(
                [
                    # Header Column
                    dbc.Col(
                        html.H1('Planned vs Actual'),
                        width=6
                    ),
                    # Date Picker and Year Controls Column
                    dbc.Col(
                        dbc.Row(
                            [
                                dbc.Col(back_year, width="auto"),
                                dbc.Col(forward_year, width="auto"),
                                dbc.Col(date_picker, width="auto"),
                            ],
                            justify="end",
                            align="center",
                            className="g-2"  # Adds small spacing between elements
                        ),
                        width=6,  # Adjust width as needed
                        className="text-end"
                    ),
                ],
                className="pb-3 pt-3"
            ),

            # Second Row: Graph
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Loading(
                                id="budget-chart-loading",
                                type="circle",
                                children=html.Div(
                                    id='budget-chart-container',
                                    children=html.P("Loading transactions...")
                                )
                            )
                        ),
                        className="pt-3"
                    )
                )
            ),

            # Third Row: Transaction Table
            dbc.Row(
                dbc.Col(
                    dbc.Container(id='transaction_table', className="pt-3")
                )
            )
        ]
    )
])

### CALLBACKS ###
@callback(
    Output('transaction-data-store', 'data'),
    Input('config-store', 'data')
)
def upload_transactions(config_json):
    """
    Store user + household transactions in browser memory at page load.

    Parameters
    ----------
    config_json : str
        JSON-serialized configuration file for the user (already loaded separately).

    Returns
    -------
    str
        JSON-serialized Pandas DataFrame containing all transactions.
    """
    uid = session.get("user_id")
    if not uid:
        raise ValueError("Error: User not found")
    
    config = SessionData.from_json(config_json)
    all_txn = fetch_all_transactions(uid, config)
    
    return all_txn.to_json(date_format="iso", orient="split")


@callback(
    Output("date-picker-range", "start_date"),
    Output("date-picker-range", "end_date"),
    Input("back-year", "n_clicks"),
    Input("forward-year", "n_clicks"),
    State("date-picker-range", "start_date"),
    State("date-picker-range", "end_date"),
    State("date-picker-range", "min_date_allowed"),
    State("date-picker-range", "max_date_allowed"),
)
def adjust_date_range(back_clicks, forward_clicks, start_date, end_date, 
                      min_date_allowed, max_date_allowed):
    """
	Updates date range of the date picker.
	
	User can select dates directly or use back and forward buttons to 
    jump by month (if single month selected) or year (if more than one
    month selected).
	
	Parameters
	----------
    back_clicks: int
        Number of clicks of back element
    forward_clicks: int
        Number of clicks of forward element
    start_date: str
        Start date selected (or current selection)
    end_date: str
        End date selected (or current end date)
    min_date_allowed: str
        Min date allowed on date picker element, if set
	max_date_allowed: str
        Max date allowed on date picker element, if set
	Returns
	-------
	Str
	    New start date
    Str
        New end date
	"""
    # Determine which button was clicked
    trigger = ctx.triggered_id
    if not trigger:
        raise PreventUpdate
    
    # Convert dates from strings to datetime objects
    start_date = dt.fromisoformat(start_date)
    end_date = dt.fromisoformat(end_date)
    min_date_allowed = dt.fromisoformat(min_date_allowed) if min_date_allowed else None
    max_date_allowed = dt.fromisoformat(max_date_allowed) if max_date_allowed else None

    def get_last_day_of_month(date):
        """Return the last day of the month for the given date."""
        next_month = date + relativedelta(months=1)
        return next_month.replace(day=1) - relativedelta(days=1)
    
    # Delta is one month if same month currently selected, else one year
    if (start_date.year == end_date.year) & (start_date.month == end_date.month):
        time_delta = relativedelta(months=1)
    else:
        time_delta = relativedelta(years=1)

    # Handle the back-year button
    if trigger == "back-year":
        # Prevent update if the current start_date is equal to or earlier than the min_date_allowed
        if min_date_allowed and start_date <= min_date_allowed:
            raise PreventUpdate
        # Calculate new dates
        new_start_date = start_date - time_delta
        new_end_date = end_date - time_delta

    # Handle the forward-year button
    elif trigger == "forward-year":
        # Prevent update if the current end_date is equal to or later than the max_date_allowed
        if max_date_allowed and end_date >= max_date_allowed:
            raise PreventUpdate
        # Calculate new dates
        new_start_date = start_date + time_delta
        new_end_date = end_date + time_delta

    else:
        raise PreventUpdate

    # Adjust the end_date to the last day of the new month if in the same month
    if new_start_date.year == new_end_date.year and new_start_date.month == new_end_date.month:
        new_end_date = get_last_day_of_month(new_end_date)

    # Clamp to min_date_allowed and max_date_allowed
    if min_date_allowed:
        new_start_date = max(new_start_date, min_date_allowed)
        new_end_date = max(new_end_date, min_date_allowed)
    if max_date_allowed:
        new_start_date = min(new_start_date, max_date_allowed)
        new_end_date = min(new_end_date, max_date_allowed)

    # Convert dates back to strings for the DatePicker component
    return new_start_date.isoformat(), new_end_date.isoformat()


@callback(
    Output('budget-chart-container', 'children'),
    [Input('transaction-data-store', 'data'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date'),
     Input('use-case', 'value')],
    [State('config-store', 'data')])
def update_plot(transactions_data, start_date, end_date, user, config_json):
    """
	Create or update budget chart.
	
	Parameters
	----------
    transaction_data: str
        JSON-serialized transaction data
    start_date: str
        Start date from date picker
    end_date: str
        End date from date picker
    config: str
        JSON-serialized configuration file for the user
    user: str
        User name from select filter
	
	Returns
	-------
	Plotly.Figure
	    Budget chart
    None
        Resets the clickData property of the budget chart
	"""
    if not user or not transactions_data or not config_json:
        raise PreventUpdate

    # Read config
    config = SessionData.from_json(config_json)
    
    # Parse dates from calendar
    start_date = dt.fromisoformat(start_date)
    end_date = dt.fromisoformat(end_date)
    
    # Read budget
    budget = functions.read_budget(config.data, user)
    
    # Read transactions
    transactions = pd.read_json(StringIO(transactions_data), orient='split')
    
    # Create budget report
    budget_report = functions.build_budget_report(
        transactions, budget, start_date, end_date, config.data, user)
    
    if budget_report['amount'].abs().sum() == 0:
        return html.P("No transactions found.")
    
    # Update chart
    fig = functions.plot_report(budget_report, start_date, end_date)

    return dcc.Graph(
        id='budget-chart',
        figure=fig,
        config={'displayModeBar': False}
    )


@callback(
    Output('budget-chart', 'clickData'),
    Input('budget-chart', 'figure'),
    prevent_initial_call=True
)
def clear_clickdata_on_update(fig):
    return None

@callback(
    Output('transaction_table', 'children'),
    [Input('budget-chart', 'clickData')],
    [State('transaction-data-store', 'data'),
     State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date'),
     State('use-case', 'value')]
)
def update_table(clickData, transactions_data, start_date, end_date,
                 user):
    """
	Show transactions table for clicked transaction category.
	
	Parameters
	----------
    clickData: str
        name of the category clicked in the budget chart
    transaction_data: str
        JSON-serialized transaction data
    start_date: str
        Start date from date picker
    end_date: str
        End date from date picker
	
	Returns
	-------
	dashBootstrapComponents.Table
	    Prettified table of transactions corresponding to clicked 
        category
	"""
    if clickData is None:
        return html.Div()
    else:
        category = clickData['points'][0]['y']
        transactions = pd.read_json(StringIO(transactions_data), orient='split')
        
        if category == 'Total Spending':
            filt = (
                (transactions['date'] >= start_date) &
                (transactions['date'] <= end_date) &
                (transactions['account_owner'] == user) &
                (transactions['csp_label'] != 'income')
            )
        elif category == 'Total Income':
            filt = (
                (transactions['date'] >= start_date) &
                (transactions['date'] <= end_date) &
                (transactions['account_owner'] == user) &
                (transactions['csp_label'] == 'income')
            )
        else:
            filt = (
                (transactions['date'] >= start_date) &
                (transactions['date'] <= end_date) & 
                (transactions['account_owner'] == user) &
                (transactions['csp'] == category)
            )
        transactions = transactions.loc[filt]
        
        # format table
        table_style = {
            "color": "darkgrey",  # Set text color to dark grey
            "font-family": "inherit",  # Use the default font of the Dash app
        }
                
        pretty_transactions = functions.format_table(transactions)
        transaction_table = dbc.Table.from_dataframe(pretty_transactions,
                                                     striped=True,
                                                     bordered=True,
                                                     hover=True,
                                                     responsive=True,
                                                     style=table_style)
        
        return transaction_table