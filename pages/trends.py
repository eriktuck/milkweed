import dash
from dash import html, dcc, callback, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from datetime import datetime as dt
import pandas as pd
from io import StringIO

from core.utils import functions

dash.register_page(__name__, path='/trends')

_today = dt.today()
_default_start = f'{_today.year - 3}-01-01'

layout = html.Div([
    dcc.Store(id='trends-drilldown', data=None),
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1('Trends'), width='auto'),
            dbc.Col(
                dbc.RadioItems(
                    id='trends-display',
                    options=[
                        {'label': '$', 'value': 'dollars'},
                        {'label': '% income', 'value': 'percent'},
                    ],
                    value='dollars',
                    inline=True,
                ),
                className='d-flex align-items-end justify-content-end',
            ),
        ], className='pt-3 pb-2'),
        dbc.Row([
            dbc.Col(
                dcc.DatePickerRange(
                    id='trends-date-picker',
                    start_date=_default_start,
                    end_date=_today.strftime('%Y-%m-%d'),
                    min_date_allowed='2020-01-01',
                    number_of_months_shown=2,
                    persistence=True,
                    updatemode='bothdates',
                    style={'borderWidth': 0},
                ),
                width='auto',
            ),
            dbc.Col(
                dbc.RadioItems(
                    id='trends-smoothing',
                    options=[
                        {'label': 'Annual', 'value': 'YE'},
                        {'label': 'Monthly', 'value': 'ME'},
                        {'label': 'Weekly', 'value': 'W'},
                        {'label': 'Daily', 'value': 'D'},
                    ],
                    value='ME',
                    inline=True,
                ),
                className='d-flex align-items-center',
            ),
        ], className='pb-3', align='center'),
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        dbc.Row(
                            dbc.Col(
                                dbc.Button(
                                    '← All Categories',
                                    id='trends-back-btn',
                                    color='secondary',
                                    outline=True,
                                    size='sm',
                                    style={'display': 'none'},
                                ),
                                width='auto',
                            ),
                            className='mb-2',
                        ),
                        dcc.Loading(
                            dcc.Graph(
                                id='trends-chart',
                                config={'displayModeBar': False},
                                style={'cursor': 'pointer'},
                            ),
                            type='circle',
                        ),
                    ]),
                    className='pt-3',
                ),
            ),
        ),
    ]),
])


@callback(
    Output('trends-drilldown', 'data'),
    Input('trends-chart', 'clickData'),
    Input('trends-back-btn', 'n_clicks'),
    Input('use-case', 'value'),
    State('trends-drilldown', 'data'),
    prevent_initial_call=True,
)
def update_drilldown(click_data, _back, _use_case, current_drilldown):
    triggered = ctx.triggered_id
    if triggered in ('trends-back-btn', 'use-case'):
        return None
    if triggered == 'trends-chart' and click_data and current_drilldown is None:
        label = click_data['points'][0].get('customdata')
        if label:
            return label
    return dash.no_update


@callback(
    Output('trends-chart', 'figure'),
    Output('trends-back-btn', 'style'),
    Input('transaction-data-store', 'data'),
    Input('trends-date-picker', 'start_date'),
    Input('trends-date-picker', 'end_date'),
    Input('trends-smoothing', 'value'),
    Input('trends-display', 'value'),
    Input('use-case', 'value'),
    Input('trends-drilldown', 'data'),
)
def update_trends_chart(transactions_data, start_date, end_date, smoothing,
                         display, use_case, drilldown):
    if not transactions_data or not use_case:
        raise PreventUpdate

    transactions = pd.read_json(StringIO(transactions_data), orient='split')

    fig = functions.plot_spending_trends(
        transactions, use_case, start_date, end_date,
        smoothing, display == 'percent', drilldown,
    )

    back_style = {'display': 'inline-block'} if drilldown else {'display': 'none'}
    return fig, back_style


