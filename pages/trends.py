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

_GROUP_DISPLAY = {
    'income': 'Income',
    'fixed': 'Fixed Expenses',
    'investments': 'Investments',
    'shrinking': 'Shrinking',
    'guilt-free': 'Guilt-free Spending',
}
_GROUP_ORDER = ['income', 'fixed', 'investments', 'shrinking', 'guilt-free']

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
        ], className='pb-2', align='center'),
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    'Filter categories',
                    id='trends-filter-btn',
                    color='light',
                    size='sm',
                    class_name='border text-muted',
                ),
                dbc.Collapse(
                    dbc.Card(
                        dbc.CardBody([
                            dbc.Row(
                                dbc.Col(
                                    dbc.ButtonGroup([
                                        dbc.Button(
                                            'Select all',
                                            id='trends-filter-select-all',
                                            color='link',
                                            size='sm',
                                            class_name='text-muted p-0 me-2',
                                        ),
                                        dbc.Button(
                                            'Deselect all',
                                            id='trends-filter-deselect-all',
                                            color='link',
                                            size='sm',
                                            class_name='text-muted p-0',
                                        ),
                                    ]),
                                    class_name='d-flex justify-content-end',
                                ),
                                class_name='mb-1',
                            ),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(
                                        label,
                                        className='fw-semibold small text-muted text-uppercase mb-1',
                                    ),
                                    dbc.Checklist(
                                        id=f'trends-filter-{group}',
                                        options=[],
                                        value=[],
                                        class_name='small',
                                    ),
                                ], md=True, xs=6)
                                for group, label in _GROUP_DISPLAY.items()
                            ]),
                        ]),
                        class_name='shadow-sm mt-2',
                    ),
                    id='trends-filter-collapse',
                    is_open=False,
                ),
            ]),
        ], className='pb-3'),
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
    Output('trends-filter-collapse', 'is_open'),
    Input('trends-filter-btn', 'n_clicks'),
    State('trends-filter-collapse', 'is_open'),
    prevent_initial_call=True,
)
def toggle_filter_panel(n_clicks, is_open):
    return not is_open


@callback(
    Output('trends-filter-income', 'value', allow_duplicate=True),
    Output('trends-filter-fixed', 'value', allow_duplicate=True),
    Output('trends-filter-investments', 'value', allow_duplicate=True),
    Output('trends-filter-shrinking', 'value', allow_duplicate=True),
    Output('trends-filter-guilt-free', 'value', allow_duplicate=True),
    Input('trends-filter-select-all', 'n_clicks'),
    Input('trends-filter-deselect-all', 'n_clicks'),
    State('trends-filter-income', 'options'),
    State('trends-filter-fixed', 'options'),
    State('trends-filter-investments', 'options'),
    State('trends-filter-shrinking', 'options'),
    State('trends-filter-guilt-free', 'options'),
    prevent_initial_call=True,
)
def bulk_select_filter(_, __, *options_per_group):
    if ctx.triggered_id == 'trends-filter-select-all':
        return tuple([o['value'] for o in opts] for opts in options_per_group)
    return tuple([] for _ in _GROUP_ORDER)


@callback(
    Output('trends-filter-income', 'options'),
    Output('trends-filter-income', 'value'),
    Output('trends-filter-fixed', 'options'),
    Output('trends-filter-fixed', 'value'),
    Output('trends-filter-investments', 'options'),
    Output('trends-filter-investments', 'value'),
    Output('trends-filter-shrinking', 'options'),
    Output('trends-filter-shrinking', 'value'),
    Output('trends-filter-guilt-free', 'options'),
    Output('trends-filter-guilt-free', 'value'),
    Input('transaction-data-store', 'data'),
    Input('use-case', 'value'),
)
def populate_filter_options(transactions_data, use_case):
    if not transactions_data or not use_case:
        return tuple([] for _ in range(10))

    df = pd.read_json(StringIO(transactions_data), orient='split')
    df = df[df['account_owner'] == use_case]

    result = []
    for group in _GROUP_ORDER:
        cats = sorted(df[df['csp_label'] == group]['category_name'].dropna().unique().tolist())
        opts = [{'label': cat, 'value': cat} for cat in cats]
        result.extend([opts, cats])

    return tuple(result)


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
    Input('trends-filter-income', 'value'),
    Input('trends-filter-fixed', 'value'),
    Input('trends-filter-investments', 'value'),
    Input('trends-filter-shrinking', 'value'),
    Input('trends-filter-guilt-free', 'value'),
)
def update_trends_chart(transactions_data, start_date, end_date, smoothing,
                         display, use_case, drilldown,
                         f_income, f_fixed, f_investments, f_shrinking, f_guilt_free):
    if not transactions_data or not use_case:
        raise PreventUpdate

    transactions = pd.read_json(StringIO(transactions_data), orient='split')

    selected = (f_income or []) + (f_fixed or []) + (f_investments or []) + (f_shrinking or []) + (f_guilt_free or [])
    if selected:
        transactions = transactions[transactions['category_name'].isin(selected)]

    fig = functions.plot_spending_trends(
        transactions, use_case, start_date, end_date,
        smoothing, display == 'percent', drilldown,
    )

    back_style = {'display': 'inline-block'} if drilldown else {'display': 'none'}
    return fig, back_style


