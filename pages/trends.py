import dash
from dash import html, dcc, callback, Input, Output, State, ctx, Patch
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_daq as daq
from datetime import datetime as dt
import json
import pandas as pd
import numpy as np
import os
from io import StringIO

import dash_ag_grid as dag
import calendar

from core.utils import functions

dash.register_page(__name__, path='/trends')


pct_or_nom = daq.ToggleSwitch(
    id='as-percent',
    label=['$', '%'],
    value=False
)


layout = html.Div([
    dbc.Container([
            dbc.Row(
                [
                    dbc.Col(html.H1('Trends'), width=10),
                    dbc.Col(
                        pct_or_nom,
                        className='d-flex align-items-end justify-content-end',
                        width=2
                    )
                ],
                class_name="pt-3 pb-3"
            ), 
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Loading(
                                children=dcc.Graph(
                                    id='csp-chart',
                                    config={'displayModeBar': False}),
                                type='circle'
                            )
                        ),
                        className="pt-3"
                    )
                )
            )
    ])
])


@callback(
    Output('csp-chart', 'figure'),
    [Input('transaction-subset-store', 'data'),
     Input('as-percent', 'value'),
     Input('csp-chart', 'clickData')],
     State('csp-chart', 'figure')
)
def update_csp_chart(transactions_data, as_percent, clickData, fig):
    transactions = pd.read_json(StringIO(transactions_data), orient='split')
    if clickData is None:
        fig = functions.plot_csp_by_label(transactions, as_percent)
    else:  #TODO: make a second chart (detail that is expanded on click)
        click = clickData['points'][0]["curveNumber"]
        csp_label_name = fig["data"][click]["name"]
        print(csp_label_name)
        fig = functions.plot_csp_by_label(transactions, as_percent)
    return fig


