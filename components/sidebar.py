import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State
import asyncio
import pandas as pd
from datetime import datetime as dt
from flask import session

from core.models.session import SessionData
from core.services.firebase import (
    sync_raw_transactions
)
from core.services.monarch import (
    login_to_monarch,
    fetch_transactions_from_monarch,
    pickle_and_encode,
    decode_and_unpickle
)


### UI COMPONENTS ###
use_case_dropdown = dcc.Dropdown(
    id='use-case',
    placeholder='Select user',
    clearable=False
)

login_button = dbc.Button("Fetch", id="open-modal-button", color="dark", outline=True)

login_form = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Email", html_for="username-input"),
                        dbc.Input(
                            type="email",
                            id="username-input",
                            placeholder="Enter Monarch email",
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Label("Password", html_for="password-input"),
                        dbc.Input(
                            type="password",
                            id="password-input",
                            placeholder="Enter Monarch password",
                        ),
                    ],
                    width=6,
                ),
            ],
            className="g-3",
        ),
        html.Div(id="login-status", style={"marginTop": "10px", "color": "bg-secondary"}),
    ]
)

login_modal = dbc.Modal(
    [
        dbc.ModalHeader("Login to Monarch Money"),
        dbc.ModalBody(
            login_form
        ),
        dbc.ModalFooter(
            [
                dbc.Button("Login", id="login-button", color="primary"),
                dbc.Button("Close", id="close-login-modal-button", color="secondary"),
            ]
        ),
    ],
    id="login-modal",
    is_open=False,
)

transaction_date_picker = dcc.DatePickerRange(
    id='transaction-date-picker',
    start_date=dt.today().strftime('%Y-%m-01'),
    end_date=dt.today().strftime('%Y-%m-%d'),
    max_date_allowed=dt.today().strftime('%Y-%m-%d'),
    number_of_months_shown=2,
    persistence=True,
    updatemode='bothdates',
    style={'borderWidth': 0}  
    )

transaction_form = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    transaction_date_picker, 
                    width="auto",  # Content width
                    className="mx-auto"  # Bootstrap class for centering
                )
            ],
            justify="center",  # Bootstrap utility to center the row contents
        ),
    ]
)

transaction_modal = dbc.Modal(
    [
        dbc.ModalHeader("Fetch Updated Transactions"),
        dbc.ModalBody(
            [html.P("Select a range of dates to fetch updated transactions."),
            transaction_form]
        ),
        dbc.ModalFooter(
            [
                dbc.Button("Fetch", id="fetch-button", color="primary"),
                dbc.Button("Close", id="close-transaction-modal-button", color="secondary"),
            ]
        ),
    ],
    id="transaction-modal",
    is_open=False,
)

### LAYOUT ###
sidebar = html.Div(
    [
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            "Please select a use case",
                            className="text-end",
                        ),
                        width=8,
                    ),
                    dbc.Col(use_case_dropdown, width=3),
                    dbc.Col(
                        dbc.Button("Fetch", id="open-modal-button", color="dark", outline=True),
                        className="text-end",
                        width=1,
                    ),
                ],
                className="align-items-center",
            ),
        ),
        login_modal,
        transaction_modal,
    ],
    className="p-2 bg-light",
)


@callback(
    Output('config-store', 'data'),
    Input('navbar', 'id')  # dummy input to fire on load
)
def store_config(dummy):
    """
    Store combined user and household config in browser memory.

    Returns
    -------
    str
        JSON-serialized config object
    """
    uid = session.get("user_id")
    if not uid:
        raise ValueError("Error: User not found")

    session_cache = SessionData.from_firestore(uid)

    return session_cache.serialize()


@callback(
    Output('use-case', 'options'),
    Output('use-case', 'value'),
    Input('config-store', 'data'),
    prevent_initial_call=True
)
def populate_use_case_dropdown(config_json):
    """
    Populate the use-case dropdown from config.

    Parameters
    ----------
    config_json : str
        JSON string of config from `config-store`.

    Returns
    -------
    list[dict], str
        Dropdown options, default value
    """
    if not config_json:
        raise dash.exceptions.PreventUpdate
    
    session_cache = SessionData.from_json(config_json)
    user_configs = session_cache.get_user_configs()

    if not user_configs:
        print("No user configurations found in this session. Cannot update dropdown.")

    # Create options for each user
    options = [
        {"label": user_config["name"].title(), "value": user_uid}
        for user_uid, user_config in user_configs.items()
    ]

    default_value = options[0]["value"] if options else None

    return options, default_value

@callback(
    [Output("login-modal", "is_open"), 
     Output("transaction-modal", "is_open"),
     Output("login-status", "children"),
     Output('transaction-data-store', 'data', allow_duplicate=True),
     Output('monarch-session-store', 'data')],
    [Input("open-modal-button", "n_clicks"),
     Input("close-login-modal-button", "n_clicks"),
     Input("close-transaction-modal-button", "n_clicks"),
     Input("login-button", "n_clicks"),
     Input("fetch-button", "n_clicks")],
    [State("username-input", "value"), 
     State("password-input", "value"),
     State("transaction-date-picker", "start_date"),
     State("transaction-date-picker", "end_date"),
     State('transaction-data-store', 'data'),
     State('monarch-session-store', 'data'),
     State('config-store', 'data')],
    prevent_initial_call=True,
)
def manage_and_handle_modals(
    open_clicks, close_login_clicks, close_transaction_clicks, 
    login_clicks, fetch_clicks, username, password, start_date, end_date, 
    stored_transaction_data, session_data, config_json
):
    """
    Manages modal states and functionality.
    
    Launches modal on open-modal-button click. If a user session is 
    found in session storage, the transactions modal is shown. 
    If not, the login modal is shown. Successful login
    triggers the transactions modal.

    User selects a date range and fetches transactions. Existing
    transactions are updated by truncating existing transactions in the
    date range selected and appending fetched transactions. Raw 
    transaction data are stored in transaction-data-store.
    """
    ctx = dash.callback_context

    if not ctx.triggered:
        return False, False, "", stored_transaction_data, session_data

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Close the modal
    if triggered_id == "close-login-modal-button":
        return False, False, "", stored_transaction_data, session_data
    if triggered_id == "close-transaction-modal-button":
        return False, False, "", stored_transaction_data, session_data
    
    # Check for existing session and open appropriate modal
    if triggered_id == "open-modal-button":
        if session_data:
            mm = decode_and_unpickle(session_data)
            try:
                # Validate session is still active
                asyncio.run(mm.get_accounts())
                return False, True, "", stored_transaction_data, session_data
            except:
                # Session expired, need to login again
                return True, False, "", stored_transaction_data, None
        else:
            # No session exists, open the login modal
            return True, False, "", stored_transaction_data, None

    # No saved session: login with username and password from login modal
    if triggered_id == "login-button":
        if not username or not password:
            return True, False, "Please enter both username and password.", stored_transaction_data, None
        
        try:
            mm = asyncio.run(login_to_monarch(username, password))
            print("LOGIN SUCCESSFUL")
            return (
                False,  # show_login_modal
                True,  # show_transaction_modal
                "",
                stored_transaction_data,
                pickle_and_encode(mm),
            )
        except Exception as e:
            print(f"Login failed: {str(e)}")
            return (
                True,  # show_login_modal
                False,  # show_transaction_modal
                f"Login failed: {str(e)}",
                stored_transaction_data,
                None,
            )
    
    # Handle the fetch button
    if triggered_id == "fetch-button":
        try:
            if not session_data:
                raise Exception("No valid session found")

            # Fetch transactions for selected dates
            config = SessionData.from_json(config_json)
            mm = decode_and_unpickle(session_data) 
            txn_raw = asyncio.run(
                fetch_transactions_from_monarch(mm, start_date, end_date)
            )

            # Sync with Firestore
            all_transactions = sync_raw_transactions(
                txn_raw, config, start_date, end_date
            )
            
            # Update transaction data store
            transactions_json = all_transactions.to_json(date_format='iso', orient='split')
            return False, False, "", transactions_json, session_data
        
        except Exception as e:
            print(f"Failed to fetch transactions: {str(e)}")
            return False, True, "", stored_transaction_data, session_data

    # Default: Both modals closed
    return False, False, "", stored_transaction_data, session_data
