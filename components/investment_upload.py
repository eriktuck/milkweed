import base64
import json

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback, Input, Output, State
from flask import session

from core.services.investments import fetch_latest_holdings, process_and_upload_vanguard_csv


investment_upload = html.Div(
    [
        dcc.Store(id="investments-data-version", data=0),
        dcc.Store(id="investments-new-accounts", storage_type="memory"),
        dcc.Upload(
            id="investment-csv-upload",
            children=html.Div(
                [
                    html.I(className="fas fa-file-upload me-2"),
                    "Drag & drop Vanguard CSV or ",
                    html.A("browse", style={"cursor": "pointer"}),
                ],
                className="text-center py-3",
            ),
            style={
                "border": "1px dashed #555",
                "borderRadius": "6px",
                "cursor": "pointer",
            },
            accept=".csv",
            multiple=False,
        ),
        html.Div(id="investment-upload-status", className="mt-2 small"),
    ]
)


@callback(
    Output("investment-upload-status", "children"),
    Output("investments-data-version", "data"),
    Output("investments-new-accounts", "data"),
    Input("investment-csv-upload", "contents"),
    State("investment-csv-upload", "filename"),
    State("investments-data-version", "data"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def handle_investment_upload(contents, filename, current_version, config_data):
    uid = session.get("user_id")
    if not uid:
        return dbc.Alert("Not logged in.", color="danger", className="py-1 mb-0"), dash.no_update, dash.no_update

    if contents is None:
        raise dash.exceptions.PreventUpdate

    try:
        _, b64 = contents.split(",", 1)
        csv_text = base64.b64decode(b64).decode("utf-8")
    except Exception:
        return (
            dbc.Alert(
                "Could not decode the uploaded file. Ensure it is a plain CSV.",
                color="danger",
                className="py-1 mb-0",
            ),
            dash.no_update,
            dash.no_update,
        )

    try:
        n_holdings, n_txns = process_and_upload_vanguard_csv(uid, csv_text)
    except ValueError as exc:
        return dbc.Alert(str(exc), color="danger", className="py-1 mb-0"), dash.no_update, dash.no_update
    except Exception as exc:
        return (
            dbc.Alert(f"Upload failed: {exc}", color="danger", className="py-1 mb-0"),
            dash.no_update,
            dash.no_update,
        )

    # Detect account numbers not yet categorised
    existing_accounts: dict = {}
    if config_data:
        try:
            cfg = json.loads(config_data)
            existing_accounts = cfg.get("users", {}).get(uid, {}).get("investment_accounts") or {}
        except Exception:
            pass

    holdings = fetch_latest_holdings(uid)
    uploaded_accounts = {h.get("account_number", "") for h in holdings if h.get("account_number")}
    new_accounts = sorted(uploaded_accounts - set(existing_accounts.keys()))

    return (
        dbc.Alert(
            f"Uploaded {filename}: {n_holdings} holdings, {n_txns} transactions.",
            color="success",
            className="py-1 mb-0",
        ),
        (current_version or 0) + 1,
        new_accounts if new_accounts else dash.no_update,
    )
