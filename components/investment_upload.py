import base64

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback, Input, Output, State
from flask import session

from core.services.investments import (
    detect_vanguard_csv_type,
    process_and_upload_cost_basis_csv,
    process_and_upload_vanguard_csv,
)

_DROPZONE_STYLE = {
    "border": "1px dashed #555",
    "borderRadius": "6px",
    "cursor": "pointer",
}


def _dropzone(id_, label):
    return dcc.Upload(
        id=id_,
        children=html.Div(
            [
                html.I(className="fas fa-file-upload me-2"),
                label,
                " or ",
                html.A("browse", style={"cursor": "pointer"}),
            ],
            className="text-center py-3",
        ),
        style=_DROPZONE_STYLE,
        accept=".csv",
        multiple=False,
    )


def _decode(contents):
    """Return decoded CSV text, or None if it can't be decoded."""
    try:
        _, b64 = contents.split(",", 1)
        return base64.b64decode(b64).decode("utf-8")
    except Exception:
        return None


def _error(msg):
    return dbc.Alert(msg, color="danger", className="py-1 mb-0")


# Exposed as separate pieces so a page can compose them into a labelled card
# with its own instructions. The backing `investments-data-version` store is
# mounted globally in app.py's protected_layout because the Settings portfolio
# uploader bumps it while the Investments page + Settings inv-table consume it
# (refresh on upload) — both pages need it in the layout.

# Portfolio export (holdings + transactions) — drives the Investments page.
portfolio_upload = html.Div(
    [
        _dropzone("investment-csv-upload", "Drag & drop the portfolio CSV"),
        html.Div(id="investment-upload-status", className="mt-2 small"),
    ]
)

# Cost-basis export (unrealized gains/losses) — feeds Retirement's gain %.
cost_basis_upload = html.Div(
    [
        _dropzone("cost-basis-csv-upload", "Drag & drop a cost-basis CSV"),
        html.Div(id="cost-basis-upload-status", className="mt-2 small"),
    ]
)


@callback(
    Output("investment-upload-status", "children"),
    Output("investments-data-version", "data"),
    Input("investment-csv-upload", "contents"),
    State("investment-csv-upload", "filename"),
    State("investments-data-version", "data"),
    prevent_initial_call=True,
)
def handle_investment_upload(contents, filename, current_version):
    uid = session.get("user_id")
    if not uid:
        return _error("Not logged in."), dash.no_update

    if contents is None:
        raise dash.exceptions.PreventUpdate

    csv_text = _decode(contents)
    if csv_text is None:
        return (
            _error("Could not decode the uploaded file. Ensure it is a plain CSV."),
            dash.no_update,
        )

    # Friendly cross-zone guard: a cost-basis export belongs in the cost-basis
    # zone (it carries no holdings/transactions).
    if detect_vanguard_csv_type(csv_text) == "cost_basis":
        return (
            _error("That looks like a cost-basis export — use the cost-basis drop zone."),
            dash.no_update,
        )

    try:
        n_holdings, n_txns = process_and_upload_vanguard_csv(uid, csv_text)
    except ValueError as exc:
        return _error(str(exc)), dash.no_update
    except Exception as exc:
        return _error(f"Upload failed: {exc}"), dash.no_update

    # Bumping the version refreshes the Investments page and the Settings
    # investment-accounts table, where any new accounts are labelled.
    return (
        dbc.Alert(
            f"Uploaded {filename}: {n_holdings} holdings, {n_txns} transactions.",
            color="success",
            className="py-1 mb-0",
        ),
        (current_version or 0) + 1,
    )


@callback(
    Output("cost-basis-upload-status", "children"),
    Input("cost-basis-csv-upload", "contents"),
    State("cost-basis-csv-upload", "filename"),
    prevent_initial_call=True,
)
def handle_cost_basis_upload(contents, filename):
    """Cost-basis export → users/{uid}/investments/cost_basis (Phase 6a).

    Feeds the Retirement page's taxable-gain %. Carries no holdings/transactions
    and introduces no new accounts, so it doesn't touch the data-version refresh
    or the account-categorisation modal.
    """
    uid = session.get("user_id")
    if not uid:
        return _error("Not logged in.")

    if contents is None:
        raise dash.exceptions.PreventUpdate

    csv_text = _decode(contents)
    if csv_text is None:
        return _error("Could not decode the uploaded file. Ensure it is a plain CSV.")

    # Friendly cross-zone guard: a portfolio export belongs in the portfolio zone.
    if detect_vanguard_csv_type(csv_text) == "portfolio":
        return _error("That looks like the portfolio CSV — use the portfolio drop zone.")

    try:
        n_accounts = process_and_upload_cost_basis_csv(uid, csv_text)
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Upload failed: {exc}")

    return dbc.Alert(
        f"Uploaded cost basis for {n_accounts} "
        f"account{'s' if n_accounts != 1 else ''} from {filename}. "
        "The Retirement page's taxable-gain % will use it.",
        color="success",
        className="py-1 mb-0",
    )
