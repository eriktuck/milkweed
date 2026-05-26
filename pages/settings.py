import asyncio
import json

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output, State, ALL, no_update
from dash.exceptions import PreventUpdate
from flask import session

from core.services.firebase import save_transaction_account_config
from core.services.investments import (
    fetch_latest_holdings,
    save_investment_account_config,
    delete_investment_data,
)
from core.services.monarch import (
    login_to_monarch,
    pickle_and_encode,
    decode_and_unpickle,
)

dash.register_page(__name__, path="/settings")

UNASSIGNED = "__none__"

_ACCOUNT_TYPES = [
    "IRA", "Roth IRA", "401k", "Roth 401k", "403b", "457b",
    "SEP IRA", "SIMPLE IRA", "Brokerage", "HSA", "529", "Other",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_config(config_data):
    if not config_data:
        return {}
    try:
        return json.loads(config_data) if isinstance(config_data, str) else dict(config_data)
    except Exception:
        return {}


def _owner_options(cfg):
    opts = [
        {"label": (c.get("name") or uid).title(), "value": uid}
        for uid, c in cfg.get("users", {}).items()
    ]
    opts.append({"label": "— Unassigned —", "value": UNASSIGNED})
    return opts


def _owner_docs(cfg):
    return {
        uid: ("households" if "members" in c else "users")
        for uid, c in cfg.get("users", {}).items()
    }


def _owner_for_account(cfg, name):
    for uid, c in cfg.get("users", {}).items():
        if name in (c.get("accounts") or []):
            return uid
    return UNASSIGNED


def _acct_setting(cfg, owner_uid, name):
    if owner_uid == UNASSIGNED:
        return True, ""
    s = (cfg.get("users", {}).get(owner_uid, {}).get("transaction_account_settings") or {}).get(name, {})
    return bool(s.get("include", True)), s.get("nickname", "")


def _rows_from_config(cfg):
    rows = []
    for uid, c in cfg.get("users", {}).items():
        for name in (c.get("accounts") or []):
            s = (c.get("transaction_account_settings") or {}).get(name, {})
            rows.append({
                "name": name, "institution": None, "type": None, "balance": None,
                "owner": uid, "include": bool(s.get("include", True)),
                "nickname": s.get("nickname", ""), "stale": False,
            })
    return rows


def _rows_from_monarch(accounts, cfg):
    rows, seen = [], set()
    for a in accounts:
        name = a.get("displayName")
        if not name:
            continue
        owner = _owner_for_account(cfg, name)
        inc, nick = _acct_setting(cfg, owner, name)
        rows.append({
            "name": name,
            "institution": (a.get("institution") or {}).get("name"),
            "type": (a.get("type") or {}).get("display") or (a.get("type") or {}).get("name"),
            "balance": a.get("currentBalance"),
            "owner": owner, "include": inc, "nickname": nick, "stale": False,
        })
        seen.add(name)
    # accounts configured but no longer returned by Monarch → stale
    for uid, c in cfg.get("users", {}).items():
        for name in (c.get("accounts") or []):
            if name in seen:
                continue
            seen.add(name)
            inc, nick = _acct_setting(cfg, uid, name)
            rows.append({
                "name": name, "institution": None, "type": None, "balance": None,
                "owner": uid, "include": inc, "nickname": nick, "stale": True,
            })
    return rows


def _txn_table(rows, owner_opts):
    if not rows:
        return html.P(
            "No accounts yet. Click “Sync from Monarch” to pull your accounts.",
            className="text-muted small mb-0",
        )

    def owner_cell(r):
        return dbc.Select(
            id={"type": "settings-owner", "index": r["name"]},
            options=owner_opts, value=r["owner"], size="sm",
        )

    def include_cell(r):
        return dbc.Switch(id={"type": "settings-include", "index": r["name"]}, value=r["include"])

    def nick_cell(r):
        return dbc.Input(
            id={"type": "settings-nick", "index": r["name"]},
            value=r["nickname"], size="sm", placeholder="optional",
        )

    body = []
    for r in rows:
        if r["stale"]:
            body.append(html.Tr([
                html.Td(r["name"], className="font-monospace small"),
                html.Td("Not found in Monarch — stale config entry", colSpan=3,
                        className="text-danger small"),
                html.Td(owner_cell(r)),
                html.Td(include_cell(r)),
                html.Td(nick_cell(r)),
            ], className="table-warning"))
        else:
            bal = r["balance"]
            bal_txt = f"${bal:,.0f}" if isinstance(bal, (int, float)) else "—"
            body.append(html.Tr([
                html.Td(r["name"], className="font-monospace small"),
                html.Td(r["institution"] or "—"),
                html.Td(r["type"] or "—"),
                html.Td(bal_txt, className="text-end"),
                html.Td(owner_cell(r)),
                html.Td(include_cell(r)),
                html.Td(nick_cell(r)),
            ]))

    header = html.Thead(html.Tr([
        html.Th("Account"), html.Th("Institution"), html.Th("Type"),
        html.Th("Balance", className="text-end"), html.Th("Owner"),
        html.Th("Include"), html.Th("Nickname"),
    ]))
    return dbc.Table([header, html.Tbody(body)], hover=True, responsive=True, className="align-middle")


def _inv_table(uid, cfg):
    holdings = fetch_latest_holdings(uid)
    accts = sorted({h.get("account_number", "") for h in holdings if h.get("account_number")})
    if not accts:
        return html.P(
            "No investment data uploaded yet. Upload a Vanguard CSV on the Investments page.",
            className="text-muted small mb-0",
        )
    user_cfg = cfg.get("users", {}).get(uid, {})
    types = user_cfg.get("investment_accounts") or {}
    nicks = user_cfg.get("investment_account_nicknames") or {}
    type_opts = [{"label": t, "value": t} for t in _ACCOUNT_TYPES]

    body = [
        html.Tr([
            html.Td(f"****{a}", className="font-monospace"),
            html.Td(dbc.Select(
                id={"type": "settings-inv-type", "index": a},
                options=type_opts, value=types.get(a),
                placeholder="Select type…", size="sm",
            )),
            html.Td(dbc.Input(
                id={"type": "settings-inv-nick", "index": a},
                value=nicks.get(a, ""), size="sm", placeholder="e.g. Work 403b",
            )),
        ])
        for a in accts
    ]
    header = html.Thead(html.Tr([html.Th("Account"), html.Th("Type"), html.Th("Nickname")]))
    return dbc.Table([header, html.Tbody(body)], hover=True, className="align-middle")


# ── modals ───────────────────────────────────────────────────────────────────

_login_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Connect to Monarch")),
    dbc.ModalBody([
        dbc.Label("Email", html_for="settings-mm-email"),
        dbc.Input(id="settings-mm-email", type="email", placeholder="Monarch email"),
        dbc.Label("Password", html_for="settings-mm-pass", className="mt-2"),
        dbc.Input(id="settings-mm-pass", type="password", placeholder="Monarch password"),
        html.Div(id="settings-login-status", className="small text-danger mt-2"),
    ]),
    dbc.ModalFooter([
        dbc.Button("Connect", id="settings-mm-login-btn", color="primary"),
        dbc.Button("Cancel", id="settings-mm-cancel", color="secondary"),
    ]),
], id="settings-login-modal", is_open=False)

_delete_modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Delete Investment Data")),
    dbc.ModalBody([
        html.P("This permanently deletes all your investment holdings, "
               "transaction history, and account labels from the database."),
        html.P("This cannot be undone.", className="fw-bold text-danger mb-0"),
    ]),
    dbc.ModalFooter([
        dbc.Button("Cancel", id="settings-delete-cancel", color="secondary"),
        dbc.Button("Delete everything", id="settings-delete-confirm", color="danger"),
    ]),
], id="settings-delete-modal", is_open=False)


# ── layout ───────────────────────────────────────────────────────────────────

layout = html.Div([
    dcc.Store(id="settings-txn-accounts-store", storage_type="memory"),
    _login_modal,
    _delete_modal,
    dbc.Container([
        html.H1("Settings", className="pt-3"),
        html.P(
            "Map your synced accounts to people and manage your investment-account labels.",
            className="text-muted",
        ),

        # ── Transaction Accounts ──
        dbc.Card([
            dbc.CardHeader(dbc.Row([
                dbc.Col([
                    html.H5("Transaction Accounts", className="mb-1"),
                    html.Div(
                        "Assign each account to a person, choose whether to include its "
                        "transactions, and set an optional nickname. Excluded accounts are "
                        "dropped on fetch.",
                        className="text-muted small",
                    ),
                ]),
                dbc.Col(
                    dbc.Button(
                        [html.I(className="fas fa-arrows-rotate me-1"), "Sync from Monarch"],
                        id="settings-sync-btn", color="primary", size="sm",
                    ),
                    width="auto", className="d-flex align-items-center",
                ),
            ], align="center")),
            dbc.CardBody(dcc.Loading(html.Div(id="settings-txn-table"), type="circle")),
            dbc.CardFooter(dbc.Row([
                dbc.Col(html.Div(id="settings-txn-status")),
                dbc.Col(dbc.Button("Save changes", id="settings-txn-save", color="primary", size="sm"),
                        width="auto"),
            ], align="center")),
        ], className="mb-4"),

        # ── Investment Accounts ──
        dbc.Card([
            dbc.CardHeader([
                html.H5("Investment Accounts", className="mb-1"),
                html.Div(
                    "Label each Vanguard account by type and nickname. Drives the retirement "
                    "vs non-retirement split on the Investments page. New accounts are prompted "
                    "automatically after an upload.",
                    className="text-muted small",
                ),
            ]),
            dbc.CardBody(html.Div(id="settings-inv-table")),
            dbc.CardFooter(dbc.Row([
                dbc.Col(html.Div(id="settings-inv-status")),
                dbc.Col(dbc.Button("Save changes", id="settings-inv-save", color="primary", size="sm"),
                        width="auto"),
            ], align="center")),
        ], className="mb-4"),

        # ── Data (destructive) ──
        dbc.Card([
            dbc.CardHeader(html.H5("Data", className="mb-0 text-danger")),
            dbc.CardBody(dbc.Row([
                dbc.Col([
                    html.Div("Delete investment data", className="fw-bold"),
                    html.Div(
                        "Removes all holdings, transaction history, and account labels "
                        "for your account.",
                        className="text-muted small",
                    ),
                    html.Div(id="settings-data-status", className="mt-1"),
                ]),
                dbc.Col(
                    dbc.Button(
                        [html.I(className="fas fa-trash me-1"), "Delete investment data"],
                        id="settings-delete-btn", color="danger", outline=True, size="sm",
                    ),
                    width="auto", className="d-flex align-items-center",
                ),
            ], align="center")),
        ], className="mb-5 border-danger"),
    ]),
])


# ── Transaction-account callbacks ─────────────────────────────────────────────

@callback(
    Output("settings-txn-accounts-store", "data"),
    Output("settings-login-modal", "is_open"),
    Output("settings-login-status", "children"),
    Input("settings-sync-btn", "n_clicks"),
    State("monarch-session-store", "data"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def sync_from_monarch(n_clicks, session_data, config_data):
    """Pull live accounts from Monarch (or prompt login if no valid session)."""
    if not n_clicks:
        raise PreventUpdate
    cfg = _parse_config(config_data)
    if not session_data:
        return no_update, True, ""  # open login modal
    try:
        mm = decode_and_unpickle(session_data)
        result = asyncio.run(mm.get_accounts())
        accounts = result.get("accounts", []) if isinstance(result, dict) else (result or [])
        return {"rows": _rows_from_monarch(accounts, cfg)}, False, ""
    except Exception:
        return no_update, True, "Your Monarch session expired — please reconnect."


@callback(
    Output("monarch-session-store", "data", allow_duplicate=True),
    Output("settings-txn-accounts-store", "data", allow_duplicate=True),
    Output("settings-login-modal", "is_open", allow_duplicate=True),
    Output("settings-login-status", "children", allow_duplicate=True),
    Input("settings-mm-login-btn", "n_clicks"),
    State("settings-mm-email", "value"),
    State("settings-mm-pass", "value"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def settings_monarch_login(n_clicks, email, password, config_data):
    if not n_clicks:
        raise PreventUpdate
    if not email or not password:
        return no_update, no_update, True, "Enter both email and password."
    try:
        mm = asyncio.run(login_to_monarch(email, password))
        result = asyncio.run(mm.get_accounts())
        accounts = result.get("accounts", []) if isinstance(result, dict) else (result or [])
        cfg = _parse_config(config_data)
        return pickle_and_encode(mm), {"rows": _rows_from_monarch(accounts, cfg)}, False, ""
    except Exception as exc:
        return no_update, no_update, True, f"Login failed: {exc}"


@callback(
    Output("settings-login-modal", "is_open", allow_duplicate=True),
    Input("settings-mm-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_settings_login(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


@callback(
    Output("settings-txn-table", "children"),
    Input("settings-txn-accounts-store", "data"),
    Input("config-store", "data"),
)
def render_txn_table(store_data, config_data):
    cfg = _parse_config(config_data)
    rows = (store_data or {}).get("rows") if store_data else None
    if rows is None:
        rows = _rows_from_config(cfg)
    return _txn_table(rows, _owner_options(cfg))


@callback(
    Output("config-store", "data", allow_duplicate=True),
    Output("settings-txn-status", "children"),
    Output("settings-txn-accounts-store", "data", allow_duplicate=True),
    Input("settings-txn-save", "n_clicks"),
    State({"type": "settings-owner", "index": ALL}, "value"),
    State({"type": "settings-owner", "index": ALL}, "id"),
    State({"type": "settings-include", "index": ALL}, "value"),
    State({"type": "settings-include", "index": ALL}, "id"),
    State({"type": "settings-nick", "index": ALL}, "value"),
    State({"type": "settings-nick", "index": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def save_txn_accounts(n_clicks, owner_vals, owner_ids, inc_vals, inc_ids,
                      nick_vals, nick_ids, config_data):
    if not n_clicks:
        raise PreventUpdate
    cfg = _parse_config(config_data)

    inc_by = {i["index"]: v for i, v in zip(inc_ids, inc_vals)}
    nick_by = {i["index"]: v for i, v in zip(nick_ids, nick_vals)}
    assignments = {}
    for i, ov in zip(owner_ids, owner_vals):
        name = i["index"]
        assignments[name] = {
            "owner": None if ov in (None, UNASSIGNED) else ov,
            "include": bool(inc_by.get(name, True)),
            "nickname": (nick_by.get(name) or "").strip(),
        }

    owner_docs = _owner_docs(cfg)
    save_transaction_account_config(owner_docs, assignments)

    # mirror the write into config-store so other pages see it immediately
    for uid, _kind in owner_docs.items():
        accts, setts = [], {}
        for name, a in assignments.items():
            if a["owner"] != uid:
                continue
            accts.append(name)
            entry = {"include": a["include"]}
            if a["nickname"]:
                entry["nickname"] = a["nickname"]
            setts[name] = entry
        cfg["users"][uid]["accounts"] = accts
        cfg["users"][uid]["transaction_account_settings"] = setts

    status = dbc.Alert("Saved.", color="success", className="py-1 mb-0")
    return json.dumps(cfg), status, {"rows": _rows_from_config(cfg)}


# ── Investment-account callbacks ──────────────────────────────────────────────

@callback(
    Output("settings-inv-table", "children"),
    Input("config-store", "data"),
)
def render_inv_table(config_data):
    uid = session.get("user_id")
    if not uid:
        raise PreventUpdate
    return _inv_table(uid, _parse_config(config_data))


@callback(
    Output("config-store", "data", allow_duplicate=True),
    Output("settings-inv-status", "children"),
    Input("settings-inv-save", "n_clicks"),
    State({"type": "settings-inv-type", "index": ALL}, "value"),
    State({"type": "settings-inv-type", "index": ALL}, "id"),
    State({"type": "settings-inv-nick", "index": ALL}, "value"),
    State({"type": "settings-inv-nick", "index": ALL}, "id"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def save_inv_accounts(n_clicks, type_vals, type_ids, nick_vals, nick_ids, config_data):
    if not n_clicks:
        raise PreventUpdate
    uid = session.get("user_id")
    if not uid:
        raise PreventUpdate
    cfg = _parse_config(config_data)

    types = {i["index"]: v for i, v in zip(type_ids, type_vals) if v}
    nicks = {i["index"]: v.strip() for i, v in zip(nick_ids, nick_vals) if v and v.strip()}

    save_investment_account_config(uid, types, nicks)

    cfg.setdefault("users", {}).setdefault(uid, {})
    cfg["users"][uid]["investment_accounts"] = types
    cfg["users"][uid]["investment_account_nicknames"] = nicks
    return json.dumps(cfg), dbc.Alert("Saved.", color="success", className="py-1 mb-0")


# ── Data (delete) callbacks ───────────────────────────────────────────────────

@callback(
    Output("settings-delete-modal", "is_open"),
    Input("settings-delete-btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_delete_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return True


@callback(
    Output("settings-delete-modal", "is_open", allow_duplicate=True),
    Input("settings-delete-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_delete_modal(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return False


@callback(
    Output("settings-delete-modal", "is_open", allow_duplicate=True),
    Output("config-store", "data", allow_duplicate=True),
    Output("settings-data-status", "children"),
    Input("settings-delete-confirm", "n_clicks"),
    State("config-store", "data"),
    prevent_initial_call=True,
)
def confirm_delete_data(n_clicks, config_data):
    if not n_clicks:
        raise PreventUpdate
    uid = session.get("user_id")
    if not uid:
        raise PreventUpdate
    delete_investment_data(uid)

    cfg = _parse_config(config_data)
    user_cfg = cfg.get("users", {}).get(uid)
    if isinstance(user_cfg, dict):
        user_cfg.pop("investment_accounts", None)
        user_cfg.pop("investment_account_nicknames", None)

    return False, json.dumps(cfg), dbc.Alert(
        "Investment data deleted.", color="success", className="py-1 mb-0",
    )
