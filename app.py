from flask import Flask, request, session, render_template, redirect, url_for
from firebase_admin import auth
import os
import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

from dotenv import load_dotenv
load_dotenv(os.getenv("ENV_PATH", "secrets/env-file"))

from components.sidebar import sidebar

firebase_api_key = os.getenv("FIREBASE_API_KEY")
firebase_auth_domain = os.getenv("FIREBASE_AUTH_DOMAIN")
firebase_project_id = os.getenv("FIREBASE_PROJECT_ID")
firebase_app_id = os.getenv("FIREBASE_APP_ID")

# Init Flask
server = Flask(__name__)
server.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret")

external_stylesheets = [dbc.themes.MINTY, dbc.icons.FONT_AWESOME]

app = Dash(__name__, 
           server=server,
           use_pages=True,
           suppress_callback_exceptions=True,
           external_stylesheets=external_stylesheets,
           url_base_pathname="/dash/")

def protected_layout():
    if "user_id" not in session:
        return html.Div("Unauthorized. Please log in at /")
    
    return html.Div([
        sidebar,
        html.Div(dash.page_container, className="app-content"),
        dcc.Store(id='config-store', storage_type="session", data={"trigger": True}),
        dcc.Store(id='transaction-data-store'),
        dcc.Store(id='transaction-subset-store'),
        dcc.Store(id='monarch-session-store', storage_type="session"),
        # Bumped by the Settings portfolio uploader (writer) and consumed by the
        # Investments page + Settings inv-table (refresh on upload) — mounted
        # globally so both pages can reach it.
        dcc.Store(id='investments-data-version', data=0),
        # The nest-egg goal handshake: written by the Retirement page (the precise
        # backward-from-spending PV) and consumed by the Forecast page as its goal,
        # replacing Forecast's crude annual_spend/4% estimate. Mounted globally so
        # both pages can reach it; session-scoped so it survives navigation.
        dcc.Store(id='retirement-goal-store', storage_type="session"),
    ])

app.layout = protected_layout

# Routes
@server.route("/")
def index():
    return render_template(
        "index.html",
        firebase_api_key=firebase_api_key,
        firebase_auth_domain=firebase_auth_domain,
        firebase_project_id=firebase_project_id,
        firebase_app_id=firebase_app_id,
    )

@server.route("/login", methods=["POST"])
def login():
    id_token = request.json.get("idToken")
    decoded_token = auth.verify_id_token(id_token)
    session["user_id"] = decoded_token["uid"]
    session["email"] = decoded_token.get("email")
    return "OK", 200

@server.route("/dash/")
def redirect_to_dash():
    if "user_id" not in session:
        return redirect(url_for("index"))
    return app.index()  # renders Dash app layout

@server.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

