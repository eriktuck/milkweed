from flask import Flask, request, session, render_template, redirect, url_for
from dotenv import load_dotenv, dotenv_values
import firebase_admin
from firebase_admin import credentials, auth
import os
import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from firebase_admin import firestore

from components.navbar import navbar
from components.sidebar import sidebar


# Load environment variables from Secret Manager content (if available)
secrets_env_content = os.getenv("SECRETS_ENV")
if secrets_env_content:
    env_vars_from_secret = {}
    for line in secrets_env_content.splitlines():
        if "=" in line:
            key, value = line.strip().split("=", 1)
            env_vars_from_secret[key] = value
    firebase_api_key = env_vars_from_secret.get("FIREBASE_API_KEY")
    firebase_auth_domain = env_vars_from_secret.get("FIREBASE_AUTH_DOMAIN")
    firebase_project_id = env_vars_from_secret.get("FIREBASE_PROJECT_ID")
    firebase_app_id = env_vars_from_secret.get("FIREBASE_APP_ID")
else:
    print("Loading secrets from local env-file.")
    load_dotenv(dotenv_path=os.path.join("secrets", "env-file"))
    # Fallback to loading from a local file (for local development) 
    firebase_api_key = os.getenv("FIREBASE_API_KEY")
    firebase_auth_domain = os.getenv("FIREBASE_AUTH_DOMAIN")
    firebase_project_id = os.getenv("FIREBASE_PROJECT_ID")
    firebase_app_id = os.getenv("FIREBASE_APP_ID")

# Init Flask
server = Flask(__name__)
server.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret")

external_stylesheets = [dbc.themes.MINTY]

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
        navbar,
        sidebar,
        dash.page_container,
        dcc.Store(id='config-store', storage_type="session", data={"trigger": True}),
        dcc.Store(id='transaction-data-store'),
        dcc.Store(id='transaction-subset-store'),
        dcc.Store(id='monarch-session-store', storage_type="session")
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

