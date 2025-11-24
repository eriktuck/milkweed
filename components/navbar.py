from dash import Input, Output, State, callback
import dash_bootstrap_components as dbc

LOGO = 'https://images.vexels.com/media/users/3/126959/isolated/preview/0000ff7cdd7b42113596a64b737403e1-3d-hand-drawn-dollar-sign-by-vexels.png'

navbar = dbc.NavbarSimple(
    id="navbar",
    children=[
        dbc.NavItem(dbc.NavLink("Actual", href="/dash/")),
        dbc.NavItem(dbc.NavLink("Budget", href="/dash/budget")),
        dbc.NavItem(dbc.NavLink("CSP", href="/dash/csp")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Trends", href="/dash/trends"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Milkweed",
    brand_href="/dash/",
    color="primary",
    dark=True,
)


@callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    """
    Toggles collapsed navbar element.

    Parameters
    ----------
    n: int
        Number of clicks

    is_open: bool
        Status of the element as opened or closed

    Returns
    -------
    bool
        Toggles element open or closed
    """
    if n:
        return not is_open
    return is_open