"""THE financial engine — pure domain logic only.

Modules here must never import firebase, yfinance, cpi, flask, or touch disk;
all data is passed in as arguments. I/O adapters live in ``core/services/``,
Plotly figure builders in ``core/charts/``.

Curated façade: the app-facing entry points are re-exported here
(``build_scenario``, ``Scenario``, ``simulate_lifetime``, the config models).
Everything else is imported from its submodule directly
(``core.domain.assumptions``, ``core.domain.trends``, ...).
"""

from core.domain.config import HouseholdConfig, RetirementConfig, UserConfig
from core.domain.lifetime import simulate_lifetime
from core.domain.scenario import Scenario, build_scenario

__all__ = [
    "HouseholdConfig",
    "RetirementConfig",
    "Scenario",
    "UserConfig",
    "build_scenario",
    "simulate_lifetime",
]
