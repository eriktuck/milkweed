"""THE financial engine — pure domain logic only.

Modules here must never import firebase, yfinance, cpi, flask, or touch disk;
all data is passed in as arguments. I/O adapters live in ``core/services/``,
Plotly figure builders in ``core/charts/``.

This package becomes the curated façade (``build_scenario``, ``simulate_lifetime``,
``Scenario``, config models) as the restructuring lands; until then, import the
submodules directly (``core.domain.glide``, ``core.domain.trends``).
"""
