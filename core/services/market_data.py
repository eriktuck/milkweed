"""Cached market-data adapters (yfinance + cpi) and the returns-payload composition.

Moved from core/services/returns.py (Phase 6 of the core/ restructuring). The
two ``functools.lru_cache`` fetchers are verbatim — module-level caches, same
mechanics, so the Dash hot path never re-hits the network. The pure return math
they feed lives in ``core/domain/returns.py`` and receives market data as
arguments (``inflation: pd.Series``, ``returns_for: Callable``); this module is
adapter #1 (live yfinance/cpi), test fixtures are adapter #2.

``cpi`` and ``yfinance`` are imported lazily inside functions so importing this
module stays cheap. This module deliberately does NOT import
``core/models/portfolio.py``, which performs ``cpi.get()`` at import time.

The heavy entry point is :func:`build_returns_payload`, which returns a
JSON-serialisable dict for a ``dcc.Store``. Call it OFF the recompute hot path.
"""

from __future__ import annotations

import functools

import pandas as pd

import core.domain.returns as returns

INFLATION_START_YEAR = 1994


# ── Inflation ────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def inflation_series() -> pd.Series:
    """Annual US CPI inflation rate indexed by year, e.g. ``{1995: 0.028, ...}``.

    Built from the ``cpi`` package: the year-over-year pct change of annual CPI
    from ``INFLATION_START_YEAR`` to the last available year. ``cpi`` is imported
    lazily. On any failure (offline, cpi data not downloaded), returns an empty
    Series — callers treat empty as "inflation unavailable" and fall back to
    nominal returns. Never calls ``cpi.update()``.
    """
    try:
        import cpi  # lazy: avoids slow import-time work elsewhere
    except Exception:
        return pd.Series(dtype="float64")

    values: dict[int, float] = {}
    for year in range(INFLATION_START_YEAR, 2100):
        try:
            v = cpi.get(year)
        except Exception:
            break  # past the last year with data
        if v is None:
            break
        values[year] = v

    if len(values) < 2:
        return pd.Series(dtype="float64")

    return pd.Series(values).sort_index().pct_change().dropna()


# ── Per-ticker returns ───────────────────────────────────────────────────────

@functools.lru_cache(maxsize=256)
def nominal_annual_returns(ticker: str) -> pd.Series:
    """Annual nominal total return for ``ticker``, indexed by calendar year.

    ``yf.Ticker(t).history(period="100y", interval="1mo")["Close"]`` resampled to
    year-end, ``pct_change`` of consecutive year-end closes. Returns an empty
    Series for unknown tickers or on any yfinance error. Cached per ticker.
    """
    if not ticker:
        return pd.Series(dtype="float64")
    try:
        import yfinance as yf  # lazy

        # auto_adjust=True (explicit, not relying on yfinance's default): "Close" is
        # the dividend- & split-adjusted TOTAL return. Critical for the bond leg, whose
        # return is almost entirely coupon income — price-only would understate it badly.
        hist = yf.Ticker(ticker).history(
            period="100y", interval="1mo", auto_adjust=True)["Close"]
        if hist.empty:
            return pd.Series(dtype="float64")
        annual = hist.resample("YE").last().pct_change().dropna()
        annual.index = annual.index.year
        return annual.astype("float64")
    except Exception:
        return pd.Series(dtype="float64")


# ── Orchestration ────────────────────────────────────────────────────────────

def build_returns_payload(holdings: list[dict]) -> dict:
    """Compute everything the Coast 'returns' section needs, in one shot.

    This is the only network-bound entry point (cached yfinance + cpi reads
    composed with the pure math in ``core.domain.returns``); call it OFF the
    recompute hot path. Returns a JSON-serialisable dict for a ``dcc.Store``.
    The MCMC fan itself is NOT computed here (it depends on the user's start
    value/contributions/horizon) — only ``real_return_pool`` is provided so the
    fan callback can run the fast numpy bootstrap.
    """
    if not holdings:
        return {
            "ok": False,
            "message": "Upload holdings on the Investments page to see return analytics.",
            "inflation": {},
            "inflation_available": False,
            "class_real_returns": {},
            "weighted_avg": {},
            "real_return_pool": [],
            "suggested_real_return_pct": 0.0,
            "n_years": 0,
            "glide": None,
            "equity_pool": [],
            "fixed_pool": [],
            "years": [],
        }

    inflation = inflation_series()
    inflation_available = not inflation.empty

    class_series = returns.asset_class_real_returns(
        holdings, inflation, nominal_annual_returns)
    weights = returns.portfolio_weights(holdings)
    weighted_avg = returns.weighted_average_real_returns(class_series, weights)

    if weighted_avg.empty and not class_series:
        return {
            "ok": False,
            "message": "No historical return data available for these holdings.",
            "inflation": {int(k): float(v) for k, v in inflation.items()},
            "inflation_available": inflation_available,
            "class_real_returns": {},
            "weighted_avg": {},
            "real_return_pool": [],
            "suggested_real_return_pct": 0.0,
            "n_years": 0,
            "glide": None,
            "equity_pool": [],
            "fixed_pool": [],
            "years": [],
        }

    return {
        "ok": True,
        "message": "" if inflation_available else "Inflation data unavailable — returns shown are nominal.",
        "inflation": {int(k): float(v) for k, v in inflation.items()},
        "inflation_available": inflation_available,
        "class_real_returns": {
            cls: {int(y): float(r) for y, r in s.items()} for cls, s in class_series.items()
        },
        "class_labels": returns.class_effective_tickers(
            holdings, inflation, nominal_annual_returns),
        "weighted_avg": {int(y): float(r) for y, r in weighted_avg.items()},
        "real_return_pool": [float(x) for x in weighted_avg.to_numpy()],
        "suggested_real_return_pct": round(
            returns.suggested_real_return(weighted_avg) * 100.0, 2),
        "n_years": int(len(weighted_avg)),
        "glide": returns.glide_return_params(holdings, inflation, nominal_annual_returns),
        **returns.glide_return_pools(holdings, inflation, nominal_annual_returns),
    }
