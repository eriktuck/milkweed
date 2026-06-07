"""Historical real-return and inflation analytics for the Forecast page.

All network/slow work (yfinance ``.history``, cpi lookups) lives here behind
``functools.lru_cache`` so the Dash hot path never hits the network. This module
deliberately does NOT import ``core/models/portfolio.py``, which performs
``cpi.get()`` at import time. ``cpi`` and ``yfinance`` are imported lazily inside
functions so importing this module stays cheap.

Conventions:
  - "real return" uses the precise geometric form ``(1+nominal)/(1+inflation)-1``.
  - All return series are annual, indexed by integer calendar year.
  - The bootstrap samples WHOLE historical years (return+inflation jointly,
    already baked into each year's real return) to preserve any return/inflation
    correlation.

The heavy entry point is :func:`build_returns_payload`, which returns a
JSON-serialisable dict for a ``dcc.Store``. Call it OFF the recompute hot path.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from core.services.investments import ASSET_CLASS_MAP, MONEY_MARKET_SYMBOLS

# ── Constants ────────────────────────────────────────────────────────────────
FALLBACK_TICKER = "VTSAX"
MIN_YEARS_HISTORY = 10          # tickers with fewer real-return obs aren't trusted standalone

# Representative ticker per asset class, used when a held ticker has too little
# history or no yfinance data. Keeps newer/illiquid assets from being over- or
# under-weighted by their (short) own history.
ASSET_CLASS_PROXY: dict[str, str] = {
    "US Equity": "VTSAX",
    "Intl Equity": "VEU",
    "Bonds": "VBTLX",
    "Cash": "VMFXX",
    "Other": "VTSAX",
}

CASH_REAL_RETURN = 0.0          # cash / money-market modelled at flat ~0% real
INFLATION_START_YEAR = 1994
N_SIMS = 1000
RNG_SEED = 42

_PERCENTILES = (10, 25, 50, 75, 90)


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

        hist = yf.Ticker(ticker).history(period="100y", interval="1mo")["Close"]
        if hist.empty:
            return pd.Series(dtype="float64")
        annual = hist.resample("YE").last().pct_change().dropna()
        annual.index = annual.index.year
        return annual.astype("float64")
    except Exception:
        return pd.Series(dtype="float64")


def real_annual_returns(ticker: str, inflation: pd.Series | None = None) -> pd.Series:
    """Inflation-adjusted annual returns for ``ticker``, indexed by year.

    Aligns nominal returns with ``inflation`` on the common year index and applies
    the precise geometric form ``(1+nominal)/(1+inflation)-1``. If ``inflation`` is
    None/empty, returns the nominal series unchanged. Empty when the ticker has no
    usable history.
    """
    nominal = nominal_annual_returns(ticker)
    if nominal.empty:
        return nominal
    if inflation is None:
        inflation = inflation_series()
    if inflation is None or inflation.empty:
        return nominal
    common = nominal.index.intersection(inflation.index)
    if len(common) == 0:
        return nominal
    return ((1 + nominal.loc[common]) / (1 + inflation.loc[common]) - 1).astype("float64")


# ── Portfolio aggregation ────────────────────────────────────────────────────

def _classify(symbol: str) -> str:
    return ASSET_CLASS_MAP.get(symbol, "Other")


def portfolio_weights(holdings: list[dict]) -> dict[str, float]:
    """Map asset class -> fraction of total portfolio value.

    Classifies each holding's ``symbol`` via ``ASSET_CLASS_MAP`` (default
    "Other"), sums ``total_value`` per class, and normalises. Returns ``{}`` for
    empty/zero-value holdings.
    """
    by_class: dict[str, float] = {}
    for h in holdings:
        value = float(h.get("total_value", 0.0) or 0.0)
        if value <= 0:
            continue
        cls = _classify(h.get("symbol", ""))
        by_class[cls] = by_class.get(cls, 0.0) + value
    total = sum(by_class.values())
    if total <= 0:
        return {}
    return {cls: v / total for cls, v in by_class.items()}


def _ticker_real_returns(
    symbol: str, asset_class: str, inflation: pd.Series
) -> pd.Series:
    """Real-return series for one ticker, with cash + short-history fallbacks."""
    # Cash / money-market: flat real return over the inflation index (per decision).
    if symbol in MONEY_MARKET_SYMBOLS or asset_class == "Cash":
        index = inflation.index if (inflation is not None and not inflation.empty) else range(2000, 2025)
        return pd.Series(CASH_REAL_RETURN, index=pd.Index(index)).astype("float64")

    series = real_annual_returns(symbol, inflation)
    if len(series) >= MIN_YEARS_HISTORY:
        return series

    # Too little (or no) history → use the asset class's representative proxy.
    proxy = ASSET_CLASS_PROXY.get(asset_class, FALLBACK_TICKER)
    return real_annual_returns(proxy, inflation)


def asset_class_real_returns(
    holdings: list[dict], inflation: pd.Series | None = None
) -> dict[str, pd.Series]:
    """Real annual return series per asset class PRESENT in the portfolio.

    For each asset class, the value-weighted average of the real-return series of
    the held tickers in that class (aligned on the common year index). A ticker
    with ``< MIN_YEARS_HISTORY`` observations or no yfinance data is replaced by
    its ``ASSET_CLASS_PROXY``; cash/money-market by a flat ``CASH_REAL_RETURN``
    series. Classes that resolve to no data are omitted.
    """
    if inflation is None:
        inflation = inflation_series()

    # Collect value per (class, symbol) so we can weight within a class.
    holdings_by_class: dict[str, list[tuple[str, float]]] = {}
    for h in holdings:
        value = float(h.get("total_value", 0.0) or 0.0)
        if value <= 0:
            continue
        cls = _classify(h.get("symbol", ""))
        holdings_by_class.setdefault(cls, []).append((h.get("symbol", ""), value))

    result: dict[str, pd.Series] = {}
    for cls, members in holdings_by_class.items():
        cls_total = sum(v for _, v in members)
        if cls_total <= 0:
            continue
        weighted = pd.Series(dtype="float64")
        for symbol, value in members:
            series = _ticker_real_returns(symbol, cls, inflation)
            if series.empty:
                continue
            contrib = series * (value / cls_total)
            weighted = contrib if weighted.empty else weighted.add(contrib, fill_value=0.0)
        if not weighted.empty:
            result[cls] = weighted.sort_index()
    return result


def weighted_average_real_returns(
    class_series: dict[str, pd.Series], weights: dict[str, float]
) -> pd.Series:
    """Portfolio weighted-average real return series indexed by year.

    For each year present across the class series, takes the weighted mean of the
    classes available that year, renormalising the weights over the available
    classes. Returns an empty Series when there is nothing to combine.
    """
    if not class_series:
        return pd.Series(dtype="float64")

    frame = pd.DataFrame(class_series)  # columns = asset classes, index = years
    w = pd.Series({cls: weights.get(cls, 0.0) for cls in frame.columns}, dtype="float64")
    if w.sum() <= 0:
        return pd.Series(dtype="float64")

    # Mask weights to classes present each year, renormalise per row, then dot.
    present = frame.notna()
    row_weights = present.mul(w, axis=1)
    row_totals = row_weights.sum(axis=1)
    row_totals = row_totals.replace(0.0, np.nan)
    normalised = row_weights.div(row_totals, axis=0)
    weighted = (frame.fillna(0.0) * normalised).sum(axis=1)
    return weighted.dropna().sort_index().astype("float64")


# ── Bootstrap simulation ─────────────────────────────────────────────────────

def bootstrap_projection(
    real_returns,
    start_value: float,
    annual_contributions,
    horizon_years: int,
    n_sims: int = N_SIMS,
    seed: int = RNG_SEED,
) -> np.ndarray:
    """Monte Carlo bootstrap of portfolio value over ``horizon_years``.

    Vectorised joint-year bootstrap: draw an ``(n_sims, horizon_years)`` matrix of
    indices into the historical real-return pool (each element is one historical
    year's portfolio real return, so any return/inflation correlation is
    preserved), then compound row-wise adding that year's contribution::

        V[:, t] = V[:, t-1] * (1 + sampled[:, t]) + contributions[t]

    Returns an ``(horizon_years, n_sims)`` matrix of simulated values in real
    (inflation-adjusted) dollars. Returns a zero matrix if the pool is empty.
    """
    pool = np.asarray(real_returns, dtype="float64")
    pool = pool[~np.isnan(pool)]
    horizon_years = int(horizon_years)
    if pool.size == 0 or horizon_years <= 0:
        return np.zeros((max(horizon_years, 0), n_sims))

    contrib = np.asarray(annual_contributions, dtype="float64")
    if contrib.size < horizon_years:
        contrib = np.pad(contrib, (0, horizon_years - contrib.size))
    else:
        contrib = contrib[:horizon_years]

    rng = np.random.default_rng(seed)
    sampled = pool[rng.integers(0, pool.size, size=(n_sims, horizon_years))]

    values = np.empty((n_sims, horizon_years), dtype="float64")
    prev = np.full(n_sims, float(start_value))
    for t in range(horizon_years):
        prev = prev * (1 + sampled[:, t]) + contrib[t]
        values[:, t] = prev
    return values.T  # (horizon_years, n_sims)


def percentile_bands(sims: np.ndarray, percentiles=_PERCENTILES) -> dict[int, list[float]]:
    """Per-year percentile bands across simulations: ``{pct: [value_per_year]}``."""
    if sims.size == 0:
        return {p: [] for p in percentiles}
    return {p: np.percentile(sims, p, axis=1).tolist() for p in percentiles}


def suggested_real_return(weighted_avg: pd.Series) -> float:
    """Median of the per-year weighted-average real returns (as a fraction).

    This is the "suggested" single number written into the Forecast rate input.
    Returns 0.0 for an empty series.
    """
    if weighted_avg is None or len(weighted_avg) == 0:
        return 0.0
    return float(np.median(weighted_avg.to_numpy()))


# ── Orchestration ────────────────────────────────────────────────────────────

def build_returns_payload(holdings: list[dict]) -> dict:
    """Compute everything the Forecast 'returns' section needs, in one shot.

    This is the only network-bound entry point (cached yfinance + cpi reads); call
    it OFF the recompute hot path. Returns a JSON-serialisable dict for a
    ``dcc.Store``. The MCMC fan itself is NOT computed here (it depends on the
    user's start value/contributions/horizon) — only ``real_return_pool`` is
    provided so the fan callback can run the fast numpy bootstrap.
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
        }

    inflation = inflation_series()
    inflation_available = not inflation.empty

    class_series = asset_class_real_returns(holdings, inflation)
    weights = portfolio_weights(holdings)
    weighted_avg = weighted_average_real_returns(class_series, weights)

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
        }

    return {
        "ok": True,
        "message": "" if inflation_available else "Inflation data unavailable — returns shown are nominal.",
        "inflation": {int(k): float(v) for k, v in inflation.items()},
        "inflation_available": inflation_available,
        "class_real_returns": {
            cls: {int(y): float(r) for y, r in s.items()} for cls, s in class_series.items()
        },
        "weighted_avg": {int(y): float(r) for y, r in weighted_avg.items()},
        "real_return_pool": [float(x) for x in weighted_avg.to_numpy()],
        "suggested_real_return_pct": round(suggested_real_return(weighted_avg) * 100.0, 2),
        "n_years": int(len(weighted_avg)),
    }
