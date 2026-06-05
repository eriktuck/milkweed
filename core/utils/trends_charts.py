"""Plotly figure builders for the Trends page (Concept C, Phase 4.2).

Pure view layer: each builder takes a frame produced by ``core/services/trends.py``
and returns a ``go.Figure`` in the approved theme — a **white Plotly canvas inside
the dark card** (matching the Investments/Forecast pages). No data access here.

Panels:
    Q1  figure_csp_shares   — stacked CSP buckets + income reference (drill-down)
    Q2  figure_composite    — composite parts stacked + combined total + baseline
    Q3  figure_income       — paychecks/other stacked + total income line
    Q4  figure_composition  — top-N bucket categories + Other (drill-down)
    rail figure_sparkline   — tiny line for a mover row
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from core.services.trends import (
    OTHER_LABEL, COMBINED_LABEL, TOTAL_INCOME_LABEL, DEFAULT_FOOD_DINING_PARTS,
)

# ── Palette (canonical CSP colors; see spec / functions.py) ──────────────────────

CSP_COLORS: dict[str, str] = {
    "fixed": "#F3969A",
    "investments": "#78C2AD",
    "sinking": "#FFCE67",
    "guilt-free": "#6f42c1",
    "income": "#9aa0aa",
}
# Q4: monochrome purple ramp for top-N guilt-free categories, grey for "Other".
GUILTFREE_RAMP: list[str] = ["#6f42c1", "#9a78d6", "#bfa3ea", "#ddccf5", "#ece3f8"]
OTHER_COLOR = "#cbd2dc"
# Q2: each composite part keeps its home-bucket base color (groceries=fixed pink,
# dining=guilt-free purple). Broken-out dining subcategories get shades of the base.
COMPOSITE_BASE_COLORS: dict[str, str] = {"Groceries": "#F3969A", "Dining out": "#6f42c1"}
# Q3: income sources.
INCOME_COLORS: dict[str, str] = {"Paychecks": "#5b7ec9", "Other sources": "#78c2ad"}

TOTAL_LINE = "#374151"      # combined / total overlay line
REFERENCE = "#9aa0aa"       # dotted reference line (income / baseline avg)
DELTA_UP = "#e0697f"        # spending up (bad)
DELTA_DOWN = "#3f9e86"      # spending down (good)

_FALLBACK = ["#5bc0be", "#d972ff", "#f6a14b", "#7fb3f5", "#90be6d"]
_TICK = {"YE": "%Y", "ME": "%b %Y", "W": "%b %d", "D": "%b %d"}


# ── Theme / helpers ──────────────────────────────────────────────────────────────

def _apply_theme(
    fig: go.Figure, *, smoothing: str, as_percent: bool, height: int,
    yaxis_range: tuple | None = None,
) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#444"),
        margin=dict(t=10, b=34, l=16, r=14),
        height=height,
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=11),
        ),
        xaxis=dict(gridcolor="#f1f2f4", tickformat=_TICK.get(smoothing, "%b %Y"), title=None),
        yaxis=dict(
            gridcolor="#e6e8ec",
            zerolinecolor="#e6e8ec",
            tickformat=".0%" if as_percent else "$,.0f",
            title="% of income" if as_percent else None,
            range=list(yaxis_range) if yaxis_range else None,
        ),
    )
    return fig


def empty_figure(height: int = 300, message: str = "No data in range") -> go.Figure:
    """A blank white-canvas figure with a centered note (for empty frames)."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(color="#9aa0aa", size=13),
    )
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white", height=height,
        margin=dict(t=10, b=34, l=16, r=14),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def _stacked_area(
    pivot: pd.DataFrame,
    *,
    colors: dict[str, str] | None,
    as_percent: bool,
    smoothing: str,
    height: int,
    total: pd.Series | None = None,
    total_name: str = "Total",
    reference: pd.Series | None = None,
    reference_name: str = "Reference",
    enable_click: bool = False,
    yaxis_range: tuple | None = None,
) -> go.Figure:
    """Shared stacked-area builder: one filled trace per column, optional total
    line and dotted reference line. `enable_click` attaches each column name as
    customdata so the page can drive drill-down from a click."""
    if pivot is None or pivot.empty:
        return empty_figure(height)

    fig = go.Figure()
    hover = "%{y:.1%}" if as_percent else "$%{y:,.0f}"
    colors = colors or {}
    for i, col in enumerate(pivot.columns):
        color = colors.get(col) or _FALLBACK[i % len(_FALLBACK)]
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[col], name=str(col),
            mode="lines", stackgroup="one",
            line=dict(width=0.5, color=color), fillcolor=color,
            customdata=([col] * len(pivot)) if enable_click else None,
            hovertemplate=f"{col}: {hover}<extra></extra>",
        ))

    if total is not None and not total.empty:
        fig.add_trace(go.Scatter(
            x=total.index, y=total.values, name=total_name, mode="lines",
            line=dict(color=TOTAL_LINE, width=2),
            hovertemplate=f"{total_name}: {hover}<extra></extra>",
        ))

    if reference is not None and not reference.empty:
        fig.add_trace(go.Scatter(
            x=reference.index, y=reference.values, name=reference_name, mode="lines",
            line=dict(color=REFERENCE, width=1.5, dash="dot"),
            hovertemplate=f"{reference_name}: {hover}<extra></extra>",
        ))

    return _apply_theme(
        fig, smoothing=smoothing, as_percent=as_percent, height=height,
        yaxis_range=yaxis_range,
    )


# ── Q1 — CSP buckets as a share of income ────────────────────────────────────────

def figure_csp_shares(
    pivot: pd.DataFrame,
    income: pd.Series,
    *,
    as_percent: bool,
    smoothing: str,
    height: int = 320,
) -> go.Figure:
    """Q1: stacked CSP buckets with an income reference line (drill-down enabled).

    In % mode the reference is a flat 100% line; in $ mode it traces actual
    income. `pivot`/`income` come from ``trends.csp_shares_over_time``.

    Shares always use the selected view's own income. Low-income months are a real
    state worth seeing — they push spending well over 100% of income — so the
    y-axis is *guarded* (capped to a robust range), not hidden: such months clip at
    the top while the income=100% line and on-hover values keep the truth visible.
    """
    if pivot is None or pivot.empty:
        return empty_figure(height)
    yaxis_range = None
    if as_percent:
        reference = pd.Series(1.0, index=pivot.index)
        totals = pivot.clip(lower=0).sum(axis=1)
        # Robust ceiling: ~90th-percentile month, floored at 120% and capped at 300%
        # so a near-zero-income month can't blow the axis to thousands of %.
        cap = min(max(1.2, float(totals.quantile(0.9)) * 1.1), 3.0) if not totals.empty else 1.5
        # Floor the lower bound too: a net-credit month ÷ a near-zero-income month
        # explodes negatively just as the upper bound explodes positively.
        lower = max(-0.5, min(0.0, float(pivot.min().min())))
        yaxis_range = (lower, cap)
    else:
        reference = income.reindex(pivot.index).fillna(0.0) if income is not None else None
    return _stacked_area(
        pivot, colors=CSP_COLORS, as_percent=as_percent, smoothing=smoothing,
        height=height, reference=reference, reference_name="Income",
        enable_click=True, yaxis_range=yaxis_range,
    )


# ── Q2 — Food & Dining composite ─────────────────────────────────────────────────

def figure_composite(
    frame: pd.DataFrame,
    *,
    smoothing: str,
    height: int = 300,
    total_label: str = COMBINED_LABEL,
    baseline: pd.Series | float | None = None,
    parts: dict[str, list[str]] | None = None,
) -> go.Figure:
    """Q2: composite parts stacked + a combined total line + optional baseline avg.

    `frame` is ``trends.composite_over_time`` output (its columns may be whole
    parts and/or broken-out subcategories, + a `total_label` column). Coloring
    keeps each part's home-bucket hue: Groceries pink, dining subcategories as
    shades of guilt-free purple — so the groceries/dining distinction stays clear.
    `baseline` may be a scalar (flat) or a Series. $-only.
    """
    if frame is None or frame.empty:
        return empty_figure(height)
    series = frame.drop(columns=[total_label], errors="ignore")
    total = frame[total_label] if total_label in frame.columns else None
    ref = _as_reference(baseline, frame.index)
    colors = _composite_color_map(
        series.columns, parts or DEFAULT_FOOD_DINING_PARTS, COMPOSITE_BASE_COLORS
    )
    return _stacked_area(
        series, colors=colors, as_percent=False, smoothing=smoothing,
        height=height, total=total, total_name=total_label,
        reference=ref, reference_name="Baseline avg",
    )


def _composite_color_map(columns, parts, base_colors) -> dict[str, str]:
    """Color each column by its part's base hue; shade parts that span >1 column.

    A column belongs to a part if it equals the part name (summed) or is one of
    the part's subcategories (broken out). Single-column parts use the base color;
    multi-column parts get progressively lighter shades of it.
    """
    by_part: dict[str, list[str]] = {p: [] for p in parts}
    for col in columns:
        for part, cats in parts.items():
            if col == part or col in cats:
                by_part[part].append(col)
                break
    mapping: dict[str, str] = {}
    for part, cols in by_part.items():
        base = base_colors.get(part, "#888888")
        if len(cols) <= 1:
            for c in cols:
                mapping[c] = base
        else:
            for c, shade in zip(cols, _shades(base, len(cols))):
                mapping[c] = shade
    return mapping


def _shades(base_hex: str, n: int, lighten: float = 0.62) -> list[str]:
    """`n` colors from `base_hex` (darkest) toward a lighter tint of itself."""
    h = base_hex.lstrip("#")
    br, bg, bb = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    out = []
    for i in range(n):
        f = (i / (n - 1)) * lighten if n > 1 else 0.0
        out.append("#%02x%02x%02x" % (
            round(br + (255 - br) * f),
            round(bg + (255 - bg) * f),
            round(bb + (255 - bb) * f),
        ))
    return out


# ── Q3 — Income over time ────────────────────────────────────────────────────────

def figure_income(
    frame: pd.DataFrame,
    *,
    smoothing: str,
    height: int = 300,
    total_label: str = TOTAL_INCOME_LABEL,
) -> go.Figure:
    """Q3: paychecks/other stacked + a total income line. `frame` from
    ``trends.income_split_over_time``. $-only (income shown as-is)."""
    if frame is None or frame.empty:
        return empty_figure(height)
    parts = frame.drop(columns=[total_label], errors="ignore")
    total = frame[total_label] if total_label in frame.columns else None
    return _stacked_area(
        parts, colors=INCOME_COLORS, as_percent=False, smoothing=smoothing,
        height=height, total=total, total_name=total_label,
    )


# ── Q4 — composition within a bucket (top-N + Other) ─────────────────────────────

def figure_composition(
    frame: pd.DataFrame,
    *,
    smoothing: str,
    height: int = 300,
    as_percent: bool = False,
) -> go.Figure:
    """Q4: stacked top-N categories + an "Other" rollup (drill-down enabled).
    `frame` from ``trends.composition_over_time``; colored as a purple ramp with
    grey "Other"."""
    if frame is None or frame.empty:
        return empty_figure(height)
    return _stacked_area(
        frame, colors=_composition_colors(frame.columns), as_percent=as_percent,
        smoothing=smoothing, height=height, enable_click=True,
    )


def _composition_colors(columns) -> dict[str, str]:
    mapping: dict[str, str] = {}
    ramp_i = 0
    for col in columns:
        if col == OTHER_LABEL:
            mapping[col] = OTHER_COLOR
        else:
            mapping[col] = GUILTFREE_RAMP[ramp_i % len(GUILTFREE_RAMP)]
            ramp_i += 1
    return mapping


# ── Movers rail sparkline ────────────────────────────────────────────────────────

def figure_sparkline(series: pd.Series, *, up: bool, height: int = 26) -> go.Figure:
    """A tiny axis-less line for a mover row, colored by direction."""
    fig = go.Figure()
    if series is not None and not series.empty:
        fig.add_trace(go.Scatter(
            x=list(range(len(series))), y=series.values, mode="lines",
            line=dict(color=DELTA_UP if up else DELTA_DOWN, width=2),
            hoverinfo="skip",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(t=2, b=2, l=2, r=2), showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


# ── internal ─────────────────────────────────────────────────────────────────────

def _as_reference(baseline, index) -> pd.Series | None:
    """Normalize a scalar/Series baseline into a Series aligned to `index`."""
    if baseline is None:
        return None
    if isinstance(baseline, pd.Series):
        return baseline.reindex(index)
    return pd.Series(float(baseline), index=index)
