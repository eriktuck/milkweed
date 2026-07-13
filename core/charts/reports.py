"""Figure builders and display formatting for the Actual/CSP pages: the
budget-vs-actual stoplight report, spending-trend area charts, and the
prettified transactions table.

Moved verbatim from core/utils/functions.py (Phase 3 of the core/
restructuring).
"""
from datetime import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz


def plot_report(budget_report, start_date, end_date):
    green = '#78C2AD'
    yellow = '#FFCE67'
    red = '#F3969A'
    body = '#888'
    heading = '#5a5a5a'

    x = abs(budget_report['proportion'])
    y = budget_report['category']
    text = budget_report['amount'] / budget_report['budget']
    hover_text = budget_report['amount']

    def stoplight_system(row):
        prop = (row['amount'] / row['budget']) if row['budget'] else row['amount']
        if row['csp_label'] == 'income':
            return green if prop > 0.99 else yellow if prop >= 0.8 else red
        else:
            return red if prop > 1.01 else yellow if prop >= 0.8 else green

    marker_color = budget_report.apply(stoplight_system, axis=1)

    bar_width = 0.8

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x,
        y=y,
        name='Budget',
        orientation='h',
        marker_color=marker_color,
        hovertext=hover_text,
        # hoverinfo='x',
        hovertemplate='Actual: $%{hovertext:,.2f} <extra></extra>',
        hoverlabel=dict(bgcolor='#888', bordercolor='#888',
                        font=dict(color='white')),
        text=text,
        textposition='auto',
        texttemplate='%{text:.0%}',
        textfont=dict(size=10, color='white'),
        insidetextanchor='start',
        width=bar_width,
        showlegend=False,
        legendgroup='Fixed Costs'
    ))

    fig.update_layout(
        # title=go.layout.Title(text="Planned vs Actual",
        #                       font=dict(color=heading)),
        xaxis=dict(showgrid=False,
                visible=False,
                range=[0,1.45]),
        paper_bgcolor='white',
        plot_bgcolor='white',
        barmode='relative', # in case negative values
        annotations=[
            go.layout.Annotation(
                x=1.31, y=len(y)+.05,
                xanchor='right',
                text='<b>Budget<b>',
                font_color=heading,
                showarrow=False
            ),
            go.layout.Annotation(
                x=1.36, y=len(y)+.05,
                xanchor='left',
                text='<b>Remaining<b>',
                font_color=heading,
                showarrow=False
            )
        ],
        height=max(45, len(budget_report) * 30),
        margin=dict(l=180, t=10, b=10, pad=10),
        yaxis=dict(
            visible=True,
            scaleanchor="x",  # Ensures proportional scaling
        ),
    )

    # Add budget annotation
    for idx, budget in enumerate(budget_report['budget']):
        if budget > 0:
            fig.add_annotation(
                x=1.31, y=idx,
                xanchor='right',
                text=f'$ {budget:,.0f}',
                font_color=body,
                showarrow=False
            )

    # Add remaining annotation
    for idx, (remain, label) in enumerate(zip(budget_report['remaining'],budget_report['csp_label'])):
        if remain >= 0:
            fig.add_annotation(
                x=1.36, y=idx,
                xanchor='left',
                text=f'$ {remain:,.0f}',
                font_color=body,
                showarrow=False
            )
        elif remain < 0:
            fig.add_annotation(
                x=1.36, y=idx,
                xanchor='left',
                text=f'<b>+ ${abs(remain):,.0f}!</b>' if label == 'income' else f'<b>$ ({abs(remain):,.0f})</b>',
                font_color = green if label == 'income' else red,
                showarrow=False
            )


    # Add line for each Group
    for group in ['Income', 'Guilt Free', 'Fixed Costs', 'Investments',
                  'Sinking']:
        fig.add_shape(
            type='line',
            x0=0,
            y0=y.tolist().index(group),
            x1=1.45,
            y1=y.tolist().index(group),
            line=dict(
                width=.5,
                color='grey'
            ),
            layer='below',
            opacity=.6
        )

    # Add line for today
    start_to_today = (dt.today() - start_date).days
    total_days = (end_date - start_date).days
    progress = start_to_today/total_days

    if progress > 0 and progress < 0.1:
        fig.add_shape(
            type='line',
            x0=progress,
            x1=progress,
            y0=0-bar_width/2,
            y1=len(y)-bar_width/2,
            line=dict(color=body, dash='dot')
        )

        fig.add_annotation(
                    x=progress, y=len(y),
                    xanchor='left',
                    text='Today',
                    font_color = body,
                    showarrow=False
                )

    elif progress >= 0.1 and progress <=1:
        fig.add_shape(
            type='line',
            x0=progress,
            x1=progress,
            y0=0-bar_width/2,
            y1=len(y)-bar_width/2,
            line=dict(color=body, dash='dot')
        )

        fig.add_annotation(
                    x=progress, y=len(y),
                    xanchor='right',
                    text='Today',
                    font_color = body,
                    showarrow=False
                )

    else:
        pass

    # Update y axis labels
    fig.update_yaxes(
        tickfont=dict(color=body),
        tickvals=np.arange(len(y)),
        ticktext=[f'<b>{label}</b>' if label in [
            'Total Income', 'Total Spending', 'Income', 'Guilt Free',
            'Fixed Costs', 'Investments', 'Sinking']
                    else label for label in y]
        )

    return fig


def format_table(transactions):
    '''prettifies transactions table for display'''
    transactions_pretty = transactions.copy()
    # format Amount
    transactions_pretty['amount'] = (
        transactions_pretty['amount'].map('${:,.2f}'.format)
    )

    # format Date
    transactions_pretty['date'] = (
        transactions_pretty['date'].dt.strftime('%Y-%m-%d')
    )

    # subset columns
    pretty_cols = [
        'date', 'plaidName','amount', 'category_name', 'account_name', 'notes'
    ]
    transactions_pretty = transactions_pretty.loc[:, pretty_cols]
    transactions_pretty.columns = [
        "Date", "Transaction", "Amount", "Category", "Account", "Notes"
    ]

    transactions_pretty.sort_values('Date', inplace=True, ascending=False)

    return transactions_pretty


def plot_spending_trends(transactions, owner_uid, start_date, end_date,
                          smoothing, as_percent, drilldown_label=None):
    """Plot spending over time, optionally drilled into a CSP label's subcategories."""
    utc = pytz.UTC

    def to_utc(d):
        if isinstance(d, str):
            d = dt.fromisoformat(d)
        return d.replace(tzinfo=utc) if d.tzinfo is None else d.astimezone(utc)

    start_date = to_utc(start_date)
    end_date = to_utc(end_date)

    df = transactions.loc[
        (transactions['account_owner'] == owner_uid) &
        (transactions['date'] >= start_date) &
        (transactions['date'] <= end_date)
    ].copy()

    if df.empty:
        return go.Figure()

    df['date'] = pd.to_datetime(df['date'], utc=True)

    # Compute income by period before any drilldown filter (used for normalization and reference line).
    # Income is stored positive — sum as-is (no abs).
    income_by_period = (
        df[df['csp_label'] == 'income']
        .groupby(pd.Grouper(key='date', freq=smoothing))['amount']
        .sum()
    )

    group_col = 'csp' if drilldown_label else 'csp_label'
    if drilldown_label:
        df = df[df['csp_label'] == drilldown_label]

    result = (
        df.groupby([pd.Grouper(key='date', freq=smoothing), group_col])['amount']
        .sum()
        .reset_index()
    )
    result.columns = ['date', group_col, 'amount']

    # Show spending as positive by negating signed amounts (expenses are stored
    # negative); never abs() — a net-credit period must stay negative. Income is
    # already positive, so leave any income column / income drill-down untouched.
    pivot = result.pivot_table(
        index='date', columns=group_col, values='amount', aggfunc='sum'
    ).fillna(0)
    if drilldown_label == 'income':
        pass  # income subcategories stay positive
    elif drilldown_label:
        pivot = pivot.mul(-1)  # subcategories of a spending bucket
    else:
        spend_cols = [c for c in pivot.columns if c != 'income']
        pivot.loc[:, spend_cols] = pivot[spend_cols].mul(-1)

    if as_percent:
        ref_income = income_by_period.reindex(pivot.index).replace(0, np.nan)
        for col in pivot.columns:
            pivot[col] = pivot[col] / ref_income
        pivot = pivot.fillna(0)

    csp_colors = {
        'fixed': '#F3969A',
        'investments': '#78C2AD',
        'sinking': '#FFCE67',
        'guilt-free': '#6f42c1',
        'income': '#888',
    }
    fallback_colors = ['#78c2ad', '#f3969a', '#ffce67', '#6f42c1', '#5bc0be', '#d972ff']

    label_order = ['fixed', 'investments', 'sinking', 'guilt-free']
    if drilldown_label:
        ordered_cols = list(pivot.columns)
    else:
        ordered_cols = [l for l in label_order if l in pivot.columns]
        ordered_cols += [c for c in pivot.columns if c not in label_order and c != 'income']

    fig = go.Figure()
    hover_fmt = '%{y:.1%}' if as_percent else '$%{y:,.0f}'

    for i, col in enumerate(ordered_cols):
        color = csp_colors.get(col, fallback_colors[i % len(fallback_colors)])
        label = col.replace('-', ' ').replace('_', ' ').title()
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=pivot[col],
            name=label,
            mode='lines',
            stackgroup='spending',
            line=dict(color=color, width=0.5),
            fillcolor=color,
            customdata=[col] * len(pivot),
            hovertemplate=f'{label}: {hover_fmt}<extra></extra>',
            hoverlabel=dict(bgcolor=color, bordercolor=color, font=dict(color='white')),
        ))

    # Income reference line (top-level view only)
    if not drilldown_label and not income_by_period.empty:
        income_aligned = income_by_period.reindex(pivot.index)
        income_y = [1.0] * len(pivot) if as_percent else income_aligned.values
        fig.add_trace(go.Scatter(
            x=pivot.index,
            y=income_y,
            name='Income',
            mode='lines',
            line=dict(color='#888', dash='dot', width=2),
            customdata=['income'] * len(pivot),
            hovertemplate=('Income: %{y:.1%}<extra></extra>' if as_percent
                          else 'Income: $%{y:,.0f}<extra></extra>'),
            hoverlabel=dict(bgcolor='#888', bordercolor='#888', font=dict(color='white')),
        ))

    tick_formats = {'YE': '%Y', 'ME': '%b %Y', 'W': '%b %d', 'D': '%b %d'}

    fig.update_layout(
        xaxis_title=None,
        yaxis_title='% of Income' if as_percent else 'Amount (USD)',
        yaxis_tickformat='.0%' if as_percent else '$,.0f',
        template='plotly_white',
        showlegend=True,
        hovermode='x unified',
        legend=dict(font=dict(color='#888')),
        font=dict(color='#888'),
        margin=dict(t=10, b=40, l=20, r=20),
        height=450,
    )
    fig.update_xaxes(tickformat=tick_formats.get(smoothing, '%Y-%m'))

    if as_percent:
        fig.update_layout(yaxis_range=[0, 1.5])

    return fig


def plot_csp_by_label(processed_transactions, as_percent):
    df = processed_transactions.groupby([processed_transactions['date'].dt.year, 'csp_label'])['amount'].sum().reset_index()
    df.columns = ['date', 'csp_label', 'value']
    df = df.loc[df['date'] >= 2020]
    df['value'] = df['value'].abs()

    # Define the desired order
    desired_order = ["fixed", "investments", "guilt-free", "sinking"]

    # Pivot the DataFrame to reshape it for calculations
    pivot_df = df.pivot(index="date", columns="csp_label", values="value").fillna(0)

    # Ensure all desired categories are in the DataFrame
    for col in desired_order:
        if col not in pivot_df.columns:
            pivot_df[col] = 0

    # Normalize by income for each year
    if as_percent:
        pivot_df[desired_order] = pivot_df[desired_order].div(pivot_df["income"], axis=0)

    # Handle missing or zero income to avoid division errors
    pivot_df[desired_order] = pivot_df[desired_order].fillna(0)

    # Define custom Minty theme-inspired colors
    minty_colors = ['#c2b2b4', '#78c2ad', '#5bc0be', '#8447ff', '#d972ff', '#ffb2e6']

    # Create the stacked area chart using stackgroup
    fig = go.Figure()

    for i, col in enumerate(desired_order):
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df[col],
            mode='lines',
            stackgroup='one',  # All traces share the same stackgroup
            name=col,
            line=dict(color=minty_colors[i % len(minty_colors)]),
            # hoverinfo='x',
            hovertemplate='{x}: $%{y:,.0f} <extra></extra>',
            hoverlabel=dict(bgcolor='#888', bordercolor='#888',
                            font=dict(color='white')),
        ))

    if not as_percent:
        fig.add_trace(go.Scatter(
            x=pivot_df.index,
            y=pivot_df['income'],
            mode='lines',
            name='income',
            line_color='#888',
            line_dash='dot',
            hovertemplate='Income: $%{y:,.0f} <extra></extra>',
            hoverlabel=dict(bgcolor='#888', bordercolor='#888',
                            font=dict(color='white')),
        ))

    if as_percent:
        for trace in fig.data:
            trace.hovertemplate='Proportion: %{y:.0%} <extra></extra>'




    # Update layout
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Total Spending",
        yaxis_tickformat="$,.0f",
        template="plotly_white",
        showlegend=True,
        yaxis_automargin=True,
        xaxis_automargin=True,
        legend=dict(
            font=dict(color="#888")
        ),
        font=dict(
            color="#888"
        )
        # colorway=plotly.colors.qualitative.Prism,
        # yaxis=dict(range=[0, 1])
    )

    if as_percent:
        fig.update_layout(
            yaxis_range=[0, 1.3],
            yaxis_title="Proportion of Income",
            yaxis_tickformat=".0%",
        )

    # Show the figure
    return fig
