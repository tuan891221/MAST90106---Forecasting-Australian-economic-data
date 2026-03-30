from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def forecast_chart(df: pd.DataFrame, variable: str):
    fig = go.Figure()

    actual = (
        df[["date", "actual"]]
        .drop_duplicates()
        .sort_values("date")
    )
    fig.add_trace(
        go.Scatter(
            x=actual["date"],
            y=actual["actual"],
            mode="lines",
            name="Actual",
        )
    )

    for model, g in df.groupby("model"):
        g = g.sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=g["date"],
                y=g["forecast"],
                mode="lines",
                name=model,
            )
        )

    fig.update_layout(
        title=f"{variable}: Actual vs Forecast",
        xaxis_title="Date",
        yaxis_title=variable,
        hovermode="x unified",
    )
    return fig


def metric_bar_chart(df: pd.DataFrame, metric: str):
    fig = px.bar(
        df.sort_values(metric),
        x="model",
        y=metric,
        color="model",
        title=f"Model comparison by {metric.upper()}",
    )
    return fig


def heatmap_chart(df: pd.DataFrame, metric: str, horizon: int):
    sub = df[df["horizon"] == horizon].copy()
    pivot = sub.pivot(index="model", columns="variable", values=metric)
    fig = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        title=f"{metric.upper()} Heatmap (Horizon={horizon})",
    )
    return fig