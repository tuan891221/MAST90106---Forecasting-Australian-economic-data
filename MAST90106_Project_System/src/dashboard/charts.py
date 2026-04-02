from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _add_vertical_split_line(fig: go.Figure, x_value, label: str = "Forecast Start") -> None:
    x_value = pd.to_datetime(x_value)

    fig.add_shape(
        type="line",
        x0=x_value,
        x1=x_value,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(width=2, dash="dot"),
    )

    fig.add_annotation(
        x=x_value,
        y=1,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        yanchor="bottom",
    )


def single_variable_history_forecast_chart(
    hist_df: pd.DataFrame,
    roll_df: pd.DataFrame,
    future_df: pd.DataFrame,
    variable: str,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["actual"],
            mode="lines",
            name="Actual",
        )
    )

    if not roll_df.empty:
        fig.add_trace(
            go.Scatter(
                x=roll_df["date"],
                y=roll_df["forecast"],
                mode="lines",
                name="Rolling Forecast",
                line=dict(dash="dot"),
            )
        )

    if not future_df.empty:
        fig.add_trace(
            go.Scatter(
                x=future_df["date"],
                y=future_df["forecast"],
                mode="lines+markers",
                name="Future Forecast",
                line=dict(dash="dash"),
            )
        )

        _add_vertical_split_line(fig, future_df["date"].min(), "Forecast Start")

    fig.update_layout(
        title=f"{variable} — Historical and Forecast",
        xaxis_title="Date",
        yaxis_title=variable,
        hovermode="x unified",
    )
    return fig


def combined_history_forecast_chart(
    historical_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    forecast_start_date,
):
    fig = go.Figure()

    variables = historical_df["variable"].dropna().unique().tolist()

    for variable in variables:
        hist_sub = historical_df[historical_df["variable"] == variable].sort_values("date")
        fcst_sub = forecast_df[forecast_df["variable"] == variable].sort_values("date")

        fig.add_trace(
            go.Scatter(
                x=hist_sub["date"],
                y=hist_sub["value"],
                mode="lines",
                name=f"{variable} (historical)",
            )
        )

        if not fcst_sub.empty:
            fig.add_trace(
                go.Scatter(
                    x=fcst_sub["date"],
                    y=fcst_sub["value"],
                    mode="lines+markers",
                    name=f"{variable} (forecast)",
                    line=dict(dash="dash"),
                )
            )

    _add_vertical_split_line(fig, forecast_start_date, "Forecast Start")

    fig.update_layout(
        title="Combined Historical and Forecast View (Standardized)",
        xaxis_title="Date",
        yaxis_title="Standardized Value (z-score)",
        hovermode="x unified",
    )
    return fig


def metric_bar_chart(metrics_df: pd.DataFrame, metric: str):
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric column '{metric}' not found in metrics dataframe.")

    fig = px.bar(
        metrics_df,
        x="model",
        y=metric,
        color="variable" if "variable" in metrics_df.columns else None,
        barmode="group",
        facet_col="horizon" if "horizon" in metrics_df.columns else None,
        title=f"{metric.upper()} by Model",
    )
    fig.update_layout(xaxis_title="Model", yaxis_title=metric.upper())
    return fig


def heatmap_chart(metrics_df: pd.DataFrame, metric: str = "rmse"):
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric column '{metric}' not found in metrics dataframe.")

    required = {"variable", "model"}
    if not required.issubset(metrics_df.columns):
        raise ValueError("metrics_df must contain 'variable' and 'model' columns for heatmap.")

    heatmap_df = metrics_df.copy()

    if "horizon" in heatmap_df.columns:
        heatmap_df["model_label"] = (
            heatmap_df["model"].astype(str) + "_h" + heatmap_df["horizon"].astype(str)
        )
        pivot = heatmap_df.pivot_table(
            index="variable",
            columns="model_label",
            values=metric,
            aggfunc="mean",
        )
    else:
        pivot = heatmap_df.pivot_table(
            index="variable",
            columns="model",
            values=metric,
            aggfunc="mean",
        )

    fig = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        title=f"{metric.upper()} Heatmap",
    )
    fig.update_layout(xaxis_title="Model", yaxis_title="Variable")
    return fig

def holdout_train_test_chart(
    actual_df,
    fitted_df,
    forecast_df,
    variable: str,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actual_df["date"],
            y=actual_df[variable],
            mode="lines",
            name="Actual",
        )
    )

    if fitted_df is not None and not fitted_df.empty and variable in fitted_df.columns:
        fig.add_trace(
            go.Scatter(
                x=fitted_df["date"],
                y=fitted_df[variable],
                mode="lines",
                name="Fitted (Train)",
                line=dict(dash="dot"),
            )
        )

    if forecast_df is not None and not forecast_df.empty and variable in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["date"],
                y=forecast_df[variable],
                mode="lines+markers",
                name="Forecast (Test)",
                line=dict(dash="dash"),
            )
        )

        _add_vertical_split_line(fig, forecast_df["date"].min(), "Test Forecast Start")

    fig.update_layout(
        title=f"{variable} — Actual / Fitted (Train) / Forecast (Test)",
        xaxis_title="Date",
        yaxis_title=variable,
        hovermode="x unified",
    )
    return fig

from plotly.subplots import make_subplots


def holdout_all_variables_chart(
    actual_df,
    fitted_df,
    forecast_df,
    variables: list[str],
):
    fig = make_subplots(
        rows=len(variables),
        cols=1,
        shared_xaxes=True,
        subplot_titles=variables,
        vertical_spacing=0.06,
    )

    forecast_start_date = None
    if forecast_df is not None and not forecast_df.empty:
        forecast_start_date = forecast_df["date"].min()

    for i, variable in enumerate(variables, start=1):
        # Actual
        fig.add_trace(
            go.Scatter(
                x=actual_df["date"],
                y=actual_df[variable],
                mode="lines",
                name="Actual" if i == 1 else None,
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
        )

        # Fitted
        if fitted_df is not None and not fitted_df.empty and variable in fitted_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=fitted_df["date"],
                    y=fitted_df[variable],
                    mode="lines",
                    name="Fitted (Train)" if i == 1 else None,
                    line=dict(dash="dot"),
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )

        # Forecast
        if forecast_df is not None and not forecast_df.empty and variable in forecast_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df[variable],
                    mode="lines+markers",
                    name="Forecast (Test)" if i == 1 else None,
                    line=dict(dash="dash"),
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )

        # Vertical split line
        if forecast_start_date is not None:
            fig.add_vline(
                x=forecast_start_date,
                line_dash="dot",
                row=i,
                col=1,
            )

    fig.update_layout(
        title="Train / Test Forecast View — All Variables",
        height=320 * len(variables),
        hovermode="x unified",
    )

    fig.update_xaxes(title_text="Date", row=len(variables), col=1)

    return fig