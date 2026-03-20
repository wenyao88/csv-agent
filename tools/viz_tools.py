import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def radar_chart(df: pd.DataFrame, sensor_cols: list, label_col: str = None) -> go.Figure:
    if label_col and label_col in df.columns:
        group_means = df.groupby(label_col)[sensor_cols].mean()
    else:
        group_means = pd.DataFrame([df[sensor_cols].mean()], index=["全部数据"])

    fig = go.Figure()
    for label, row in group_means.iterrows():
        values = row.tolist()
        values.append(values[0])
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=sensor_cols + [sensor_cols[0]],
            fill="toself",
            name=str(label)
        ))
    fig.update_layout(
        title="传感器响应雷达图",
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True
    )
    return fig


def pca_scatter(pca_data: pd.DataFrame, labels=None, label_name: str = "类别") -> go.Figure:
    df_plot = pca_data.copy()
    if labels is not None:
        df_plot[label_name] = [str(l) for l in labels]
        fig = px.scatter(
            df_plot, x="PC1", y="PC2", color=label_name,
            title="PCA 主成分散点图",
            labels={"PC1": "第一主成分 (PC1)", "PC2": "第二主成分 (PC2)"}
        )
    else:
        fig = px.scatter(
            df_plot, x="PC1", y="PC2",
            title="PCA 主成分散点图",
            labels={"PC1": "第一主成分 (PC1)", "PC2": "第二主成分 (PC2)"}
        )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    return fig


def sensor_heatmap(df: pd.DataFrame, sensor_cols: list, label_col: str = None) -> go.Figure:
    if label_col and label_col in df.columns:
        matrix = df.groupby(label_col)[sensor_cols].mean()
        y_labels = [str(l) for l in matrix.index.tolist()]
        z_values = matrix.values
    else:
        z_values = df[sensor_cols].corr().values
        y_labels = sensor_cols

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=sensor_cols,
        y=y_labels,
        colorscale="RdBu_r",
        text=np.round(z_values, 3),
        texttemplate="%{text}",
    ))
    title = "各气体传感器平均响应热力图" if label_col else "传感器响应相关性热力图"
    fig.update_layout(title=title)
    return fig


def response_curves(df: pd.DataFrame, sensor_cols: list,
                    label_col: str = None, max_sensors: int = 6) -> go.Figure:
    cols = sensor_cols[:max_sensors]
    n = len(cols)
    rows = (n + 1) // 2
    fig = make_subplots(rows=rows, cols=2, subplot_titles=cols)

    for i, col in enumerate(cols):
        row = i // 2 + 1
        c = i % 2 + 1
        if label_col and label_col in df.columns:
            for lbl in df[label_col].unique():
                subset = df[df[label_col] == lbl][col].reset_index(drop=True)
                fig.add_trace(
                    go.Scatter(y=subset.values, name=str(lbl),
                               showlegend=(i == 0)),
                    row=row, col=c
                )
        else:
            fig.add_trace(
                go.Scatter(y=df[col].values, name=col, showlegend=False),
                row=row, col=c
            )

    fig.update_layout(title="传感器响应曲线", height=300 * rows)
    return fig


def discrimination_bar(scores: dict) -> go.Figure:
    sensors = list(scores.keys())
    values = list(scores.values())
    fig = px.bar(
        x=sensors, y=values,
        labels={"x": "传感器", "y": "区分能力得分"},
        title="传感器气体区分能力排名",
        color=values,
        color_continuous_scale="Blues"
    )
    fig.update_layout(showlegend=False)
    return fig