from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


STATE_COLORS = {
    0: "#440154",
    1: "#31688e",
    2: "#35b779",
    3: "#fde725",
}


def behavior_map(latest: pd.DataFrame, title: str = "Biểu đồ hành vi thị trường VN30") -> go.Figure:
    fig = go.Figure()
    labels = latest["ClusterLabel"].dropna().unique().tolist()
    for label in labels:
        x = latest[latest["ClusterLabel"] == label]
        cluster_id = int(x["Cluster"].iloc[0])
        fig.add_trace(go.Scatter(
            x=x["MomentumScore"],
            y=x["RiskScore"],
            mode="markers+text",
            text=x["Ticker"],
            textposition="top center",
            name=label,
            marker=dict(size=11, color=STATE_COLORS.get(cluster_id, "#636efa"), line=dict(width=0.5, color="white")),
            customdata=x[["Ticker", "Cluster", "AssignmentConfidence", "FlowScore"]].to_numpy(),
            hovertemplate="Mã: %{customdata[0]}<br>Nhóm: %{customdata[1]}<br>Confidence: %{customdata[2]:.2f}<br>Flow: %{customdata[3]:.2f}<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="Động lượng / Sức mạnh tương đối",
        yaxis_title="Rủi ro / Mức độ biến động",
        height=680,
        legend_title="Nhóm hành vi",
        margin=dict(l=30, r=30, t=60, b=40),
    )
    return fig


def migration_heatmap(migration: pd.DataFrame, change_dates: list[pd.Timestamp] | None = None) -> go.Figure:
    pivot = migration.pivot(index="Ticker", columns="Date", values="Cluster")
    tickers = pivot.index.tolist()
    dates = pivot.columns.tolist()
    z = pivot.to_numpy(dtype=float)
    text = pivot.fillna("").astype(object).to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z,
        x=dates,
        y=tickers,
        text=text,
        hovertemplate="Mã: %{y}<br>Ngày: %{x|%d/%m/%Y}<br>Cluster: %{z}<extra></extra>",
        colorscale=[
            [0.00, "#440154"], [0.25, "#440154"],
            [0.25, "#31688e"], [0.50, "#31688e"],
            [0.50, "#35b779"], [0.75, "#35b779"],
            [0.75, "#fde725"], [1.00, "#fde725"],
        ],
        zmin=0,
        zmax=max(3, int(pd.Series(migration["Cluster"]).max())),
        colorbar=dict(title="Cluster ID"),
        xgap=0,
        ygap=0,
    ))
    for d in change_dates or []:
        fig.add_vline(x=pd.Timestamp(d), line_dash="dash", line_color="red", opacity=0.8)
        fig.add_annotation(x=pd.Timestamp(d), y=1.02, yref="paper", text=pd.Timestamp(d).strftime("%d/%m"), showarrow=False, font=dict(color="red"))
    fig.update_layout(
        title="Quá trình dịch chuyển cụm của rổ VN30",
        xaxis_title="Ngày quan sát",
        yaxis_title="Mã cổ phiếu",
        height=max(650, 18 * len(tickers) + 180),
        margin=dict(l=80, r=30, t=70, b=80),
    )
    return fig


def cluster_count_chart(rolling_result: pd.DataFrame) -> go.Figure:
    counts = rolling_result.groupby(["Date", "Cluster"]).size().reset_index(name="Count")
    fig = go.Figure()
    for cluster_id in sorted(counts["Cluster"].unique()):
        x = counts[counts["Cluster"] == cluster_id]
        fig.add_trace(go.Scatter(x=x["Date"], y=x["Count"], mode="lines+markers", name=f"Cluster {int(cluster_id)}"))
    fig.update_layout(title="Quy mô từng nhóm hành vi theo thời gian", xaxis_title="Ngày", yaxis_title="Số cổ phiếu", height=420)
    return fig
