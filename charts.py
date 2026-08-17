from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from membership import ALL_STUDY_SYMBOLS, membership_at


STATE_COLORS = {
    0: "#440154",
    1: "#31688e",
    2: "#35b779",
    3: "#fde725",
}


def behavior_map(latest: pd.DataFrame, title: str = "Biểu đồ hành vi thị trường VN30") -> go.Figure:
    """Behavior map for the active VN30 constituents at one observation date.

    Only constituents active on the selected date are plotted. Centroids are
    shown separately so the map describes the rolling model rather than only
    the stock observations.
    """
    latest = latest.copy().dropna(subset=["MomentumScore", "RiskScore", "Cluster"])
    fig = go.Figure()

    for cluster_id in sorted(latest["Cluster"].astype(int).unique()):
        x = latest[latest["Cluster"].astype(int) == cluster_id]
        label = x["ClusterLabel"].mode().iloc[0] if not x["ClusterLabel"].dropna().empty else f"State {cluster_id}"
        fig.add_trace(go.Scatter(
            x=x["MomentumScore"],
            y=x["RiskScore"],
            mode="markers+text",
            text=x["Ticker"],
            textposition="top center",
            name=label,
            marker=dict(
                size=12,
                color=STATE_COLORS.get(cluster_id, "#636efa"),
                line=dict(width=0.8, color="white"),
            ),
            customdata=x[["Ticker", "Cluster", "AssignmentConfidence", "FlowScore", "CentroidDistance"]].to_numpy(),
            hovertemplate=(
                "Mã: %{customdata[0]}<br>"
                "Nhóm: %{customdata[1]}<br>"
                "Confidence: %{customdata[2]:.2f}<br>"
                "Flow: %{customdata[3]:.2f}<br>"
                "Khoảng cách centroid: %{customdata[4]:.2f}<extra></extra>"
            ),
        ))

    # The centroid of each active state is the mean location of its assigned
    # stocks in the displayed behavior coordinates. This avoids mixing the
    # six-dimensional clustering centroid with the two-dimensional projection.
    centroids = (
        latest.groupby("Cluster", as_index=False)[["MomentumScore", "RiskScore"]]
        .mean()
        .sort_values("Cluster")
    )
    if not centroids.empty:
        fig.add_trace(go.Scatter(
            x=centroids["MomentumScore"],
            y=centroids["RiskScore"],
            mode="markers+text",
            text=[f"Tâm cụm {int(c)}" for c in centroids["Cluster"]],
            textposition="bottom center",
            name="Tâm cụm",
            marker=dict(symbol="x", size=14, color="#111111", line=dict(width=2)),
            hovertemplate="Cluster: %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.7)
    fig.update_layout(
        title=title,
        xaxis_title="Động lượng / Sức mạnh tương đối",
        yaxis_title="Rủi ro / Mức độ biến động",
        height=680,
        legend_title="Nhóm hành vi",
        margin=dict(l=30, r=30, t=60, b=40),
        hovermode="closest",
    )
    return fig


def migration_heatmap(
    migration: pd.DataFrame,
    change_table: pd.DataFrame | None = None,
) -> go.Figure:
    """Constituent-aware cluster migration heatmap.

    Missing cells are intentionally left blank because the ticker was not a
    VN30 constituent at that observation date. This prevents a stock from
    appearing to have a cluster state before it entered, or after it left, the
    index. Vertical markers identify actual constituent effective dates.
    """
    dates = sorted(pd.to_datetime(migration["Date"].dropna().unique()))
    if not dates:
        return go.Figure()

    dates = [pd.Timestamp(d) for d in dates]
    tickers = [t for t in ALL_STUDY_SYMBOLS if t in set(migration["Ticker"])]

    pivot = (
        migration.assign(Date=pd.to_datetime(migration["Date"]))
        .pivot(index="Ticker", columns="Date", values="Cluster")
        .reindex(index=tickers, columns=dates)
    )

    z = pivot.to_numpy(dtype=float)
    text = np.empty_like(z, dtype=object)

    for i, ticker in enumerate(tickers):
        for j, d in enumerate(dates):
            active = ticker in set(membership_at(d))
            value = pivot.iloc[i, j]
            if not active:
                text[i, j] = "Không thuộc VN30"
            elif pd.isna(value):
                text[i, j] = "Thiếu dữ liệu / chưa phân loại"
            else:
                text[i, j] = f"Cluster {int(value)}"

    colorscale = [
        [0.00, "#440154"], [0.2499, "#440154"],
        [0.25, "#31688e"], [0.4999, "#31688e"],
        [0.50, "#35b779"], [0.7499, "#35b779"],
        [0.75, "#fde725"], [1.00, "#fde725"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=dates,
        y=tickers,
        text=text,
        customdata=text,
        colorscale=colorscale,
        zmin=0,
        zmax=3,
        connectgaps=False,
        colorbar=dict(title="Cluster ID", tickvals=[0, 1, 2, 3]),
        hovertemplate="Mã: %{y}<br>Ngày: %{x|%d/%m/%Y}<br>%{customdata}<extra></extra>",
        xgap=0,
        ygap=0,
    ))

    if change_table is not None and not change_table.empty:
        for _, row in change_table.iterrows():
            d = pd.Timestamp(row["EffectiveDate"])
            if dates[0] <= d <= dates[-1]:
                added = row.get("Added", "") or ""
                removed = row.get("Removed", "") or ""
                fig.add_vline(x=d, line_dash="dash", line_color="red", opacity=0.85)
                fig.add_annotation(
                    x=d,
                    y=1.03,
                    yref="paper",
                    text=f"{d:%d/%m}<br>+ {added}<br>− {removed}",
                    showarrow=False,
                    align="center",
                    font=dict(color="red", size=10),
                )

    fig.update_layout(
        title="Quá trình dịch chuyển cụm của rổ VN30 theo thành phần thực tế",
        xaxis_title="Ngày quan sát",
        yaxis_title="Mã cổ phiếu",
        height=max(700, 20 * len(tickers) + 210),
        margin=dict(l=80, r=30, t=105, b=80),
        hovermode="closest",
    )
    return fig


def cluster_count_chart(rolling_result: pd.DataFrame) -> go.Figure:
    counts = rolling_result.groupby(["Date", "Cluster"]).size().reset_index(name="Count")
    fig = go.Figure()
    for cluster_id in sorted(counts["Cluster"].unique()):
        x = counts[counts["Cluster"] == cluster_id]
        fig.add_trace(go.Scatter(
            x=x["Date"],
            y=x["Count"],
            mode="lines+markers",
            name=f"Cluster {int(cluster_id)}",
        ))
    fig.update_layout(
        title="Quy mô từng nhóm hành vi theo thời gian",
        xaxis_title="Ngày",
        yaxis_title="Số cổ phiếu",
        height=420,
    )
    return fig


def membership_count_chart(dates: list[pd.Timestamp]) -> go.Figure:
    rows = []
    for d in dates:
        rows.append({"Date": pd.Timestamp(d), "Constituents": len(membership_at(pd.Timestamp(d)))})
    frame = pd.DataFrame(rows)
    fig = go.Figure(go.Scatter(
        x=frame["Date"],
        y=frame["Constituents"],
        mode="lines+markers",
        name="VN30",
    ))
    fig.update_layout(
        title="Số thành phần VN30 theo thời gian",
        xaxis_title="Ngày",
        yaxis_title="Số mã đang thuộc rổ",
        yaxis=dict(range=[29.5, 30.5], dtick=1),
        height=320,
    )
    return fig
