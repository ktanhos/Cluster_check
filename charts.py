from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from membership import ALL_STUDY_SYMBOLS, membership_at

STATE_COLORS = {0: "#440154", 1: "#31688e", 2: "#35b779", 3: "#fde725"}


def behavior_map(latest: pd.DataFrame, title: str = "Bản đồ hành vi VN30") -> go.Figure:
    frame = latest.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Ticker", "MomentumScore", "RiskScore", "Cluster"])
    if frame.empty:
        return go.Figure()
    latest_date = frame["Date"].max()
    frame = frame[frame["Date"] == latest_date].copy()
    frame = frame.sort_values(["Ticker", "CentroidDistance"], na_position="last").drop_duplicates("Ticker", keep="first")
    frame = frame[frame["Ticker"].isin(set(membership_at(latest_date)))].copy()
    fig = go.Figure()
    for cluster_id in sorted(frame["Cluster"].astype(int).unique()):
        x = frame[frame["Cluster"].astype(int) == cluster_id]
        label = x["ClusterLabel"].mode().iloc[0] if not x["ClusterLabel"].dropna().empty else f"Nhóm {cluster_id + 1}"
        cols = ["Ticker", "ClusterLabel", "AssignmentConfidence", "FlowScore"]
        fig.add_trace(go.Scatter(x=x["MomentumScore"], y=x["RiskScore"], mode="markers+text", text=x["Ticker"], textposition="top center", name=label, marker=dict(size=12, color=STATE_COLORS.get(cluster_id, "#636efa"), line=dict(width=0.8, color="white")), customdata=x[cols].to_numpy(), hovertemplate="Mã: %{customdata[0]}<br>Nhóm: %{customdata[1]}<br>Độ chắc chắn: %{customdata[2]:.2f}<br>Hoạt động dòng tiền: %{customdata[3]:.2f}<extra></extra>"))
    centroids = frame.groupby("Cluster", as_index=False)[["MomentumScore", "RiskScore"]].mean().sort_values("Cluster")
    if not centroids.empty:
        fig.add_trace(go.Scatter(x=centroids["MomentumScore"], y=centroids["RiskScore"], mode="markers+text", text=[f"Tâm nhóm {int(c) + 1}" for c in centroids["Cluster"]], textposition="bottom center", name="Tâm nhóm", marker=dict(symbol="x", size=14, color="#111111", line=dict(width=2)), hovertemplate="%{text}<br>Vị trí tương đối: %{x:.2f}, %{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.7)
    fig.update_layout(title=f"{title} · {latest_date:%d/%m/%Y}", xaxis_title="Sức mạnh tương đối", yaxis_title="Mức biến động", height=680, legend_title="Nhóm hành vi", margin=dict(l=30, r=30, t=60, b=40), hovermode="closest")
    return fig


def behavior_history_chart(rolling_result: pd.DataFrame, tickers: list[str], start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> go.Figure:
    frame = rolling_result.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Ticker", "Cluster"]).sort_values(["Ticker", "Date"])
    if start is not None:
        frame = frame[frame["Date"] >= pd.Timestamp(start)]
    if end is not None:
        frame = frame[frame["Date"] <= pd.Timestamp(end)]
    frame = frame[frame["Ticker"].isin(tickers)].drop_duplicates(["Ticker", "Date"], keep="last")
    if frame.empty:
        return go.Figure()
    fig = go.Figure()
    labels = {}
    for _, r in frame.iterrows():
        labels[int(r["Cluster"])] = r.get("ClusterLabel", f"Nhóm {int(r['Cluster']) + 1}")
    for ticker in tickers:
        x = frame[frame["Ticker"] == ticker]
        if x.empty:
            continue
        fig.add_trace(go.Scatter(x=x["Date"], y=x["Cluster"].astype(int) + 1, mode="lines+markers", name=ticker, customdata=x[["ClusterLabel", "AssignmentConfidence", "MigrationType"]].to_numpy(), hovertemplate="Mã: " + ticker + "<br>Ngày: %{x|%d/%m/%Y}<br>%{customdata[0]}<br>Độ chắc chắn: %{customdata[1]:.2f}<br>%{customdata[2]}<extra></extra>"))
    ordered = sorted(labels)
    fig.update_layout(title="Hành trình của cổ phiếu qua các nhóm", xaxis_title="Ngày quan sát", yaxis_title="Nhóm hành vi", yaxis=dict(tickmode="array", tickvals=[i + 1 for i in ordered], ticktext=[labels[i] for i in ordered]), height=500, hovermode="closest")
    return fig


def behavior_trajectory_chart(rolling_result: pd.DataFrame, ticker: str) -> go.Figure:
    frame = rolling_result.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame[frame["Ticker"] == ticker].dropna(subset=["Date", "MomentumScore", "RiskScore"]).sort_values("Date").drop_duplicates("Date")
    if frame.empty:
        return go.Figure()
    fig = go.Figure(go.Scatter(x=frame["MomentumScore"], y=frame["RiskScore"], mode="lines+markers+text", text=frame["Date"].dt.strftime("%d/%m"), textposition="top center", customdata=frame[["Date", "ClusterLabel", "AssignmentConfidence", "MigrationType"]].to_numpy(), marker=dict(size=9), line=dict(width=2), hovertemplate="Ngày: %{customdata[0]|%d/%m/%Y}<br>%{customdata[1]}<br>Độ chắc chắn: %{customdata[2]:.2f}<br>%{customdata[3]}<extra></extra>", name=ticker))
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.7)
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.7)
    fig.update_layout(title=f"Hành trình của {ticker} trên bản đồ hành vi", xaxis_title="Sức mạnh tương đối", yaxis_title="Mức biến động", height=560, showlegend=False)
    return fig


def migration_heatmap(migration: pd.DataFrame, change_table: pd.DataFrame | None = None) -> go.Figure:
    if migration is None or migration.empty:
        return go.Figure()
    frame = migration.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date", "Ticker"])
    frame = frame.sort_values(["Date", "Ticker", "CentroidDistance"], na_position="last").drop_duplicates(["Ticker", "Date"], keep="first")
    dates = [pd.Timestamp(d) for d in sorted(frame["Date"].unique())]
    if not dates:
        return go.Figure()
    active_by_date = {d: set(membership_at(d)) for d in dates}
    tickers = [t for t in ALL_STUDY_SYMBOLS if any(t in active for active in active_by_date.values())]
    pivot = frame.pivot(index="Ticker", columns="Date", values="Cluster").reindex(index=tickers, columns=dates)
    z = pivot.to_numpy(dtype=float)
    text = np.empty_like(z, dtype=object)
    for i, ticker in enumerate(tickers):
        for j, d in enumerate(dates):
            value = pivot.iloc[i, j]
            if ticker not in active_by_date[d]:
                text[i, j] = "Không thuộc VN30 tại thời điểm này"
                z[i, j] = np.nan
            elif pd.isna(value):
                text[i, j] = "Chưa có kết quả phân nhóm"
            else:
                text[i, j] = f"Nhóm {int(value) + 1}"
    colorscale = [[0.00, "#440154"], [0.2499, "#440154"], [0.25, "#31688e"], [0.4999, "#31688e"], [0.50, "#35b779"], [0.7499, "#35b779"], [0.75, "#fde725"], [1.00, "#fde725"]]
    fig = go.Figure(go.Heatmap(z=z, x=dates, y=tickers, text=text, customdata=text, colorscale=colorscale, zmin=0, zmax=3, connectgaps=False, colorbar=dict(title="Nhóm hành vi", tickvals=[0, 1, 2, 3], ticktext=["Nhóm 1", "Nhóm 2", "Nhóm 3", "Nhóm 4"]), hovertemplate="Mã: %{y}<br>Ngày: %{x|%d/%m/%Y}<br>%{customdata}<extra></extra>", xgap=0, ygap=0))
    if change_table is not None and not change_table.empty:
        for _, row in change_table.iterrows():
            d = pd.Timestamp(row["EffectiveDate"])
            if dates[0] <= d <= dates[-1]:
                fig.add_vline(x=d, line_dash="dash", line_color="red", opacity=0.85)
                fig.add_annotation(x=d, y=1.03, yref="paper", text=f"{d:%d/%m}<br>+ {row.get('Added', '') or ''}<br>− {row.get('Removed', '') or ''}", showarrow=False, align="center", font=dict(color="red", size=10))
    fig.update_layout(title="Chuyển nhóm của từng cổ phiếu theo thời gian", xaxis_title="Ngày quan sát", yaxis_title="Mã cổ phiếu", height=max(700, 20 * len(tickers) + 210), margin=dict(l=80, r=30, t=105, b=80), hovermode="closest")
    return fig


def cluster_count_chart(rolling_result: pd.DataFrame) -> go.Figure:
    frame = rolling_result.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.drop_duplicates(["Date", "Ticker"])
    counts = frame.groupby(["Date", "Cluster", "ClusterLabel"], as_index=False).size().rename(columns={"size": "Count"})
    fig = go.Figure()
    for label in counts["ClusterLabel"].dropna().unique():
        x = counts[counts["ClusterLabel"] == label]
        fig.add_trace(go.Scatter(x=x["Date"], y=x["Count"], mode="lines+markers", name=label))
    fig.update_layout(title="Quy mô các nhóm hành vi theo thời gian", xaxis_title="Ngày", yaxis_title="Số cổ phiếu", height=420)
    return fig
