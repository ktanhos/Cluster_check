from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from config import FEATURES, VN30


def _align_labels(previous_centers: np.ndarray, current_centers: np.ndarray):
    d = np.linalg.norm(previous_centers[:, None, :] - current_centers[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(d)
    mapping = {int(c): int(r) for r, c in zip(rows, cols)}
    return mapping


def rolling_cluster(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, train_window: int = 60, k: int = 4, step: int = 5):
    dates = sorted(pd.to_datetime(panel["time"].dropna().unique()))
    dates = [pd.Timestamp(d) for d in dates if start <= pd.Timestamp(d) <= end]
    results = []
    diagnostics = []
    previous_centers = None

    for idx in range(train_window, len(dates), step):
        current_date = dates[idx]
        train_dates = dates[max(0, idx - train_window):idx]
        train = panel[panel["time"].isin(train_dates)].dropna(subset=FEATURES).copy()
        current = panel[panel["time"] == current_date].dropna(subset=FEATURES).copy()
        if len(current) != len(VN30):
            continue
        if train.empty or train["Ticker"].nunique() < k:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURES])
        X_current = scaler.transform(current[FEATURES])

        model = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels_train = model.fit_predict(X_train)
        labels_current = model.predict(X_current)
        centers = model.cluster_centers_

        if previous_centers is not None and previous_centers.shape == centers.shape:
            mapping = _align_labels(previous_centers, centers)
            labels_current = np.array([mapping[int(x)] for x in labels_current])
            centers = np.array([centers[next(c for c, stable in mapping.items() if stable == s)] for s in range(k)])
        else:
            centers = centers.copy()

        current = current.copy()
        current["Cluster"] = labels_current
        current["ClusterLabel"] = current["Cluster"].map({0: "State 0", 1: "State 1", 2: "State 2", 3: "State 3"})
        current["Date"] = current_date
        results.append(current[["Date", "Ticker"] + FEATURES + ["Cluster", "ClusterLabel"]])

        diagnostics.append({
            "Date": current_date,
            "K": k,
            "Silhouette": silhouette_score(X_train, labels_train) if len(np.unique(labels_train)) > 1 else np.nan,
            "Calinski_Harabasz": calinski_harabasz_score(X_train, labels_train) if len(np.unique(labels_train)) > 1 else np.nan,
            "Davies_Bouldin": davies_bouldin_score(X_train, labels_train) if len(np.unique(labels_train)) > 1 else np.nan,
        })
        previous_centers = centers

    if not results:
        raise ValueError("Không đủ dữ liệu để chạy rolling clustering với cửa sổ đã chọn.")
    return pd.concat(results, ignore_index=True), pd.DataFrame(diagnostics)
