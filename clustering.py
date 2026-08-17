from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from config import FEATURES, VN30
from features import Z_FEATURES


def _align_centroids(previous_centers: np.ndarray, current_centers: np.ndarray) -> dict[int, int]:
    distance = np.linalg.norm(
        previous_centers[:, None, :] - current_centers[None, :, :], axis=2
    )
    rows, cols = linear_sum_assignment(distance)
    return {int(current): int(stable) for stable, current in zip(rows, cols)}


def _align_current_labels(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.asarray([mapping[int(label)] for label in labels], dtype=int)


def _reorder_centers(current_centers: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    aligned = np.empty_like(current_centers)
    for current_id, stable_id in mapping.items():
        aligned[stable_id] = current_centers[current_id]
    return aligned


def _assignment_metrics(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    order = np.argsort(distances, axis=1)
    best = distances[np.arange(len(X)), order[:, 0]]
    second = distances[np.arange(len(X)), order[:, 1]]
    margin = second - best
    confidence = margin / np.maximum(second, 1e-12)
    return best, second, confidence


def _state_profile(centers: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(centers, columns=Z_FEATURES)
    frame["MomentumScore"] = (frame["Z_Return20"] + frame["Z_RS20"]) / 2
    frame["RiskScore"] = (frame["Z_Volatility20"] + frame["Z_Beta60"]) / 2
    frame["FlowScore"] = frame["Z_VolumeZ20"]
    return frame


def rolling_cluster(
    panel: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_window: int = 60,
    k: int = 4,
    step: int = 5,
):
    dates = sorted(pd.to_datetime(panel["time"].dropna().unique()))
    dates = [pd.Timestamp(d) for d in dates if start <= pd.Timestamp(d) <= end]
    results = []
    diagnostics = []
    previous_centers = None

    for idx in range(train_window, len(dates), step):
        current_date = dates[idx]
        train_dates = dates[max(0, idx - train_window):idx]
        train = panel[panel["time"].isin(train_dates)].dropna(subset=Z_FEATURES).copy()
        current = panel[panel["time"] == current_date].dropna(subset=Z_FEATURES).copy()

        if current["Ticker"].nunique() != len(VN30):
            continue
        if train.empty or train["Ticker"].nunique() < k:
            continue

        X_train = train[Z_FEATURES].to_numpy(dtype=float)
        X_current = current[Z_FEATURES].to_numpy(dtype=float)

        model = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels_train = model.fit_predict(X_train)
        raw_labels_current = model.predict(X_current)
        raw_centers = model.cluster_centers_.copy()

        if previous_centers is None:
            stable_labels_current = raw_labels_current.copy()
            stable_centers = raw_centers.copy()
        else:
            mapping = _align_centroids(previous_centers, raw_centers)
            stable_labels_current = _align_current_labels(raw_labels_current, mapping)
            stable_centers = _reorder_centers(raw_centers, mapping)

        best, second, confidence = _assignment_metrics(X_current, stable_centers)

        current = current.copy()
        current["Cluster"] = stable_labels_current
        current["ClusterLabel"] = current["Cluster"].map(
            {state: f"State {state}" for state in range(k)}
        )
        current["CentroidDistance"] = best
        current["SecondCentroidDistance"] = second
        current["AssignmentConfidence"] = confidence
        current["Date"] = current_date
        results.append(
            current[
                ["Date", "Ticker"] + FEATURES + Z_FEATURES + [
                    "Cluster", "ClusterLabel", "CentroidDistance",
                    "SecondCentroidDistance", "AssignmentConfidence",
                ]
            ]
        )

        unique_train = len(np.unique(labels_train))
        diagnostics.append({
            "Date": current_date,
            "K": k,
            "TrainObservations": len(train),
            "Silhouette": silhouette_score(X_train, labels_train) if unique_train > 1 else np.nan,
            "Calinski_Harabasz": calinski_harabasz_score(X_train, labels_train) if unique_train > 1 else np.nan,
            "Davies_Bouldin": davies_bouldin_score(X_train, labels_train) if unique_train > 1 else np.nan,
            "MeanAssignmentConfidence": float(np.mean(confidence)),
            "MeanCentroidDistance": float(np.mean(best)),
        })
        previous_centers = stable_centers

    if not results:
        raise ValueError("Không đủ dữ liệu để chạy rolling clustering với cửa sổ đã chọn.")

    return pd.concat(results, ignore_index=True), pd.DataFrame(diagnostics)
