from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from config import FEATURES
from features import Z_FEATURES
from membership import membership_at


def _align_centroids(previous_centers: np.ndarray, current_centers: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    distance = np.linalg.norm(previous_centers[:, None, :] - current_centers[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(distance)
    mapping = {int(current): int(stable) for stable, current in zip(rows, cols)}
    return mapping, distance


def _align_current_labels(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.asarray([mapping[int(label)] for label in labels], dtype=int)


def _reorder_centers(current_centers: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    aligned = np.empty_like(current_centers)
    for current_id, stable_id in mapping.items():
        aligned[stable_id] = current_centers[current_id]
    return aligned


def _assign(X: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    order = np.argsort(distances, axis=1)
    best_idx = order[:, 0]
    best = distances[np.arange(len(X)), best_idx]
    second = distances[np.arange(len(X)), order[:, 1]]
    margin = second - best
    confidence = margin / np.maximum(second, 1e-12)
    return best_idx.astype(int), best, second, confidence


def _state_profile(centers: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(centers, columns=Z_FEATURES)
    frame["MomentumScore"] = (frame["Z_Return20"] + frame["Z_RS20"]) / 2
    frame["RiskScore"] = (frame["Z_Volatility20"] + frame["Z_Beta60"]) / 2
    frame["FlowScore"] = frame["Z_VolumeZ20"]
    return frame


def _state_names(profile: pd.DataFrame) -> dict[int, str]:
    ids = list(profile.index)
    if not ids:
        return {}
    names: dict[int, str] = {}
    momentum = profile["MomentumScore"]
    risk = profile["RiskScore"]
    leader = int(momentum.idxmax())
    weak = int(momentum.idxmin())
    names[leader] = "Nhóm dẫn đầu"
    if weak != leader:
        names[weak] = "Nhóm suy yếu"
    remaining = [i for i in ids if i not in names]
    if remaining:
        risk_on = int(risk.loc[remaining].idxmax())
        names[risk_on] = "Nhóm chấp nhận rủi ro"
    remaining = [i for i in ids if i not in names]
    if remaining:
        names[int(remaining[0])] = "Nhóm phòng thủ"
    return names


def rolling_cluster(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, train_window: int = 60, k: int = 4, step: int = 5):
    all_dates = sorted(pd.to_datetime(panel["time"].dropna().unique()))
    all_dates = [pd.Timestamp(d) for d in all_dates if pd.Timestamp(d) <= end]
    observation_dates = [d for d in all_dates if d >= start]
    selected_dates = observation_dates[::step]

    results = []
    diagnostics = []
    previous_centers = None
    previous_current = None
    previous_date = None

    for current_date in selected_dates:
        idx = all_dates.index(current_date)
        if idx < train_window:
            continue

        train_dates = all_dates[idx - train_window:idx]
        train_parts = []
        for d in train_dates:
            active = membership_at(d)
            part = panel[(panel["time"] == d) & panel["Ticker"].isin(active)]
            train_parts.append(part)
        train = pd.concat(train_parts, ignore_index=True).dropna(subset=Z_FEATURES)

        active_current = membership_at(current_date)
        current = panel[(panel["time"] == current_date) & panel["Ticker"].isin(active_current)].dropna(subset=Z_FEATURES).copy()

        if current["Ticker"].nunique() != 30:
            continue
        if train.empty or train["Ticker"].nunique() < k:
            continue

        X_train = train[Z_FEATURES].to_numpy(dtype=float)
        X_current = current[Z_FEATURES].to_numpy(dtype=float)
        model = KMeans(n_clusters=k, random_state=42, n_init=50)
        labels_train = model.fit_predict(X_train)
        raw_labels_current = model.predict(X_current)
        raw_centers = model.cluster_centers_.copy()

        centroid_drift = np.zeros(k, dtype=float)
        if previous_centers is None:
            stable_labels_current = raw_labels_current.copy()
            stable_centers = raw_centers.copy()
            mapping = {i: i for i in range(k)}
        else:
            mapping, distance_matrix = _align_centroids(previous_centers, raw_centers)
            stable_labels_current = _align_current_labels(raw_labels_current, mapping)
            stable_centers = _reorder_centers(raw_centers, mapping)
            centroid_drift = np.linalg.norm(stable_centers - previous_centers, axis=1)

        profile = _state_profile(stable_centers)
        names = _state_names(profile)
        current_raw_idx, best, second, confidence = _assign(X_current, stable_centers)

        current["Cluster"] = stable_labels_current
        current["ClusterLabel"] = current["Cluster"].map(names).fillna(current["Cluster"].map(lambda x: f"State {int(x)}"))
        current["CentroidDistance"] = best
        current["SecondCentroidDistance"] = second
        current["AssignmentConfidence"] = confidence
        current["MomentumScore"] = (current["Z_Return20"] + current["Z_RS20"]) / 2
        current["RiskScore"] = (current["Z_Volatility20"] + current["Z_Beta60"]) / 2
        current["FlowScore"] = current["Z_VolumeZ20"]
        current["Date"] = current_date

        # Counterfactual test: what would today's feature vector be classified as
        # by yesterday's model? This separates stock-driven migration from
        # migration caused mainly by centroid movement.
        current["PreviousModelCluster"] = np.nan
        current["ModelDrivenChange"] = False
        current["FeatureDrivenChange"] = False
        current["MigrationType"] = "First observation"

        if previous_centers is not None and previous_current is not None:
            prev_map = previous_current.set_index("Ticker")["Cluster"]
            common = current["Ticker"].isin(prev_map.index)
            current.loc[common, "PreviousObservedCluster"] = current.loc[common, "Ticker"].map(prev_map)

            previous_raw_idx, _, _, _ = _assign(X_current, previous_centers)
            current.loc[common, "PreviousModelCluster"] = previous_raw_idx[common.to_numpy()]

            prev_obs = current["PreviousObservedCluster"]
            prev_model = current["PreviousModelCluster"]
            now = current["Cluster"]

            model_driven = common & prev_model.notna() & (prev_model != now) & (prev_obs == prev_model)
            feature_driven = common & prev_model.notna() & (prev_obs != prev_model)
            mixed = common & prev_model.notna() & (prev_obs != prev_model) & (prev_model != now)

            current.loc[model_driven, "ModelDrivenChange"] = True
            current.loc[feature_driven, "FeatureDrivenChange"] = True
            current.loc[mixed, "MigrationType"] = "Mixed"
            current.loc[model_driven & ~mixed, "MigrationType"] = "Model-driven"
            current.loc[feature_driven & ~mixed, "MigrationType"] = "Feature-driven"
            current.loc[common & (prev_obs == now) & ~model_driven & ~feature_driven, "MigrationType"] = "Stable"

        for state_id in range(k):
            current.loc[current["Cluster"] == state_id, "CentroidDrift"] = centroid_drift[state_id]

        results.append(current[[
            "Date", "Ticker", *FEATURES, *Z_FEATURES,
            "Cluster", "ClusterLabel", "CentroidDistance", "SecondCentroidDistance",
            "AssignmentConfidence", "MomentumScore", "RiskScore", "FlowScore",
            "PreviousObservedCluster", "PreviousModelCluster", "ModelDrivenChange",
            "FeatureDrivenChange", "MigrationType", "CentroidDrift",
        ]])

        unique_train = len(np.unique(labels_train))
        diagnostics.append({
            "Date": current_date,
            "K": k,
            "ActiveConstituents": len(active_current),
            "TrainObservations": len(train),
            "Silhouette": silhouette_score(X_train, labels_train) if unique_train > 1 else np.nan,
            "Calinski_Harabasz": calinski_harabasz_score(X_train, labels_train) if unique_train > 1 else np.nan,
            "Davies_Bouldin": davies_bouldin_score(X_train, labels_train) if unique_train > 1 else np.nan,
            "MeanAssignmentConfidence": float(np.mean(confidence)),
            "MeanCentroidDistance": float(np.mean(best)),
            "MeanCentroidDrift": float(np.mean(centroid_drift)),
            "MaxCentroidDrift": float(np.max(centroid_drift)),
        })
        previous_centers = stable_centers
        previous_current = current[["Ticker", "Cluster"]].copy()
        previous_date = current_date

    if not results:
        raise ValueError("Không đủ dữ liệu để chạy rolling clustering với cửa sổ đã chọn.")
    return pd.concat(results, ignore_index=True), pd.DataFrame(diagnostics)
