import numpy as np
import pandas as pd

from clustering import _align_centroids
from migration import build_migration_table


def test_centroid_alignment_is_invariant_to_kmeans_label_permutation():
    previous = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, 1.0]])
    current = np.array([[-3.0, 1.0], [0.0, 0.0], [3.0, 3.0]])
    mapping = _align_centroids(previous, current)
    assert mapping == {1: 0, 2: 1, 0: 2}


def test_confirmed_migration_requires_persistence():
    rows = []
    dates = pd.date_range("2026-01-01", periods=4, freq="5D")
    clusters = [0, 1, 1, 1]
    for date, cluster in zip(dates, clusters):
        rows.append(
            {
                "Date": date,
                "Ticker": "VCB",
                "Cluster": cluster,
                "AssignmentConfidence": 0.5,
            }
        )

    result = build_migration_table(
        pd.DataFrame(rows), confirmation_steps=2, confidence_threshold=0.1
    )
    assert result.loc[result["Date"] == dates[1], "Migration"].iloc[0]
    assert not result.loc[result["Date"] == dates[1], "MigrationConfirmed"].iloc[0]
    assert result.loc[result["Date"] == dates[2], "MigrationConfirmed"].iloc[0]
