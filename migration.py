from __future__ import annotations

import pandas as pd


def build_migration_table(
    rolling_result: pd.DataFrame,
    confirmation_steps: int = 2,
    confidence_threshold: float = 0.0,
) -> pd.DataFrame:
    """Create raw and confirmed cluster migrations."""
    x = rolling_result.sort_values(["Ticker", "Date"]).copy()
    x["PreviousCluster"] = x.groupby("Ticker")["Cluster"].shift(1)
    x["Migration"] = (
        x["PreviousCluster"].notna()
        & (x["PreviousCluster"] != x["Cluster"])
    )

    x["Transition"] = x.apply(
        lambda r: (
            f"{int(r['PreviousCluster'])} → {int(r['Cluster'])}"
            if r["Migration"] else "Stable"
        ),
        axis=1,
    )

    state_change = x.groupby("Ticker")["Cluster"].transform(
        lambda s: s.ne(s.shift()).cumsum()
    )
    x["StateRunID"] = state_change
    x["StateRunLength"] = x.groupby(["Ticker", "StateRunID"]).cumcount() + 1

    if "AssignmentConfidence" in x.columns:
        confidence_ok = x["AssignmentConfidence"] >= confidence_threshold
    else:
        confidence_ok = pd.Series(True, index=x.index)

    x["MigrationConfirmed"] = (
        x["Migration"]
        & (x["StateRunLength"] >= max(1, int(confirmation_steps)))
        & confidence_ok
    )
    x["StateDuration"] = x["StateRunLength"]
    return x.drop(columns=["StateRunID"])
