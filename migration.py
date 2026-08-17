from __future__ import annotations

import pandas as pd


def build_migration_table(
    rolling_result: pd.DataFrame,
    confirmation_steps: int = 2,
    confidence_threshold: float = 0.0,
) -> pd.DataFrame:
    """Create migration events without confusing model drift with stock drift.

    MigrationSignal is observable at the event date and is safe for forward-return
    studies. MigrationConfirmedRetrospective uses subsequent observations only to
    verify persistence and must not be used as a real-time predictor.
    """
    x = rolling_result.sort_values(["Ticker", "Date"]).copy()
    x["PreviousCluster"] = x.groupby("Ticker")["Cluster"].shift(1)
    x["Migration"] = x["PreviousCluster"].notna() & (x["PreviousCluster"] != x["Cluster"])
    x["MigrationSignal"] = x["Migration"] & (x["AssignmentConfidence"] >= confidence_threshold)

    x["Transition"] = x.apply(
        lambda r: (
            f"{int(r['PreviousCluster'])} → {int(r['Cluster'])}"
            if r["Migration"] else "Stable"
        ),
        axis=1,
    )

    x["StateRunID"] = x.groupby("Ticker")["Cluster"].transform(lambda s: s.ne(s.shift()).cumsum())
    x["StateDuration"] = x.groupby(["Ticker", "StateRunID"]).cumcount() + 1

    # Retrospective persistence test. This deliberately looks forward only to
    # label an event as persistent; it is not used by the forward-return test.
    confirm_n = max(1, int(confirmation_steps))
    x["MigrationConfirmedRetrospective"] = False
    for ticker, idx in x.groupby("Ticker").groups.items():
        rows = list(idx)
        for pos, row_idx in enumerate(rows):
            if not bool(x.at[row_idx, "MigrationSignal"]):
                continue
            target = x.at[row_idx, "Cluster"]
            future_rows = rows[pos + 1: pos + confirm_n]
            if len(future_rows) == confirm_n - 1 and all(x.at[j, "Cluster"] == target for j in future_rows):
                x.at[row_idx, "MigrationConfirmedRetrospective"] = True

    # A migration is economically more credible when the stock itself changed
    # model assignment under the previous centroids, rather than only because
    # the centroids moved.
    x["EconomicallyDrivenMigration"] = (
        x["MigrationSignal"]
        & x["FeatureDrivenChange"]
        & ~x["ModelDrivenChange"]
    )

    # Keep the old field as an explicit retrospective alias for compatibility.
    x["MigrationConfirmed"] = x["MigrationConfirmedRetrospective"]
    return x.drop(columns=["StateRunID"])
