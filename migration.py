from __future__ import annotations

import pandas as pd


def build_migration_table(rolling_result: pd.DataFrame, confirmation_steps: int = 2, confidence_threshold: float = 0.0) -> pd.DataFrame:
    """Build observable migration events and a retrospective persistence flag.

    MigrationSignal is available at the event date and is appropriate for forward
    return studies. MigrationConfirmedRetrospective deliberately looks ahead and
    is therefore descriptive, not a real-time trading signal.
    """
    x = rolling_result.sort_values(["Ticker", "Date"]).copy()
    x["PreviousCluster"] = x.groupby("Ticker")["Cluster"].shift(1)
    x["Migration"] = x["PreviousCluster"].notna() & (x["PreviousCluster"] != x["Cluster"])
    x["MigrationSignal"] = x["Migration"] & (x["AssignmentConfidence"] >= confidence_threshold)
    x["Transition"] = x.apply(lambda r: f"{int(r['PreviousCluster'])} → {int(r['Cluster'])}" if r["Migration"] else "Stable", axis=1)
    x["StateRunID"] = x.groupby("Ticker")["Cluster"].transform(lambda s: s.ne(s.shift()).cumsum())
    x["StateDuration"] = x.groupby(["Ticker", "StateRunID"]).cumcount() + 1

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

    x["EconomicallyDrivenMigration"] = x["MigrationType"].eq("Feature-driven") & x["MigrationSignal"]
    x["MigrationConfirmed"] = x["MigrationConfirmedRetrospective"]
    return x.drop(columns=["StateRunID"])
