from __future__ import annotations

import pandas as pd


def build_migration_table(rolling_result: pd.DataFrame) -> pd.DataFrame:
    x = rolling_result.sort_values(["Ticker", "Date"]).copy()
    x["PreviousCluster"] = x.groupby("Ticker")["Cluster"].shift(1)
    x["Migration"] = (x["PreviousCluster"].notna() & (x["PreviousCluster"] != x["Cluster"]))
    x["Transition"] = x.apply(
        lambda r: f"{int(r['PreviousCluster'])} → {int(r['Cluster'])}" if r["Migration"] else "Stable",
        axis=1,
    )
    return x
