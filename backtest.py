from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_forward_returns(stock: pd.DataFrame, migration: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    s = stock.copy().sort_values(["symbol", "time"])
    close = s[["symbol", "time", "close"]].copy()
    rows = []
    for _, r in migration.iterrows():
        px = close[(close["symbol"] == r["Ticker"]) & (close["time"] >= r["Date"])].sort_values("time")
        if px.empty:
            continue
        base = px.iloc[0]["close"]
        out = r.to_dict()
        for h in horizons:
            if len(px) > h:
                out[f"ForwardReturn{h}D"] = px.iloc[h]["close"] / base - 1
            else:
                out[f"ForwardReturn{h}D"] = np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_forward_returns(forward: pd.DataFrame) -> pd.DataFrame:
    if forward.empty:
        return pd.DataFrame()
    events = forward[forward["Migration"]].copy()
    cols = [c for c in forward.columns if c.startswith("ForwardReturn")]
    rows = []
    for transition, g in events.groupby("Transition", dropna=True):
        row = {"Transition": transition, "Events": len(g)}
        for c in cols:
            row[f"{c}_Mean"] = g[c].mean()
            row[f"{c}_Median"] = g[c].median()
            row[f"{c}_PositiveRate"] = (g[c] > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Events", ascending=False)
