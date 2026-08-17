from __future__ import annotations

import numpy as np
import pandas as pd

from membership import membership_at


def calculate_forward_returns(stock: pd.DataFrame, migration: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    s = stock.copy().sort_values(["symbol", "time"])
    close = s[["symbol", "time", "close"]].copy()
    rows = []
    for _, r in migration.iterrows():
        event_date = pd.Timestamp(r["Date"])
        ticker = r["Ticker"]
        active_at_event = ticker in membership_at(event_date)
        px = close[(close["symbol"] == ticker) & (close["time"] >= event_date)].sort_values("time")
        if px.empty or not active_at_event:
            continue
        base = px.iloc[0]["close"]
        out = r.to_dict()
        out["ConstituentAtEvent"] = True
        for h in horizons:
            if len(px) > h:
                horizon_date = pd.Timestamp(px.iloc[h]["time"])
                out[f"ConstituentThrough{h}D"] = ticker in membership_at(horizon_date)
                out[f"ForwardReturn{h}D"] = px.iloc[h]["close"] / base - 1
            else:
                out[f"ConstituentThrough{h}D"] = False
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
            horizon = c.replace("ForwardReturn", "")
            membership_col = f"ConstituentThrough{horizon}"
            if membership_col in g:
                valid = g[g[membership_col]]
            else:
                valid = g
            row[f"{c}_EventsInBasket"] = len(valid)
            row[f"{c}_Mean"] = valid[c].mean()
            row[f"{c}_Median"] = valid[c].median()
            row[f"{c}_PositiveRate"] = (valid[c] > 0).mean() if len(valid) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Events", ascending=False)
