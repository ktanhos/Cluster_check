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
            valid = g[g[membership_col]] if membership_col in g else g
            row[f"{c}_EventsInBasket"] = len(valid)
            row[f"{c}_Mean"] = valid[c].mean()
            row[f"{c}_Median"] = valid[c].median()
            row[f"{c}_PositiveRate"] = (valid[c] > 0).mean() if len(valid) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Events", ascending=False)


def build_reference_forecast(stock: pd.DataFrame, rolling_result: pd.DataFrame, horizons=(5, 10, 20), min_history: int = 5) -> pd.DataFrame:
    """Descriptive historical conditional-return ranking, not a predictive model."""
    if rolling_result is None or rolling_result.empty:
        return pd.DataFrame()
    hist = calculate_forward_returns(stock, rolling_result, horizons=horizons)
    if hist.empty:
        return pd.DataFrame()
    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    latest_date = hist["Date"].max()

    current = rolling_result.copy()
    current["Date"] = pd.to_datetime(current["Date"], errors="coerce")
    current_date = current["Date"].max()
    current = current[current["Date"] == current_date].copy()
    current = current.sort_values("CentroidDistance", na_position="last").drop_duplicates("Ticker")
    current = current[current["Ticker"].isin(set(membership_at(current_date)))].copy()

    rows = []
    return_cols = [f"ForwardReturn{h}D" for h in horizons]
    for _, cur in current.iterrows():
        state_id = int(cur["Cluster"])
        state_name = cur.get("ClusterLabel", f"Nhóm {state_id + 1}")
        migration_type = cur.get("MigrationType", "Stable")
        transition = cur.get("Transition", "Stable")

        # Stable state IDs are safer than human labels because labels are generated
        # from each window's profile and may change wording while IDs are aligned.
        state_hist = hist[(hist["Cluster"] == state_id) & (hist["Date"] < latest_date)].copy()
        transition_hist = hist[(hist["Transition"] == transition) & (hist["Date"] < latest_date)].copy() if transition != "Stable" else pd.DataFrame()

        out = {
            "Ticker": cur["Ticker"],
            "CurrentGroup": state_name,
            "CurrentStatus": migration_type,
            "CurrentTransition": transition,
            "Confidence": cur.get("AssignmentConfidence", np.nan),
            "HistoricalObservations": len(state_hist),
            "TransitionObservations": len(transition_hist),
        }
        state_scores = []
        transition_scores = []
        for h, c in zip(horizons, return_cols):
            state_valid = state_hist[c].dropna()
            trans_valid = transition_hist[c].dropna()
            out[f"HistoricalMean{h}D"] = state_valid.mean() if len(state_valid) else np.nan
            out[f"HistoricalMedian{h}D"] = state_valid.median() if len(state_valid) else np.nan
            out[f"PositiveRate{h}D"] = (state_valid > 0).mean() if len(state_valid) else np.nan
            out[f"TransitionMean{h}D"] = trans_valid.mean() if len(trans_valid) else np.nan
            if len(state_valid) >= min_history:
                state_scores.append(float(state_valid.mean()))
            else:
                state_scores.append(np.nan)
            if len(trans_valid) >= min_history:
                transition_scores.append(float(trans_valid.mean()))
            else:
                transition_scores.append(np.nan)

        state_available = [x for x in state_scores if pd.notna(x)]
        trans_available = [x for x in transition_scores if pd.notna(x)]
        state_score = float(np.mean(state_available)) if state_available else np.nan
        transition_score = float(np.mean(trans_available)) if trans_available else np.nan

        # If a current stock has enough historical examples of the same transition,
        # prefer that more specific evidence. Otherwise fall back to its current state.
        if pd.notna(transition_score) and len(transition_hist) >= min_history:
            reference_score = transition_score
            forecast_basis = "Lịch sử cùng kiểu chuyển nhóm"
        elif pd.notna(state_score):
            reference_score = state_score
            forecast_basis = "Lịch sử cùng nhóm"
        else:
            reference_score = np.nan
            forecast_basis = "Chưa đủ dữ liệu"

        out["StateScore"] = state_score
        out["TransitionScore"] = transition_score
        out["ReferenceScore"] = reference_score
        out["ForecastBasis"] = forecast_basis
        out["EnoughHistory"] = len(state_hist) >= min_history
        rows.append(out)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["Rank"] = result["ReferenceScore"].rank(method="min", ascending=False).astype("Int64")
    return result.sort_values(["EnoughHistory", "ReferenceScore"], ascending=[False, False])
