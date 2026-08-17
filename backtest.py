from __future__ import annotations

import numpy as np
import pandas as pd

from membership import membership_at


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """Normalize either a migration table or rolling observations.

    The app historically passed rolling_result to calculate_forward_returns.
    Convert that input into an event table here so the analysis remains backward
    compatible and never depends on a fragile column existing upstream.
    """
    if events is None or events.empty:
        return pd.DataFrame()
    x = events.copy()
    if "Ticker" not in x.columns or "Date" not in x.columns:
        return pd.DataFrame()
    x["Date"] = pd.to_datetime(x["Date"], errors="coerce")
    x = x.dropna(subset=["Date", "Ticker"]).sort_values(["Ticker", "Date"]).copy()
    if "Migration" not in x.columns:
        if "Cluster" in x.columns:
            previous = x.groupby("Ticker")["Cluster"].shift(1)
            x["Migration"] = previous.notna() & previous.ne(x["Cluster"])
            x["Transition"] = np.where(
                x["Migration"],
                previous.astype("Int64").astype(str) + " → " + x["Cluster"].astype("Int64").astype(str),
                "Stable",
            )
        else:
            x["Migration"] = False
    if "Transition" not in x.columns:
        x["Transition"] = "Stable"
    return x


def calculate_forward_returns(stock: pd.DataFrame, events: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    """Calculate future returns after observations, including migration events."""
    events = _prepare_events(events)
    if events.empty:
        return pd.DataFrame()

    s = stock.copy().sort_values(["symbol", "time"])
    s["symbol"] = s["symbol"].astype(str)
    close = s[["symbol", "time", "close"]].copy()
    rows = []
    for _, r in events.iterrows():
        event_date = pd.Timestamp(r["Date"])
        ticker = str(r["Ticker"])
        if ticker not in membership_at(event_date):
            continue
        px = close[(close["symbol"] == ticker) & (close["time"] >= event_date)].sort_values("time")
        if px.empty:
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
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def summarize_forward_returns(forward: pd.DataFrame) -> pd.DataFrame:
    """Summarize future returns for actual migration events without KeyError."""
    if forward is None or forward.empty:
        return pd.DataFrame()
    x = _prepare_events(forward)
    if x.empty or "Migration" not in x.columns:
        return pd.DataFrame()
    events = x[x["Migration"].fillna(False).astype(bool)].copy()
    if events.empty:
        return pd.DataFrame(columns=["Transition", "Events"])

    cols = [c for c in events.columns if c.startswith("ForwardReturn")]
    rows = []
    for transition, g in events.groupby("Transition", dropna=True):
        row = {"Transition": transition, "Events": len(g)}
        for c in cols:
            horizon = c.replace("ForwardReturn", "")
            membership_col = f"ConstituentThrough{horizon}"
            valid = g[g[membership_col].fillna(False)] if membership_col in g.columns else g
            valid = valid[valid[c].notna()]
            row[f"{c}_EventsInBasket"] = len(valid)
            row[f"{c}_Mean"] = valid[c].mean() if len(valid) else np.nan
            row[f"{c}_Median"] = valid[c].median() if len(valid) else np.nan
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
    for _, cur in current.iterrows():
        state_id = int(cur["Cluster"])
        state_name = cur.get("ClusterLabel", f"Nhóm {state_id + 1}")
        transition = cur.get("Transition", "Stable")
        state_hist = hist[(hist["Cluster"] == state_id) & (hist["Date"] < latest_date)].copy() if "Cluster" in hist.columns else pd.DataFrame()
        transition_hist = hist[(hist["Transition"] == transition) & (hist["Date"] < latest_date)].copy() if transition != "Stable" else pd.DataFrame()

        out = {
            "Ticker": cur["Ticker"],
            "CurrentGroup": state_name,
            "CurrentStatus": cur.get("MigrationType", "Ổn định"),
            "CurrentTransition": transition,
            "Confidence": cur.get("AssignmentConfidence", np.nan),
            "HistoricalObservations": len(state_hist),
            "TransitionObservations": len(transition_hist),
        }
        state_scores = []
        transition_scores = []
        for h in horizons:
            c = f"ForwardReturn{h}D"
            state_valid = state_hist[c].dropna() if c in state_hist.columns else pd.Series(dtype=float)
            trans_valid = transition_hist[c].dropna() if c in transition_hist.columns else pd.Series(dtype=float)
            out[f"HistoricalMean{h}D"] = state_valid.mean() if len(state_valid) else np.nan
            out[f"HistoricalMedian{h}D"] = state_valid.median() if len(state_valid) else np.nan
            out[f"PositiveRate{h}D"] = (state_valid > 0).mean() if len(state_valid) else np.nan
            out[f"TransitionMean{h}D"] = trans_valid.mean() if len(trans_valid) else np.nan
            state_scores.append(float(state_valid.mean()) if len(state_valid) >= min_history else np.nan)
            transition_scores.append(float(trans_valid.mean()) if len(trans_valid) >= min_history else np.nan)

        state_available = [x for x in state_scores if pd.notna(x)]
        trans_available = [x for x in transition_scores if pd.notna(x)]
        state_score = float(np.mean(state_available)) if state_available else np.nan
        transition_score = float(np.mean(trans_available)) if trans_available else np.nan

        if pd.notna(transition_score) and len(transition_hist) >= min_history:
            reference_score = transition_score
            forecast_basis = "Cùng kiểu chuyển nhóm"
        elif pd.notna(state_score):
            reference_score = state_score
            forecast_basis = "Cùng nhóm hành vi"
        else:
            reference_score = np.nan
            forecast_basis = "Chưa đủ dữ liệu lịch sử"

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
