from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURES
from membership import membership_at

Z_FEATURES = [f"Z_{f}" for f in FEATURES]


def _cross_sectional_zscore(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    # Normalize only against the actual VN30 cross-section on each date.
    # Former and future constituents must not affect today's relative scores.
    active_cache: dict[pd.Timestamp, set[str]] = {}
    active_mask = []
    for d, ticker in zip(out["time"], out["Ticker"]):
        d = pd.Timestamp(d)
        active_cache.setdefault(d, set(membership_at(d)))
        active_mask.append(ticker in active_cache[d])
    active_mask = pd.Series(active_mask, index=out.index)

    for col in columns:
        active_values = out.loc[active_mask, ["time", col]].copy()
        mean = active_values.groupby("time")[col].transform("mean")
        std = active_values.groupby("time")[col].transform("std")
        active_values[f"Z_{col}"] = (active_values[col] - mean) / std.replace(0, np.nan)
        out[f"Z_{col}"] = np.nan
        out.loc[active_mask, f"Z_{col}"] = active_values[f"Z_{col}"].to_numpy()
    return out


def build_feature_panel(stock: pd.DataFrame, vn: pd.DataFrame) -> pd.DataFrame:
    s = stock.copy()
    i = vn.copy()
    s["Ret1D"] = s.groupby("symbol")["close"].pct_change()
    i["Ret1D"] = i["close"].pct_change()
    vn_small = i[["time", "close", "Ret1D"]].rename(columns={"close": "vn_close", "Ret1D": "vn_ret"})

    frames = []
    for ticker in sorted(s["symbol"].dropna().unique()):
        x = s[s["symbol"] == ticker].copy().sort_values("time")
        x = x.merge(vn_small, on="time", how="inner")
        x["Return20"] = x["close"].pct_change(20)
        x["Volatility20"] = x["Ret1D"].rolling(20).std()
        cov = x["Ret1D"].rolling(60).cov(x["vn_ret"])
        var = x["vn_ret"].rolling(60).var()
        x["Beta60"] = cov / var.replace(0, np.nan)
        x["RS20"] = x["Return20"] - x["vn_close"].pct_change(20)
        vmean = x["volume"].rolling(20).mean()
        vstd = x["volume"].rolling(20).std()
        x["VolumeZ20"] = (x["volume"] - vmean) / vstd.replace(0, np.nan)
        corr = x["Ret1D"].rolling(60).corr(x["vn_ret"])
        x["DistanceVN60"] = 1 - corr
        x["Ticker"] = ticker
        frames.append(x[["time", "Ticker"] + FEATURES])

    if not frames:
        raise ValueError("Không có dữ liệu cổ phiếu để tính feature.")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["time", "Ticker"]).reset_index(drop=True)
    panel = _cross_sectional_zscore(panel, FEATURES)
    return panel


def get_feature_snapshot(panel: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    return panel[panel["time"] == pd.Timestamp(date)].dropna(subset=FEATURES + Z_FEATURES).copy()
