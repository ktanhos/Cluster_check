from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEATURES, VN30


def build_feature_panel(stock: pd.DataFrame, vn: pd.DataFrame) -> pd.DataFrame:
    s = stock.copy()
    i = vn.copy()
    s["Ret1D"] = s.groupby("symbol")["close"].pct_change()
    i["Ret1D"] = i["close"].pct_change()

    frames = []
    vn_small = i[["time", "close", "Ret1D"]].rename(
        columns={"close": "vn_close", "Ret1D": "vn_ret"}
    )

    for ticker in VN30:
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

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["time", "Ticker"]).reset_index(drop=True)
    return panel


def get_feature_snapshot(panel: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    return panel[panel["time"] == pd.Timestamp(date)].dropna(subset=FEATURES).copy()
