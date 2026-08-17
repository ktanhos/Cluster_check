from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from vnstock import Vnstock

from config import VN30

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.csv"


def _read_cache(symbol: str) -> pd.DataFrame | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    if df.empty:
        return None
    return df


def _save_cache(symbol: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time", keep="last")
    df.to_csv(_cache_path(symbol), index=False)


def _fetch_ohlcv(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    stock = Vnstock().stock(symbol=symbol, source="VCI")
    raw = stock.quote.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    df = raw.copy()
    if df.empty:
        raise ValueError(f"Không có dữ liệu cho {symbol}")
    df = df.rename(columns={"date": "time", "Date": "time"})
    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột {missing} cho {symbol}")
    return df[required].assign(symbol=symbol)


def _load_symbol(symbol: str, fetch_start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cached = _read_cache(symbol)
    if cached is None:
        fetched = _fetch_ohlcv(symbol, fetch_start, end)
        _save_cache(symbol, fetched)
        time.sleep(0.4)
        cached = _read_cache(symbol)
    else:
        cached["time"] = pd.to_datetime(cached["time"])
        missing_left = fetch_start < cached["time"].min()
        missing_right = end > cached["time"].max()

        if missing_left:
            left = _fetch_ohlcv(symbol, fetch_start, cached["time"].min())
            cached = pd.concat([cached, left], ignore_index=True)
            time.sleep(0.4)
        if missing_right:
            right = _fetch_ohlcv(symbol, cached["time"].max(), end)
            cached = pd.concat([cached, right], ignore_index=True)
            time.sleep(0.4)
        _save_cache(symbol, cached)

    cached = cached.sort_values("time").drop_duplicates("time", keep="last")
    return cached[(cached["time"] >= fetch_start) & (cached["time"] <= end)].copy()


def load_market_data(start: pd.Timestamp, end: pd.Timestamp, warmup: int = 80):
    fetch_start = start - pd.Timedelta(days=int(warmup * 1.7))
    stock_frames = []
    for symbol in VN30:
        stock_frames.append(_load_symbol(symbol, fetch_start, end))
    stock = pd.concat(stock_frames, ignore_index=True)

    vn_cached = _read_cache("VNINDEX")
    if vn_cached is None:
        index = Vnstock().index(symbol="VNINDEX", source="VCI")
        raw = index.quote.history(
            start=fetch_start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
        vn = raw.rename(columns={"date": "time", "Date": "time"}).copy()
        vn = vn[["time", "open", "high", "low", "close", "volume"]].assign(symbol="VNINDEX")
        _save_cache("VNINDEX", vn)
        time.sleep(0.4)
    else:
        vn_cached["time"] = pd.to_datetime(vn_cached["time"])
        if fetch_start < vn_cached["time"].min():
            left = Vnstock().index(symbol="VNINDEX", source="VCI").quote.history(
                start=fetch_start.strftime("%Y-%m-%d"),
                end=vn_cached["time"].min().strftime("%Y-%m-%d"),
            )
            left = left.rename(columns={"date": "time", "Date": "time"})
            left = left[["time", "open", "high", "low", "close", "volume"]].assign(symbol="VNINDEX")
            vn_cached = pd.concat([vn_cached, left], ignore_index=True)
            time.sleep(0.4)
        if end > vn_cached["time"].max():
            right = Vnstock().index(symbol="VNINDEX", source="VCI").quote.history(
                start=vn_cached["time"].max().strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
            )
            right = right.rename(columns={"date": "time", "Date": "time"})
            right = right[["time", "open", "high", "low", "close", "volume"]].assign(symbol="VNINDEX")
            vn_cached = pd.concat([vn_cached, right], ignore_index=True)
            time.sleep(0.4)
        _save_cache("VNINDEX", vn_cached)
        vn = vn_cached

    for df in (stock, vn):
        df["time"] = pd.to_datetime(df["time"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["time", "close"], inplace=True)
        df.sort_values(["symbol", "time"], inplace=True)

    vn = vn[(vn["time"] >= fetch_start) & (vn["time"] <= end)].copy()
    return stock, vn
