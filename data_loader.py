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
    rename = {"date": "time", "Date": "time"}
    df = df.rename(columns=rename)
    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột {missing} cho {symbol}")
    return df[required].assign(symbol=symbol)


def _load_symbol(symbol: str, fetch_start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cached = _read_cache(symbol)
    if cached is not None:
        cached["time"] = pd.to_datetime(cached["time"])
        if cached["time"].min() <= fetch_start and cached["time"].max() >= end:
            return cached[(cached["time"] >= fetch_start) & (cached["time"] <= end)].copy()
    df = _fetch_ohlcv(symbol, fetch_start, end)
    _save_cache(symbol, df)
    time.sleep(0.4)
    return df


def load_market_data(start: pd.Timestamp, end: pd.Timestamp, warmup: int = 80):
    fetch_start = start - pd.Timedelta(days=int(warmup * 1.7))
    stock_frames = []
    for symbol in VN30:
        stock_frames.append(_load_symbol(symbol, fetch_start, end))
    stock = pd.concat(stock_frames, ignore_index=True)

    vn_cached = _read_cache("VNINDEX")
    if vn_cached is not None and vn_cached["time"].min() <= fetch_start and vn_cached["time"].max() >= end:
        vn = vn_cached[(vn_cached["time"] >= fetch_start) & (vn_cached["time"] <= end)].copy()
    else:
        index = Vnstock().index(symbol="VNINDEX", source="VCI")
        raw = index.quote.history(start=fetch_start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        vn = raw.rename(columns={"date": "time", "Date": "time"}).copy()
        vn = vn[["time", "open", "high", "low", "close", "volume"]].assign(symbol="VNINDEX")
        _save_cache("VNINDEX", vn)

    for df in (stock, vn):
        df["time"] = pd.to_datetime(df["time"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["time", "close"], inplace=True)
        df.sort_values(["symbol", "time"], inplace=True)

    return stock, vn
