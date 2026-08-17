from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

try:
    from vnstock import Market
except ImportError:
    from vnstock.ui import Market

try:
    from vnstock import register_user
except ImportError:
    register_user = None

from membership import symbols_for_period

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)
REQUEST_INTERVAL_SECONDS = 1.35
MAX_FETCH_RETRIES = 4
_last_request_at = 0.0


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.csv"


def _read_cache(symbol: str) -> pd.DataFrame | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["time"])
    return None if df.empty else df


def _save_cache(symbol: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time", keep="last")
    df.to_csv(_cache_path(symbol), index=False)


def _configure_vnstock(api_key: str | None = None) -> None:
    key = (api_key or os.getenv("VNSTOCK_API_KEY", "")).strip()
    if not key:
        raise ValueError("Thiếu VNstock API Key.")
    os.environ["VNSTOCK_API_KEY"] = key
    if register_user is not None:
        try:
            register_user(api_key=key)
        except TypeError:
            pass


def _rate_limit_gate() -> None:
    global _last_request_at
    wait = REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _call_with_retry(fetch_fn, symbol: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, MAX_FETCH_RETRIES + 1):
        _rate_limit_gate()
        try:
            return fetch_fn()
        except Exception as exc:
            last_error = exc
            if attempt < MAX_FETCH_RETRIES:
                time.sleep(min(20.0, 2.0 ** (attempt - 1) * 2.0))
    error_type = type(last_error).__name__ if last_error is not None else "UnknownError"
    error_text = str(last_error).replace(os.getenv("VNSTOCK_API_KEY", ""), "[API_KEY]")[:300] if last_error else ""
    raise RuntimeError(f"VNstock lỗi với {symbol} sau {MAX_FETCH_RETRIES} lần thử. Loại lỗi: {error_type}. Chi tiết: {error_text}") from last_error


def _normalize_ohlcv(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError(f"Không có dữ liệu cho {symbol}")
    df = raw.copy().rename(columns={"date": "time", "Date": "time"})
    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột {missing} cho {symbol}")
    df = df[required].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["time", "close"], inplace=True)
    return df.assign(symbol=symbol)


def _fetch_equity(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = _call_with_retry(lambda: Market().equity(symbol).ohlcv(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")), symbol)
    return _normalize_ohlcv(raw, symbol)


def _fetch_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    raw = _call_with_retry(lambda: Market().index("VNINDEX").ohlcv(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")), "VNINDEX")
    return _normalize_ohlcv(raw, "VNINDEX")


def _load_symbol(symbol: str, fetch_start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cached = _read_cache(symbol)
    if cached is None:
        cached = _fetch_equity(symbol, fetch_start, end)
    else:
        cached["time"] = pd.to_datetime(cached["time"])
        if fetch_start < cached["time"].min():
            cached = pd.concat([cached, _fetch_equity(symbol, fetch_start, cached["time"].min() - pd.Timedelta(days=1))], ignore_index=True)
        if end > cached["time"].max():
            cached = pd.concat([cached, _fetch_equity(symbol, cached["time"].max() + pd.Timedelta(days=1), end)], ignore_index=True)
    _save_cache(symbol, cached)
    cached = cached.sort_values("time").drop_duplicates("time", keep="last")
    return cached[(cached["time"] >= fetch_start) & (cached["time"] <= end)].copy()


def _load_index(fetch_start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cached = _read_cache("VNINDEX")
    if cached is None:
        cached = _fetch_index(fetch_start, end)
    else:
        cached["time"] = pd.to_datetime(cached["time"])
        if fetch_start < cached["time"].min():
            cached = pd.concat([cached, _fetch_index(fetch_start, cached["time"].min() - pd.Timedelta(days=1))], ignore_index=True)
        if end > cached["time"].max():
            cached = pd.concat([cached, _fetch_index(cached["time"].max() + pd.Timedelta(days=1), end)], ignore_index=True)
    _save_cache("VNINDEX", cached)
    cached = cached.sort_values("time").drop_duplicates("time", keep="last")
    return cached[(cached["time"] >= fetch_start) & (cached["time"] <= end)].copy()


def load_market_data(start: pd.Timestamp, end: pd.Timestamp, warmup: int = 80, api_key: str | None = None):
    _configure_vnstock(api_key)
    fetch_start = start - pd.Timedelta(days=int(warmup * 1.7))
    symbols = symbols_for_period(fetch_start, end)
    stock = pd.concat([_load_symbol(symbol, fetch_start, end) for symbol in symbols], ignore_index=True)
    vn = _load_index(fetch_start, end)
    for df in (stock, vn):
        df["time"] = pd.to_datetime(df["time"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["time", "close"], inplace=True)
        df.sort_values(["symbol", "time"], inplace=True)
    return stock, vn
