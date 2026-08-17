from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

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
REQUEST_INTERVAL_SECONDS = 1.25
_last_request_at = 0.0
_market = None


def _get_market():
    global _market
    if _market is None:
        _market = Market()
    return _market


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.csv"


def _read_cache(symbol: str) -> pd.DataFrame | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["time"])
        return None if df.empty else df
    except Exception:
        return None


def _save_cache(symbol: str, df: pd.DataFrame) -> None:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates("time", keep="last")
    df.to_csv(_cache_path(symbol), index=False)


def cache_status(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    cached = _read_cache(symbol)
    if cached is None:
        return "Chưa có cache"
    cached["time"] = pd.to_datetime(cached["time"])
    if cached["time"].min() <= start and cached["time"].max() >= end:
        return "Đã đủ cache"
    return "Cache chưa đủ, cần cập nhật"


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


def _safe_error(exc: Exception) -> str:
    key = os.getenv("VNSTOCK_API_KEY", "")
    return str(exc).replace(key, "[API_KEY]")[:700]


def _call(fetch_fn, symbol: str) -> pd.DataFrame:
    # Do not add another retry layer around Vnstock. The library itself may
    # retry internally; wrapping it again can make a single symbol appear
    # frozen for several minutes. We deliberately fail fast here so the app
    # can report the exact symbol and continue with the remaining cache/data.
    _rate_limit_gate()
    started = time.monotonic()
    try:
        result = fetch_fn()
    except Exception as exc:
        elapsed = time.monotonic() - started
        raise RuntimeError(
            f"VNstock lỗi khi tải {symbol} sau {elapsed:.1f} giây. "
            f"Loại lỗi: {type(exc).__name__}. Chi tiết: {_safe_error(exc)}"
        ) from exc
    if result is None or getattr(result, "empty", False):
        elapsed = time.monotonic() - started
        raise RuntimeError(f"VNstock trả dữ liệu rỗng cho {symbol} sau {elapsed:.1f} giây.")
    return result


def _required_count(start: pd.Timestamp, end: pd.Timestamp) -> int:
    # ohlcv supports count when start/end are omitted. This avoids some
    # provider-side date-range handling issues and fetches one contiguous
    # block per symbol. Add a safety margin for weekends and holidays.
    calendar_days = max(1, (pd.Timestamp(end) - pd.Timestamp(start)).days)
    return max(250, int(calendar_days * 1.7) + 30)


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
    market = _get_market()
    count = _required_count(start, end)
    raw = _call(
        lambda: market.equity(symbol).ohlcv(interval="1D", count=count),
        symbol,
    )
    df = _normalize_ohlcv(raw, symbol)
    return df[(df["time"] >= start) & (df["time"] <= end)].copy()


def _fetch_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    market = _get_market()
    count = _required_count(start, end)
    raw = _call(
        lambda: market.index("VNINDEX").ohlcv(interval="1D", count=count),
        "VNINDEX",
    )
    df = _normalize_ohlcv(raw, "VNINDEX")
    return df[(df["time"] >= start) & (df["time"] <= end)].copy()


def _load_symbol(symbol: str, fetch_start: pd.Timestamp, end: pd.Timestamp, force_refresh: bool = False) -> tuple[pd.DataFrame, bool]:
    cached = None if force_refresh else _read_cache(symbol)
    fetched = False
    if cached is None:
        cached = _fetch_equity(symbol, fetch_start, end)
        fetched = True
    else:
        cached["time"] = pd.to_datetime(cached["time"])
        if fetch_start < cached["time"].min():
            left = _fetch_equity(symbol, fetch_start, cached["time"].min() - pd.Timedelta(days=1))
            cached = pd.concat([cached, left], ignore_index=True)
            fetched = True
        if end > cached["time"].max():
            right = _fetch_equity(symbol, cached["time"].max() + pd.Timedelta(days=1), end)
            cached = pd.concat([cached, right], ignore_index=True)
            fetched = True
    _save_cache(symbol, cached)
    cached = cached.sort_values("time").drop_duplicates("time", keep="last")
    result = cached[(cached["time"] >= fetch_start) & (cached["time"] <= end)].copy()
    if result.empty:
        raise RuntimeError(f"Không có dữ liệu hợp lệ cho {symbol} trong khoảng đã chọn.")
    return result, fetched


def _load_index(fetch_start: pd.Timestamp, end: pd.Timestamp, force_refresh: bool = False) -> tuple[pd.DataFrame, bool]:
    cached = None if force_refresh else _read_cache("VNINDEX")
    fetched = False
    if cached is None:
        cached = _fetch_index(fetch_start, end)
        fetched = True
    else:
        cached["time"] = pd.to_datetime(cached["time"])
        if fetch_start < cached["time"].min():
            left = _fetch_index(fetch_start, cached["time"].min() - pd.Timedelta(days=1))
            cached = pd.concat([cached, left], ignore_index=True)
            fetched = True
        if end > cached["time"].max():
            right = _fetch_index(cached["time"].max() + pd.Timedelta(days=1), end)
            cached = pd.concat([cached, right], ignore_index=True)
            fetched = True
    _save_cache("VNINDEX", cached)
    cached = cached.sort_values("time").drop_duplicates("time", keep="last")
    result = cached[(cached["time"] >= fetch_start) & (cached["time"] <= end)].copy()
    if result.empty:
        raise RuntimeError("Không có dữ liệu VNINDEX trong khoảng đã chọn.")
    return result, fetched


def load_market_data(start: pd.Timestamp, end: pd.Timestamp, warmup: int = 80, api_key: str | None = None, progress_callback: Callable[[int, int, str, str], None] | None = None, force_refresh: bool = False):
    _configure_vnstock(api_key)
    fetch_start = start - pd.Timedelta(days=int(warmup * 1.7))
    symbols = symbols_for_period(fetch_start, end)
    total = len(symbols) + 1
    stock_frames = []
    failed = []

    for i, symbol in enumerate(symbols, 1):
        status = cache_status(symbol, fetch_start, end) if not force_refresh else "Bỏ qua cache, đang gọi VNstock"
        if progress_callback:
            progress_callback(i - 1, total, symbol, status)
            progress_callback(i - 1, total, symbol, "Đang gọi VNstock cho mã này...")
        try:
            frame, fetched = _load_symbol(symbol, fetch_start, end, force_refresh=force_refresh)
            stock_frames.append(frame)
            if progress_callback:
                progress_callback(i, total, symbol, "Đã tải API" if fetched else "Dùng cache")
        except Exception as exc:
            failed.append((symbol, str(exc)))
            if progress_callback:
                progress_callback(i, total, symbol, f"LỖI: {str(exc)[:350]}")

    if failed:
        details = "; ".join(f"{s}: {e}" for s, e in failed)
        raise RuntimeError(f"Không tải được {len(failed)} mã: {details}")

    vn, fetched = _load_index(fetch_start, end, force_refresh=force_refresh)
    if progress_callback:
        progress_callback(total, total, "VNINDEX", "Đã tải API" if fetched else "Dùng cache")

    stock = pd.concat(stock_frames, ignore_index=True)
    for df in (stock, vn):
        df["time"] = pd.to_datetime(df["time"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["time", "close"], inplace=True)
        df.sort_values(["symbol", "time"], inplace=True)
    return stock, vn
