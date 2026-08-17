from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

# Use the Community package, but bypass Unified UI for the primary OHLCV
# request. Direct KBS/VCI adapters give us a controlled fallback path when a
# single symbol causes Unified UI to wait or fail on Streamlit Cloud.
try:
    from vnstock import Vnstock
except ImportError:
    Vnstock = None

try:
    from vnstock import Market
except ImportError:
    try:
        from vnstock.ui import Market
    except ImportError:
        Market = None

from membership import symbols_for_period

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)
REQUEST_INTERVAL_SECONDS = 1.1
_last_request_at = 0.0


def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.csv"


def _read_cache(symbol: str) -> pd.DataFrame | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, parse_dates=["time"])
        if df.empty:
            return None
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        return df.dropna(subset=["time"]).sort_values("time")
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
    if cached["time"].min() <= start and cached["time"].max() >= end:
        return "Đã đủ cache"
    return "Cache chưa đủ, cần cập nhật"


def _configure_vnstock(api_key: str | None = None) -> None:
    key = (api_key or os.getenv("VNSTOCK_API_KEY", "")).strip()
    if not key:
        raise ValueError("Thiếu VNstock API Key.")
    os.environ["VNSTOCK_API_KEY"] = key


def _rate_limit_gate() -> None:
    global _last_request_at
    wait = REQUEST_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _safe_error(exc: Exception) -> str:
    key = os.getenv("VNSTOCK_API_KEY", "")
    return str(exc).replace(key, "[API_KEY]")[:500]


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


def _legacy_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp, source: str) -> pd.DataFrame:
    if Vnstock is None:
        raise RuntimeError("Phiên bản vnstock hiện tại không có Vnstock wrapper.")
    stock = Vnstock().stock(symbol=symbol, source=source)
    raw = stock.quote.history(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1D",
    )
    return _normalize_ohlcv(raw, symbol)


def _market_history(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if Market is None:
        raise RuntimeError("Không tìm thấy Market trong vnstock.")
    raw = Market().equity(symbol).ohlcv(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1D",
    )
    return _normalize_ohlcv(raw, symbol)


def _fetch_equity(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch one symbol with direct provider fallback.

    KBS and VCI are both documented equity market sources. We try the direct
    adapters first, then Unified UI only as a last resort. This prevents one
    problematic Unified UI route from being the only path for the entire data
    update.
    """
    errors = []
    for source in ("KBS", "VCI"):
        _rate_limit_gate()
        try:
            return _legacy_history(symbol, start, end, source)
        except Exception as exc:
            errors.append(f"{source}: {_safe_error(exc)}")

    if Market is not None:
        _rate_limit_gate()
        try:
            return _market_history(symbol, start, end)
        except Exception as exc:
            errors.append(f"Unified: {_safe_error(exc)}")

    raise RuntimeError(f"VNstock không lấy được {symbol}: " + " | ".join(errors))


def _fetch_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    errors = []
    if Vnstock is not None:
        for source in ("KBS", "VCI"):
            _rate_limit_gate()
            try:
                stock = Vnstock().stock(symbol="VNINDEX", source=source)
                raw = stock.quote.history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="1D",
                )
                return _normalize_ohlcv(raw, "VNINDEX")
            except Exception as exc:
                errors.append(f"{source}: {_safe_error(exc)}")

    if Market is not None:
        _rate_limit_gate()
        try:
            raw = Market().index("VNINDEX").ohlcv(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval="1D",
            )
            return _normalize_ohlcv(raw, "VNINDEX")
        except Exception as exc:
            errors.append(f"Unified: {_safe_error(exc)}")

    raise RuntimeError("VNstock không lấy được VNINDEX: " + " | ".join(errors))


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
    result = cached[(cached["time"] >= fetch_start) & (cached["time"] <= end)].copy()
    if result.empty:
        raise RuntimeError("Không có dữ liệu VNINDEX trong khoảng đã chọn.")
    return result, fetched


def load_market_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    warmup: int = 80,
    api_key: str | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    force_refresh: bool = False,
):
    _configure_vnstock(api_key)
    fetch_start = start - pd.Timedelta(days=int(warmup * 1.7))
    symbols = symbols_for_period(fetch_start, end)
    total = len(symbols) + 1
    stock_frames = []
    failed = []

    for i, symbol in enumerate(symbols, 1):
        status = "Đang tải phần còn thiếu" if not force_refresh else "Đang tải lại từ VNstock"
        if progress_callback:
            progress_callback(i - 1, total, symbol, status)
        try:
            frame, fetched = _load_symbol(symbol, fetch_start, end, force_refresh=force_refresh)
            stock_frames.append(frame)
            if progress_callback:
                progress_callback(i, total, symbol, "Đã tải API" if fetched else "Dùng cache")
        except Exception as exc:
            failed.append((symbol, str(exc)))
            if progress_callback:
                progress_callback(i, total, symbol, f"LỖI: {str(exc)[:350]}")
            continue

    if failed:
        details = "; ".join(f"{s}: {e}" for s, e in failed)
        raise RuntimeError(f"Không tải được {len(failed)} mã: {details}")

    vn, fetched = _load_index(fetch_start, end, force_refresh=force_refresh)
    if progress_callback:
        progress_callback(total, total, "VNINDEX", "Đã tải API" if fetched else "Dùng cache")

    stock = pd.concat(stock_frames, ignore_index=True)
    stock["time"] = pd.to_datetime(stock["time"])
    vn["time"] = pd.to_datetime(vn["time"])
    for df in (stock, vn):
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["time", "close"], inplace=True)
    stock.sort_values(["symbol", "time"], inplace=True)
    vn.sort_values(["symbol", "time"], inplace=True)
    return stock, vn
