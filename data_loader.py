from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from membership import symbols_for_period

CACHE_DIR = Path("data_cache")
CACHE_DIR.mkdir(exist_ok=True)
REQUEST_INTERVAL_SECONDS = 1.1
REQUEST_TIMEOUT_SECONDS = 25


class WorkerTimeout(Exception):
    pass


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


def _worker_equity(queue, symbol: str, start: str, end: str, source: str, api_key: str):
    """Run VNstock in a real child process.

    Streamlit executes the user script in a worker thread rather than the
    interpreter main thread. VNstock's internal timeout implementation uses
    signal.signal, which Python only permits in the main thread. A child
    process gives VNstock an actual interpreter main thread and therefore
    avoids: "signal only works in main thread of the main interpreter".
    """
    try:
        os.environ["VNSTOCK_API_KEY"] = api_key
        try:
            from vnstock.config import Config
            Config.REQUEST_TIMEOUT = 15
            Config.RETRIES = 1
            Config.BACKOFF_MIN = 1
            Config.BACKOFF_MAX = 2
        except Exception:
            pass

        from vnstock import Vnstock

        stock = Vnstock().stock(symbol=symbol, source=source)
        raw = stock.quote.history(start=start, end=end, interval="1D")
        queue.put((True, _normalize_ohlcv(raw, symbol), ""))
    except BaseException as exc:
        queue.put((False, None, f"{type(exc).__name__}: {str(exc)[:500]}"))


def _worker_index(queue, start: str, end: str, source: str, api_key: str):
    try:
        os.environ["VNSTOCK_API_KEY"] = api_key
        try:
            from vnstock.config import Config
            Config.REQUEST_TIMEOUT = 15
            Config.RETRIES = 1
            Config.BACKOFF_MIN = 1
            Config.BACKOFF_MAX = 2
        except Exception:
            pass

        from vnstock import Vnstock

        stock = Vnstock().stock(symbol="VNINDEX", source=source)
        raw = stock.quote.history(start=start, end=end, interval="1D")
        queue.put((True, _normalize_ohlcv(raw, "VNINDEX"), ""))
    except BaseException as exc:
        queue.put((False, None, f"{type(exc).__name__}: {str(exc)[:500]}"))


def _run_worker(target, args, timeout: int = REQUEST_TIMEOUT_SECONDS):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=target, args=(queue, *args))
    process.daemon = True
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join(5)
        return None, f"timeout {timeout}s"

    if queue.empty():
        return None, f"worker exited with code {process.exitcode} without a result"

    ok, data, error = queue.get()
    if ok:
        return data, ""
    return None, error


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
    state = getattr(_rate_limit_gate, "state", 0.0)
    wait = REQUEST_INTERVAL_SECONDS - (time.monotonic() - state)
    if wait > 0:
        time.sleep(wait)
    _rate_limit_gate.state = time.monotonic()


def _safe_error(exc: BaseException) -> str:
    key = os.getenv("VNSTOCK_API_KEY", "")
    return str(exc).replace(key, "[API_KEY]")[:500]


def _fetch_equity(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    errors = []
    for source in ("KBS", "VCI"):
        _rate_limit_gate()
        data, error = _run_worker(
            _worker_equity,
            (symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), source, os.getenv("VNSTOCK_API_KEY", "")),
        )
        if data is not None:
            return data
        errors.append(f"{source}: {error}")
    raise RuntimeError(f"VNstock không lấy được {symbol}: " + " | ".join(errors))


def _fetch_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    errors = []
    for source in ("KBS", "VCI"):
        _rate_limit_gate()
        data, error = _run_worker(
            _worker_index,
            (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), source, os.getenv("VNSTOCK_API_KEY", "")),
        )
        if data is not None:
            return data
        errors.append(f"{source}: {error}")
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
        if progress_callback:
            progress_callback(i - 1, total, symbol, "Kiểm tra cache")
        try:
            frame, fetched = _load_symbol(symbol, fetch_start, end, force_refresh=force_refresh)
            stock_frames.append(frame)
            if progress_callback:
                progress_callback(i, total, symbol, "Đã tải API" if fetched else "Dùng cache")
        except Exception as exc:
            failed.append((symbol, _safe_error(exc)))
            if progress_callback:
                progress_callback(i, total, symbol, f"LỖI: {_safe_error(exc)}")
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
