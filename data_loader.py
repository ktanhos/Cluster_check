from __future__ import annotations

import os
import time
from typing import Callable

import pandas as pd

from membership import symbols_for_period

# Deliberately no disk cache in the Data Layer.
# Every press of "Cập nhật dữ liệu VNstock" starts a fresh download.
REQUEST_INTERVAL_SECONDS = 1.1


def _normalize_ohlcv(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError(f"Không có dữ liệu cho {symbol}")

    df = raw.copy()
    rename_map = {
        "date": "time",
        "Date": "time",
        "time": "time",
        "Time": "time",
    }
    df.rename(columns=rename_map, inplace=True)

    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột {missing} cho {symbol}")

    df = df[required].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["time", "close"], inplace=True)
    df.sort_values("time", inplace=True)
    df.drop_duplicates("time", keep="last", inplace=True)
    df["symbol"] = symbol
    return df


def _configure_vnstock(api_key: str | None = None) -> None:
    key = (api_key or os.getenv("VNSTOCK_API_KEY", "")).strip()
    if not key:
        raise ValueError("Thiếu VNstock API Key.")
    os.environ["VNSTOCK_API_KEY"] = key


def _rate_limit_gate() -> None:
    previous = getattr(_rate_limit_gate, "last_request", 0.0)
    wait = REQUEST_INTERVAL_SECONDS - (time.monotonic() - previous)
    if wait > 0:
        time.sleep(wait)
    _rate_limit_gate.last_request = time.monotonic()


def _safe_error(exc: BaseException) -> str:
    key = os.getenv("VNSTOCK_API_KEY", "")
    text = str(exc).replace(key, "[API_KEY]")
    return text[:700]


def _fetch_equity(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fresh VNstock download with no disk cache and one Unified UI route.

    Vnstock 4 recommends the Unified UI Market interface. We intentionally
    do not wrap the request in signal, multiprocessing, custom timeout or
    nested retry logic. Those layers caused the previous Streamlit failures.
    """
    _rate_limit_gate()
    from vnstock.ui import Market

    market = Market()
    raw = market.equity(symbol).ohlcv(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    return _normalize_ohlcv(raw, symbol)


def _fetch_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    _rate_limit_gate()
    from vnstock.ui import Market

    market = Market()
    raw = market.index("VNINDEX").ohlcv(
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    return _normalize_ohlcv(raw, "VNINDEX")


def load_market_data(
    start: pd.Timestamp,
    end: pd.Timestamp,
    warmup: int = 80,
    api_key: str | None = None,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
    force_refresh: bool = True,
):
    """Download a completely fresh dataset for the selected research period.

    There is intentionally no disk cache. Data is held only in Streamlit
    session state after this function returns, so the model layer can be
    rerun repeatedly without making another VNstock request.
    """
    _configure_vnstock(api_key)

    fetch_start = start - pd.Timedelta(days=int(warmup * 1.7))
    symbols = symbols_for_period(fetch_start, end)
    total = len(symbols) + 1
    stock_frames: list[pd.DataFrame] = []
    failed: list[tuple[str, str]] = []

    for i, symbol in enumerate(symbols, 1):
        if progress_callback:
            progress_callback(i - 1, total, symbol, "Đang gọi VNstock")
        try:
            frame = _fetch_equity(symbol, fetch_start, end)
            stock_frames.append(frame)
            if progress_callback:
                progress_callback(i, total, symbol, "Đã tải")
        except Exception as exc:
            message = _safe_error(exc)
            failed.append((symbol, message))
            if progress_callback:
                progress_callback(i, total, symbol, f"LỖI: {message}")

    if failed:
        details = "; ".join(f"{symbol}: {error}" for symbol, error in failed)
        raise RuntimeError(
            f"Không tải được {len(failed)} mã: {details}"
        )

    if progress_callback:
        progress_callback(total - 1, total, "VNINDEX", "Đang gọi VNstock")
    try:
        index = _fetch_index(fetch_start, end)
    except Exception as exc:
        raise RuntimeError(f"Không tải được VNINDEX: {_safe_error(exc)}") from exc

    if progress_callback:
        progress_callback(total, total, "VNINDEX", "Đã tải")

    stock = pd.concat(stock_frames, ignore_index=True)
    stock["time"] = pd.to_datetime(stock["time"])
    index["time"] = pd.to_datetime(index["time"])

    for df in (stock, index):
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["time", "close"], inplace=True)

    stock.sort_values(["symbol", "time"], inplace=True)
    index.sort_values(["symbol", "time"], inplace=True)
    return stock, index
