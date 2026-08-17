from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from config import VN30


@dataclass(frozen=True)
class MembershipPeriod:
    effective_date: date
    added: tuple[str, ...]
    removed: tuple[str, ...]
    reason: str


PERIODS = [
    MembershipPeriod(date(2026, 2, 2), ("VPL",), ("BCM",), "Rà soát Q1/2026"),
    MembershipPeriod(date(2026, 5, 13), ("BSR",), ("DGC",), "Thay thế DGC"),
    MembershipPeriod(date(2026, 8, 3), ("MCH", "TCX"), ("PLX", "TPB"), "Rà soát Q3/2026"),
]

ALL_STUDY_SYMBOLS = sorted(
    set(VN30)
    .union(*(set(period.added) for period in PERIODS))
    .union(*(set(period.removed) for period in PERIODS))
)


def membership_at(date_value: pd.Timestamp) -> list[str]:
    date_value = pd.Timestamp(date_value)
    current = set(VN30)
    for period in reversed(PERIODS):
        effective = pd.Timestamp(period.effective_date)
        if date_value >= effective:
            break
        current.difference_update(period.added)
        current.update(period.removed)
    return sorted(current)


def symbols_for_period(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    """Return only symbols that can actually be members during the requested period."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    symbols = set(membership_at(start))
    for period in PERIODS:
        d = pd.Timestamp(period.effective_date)
        if start <= d <= end:
            symbols.update(period.added)
            symbols.update(period.removed)
    return sorted(symbols)


def membership_series(dates: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for d in dates:
        for ticker in membership_at(pd.Timestamp(d)):
            rows.append({"Date": pd.Timestamp(d), "Ticker": ticker, "Active": True})
    return pd.DataFrame(rows)


def change_table() -> pd.DataFrame:
    return pd.DataFrame([{"EffectiveDate": pd.Timestamp(p.effective_date), "Added": ", ".join(p.added), "Removed": ", ".join(p.removed), "Reason": p.reason} for p in PERIODS])


def validate_membership() -> None:
    for p in PERIODS:
        before = set(membership_at(pd.Timestamp(p.effective_date) - pd.Timedelta(days=1)))
        after = set(membership_at(pd.Timestamp(p.effective_date)))
        if len(before) != 30 or len(after) != 30:
            raise ValueError(f"VN30 membership không đủ 30 mã tại {p.effective_date}")
