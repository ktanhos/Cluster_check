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


# VN30 membership timeline for 2026 used by the rolling study.
# The base VN30 list in config.py is the membership at the latest study date.
# Each period below describes the change that took effect on that date.
PERIODS = [
    MembershipPeriod(date(2026, 2, 2), ("VPL",), ("BCM",), "Rà soát Q1/2026"),
    MembershipPeriod(date(2026, 5, 13), ("BSR",), ("DGC",), "Thay thế đặc biệt DGC"),
    MembershipPeriod(date(2026, 8, 3), ("MCH", "TCX"), ("PLX", "TPB"), "Rà soát Q3/2026"),
]

# Every symbol that can be required during the study period.
# data_loader.py uses this list so that symbols entering or leaving VN30
# are available for the full historical membership-aware calculation.
ALL_STUDY_SYMBOLS = sorted(
    set(VN30)
    .union(*(set(period.added) for period in PERIODS))
    .union(*(set(period.removed) for period in PERIODS))
)


def membership_at(date_value: pd.Timestamp) -> list[str]:
    """Return the 30 VN30 constituents active on a given date.

    VN30 in config.py represents the latest membership state. We walk the
    change timeline backwards and reverse every change that occurred after
    the requested date.
    """
    date_value = pd.Timestamp(date_value)
    current = set(VN30)

    for period in reversed(PERIODS):
        effective = pd.Timestamp(period.effective_date)
        if date_value >= effective:
            break
        current.difference_update(period.added)
        current.update(period.removed)

    return sorted(current)


def membership_series(dates: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for d in dates:
        active = membership_at(pd.Timestamp(d))
        for ticker in active:
            rows.append({"Date": pd.Timestamp(d), "Ticker": ticker, "Active": True})
    return pd.DataFrame(rows)


def change_table() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "EffectiveDate": pd.Timestamp(p.effective_date),
            "Added": ", ".join(p.added),
            "Removed": ", ".join(p.removed),
            "Reason": p.reason,
        }
        for p in PERIODS
    ])


def validate_membership() -> None:
    for p in PERIODS:
        before = set(membership_at(pd.Timestamp(p.effective_date) - pd.Timedelta(days=1)))
        after = set(membership_at(pd.Timestamp(p.effective_date)))
        if len(before) != 30 or len(after) != 30:
            raise ValueError(f"VN30 membership không đủ 30 mã tại {p.effective_date}")
