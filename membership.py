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
# 02/02/2026: VPL replaced BCM.
# 13/05/2026: BSR replaced DGC after DGC became ineligible.
# 03/08/2026: MCH and TCX replaced PLX and TPB.
PERIODS = [
    MembershipPeriod(date(2026, 2, 2), ("VPL",), ("BCM",), "Rà soát Q1/2026"),
    MembershipPeriod(date(2026, 5, 13), ("BSR",), ("DGC",), "Thay thế đặc biệt DGC"),
    MembershipPeriod(date(2026, 8, 3), ("MCH", "TCX"), ("PLX", "TPB"), "Rà soát Q3/2026"),
]


def membership_at(date_value: pd.Timestamp) -> list[str]:
    current = set(VN30)
    for period in reversed(PERIODS):
        if date_value >= pd.Timestamp(period.effective_date):
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
