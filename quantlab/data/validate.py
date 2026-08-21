"""Layer 1, data integrity validators (QA Checklist Section A).

Each function returns a `Check` and nothing raises: the QA report should surface
*every* problem in a single pass rather than stopping at the first. Severity, not
the exception mechanism, then decides whether the pipeline may continue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

Severity = Literal["pass", "warn", "fail"]


@dataclass
class Check:
    """One checklist line item: its verdict, and the evidence standing behind it."""

    section: str
    name: str
    status: Severity
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status != "fail"

    def __str__(self) -> str:
        icon = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}[self.status]
        return f"{icon} {self.section} :: {self.name} -- {self.detail}"


# ---------------------------------------------------------------------------
# Section A: data integrity
# ---------------------------------------------------------------------------


def check_adjustment(source_name: str, auto_adjust: bool) -> Check:
    """A1: prices adjusted for splits and dividends (where the vendor supports it)."""
    if auto_adjust:
        return Check(
            "A", "Split/dividend adjusted", "pass",
            f"{source_name} returning adjusted OHLC (auto_adjust=True).",
            {"source": source_name},
        )
    return Check(
        "A", "Split/dividend adjusted", "fail",
        f"{source_name} returning RAW prices. A 4:1 split reads as a -75% return "
        "and will fire every momentum and mean-reversion signal you own.",
        {"source": source_name},
    )


def check_missing_dates(prices: pd.DataFrame, max_gap_days: int = 5) -> Check:
    """A2a: no missing sessions, i.e., gaps beyond `max_gap_days` calendar days."""
    idx = pd.DatetimeIndex(prices.index)
    if len(idx) < 2:
        return Check("A", "Calendar continuity", "fail", "Fewer than 2 rows of data.")

    gaps = idx.to_series().diff().dt.days.dropna()
    # A weekend is a legitimate 3-day gap; a holiday alongside it pushes that to 4.
    big = gaps[gaps > max_gap_days]
    if big.empty:
        return Check(
            "A", "Calendar continuity", "pass",
            f"No gaps > {max_gap_days}d across {len(idx)} sessions "
            f"({idx[0]:%Y-%m-%d} to {idx[-1]:%Y-%m-%d}).",
            {"sessions": len(idx)},
        )
    worst = big.sort_values(ascending=False).head(5)
    return Check(
        "A", "Calendar continuity", "warn",
        f"{len(big)} gaps > {max_gap_days}d. Largest {int(worst.iloc[0])}d at "
        f"{worst.index[0]:%Y-%m-%d}. Check for holidays vs. vendor outage.",
        {"n_gaps": int(len(big)), "largest_days": int(worst.iloc[0])},
    )


def check_missing_values(prices: pd.DataFrame, max_nan_frac: float = 0.10) -> Check:
    """A2b: NaN density per symbol, measured as a fraction of all sessions."""
    frac = prices.isna().mean()
    bad = frac[frac > max_nan_frac].sort_values(ascending=False)
    if bad.empty:
        return Check(
            "A", "Missing values", "pass",
            f"All {prices.shape[1]} symbols below {max_nan_frac:.0%} NaN.",
            {"max_nan_frac": float(frac.max()) if len(frac) else 0.0},
        )
    return Check(
        "A", "Missing values", "warn",
        f"{len(bad)} symbol(s) exceed {max_nan_frac:.0%} NaN: "
        + ", ".join(f"{s} ({v:.0%})" for s, v in bad.head(5).items())
        + ". Usually a late listing (correct) or a vendor hole (not).",
        {"symbols": bad.head(10).round(3).to_dict()},
    )


def check_bad_ticks(prices: pd.DataFrame, threshold: float = 0.50) -> Check:
    """A2c: bad ticks, i.e., implausible single-day moves.

    On adjusted data a single-day move beyond 50% is rare but genuinely occurs
    (biotech readouts, energy in 2020). We flag it and never auto-remove it:
    silently deleting outliers is itself a bias, and usually a flattering one.
    """
    rets = prices.pct_change(fill_method=None)
    hits = rets.abs() > threshold
    n = int(hits.to_numpy().sum())
    if n == 0:
        return Check(
            "A", "Bad ticks", "pass",
            f"No single-day moves beyond {threshold:.0%}.", {"n_outliers": 0},
        )
    stacked = rets.where(hits).stack()
    top = stacked.reindex(stacked.abs().sort_values(ascending=False).index).head(5)
    return Check(
        "A", "Bad ticks", "warn",
        f"{n} move(s) beyond {threshold:.0%}. Largest: "
        + "; ".join(f"{sym} {d:%Y-%m-%d} {v:+.0%}" for (d, sym), v in top.items())
        + ". Verify each is a real event, not an unadjusted split.",
        {"n_outliers": n},
    )


def check_survivorship(universe_is_pit: bool, source_survivorship_safe: bool) -> Check:
    """A3 / B4: survivorship bias addressed.

    A clean pass requires both halves: point-in-time membership, and a vendor that
    retains delisted history. One without the other still leaks.
    """
    if universe_is_pit and source_survivorship_safe:
        return Check(
            "A", "Survivorship bias", "pass",
            "Point-in-time membership + vendor carries delisted history.",
        )
    if universe_is_pit and not source_survivorship_safe:
        return Check(
            "A", "Survivorship bias", "warn",
            "Membership is point-in-time but the vendor has no delisted prices, "
            "so names that left the index still vanish from the panel.",
        )
    return Check(
        "A", "Survivorship bias", "warn",
        "Static universe of currently-listed names. Every symbol here is known "
        "to have survived to today. Historical returns are biased UPWARD and the "
        "bias is largest exactly where it hurts -- in crisis periods, because the "
        "companies that did not make it are absent. Treat results as an upper bound.",
        {"point_in_time": universe_is_pit, "vendor_safe": source_survivorship_safe},
    )


def check_monotonic_unique(prices: pd.DataFrame) -> Check:
    """A4: index sanity (sorted ascending, free of duplicates, and timezone-naive)."""
    idx = pd.DatetimeIndex(prices.index)
    problems = []
    if not idx.is_monotonic_increasing:
        problems.append("index is not sorted ascending")
    if idx.has_duplicates:
        problems.append(f"{int(idx.duplicated().sum())} duplicate timestamps")
    if idx.tz is not None:
        problems.append(f"index is tz-aware ({idx.tz}); mixing tz-aware and naive silently misaligns joins")
    if problems:
        return Check("A", "Index sanity", "fail", "; ".join(problems))
    return Check("A", "Index sanity", "pass", "Sorted, unique, timezone-naive.")


def check_sufficient_history(prices: pd.DataFrame, min_obs: int, lookback: int) -> Check:
    """A5: enough history to warm the signal up and still leave a meaningful test window."""
    n = len(prices)
    if n < lookback * 2:
        return Check(
            "A", "History length", "fail",
            f"{n} sessions is less than 2x the {lookback}-session lookback. "
            "There is no out-of-sample left after the signal warms up.",
            {"n_obs": n, "lookback": lookback},
        )
    if n < min_obs:
        return Check(
            "A", "History length", "warn",
            f"{n} sessions (~{n/252:.1f}y) is below the {min_obs}-session floor. "
            "Sharpe estimates over short samples have enormous standard errors.",
            {"n_obs": n},
        )
    return Check(
        "A", "History length", "pass",
        f"{n} sessions (~{n/252:.1f}y) covering {lookback}-session lookback with room to spare.",
        {"n_obs": n},
    )


def run_data_checks(
    prices: pd.DataFrame,
    *,
    source_name: str = "unknown",
    auto_adjust: bool = True,
    source_survivorship_safe: bool = False,
    universe_is_pit: bool = False,
    lookback: int = 252,
    min_obs: int = 756,
) -> list[Check]:
    """Run every Section A check and return the verdicts in report order."""
    return [
        check_adjustment(source_name, auto_adjust),
        check_monotonic_unique(prices),
        check_missing_dates(prices),
        check_missing_values(prices),
        check_bad_ticks(prices),
        check_survivorship(universe_is_pit, source_survivorship_safe),
        check_sufficient_history(prices, min_obs, lookback),
    ]


def clean_prices(
    prices: pd.DataFrame,
    ffill_limit: int = 5,
    min_price: float = 1.0,
) -> pd.DataFrame:
    """Conservative cleaning, deliberately doing very little.

    - Forward-fill short holes only (`ffill_limit` sessions): a halted name that
      never reopens should stay NaN rather than carry a stale price forever.
    - Never back-fill. Back-filling writes future information into the past; it
      is look-ahead bias under a friendlier name.
    - Drop sub-$1 names, where the bid-ask spread swamps any realistic edge.
    """
    out = prices.copy()
    out = out.ffill(limit=ffill_limit)
    out = out.where(out >= min_price)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out
