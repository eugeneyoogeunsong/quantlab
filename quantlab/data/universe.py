"""Layer 1 - Universe definition.

QA Section A: "Universe definition is explicit (what symbols, when, and why)."

A `Universe` is not just a ticker list. It records *why* those tickers, and
whether membership is point-in-time. That distinction is the whole survivorship
bias question: a static list of today's S&P 500 members, backtested to 2010, is
a list of companies that we know in advance survived and thrived.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

import pandas as pd


@dataclass
class Universe:
    """An explicit, documented tradable set.

    Parameters
    ----------
    name        : short identifier used in reports.
    symbols     : the static ticker list (used when `membership` is None).
    rationale   : free text -- why these names. Forced field; the QA layer
                  fails the run if it is empty.
    membership  : optional point-in-time membership matrix (date x symbol, bool).
                  When supplied, the universe is survivorship-safe and the
                  backtester will only hold a name on dates where it was a member.
    """

    name: str
    symbols: list[str]
    rationale: str
    membership: pd.DataFrame | None = None
    created: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))

    def __post_init__(self) -> None:
        self.symbols = list(dict.fromkeys(self.symbols))
        if not self.symbols:
            raise ValueError("Universe must contain at least one symbol")
        if not self.rationale.strip():
            raise ValueError(
                "Universe.rationale is required -- QA Section A demands an explicit "
                "statement of what symbols, when, and why."
            )

    @property
    def is_point_in_time(self) -> bool:
        return self.membership is not None

    def active_on(self, date) -> list[str]:
        """Symbols tradable on `date`."""
        if self.membership is None:
            return list(self.symbols)
        date = pd.Timestamp(date)
        if date not in self.membership.index:
            idx = self.membership.index[self.membership.index <= date]
            if len(idx) == 0:
                return []
            date = idx[-1]
        row = self.membership.loc[date]
        return list(row[row.astype(bool)].index)

    def mask_for(self, index: pd.DatetimeIndex, columns: Sequence[str]) -> pd.DataFrame:
        """Boolean date x symbol tradability mask aligned to a price frame."""
        if self.membership is None:
            return pd.DataFrame(True, index=index, columns=list(columns))
        m = self.membership.reindex(index=index, columns=list(columns))
        return m.ffill().fillna(False).astype(bool)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "n_symbols": len(self.symbols),
            "point_in_time": self.is_point_in_time,
            "rationale": self.rationale,
            "created": self.created,
        }


# --------------------------------------------------------------------------
# Prebuilt universes. Deliberately small and liquid: the point of a reference
# universe is to make the plumbing testable, not to be a production selection.
# --------------------------------------------------------------------------

MEGA_CAP_TECH = Universe(
    name="mega_cap_tech",
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AVGO", "ORCL", "CRM", "AMD"],
    rationale=(
        "Ten mega-cap US technology names. Chosen for tight spreads and deep liquidity so "
        "cost assumptions are not the dominant term. NOT survivorship-safe: this is a "
        "list of today's winners, so any backtest on it is optimistic by construction. "
        "Use for plumbing validation, not for return estimates."
    ),
)

SECTOR_ETFS = Universe(
    name="sector_etfs",
    symbols=["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"],
    rationale=(
        "SPDR Select Sector ETFs spanning the full GICS map. ETFs largely sidestep "
        "single-name survivorship bias -- the sector persists even as constituents "
        "change -- which makes this the most defensible of the built-in universes for "
        "cross-sectional work. XLRE (2015) and XLC (2018) launched late, so pre-launch "
        "dates are correctly NaN rather than backfilled."
    ),
)

GLOBAL_MACRO_ETFS = Universe(
    name="global_macro_etfs",
    symbols=["SPY", "EFA", "EEM", "IEF", "TLT", "LQD", "HYG", "GLD", "DBC", "VNQ"],
    rationale=(
        "Multi-asset ETF sleeve: US equity, developed ex-US, emerging, intermediate and "
        "long Treasuries, IG and HY credit, gold, broad commodities, REITs. This is the "
        "standard proving ground for time-series momentum / trend following, which was "
        "documented across asset classes rather than within equities."
    ),
)

BUILTIN_UNIVERSES = {
    u.name: u for u in (MEGA_CAP_TECH, SECTOR_ETFS, GLOBAL_MACRO_ETFS)
}


def get_universe(name: str) -> Universe:
    if name not in BUILTIN_UNIVERSES:
        raise KeyError(f"Unknown universe {name!r}. Available: {sorted(BUILTIN_UNIVERSES)}")
    return BUILTIN_UNIVERSES[name]
