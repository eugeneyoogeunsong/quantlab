"""Layer 3: transaction cost models (QA Section D).

Cost assumptions are where most backtests quietly die: a strategy that turns
over 100% monthly and pays 20bps round trip hands over ~2.4%/yr before it earns
anything, which is frequently the entire alpha.

Reference points from the practitioner literature:
  - deep futures, institutional : ~5-7 bps round trip
  - liquid large-cap equities   : ~10-15 bps round trip (retail)
  - small caps                  : 20-50+ bps, spread-dominated

We default to 10 bps round trip on large-cap equities (5 bps commission plus
5 bps half-spread each way). The library refuses to run a zero-cost backtest
without an explicit override, because "I forgot to add costs" and "costs are
zero" produce identical numbers and only one of them is a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class CostModel(Protocol):
    name: str

    def cost(self, turnover: pd.DataFrame, prices: pd.DataFrame, volume: pd.DataFrame | None) -> pd.DataFrame:
        """Cost as a fraction of portfolio value, per date and per symbol."""
        ...


@dataclass
class ZeroCost:
    """No costs at all: for unit tests, and for measuring cost drag by difference.

    Never report a headline number from this. QA Section G exists precisely
    because performance that survives only at zero cost is not performance.
    """

    name: str = "zero"

    def cost(self, turnover, prices=None, volume=None):
        return pd.DataFrame(0.0, index=turnover.index, columns=turnover.columns)


@dataclass
class FixedBpsCost:
    """Flat basis-point charge on traded notional.

    `commission_bps` and `spread_bps` are ONE-WAY; a round trip therefore pays
    both twice. Turnover arriving here is already |w_t - w_{t-1}| (i.e., one-way
    notional traded), so the per-unit charge lands once per side automatically.
    """

    commission_bps: float = 5.0
    spread_bps: float = 5.0
    name: str = "fixed_bps"

    @property
    def one_way_bps(self) -> float:
        return self.commission_bps + self.spread_bps

    @property
    def round_trip_bps(self) -> float:
        return 2 * self.one_way_bps

    def cost(self, turnover, prices=None, volume=None):
        return turnover.abs() * (self.one_way_bps / 1e4)


@dataclass
class SquareRootImpactCost:
    """Fixed cost plus a square-root market-impact term.

    Impact grows with the square root of the participation rate (the standard
    practitioner model): trading 1% of daily volume does not cost ten times what
    trading 0.1% costs, it costs about 3.2 times as much.

        impact_bps = coefficient * volatility * sqrt(order_size / daily_volume)

    The intuition is a wake. A small boat barely disturbs the water, and a ship
    ten times heavier makes far less than ten times the wake; it does make one,
    though, and it has to sail through it.

    This is the model that exposes capacity limits. Run it at $100k and at $100M
    of AUM; if the edge exists only at $100k, that is worth knowing before we
    scale rather than after.
    """

    commission_bps: float = 5.0
    spread_bps: float = 5.0
    impact_coefficient: float = 0.10
    capital: float = 1_000_000.0
    name: str = "sqrt_impact"

    def cost(self, turnover, prices=None, volume=None):
        base = turnover.abs() * ((self.commission_bps + self.spread_bps) / 1e4)
        if volume is None or prices is None:
            return base

        dollar_volume = (volume * prices).replace(0, np.nan)
        order_dollars = turnover.abs() * self.capital
        participation = (order_dollars / dollar_volume).clip(0, 1.0).fillna(0.0)

        vol = prices.pct_change(fill_method=None).rolling(63).std().fillna(0.02)
        impact = self.impact_coefficient * vol * np.sqrt(participation)
        return base + turnover.abs() * impact


@dataclass
class SlippageModel:
    """Execution slippage, kept separate from spread and commission.

    Two components:
      - `fixed_bps`: baseline shortfall against the assumed fill price.
      - `vol_multiple`: additional slippage proportional to that day's
        volatility. Fills get worse on turbulent days, which is inconveniently
        when most signals want to trade.
    """

    fixed_bps: float = 2.0
    vol_multiple: float = 0.0
    name: str = "slippage"

    def cost(self, turnover, prices=None, volume=None):
        base = turnover.abs() * (self.fixed_bps / 1e4)
        if self.vol_multiple and prices is not None:
            vol = prices.pct_change(fill_method=None).rolling(21).std().fillna(0.0)
            base = base + turnover.abs() * vol * self.vol_multiple
        return base


@dataclass
class CompositeCost:
    """Sum of several models (e.g., FixedBpsCost plus SlippageModel)."""

    models: list
    name: str = "composite"

    def cost(self, turnover, prices=None, volume=None):
        total = pd.DataFrame(0.0, index=turnover.index, columns=turnover.columns)
        for m in self.models:
            total = total + m.cost(turnover, prices, volume)
        return total


# Named presets, so that a config file can say `cost_preset: retail_equity`
# instead of scattering magic numbers through the research code.
PRESETS = {
    "institutional_futures": FixedBpsCost(commission_bps=2.0, spread_bps=1.5),
    "large_cap_equity": FixedBpsCost(commission_bps=5.0, spread_bps=5.0),
    "retail_equity": CompositeCost([FixedBpsCost(5.0, 7.5), SlippageModel(2.0, 0.5)]),
    "small_cap_equity": CompositeCost([FixedBpsCost(5.0, 20.0), SlippageModel(5.0, 1.0)]),
    "crypto": FixedBpsCost(commission_bps=10.0, spread_bps=5.0),
    "zero": ZeroCost(),
}


def get_cost_model(name: str):
    if name not in PRESETS:
        raise KeyError(f"Unknown cost preset {name!r}. Available: {sorted(PRESETS)}")
    return PRESETS[name]
