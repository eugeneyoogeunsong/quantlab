"""Layer 3 - Transaction cost models (QA Section D).

Cost assumptions are where most backtests quietly die. A strategy that turns
over 100% monthly and pays 20bps round-trip is handing over ~2.4%/yr before it
earns anything -- often the entire alpha.

Reference points from the practitioner literature:
  - deep futures, institutional : ~5-7 bps round trip
  - liquid large-cap equities   : ~10-15 bps round trip (retail)
  - small caps                  : 20-50+ bps, spread-dominated

Default here is 10 bps round-trip on large-cap equities (5 bps commission +
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
        """Cost as a fraction of portfolio value, per date per symbol."""
        ...


@dataclass
class ZeroCost:
    """No costs. For unit tests and for measuring the cost drag by difference.

    Never report a headline number from this. QA Section G exists because
    performance that only survives at zero cost is not performance.
    """

    name: str = "zero"

    def cost(self, turnover, prices=None, volume=None):
        return pd.DataFrame(0.0, index=turnover.index, columns=turnover.columns)


@dataclass
class FixedBpsCost:
    """Flat basis-point charge on traded notional.

    `commission_bps` and `spread_bps` are ONE-WAY. A round-trip pays both twice.
    Turnover here is already |w_t - w_{t-1}|, i.e. one-way notional traded, so
    the per-unit charge is applied once per side automatically.
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

    Impact grows with the square root of participation rate -- the standard
    practitioner model. Trading 1% of daily volume costs far less than 10x the
    cost of trading 0.1%; it costs about 3.2x.

        impact_bps = coefficient * volatility * sqrt(order_size / daily_volume)

    Analogy: a small boat barely disturbs the water. A ship ten times heavier
    does not make ten times the wake -- but it does make a wake, and it has to
    sail through its own.

    This is the model that reveals capacity limits. Run it at $100k and $100M
    of AUM; if the edge only exists at $100k, that is worth knowing before you
    scale.
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
    """Execution slippage separate from spread and commission.

    Two components:
      - `fixed_bps`: baseline shortfall vs. the assumed fill price.
      - `vol_multiple`: extra slippage proportional to that day's volatility.
        You get worse fills on turbulent days, which is inconveniently when
        most signals want to trade.
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
    """Sum several models. e.g. FixedBpsCost + SlippageModel."""

    models: list
    name: str = "composite"

    def cost(self, turnover, prices=None, volume=None):
        total = pd.DataFrame(0.0, index=turnover.index, columns=turnover.columns)
        for m in self.models:
            total = total + m.cost(turnover, prices, volume)
        return total


# Named presets so a config file can say `cost_preset: retail_equity`
# rather than scattering magic numbers through research code.
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
