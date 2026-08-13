"""Layer 4 - Risk controls (QA Section E).

The blueprint's warning is that most people skip layers 3-5. This module is
layer 4's core: the rules that decide not what to buy, but what you refuse to do
regardless of how good the signal looks.

Every control here is causal. In particular `drawdown_control` uses only equity
observed up to t-1 when deciding exposure for t. Computing a drawdown from the
full equity curve and then "de-risking" at the peaks is a spectacular and
surprisingly common form of look-ahead bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class RiskLimits:
    """Hard constraints applied after sizing, before execution."""

    max_position: float = 0.25          # per name, fraction of capital
    max_gross_exposure: float = 1.0     # sum |w|
    max_net_exposure: float = 1.0       # sum w
    max_sector_exposure: float = 0.40
    max_drawdown_stop: float = 0.25     # de-risk beyond this
    drawdown_derisk_to: float = 0.50    # scale book to this on breach
    vol_target: float | None = None
    max_leverage: float = 1.0

    def describe(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def apply_position_limits(weights: pd.DataFrame, max_position: float = 0.25) -> pd.DataFrame:
    """Cap any single name, then renormalise so the book still sums correctly.

    Concentration is the fastest route to ruin. A 40% position in one name means
    a single accounting scandal is a 40% portfolio event.
    """
    capped = weights.clip(-max_position, max_position)
    original_gross = weights.abs().sum(axis=1)
    new_gross = capped.abs().sum(axis=1)
    # Redistribute only where capping actually removed exposure.
    need = (original_gross > 0) & (new_gross > 0) & (new_gross < original_gross - 1e-12)
    scale = pd.Series(1.0, index=weights.index)
    scale[need] = (original_gross[need] / new_gross[need]).clip(upper=1.0 / max(max_position, 1e-9))
    out = capped.mul(scale, axis=0)
    return out.clip(-max_position, max_position)


def apply_exposure_limits(weights: pd.DataFrame, max_gross: float = 1.0,
                          max_net: float = 1.0) -> pd.DataFrame:
    """Scale the whole book down when gross or net exposure breaches its cap."""
    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)

    gross_scale = np.where(gross > max_gross, max_gross / gross.replace(0, np.nan), 1.0)
    net_scale = np.where(net.abs() > max_net, max_net / net.abs().replace(0, np.nan), 1.0)
    scale = pd.Series(np.minimum(gross_scale, net_scale), index=weights.index).fillna(1.0)
    return weights.mul(scale, axis=0)


def apply_sector_limits(weights: pd.DataFrame, sector_map: dict[str, str],
                        max_sector: float = 0.40) -> pd.DataFrame:
    """Cap aggregate exposure per sector.

    Matters because signals cluster. A momentum screen in 2020 would happily
    have handed you a portfolio of nine software names and called it diversified.
    """
    out = weights.copy()
    sectors = {}
    for sym, sec in sector_map.items():
        if sym in weights.columns:
            sectors.setdefault(sec, []).append(sym)

    for sec, names in sectors.items():
        exposure = out[names].abs().sum(axis=1)
        breach = exposure > max_sector
        if breach.any():
            scale = pd.Series(1.0, index=out.index)
            scale[breach] = max_sector / exposure[breach]
            out.loc[:, names] = out[names].mul(scale, axis=0)
            log.info("Sector %s scaled on %d date(s)", sec, int(breach.sum()))
    return out


def drawdown_control(weights: pd.DataFrame, equity: pd.Series,
                     stop_level: float = 0.25, derisk_to: float = 0.50,
                     recovery_level: float = 0.10) -> pd.DataFrame:
    """Cut exposure after a drawdown threshold; restore on recovery.

    CAUSALITY: exposure for date t is decided from equity through t-1 only.

    Whether this helps is genuinely contested. It reliably reduces the depth of
    the worst drawdown. It also reliably locks in losses at the bottom and misses
    the sharpest rebounds, which historically cluster immediately after the worst
    days. The honest framing is that it buys psychological survivability at some
    cost in expected return -- and a system you actually stick with beats a
    better one you abandon.
    """
    eq = equity.reindex(weights.index).ffill()
    prior_eq = eq.shift(1)
    peak = prior_eq.cummax()
    dd = (prior_eq / peak - 1).fillna(0.0)

    scale = pd.Series(1.0, index=weights.index)
    derisked = False
    for i, dt in enumerate(weights.index):
        d = dd.iloc[i]
        if not derisked and d <= -abs(stop_level):
            derisked = True
        elif derisked and d >= -abs(recovery_level):
            derisked = False
        scale.iloc[i] = derisk_to if derisked else 1.0
    return weights.mul(scale, axis=0)


def volatility_scaling(weights: pd.DataFrame, prices: pd.DataFrame,
                       target_vol: float = 0.10, window: int = 63,
                       max_scale: float = 1.5) -> pd.DataFrame:
    """Alias into the sizing module's vol targeting, kept here for discoverability."""
    from .sizing import volatility_target
    return volatility_target(weights, prices, target_vol, window, max_scale)


@dataclass
class RiskManager:
    """Applies the full stack of controls in a deliberate order.

    Order matters and is not arbitrary:
      1. position caps    -- fix concentration first
      2. sector caps      -- then cluster risk
      3. vol targeting    -- then scale total risk
      4. exposure caps    -- then enforce the hard leverage ceiling
      5. drawdown control -- last, so it can override everything above

    Reversing 3 and 4 would let vol targeting push you back above the leverage
    limit after it had been enforced.
    """

    limits: RiskLimits = field(default_factory=RiskLimits)
    sector_map: dict[str, str] | None = None
    applied: list[str] = field(default_factory=list)

    def apply(self, weights: pd.DataFrame, prices: pd.DataFrame,
              equity: pd.Series | None = None) -> pd.DataFrame:
        w = weights.copy()
        self.applied = []

        w = apply_position_limits(w, self.limits.max_position)
        self.applied.append(f"position cap {self.limits.max_position:.0%}")

        if self.sector_map:
            w = apply_sector_limits(w, self.sector_map, self.limits.max_sector_exposure)
            self.applied.append(f"sector cap {self.limits.max_sector_exposure:.0%}")

        if self.limits.vol_target:
            w = volatility_scaling(w, prices, self.limits.vol_target,
                                   max_scale=self.limits.max_leverage)
            self.applied.append(f"vol target {self.limits.vol_target:.0%}")

        w = apply_exposure_limits(w, self.limits.max_gross_exposure, self.limits.max_net_exposure)
        self.applied.append(f"gross cap {self.limits.max_gross_exposure:.0%}")

        if equity is not None and self.limits.max_drawdown_stop:
            w = drawdown_control(w, equity, self.limits.max_drawdown_stop,
                                 self.limits.drawdown_derisk_to)
            self.applied.append(f"DD stop {self.limits.max_drawdown_stop:.0%}")

        return w.fillna(0.0)

    def audit(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Post-hoc check that no limit was violated. Trust, then verify."""
        rows = []
        maxpos = weights.abs().max(axis=1)
        gross = weights.abs().sum(axis=1)
        net = weights.sum(axis=1)
        rows.append({"limit": "max_position", "cap": self.limits.max_position,
                     "observed_max": round(float(maxpos.max()), 4),
                     "breaches": int((maxpos > self.limits.max_position + 1e-6).sum())})
        rows.append({"limit": "max_gross", "cap": self.limits.max_gross_exposure,
                     "observed_max": round(float(gross.max()), 4),
                     "breaches": int((gross > self.limits.max_gross_exposure + 1e-6).sum())})
        rows.append({"limit": "max_net", "cap": self.limits.max_net_exposure,
                     "observed_max": round(float(net.abs().max()), 4),
                     "breaches": int((net.abs() > self.limits.max_net_exposure + 1e-6).sum())})
        return pd.DataFrame(rows)
