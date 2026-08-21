"""Layer 2: strategies.

A Strategy maps prices to target weights. It does NOT decide position size in
dollars, apply risk limits, or know about costs; those belong to Layer 4 and
Layer 3. Keeping the seam here means we can run the same signal through three
different risk configurations and attribute the difference honestly.

The contract
------------
`generate_weights(prices) -> DataFrame` (date x symbol), where row `t` holds the
weights we *want* on date t, decided from information available at the close of
t. The backtester applies them at the next open. Strategies must never shift
signals forward themselves: doing it in two places is how you end up with an
accidental double-shift and a suspiciously good backtest.

Evidence grades
---------------
Each strategy carries a `.evidence` string. Published, replicated, decades of
out-of-sample data is not the same thing as a blog backtest, and the report
should say which is which.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import features as F


@dataclass
class Strategy(ABC):
    """Base class; subclasses implement `raw_signal`."""

    name: str = "unnamed"
    params: dict[str, Any] = field(default_factory=dict)
    long_only: bool = True
    citation: str = ""
    evidence: str = "ungraded"

    @abstractmethod
    def raw_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Higher value = more attractive; NaN = not tradable on that date."""

    @property
    def min_history(self) -> int:
        """Rows needed before the first valid signal, used to trim the warm-up."""
        return int(max([v for v in self.params.values() if isinstance(v, (int, float))] or [252]))

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Default: rank the raw signal, then go long the top decile-ish slice."""
        sig = self.raw_signal(prices)
        return self._to_weights(sig)

    def _to_weights(self, sig: pd.DataFrame) -> pd.DataFrame:
        """Equal-weight the selected names; used by most subclasses."""
        sel = sig > 0
        counts = sel.sum(axis=1)
        w = sel.astype(float).div(counts.replace(0, np.nan), axis=0)
        return w.fillna(0.0)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "params": dict(self.params),
            "long_only": self.long_only,
            "citation": self.citation,
            "evidence": self.evidence,
        }


# ---------------------------------------------------------------------------
# 1. Cross-sectional momentum
# ---------------------------------------------------------------------------

@dataclass
class CrossSectionalMomentum(Strategy):
    """Buy the recent relative winners, rebalance monthly.

    Jegadeesh & Titman (1993) sorted US stocks on the prior 12-month return
    skipping the most recent month, and found that the top-minus-bottom decile
    earned roughly 1%/month over 1965-1989. The result has been replicated across
    countries and asset classes and remains one of the most-studied anomalies in
    finance, whilst also being famous for rare, violent crashes during sharp
    market rebounds (e.g., 2009), when the short leg of beaten-down names rockets.

    This implementation is long-only top-N, which sidesteps the crash risk of the
    short leg but also gives up most of the documented spread.
    """

    name: str = "xs_momentum"
    lookback: int = 252
    skip: int = 21
    top_n: int = 3
    citation: str = "Jegadeesh & Titman (1993), Journal of Finance 48(1)"
    evidence: str = "strong -- 30+ years of replication across markets and asset classes"

    def __post_init__(self) -> None:
        self.params = {"lookback": self.lookback, "skip": self.skip, "top_n": self.top_n}

    def raw_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        return F.momentum(prices, self.lookback, self.skip)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        sig = self.raw_signal(prices)
        ranks = sig.rank(axis=1, ascending=False, na_option="keep")
        sel = ranks <= self.top_n
        counts = sel.sum(axis=1)
        return sel.astype(float).div(counts.replace(0, np.nan), axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# 2. Time-series momentum / trend following
# ---------------------------------------------------------------------------

@dataclass
class TimeSeriesMomentum(Strategy):
    """Hold each asset only whilst its own 12-month return is positive.

    Moskowitz, Ooi & Pedersen (2012) documented this across 58 futures and
    forward contracts spanning equity indices, currencies, commodities, and
    sovereign bonds. Unlike cross-sectional momentum it is an absolute rule, so
    the book de-risks to cash in broad bear markets: the source of trend
    following's reputation for paying off in crises.

    The analogy: cross-sectional momentum always picks the fastest horse in the
    race, whereas time-series momentum declines to bet if every horse is walking.
    """

    name: str = "ts_momentum"
    lookback: int = 252
    vol_target_scale: bool = False
    vol_window: int = 63
    citation: str = "Moskowitz, Ooi & Pedersen (2012), Journal of Financial Economics 104(2)"
    evidence: str = "strong -- multi-asset, multi-decade, though post-2010 returns are weaker"

    def __post_init__(self) -> None:
        self.params = {"lookback": self.lookback, "vol_window": self.vol_window}

    def raw_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        return F.time_series_momentum(prices, self.lookback)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        sig = self.raw_signal(prices)
        in_trend = (sig > 0) & sig.notna()

        if self.vol_target_scale:
            # Inverse-vol tilt inside the trend sleeve, so that a quiet bond ETF and
            # a loud commodity ETF contribute comparable risk, not comparable dollars.
            vol = F.realized_vol(prices, self.vol_window)
            raw = in_trend.astype(float) / vol.replace(0, np.nan)
        else:
            raw = in_trend.astype(float)

        total = raw.sum(axis=1)
        # Deliberately NOT renormalised to 1.0 when few assets trend: sitting
        # partly in cash through downtrends is the strategy, not a bug.
        n_assets = prices.notna().sum(axis=1).replace(0, np.nan)
        if self.vol_target_scale:
            w = raw.div(total.replace(0, np.nan), axis=0).mul(
                (in_trend.sum(axis=1) / n_assets).clip(0, 1), axis=0
            )
        else:
            w = raw.div(n_assets, axis=0)
        return w.fillna(0.0)


# ---------------------------------------------------------------------------
# 3. Low volatility
# ---------------------------------------------------------------------------

@dataclass
class LowVolatility(Strategy):
    """Hold the lowest-volatility names. The low-risk anomaly.

    Low-beta and low-volatility stocks have historically delivered higher
    risk-adjusted returns than CAPM predicts. The leading explanation is leverage
    constraints: investors who want more return but cannot borrow bid up
    high-beta stocks instead, depressing their forward returns. The effect is
    closely related to Frazzini & Pedersen's betting-against-beta factor.

    One caveat worth stating plainly: the raw return is usually *lower* than the
    market. The claim is about return per unit of risk, and capturing it in
    absolute terms has historically required leverage, which reintroduces exactly
    the risk you were avoiding.
    """

    name: str = "low_vol"
    vol_window: int = 126
    top_n: int = 3
    citation: str = "Frazzini & Pedersen (2014); Baker, Bradley & Wurgler (2011)"
    evidence: str = "strong on risk-adjusted basis; weak on absolute return"

    def __post_init__(self) -> None:
        self.params = {"vol_window": self.vol_window, "top_n": self.top_n}

    def raw_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        return -F.realized_vol(prices, self.vol_window)  # negate: higher is better

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        sig = self.raw_signal(prices)
        ranks = sig.rank(axis=1, ascending=False, na_option="keep")
        sel = ranks <= self.top_n
        counts = sel.sum(axis=1)
        return sel.astype(float).div(counts.replace(0, np.nan), axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# 4. Mean reversion
# ---------------------------------------------------------------------------

@dataclass
class MeanReversion(Strategy):
    """Buy the short-term losers: momentum's mirror image, at a shorter horizon.

    Short-horizon reversal (1 week to 1 month) is well documented, and is exactly
    why the 12-1 momentum signal skips the most recent month. It is also the
    strategy most sensitive to trading costs in this library: it turns over
    constantly, and gross edges of a few basis points per trade do not survive
    realistic spreads. Watch the cost-drag line in the report.
    """

    name: str = "mean_reversion"
    zscore_window: int = 21
    entry_z: float = -1.0
    top_n: int = 3
    citation: str = "Lehmann (1990); Lo & MacKinlay (1990)"
    evidence: str = "moderate -- robust in gross terms, frequently negative after costs"

    def __post_init__(self) -> None:
        self.params = {
            "zscore_window": self.zscore_window,
            "entry_z": self.entry_z,
            "top_n": self.top_n,
        }

    def raw_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        z = F.zscore(prices, self.zscore_window)
        return -z  # flip the sign: the most oversold name is the most attractive

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        sig = self.raw_signal(prices)
        eligible = sig > -self.entry_z  # i.e., z below entry_z, sign already flipped
        ranks = sig.where(eligible).rank(axis=1, ascending=False, na_option="keep")
        sel = ranks <= self.top_n
        counts = sel.sum(axis=1)
        return sel.astype(float).div(counts.replace(0, np.nan), axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# 5. Dual momentum
# ---------------------------------------------------------------------------

@dataclass
class DualMomentum(Strategy):
    """Relative momentum to pick, absolute momentum to gate.

    Two filters stacked: (i) rank the assets against each other
    (cross-sectional), then (ii) require the winner to be beating cash on its own
    (time-series). If the best available asset is still falling, hold nothing.

    Popularised by Gary Antonacci. The construction is a sensible combination of
    two independently well-evidenced effects; note, however, that the specific
    published parameterisation has a much thinner out-of-sample record than the
    underlying momentum literature it draws on.
    """

    name: str = "dual_momentum"
    lookback: int = 252
    skip: int = 21
    top_n: int = 2
    citation: str = "Antonacci (2014), 'Dual Momentum Investing'"
    evidence: str = "moderate -- components well-evidenced, specific recipe less so"

    def __post_init__(self) -> None:
        self.params = {"lookback": self.lookback, "skip": self.skip, "top_n": self.top_n}

    def raw_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        return F.momentum(prices, self.lookback, self.skip)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        rel = self.raw_signal(prices)
        absolute = F.time_series_momentum(prices, self.lookback)
        ranks = rel.rank(axis=1, ascending=False, na_option="keep")
        sel = (ranks <= self.top_n) & (absolute > 0)  # both gates must pass, jointly
        counts = sel.sum(axis=1)
        return sel.astype(float).div(counts.replace(0, np.nan), axis=0).fillna(0.0)


# ---------------------------------------------------------------------------
# 6. Risk-managed benchmark
# ---------------------------------------------------------------------------

@dataclass
class BuyAndHold(Strategy):
    """Equal-weight everything, always. The benchmark you must beat.

    QA Section C requires a benchmark comparison, and this is the honest one: if
    a strategy cannot beat equal-weight buy-and-hold after costs, its complexity
    is not earning anything.
    """

    name: str = "buy_and_hold"
    citation: str = "n/a"
    evidence: str = "benchmark"

    def __post_init__(self) -> None:
        self.params = {}

    @property
    def min_history(self) -> int:
        return 1

    def raw_signal(self, prices: pd.DataFrame) -> pd.DataFrame:
        return prices.notna().astype(float)

    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        avail = prices.notna()
        counts = avail.sum(axis=1)
        return avail.astype(float).div(counts.replace(0, np.nan), axis=0).fillna(0.0)


STRATEGY_REGISTRY = {
    "xs_momentum": CrossSectionalMomentum,
    "ts_momentum": TimeSeriesMomentum,
    "low_vol": LowVolatility,
    "mean_reversion": MeanReversion,
    "dual_momentum": DualMomentum,
    "buy_and_hold": BuyAndHold,
}


def get_strategy(name: str, **kwargs) -> Strategy:
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Available: {sorted(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](**kwargs)
