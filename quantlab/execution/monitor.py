# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Layer 5: monitoring, i.e., answering "how do you know when a strategy is dying?"

That question appears in the blueprint's Q&A section, and it is the hardest one,
because the answer has to separate two things that look identical over the short
run: a normal drawdown, and a broken edge.

We therefore compare live results against the backtest's *own* distribution. A
drawdown is unremarkable if the backtest produced several like it; the same
drawdown is a red flag if the backtest never saw anything close.

Nothing here is a trading signal by itself: these are prompts to investigate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from ..backtest import metrics as M

Level = Literal["ok", "watch", "alert"]


@dataclass
class Alert:
    level: Level
    metric: str
    message: str
    value: float = float("nan")
    threshold: float = float("nan")
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds"))

    def __str__(self) -> str:
        return f"[{self.level.upper():5s}] {self.metric}: {self.message}"


@dataclass
class StrategyMonitor:
    """Compares live returns against the backtest baseline.

    `baseline_returns` should be the out-of-sample series wherever one exists:
    comparing live results to an in-sample backtest sets the bar too high and
    guarantees a stream of false alarms.
    """

    baseline_returns: pd.Series
    dd_percentile: float = 0.95
    sharpe_floor_ratio: float = 0.5
    vol_ratio_limit: float = 1.5

    def __post_init__(self) -> None:
        b = self.baseline_returns.dropna()
        self.baseline_sharpe = M.sharpe_ratio(b)
        self.baseline_vol = M.volatility(b)
        self.baseline_max_dd = M.max_drawdown(b)
        # Distribution of rolling 63-day drawdowns in the backtest, so that
        # "unusual" is defined by history, not by a round number someone liked.
        dd = M.drawdown_series(b)
        self.dd_threshold = float(np.quantile(dd, 1 - self.dd_percentile)) if len(dd) else -0.2

    def check(self, live_returns: pd.Series, min_obs: int = 42) -> list[Alert]:
        live = live_returns.dropna()
        if len(live) < min_obs:
            return [Alert("ok", "sample",
                          f"{len(live)} live observations; need {min_obs} before "
                          "any comparison is meaningful.", len(live), min_obs)]

        alerts: list[Alert] = []

        # (i) Drawdown, judged against the worst the backtest ever produced
        live_dd = M.max_drawdown(live)
        if live_dd < self.baseline_max_dd:
            alerts.append(Alert("alert", "drawdown",
                f"Live drawdown {live_dd:.1%} is worse than anything in the backtest "
                f"({self.baseline_max_dd:.1%}). Either the regime changed or the "
                "implementation differs from the simulation.", live_dd, self.baseline_max_dd))
        elif live_dd < self.dd_threshold:
            alerts.append(Alert("watch", "drawdown",
                f"Live drawdown {live_dd:.1%} is in the worst {(1-self.dd_percentile):.0%} "
                "of the backtest distribution. Unusual but not unprecedented.",
                live_dd, self.dd_threshold))

        # (ii) Sharpe decay relative to the baseline, reported with its standard error
        live_sharpe = M.sharpe_ratio(live)
        floor = self.baseline_sharpe * self.sharpe_floor_ratio
        if self.baseline_sharpe > 0 and live_sharpe < floor:
            level = "alert" if live_sharpe < 0 else "watch"
            alerts.append(Alert(level, "sharpe",
                f"Live Sharpe {live_sharpe:.2f} vs backtest {self.baseline_sharpe:.2f}. "
                f"Note the standard error over {len(live)/252:.1f}y is roughly "
                f"{1/np.sqrt(max(len(live)/252, 0.1)):.2f} -- this may still be noise.",
                live_sharpe, floor))

        # (iii) Volatility regime shift, which invalidates the calibrated position sizes
        live_vol = M.volatility(live)
        if self.baseline_vol > 0:
            ratio = live_vol / self.baseline_vol
            if ratio > self.vol_ratio_limit:
                alerts.append(Alert("watch", "volatility",
                    f"Live vol {live_vol:.1%} is {ratio:.1f}x the backtest's "
                    f"{self.baseline_vol:.1%}. Position sizes calibrated to the old "
                    "regime are now too large.", live_vol, self.baseline_vol * self.vol_ratio_limit))

        # (iv) Distribution shift (two-sample Kolmogorov-Smirnov; scipy is optional here)
        try:
            from scipy import stats as sps
            ks, p = sps.ks_2samp(live.values, self.baseline_returns.dropna().values)
            if p < 0.01:
                alerts.append(Alert("watch", "distribution",
                    f"Live return distribution differs from backtest (KS={ks:.3f}, p={p:.4f}). "
                    "Worth checking whether fills, costs, or the universe drifted.", ks, 0.01))
        except Exception:
            pass

        if not alerts:
            alerts.append(Alert("ok", "all",
                f"Live Sharpe {live_sharpe:.2f}, drawdown {live_dd:.1%}, vol {live_vol:.1%} "
                "-- all within the backtest's range.", live_sharpe, self.baseline_sharpe))
        return alerts

    def report(self, live_returns: pd.Series) -> str:
        lines = ["STRATEGY HEALTH CHECK", "=" * 60,
                 f"Baseline: Sharpe {self.baseline_sharpe:.2f}, vol {self.baseline_vol:.1%}, "
                 f"max DD {self.baseline_max_dd:.1%}", "-" * 60]
        lines += [str(a) for a in self.check(live_returns)]
        return "\n".join(lines)


def turnover_drift(backtest_turnover: float, live_turnover: float,
                   tolerance: float = 0.30) -> Alert:
    """Flag live turnover running above the backtest: realised costs then exceed modelled."""
    if backtest_turnover <= 0:
        return Alert("ok", "turnover", "No backtest turnover baseline.")
    ratio = live_turnover / backtest_turnover
    if ratio > 1 + tolerance:
        return Alert("watch", "turnover",
            f"Live turnover {live_turnover:.1f}x vs backtest {backtest_turnover:.1f}x "
            f"({ratio:.1f}x). Real costs are running above the modelled ones.",
            live_turnover, backtest_turnover * (1 + tolerance))
    return Alert("ok", "turnover",
                 f"Live turnover {live_turnover:.1f}x vs backtest {backtest_turnover:.1f}x.",
                 live_turnover, backtest_turnover)
