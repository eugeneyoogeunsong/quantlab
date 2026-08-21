"""Layer 1, data: vendor-agnostic price loading backed by an on-disk cache.

Design note
-----------
`DataSource` is a Protocol, not a base class: any object carrying a matching
`fetch` signature qualifies, so swapping yfinance for Polygon or Tiingo later
costs one new class and one line of config, with no edits to the research,
backtest, or portfolio layers.

The cache is content-addressed by (source, symbols, dates, interval); after the
first pull, re-running research is free and the results are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Canonical column names, in the order every loader must return them.
OHLCV = ["open", "high", "low", "close", "volume"]


class DataSource(Protocol):
    """Structural contract for a price vendor adapter: implement `fetch`, nothing else."""

    name: str

    def fetch(
        self,
        symbols: Sequence[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return a long-format frame indexed by (date, symbol), carrying the OHLCV columns."""
        ...


@dataclass
class YFinanceSource:
    """yfinance adapter.

    `auto_adjust=True` returns OHLC adjusted for splits and dividends alike, which
    satisfies QA Section A item 1. The caveat is a real one: yfinance serves only
    *currently listed* tickers, so there is no delisted history and any universe
    built from it carries survivorship bias. Setting `survivorship_safe = False`
    propagates that fact into the QA report instead of leaving it to sit silently
    in a footnote.
    """

    name: str = "yfinance"
    auto_adjust: bool = True
    survivorship_safe: bool = False

    def fetch(
        self,
        symbols: Sequence[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        import yfinance as yf

        symbols = list(dict.fromkeys(symbols))  # de-duplicate, preserving order
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(
                tickers=symbols,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=self.auto_adjust,
                progress=False,
                group_by="column",
                threads=True,
            )

        if raw is None or raw.empty:
            raise RuntimeError(
                f"yfinance returned no data for {symbols[:5]}"
                f"{'...' if len(symbols) > 5 else ''} ({start} to {end}).\n"
                "Common causes:\n"
                "  - No internet access, or a proxy/firewall blocking Yahoo Finance.\n"
                "  - Invalid ticker symbols.\n"
                "  - A date range with no trading sessions.\n"
                "To work offline, use the synthetic source instead:\n"
                "  PipelineConfig(data_source='synthetic')   or   --source synthetic"
            )

        # yfinance returns a flat frame for a single symbol, a MultiIndex for several.
        if isinstance(raw.columns, pd.MultiIndex):
            frame = raw.stack(level=1, future_stack=True)
            frame.index.names = ["date", "symbol"]
        else:
            frame = raw.copy()
            frame["symbol"] = symbols[0]
            frame = frame.set_index("symbol", append=True)
            frame.index.names = ["date", "symbol"]

        frame.columns = [str(c).lower().replace(" ", "_") for c in frame.columns]
        missing = [c for c in OHLCV if c not in frame.columns]
        if missing:
            raise RuntimeError(f"Vendor did not return required columns: {missing}")

        frame = frame[OHLCV].sort_index()
        frame = frame.dropna(subset=["close"])
        return frame


@dataclass
class CSVSource:
    """Load from local CSVs, one file per symbol, named ``<SYMBOL>.csv``.

    Use this to plug in a paid vendor export, or point-in-time data that includes
    delisted names (which is how survivorship bias actually gets fixed, rather
    than merely disclosed).
    """

    directory: Path
    name: str = "csv"
    survivorship_safe: bool = True

    def fetch(
        self,
        symbols: Sequence[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        parts = []
        for sym in symbols:
            path = Path(self.directory) / f"{sym}.csv"
            if not path.exists():
                log.warning("CSVSource: no file for %s at %s", sym, path)
                continue
            df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
            df.columns = [c.lower() for c in df.columns]
            df["symbol"] = sym
            parts.append(df.set_index("symbol", append=True)[OHLCV])
        if not parts:
            raise RuntimeError(f"No CSV files found in {self.directory}")
        out = pd.concat(parts).sort_index()
        out.index.names = ["date", "symbol"]
        return out.loc[str(start) : str(end)]


@dataclass
class SyntheticSource:
    """Deterministic simulated prices, with no network required.

    It exists for three reasons:

    1. CI. Tests that hit a live vendor are flaky by construction: they fail on a
       Sunday, during an outage, or behind a firewall.
    2. Offline development.
    3. Null-hypothesis testing. Prices here are a random walk plus the drift you
       specify, so there is genuinely no signal beyond that drift. If a strategy
       posts a strong Sharpe on `regime="random_walk"` data, it has found
       structure that does not exist, and the finding is about the code, not the
       market; as bug detectors go, this one costs nothing and catches plenty.
    """

    seed: int = 42
    annual_drift: float = 0.07
    annual_vol: float = 0.18
    regime: str = "mixed"  # 'random_walk' | 'trending' | 'mixed'
    name: str = "synthetic"
    survivorship_safe: bool = True
    auto_adjust: bool = True

    def fetch(
        self,
        symbols: Sequence[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        dates = pd.bdate_range(start=start, end=end)
        n = len(dates)
        if n < 2:
            raise RuntimeError(f"Date range {start}..{end} yields fewer than 2 sessions")

        # A shared market factor, so the assets co-move the way real ones do:
        # without it, cross-sectional strategies collect a diversification
        # benefit that no real panel would hand them.
        market = rng.normal(0, self.annual_vol / np.sqrt(252) * 0.6, n)

        parts = []
        for i, sym in enumerate(dict.fromkeys(symbols)):
            # Spread drift and volatility across symbols so cross-sectional
            # strategies have something to rank on.
            drift = self.annual_drift * (1 + 0.5 * ((i % 5) - 2) / 2)
            vol = self.annual_vol * (1 + 0.3 * ((i % 4) - 1.5) / 1.5)
            daily_vol = vol / np.sqrt(252)

            idio = rng.normal(drift / 252, daily_vol * 0.8, n)
            shocks = idio + market * (0.8 + 0.4 * ((i % 3) / 2))  # beta varies by symbol

            if self.regime in ("trending", "mixed"):
                # Stochastic regime switching, NOT a deterministic cycle.
                #
                # An earlier version used a sine wave here, and it was a mistake:
                # a fixed-period oscillation is perfectly predictable, and it
                # produced multi-year drawdowns (-61%) that are impossible given
                # the stated 8.9% volatility, so any momentum strategy tested on
                # it looked far better than it deserved. A two-state Markov chain
                # with persistent but random switching gives realistic trends
                # without handing strategies a free clock to trade against.
                p_switch = 1 / 189 if self.regime == "trending" else 1 / 252
                state, states = 1, np.empty(n)
                for t in range(n):
                    if rng.random() < p_switch:
                        state = -state
                    states[t] = state
                amplitude = daily_vol * (0.35 if self.regime == "trending" else 0.20)
                shocks = shocks + states * amplitude

            close = 100 * np.exp(np.cumsum(shocks))
            intraday = np.abs(rng.normal(0, vol / np.sqrt(252) * 0.5, n))
            df = pd.DataFrame({
                "open": close * (1 + rng.normal(0, 0.001, n)),
                "high": close * (1 + intraday),
                "low": close * (1 - intraday),
                "close": close,
                "volume": rng.lognormal(15, 0.4, n).round(),
            }, index=dates)
            df["symbol"] = sym
            parts.append(df.set_index("symbol", append=True))

        out = pd.concat(parts).sort_index()
        out.index.names = ["date", "symbol"]
        return out[OHLCV]


@dataclass
class PriceCache:
    """Parquet cache keyed by a fingerprint of the request.

    Reproducibility matters more here than it sounds: if the vendor silently
    restates history between two runs, an uncached pipeline hands you two
    different backtests for identical code, and nothing in the output tells you
    why.
    """

    root: Path = field(default_factory=lambda: Path("./.cache/prices"))

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, source: str, symbols: Iterable[str], start: str, end: str, interval: str) -> str:
        payload = json.dumps(
            {"src": source, "sym": sorted(symbols), "s": str(start), "e": str(end), "i": interval},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get_or_fetch(
        self,
        source: DataSource,
        symbols: Sequence[str],
        start: str,
        end: str,
        interval: str = "1d",
        force: bool = False,
    ) -> pd.DataFrame:
        key = self._key(source.name, symbols, start, end, interval)
        path = self.root / f"{key}.parquet"
        if path.exists() and not force:
            log.info("cache hit %s", path.name)
            return pd.read_parquet(path)
        log.info("cache miss -> fetching %d symbols from %s", len(symbols), source.name)
        frame = source.fetch(symbols, start, end, interval)
        try:
            frame.to_parquet(path)
        except Exception as exc:  # pragma: no cover (cache writes are best-effort)
            log.warning("could not write cache: %s", exc)
        return frame


def to_wide(frame: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Pivot long (date, symbol) rows into a wide date x symbol matrix for one field.

    Every downstream layer speaks wide matrices; doing the pivot once, here, keeps
    a single auditable place where date alignment happens.
    """
    if field not in frame.columns:
        raise KeyError(f"{field!r} not in {list(frame.columns)}")
    wide = frame[field].unstack("symbol").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide


def load_prices(
    symbols: Sequence[str],
    start: str,
    end: str,
    source: DataSource | None = None,
    cache_dir: str | Path = "./.cache/prices",
    interval: str = "1d",
    force: bool = False,
) -> pd.DataFrame:
    """Convenience entry point: cached, long-format OHLCV for the requested symbols."""
    source = source or YFinanceSource()
    cache = PriceCache(Path(cache_dir))
    return cache.get_or_fetch(source, symbols, start, end, interval, force=force)
