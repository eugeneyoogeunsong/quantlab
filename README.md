# quantlab

[![tests](https://github.com/eugeneyoogeunsong/quantlab/actions/workflows/tests.yml/badge.svg)](https://github.com/eugeneyoogeunsong/quantlab/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An algorithmic trading research framework in Python, built to the 5-layer system
map from the *Algorithmic Trading System Blueprint*, with the *Backtest QA
Checklist* implemented as executable code rather than a document you are meant
to remember.

The organising idea: **a backtest is not a result until it has a QA report
attached.** `Pipeline.run()` returns performance and the checklist verdict in the
same object, and `qa.gate()` raises on any blocking failure. Looking at a Sharpe
ratio here without also seeing whether it is trustworthy is deliberately awkward.

Built by [**Eugene (Yoogeun) Song**](https://www.linkedin.com/in/yoogeunsong) - a PhD researcher at Imperial College London -
as an independent side project: a hobby, written in my own time, out of curiosity
about how much of a backtest survives honest scrutiny.

---

## Install

```bash
git clone https://github.com/eugeneyoogeunsong/quantlab.git
cd quantlab
pip install -e ".[dev]"
pytest                    # 289 tests, fully offline
```

Python 3.10+. Core dependencies: pandas, numpy, scipy, yfinance, pyarrow.

## Run it

```bash
# Full pipeline with the complete QA checklist
python -m quantlab.cli backtest --strategy xs_momentum --universe sector_etfs -o report.html

# No network? Use the offline simulator
python -m quantlab.cli backtest --strategy ts_momentum --source synthetic

# All six strategies side by side
python -m quantlab.cli compare --universe sector_etfs

# Today's target book as paper orders
python -m quantlab.cli orders --strategy dual_momentum --capital 100000
```

Or from Python:

```python
from quantlab import Pipeline, PipelineConfig
from quantlab.report import build_report

pr = Pipeline(PipelineConfig(
    universe="sector_etfs",
    strategy="xs_momentum",
    start="2012-01-01", end="2024-12-31",
    cost_preset="large_cap_equity",
)).run()

pr.qa.print_report()          # the checklist, section by section
pr.qa.gate()                  # raises if anything blocking failed
build_report(pr, "report.html")
```

---

## The five layers

| Layer | Module | Responsibility | Output |
|---|---|---|---|
| 1. Data | `quantlab.data` | Loaders, universe definition, integrity checks | Clean, versioned, cached panels |
| 2. Research | `quantlab.research` | Features and signals | Target weights |
| 3. Backtest | `quantlab.backtest` | Engine, costs, metrics, validation | Performance + risk metrics |
| 4. Portfolio | `quantlab.portfolio` | Sizing, exposure and drawdown limits | Risk-adjusted positions |
| 5. Execution | `quantlab.execution` | Paper broker, order generation, monitoring | Orders + health alerts |

Two further modules sit alongside the five layers:

| Module | Responsibility |
|---|---|
| `quantlab.derivatives` | Options pricing (analytic, lattice, PDE and Monte Carlo), European and American, plus Greeks and implied volatility |
| `quantlab.portfolio.optimisation` | Markowitz mean-variance with CAPM/EWMA inputs |

The blueprint issues one warning above all others: most people fail because they
skip layers 3 to 5. That is precisely why layers 3, 4 and 5 are the largest part
of this codebase, not the smallest.

---

## How look-ahead bias is prevented

Not by care and attention. Structurally, in one place.

Strategies return **unshifted** weights: row `t` holds what you want given data
through `t`'s close. The engine applies the lag, at exactly one line in
`backtest/engine.py`:

```python
effective = held.shift(cfg.execution_lag)
```

`BacktestConfig` refuses `execution_lag=0` with an error. Nothing else in the
library shifts a signal. If a strategy shifts as well, the result is a double
lag: worse performance, never better, so the failure mode is conservative.

Four independent tests defend this:

1. **Truncation:** recompute every feature on history cut to 70% and require
   identical values on the overlap. Catches `bfill()`, `rolling(center=True)`,
   and full-sample scaling.
2. **Correlation:** `corr(weight_t, return_t)` must stay below 0.10. Real daily
   alpha sits around 0.01 to 0.05; leakage sits above 0.3.
3. **Perfect foresight:** a strategy that holds tomorrow's best asset must be
   caught. If the detector misses it, the detector is broken.
4. **Null hypothesis:** every strategy is run on 40 zero-drift random walks.
   Anything that reliably profits there is finding structure that does not
   exist. (`tests/test_null_hypothesis.py`)

That last one is the strongest check in the suite. Current results on pure noise:

| Strategy | Mean Sharpe on noise |
|---|---|
| xs_momentum | 0.02 |
| ts_momentum | 0.06 |
| low_vol | 0.17 |
| mean_reversion | 0.05 |
| dual_momentum | 0.04 |
| buy_and_hold | 0.19 |

All near zero, as they must be. The small positive bias for long-only books is
Jensen's inequality, not leakage: zero *log* drift implies a slightly positive
*arithmetic* drift of σ²/2.

---

## The QA checklist, in code

Every section of the source checklist maps to runnable checks in `quantlab/qa/`
and `quantlab/data/validate.py`:

| Section | Checks |
|---|---|
| **A: Data Integrity** | adjustment, index sanity, calendar gaps, NaN density, bad ticks, survivorship, history length |
| **B: Bias + Leakage** | execution lag, weight causality, feature causality, rebalance timing |
| **C: Validation** | walk-forward OOS, parameter sensitivity, regime coverage, benchmark, deflated Sharpe |
| **D: Trading Reality** | costs applied, turnover sanity, execution spec |
| **E: Risk & Portfolio** | position/exposure/drawdown limits, with a post-hoc breach audit |
| **F: Reporting** | required metrics present, sample adequacy vs. minimum track record |
| **G: Robustness triad** | does it survive costs **and** OOS **and** parameter variation? |

Three statuses. `pass` and `warn` let the run proceed; `fail` blocks it. A
`ZeroCost` model is a **failure**, not a warning: "I forgot to add costs" and
"costs are zero" produce identical numbers, and only one of them is a decision.

Survivorship bias is always at least a warning when the universe is a static
list of currently-listed tickers, because that is what it is.

---

## Strategies

Six reference implementations, each carrying a citation and an evidence grade.
Published-and-replicated is not the same as a blog backtest, and the report says
which is which.

| Strategy | Signal | Evidence |
|---|---|---|
| `xs_momentum` | 12-1 relative return, top N | **Strong:** Jegadeesh & Titman (1993), 30+ years of replication |
| `ts_momentum` | Own 12-month return > 0 | **Strong:** Moskowitz, Ooi & Pedersen (2012), 58 markets |
| `low_vol` | Lowest trailing volatility | **Strong** risk-adjusted, **weak** absolute |
| `mean_reversion` | Short-horizon z-score reversal | **Moderate:** robust gross, often negative after costs |
| `dual_momentum` | Relative rank + absolute gate | **Moderate:** components strong, specific recipe less so |
| `buy_and_hold` | Equal weight, always | Benchmark |

Adding your own means implementing one method:

```python
from quantlab.research.strategies import Strategy
from dataclasses import dataclass

@dataclass
class MyStrategy(Strategy):
    name: str = "my_strategy"
    lookback: int = 60

    def raw_signal(self, prices):
        # Higher = more attractive. Use only data at or before each row.
        return prices.pct_change(self.lookback)
```

The QA layer will test it for causality automatically. It does not take your
word for it.

---

## Cost model

Costs are where most backtests quietly die. Presets:

| Preset | One-way | Notes |
|---|---|---|
| `institutional_futures` | 3.5 bps | Deep futures, institutional execution |
| `large_cap_equity` | 10 bps | Default |
| `retail_equity` | 12.5 bps + slippage | Realistic retail |
| `small_cap_equity` | 25 bps + slippage | Spread-dominated |
| `crypto` | 15 bps | |

`SquareRootImpactCost` adds market impact scaling with the square root of
participation rate: doubling order size raises per-share impact by 1.41x rather
than 2x, so total cost grows as size^1.5. Run it at two capital levels to find
where your capacity limit actually is.

Every run reports **annual cost drag** next to the Sharpe. `cost_sensitivity()`
re-runs at escalating cost levels and reports the break-even.

---

## Overfitting controls

- **Deflated Sharpe ratio** (Bailey & López de Prado): corrects for how many
  variants you tried. The expected maximum of N noisy Sharpes grows with N, and
  DSR subtracts it before judging. The pipeline counts trials honestly: every
  parameter combination evaluated in the sweep is fed into the correction.
- **Probabilistic Sharpe ratio:** adjusts for skew and fat tails.
- **Minimum track record length:** how many years you would need before the
  Sharpe is distinguishable from zero. Frequently sobering.
- **Walk-forward with embargo:** parameters selected on training data only, with
  a gap before the test window so lookback windows cannot straddle the boundary.

---

## Tests

```bash
pytest                    # 289 tests
pytest -m "not slow"      # skip the Monte Carlo null tests
```

Coverage includes leakage detection, engine lag mechanics, cost monotonicity,
metric cross-checks against an independent implementation, risk-limit
enforcement, walk-forward window integrity, and the null-hypothesis suite.

Two bugs found and fixed by these tests during development, both documented as
regression tests:

- `sharpe_ratio` returned 3.7e16 for a constant return series: `np.std` of a
  constant array is ~1e-19, not `0.0`, so the `sd == 0` guard never fired.
- The synthetic data generator used a deterministic sine wave for regimes,
  producing a −61% drawdown at 8.9% annual volatility; that is impossible under
  a random walk (0 of 3000 Monte Carlo paths came close). Replaced with
  stochastic regime switching.

---

## What this framework will not do for you

- **Survivorship-safe equity data.** yfinance has no delisted names. The QA
  layer flags this on every run rather than letting it pass quietly. Fixing it
  properly needs a point-in-time vendor and the `Universe.membership` matrix.
- **Fundamental data.** The value and quality factors described in
  `docs/STRATEGIES.md` need financial statements the free loader does not
  provide.
- **Intraday microstructure.** The engine is daily and vectorised. Signals that
  live inside the day need an event-driven engine.
- **Tell you a strategy will work.** Passing QA means the backtest is not
  obviously broken; that is a floor, not a recommendation. The base rate for
  retail strategies that survive contact with live markets is low, and nothing
  in this repo changes that.

---

## Repository layout

```
quantlab/
├── quantlab/
│   ├── data/          Layer 1 — loaders, universe, integrity checks
│   ├── research/      Layer 2 — features, strategies
│   ├── backtest/      Layer 3 — engine, costs, metrics, validation
│   ├── portfolio/     Layer 4 — sizing, risk limits, mean-variance optimisation
│   ├── execution/     Layer 5 — broker, monitoring
│   ├── derivatives/   Options pricing — 4 independent methods + implied vol
│   ├── qa/            The checklist, executable
│   ├── pipeline.py    Wires all five layers together
│   ├── report.py      Standalone HTML reports
│   └── cli.py         backtest / compare / orders
├── tests/             289 tests incl. null-hypothesis suite
├── docs/              QA_CHECKLIST.md, STRATEGIES.md, DERIVATIVES.md
└── examples/
```

## Credits

The derivatives pricing modules and the mean-variance optimiser implement
standard published methods. See [CREDITS.md](CREDITS.md) for the references each
one follows, and for the barrier-convergence finding behind the trinomial
lattice.

## Contributing

New strategies need one method (`raw_signal`) and are automatically subjected to
the causality and null-hypothesis tests: the QA layer does not take anyone's
word that a signal is causal, including mine.

If you find a bug in the leakage detection, that is the most valuable possible
bug report. Open an issue.

## Author and copyright

quantlab is built by
[**Eugene (Yoogeun) Song**](https://www.linkedin.com/in/yoogeunsong) - a PhD
researcher at Imperial College London. It is an independent side project: a
hobby, built in my own time and for my own curiosity, not work carried out for
or on behalf of any employer, university, or client. Nothing in this
repository represents the position of any institution I am affiliated with.

Copyright © 2026 Eugene (Yoogeun) Song. Released under the
[MIT License](LICENSE): use it, fork it, build on it, provided the copyright
notice and permission notice travel with it.

## Disclaimer

Backtested results are hypothetical and do not represent actual trading. Past
performance does not indicate future results. This is research software, not
investment advice.
