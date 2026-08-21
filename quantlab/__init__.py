"""quantlab: an algorithmic trading research framework organised around what it refuses to do.

A backtest is not evidence; it is a hypothesis with a flattering chart attached. quantlab
therefore couples every performance figure to the checks that qualify it: ``Pipeline.run()``
hands back the equity curve and the QA report in one object, and it is the QA verdict, not the
Sharpe ratio, that decides whether a result may inform a decision. Specifically, we refuse to
(i) quote gross returns, since costs decide whether an edge survives, (ii) report a Sharpe
ratio without the multiple-testing penalty the parameter search has already incurred, and
(iii) pass a strategy tested on one sample, one regime, and one cost assumption. Failing
numbers are still printed, in full: they simply arrive labelled as diagnostics rather than as
results.

The system is five layers, each testable on its own:

    Layer 1  Data        quantlab.data       prices, universe, integrity checks
    Layer 2  Research    quantlab.research   features, signals, strategies
    Layer 3  Backtest    quantlab.backtest   engine, costs, metrics, validation
    Layer 4  Portfolio   quantlab.portfolio  sizing, risk limits
    Layer 5  Execution   quantlab.execution  broker, monitoring
             Reporting   quantlab.report     HTML output
             QA          quantlab.qa         the Backtest QA Checklist, executable

Quick start
-----------
    from quantlab import Pipeline, PipelineConfig

    pr = Pipeline(PipelineConfig(
        universe="sector_etfs",
        strategy="xs_momentum",
        start="2010-01-01", end="2024-12-31",
    )).run()

    pr.qa.print_report()
    print(pr.stats["sharpe"], pr.tradeable)
"""

__version__ = "0.1.0"

from .pipeline import Pipeline, PipelineConfig, PipelineResult

__all__ = ["Pipeline", "PipelineConfig", "PipelineResult", "__version__"]


def __getattr__(name):
    """Lazy re-exports: ``import quantlab`` should not drag in every submodule at start-up."""
    mapping = {
        "BacktestEngine": ("quantlab.backtest.engine", "BacktestEngine"),
        "BacktestConfig": ("quantlab.backtest.engine", "BacktestConfig"),
        "get_strategy": ("quantlab.research.strategies", "get_strategy"),
        "get_universe": ("quantlab.data.universe", "get_universe"),
        "load_prices": ("quantlab.data.loaders", "load_prices"),
        "build_report": ("quantlab.report", "build_report"),
        "QAReport": ("quantlab.qa.checklist", "QAReport"),
    }
    if name in mapping:
        import importlib
        mod, attr = mapping[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'quantlab' has no attribute {name!r}")
