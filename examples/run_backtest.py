#!/usr/bin/env python3
# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""End-to-end example: data -> signals -> backtest -> QA -> report.

    python examples/run_backtest.py

The first run needs a network connection to fetch prices; afterwards everything is served
from the on-disk cache. Set ``data_source="synthetic"`` in the config below to skip the
network entirely (useful on a machine with no outbound access, and for reproducible tests).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantlab import Pipeline, PipelineConfig
from quantlab.report import build_report

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")


def main() -> int:
    cfg = PipelineConfig(
        universe="sector_etfs",
        strategy="xs_momentum",
        strategy_params={"lookback": 252, "skip": 21, "top_n": 3},
        start="2012-01-01",
        end="2024-12-31",
        cost_preset="large_cap_equity",   # 10 bps one-way, charged on traded notional
        rebalance="M",
        sizing="equal_weight",
        max_position=0.40,
        max_drawdown_stop=0.25,
        benchmark_symbol="SPY",
    )

    print("Running the full 5-layer pipeline...\n")
    pr = Pipeline(cfg).run()

    pr.qa.print_report()

    print("\nHEADLINE NUMBERS")
    print("-" * 50)
    for k in ("years", "cagr", "volatility", "sharpe", "max_drawdown", "deflated_sharpe"):
        v = pr.stats.get(k)
        s = f"{v:.2%}" if k in ("cagr", "volatility", "max_drawdown") else f"{v:.3f}"
        print(f"  {k:20s} {s}")
    print(f"  {'annual turnover':20s} {pr.result.annual_turnover:.2f}x")
    print(f"  {'annual cost drag':20s} {pr.result.cost_drag_annual:.2%}")

    out = build_report(pr, "reports/xs_momentum.html")
    print(f"\nHTML report: {out.resolve()}")

    if not pr.tradeable:
        print("\nQA gate failed -- treat these numbers as diagnostics, not results.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
