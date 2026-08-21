# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Command-line interface: one strategy in full, many strategies quickly, or today's orders.

    python -m quantlab.cli backtest --strategy xs_momentum --universe sector_etfs
    python -m quantlab.cli compare  --universe sector_etfs
    python -m quantlab.cli orders   --strategy ts_momentum --capital 100000

``backtest`` runs the whole checklist and exits non-zero when the QA gate fails, so it drops
straight into CI. ``compare`` and ``orders`` buy speed by skipping the expensive validation
work; both say so in their own output, because a partial verdict quoted as a full one is how
a research pipeline starts lying to its owner.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .pipeline import Pipeline, PipelineConfig
from .report import build_report
from .research.strategies import STRATEGY_REGISTRY
from .data.universe import BUILTIN_UNIVERSES


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_backtest(args) -> int:
    cfg = PipelineConfig(
        universe=args.universe, strategy=args.strategy,
        start=args.start, end=args.end, data_source=args.source,
        cost_preset=args.costs, rebalance=args.rebalance,
        initial_capital=args.capital, sizing=args.sizing,
        vol_target=args.vol_target, benchmark_symbol=args.benchmark,
        run_walk_forward=not args.fast, run_param_sweep=not args.fast,
        run_cost_sensitivity=True,
    )
    pr = Pipeline(cfg).run()

    print()
    pr.qa.print_report()
    print()
    print(f"{'PERFORMANCE':^78}")
    print("=" * 78)
    for k in ("years", "cagr", "volatility", "sharpe", "sortino", "max_drawdown",
              "max_dd_duration_days", "calmar", "deflated_sharpe"):
        v = pr.stats.get(k)
        fmt = f"{v:.2%}" if k in ("cagr", "volatility", "max_drawdown") else (
            f"{v:.3f}" if isinstance(v, float) else str(v))
        print(f"  {k:26s} {fmt}")
    print(f"  {'annual_turnover':26s} {pr.result.annual_turnover:.2f}x")
    print(f"  {'annual_cost_drag':26s} {pr.result.cost_drag_annual:.2%}")
    print("=" * 78)

    if args.output:
        out = build_report(pr, args.output)
        print(f"\nReport written to {out}")

    if not pr.tradeable:
        print("\nQA gate FAILED. These results should not drive decisions.")
        return 1
    return 0


def cmd_compare(args) -> int:
    rows = []
    names = args.strategies or list(STRATEGY_REGISTRY)
    for name in names:
        try:
            cfg = PipelineConfig(
                universe=args.universe, strategy=name,
                start=args.start, end=args.end, cost_preset=args.costs,
                data_source=args.source,
                run_walk_forward=False, run_param_sweep=False,
                run_cost_sensitivity=False, benchmark_symbol=args.benchmark,
            )
            pr = Pipeline(cfg).run()

            # Compare mode skips walk-forward, the parameter sweep and cost
            # sensitivity on purpose: running all three for every strategy is
            # slow. Those Section C/G checks then report "fail" for want of
            # evidence, which is the right answer in the full pipeline and the
            # wrong one here, since they were never run at all. We therefore
            # score only the sections that did run, and say so in the footer.
            ran = [c for c in pr.qa.checks if c.section in ("A", "B", "D", "E", "F")]
            n_fail = sum(1 for c in ran if c.status == "fail")

            rows.append({
                "strategy": name,
                "cagr": round(pr.stats["cagr"], 4),
                "vol": round(pr.stats["volatility"], 4),
                "sharpe": round(pr.stats["sharpe"], 3),
                "max_dd": round(pr.stats["max_drawdown"], 4),
                "calmar": round(pr.stats["calmar"], 3),
                "turnover": round(pr.result.annual_turnover, 2),
                "cost_drag": round(pr.result.cost_drag_annual, 4),
                "qa_partial": "ok" if n_fail == 0 else f"FAIL({n_fail})",
            })
        except Exception as exc:
            rows.append({"strategy": name, "qa_partial": f"ERROR: {exc}"})

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False, na_position="last")
    print()
    print(df.to_string(index=False))
    print(
        "\nNOTE: qa_partial covers data integrity, leakage, costs, risk limits and\n"
        "reporting only. Walk-forward, parameter sensitivity and cost sensitivity\n"
        "were NOT run in compare mode -- so an 'ok' here is not a robustness verdict.\n"
        "Run `backtest --strategy <name>` for the full checklist before trusting any row.\n"
        "Ranking by Sharpe across untested strategies is itself a multiple-testing\n"
        "exercise: the top row is the best of N draws and is biased upward."
    )
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nWritten to {args.output}")
    return 0


def cmd_orders(args) -> int:
    """Target book implied by the latest signal, priced at the last close. Paper only."""
    from .execution.broker import generate_orders, write_order_blotter

    cfg = PipelineConfig(
        universe=args.universe, strategy=args.strategy,
        start=args.start, end=args.end, initial_capital=args.capital,
        data_source=args.source,
        run_walk_forward=False, run_param_sweep=False, run_cost_sensitivity=False,
    )
    pr = Pipeline(cfg).run()

    latest_w = pr.result.weights.iloc[-1]
    latest_px = pr.prices.iloc[-1]
    as_of = pr.result.weights.index[-1]

    orders = generate_orders(
        target_weights=latest_w,
        current_positions={},           # assumed flat; wire a broker in for real state
        prices=latest_px,
        equity=args.capital,
    )

    print(f"\nTarget book as of {as_of:%Y-%m-%d} (signal from that close):")
    for sym, w in latest_w[latest_w.abs() > 1e-6].sort_values(ascending=False).items():
        print(f"  {sym:8s} {w:7.2%}   @ {latest_px[sym]:10,.2f}")

    print(f"\n{len(orders)} order(s) to reach it from a flat book:")
    for o in orders:
        print(f"  {o.side.upper():4s} {o.quantity:10.2f} {o.symbol:8s} "
              f"(${o.notional:,.0f}) -- {o.reason}")

    if args.output:
        write_order_blotter(orders, args.output)
        print(f"\nBlotter written to {args.output}")

    print("\nPaper output only. No orders were placed anywhere.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="quantlab", description="Algorithmic trading research framework")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--universe", default="sector_etfs",
                        choices=sorted(BUILTIN_UNIVERSES))
        sp.add_argument("--start", default="2010-01-01")
        sp.add_argument("--end", default="2024-12-31")
        sp.add_argument("--costs", default="large_cap_equity")
        sp.add_argument("--source", default="yfinance", choices=["yfinance", "synthetic"],
                        help="synthetic = offline simulated data, no network needed")
        sp.add_argument("--benchmark", default="SPY")
        sp.add_argument("-o", "--output")

    b = sub.add_parser("backtest", help="Run one strategy with full QA")
    common(b)
    b.add_argument("--strategy", default="xs_momentum", choices=sorted(STRATEGY_REGISTRY))
    b.add_argument("--rebalance", default="M", choices=["D", "W", "M", "Q"])
    b.add_argument("--capital", type=float, default=1_000_000)
    b.add_argument("--sizing", default="equal_weight",
                   choices=["equal_weight", "inverse_vol", "risk_parity"])
    b.add_argument("--vol-target", type=float, default=None, dest="vol_target")
    b.add_argument("--fast", action="store_true",
                   help="Skip walk-forward and parameter sweep")
    b.set_defaults(func=cmd_backtest)

    c = sub.add_parser("compare", help="Compare all strategies")
    common(c)
    c.add_argument("--strategies", nargs="*", choices=sorted(STRATEGY_REGISTRY))
    c.set_defaults(func=cmd_compare)

    o = sub.add_parser("orders", help="Generate today's paper orders")
    common(o)
    o.add_argument("--strategy", default="xs_momentum", choices=sorted(STRATEGY_REGISTRY))
    o.add_argument("--capital", type=float, default=100_000)
    o.set_defaults(func=cmd_orders)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
