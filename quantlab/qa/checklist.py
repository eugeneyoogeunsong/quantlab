"""The Backtest QA Checklist, as executable code.

Sections A to G of the source checklist, each mapped to a check that either
inspects the run or actively tries to break it.

The design intent: a backtest is not a result until it carries a QA report.
`QAReport.gate()` raises on any FAIL, so a broken run cannot quietly become a
number in a slide deck.

Section G is the section that matters most: "If performance collapses when you
add costs, test OOS, or change parameters, it's probably not robust." We
implement that as a real, run-it-and-see verdict, not as a reminder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..backtest import metrics as M
from ..data.validate import Check

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Section B: bias and leakage, i.e., the failures that flatter a backtest most
# ---------------------------------------------------------------------------

def check_execution_lag(config) -> Check:
    """B1: no look-ahead bias; a signal may only use information available at the time."""
    lag = getattr(config, "execution_lag", 0)
    if lag < 1:
        return Check("B", "Execution lag", "fail",
                     f"execution_lag={lag}. Positions earn the same bar the signal "
                     "was computed from. This is look-ahead bias.")
    return Check("B", "Execution lag", "pass",
                 f"execution_lag={lag} bar(s): signal at close t, position effective t+{lag}.",
                 {"lag": lag})


def check_weight_causality(weights: pd.DataFrame, returns_by_asset: pd.DataFrame,
                           threshold: float = 0.10) -> Check:
    """B2: empirical leakage test; do today's weights anticipate today's returns?

    A causal strategy fixed its weights before the bar's return existed, so the
    contemporaneous cell-by-cell correlation should be small; leakage produces a
    large one.

    Two design points carry the test:

    1. We compute the correlation over ALL cells, including the zero weights,
       rather than over held positions alone. Most strategies equal-weight their
       picks, so among held cells the weight is a constant and the correlation is
       undefined; an earlier version of this check silently degraded to "warn" on
       precisely the strategies it most needed to test. Including the zeros
       restores the variance, and the question becomes "did held names outperform
       unheld names on the same bar", which is exactly the leakage signature.

    2. The threshold is deliberately not zero. A genuinely predictive strategy
       shows a small positive correlation: that is what alpha looks like. Daily
       equity signals typically land in 0.01 to 0.05, whilst leakage lands at 0.3
       and above; the gap is wide enough that 0.10 separates the two cleanly.
    """
    idx = weights.index.intersection(returns_by_asset.index)
    if len(idx) < 30:
        return Check("B", "Weight causality", "warn",
                     "Fewer than 30 overlapping dates; correlation test not meaningful.")

    w, r = weights.loc[idx], returns_by_asset.loc[idx]
    active_dates = w.abs().sum(axis=1) > 1e-9
    if active_dates.sum() < 30:
        return Check("B", "Weight causality", "warn",
                     "Fewer than 30 dates with any position; nothing to test.")

    w, r = w[active_dates], r[active_dates]
    flat_w, flat_r = w.to_numpy().ravel(), r.to_numpy().ravel()
    mask = np.isfinite(flat_w) & np.isfinite(flat_r)
    if mask.sum() < 100:
        return Check("B", "Weight causality", "warn", "Too few finite observations to test.")

    fw, fr = flat_w[mask], flat_r[mask]
    if fw.std() < 1e-12:
        return Check("B", "Weight causality", "pass",
                     "Weights are constant across all assets and dates (e.g. static "
                     "buy-and-hold). There is no cross-sectional bet, so no timing "
                     "information can leak through the weights.")
    if fr.std() < 1e-12:
        return Check("B", "Weight causality", "warn", "Asset returns have zero variance.")

    corr = float(np.corrcoef(fw, fr)[0, 1])
    if not np.isfinite(corr):
        return Check("B", "Weight causality", "warn", "Correlation undefined.")

    if corr > threshold:
        return Check("B", "Weight causality", "fail",
                     f"corr(weight_t, return_t) = {corr:+.3f}, above the {threshold:.2f} limit. "
                     "Weights appear to anticipate same-bar returns -- this is leakage, "
                     "not alpha; real daily signals correlate around 0.01-0.05.",
                     {"correlation": round(corr, 4)})
    return Check("B", "Weight causality", "pass",
                 f"corr(weight_t, return_t) = {corr:+.3f}, consistent with causal execution.",
                 {"correlation": round(corr, 4)})


def check_no_future_features(strategy, prices: pd.DataFrame) -> Check:
    """B3: truncation test applied to the strategy's own weight generation.

    We recompute the weights on the first 70% of the sample and compare them, cell
    by cell, against the same dates taken from the full run: any divergence means
    the strategy is reading data that did not yet exist on those dates.
    """
    try:
        cut = int(len(prices) * 0.7)
        if cut < 50:
            return Check("B", "Feature causality", "warn", "Sample too short to truncate.")
        full = strategy.generate_weights(prices)
        part = strategy.generate_weights(prices.iloc[:cut])
        common = full.index[:cut]
        a = full.loc[common]
        b = part.reindex(index=common, columns=a.columns)
        diff = (a.fillna(-999) - b.fillna(-999)).abs()
        n_bad = int((diff > 1e-9).to_numpy().sum())
        if n_bad:
            where = diff.stack()
            where = where[where > 1e-9]
            d, s = where.index[0]
            return Check("B", "Feature causality", "fail",
                         f"{n_bad} weight(s) changed when future data was withheld. "
                         f"First at {d:%Y-%m-%d}/{s}. The strategy reads the future.",
                         {"n_divergent": n_bad})
        return Check("B", "Feature causality", "pass",
                     "Weights identical when recomputed on truncated history.")
    except Exception as exc:
        return Check("B", "Feature causality", "warn", f"Test could not run: {exc}")


def check_rebalance_timing(config) -> Check:
    """B5: rebalance timing is realistic, i.e., we could actually have traded it."""
    px = getattr(config, "execution_price", "close")
    lag = getattr(config, "execution_lag", 1)
    if px == "close" and lag >= 1:
        return Check("B", "Rebalance timing", "warn",
                     "Executing at the close of the bar after the signal. Workable, but "
                     "assumes you can trade the closing print; next-open is more honest.",
                     {"execution_price": px, "lag": lag})
    return Check("B", "Rebalance timing", "pass",
                 f"Signal at close, execution at next {px} (lag={lag}).",
                 {"execution_price": px, "lag": lag})


# ---------------------------------------------------------------------------
# Section C: validation, i.e., evidence that the result survives outside the fit
# ---------------------------------------------------------------------------

def check_oos_reported(wf_result) -> Check:
    """C1/C2: walk-forward was run, and the out-of-sample results are the ones reported."""
    if wf_result is None:
        return Check("C", "Out-of-sample", "fail",
                     "No walk-forward run. In-sample results alone are not evidence.")
    n = len(wf_result.windows)
    if n < 3:
        return Check("C", "Out-of-sample", "warn",
                     f"Only {n} walk-forward fold(s). Too few to judge consistency.")
    oos_sr = M.sharpe_ratio(wf_result.oos_returns)
    deg, cons = wf_result.degradation, wf_result.consistency

    if oos_sr <= 0:
        return Check("C", "Out-of-sample", "fail",
                     f"OOS Sharpe {oos_sr:.2f} across {n} folds. The strategy does not "
                     "work on data it was not fitted to.",
                     {"oos_sharpe": round(oos_sr, 3), "consistency": cons})
    if deg > 1.0:
        return Check("C", "Out-of-sample", "warn",
                     f"OOS Sharpe {oos_sr:.2f} but degradation {deg:.2f} from train to test. "
                     "Large decay suggests the optimiser is fitting noise.",
                     {"oos_sharpe": round(oos_sr, 3), "degradation": round(deg, 3)})
    return Check("C", "Out-of-sample", "pass",
                 f"OOS Sharpe {oos_sr:.2f} over {n} folds, {cons:.0%} of folds positive, "
                 f"degradation {deg:.2f}.",
                 {"oos_sharpe": round(oos_sr, 3), "consistency": cons})


def check_parameter_sensitivity(verdict: dict | None) -> Check:
    """C3: parameter sensitivity checked; one lucky parameter set is not a result."""
    if verdict is None:
        return Check("C", "Parameter sensitivity", "fail",
                     "No parameter sweep run. A single parameter set proves nothing.")
    v = verdict.get("verdict")
    if v == "robust":
        return Check("C", "Parameter sensitivity", "pass", verdict["detail"], verdict)
    if v == "fragile":
        return Check("C", "Parameter sensitivity", "fail", verdict["detail"], verdict)
    return Check("C", "Parameter sensitivity", "warn", verdict["detail"], verdict)


def check_regimes(verdict: dict | None) -> Check:
    """C4: tested across multiple market regimes (bull, bear, and high-volatility)."""
    if verdict is None:
        return Check("C", "Regime coverage", "warn", "No regime analysis run.")
    v = verdict.get("verdict")
    status = {"robust": "pass", "acceptable": "pass",
              "regime_dependent": "warn", "insufficient": "warn"}.get(v, "warn")
    return Check("C", "Regime coverage", status, verdict["detail"], verdict)


def check_benchmark(stats: dict) -> Check:
    """C5: benchmark comparison included, since absolute return alone says little."""
    if "benchmark_cagr" not in stats:
        return Check("C", "Benchmark", "fail",
                     "No benchmark supplied. Absolute returns are uninterpretable "
                     "without one -- a 12% CAGR is excellent or dreadful depending "
                     "on what buy-and-hold did.")
    excess = stats.get("excess_cagr", 0.0)
    ir = stats.get("information_ratio", 0.0)
    if excess <= 0:
        return Check("C", "Benchmark", "warn",
                     f"Underperforms benchmark by {abs(excess):.1%}/yr (IR {ir:.2f}). "
                     "May still be justified on risk-adjusted grounds -- check drawdown.",
                     {"excess_cagr": round(excess, 4), "information_ratio": round(ir, 3)})
    return Check("C", "Benchmark", "pass",
                 f"Beats benchmark by {excess:+.1%}/yr, information ratio {ir:.2f}.",
                 {"excess_cagr": round(excess, 4), "information_ratio": round(ir, 3)})


def check_multiple_testing(stats: dict, min_dsr: float = 0.90) -> Check:
    """C6: deflated Sharpe, i.e., the Sharpe corrected for how many variants we tried."""
    dsr = stats.get("deflated_sharpe", float("nan"))
    n = stats.get("n_trials_assumed", 1)
    if not np.isfinite(dsr):
        return Check("C", "Multiple testing", "warn", "Deflated Sharpe not computable.")
    if dsr < 0.5:
        return Check("C", "Multiple testing", "fail",
                     f"Deflated Sharpe {dsr:.2f} assuming {n} trials. This result is "
                     "consistent with having found nothing -- the best of N noisy "
                     "strategies looks like this.",
                     {"dsr": round(dsr, 3), "n_trials": n})
    if dsr < min_dsr:
        return Check("C", "Multiple testing", "warn",
                     f"Deflated Sharpe {dsr:.2f} assuming {n} trials, below the "
                     f"{min_dsr:.2f} bar. Weak evidence once selection is accounted for.",
                     {"dsr": round(dsr, 3), "n_trials": n})
    return Check("C", "Multiple testing", "pass",
                 f"Deflated Sharpe {dsr:.2f} assuming {n} trials.",
                 {"dsr": round(dsr, 3), "n_trials": n})


# ---------------------------------------------------------------------------
# Section D: trading reality, i.e., costs, turnover, and when orders actually fill
# ---------------------------------------------------------------------------

def check_costs_included(cost_model_name: str, result) -> Check:
    """D1/D2: transaction costs and slippage are both included, and non-trivial."""
    if cost_model_name == "zero":
        return Check("D", "Transaction costs", "fail",
                     "ZeroCost model. Every number in this report is fictional; "
                     "the only question is by how much.")
    drag = result.cost_drag_annual
    return Check("D", "Transaction costs", "pass",
                 f"Model '{cost_model_name}' applied. Annual cost drag {drag:.2%}, "
                 f"annual turnover {result.annual_turnover:.1f}x.",
                 {"model": cost_model_name, "annual_drag": round(drag, 5),
                  "annual_turnover": round(result.annual_turnover, 2)})


def check_turnover_sanity(result, max_annual_turnover: float = 12.0) -> Check:
    """D3: liquidity and capacity awareness, proxied here by annualised turnover."""
    t = result.annual_turnover
    if t > max_annual_turnover:
        return Check("D", "Turnover", "warn",
                     f"Annual turnover {t:.1f}x exceeds the {max_annual_turnover:.0f}x "
                     "guideline. At this rate cost assumptions dominate the result, "
                     "and small errors in them flip the sign.",
                     {"annual_turnover": round(t, 2)})
    return Check("D", "Turnover", "pass", f"Annual turnover {t:.1f}x.",
                 {"annual_turnover": round(t, 2)})


def check_execution_specified(config) -> Check:
    """D4: execution timing is specified, not left to whatever the engine defaults to."""
    px = getattr(config, "execution_price", None)
    if px not in ("open", "close"):
        return Check("D", "Execution spec", "fail", f"execution_price={px!r} is not specified.")
    return Check("D", "Execution spec", "pass",
                 f"Orders modelled at the {px} of the bar following the signal.")


# ---------------------------------------------------------------------------
# Section E: risk and portfolio controls, both as configured and as observed
# ---------------------------------------------------------------------------

def check_risk_limits(limits, audit: pd.DataFrame | None = None) -> Check:
    """E1 to E3: sizing rule, exposure limits, and drawdown stop defined and honoured."""
    if limits is None:
        return Check("E", "Risk limits", "fail", "No RiskLimits configured.")
    if audit is not None and "breaches" in audit.columns:
        total = int(audit["breaches"].sum())
        if total:
            rows = audit[audit["breaches"] > 0]
            return Check("E", "Risk limits", "fail",
                         f"{total} limit breach(es): "
                         + "; ".join(f"{r.limit} cap {r.cap} observed {r.observed_max}"
                                     for r in rows.itertuples()),
                         {"breaches": total})
    return Check("E", "Risk limits", "pass",
                 f"Position cap {limits.max_position:.0%}, gross cap "
                 f"{limits.max_gross_exposure:.0%}, DD stop {limits.max_drawdown_stop:.0%}. "
                 "No breaches observed.",
                 limits.describe())


def check_rebalance_justified(config, result) -> Check:
    """E4: rebalance frequency justified against the turnover it generates."""
    freq = getattr(config, "rebalance", "?")
    t = result.annual_turnover
    return Check("E", "Rebalance frequency", "pass",
                 f"Rebalance '{freq}' produces {t:.1f}x annual turnover. Verify this is a "
                 "deliberate trade-off between signal freshness and cost, not a default.",
                 {"rebalance": freq, "annual_turnover": round(t, 2)})


# ---------------------------------------------------------------------------
# Section F: reporting, i.e., the metrics a reader needs before forming a view
# ---------------------------------------------------------------------------

def check_reporting_complete(stats: dict) -> Check:
    """F1 to F4: the required metrics are present and finite."""
    required = ["cagr", "volatility", "sharpe", "max_drawdown", "max_dd_duration_days"]
    missing = [k for k in required if k not in stats or not np.isfinite(stats.get(k, np.nan))]
    if missing:
        return Check("F", "Reporting completeness", "fail",
                     f"Missing required metrics: {missing}")
    return Check("F", "Reporting completeness", "pass",
                 f"CAGR {stats['cagr']:.2%}, vol {stats['volatility']:.2%}, "
                 f"Sharpe {stats['sharpe']:.2f}, max DD {stats['max_drawdown']:.2%} "
                 f"lasting {stats['max_dd_duration_days']} days.")


def check_sample_adequacy(stats: dict, min_years: float = 3.0) -> Check:
    """F5: is the track record long enough for the reported Sharpe to mean anything?"""
    years = stats.get("years", 0)
    needed = stats.get("min_track_record_years", float("nan"))
    if years < min_years:
        return Check("F", "Sample adequacy", "warn",
                     f"Only {years:.1f} years. Sharpe standard error is roughly "
                     f"1/sqrt(years) = {1/np.sqrt(max(years,0.1)):.2f} -- wide enough to "
                     "contain almost any conclusion.",
                     {"years": years})
    if np.isfinite(needed) and needed > years:
        return Check("F", "Sample adequacy", "warn",
                     f"{years:.1f} years available, but ~{needed:.1f} years are needed to "
                     "distinguish this Sharpe from zero at 95% confidence.",
                     {"years": years, "needed_years": needed})
    return Check("F", "Sample adequacy", "pass",
                 f"{years:.1f} years of data; sufficient for the observed Sharpe.",
                 {"years": years})


# ---------------------------------------------------------------------------
# Section G: "If this fails, it fails here"
# ---------------------------------------------------------------------------

def check_robustness_triad(cost_v: dict | None, oos_check: Check,
                           param_v: dict | None) -> Check:
    """G: the three-way stress test from the checklist's closing box.

    We require survival on all three axes (costs, out-of-sample, and parameter
    variation); a strategy that collapses on any one of them is not robust, and a
    strategy that passes all three has cleared a necessary bar, not a sufficient one.
    """
    failures, passes = [], []

    if cost_v:
        (failures if cost_v["verdict"] in ("fails_on_costs", "cost_fragile") else passes).append(
            f"costs ({cost_v['verdict']})")
    if oos_check.status == "fail":
        failures.append("out-of-sample")
    elif oos_check.status == "pass":
        passes.append("out-of-sample")
    if param_v:
        (failures if param_v["verdict"] == "fragile" else passes).append(
            f"parameters ({param_v['verdict']})")

    if failures:
        return Check("G", "Robustness triad", "fail",
                     f"Collapses under: {', '.join(failures)}. Per the checklist, "
                     "this is probably not robust. Do not trade it.",
                     {"failed": failures, "passed": passes})
    if len(passes) < 3:
        return Check("G", "Robustness triad", "warn",
                     f"Only {len(passes)}/3 stress tests conclusively passed ({', '.join(passes)}). "
                     "Run the missing ones before drawing conclusions.",
                     {"passed": passes})
    return Check("G", "Robustness triad", "pass",
                 "Survives added costs, out-of-sample testing, and parameter variation. "
                 "That is the bar from the checklist -- necessary, not sufficient.",
                 {"passed": passes})


# ---------------------------------------------------------------------------
# Report: the verdicts collected into one object that can gate a run
# ---------------------------------------------------------------------------

@dataclass
class QAReport:
    """Aggregated checklist verdicts, with the context in which they were produced."""

    checks: list[Check] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def add(self, *checks: Check) -> "QAReport":
        self.checks.extend(c for c in checks if c is not None)
        return self

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warn"]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"section": c.section, "check": c.name, "status": c.status.upper(), "detail": c.detail}
            for c in self.checks
        ]).sort_values(["section", "check"]).reset_index(drop=True)

    def summary_line(self) -> str:
        n = len(self.checks)
        return (f"{n - len(self.failures) - len(self.warnings)}/{n} passed, "
                f"{len(self.warnings)} warning(s), {len(self.failures)} failure(s)")

    def gate(self, allow_warnings: bool = True) -> None:
        """Raise if the run is not fit to be reported.

        Any FAIL blocks unconditionally; set ``allow_warnings=False`` to block on
        warnings too, which is the sensible setting in CI.
        """
        if self.failures:
            lines = "\n".join(f"  - {c}" for c in self.failures)
            raise AssertionError(
                f"QA GATE FAILED -- {len(self.failures)} blocking issue(s):\n{lines}\n\n"
                "These results should not be used to make decisions."
            )
        if not allow_warnings and self.warnings:
            lines = "\n".join(f"  - {c}" for c in self.warnings)
            raise AssertionError(f"QA GATE FAILED (strict mode) -- warnings present:\n{lines}")

    def print_report(self) -> None:
        print("=" * 78)
        print("BACKTEST QA REPORT".center(78))
        print("=" * 78)
        current = None
        titles = {
            "A": "Data Integrity", "B": "Bias + Leakage", "C": "Validation",
            "D": "Trading Reality", "E": "Risk & Portfolio Controls",
            "F": "Reporting", "G": "If this fails, it fails here",
        }
        for c in sorted(self.checks, key=lambda x: (x.section, x.name)):
            if c.section != current:
                current = c.section
                print(f"\nSection {current} -- {titles.get(current, '')}")
                print("-" * 78)
            icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[c.status]
            print(f"  [{icon}] {c.name}")
            for line in _wrap(c.detail, 70):
                print(f"         {line}")
        print("\n" + "=" * 78)
        print(f"RESULT: {self.summary_line()}")
        print("=" * 78)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines
