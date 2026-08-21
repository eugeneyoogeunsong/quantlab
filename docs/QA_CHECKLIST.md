# Backtest QA Checklist: where each item lives in the code

The source checklist, mapped item by item to the function that enforces it.
Nothing here relies on remembering to check something.

Run it:

```python
pr = Pipeline(config).run()
pr.qa.print_report()
pr.qa.gate()                      # raises on any FAIL
pr.qa.gate(allow_warnings=False)  # strict: raises on warnings too
```

**Statuses:** `pass` proceeds · `warn` proceeds with a caveat · `fail` blocks.

---

## Section A: Data Integrity

| Checklist item | Function | Notes |
|---|---|---|
| Prices adjusted for splits/dividends | `check_adjustment` | **FAIL** on raw prices. A 4:1 split reads as −75% and fires every signal you own. |
| No missing dates | `check_missing_dates` | Warns on gaps > 5 days. Weekends and holidays are expected; a two-week hole is not. |
| No bad ticks | `check_bad_ticks` | Flags >50% single-day moves. Never auto-removes: silently deleting outliers is itself a bias, and usually a flattering one. |
| Missing values | `check_missing_values` | Per-symbol NaN density. |
| Corporate actions consistent | `check_adjustment` + `check_bad_ticks` | An unadjusted split shows up as a bad tick. |
| Universe definition explicit | `Universe.rationale` | **Constructor raises** if rationale is empty. Not a check you can skip. |
| Survivorship addressed | `check_survivorship` | Always at least **WARN** for a static list of currently-listed tickers. |
| Sufficient history | `check_sufficient_history` | **FAIL** if history < 2× the signal lookback; there is no out-of-sample left after warm-up. |

`clean_prices()` is deliberately minimal: forward-fill short holes only, never
back-fill (that is look-ahead with a friendly name), drop sub-$1 names.

---

## Section B: Bias + Leakage

| Checklist item | Function | Notes |
|---|---|---|
| No look-ahead bias | `check_execution_lag` | `BacktestConfig` **raises** on `execution_lag=0`. |
| No look-ahead bias (empirical) | `check_weight_causality` | `corr(w_t, r_t)` must be < 0.10. Real daily alpha is 0.01–0.05; leakage is 0.3+. |
| No data leakage | `check_no_future_features` | Truncation test: recompute on 70% of history, require identical overlap. |
| No data leakage (features) | `features.assert_no_lookahead` | Same test, per feature function. |
| Survivorship bias | `check_survivorship` | See Section A. |
| Rebalance timing realistic | `check_rebalance_timing` | Signal at close → fill at next open. Warns if filling at close. |

**Design note.** The correlation check counts *all* cells, including zero weights,
not only held positions. Most strategies equal-weight their picks, so among held
cells the weight is constant and the correlation is undefined; an earlier version
silently degraded to "warn" on exactly the strategies it most needed to test.
Including the zeros reframes the question as "did held names outperform unheld
names on the same bar", which is the leakage signature.

---

## Section C: Validation

| Checklist item | Function | Notes |
|---|---|---|
| Train/test or walk-forward | `walk_forward` | Parameters chosen on train only, with an **embargo** gap before test. |
| Out-of-sample reported | `check_oos_reported` | **FAIL** if OOS Sharpe ≤ 0 or no walk-forward was run. |
| Parameter sensitivity | `parameter_sensitivity` → `sensitivity_verdict` | Looks for a broad plateau. A lone spike with weak neighbours is **FAIL**. |
| Multiple market regimes | `split_regimes` → `regime_performance` | Bull/bear × calm/volatile, labelled **causally** from trailing windows. |
| Benchmark comparison | `check_benchmark` | **FAIL** if no benchmark. Absolute returns are uninterpretable alone. |
| Multiple testing | `check_multiple_testing` | Deflated Sharpe. **FAIL** below 0.5, **WARN** below 0.90. |

**Why the embargo matters.** If a signal uses a 252-day lookback and the test set
begins the day after training ends, the first test observations were computed
mostly from training data; they are not really out-of-sample. The embargo drops a
gap so the lookback window cannot straddle the boundary.

**Honest trial counting.** The pipeline feeds the *actual* number of parameter
combinations evaluated into the deflated Sharpe. Under-reporting trials defeats
the entire correction.

---

## Section D: Trading Reality

| Checklist item | Function | Notes |
|---|---|---|
| Transaction costs included | `check_costs_included` | `ZeroCost` is a **FAIL**, not a warning. |
| Slippage realistic | `SlippageModel` | Fixed bps plus a volatility-proportional term: worse fills on turbulent days, which is when signals want to trade. |
| Liquidity constraints | `SquareRootImpactCost` | Impact ∝ √(participation rate). Reveals capacity limits. |
| Execution timing specified | `check_execution_specified` | **FAIL** if unspecified. |
| Turnover sanity | `check_turnover_sanity` | Warns above 12× annual; cost assumptions then dominate the result. |

Reference costs from the practitioner literature: ~5–7 bps round-trip for
institutional futures, 15–25 bps for retail equities, 40–100 bps for small caps.
The default is 10 bps one-way for large-cap equity.

---

## Section E: Risk & Portfolio Controls

| Checklist item | Function | Notes |
|---|---|---|
| Position sizing rule | `portfolio.sizing` | Equal weight, inverse-vol, risk parity, vol targeting, fractional Kelly. |
| Exposure limits | `apply_exposure_limits` | Gross and net caps. |
| Drawdown controls | `drawdown_control` | **Causal**: exposure at t uses equity through t−1 only. |
| Rebalance frequency justified | `check_rebalance_justified` | Reports the turnover consequence and asks you to confirm it was deliberate. |
| Limits actually honoured | `RiskManager.audit` | Post-hoc breach count. Trust, then verify. |

**Order of application matters** and is not arbitrary: position caps → sector
caps → vol targeting → exposure caps → drawdown control. Applying vol targeting
*after* the exposure cap would let it push the book back above the leverage
ceiling that had just been enforced.

**On drawdown stops.** Whether they help is genuinely contested. They reliably
reduce the depth of the worst drawdown; they also reliably lock in losses at the
bottom and miss the sharpest rebounds, which cluster right after the worst days.
The honest framing is that they buy psychological survivability at some cost in
expected return, and a system you actually stick with beats a better one you
abandon.

---

## Section F: Reporting

| Checklist item | Function |
|---|---|
| Equity curve + drawdowns | `report.build_report` |
| Monthly/annual returns | `monthly_returns_table` |
| Volatility, max DD, Sharpe | `summarize` |
| Turnover + capacity | `BacktestResult.annual_turnover`, `SquareRootImpactCost` |
| Sample adequacy | `check_sample_adequacy` vs `min_track_record_length` |

The HTML report puts the **QA verdict above the equity curve**, deliberately. If a
strategy failed its checks, that should be the first thing a reader sees, not a
footnote beneath a flattering chart.

---

## Section G: "If this fails, it fails here"

> *If performance collapses when you add costs, test OOS, or change parameters →
> it's probably not robust.*

`check_robustness_triad` implements this as a real verdict, not a reminder:

1. **Costs**: `cost_sensitivity()` re-runs at 0/5/10/20/40 bps and finds the
   break-even. Below ~10 bps one-way → `cost_fragile`.
2. **Out-of-sample**: walk-forward OOS Sharpe and train→test degradation.
3. **Parameters**: is there a plateau, or one lucky spike?

**Any leg failing → the whole check fails.** Fewer than three legs conclusively
passing → warn. This is a floor, not a recommendation: passing means the backtest
is not obviously broken, which is a much weaker claim than "this will make
money."

---

## What a good report looks like

- Section A: mostly pass, with survivorship warned honestly.
- Section B: **all pass**. Any failure here invalidates everything downstream.
- Section C: OOS Sharpe positive, degradation < 1.0, broad parameter plateau,
  positive in most regimes, deflated Sharpe > 0.90.
- Section D: real cost model, turnover you can defend.
- Section E: no limit breaches.
- Section F: 5+ years of data, minimum track record satisfied.
- Section G: pass.

If Section B fails, stop. Nothing else in the report means anything.
