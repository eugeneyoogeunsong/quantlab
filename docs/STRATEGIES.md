# Strategy Catalogue

Strategies implemented in `quantlab`, plus researched candidates not yet built.
Each carries an **evidence grade**, because "published in the Journal of Finance
and replicated across 20 countries for 30 years" and "worked in someone's
backtest" are not the same claim and should not be recorded the same way.

**Evidence grades**

- **Strong** — peer-reviewed, replicated independently, out-of-sample evidence
  across markets and decades.
- **Moderate** — published, but with thinner replication, or components that are
  well-evidenced assembled into a specific recipe that is not.
- **Weak** — plausible mechanism, limited or in-sample-only evidence.
- **Unvalidated** — circulating in practitioner writing without rigorous testing.

A universal caveat: every strategy below was discovered by someone examining
historical data. Publication itself tends to erode returns as capital arrives,
and several of these factors have visibly weaker post-publication performance.

---

## Implemented

### 1. Cross-sectional momentum — `xs_momentum`

**Evidence: Strong**

Rank assets on total return over the past 12 months excluding the most recent
month; hold the top N; rebalance monthly.

Jegadeesh & Titman (1993) found that buying the top decile and shorting the
bottom decile of US stocks by prior 12-month return produced roughly 1% per
month over 1965–1989. It has since been replicated across countries, asset
classes and time periods, and remains one of the most-studied anomalies in
finance.

**The one-month skip is not optional.** Very short-horizon returns exhibit
reversal from microstructure effects — bid-ask bounce, liquidity provision.
Without the skip you are partly betting on one-month reversal, which points the
opposite way and dilutes the signal.

**Known failure mode: momentum crashes.** The strategy suffers rare, violent
losses during sharp market rebounds — 2009 being the canonical example — when
the beaten-down names in the short leg rocket. The long-only implementation here
avoids the worst of that but also forgoes most of the documented spread.

*Parameters:* `lookback=252, skip=21, top_n=3`
*Citation:* Jegadeesh & Titman (1993), *Journal of Finance* 48(1)

---

### 2. Time-series momentum — `ts_momentum`

**Evidence: Strong**

Hold each asset only while its own trailing 12-month return is positive.
Otherwise hold cash.

Moskowitz, Ooi & Pedersen (2012) documented this across 58 futures and forward
contracts spanning equity indices, currencies, commodities and sovereign bonds,
over more than 25 years. The 12-month lookback with 1-month holding produced
significant returns in nearly every asset class, and — importantly —
independently of cross-sectional momentum.

The critical difference from cross-sectional momentum is that this is an
**absolute** rule. Cross-sectional momentum always picks the fastest horse in
the race. Time-series momentum will decline to bet at all if every horse is
walking. That is the source of trend following's reputation for performing well
in crises: the book de-risks to cash during sustained declines.

*Caveat:* post-2010 returns have been noticeably weaker than the published
sample, consistent with either crowding or an unusually trend-hostile decade.

*Parameters:* `lookback=252`
*Citation:* Moskowitz, Ooi & Pedersen (2012), *JFE* 104(2)

---

### 3. Low volatility — `low_vol`

**Evidence: Strong on risk-adjusted basis, weak on absolute return**

Hold the lowest trailing-volatility assets.

Low-beta and low-volatility stocks have historically delivered higher
risk-adjusted returns than CAPM predicts. The leading explanation is leverage
constraints: investors who want higher returns but cannot or will not borrow bid
up high-beta stocks instead, depressing their forward returns. Frazzini &
Pedersen formalised this as betting-against-beta.

**State the caveat plainly:** raw returns are usually *lower* than the market.
The claim is about return per unit of risk. Capturing it in absolute terms
historically required leverage — which reintroduces exactly the risk the
strategy was avoiding, and the leverage constraint is the reason the anomaly
exists in the first place.

*Parameters:* `vol_window=126, top_n=3`
*Citation:* Frazzini & Pedersen (2014); Baker, Bradley & Wurgler (2011)

---

### 4. Short-horizon mean reversion — `mean_reversion`

**Evidence: Moderate — robust gross, frequently negative net**

Buy assets whose price is most depressed relative to a trailing 21-day mean.

Short-horizon reversal over 1 week to 1 month is well documented, and it is
precisely why the 12-1 momentum signal skips the most recent month.

**This is the most cost-sensitive strategy in the library.** It turns over
constantly, and gross edges of a few basis points per trade do not survive
realistic spreads. Watch the cost-drag line and the `cost_sensitivity()` table
before drawing any conclusion — in the framework's own synthetic tests it is
reliably the worst performer once costs are applied.

*Parameters:* `zscore_window=21, entry_z=-1.0, top_n=3`
*Citation:* Lehmann (1990); Lo & MacKinlay (1990)

---

### 5. Dual momentum — `dual_momentum`

**Evidence: Moderate — components strong, specific recipe less so**

Rank assets against each other (relative momentum), then require the winner to
also be beating cash on its own (absolute momentum). If the best available asset
is still falling, hold nothing.

Popularised by Gary Antonacci. The construction is a sensible combination of two
independently well-evidenced effects. The distinction worth preserving: the
*components* have strong evidence; the *specific parameterisation* has a much
thinner out-of-sample record than the underlying literature it draws on.

*Parameters:* `lookback=252, skip=21, top_n=2`
*Citation:* Antonacci (2014), *Dual Momentum Investing*

---

### 6. Buy and hold — `buy_and_hold`

**Benchmark.** Equal-weight everything, always.

QA Section C requires a benchmark comparison, and this is the honest one. If a
strategy cannot beat equal-weight buy-and-hold after costs, its complexity is
not earning anything. On signal-free data it scores the *highest* of all six
strategies — as it must, since active trading on noise can only dilute drift
exposure and add costs.

---

## Researched but not implemented

These need data the free loader does not supply.

### Value (book-to-market)

**Evidence: Strong, but with a long recent drawdown**

Rank on book-to-market or a composite of earnings, cash-flow and sales yields.
Fama & French (1992) established the size and value factors as systematic
sources of return.

Requires fundamental data with **point-in-time** discipline: using restated
financials, or financials before their filing date, is look-ahead bias of the
most flattering kind. Value also suffered an unusually long underperformance
from roughly 2007 to 2020, which is an open question rather than a settled one.

### Quality / gross profitability

**Evidence: Strong**

Rank on gross profits (revenue minus COGS) divided by total assets.

Novy-Marx (2013) showed gross profitability has roughly the same predictive
power for the cross-section of returns as book-to-market, and the two are
negatively correlated — combining them works better than either alone. The
insight is to look at the *top* of the income statement rather than the bottom:
gross profit is harder to manipulate than net earnings.

Requires income statement and balance sheet data.

### Carry

**Evidence: Strong in FX and futures**

Hold high-yielding assets, short low-yielding ones. Needs forward curves or
interest rate differentials.

### Volatility risk premium

**Evidence: Moderate to strong, with severe tail risk**

Systematically selling options tends to earn a premium because implied
volatility exceeds subsequent realised volatility on average. The return
distribution is sharply negatively skewed — many small gains, occasional
catastrophic losses. Backtests of option selling are especially prone to
understating risk, because the tail event may simply not appear in the sample.

Requires options data and a materially different risk framework than the one in
this library.

---

## Combining strategies

Cross-sectional and time-series momentum are documented as *independent*
effects, so combining them is more defensible than combining two variants of the
same idea. Two approaches:

1. **Signal blending** — average standardised signals before ranking.
2. **Portfolio blending** — run each strategy separately, allocate across the
   resulting equity curves (inverse-vol or risk parity across strategies).

Portfolio blending is easier to attribute when something goes wrong, which
matters more than it sounds: when a combined system starts losing money, you
want to know which component is responsible.

Note that combining strategies **increases** your effective trial count. If you
tested five strategies and are now testing combinations of them, the
multiple-testing penalty grows accordingly, and `deflated_sharpe_ratio` should be
given the honest `n_trials`.

---

## Sources

- [Jegadeesh & Titman (1993), Returns to Buying Winners and Selling Losers](https://breesefine7110.tulane.edu/wp-content/uploads/sites/16/2015/10/Momentum-2001.pdf)
- [Momentum: what do we know 30 years after Jegadeesh and Titman's seminal paper?](https://link.springer.com/article/10.1007/s11408-022-00417-8)
- [Moskowitz, Ooi & Pedersen (2012), Time Series Momentum](https://www.sciencedirect.com/science/article/pii/S0304405X11002613)
- [Time Series Momentum (AQR)](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)
- [Novy-Marx (2013), The Other Side of Value: The Gross Profitability Premium](https://mysimon.rochester.edu/novy-marx/research/OSoV.pdf)
- [Betting Against Beta (BAB) Construction — Alpha Architect](https://alphaarchitect.com/betting-against-beta-bab-construction/)
- [Understanding the low volatility anomaly (NYU Stern)](https://pages.stern.nyu.edu/~jwurgler/papers/faj-benchmarks.pdf)
- [Bailey & López de Prado, The Deflated Sharpe Ratio](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- [Modelling Transaction Costs and Market Impact (BSIC)](https://bsic.it/wp-content/uploads/2023/04/Modelling-transaction-costs-for-pdf.pdf)
- [Backtesting Series: Transaction Cost Modelling (BSIC)](https://bsic.it/backtesting-series-episode-5-transaction-cost-modelling/)
