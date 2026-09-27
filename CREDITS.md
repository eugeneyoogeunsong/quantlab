# Credits and references

## Provenance

An earlier iteration of this repository ported
[Adrian Phillips-Hernaez's](https://github.com/Adrian-pH) public option-pricing
and portfolio-optimisation scripts (2025) into quantlab. That code was removed
and the two packages below rebuilt from the sources listed here, with a
different architecture. The README carries the full statement. His original
repositories:

- American-European-Option-Binomial-Model
- Black-Scholes-Implied-Volatility-Calculator
- Evaluation-of-Numerical-Methods-for-Pricing-European-Calls
- Evaluation-of-Numerical-Methods-for-Pricing-Up-and-Out-Calls
- Portfolio-Analysis-using-MPT-and-CAPM-Informed-Inputs

## `quantlab.pricing`

The design — a contract, a model, and an engine as three separate objects —
follows the way Hull organises the material and the way QuantLib organises
its code.

| Component | Source |
|---|---|
| Black-Scholes-Merton via the forward, Greeks | Hull, *Options, Futures, and Other Derivatives*, Ch. 15, 17, 19 |
| Cash-or-nothing digitals | Hull, Ch. 26 |
| Barrier options by in-out parity from knock-in closed forms | Hull, Ch. 26, presenting Reiner & Rubinstein (1991) |
| Cox-Ross-Rubinstein lattice | Cox, Ross & Rubinstein (1979); Hull, Ch. 21 |
| Jarrow-Rudd equal-probability lattice | Jarrow & Rudd (1983) |
| Tian three-moment lattice | Tian (1993) |
| Leisen-Reimer lattice, Peizer-Pratt inversion | Leisen & Reimer (1996) |
| Stretched trinomial for barriers | Ritchken (1995) |
| Continuity correction (kept as an option, off by default) | Broadie, Glasserman & Kou (1997) |
| Theta-scheme finite differences, boundary conditions | Wilmott, *Paul Wilmott on Quantitative Finance*, Ch. 77 |
| Projected SOR for the American LCP | Wilmott, Ch. 78 |
| Implicit start-up steps to damp Crank-Nicolson ringing | Rannacher (1984) |
| Exact lognormal path generation | Glasserman, *Monte Carlo Methods in Financial Engineering*, Ch. 3 |
| Antithetic variates, control variates | Glasserman, Ch. 4; Hull, Ch. 21 |
| Brownian-bridge barrier crossing probability | Glasserman, Ch. 6.4 |
| Pathwise and likelihood-ratio Greeks | Glasserman, Ch. 7 |
| Longstaff-Schwartz least-squares Monte Carlo | Longstaff & Schwartz (2001); Glasserman, Ch. 8.6 |
| Brent's method for implied volatility | Brent (1973); Hull, Ch. 20 |

## `quantlab.portfolio.meanvariance`

| Component | Source |
|---|---|
| Mean-variance optimisation | Markowitz (1952) |
| Constant-correlation covariance shrinkage, analytic intensity | Ledoit & Wolf (2003), "Honey, I shrunk the sample covariance matrix" |
| Bayes-Stein shrinkage of expected returns | Jorion (1986) |
| CAPM-implied expected returns | Sharpe (1964) |
| EWMA covariance, half-life parameterisation | J.P. Morgan/Reuters, *RiskMetrics Technical Document* (1996) |
| Why any of this is needed | DeMiguel, Garlappi & Uppal (2009), "Optimal versus naive diversification" |

## Numerical findings made during the rebuild

Three things the test suite turned up that are worth knowing if you use this
code.

**An aligned trinomial does not want a continuity correction.** The
Broadie-Glasserman-Kou shift compensates for a barrier that is observed only
at discrete times while the underlying moves continuously. A Ritchken lattice
with a node layer sitting exactly on the barrier has no such gap — a path
cannot reach the far side without landing on the barrier node — so the
monitoring is already effectively continuous. Applying the shift anyway biases
the price low: error 3.1e-1 at 250 steps with the correction versus 1.1e-2
without, and still 7.4e-2 versus 7.0e-4 at 4000 steps. The correction is
retained as a flag for experimentation and defaults to off.

**Richardson extrapolation helps Leisen-Reimer and hurts CRR and
Jarrow-Rudd.** Extrapolation assumes a smooth error in the step size.
Leisen-Reimer's is (it centres the tree on the strike); CRR's and Jarrow-Rudd's
oscillate as the strike drifts between node layers, and extrapolating an
oscillation amplifies it. Measured at 201 steps on a vanilla call: Jarrow-Rudd
1.4e-3 → 3.6e-3, CRR 3.5e-3 → 5.2e-3; Leisen-Reimer at 51 steps 1.4e-4 →
7.1e-5.

**Antithetic samples are pairs, and the standard error has to know that.**
Treating 2m antithetic draws as 2m independent observations reports a standard
error that ignores the variance reduction entirely — the test asserting
`antithetic SE < plain SE` failed on exactly this. The estimator now averages
within each pair and takes the standard error over the m pair-means
(Glasserman, Ch. 4.2).

## Bibliography

- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *J. Political Economy* 81(3).
- Brent, R. P. (1973). *Algorithms for Minimization without Derivatives*. Prentice-Hall.
- Broadie, M., Glasserman, P. & Kou, S. (1997). A Continuity Correction for Discrete Barrier Options. *Mathematical Finance* 7(4).
- Cox, J., Ross, S. & Rubinstein, M. (1979). Option Pricing: A Simplified Approach. *J. Financial Economics* 7(3).
- DeMiguel, V., Garlappi, L. & Uppal, R. (2009). Optimal versus Naive Diversification. *Review of Financial Studies* 22(5).
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Hull, J. C. (2022). *Options, Futures, and Other Derivatives*, 11th ed. Pearson.
- Jarrow, R. & Rudd, A. (1983). *Option Pricing*. Irwin.
- Jorion, P. (1986). Bayes-Stein Estimation for Portfolio Analysis. *J. Financial and Quantitative Analysis* 21(3).
- Ledoit, O. & Wolf, M. (2003). Honey, I Shrunk the Sample Covariance Matrix. *J. Portfolio Management* 30(4).
- Leisen, D. & Reimer, M. (1996). Binomial Models for Option Valuation — Examining and Improving Convergence. *Applied Mathematical Finance* 3(4).
- Longstaff, F. & Schwartz, E. (2001). Valuing American Options by Simulation. *Review of Financial Studies* 14(1).
- Markowitz, H. (1952). Portfolio Selection. *J. Finance* 7(1).
- Merton, R. C. (1973). Theory of Rational Option Pricing. *Bell J. Economics* 4(1).
- Rannacher, R. (1984). Finite Element Solution of Diffusion Problems with Irregular Data. *Numerische Mathematik* 43.
- Reiner, E. & Rubinstein, M. (1991). Breaking Down the Barriers. *Risk* 4(8).
- Ritchken, P. (1995). On Pricing Barrier Options. *J. Derivatives* 3(2).
- Sharpe, W. F. (1964). Capital Asset Prices. *J. Finance* 19(3).
- Tian, Y. (1993). A Modified Lattice Approach to Option Pricing. *J. Futures Markets* 13(5).
- Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance*, 2nd ed. Wiley.

The factor strategies in `quantlab/research/strategies.py` carry their own
citations in code and in `docs/STRATEGIES.md`.
