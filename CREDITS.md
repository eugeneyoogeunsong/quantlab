# Credits

## Derivatives pricing and mean-variance optimisation: Adrian (`Adrian.ph689`)

The following modules are derived from Adrian's independent quantitative finance
projects (2025), contributed with permission:

| quantlab module | Original project |
|---|---|
| `quantlab/derivatives/analytic.py` | Black-Scholes Implied Volatility Calculator; Evaluation of Numerical Methods for Pricing European Calls; Evaluation of Numerical Methods for Pricing Up-and-Out Calls |
| `quantlab/derivatives/binomial.py` | American & European Option Valuer: Binomial Model; the two numerical-methods studies |
| `quantlab/derivatives/monte_carlo.py` | Both numerical-methods studies |
| `quantlab/derivatives/finite_difference.py` | Both numerical-methods studies |
| `quantlab/derivatives/implied_vol.py` | Black-Scholes Implied Volatility Calculator |
| `quantlab/portfolio/optimisation.py` | Portfolio Analysis using MPT and CAPM-Informed Inputs |

### The original work

**American & European Option Valuer: Binomial Model.** Prices American and
European calls and puts on a vectorised binomial lattice, following Shreve's
*Stochastic Calculus for Finance I* (2004). It demonstrates that the American and
European call coincide without dividends whilst the American put is strictly more
valuable, and it highlights the optimal early-exercise region.

**Black-Scholes Implied Volatility Calculator.** Newton-Raphson inversion of the
Black-Scholes call price using vega, with 2-D and 3-D visualisation of the
implied volatility surface. Built on Wilmott's *Paul Wilmott Introduces
Quantitative Finance*, Ch. 8.

**Evaluation of Numerical Methods for Pricing European Calls.** Benchmarks
binomial trees, Crank-Nicolson finite differences and Monte Carlo against the
Black-Scholes-Merton formula on both accuracy and runtime.

**Evaluation of Numerical Methods for Pricing Up-and-Out Calls.** The same
comparison for barrier options against the closed-form solution, using the
Broadie-Glasserman-Kou adjustment on the lattice, Rannacher timestepping in the
PDE, and Brownian-bridge interpolation in the simulation.

**Portfolio Analysis using MPT and CAPM-Informed Inputs.** Benchmarks CAPM- and
EWMA-informed mean-variance portfolios with quarterly rebalancing against the
S&P 500 and an equal-weight book over 2019–2023, comparing Sharpe maximisation
against variance minimisation on cumulative growth, drawdowns, turnover and
annual Sharpe.

### What changed in the port

The algorithms are Adrian's and are preserved. The refactoring covers packaging
and numerical robustness:

- Standalone Spyder scripts with `#%%` cells and module-level plotting became
  importable, side-effect-free functions.
- Hardcoded parameters became arguments.
- Monte Carlo was vectorised across paths (the original looped path by path;
  clearer to read, roughly 100x slower) and gained optional antithetic variates
  and a reported standard error.
- The finite-difference solver moved from a dense `np.linalg.solve` to a banded
  tridiagonal solve: identical arithmetic, O(n) instead of O(n³) per timestep.
- Implied volatility gained a bisection fallback for the deep ITM/OTM cases where
  vega collapses and Newton-Raphson diverges. The original retried from random
  starting points; bisection is deterministic and cannot fail on a bracketed
  root.
- `trinomial_up_and_out` (Ritchken 1995) was added after testing showed the
  binomial barrier price converges non-monotonically (see below).
- 66 tests were added covering all of the above.

### One substantive numerical change

Testing found that the binomial barrier pricer converges non-monotonically: error
against the closed form moved 1.2e-1 → 1.4e-2 → 2.7e-2 as steps went 500 → 2000 →
5000.

The cause is structural rather than a coding error. A CRR lattice can only take
values `S0·u^k`, the barrier generally falls between two of them, and changing the
step count moves that misalignment erratically. Aligning the barrier by setting
`u = (B/S0)^(1/m)` fixes the placement but shifts the lattice's effective
volatility by ~1%, which does comparable damage. Two branches cannot satisfy both
constraints at once.

Ritchken's stretched trinomial tree adds a third branch and therefore the missing
degree of freedom, matching mean, variance and barrier position simultaneously. It
converges smoothly (1.1e-2 → 4.3e-3 → 2.8e-3 → 1.1e-3 → 7.0e-4 over the same step
counts).

`binomial_up_and_out` is retained, since comparing the two is the point of the
original study, with the limitation documented in its docstring, the
Broadie-Glasserman-Kou continuity correction applied, and a floating-point issue
fixed in the barrier comparison. `trinomial_up_and_out` is the recommended
lattice method.

---

## References

Formulas and methods implemented here come from the following standard sources.

- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate
  Liabilities. *Journal of Political Economy*, 81(3).
- Merton, R. C. (1973). Theory of Rational Option Pricing. *Bell Journal of
  Economics and Management Science*, 4(1).
- Cox, J., Ross, S. & Rubinstein, M. (1979). Option Pricing: A Simplified
  Approach. *Journal of Financial Economics*, 7(3).
- Boyle, P. & Lau, S. H. (1994). Bumping Up Against the Barrier with the
  Binomial Method. *Journal of Derivatives*, 1(4).
- Ritchken, P. (1995). On Pricing Barrier Options. *Journal of Derivatives*,
  3(2).
- Broadie, M., Glasserman, P. & Kou, S. (1997). A Continuity Correction for
  Discrete Barrier Options. *Mathematical Finance*, 7(4).
- Rannacher, R. (1984). Finite Element Solution of Diffusion Problems with
  Irregular Data. *Numerische Mathematik*, 43.
- Shreve, S. E. (2004). *Stochastic Calculus for Finance I: The Binomial Asset
  Pricing Model*. Springer.
- Wilmott, P. (2007). *Paul Wilmott Introduces Quantitative Finance*, 2nd ed.
  Wiley.
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1).
- Sharpe, W. F. (1964). Capital Asset Prices. *Journal of Finance*, 19(3).
- Michaud, R. O. (1989). The Markowitz Optimization Enigma: Is 'Optimized'
  Optimal? *Financial Analysts Journal*, 45(1).
- J.P. Morgan/Reuters (1996). *RiskMetrics Technical Document*, 4th ed.
  (EWMA covariance estimation.)

The factor strategies in `quantlab/research/strategies.py` carry their own
citations in code and in `docs/STRATEGIES.md`.
