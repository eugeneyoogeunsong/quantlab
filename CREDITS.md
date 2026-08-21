# Credits

The derivatives pricing and mean-variance optimisation modules implement standard,
published methods. This file records which sources each one follows, plus the two
implementation decisions that are not obvious from the papers.

## Implementation notes

The formulas are textbook; the choices around them are not, and each was made for
a stated reason:

- Monte Carlo is vectorised across paths rather than looped path by path (roughly
  100x faster for the same arithmetic), with optional antithetic variates and a
  reported standard error.
- The finite-difference solver uses a banded tridiagonal solve rather than a dense
  `np.linalg.solve`: identical arithmetic, O(n) instead of O(n³) per timestep.
- Implied volatility falls back to bisection in the deep ITM/OTM cases where vega
  collapses and Newton-Raphson diverges: slower, but it cannot fail on a bracketed
  root.
- `trinomial_up_and_out` (Ritchken 1995) exists because the binomial barrier price
  does not converge cleanly (see below).

## One substantive numerical finding

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

`binomial_up_and_out` is retained, since the comparison between the two lattices is
worth keeping visible, with the limitation documented in its docstring, the
Broadie-Glasserman-Kou continuity correction applied, and a floating-point issue
fixed in the barrier comparison. `trinomial_up_and_out` is the recommended lattice
method.

---

## References

Formulas and methods implemented here come from the following standard sources.

- Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate
  Liabilities. *Journal of Political Economy*, 81(3).
- Merton, R. C. (1973). Theory of Rational Option Pricing. *Bell Journal of
  Economics and Management Science*, 4(1).
- Cox, J., Ross, S. & Rubinstein, M. (1979). Option Pricing: A Simplified
  Approach. *Journal of Financial Economics*, 7(3).
- Jarrow, R. A. & Rudd, A. (1983). *Option Pricing*. Irwin. (The
  equal-probability tree.)
- Tian, Y. (1993). A Modified Lattice Approach to Option Pricing. *Journal of
  Futures Markets*, 13(5).
- Leisen, D. P. J. & Reimer, M. (1996). Binomial Models for Option Valuation:
  Examining and Improving Convergence. *Applied Mathematical Finance*, 3(4).
- Peizer, D. B. & Pratt, J. W. (1968). A Normal Approximation for Binomial, F,
  Beta, and Other Common, Related Tail Probabilities, I. *Journal of the
  American Statistical Association*, 63(324). (The normal-to-binomial inversion
  Leisen and Reimer build on.)
- Hull, J. C. *Options, Futures, and Other Derivatives*. Pearson. (Greeks read
  off the tree nodes, and the dividend-yield adjustment.)
- Boyle, P. & Lau, S. H. (1994). Bumping Up Against the Barrier with the
  Binomial Method. *Journal of Derivatives*, 1(4).
- Ritchken, P. (1995). On Pricing Barrier Options. *Journal of Derivatives*,
  3(2).
- Broadie, M., Glasserman, P. & Kou, S. (1997). A Continuity Correction for
  Discrete Barrier Options. *Mathematical Finance*, 7(4).
- Rannacher, R. (1984). Finite Element Solution of Diffusion Problems with
  Irregular Data. *Numerische Mathematik*, 43.
- Brennan, M. J. & Schwartz, E. S. (1977). The Valuation of American Put
  Options. *Journal of Finance*, 32(2). (The first finite-difference treatment
  of the early-exercise boundary.)
- Cryer, C. W. (1971). The Solution of a Quadratic Programming Problem Using
  Systematic Overrelaxation. *SIAM Journal on Control*, 9(3). (Projected SOR
  and its convergence.)
- Jaillet, P., Lamberton, D. & Lapeyre, B. (1990). Variational Inequalities and
  the Pricing of American Options. *Acta Applicandae Mathematicae*, 21(3). (The
  equivalence of the optimal-stopping and linear-complementarity formulations.)
- Shreve, S. E. (2004). *Stochastic Calculus for Finance I: The Binomial Asset
  Pricing Model*. Springer.
- Wilmott, P. (2007). *Paul Wilmott Introduces Quantitative Finance*, 2nd ed.
  Wiley.
- Shreve, S. E. (2004). *Stochastic Calculus for Finance II: Continuous-Time
  Models*. Springer.
- Reiner, E. & Rubinstein, M. (1991). Breaking Down the Barriers. *Risk*, 4(8).
  (The closed-form barrier family, with cost of carry.)
- Haug, E. G. (2007). *The Complete Guide to Option Pricing Formulas*, 2nd ed.
  McGraw-Hill. (Second-order Greeks: vanna, volga and charm.)
- Brenner, M. & Subrahmanyam, M. G. (1988). A Simple Formula to Compute the
  Implied Standard Deviation. *Financial Analysts Journal*, 44(5).
- Manaster, S. & Koehler, G. (1982). The Calculation of Implied Variances from
  the Black-Scholes Model: A Note. *Journal of Finance*, 37(1). (The
  vega-maximising starting point for the Newton inversion.)
- Boyle, P. P. (1977). Options: A Monte Carlo Approach. *Journal of Financial
  Economics*, 4(3).
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering*.
  Springer. (Variance reduction Ch. 4, quasi-Monte Carlo Ch. 5, pathwise and
  likelihood-ratio Greeks Ch. 7, American pricing Ch. 8.)
- Longstaff, F. A. & Schwartz, E. S. (2001). Valuing American Options by
  Simulation: A Simple Least-Squares Approach. *Review of Financial Studies*,
  14(1).
- McDonald, R. L. & Schroder, M. D. (1998). A Parity Result for American
  Options. *Journal of Computational Finance*, 1(3).
- Sobol', I. M. (1967). On the Distribution of Points in a Cube and the
  Approximate Evaluation of Integrals. *USSR Computational Mathematics and
  Mathematical Physics*, 7(4).
- Owen, A. B. (1997). Scrambled Net Variance for Integrals of Smooth Functions.
  *Annals of Statistics*, 25(4).
- Markowitz, H. (1952). Portfolio Selection. *Journal of Finance*, 7(1).
- Sharpe, W. F. (1964). Capital Asset Prices. *Journal of Finance*, 19(3).
- Michaud, R. O. (1989). The Markowitz Optimization Enigma: Is 'Optimized'
  Optimal? *Financial Analysts Journal*, 45(1).
- J.P. Morgan/Reuters (1996). *RiskMetrics Technical Document*, 4th ed.
  (EWMA covariance estimation.)

The factor strategies in `quantlab/research/strategies.py` carry their own
citations in code and in `docs/STRATEGIES.md`.
