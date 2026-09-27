# Option pricing

`quantlab.pricing` prices a contract four independent ways and makes them
agree. Provenance and references are in [CREDITS.md](../CREDITS.md).

```python
from quantlab.pricing import (
    VanillaOption, BarrierOption, DigitalOption, BlackScholes,
    AnalyticEngine, LatticeEngine, PDEEngine, MonteCarloEngine, PathGenerator,
    implied_volatility, bump_greeks,
)

market   = BlackScholes(spot=100, rate=0.05, vol=0.20, dividend_yield=0.0)
call     = VanillaOption(expiry=1.0, strike=110, option_type="call")

AnalyticEngine().price(call, market)                       # 6.040088, with Greeks
LatticeEngine(n_steps=501).price(call, market)             # 6.040087  (Leisen-Reimer)
PDEEngine(n_space=600, n_time=600).price(call, market)     # 6.0405 ± discretisation
MonteCarloEngine(PathGenerator(100_000, seed=1)).price(call, market)
#                                                          # 6.03 ± 0.02, stderr reported

american = call.with_exercise("american")
LatticeEngine().price(american, market)                    # early exercise handled
PDEEngine(american_solver="psor").price(american, market)  # or as an LCP
MonteCarloEngine().price(american, market)                 # Longstaff-Schwartz

implied_volatility(6.040088, call, market)                 # 0.2 exactly
```

## The three nouns

The design is Hull's: a **contract** is a payoff and an exercise right, a
**model** is market data and dynamics, an **engine** is a numerical procedure
that takes one of each. They never know about each other. That is what lets
one `VanillaOption` go through four engines that share no code, and what lets
the test suite line them up and insist they agree.

| | Knows about | Does not know about |
|---|---|---|
| `Instrument` | strike, expiry, payoff, exercise right, barrier | volatility, rates, numerics |
| `BlackScholes` | spot, rate, vol, dividend yield; forwards, discounts, `d±`; how to simulate itself | what a call is |
| `Engine` | grid sizes, path counts, schemes | market data (the model's), payoffs (the instrument's) |

## The four engines

| Engine | Strength | Cost |
|---|---|---|
| `AnalyticEngine` | Exact; instant; the reference | Only exists for simple payoffs |
| `LatticeEngine` | Early exercise is natural; pluggable schemes | Barriers need the trinomial |
| `PDEEngine` | Whole price surface; barriers are just a boundary; Greeks free | Grid design |
| `MonteCarloEngine` | Path dependence, high dimension | Error falls only as 1/√n |

## Lattice schemes

The engine does backward induction; a `Scheme` supplies the moves and
probabilities. Four binomial schemes differ only in how they match the
lognormal's moments — the induction loop is the same for all of them.

| Scheme | Convergence | Note |
|---|---|---|
| `CoxRossRubinstein` | O(1/n), oscillating | The textbook default |
| `JarrowRudd` | O(1/n), oscillating | Equal probabilities, drift in the moves |
| `Tian` | O(1/n) | Matches three moments |
| `LeisenReimer` | **O(1/n²), monotone** | Centres the tree on the strike; 51 steps beat 1000 of CRR |
| `RitchkenTrinomial` | smooth | For barriers; a node layer sits exactly on the barrier |

Measured on the reference call (S=100, K=110, T=1, r=5%, σ=20%):

```
scheme            n=25       n=101      n=501     err@501
crr               6.080298   6.031102   6.039674  4.1e-04
jarrow-rudd       6.090857   6.037821   6.039958  1.3e-04
tian              6.116874   6.056922   6.041242  1.2e-03
leisen-reimer     6.039517   6.040051   6.040087  1.5e-06
```

## Why barriers need a trinomial

A binomial tree has two free parameters per step and must spend both matching
the mean and variance of the log-return. Nothing is left to control where the
barrier falls, so the barrier the tree *enforces* is whichever node layer is
nearest — and that jumps around as `n` changes. Adding steps does not converge;
it re-rolls the misalignment.

Ritchken's third branch adds a stretch parameter λ ≥ 1 that is free to be
chosen so an integer number of up-moves lands exactly on `ln(H/S₀)`. Moments
still match for any λ. The result converges smoothly:

```
n=250    5.141268   err 1.1e-02
n=500    5.147701   err 4.3e-03
n=1000   5.149159   err 2.8e-03
n=2000   5.150926   err 1.1e-03
n=4000   5.151298   err 7.0e-04          exact: 5.151995
```

An aligned trinomial does **not** want the Broadie-Glasserman-Kou continuity
correction — see CREDITS.md for the measurement. The flag exists; leave it off.

## Monte Carlo

`PathGenerator` draws exact lognormal paths (no Euler bias). `MonteCarloEngine`
composes plugins:

- **Antithetic variates** — pair each `z` with `−z`. The standard error is
  computed over pair-means, not over the 2m correlated samples.
- **Control variate** — the discounted terminal spot has a known expectation
  `S₀e^{−qT}`; regressing the payoff on it and subtracting removes the
  explained variance. Halves the standard error on a vanilla.
- **Brownian-bridge survival weights** for barriers — between two observed
  points on the safe side, the probability the continuous path touched the
  barrier is closed-form. Checking only the grid points misses those crossings
  and overprices knock-outs by ~5%.
- **Longstaff-Schwartz** for American exercise — regress continuation value on
  a polynomial in spot over in-the-money paths, backwards. Biased low by
  construction; the lattice is the check.
- **Pathwise and likelihood-ratio Greeks** — unbiased delta without bumping.
  Pathwise fails on digitals (zero derivative almost everywhere); the
  likelihood-ratio estimator differentiates the density instead and does not
  care about the payoff.

## Implied volatility

Root-finding on a monotone function with Brent's method. Newton on vega is
faster near the money and diverges where vega is tiny — which is exactly the
deep in/out-of-the-money region where quotes are least reliable. Brent gets
Newton's speed where Newton works and bisection's safety where it doesn't.
Prices outside the no-arbitrage bounds return `NaN`.

## What the tests check

Nothing is pinned to a number this code generated. Every reference is external:

- Put-call parity to 1e-12; in-out parity for all four barrier types to 1e-10
- No-arbitrage bounds; monotonicity in vol, spot and strike
- Greek identities: `Δ_call − Δ_put = 1`, `Γ = vega/(S²σT)`, digital = call-spread limit
- American call = European call without dividends; strictly greater with them
- `E[S_T] = S₀e^{(r−q)T}` and `Var[ln S_T] = σ²T` for the simulated paths
- Monte Carlo within 4 standard errors, with SE falling as 1/√n
- Discrete barrier monitoring overprices; the bridge correction does not
- Four engines on a vanilla, four on a barrier, three on an American put
- Implied vol round-trips across 18 (strike, vol) pairs to 1e-8
