# Derivatives Pricing

Four independent routes to the same number, plus implied volatility inversion.

Original implementations by **Adrian** (`Adrian.ph689`), 2025 — see
[CREDITS.md](../CREDITS.md).

```python
from quantlab.derivatives import black_scholes_call, binomial_american

black_scholes_call(S=100, K=110, T=1.0, r=0.05, sigma=0.20)   # 6.040088
binomial_american(100, 110, 1.0, 0.05, 0.20, option="put")    # 11.973757
```

---

## Why four methods for one number

Because agreement between independent methods is evidence, and disagreement
localises a bug. Each has a domain where it is the only practical choice:

| Method | Strength | Weakness |
|---|---|---|
| **Analytic** (`analytic.py`) | Exact, instant | Only exists for simple payoffs |
| **Binomial** (`binomial.py`) | Handles early exercise | Poor for barriers; O(n²) |
| **Trinomial** (`binomial.py`) | Handles barriers correctly | More complex |
| **Finite difference** (`finite_difference.py`) | Whole price surface at once; barriers are just a boundary condition | Needs a well-chosen grid |
| **Monte Carlo** (`monte_carlo.py`) | Path dependence, many underlyings | Slow; error only falls as 1/√n |

Analogy: Black-Scholes computes a journey's value knowing only the destination.
The lattice walks every fork in the road — the only way to notice that stopping
early is sometimes better. The PDE maps the whole terrain. Monte Carlo just
drives it ten thousand times and averages.

---

## Verified agreement

European call, S=100, K=110, T=1, r=5%, σ=20%:

| Method | Price | Error vs exact |
|---|---|---|
| Black-Scholes | 6.040088 | — |
| Crank-Nicolson (800×800) | 6.039920 | 1.7e-4 |
| Binomial (2000 steps) | 6.040668 | 5.8e-4 |
| Monte Carlo (200k paths) | 6.032180 | 0.3 standard errors |

Put-call parity holds to 7e-15 — machine precision.

Up-and-out call, S=100, K=95, B=130:

| Method | Price | Error |
|---|---|---|
| Closed form | 5.151995 | — |
| Crank-Nicolson (600×600) | 5.151880 | 1.1e-4 |
| Trinomial (2000 steps) | 5.150926 | 1.1e-3 |
| Monte Carlo + Brownian bridge (200k) | 5.141429 | 0.6 standard errors |

---

## Three subtleties that actually matter

### 1. Barrier options break binomial trees

A CRR lattice can only represent prices `S0·u^k`. The barrier almost never
lands on one, so the barrier the lattice *enforces* is displaced by up to half a
level — and changing the step count moves that displacement erratically.

Measured error against the closed form went **1.2e-1 → 1.4e-2 → 2.7e-2** as
steps rose 500 → 2000 → 5000. More computation, worse answer.

Aligning the barrier via `u = (B/S0)^(1/m)` fixes the placement but throws the
effective volatility off by ~1%, which does comparable damage. Two branches
cannot satisfy both constraints.

**Ritchken's trinomial** adds a third branch — the missing degree of freedom.
Choose a stretch parameter λ ≥ 1 so exactly *m* up-moves land on the barrier:

```
m = floor( ln(B/S0) / (σ√dt) )
λ = ln(B/S0) / (m·σ√dt)

pu = 1/(2λ²) + (r − σ²/2)√dt / (2λσ)
pd = 1/(2λ²) − (r − σ²/2)√dt / (2λσ)
pm = 1 − 1/λ²
```

Barrier aligned exactly, mean and variance matched exactly. Errors then fall
smoothly: 1.1e-2 → 4.3e-3 → 2.8e-3 → 1.1e-3 → 7.0e-4.

Use `trinomial_up_and_out`. `binomial_up_and_out` is kept for comparison, with
the limitation documented in its docstring.

### 2. Discrete monitoring overprices knockouts

Checking the barrier only at simulation grid points misses crossings that
happen *between* observations, so too many paths survive and the option is
overpriced. Measured: 5.428 vs the true 5.152, a 5% overstatement.

The Brownian bridge correction prices those unseen crossings analytically.
Conditional on the two endpoints of a step, the probability the path touched the
barrier is closed-form:

```
p_cross = exp( −2·ln(S_prev/B)·ln(S_next/B) / (σ²·dt) )
```

Each path is weighted by its survival probability instead of being counted as a
binary survivor. With the correction: 5.141 — inside one standard error.

Set `brownian_bridge=False` to reproduce the bias; the test suite asserts it.

### 3. Vega vanishes, and Newton-Raphson goes with it

Implied volatility inverts Black-Scholes using vega as the derivative. Deep in
or out of the money, vega collapses toward zero and the Newton step divides by
almost nothing, sending the iteration somewhere useless.

`implied_volatility` detects this and falls back to **bisection** — slower, but
it cannot diverge on a bracketed root. Prices outside the no-arbitrage bounds
return `NaN` rather than a plausible-looking wrong number.

The original implementation instead retried from several random starting
guesses. Bisection is deterministic and provably convergent, so it is preferred
here; the Brenner-Subrahmanyam approximation supplies the initial guess.

---

## What the tests actually check

Not "does this reproduce the number it produced last time" — nothing is pinned
to a value this code generated. The references are external:

- **Put-call parity**: `C − P = S − K·e^(−rT)`, model-independent, to 1e-12.
- **No-arbitrage bounds**: `max(S − Ke^(−rT), 0) ≤ C ≤ S`.
- **Monotonicity**: price rises in volatility and spot, falls in strike.
- **Greeks vs finite differences**, plus the exact identity `Γ = vega/(S²σT)`.
- **American call = European call** without dividends (Shreve) — early exercise
  is worthless, so the two must agree exactly.
- **American put > European put**, and never below intrinsic value.
- **Barrier < vanilla**, converging to vanilla as `B → ∞`.
- **Monte Carlo within 4 standard errors**, with error falling as 1/√n.
- **`E[S_T] = S₀e^(rT)`** and **`Var[ln S_T] = σ²T`** for the simulated paths.
- **Implied vol round trip**: price at σ, invert, recover σ across strikes.

A note on step sizes: the Greek finite-difference test uses `h ≈ 1e-4·S` for
first derivatives and `h ≈ 1e-2·S` for gamma. Using `h = 1e-5` for a *second*
difference produces a 0.9% discrepancy that looks exactly like a wrong formula
but is pure floating-point cancellation — the numerator is ~2e-12 formed by
subtracting numbers of order 6. Optimal step for a second derivative is
`ε^(1/4)`, not `ε^(1/2)`.

---

## API

```
analytic.py
    black_scholes_call / black_scholes_put   European prices
    black_scholes_greeks                     delta, gamma, vega, theta, rho
    vega                                     dV/dσ
    up_and_out_call_closed_form              barrier, continuous monitoring

binomial.py
    binomial_european / binomial_american    CRR lattice
    binomial_tree_full                       full lattice for plotting
    binomial_up_and_out                      barrier (see limitation above)
    trinomial_up_and_out                     barrier, Ritchken — recommended

finite_difference.py
    crank_nicolson_european                  vanilla, whole surface
    crank_nicolson_up_and_out                barrier, Rannacher start-up

monte_carlo.py
    simulate_gbm_paths                       exact lognormal, optional antithetic
    monte_carlo_european                     optional standard error
    monte_carlo_up_and_out                   optional Brownian bridge

implied_vol.py
    implied_volatility                       Newton-Raphson + bisection fallback
    implied_vol_surface                      grid of spots × expiries
```
