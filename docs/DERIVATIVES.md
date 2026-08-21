# Derivatives Pricing

Four independent routes to the same number (analytic, lattice, PDE, and
simulation), plus implied volatility inversion.

Every method follows a published reference; see [CREDITS.md](../CREDITS.md).

```python
from quantlab.derivatives import black_scholes_call, binomial_american

black_scholes_call(S=100, K=110, T=1.0, r=0.05, sigma=0.20)   # 6.040088
binomial_american(100, 110, 1.0, 0.05, 0.20, option="put")    # 11.974393
black_scholes_call(100, 110, 1.0, 0.05, 0.20, q=0.03)         # 4.797754
```

A continuous dividend yield `q` runs through every pricer in the package. It is
appended last in every signature and defaults to `0.0`, so existing calls are
unchanged: the tests assert that with `==`, not with `approx`, because `r - 0.0`
is exactly `r` in IEEE 754 and anything else would mean the refactor moved
arithmetic it had no business moving. The convention is identical in all four
routes: the risk-neutral drift of `S` is `(r - q)`, whilst discounting stays at
`r`, never at `r - q`. Getting that pairing wrong shifts a price by roughly
`q*T*V`, which no amount of grid refinement removes, so the tests check the two
halves separately (Shreve II, Ch. 5, for the measure change that makes the drift
`r - q` in the first place).

---

## Why four methods for one number

Agreement between independent methods is evidence; disagreement localises a bug.
Each method also owns a domain in which it is the only practical choice:

| Method | Strength | Weakness |
|---|---|---|
| **Analytic** (`analytic.py`) | Exact, instant; the reference everything else is scored against | Only exists for a handful of payoffs |
| **Lattice** (`binomial.py`) | Early exercise node by node; four parameterisations; cheap Greeks | One dimension only; barriers need the trinomial |
| **Trinomial** (`binomial.py`) | The third branch aligns the barrier exactly | Barriers only, in this package |
| **Finite difference** (`finite_difference.py`) | Whole price surface at once; barriers are a boundary condition; American exercise is a pointwise constraint | Needs a well-chosen grid; the free boundary costs order |
| **Monte Carlo** (`monte_carlo.py`) | Path dependence, many underlyings, honest error bars | Slow; error falls only as 1/sqrt(n) |

The geography is worth stating as an analogy: Black-Scholes computes a journey's
value knowing only the destination, whereas the lattice walks every fork in the
road (the only way to notice that stopping early is sometimes better). The PDE
maps the whole terrain; Monte Carlo simply drives the route ten thousand times
and averages.

### Choosing between them

The four are not interchangeable, and the differences are the point:

- **One-dimensional American contract, price only.** Use the lattice.
  `binomial_american` at 5,000 steps settles to 11.97284 on the reference put
  (stable to 1e-4 against 40,000 steps) in about a quarter of a second. Nothing
  else here is competitive on that problem, and `longstaff_schwartz_american`
  in particular is not (see below).
- **You want the Greeks, the exercise boundary, or a barrier.** Use the PDE. One
  backward integration produces the value at every node and every timestep, so
  delta, gamma and theta are differences of a surface we already have
  (`crank_nicolson_greeks`), the free boundary falls out of the projection
  (`crank_nicolson_american(..., return_boundary=True)`), and a knockout is
  simply a Dirichlet condition on the domain edge.
- **Path dependence, or more than about two underlyings.** Use simulation, and
  only then. On a vanilla European it is the slowest and least accurate route
  here, which is exactly why it should never be used for one.
- **Anything at all, as a check.** Use the closed form. A numerical method that
  cannot reproduce Black-Scholes on a vanilla call has not earned the right to
  be trusted on an exotic one.

---

## Verified agreement

European call, S=100, K=110, T=1, r=5%, sigma=20%, q=0:

| Method | Price | Error vs closed form |
|---|---|---|
| Black-Scholes | 6.040088 | reference |
| Leisen-Reimer lattice (801 steps) | 6.040088 | 5.9e-7 |
| Crank-Nicolson (800x800) | 6.039920 | 1.7e-4 |
| Cox-Ross-Rubinstein lattice (801 steps) | 6.040704 | 6.2e-4 |
| Monte Carlo (200k paths, seed 3) | 6.042892 | 0.11 standard errors |

```python
from quantlab.derivatives import (black_scholes_call, binomial_european,
                                  crank_nicolson_european, monte_carlo_european)

args = (100.0, 110.0, 1.0, 0.05, 0.20)
black_scholes_call(*args)                                    # 6.040088
binomial_european(*args, n_steps=801, method="lr")           # 6.040088
crank_nicolson_european(*args, n_space=800, n_time=800)      # 6.039920
monte_carlo_european(*args, n_paths=200_000, seed=3, return_stderr=True)
#                                          (6.042892, 0.025985)
```

Put-call parity holds to 7e-15: machine precision, at any volatility and any
dividend yield (`black_scholes_put_call_parity_residual`).

Up-and-out call, S=100, K=95, B=130, T=1, r=5%, sigma=20%:

| Method | Price | Error |
|---|---|---|
| Closed form (reflection principle) | 5.151995 | reference |
| Crank-Nicolson (600x600) | 5.151880 | 1.1e-4 |
| Trinomial, Ritchken (2000 steps) | 5.150926 | 1.1e-3 |
| Monte Carlo + Brownian bridge (200k, seed 5) | 5.163223 | 0.65 standard errors |
| CRR binomial (2000 steps) | 5.213039 | 6.1e-2 |

The last row is not a bug, and it is the subject of the first subtlety below.

---

## The lattice: which parameterisation, and why it matters

A recombining tree carries three free numbers per step (u, d, p) and only two
constraints (the mean and the variance of the log-price over `dt`, the second of
which need only hold to leading order). One degree of freedom is therefore left
over, and each named method spends it differently: `crr` imposes `d = 1/u`, `jr`
imposes `p = 1/2`, `tian` matches the third moment as well, and `lr`
(Leisen-Reimer) spends it on centring the tree on the strike via a Peizer-Pratt
inversion of the normal CDF. Shreve I, Ch. 1 to 2, is the underlying replication
argument; Ch. 4 is the optimal-stopping problem that makes early exercise
natural on a tree.

Only the last of those changes the convergence *order*, and that is the whole
argument for having four of them. Absolute error on the reference European call,
over the step ladder n = 51, 101, 201, 401, 801, 1601:

```python
from quantlab.derivatives import binomial_european, black_scholes_call

exact = float(black_scholes_call(100, 110, 1.0, 0.05, 0.20))
for method in ("crr", "jr", "tian", "lr"):
    errors = [abs(binomial_european(100, 110, 1.0, 0.05, 0.20,
                                    n_steps=n, method=method) - exact)
              for n in (51, 101, 201, 401, 801, 1601)]
    print(method, " ".join(f"{e:.1e}" for e in errors))

# crr    2.0e-3   9.0e-3   3.5e-3   1.8e-3   6.2e-4   1.2e-3
# jr     2.8e-2   2.3e-3   1.4e-3   1.7e-3   5.6e-4   1.0e-3
# tian   3.6e-2   1.7e-2   7.1e-3   2.1e-3   3.3e-4   9.5e-4
# lr     1.4e-4   3.7e-5   9.3e-6   2.4e-6   5.9e-7   1.5e-7
```

Read the last column first: CRR, JR and Tian are all *less* accurate with 1601
steps than they were with 801, whilst LR's error ratio per doubling runs 3.86,
3.93, 3.96, 3.98, 3.99, converging on the factor of 4 that defines O(1/n^2). At
n = 801 LR is ~1000x closer than CRR; at n = 1601, ~8000x.

**Why Leisen-Reimer is the one worth reaching for.** The accuracy is the
headline, but the monotonicity is the property one actually pays for. An
oscillating error is not merely untidy: it means the change between two step
counts is worthless as an error estimate (CRR's error *grows* 4.5x between
n = 51 and n = 101, then falls again), and it leaves extrapolation with nothing
stable to extrapolate.
The cause is simple. A CRR, JR or Tian lattice knows nothing about the strike,
so as `n` changes the strike drifts across the terminal nodes, the payoff kink
lands at a different place inside a cell each time, and the quadrature error of
the terminal distribution jumps with it. Leisen and Reimer remove the problem at
its source by choosing `u` and `d` so the strike sits at the centre of the
terminal distribution for every `n`.

Two caveats come with LR, and both are enforced rather than hidden. First, the
Peizer-Pratt inversion is defined for an *odd* number of steps, so an even
`n_steps` is rounded up: used as-is, an even-step LR tree is out by 4.9e-3 at
n = 802 against 5.9e-7 at n = 801, because the strike then lands exactly on the
terminal node the construction exists to avoid. Second, LR buys its accuracy by
ordering the terminal nodes, not by being a better one-step approximation: its
per-step variance is off by 5.7e-3 relative at n = 101 against CRR's 2.2e-4.
That error decays as O(dt) and is repaid many times over, but it is real.

### Richardson extrapolation helps LR and misleads CRR

`binomial_richardson` forms `2*P(2n) - P(n)`, which cancels a leading `C/n`
error term. The premise is that the error is a smooth function of `1/n` with a
stable sign, and that premise is exactly what CRR violates:

```python
from quantlab.derivatives import (binomial_european, binomial_richardson,
                                  black_scholes_call)

exact = float(black_scholes_call(100, 110, 1.0, 0.05, 0.20))
for method in ("lr", "crr"):
    base = abs(binomial_european(100, 110, 1.0, 0.05, 0.20,
                                 n_steps=51, method=method) - exact)
    extra = abs(binomial_richardson(100, 110, 1.0, 0.05, 0.20,
                                    n_steps=51, method=method) - exact)
    print(f"{method}: {base:.2e} -> {extra:.2e}  ({base / extra:.2f}x)")
# lr:  1.42e-04 -> 7.11e-05  (1.99x)
# crr: 1.98e-03 -> 2.54e-02  (0.08x)
```

LR gains a clean factor of two at n = 51, 101, 201 and 401 (measured 1.99, 2.00,
2.00, 2.00). CRR is made 12.8x *worse* at n = 51 and is worse than doing nothing
in five of the eight step counts tested, because differencing two step counts
amplifies an oscillation instead of cancelling a trend. `binomial_richardson`
therefore defaults to `method="lr"` (unlike `binomial_european` and
`binomial_american`, which keep `"crr"` so that existing calls are unchanged),
and extrapolating a CRR tree should be treated as unsupported rather than merely
inadvisable.

Why the gain on LR is 2x and not a whole order: LR's leading error is O(1/n^2),
so the classical two-point weights are matched to the wrong power and halve the
term rather than annihilating it. The order-matched combination
`(4*P(2n) - P(n))/3` reaches 1.9e-7 at n = 51 where this one reaches 7.1e-5; a
caller who knows their scheme is second-order can build it from two calls to
`binomial_european`.

### The convergence table does not carry over to American contracts

An American option's error is dominated by the early-exercise boundary, about
which the Leisen-Reimer construction says nothing. Measured on the standard put
against a 40,000-step reference, LR's error at n = 51 is 2.6e-2 against CRR's
9.4e-3; by n = 801 the two are level (1.2e-3 and 7.0e-4). Use `lr` for European
contracts, and do not expect it to pay for itself on American ones.

### Tree Greeks

`binomial_greeks` returns the same five keys as `black_scholes_greeks`, so the
two are directly comparable. Delta, gamma and theta are read off nodes the
backward induction has already computed and are therefore essentially free; vega
and rho are central differences over re-priced trees, i.e. four extra lattices.

```python
from quantlab.derivatives import binomial_greeks, black_scholes_greeks

exact = black_scholes_greeks(100, 110, 1.0, 0.05, 0.20, option="call")
for method in ("crr", "lr"):
    tree = binomial_greeks(100, 110, 1.0, 0.05, 0.20, n_steps=501,
                           option="call", method=method)
    print(method, " ".join(f"{k} {abs(tree[k] - exact[k]):.1e}" for k in exact))
# crr delta 2.9e-04 gamma 1.8e-05 vega 5.7e-01 theta 2.7e-03 rho 1.7e-02
# lr  delta 3.5e-04 gamma 1.2e-05 vega 9.1e-06 theta 4.5e-03 rho 5.4e-06
```

Delta, gamma and theta are equally good on either tree (their error is the O(dt)
truncation of the node differences, common to all four parameterisations), but
CRR's vega is out by 1.4% relative. The reason is worth internalising: a finite
difference in sigma differences the lattice's *own* error as well as the price,
and CRR's error swings with sigma. Use `method="lr"` whenever vega or rho is
wanted.

One implementation detail is load-bearing rather than cosmetic. Theta compares
the central step-2 node with the root, and that node sits at `S0*u*d`, which
equals `S0` only in a CRR tree. The pricer subtracts `delta*(S_ud - S0)` before
dividing by `2*dt`; drop that correction and theta reads -1.62 under `lr` and
-4.56 under `jr` against a true -5.90, since a displacement of order `dt`
divided by `2*dt` contaminates the answer at O(1).

---

## The PDE: the whole surface, and free boundaries

Rather than simulating or enumerating outcomes, `finite_difference.py` solves
the Black-Scholes equation on a grid of (spot, time), integrating backwards from
the payoff. Feynman-Kac (Shreve II, Ch. 6) is the bridge between that equation
and the expectation the other three modules compute; Wilmott, Ch. 8, is the
source for the schemes, their stability, and projected SOR.

### The theta family

Every scheme here is a weighted average of the explicit and implicit Euler
steps, selected by `theta_scheme`, the weight on the *old* time level:

```
theta_scheme = 0     fully implicit,   O(dt),    unconditionally stable
theta_scheme = 0.5   Crank-Nicolson,   O(dt^2),  unconditionally stable  (default)
theta_scheme = 1     fully explicit,   O(dt),    stable only under a step limit
```

```python
from quantlab.derivatives import crank_nicolson_european

crank_nicolson_european(100, 110, 1.0, 0.05, 0.20, 400, 400)                    # 6.038623
crank_nicolson_european(100, 110, 1.0, 0.05, 0.20, 400, 400, theta_scheme=0.0)  # 6.037062

try:
    crank_nicolson_european(100, 110, 1.0, 0.05, 0.20, 400, 400, theta_scheme=1.0)
except ValueError as exc:
    print(exc)   # dt=2.500e-03 exceeds the limit ... =1.562e-04
```

Dropping to first order costs a factor of two on this grid (3.0e-3 against
1.5e-3). The explicit end is refused rather than run: von Neumann analysis gives
`dt <= 1/(sigma^2 * n_space^2 * (2*theta_scheme - 1))`, which at sigma = 0.2 and
n_space = 400 caps `dt` at 1.6e-4, i.e. more than 6400 timesteps for a one-year
contract. Violating it does not degrade gracefully: the solution overflows to
NaN within a few dozen steps and surfaces as an opaque error from the banded
solver several layers down, so we say why up front.

Crank-Nicolson buys its second order at a price worth knowing: its amplification
factor tends to -1 rather than 0 for the highest-frequency modes, so a kink or a
jump in the initial data rings rather than decays. Rannacher (1984) start-up,
two fully implicit steps before switching, restores the damping without giving
up the second-order tail. `crank_nicolson_up_and_out` uses it unconditionally,
which is why that pricer does not expose `theta_scheme`: the scheme is chosen by
the discontinuity, not by the caller.

### American exercise as a linear complementarity problem

An American option is an optimal-stopping problem (Shreve I, Ch. 4; Shreve II,
Ch. 8). Stated as "solve the PDE on the continuation region" it is circular,
because the region is defined by the answer. The complementarity form removes
the circularity by posing the constraint pointwise on the whole domain: with `L`
the Black-Scholes operator and `g` the intrinsic value,

```
(i)   V >= g
(ii)  dV/dtau - L V >= 0
(iii) (V - g) * (dV/dtau - L V) = 0
```

`crank_nicolson_american` solves the discretised version by projected SOR: the
Gauss-Seidel sweep is sequential, so each node can be projected onto `V >= g`
the moment its new value is computed, which makes the projection part of the
solve rather than a repair applied to a finished answer. The exercise boundary
is then read off afterwards.

```python
from quantlab.derivatives import crank_nicolson_american, binomial_american

price, tau, S_free = crank_nicolson_american(100, 110, 1.0, 0.05, 0.20, 400, 400,
                                             option="put", return_boundary=True)
price                        # 11.971069
S_free[0], S_free[-1]        # 107.8 just before expiry, 89.1 at tau = 1
binomial_american(100, 110, 1.0, 0.05, 0.20, n_steps=10_000, option="put")
#                            # 11.972845, from entirely different mathematics
```

The obvious alternative, not implemented here, is to take a plain
Crank-Nicolson step and then set `V <- max(V, g)`. That scheme is valid and it
converges, but it is the weaker method for a structural reason: the implicit half of the step couples every node
to every other node *within* the step, so nodes that should have been pinned at
intrinsic instead feed sub-intrinsic values to their neighbours, and the final
`max()` repairs the node itself but not the contamination it has propagated.
Measured against a 40,000-step binomial on an aligned grid:

```
n_space = n_time     100       200       400       800
PSOR error         3.0e-2    9.8e-3    2.7e-3    6.7e-4
shortcut error     3.9e-2    1.4e-2    4.9e-3    1.8e-3
```

PSOR divides its error by 3.1, 3.7, then 4.0 per halving of `dS`; the shortcut
manages only 2.8, 2.8, 2.7, so the gap *widens* with refinement rather than
closing. The shortcut is roughly 12x faster per solve, which is a real argument
for it when a coarse American price is all that is wanted, and not an argument
for it when the free boundary, the early-exercise premium, or gamma near the
boundary is the quantity of interest.

Three practical notes. The error is dominated by the spatial grid, not the
timestep (taking `n_time` from 100 to 1600 moved the price by 1.9e-4, whilst
taking `n_space` from 100 to 800 moved it by 3.0e-2), so spend the budget on
`n_space`. Set `S_max` so that `S0` lands exactly on a node whenever possible,
since the final price is read off by linear interpolation and that bias has the
opposite sign to the free-boundary error, making the total oscillate on an
unaligned grid. Finally, `omega` changes how fast we converge and never what we
converge to: across omega in {0.9, 1.2, 1.6} the price spread is 7.6e-8, of the
order of `tol` itself, which is the evidence that we are solving the LCP rather
than reporting wherever the relaxation happened to stop.

### Greeks off the grid

This is the argument for taking the PDE route at all. Delta and gamma are
central differences of the final slice, theta is the difference of the last two
time slices, and only vega and rho need re-solving (sigma and `r` are parameters
of the operator rather than coordinates of the grid).

```python
from quantlab.derivatives import crank_nicolson_greeks, black_scholes_greeks

fd = crank_nicolson_greeks(100, 110, 1.0, 0.05, 0.20, 400, 400, option="call")
bs = black_scholes_greeks(100, 110, 1.0, 0.05, 0.20, option="call")
print({k: float(f"{fd[k] - bs[k]:.2e}") for k in bs})
# {'delta': -9.64e-05, 'gamma': 3e-06, 'vega': 0.0048, 'theta': -0.00187, 'rho': -0.00343}
```

Accuracy is not uniform across the five, and the differences are worth knowing
before quoting any of them. On a 400x400 grid the worst case over calls, puts
and `q` in {0, 4%} is: delta 9.6e-5, gamma 3.0e-6, vega 4.8e-3 (1.2e-4
relative), rho 3.4e-3 (8.8e-5 relative), theta 2.5e-3. Theta is the weak one by
construction: a two-slice difference is only O(dt) when reported at tau = T,
because formally it is a centred estimate at tau - dt/2. Its error falls only as
fast as 1/sqrt(n_time) rather than as 1/n_time^2 (1.9e-3, 7.0e-4, 4.1e-4 at
n_time = 400, 1600, 6400), so buy accuracy there with timesteps and nowhere else.

---

## Simulation: for the problems the other two cannot reach

Monte Carlo is the slowest and least accurate method here on a vanilla
European. It earns its place on payoffs that depend on the whole path, or on
many underlyings at once. Its error shrinks as 1/sqrt(n_paths): 100x the compute
buys 10x the accuracy, which is a poor exchange rate and the reason variance
reduction exists at all (Glasserman, Ch. 4).

### Variance reduction

`antithetic=True` (the default on `monte_carlo_european`) pairs each path with
its mirror image. `control_variate=True` subtracts an optimally-scaled control
with a known mean: the discounted terminal underlying for the European pricer,
the vanilla European call for the barrier pricer. The gain tracks
`corr(Y, X)` and nothing else, so it is large where the payoff nearly *is* the
control and negligible where it is not:

```python
from quantlab.derivatives import monte_carlo_european

for strike in (80.0, 100.0, 110.0, 130.0):
    _, plain = monte_carlo_european(100.0, strike, 1.0, 0.05, 0.20, n_paths=200_000,
                                    n_steps=50, seed=6, return_stderr=True)
    _, cv = monte_carlo_european(100.0, strike, 1.0, 0.05, 0.20, n_paths=200_000,
                                 n_steps=50, seed=6, return_stderr=True,
                                 control_variate=True)
    print(strike, f"{plain / cv:.2f}x")
# 80.0 7.99x   100.0 2.62x   110.0 1.90x   130.0 1.30x
```

Deep in the money the payoff is `S_T - K` on nearly every path and the control
removes nearly everything; deep out of the money it is zero on nearly every
path, the linear projection has almost nothing to explain, and the technique
earns almost nothing. The same collapse happens on the barrier pricer as the
knockout approaches the money: corr(Y, X) runs -0.03 / 0.15 / 0.39 / 0.61 / 0.93
for B = 120 / 130 / 140 / 150 / 180, giving reductions of 1.00x / 1.01x / 1.09x
/ 1.26x / 2.76x. A near barrier knocks out exactly the paths on which the
vanilla pays most.

`b*` is fitted on the same paths it prices on, which makes the estimator a ratio
of correlated sample moments and therefore biased. We measured the bias rather
than assuming it away: over 2,000 replications the paired difference is
-2.0e-3 +/- 4.9e-3 at 2,000 paths and +1.9e-3 +/- 1.6e-3 at 20,000, against
per-run standard errors of 0.136 and 0.044. It is O(1/n) whilst the standard
error it buys down is O(1/sqrt(n)), so we spend the paths on variance instead.

### Quasi-Monte Carlo

`sobol=True` replaces the pseudo-random draws with a scrambled Sobol' net
inverted through the normal quantile function. It requires `n_paths` to be a
power of two (the equidistribution properties only hold on 2^m points) and
cannot be combined with `antithetic`, since mirroring a balanced net destroys
its balance. Because the points are deliberately dependent, `return_stderr=True`
returns NaN rather than a sample standard deviation that would look like a
confidence measure and would not be one:

```python
from quantlab.derivatives import monte_carlo_european

monte_carlo_european(100, 110, 1.0, 0.05, 0.20, n_paths=2**12, n_steps=8,
                     seed=1, sobol=True, return_stderr=True)   # (6.038672, nan)
```

To get an honest error bar under QMC, average several independent scrambles and
take the standard deviation across replications (Glasserman, Ch. 5.4). The gain
is real but dimension-dependent: over 2^10 to 2^16 points a scrambled net cuts
the RMSE by 10x to 23x at `n_steps=8` and by only 2.4x to 6.3x at `n_steps=64`.
Recovering it at 252 steps needs a Brownian-bridge construction, which is not
implemented here.

### Longstaff-Schwartz, and why it is the wrong tool here

`longstaff_schwartz_american` estimates the continuation value by
cross-sectional regression on the in-the-money paths, working backwards from
expiry (Longstaff & Schwartz 2001; Glasserman, Ch. 8). It is implemented, it is
correct, and on a one-dimensional American put it is the wrong choice:

```python
from quantlab.derivatives import longstaff_schwartz_american, binomial_american

longstaff_schwartz_american(100, 110, 1.0, 0.05, 0.20, n_paths=50_000,
                            n_steps=50, seed=3, return_stderr=True)
#                                       (11.911390, 0.038351)
longstaff_schwartz_american(100, 110, 1.0, 0.05, 0.20, n_paths=50_000,
                            n_steps=50, seed=3, out_of_sample=True)
#                                       11.951182
binomial_american(100, 110, 1.0, 0.05, 0.20, n_steps=5_000, option="put")
#                                       11.972840
```

The lattice settles to 11.97284 (stable to 1e-4 against 40,000 steps) in about
0.25 s; 50,000 LSM paths across 50 exercise dates cost the same order of
compute and land 0.024 low averaged over eight seeds, with a per-run standard
error of 0.038. Same compute, two orders of magnitude more error. LSM exists for
problems a lattice cannot reach at all: a basket of five assets, a swaption in a
multi-factor model, anything where the grid would need five dimensions. We
implement it on the one-dimensional case because that is where it can be checked
against an independent method, not because it is competitive there.

Two biases sit underneath the noise, and they point in opposite directions.
(i) The fitted rule is suboptimal, and any suboptimal stopping rule undervalues
the option, so the price is pushed down. (ii) The rule is fitted on the very
paths it is then evaluated on, so it exercises with partial foresight of that
sample's noise, which pushes the price up. At realistic path counts the first
dominates. Most of the remaining gap is the Bermudan restriction rather than the
regression: raising the exercise dates from 25 to 100 moves the mean error from
-0.036 to -0.014. `out_of_sample=True` freezes the coefficients, draws fresh
paths from a spawned seed, and replays the rule forward, which removes the
second bias outright and leaves a genuine lower bound on the American value.

### Simulated Greeks: pathwise for delta, likelihood-ratio for gamma

This distinction is the most instructive thing in the module (Glasserman,
Ch. 7.2 and 7.3). The pathwise method differentiates the payoff along the path
and takes the expectation of the derivative, swapping the order of
differentiation and expectation. That swap is legitimate only when the payoff is
Lipschitz in the parameter. The call payoff qualifies: it is continuous, and
non-differentiable only at `S_T = K`, which the lognormal law assigns
probability zero. Under GBM `dS_T/dS0 = S_T/S0`, so the pathwise delta is
`exp(-r*T) * 1{S_T > K} * S_T/S0`, and vega follows the same way.

Gamma is where the method breaks, and not by a little. It is the second
derivative of the payoff, and the second derivative of `(S_T - K)^+` is a Dirac
delta at the strike: it is not a function, its pathwise "estimator" is zero on
every path that does not land exactly on `K`, and no amount of sampling recovers
the mass concentrated at that point. Differentiating the *density* instead
sidesteps this entirely, because the lognormal density is smooth in `S0` however
rough the payoff is. `monte_carlo_greeks` therefore returns pathwise delta and
vega, a likelihood-ratio gamma, and a likelihood-ratio delta alongside for
comparison:

```python
from quantlab.derivatives import monte_carlo_greeks, black_scholes_greeks

mc = monte_carlo_greeks(100, 110, 1.0, 0.05, 0.20, n_paths=200_000,
                        seed=13, return_stderr=True)
mc["delta"], mc["delta_stderr"]          # 0.449575, 0.001321   (pathwise)
mc["delta_lr"], mc["delta_lr_stderr"]    # 0.451819, 0.002741   (likelihood ratio)
mc["gamma"], mc["gamma_stderr"]          # 0.019953, 0.000268
black_scholes_greeks(100, 110, 1.0, 0.05, 0.20)["gamma"]   # 0.019788
```

The trade-off goes the way the theory predicts. Where both estimators are valid
the pathwise one carries a standard error 2.1x lower (0.0013 against 0.0027 on
delta at 200,000 paths). Likelihood ratio is not a better estimator; it is the
estimator that survives a payoff the pathwise method cannot differentiate. The
rule follows: use pathwise wherever the payoff is smooth enough to allow it, and
keep likelihood ratio for gamma, digitals, and anything with a jump in the
payoff.

`n_steps` defaults to 1 because every estimator here depends on the path only
through `S_T` and the lognormal step is exact: 252 steps would cost 252x the
work for identical statistics.

---

## Closed forms beyond the vanilla

`analytic.py` carries three additions worth knowing about, all of them checked
against independent routes rather than against themselves.

```python
from quantlab.derivatives import (digital_call, digital_put,
                                  black_scholes_put_call_parity_residual,
                                  black_scholes_greeks)

digital_call(100, 110, 1.0, 0.05, 0.20)                        # 0.353861
(digital_call(100, 110, 1.0, 0.05, 0.20)
 + digital_put(100, 110, 1.0, 0.05, 0.20))                     # 0.951229 = exp(-0.05)
black_scholes_put_call_parity_residual(100, 110, 1.0, 0.05, 0.20, q=0.03)
#                                                              # 7.1e-15
black_scholes_greeks(100, 110, 1.0, 0.05, 0.20, q=0.03, option="call",
                     second_order=True)
# {..., 'vanna': 0.887877, 'volga': 24.554330, 'charm': -0.114665}
```

- **Digitals.** A cash-or-nothing digital is the negative strike-derivative of
  the vanilla, so a tight call spread converges to it; that is the cross-check
  the tests use, to 1e-7. The payoff is discontinuous at `K`, so delta and gamma
  blow up as expiry approaches: near the strike a digital cannot be hedged with
  any bounded position, which is why desks quote and risk-manage it as the call
  spread rather than as the limit.
- **The parity residual.** Parity is not a Black-Scholes result: it is a static
  replication argument, so it holds for any arbitrage-free model, any
  volatility, and any smile. That makes it the sharpest available check on the
  two pricers and on the dividend convention. Exposing it as a function makes it
  something callers can assert on rather than something they have to rederive.
- **Second-order Greeks**, behind `second_order=True` so the returned keys are
  unchanged for every existing caller. `vanna` (d(vega)/dS = d(delta)/dsigma) is
  what makes a risk-reversal directional in vol and is what the skew is
  charging for; `volga` (d2V/dsigma2) is the convexity of the position in
  volatility, zero exactly at `d1*d2 = 0` and negative strictly between the two
  roots; `charm` (d(delta)/dt) is what forces a delta rebalance over a weekend
  when nothing trades. All three are checked against central differences of the
  first-order Greeks to 1e-5 relative.

---

## Implied volatility

Every other function in this package maps volatility to a price; `implied_vol.py`
runs the map backwards (Wilmott, Ch. 8). Newton-Raphson converges quadratically
using vega as the derivative, with a bisection fallback for the cases where vega
collapses.

```python
from quantlab.derivatives import black_scholes_call, implied_volatility

price = float(black_scholes_call(100, 110, 1.0, 0.05, 0.20))
implied_volatility(price, 100, 110, 1.0, 0.05, return_diagnostics=True)
# (0.2, {'route': 'newton', 'iterations': 3, 'newton_iterations': 3,
#        'bisection_iterations': 0, 'sigma_initial': 0.301032, 'converged': True})

implied_volatility(200.0, 100, 110, 1.0, 0.05)   # nan: outside the no-arbitrage bounds
```

`return_diagnostics=True` is appended last and defaults to False, so callers
still receive a bare float unless they ask otherwise. The dict reports which
route was taken, how many price evaluations the solve cost, and the starting
volatility, which is what makes the claims below auditable rather than
anecdotal.

### The starting guess is gated, not applied everywhere

Brenner and Subrahmanyam (1988) observed that the at-the-money-forward call price
is almost exactly linear in volatility, which inverts to
`sigma_0 = sqrt(2*pi/T) * price / (S*exp(-q*T))`. It is very sharp there: 0.04%
relative error at sigma = 0.2, T = 0.25, and still only 1.5% at sigma = 0.6,
T = 1. In the wings it is actively harmful, because a deep out-of-the-money
price tends to zero and the formula returns `sigma_0 ~ 0`, exactly where vega is
degenerate and the iteration cannot start.

We therefore accept the approximation only when `|ln(F/K)| <= 0.25 * sigma_0 *
sqrt(T)`, i.e. only when the option really is near the forward measured in the
units the approximation itself supplies (a band in raw log-moneyness would be
wrong: 10% out of the money is nearly at the money for a five-year option and
four standard deviations out for a two-week one). Outside the band we fall back
to the Manaster-Koehler (1982) point, `sqrt(2*|ln(F/K)|/T)`, which is the
vega-maximising volatility and therefore the largest derivative Newton can
possibly have for that contract.

Measured on a grid of 1232 contracts (11 forward moneynesses from 0.6 to 1.7,
seven maturities from 0.05 to 5 years, four volatilities from 0.10 to 0.60,
calls and puts, `q` in {0, 0.03}), holding the fallback fixed and toggling only
the Brenner-Subrahmanyam branch:

```
band                 mean Newton iterations
|ln(F/K)| <= 0.05    3.143 -> 2.598   (-17.3%)
|ln(F/K)| <= 0.10    3.598 -> 3.326   ( -7.6%)
whole grid           6.169 -> 6.080   ( -1.4%)
```

The whole-grid figure is small because most of that grid is in the wings, where
the approximation is deliberately not used. Gating it, rather than applying it
everywhere as this module previously did, is the larger win: it removes every
bisection fallback on the grid (211 of 1232 down to none), and since a fallback
costs 40 or more further price evaluations the mean *total* iteration count
drops from 8.686 to 6.080, i.e. by 30%.

### What the surface is, and is not

`implied_vol_surface` inverts across a grid of spots and expiries. Plotted, it
is the standard picture of the smile (or, in equity index markets since 1987,
the downward skew), and the honest reading of that picture is that the model is
wrong about the terminal distribution: real returns have fatter tails than a
lognormal, out-of-the-money options are therefore worth more than a lognormal
says, and `sigma` is the only dial the formula leaves for expressing "worth
more". The surface is the shape of the model's error, plotted in volatility
units.

So it is a change of units, and a bijective one on the no-arbitrage interval: no
information is created and none destroyed. It is comparable across strikes,
expiries and underlyings in a way raw premia are not, which is the entire reason
the convention exists. It is *not* a forecast of realised volatility (implied
consistently exceeds subsequent realised: the variance risk premium), and it is
*not* evidence that the underlying has a strike-dependent volatility, which is
not a coherent statement about a single process. If a strike-dependent
volatility is what one wants to model, the correct objects are a local-volatility
surface (Dupire) or a stochastic-vol model (Heston, SABR), calibrated *to* this
surface rather than being it. Nothing here does that: this module inverts, and
stops.

---

## Three subtleties that actually matter

### 1. Barrier options break binomial trees

A CRR lattice can only represent prices `S0*u^k`. The barrier almost never lands
on one, so the barrier the lattice *enforces* is displaced by up to half a
level, and changing the step count moves that displacement erratically.

Measured error against the closed form, at n = 250, 500, 1000, 2000, 4000, 5000:

```
1.3e-1   1.2e-1   1.2e-1   6.1e-2   4.5e-3   2.7e-2
```

The last step is the point: raising the step count from 4,000 to 5,000 makes the
answer six times worse.

Aligning the barrier via `u = (B/S0)^(1/m)` fixes the placement but throws the
effective volatility off by ~1%, which does comparable damage. Two branches
cannot satisfy both constraints.

**Ritchken's trinomial** adds a third branch, and with it the missing degree of
freedom: choose a stretch parameter lambda >= 1 so that exactly *m* up-moves
land on the barrier:

```
m      = floor( ln(B/S0) / (sigma*sqrt(dt)) )
lambda = ln(B/S0) / (m*sigma*sqrt(dt))

pu = 1/(2*lambda^2) + (r - q - sigma^2/2)*sqrt(dt) / (2*lambda*sigma)
pd = 1/(2*lambda^2) - (r - q - sigma^2/2)*sqrt(dt) / (2*lambda*sigma)
pm = 1 - 1/lambda^2
```

Barrier aligned exactly, mean and variance matched exactly. Errors then fall
smoothly: 1.1e-2, 4.3e-3, 2.8e-3, 1.1e-3, 7.0e-4 at n = 250, 500, 1000, 2000,
4000.

Use `trinomial_up_and_out`. `binomial_up_and_out` is kept for comparison, with
the limitation documented in its docstring and the Broadie-Glasserman-Kou
continuity correction applied by default (`bgk_adjust=True`), which shifts the
lattice barrier to `B*exp(-beta*sigma*sqrt(dt))` to compensate for the fact that
a lattice monitors the barrier only at its own timesteps. The correction helps;
it does not repair the alignment problem, because nothing with two branches can.

For a barrier the PDE is better than either: the domain simply ends at `B` with
`V = 0` there, so no correction term is needed at all.

### 2. Discrete monitoring overprices knockouts

Checking the barrier only at simulation grid points misses crossings that happen
*between* observations, so too many paths survive and the option is overpriced.
Measured on the reference contract with 200,000 paths: 5.4526 against the true
5.1520, a 5.8% overstatement.

The Brownian bridge correction prices those unseen crossings analytically.
Conditional on the two endpoints of a step, the probability the path touched the
barrier is closed-form:

```
p_cross = exp( -2 * ln(S_prev/B) * ln(S_next/B) / (sigma^2 * dt) )
```

Each path is weighted by its survival probability instead of being counted as a
binary survivor, and the products are accumulated as `exp(sum(log(.)))` to avoid
underflow. With the correction: 5.1632, inside one standard error.

Set `brownian_bridge=False` to reproduce the bias; the test suite asserts it.

### 3. Vega vanishes, and Newton-Raphson goes with it

Implied volatility inverts Black-Scholes using vega as the derivative. Deep in
or out of the money, vega collapses toward zero and the Newton step divides by
almost nothing, sending the iteration somewhere useless.

`implied_volatility` detects this and falls back to **bisection**: slower, but it
cannot diverge on a bracketed root. Prices outside the no-arbitrage bounds return
`NaN` rather than a plausible-looking wrong number. The original implementation
instead retried from several random starting guesses; bisection is deterministic
and provably convergent, so we prefer it here, and the starting point is chosen
by the gated rule above rather than drawn at random.

The fallback is now rare rather than routine: on the 1232-contract grid it never
fires. It remains in place because "rare on a synthetic grid" is not "impossible
on a real chain", and a solver that can diverge is worse than one that is
occasionally slow.

---

## What the tests actually check

Not "does this reproduce the number it produced last time": almost nothing is
pinned to a value this code generated. The references are external.

- **Put-call parity**: `C - P = S*exp(-qT) - K*exp(-rT)`, model-independent, to
  1e-12 in closed form and to 3e-12 on every lattice except Jarrow-Rudd.
- **No-arbitrage bounds**: `max(S - K*exp(-rT), 0) <= C <= S`.
- **Monotonicity**: price rises in volatility and spot, falls in strike; a
  dividend yield lowers calls and raises puts.
- **Moment conditions on the lattice itself**: Tian must match the first three
  moments of the lognormal to 1e-15; CRR, Tian and LR must reproduce the forward
  exactly; JR must miss it at O(dt^2), and does (1.3e-8 at n = 101, 5.2e-11 at
  n = 1601).
- **Greeks against finite differences**, plus the exact identity
  `gamma = vega/(S^2*sigma*T)`, plus vanna computed both ways (Clairaut).
- **American call = European call** without dividends (Shreve): early exercise is
  worthless, so the two must agree exactly, and PSOR reproduces the European
  solve to 2.7e-15. With `q = 8%` the premium reappears (11.92 against 10.93).
- **American put > European put**, and never below intrinsic value; the PDE and a
  10,000-step lattice agree to 1.8e-3 on a problem with no closed form at all.
- **The free boundary is monotone in time**, at every one of 400 timesteps, and
  is NaN where no exercise region exists (an American call with `q = 0`), rather
  than an invented number at the edge of the grid.
- **Barrier < vanilla**, converging to vanilla as `B -> infinity`, with and
  without a dividend yield.
- **Digitals against a tight call spread**, to 1e-7; the pair sums to
  `cash*exp(-rT)` to 1e-14.
- **Monte Carlo within 3 to 4 standard errors**, with error falling as
  1/sqrt(n); `E[S_T] = S0*exp(rT)` and `Var[ln S_T] = sigma^2*T` for the
  simulated paths.
- **LSM against a 5,000-step lattice**, and separately against American put-call
  symmetry `C_A(S, K, r, q) = P_A(K, S, q, r)` (McDonald & Schroder 1998), which
  lets a lattice with no dividend argument price the symmetric contract and
  turns the check into a genuinely independent one.
- **Implied vol round trip**: price at sigma, invert, recover the *price* to
  1e-10 across 400 contracts. We assert on the price rather than the volatility
  on purpose: deep in the wings the map is nearly flat, so a price reproduced to
  1e-10 can still correspond to a volatility wrong in the third decimal, and
  claiming to recover sigma there would be dishonest.

The `q = 0` regression locks are the exception, and they are deliberate:
`test_default_arguments_reproduce_pre_dividend_prices` (lattices) and
`test_zero_dividend_reproduces_frozen_prices` (PDE) pin literal floats captured
before `q` existed, and `test_zero_dividend_reproduces_legacy_paths_bitwise`
transcribes the old path generator rather than importing the module it is
policing. None of the three is a correctness claim; they exist so that appending
a parameter with a zero default cannot perturb the arithmetic by even one unit in
the last place.

A note on step sizes: the Greek finite-difference tests use `h ~ 1e-4*S` for
first derivatives and `h ~ 1e-2*S` for gamma. Using `h = 1e-5` for a *second*
difference produces a 0.9% discrepancy that looks exactly like a wrong formula
but is pure floating-point cancellation: the numerator is ~2e-12 formed by
subtracting numbers of order 6. The optimal step for a second derivative is
`eps^(1/4)`, not `eps^(1/2)`.

---

## API

Every pricer takes `q` last, defaulting to `0.0`.

```
analytic.py
    black_scholes_call / black_scholes_put         European prices
    black_scholes_greeks                           delta, gamma, vega, theta, rho;
                                                   second_order=True adds vanna,
                                                   volga, charm
    black_scholes_put_call_parity_residual         model-free invariant, ~1e-14
    digital_call / digital_put                     cash-or-nothing, cash=1.0
    vega                                           dV/dsigma
    up_and_out_call_closed_form                    barrier, continuous monitoring

binomial.py
    binomial_european / binomial_american          method='crr'|'jr'|'tian'|'lr'
    binomial_richardson                            2*P(2n) - P(n); default method='lr'
    binomial_greeks                                five Greeks off the tree
    binomial_tree_full                             full lattice for plotting (CRR)
    binomial_up_and_out                            barrier (see limitation above)
    trinomial_up_and_out                           barrier, Ritchken: recommended

finite_difference.py
    crank_nicolson_european                        vanilla, whole surface
    crank_nicolson_american                        projected SOR on the LCP;
                                                   return_boundary=True gives S*(tau)
    crank_nicolson_greeks                          five Greeks from one solve
    crank_nicolson_up_and_out                      barrier, Rannacher start-up
    (theta_scheme: 0 implicit, 0.5 Crank-Nicolson, 1 explicit)

monte_carlo.py
    simulate_gbm_paths                             exact lognormal; antithetic or sobol
    monte_carlo_european                           optional stderr, control variate, QMC
    monte_carlo_up_and_out                         optional Brownian bridge
    longstaff_schwartz_american                    LSM; basis='power'|'laguerre',
                                                   out_of_sample=True for a lower bound
    monte_carlo_greeks                             pathwise delta and vega,
                                                   likelihood-ratio gamma and delta

implied_vol.py
    implied_volatility                             gated Brenner-Subrahmanyam start,
                                                   Manaster-Koehler fallback,
                                                   Newton with bisection backstop;
                                                   return_diagnostics=True for the route
    implied_vol_surface                            grid of spots x expiries
```
