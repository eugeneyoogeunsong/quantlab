# quantlab: Eugene (Yoogeun) Song (https://www.linkedin.com/in/yoogeunsong)
# Independent side project. MIT licensed; see LICENSE.

"""Binomial lattice pricing (Cox-Ross-Rubinstein, Jarrow-Rudd, Tian, Leisen-Reimer).

Following Shreve, *Stochastic Calculus for Finance I* (2004): the one-period
replication argument and the risk-neutral measure (Ch. 1), the multi-period tree
and the martingale property of the discounted price (Ch. 2), and American
securities as an optimal-stopping problem (Ch. 4).

The lattice discretises the underlying into up/down moves and works backwards
from expiry, discounting under the risk-neutral measure. Its advantage over the
closed form is that it can price American options: at every node we can ask
whether exercising now is worth more than holding, a question the Black-Scholes
PDE has no room for.

Analogy: Black-Scholes computes the value of a journey knowing only the
destination, whilst the lattice walks every fork in the road, which is the only
way to notice that stopping early is sometimes better.

Which parameterisation, and why it matters
------------------------------------------
A recombining tree carries three free numbers per step (u, d, p) and only two
constraints, namely the mean and the variance of the log-price over dt, and the
second of those only has to hold to leading order in dt. One degree of freedom
is therefore left over, and each named method spends it differently:

- ``crr``  Cox-Ross-Rubinstein (1979): impose d = 1/u, so the tree is symmetric
           in log-space and returns to its starting level after an up-down pair.
- ``jr``   Jarrow-Rudd (1983): impose p = 1/2, so every terminal node carries a
           binomial weight and the drift is pushed entirely into u and d.
- ``tian`` Tian (1993): spend it on matching the third moment of the lognormal
           as well, which removes the leading skew error of the discretisation.
- ``lr``   Leisen-Reimer (1996): spend it on centring the tree on the strike,
           inverting a normal CDF onto the binomial via Peizer-Pratt. This is
           the one method that gives up something for it: its per-step variance
           is off by 5.7e-3 relative at n = 101 (against CRR's 2.2e-4), an error
           that decays as O(dt) and is repaid many times over by the ordering of
           the terminal nodes.

Only the last of those changes the convergence ORDER, and that is the whole
argument for having them. Measured on a European call (S0=100, K=110, T=1,
r=0.05, sigma=0.20) against Black-Scholes, absolute error over the step ladder
n = 51, 101, 201, 401, 801, 1601:

    crr    2.0e-3   9.0e-3   3.5e-3   1.8e-3   6.2e-4   1.2e-3
    jr     2.8e-2   2.3e-3   1.4e-3   1.7e-3   5.6e-4   1.0e-3
    tian   3.6e-2   1.7e-2   7.1e-3   2.1e-3   3.3e-4   9.5e-4
    lr     1.4e-4   3.7e-5   9.3e-6   2.4e-6   5.9e-7   1.5e-7

Read the last column first: all three of CRR, JR and Tian are LESS accurate with
1601 steps than they were with 801, whilst LR's error ratio per doubling runs
3.86, 3.93, 3.96, 3.98, 3.99, converging on the factor of 4 that defines
O(1/n^2). At n = 801 LR is ~1000x closer than CRR; at n = 1601, ~8000x.

The more useful property, though, is not the accuracy but the monotonicity. An
oscillating error is not merely untidy: it means the change between two step
counts is worthless as an error estimate (CRR's error grows 4.5x between n = 51
and n = 101, then falls again), and it leaves Richardson extrapolation (see
`binomial_richardson`) with nothing stable to extrapolate.

The oscillation has a simple cause. A CRR, JR or Tian lattice knows nothing
about the strike, so as n changes the strike drifts across the terminal nodes;
the payoff kink then falls at a different place inside a cell each time, and the
quadrature error of the terminal distribution jumps with it. Leisen and Reimer
remove the problem at the source by construction: they choose u and d so the
strike sits at the centre of the terminal distribution for every n.

Dividend yield
--------------
Every pricer takes `q` last and defaults it to 0.0, so existing calls are
unaffected. The quantlab convention, applied identically in all four pricing
routes: the risk-neutral drift of S is (r - q), discounting of payoffs is always
at r (never at r - q), and the CRR probability becomes
p = (exp((r - q)*dt) - d) / (u - d).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "binomial_european",
    "binomial_american",
    "binomial_richardson",
    "binomial_greeks",
    "binomial_tree_full",
    "binomial_up_and_out",
]

TREE_METHODS = ("crr", "jr", "tian", "lr")


def _crr_params(T: float, r: float, sigma: float, n_steps: int, q: float = 0.0):
    """Cox-Ross-Rubinstein up/down factors and the risk-neutral probability.

    We set u = exp(sigma*sqrt(dt)) and d = 1/u, so the lattice recombines; p is
    then fixed by requiring the discounted spot to be a martingale under the
    (r - q) drift. A p outside [0, 1] means the parameters do not admit a
    no-arbitrage lattice at all.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} outside [0,1]. The lattice is "
            f"arbitrageable: with dt={dt:.5f}, u={u:.5f}, d={d:.5f}, the rate r={r} "
            "moves the forward outside the up/down range. Use more steps or check "
            "that sigma is not far too small for r."
        )
    return dt, u, d, p


def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt inversion, method 2: a normal CDF read backwards onto a binomial.

    Leisen and Reimer need the binomial probability whose n-step tree reproduces
    N(z) as closely as possible. Peizer and Pratt (1968) give a closed form for
    that inverse; method 2 is the sharper of their two variants, and the one LR
    adopt:

        h(z, n) = 1/2 + sgn(z)/2 * sqrt(1 - exp(-(z/a)^2 * (n + 1/6))),
        a = n + 1/3 + 0.1/(n + 1)

    Its accuracy is what buys the O(1/n^2) convergence of the resulting tree: the
    inversion error is O(1/n^2) whilst a naive normal approximation would leave an
    O(1/n) term behind. We require ODD n (see `_tree_params`).
    """
    a = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    return 0.5 + np.sign(z) * 0.5 * np.sqrt(1.0 - np.exp(-((z / a) ** 2) * (n + 1.0 / 6.0)))


def _tree_params(method: str, T: float, r: float, q: float, sigma: float, n_steps: int,
                 S0=None, K=None):
    """Lattice parameters for one of the four parameterisations.

    Parameters
    ----------
    method : str    one of 'crr', 'jr', 'tian', 'lr'
    T, r, q, sigma  contract and market inputs; drift is (r - q), discounting is at r
    n_steps : int   requested number of steps
    S0, K           spot and strike, required by 'lr' only (its tree depends on both)

    Returns
    -------
    dt, u, d, p, n_steps

    `n_steps` is returned because 'lr' may adjust it: the Peizer-Pratt inversion
    is defined for an odd number of steps, and we ROUND UP to the next odd
    integer rather than raising, so that `binomial_richardson` (which prices at n
    and 2n, and 2n is never odd) stays usable. The adjustment is at most one step
    and is reported here rather than hidden.

    Using an even n regardless would be a silent disaster, not a rounding
    nuisance: on the reference contract the even-step LR tree is out by 4.9e-3 at
    n = 802 against 5.9e-7 at n = 801, and the even ladder decays only as O(1/n).
    The strike lands exactly on a terminal node when n is even, so the payoff kink
    sits at the node the construction is trying to centre.
    """
    if method not in TREE_METHODS:
        raise ValueError(f"method must be one of {TREE_METHODS}, got {method!r}")
    if method == "crr":
        dt, u, d, p = _crr_params(T, r, sigma, n_steps, q)
        return dt, u, d, p, n_steps

    if method == "lr":
        if S0 is None or K is None:
            raise ValueError("method='lr' needs S0 and K: its tree is centred on the strike")
        if n_steps % 2 == 0:
            n_steps += 1

    dt = T / n_steps
    drift = np.exp((r - q) * dt)

    if method == "jr":
        # Jarrow-Rudd: fix p = 1/2 and put the whole drift into u and d. The
        # log-price moves by (r - q - sigma^2/2)*dt +/- sigma*sqrt(dt), i.e. the
        # Euler step of the GBM SDE with a two-point noise. Matching the mean is
        # then only asymptotic (the discrete forward is off by O(dt^2) per step),
        # which is why put-call parity holds here to ~1e-5 rather than to machine
        # precision; the other three trees are exact discrete martingales.
        nu = (r - q - 0.5 * sigma**2) * dt
        vol_step = sigma * np.sqrt(dt)
        u = np.exp(nu + vol_step)
        d = np.exp(nu - vol_step)
        p = 0.5
    elif method == "tian":
        # Tian (1993): match the first THREE moments of S(t+dt)/S(t). Writing
        # M = exp((r-q)*dt) and V = exp(sigma^2*dt), the three moment conditions
        # p*u + (1-p)*d = M, p*u^2 + (1-p)*d^2 = M^2*V and
        # p*u^3 + (1-p)*d^3 = M^3*V^3 have the closed-form solution below.
        # Matching the skew removes one error term but not the strike-alignment
        # problem, so the order stays O(1/n): a better constant, not a better rate.
        V = np.exp(sigma**2 * dt)
        radical = np.sqrt(V * V + 2.0 * V - 3.0)
        u = 0.5 * drift * V * (V + 1.0 + radical)
        d = 0.5 * drift * V * (V + 1.0 - radical)
        p = (drift - d) / (u - d)
    else:  # 'lr'
        # Leisen-Reimer (1996): invert the Black-Scholes d1 and d2 onto the
        # binomial with Peizer-Pratt, so p plays the role of N(d2) and p_star the
        # role of N(d1). Then u and d follow from the martingale condition,
        # p*u + (1-p)*d = exp((r-q)*dt), and the strike sits at the centre of the
        # terminal distribution for every n. The tree is contract-specific: change
        # K and you get a different tree, which is exactly the point.
        sqrt_T = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / sqrt_T
        d2 = d1 - sqrt_T
        p = _peizer_pratt(d2, n_steps)
        p_star = _peizer_pratt(d1, n_steps)
        u = drift * p_star / p
        d = drift * (1.0 - p_star) / (1.0 - p)

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"Risk-neutral probability p={p:.4f} outside [0,1] for method={method!r} "
            f"(dt={dt:.5f}, u={u:.5f}, d={d:.5f}). Check sigma > 0 and use more steps."
        )
    return dt, u, d, p, n_steps


def _payoff(S, K: float, option: str):
    if option == "call":
        return np.maximum(S - K, 0.0)
    if option == "put":
        return np.maximum(K - S, 0.0)
    raise ValueError(f"option must be 'call' or 'put', got {option!r}")


def _rollback(S0, K, T, r, sigma, n_steps: int, option: str, american: bool,
              method: str, q: float, capture_step: int | None = None):
    """Backward induction on the lattice: the one engine behind every pricer here.

    Returns (price, dt, nodes). `nodes` is None unless `capture_step` is given, in
    which case it is the (stock, value) pair of arrays at that step, which is what
    `binomial_greeks` reads its delta, gamma and theta from.

    Keeping a single engine matters for more than tidiness: the CRR arithmetic
    lives in exactly one place, so the guarantee that q=0.0 reproduces the
    pre-dividend prices bit-for-bit has one place to hold rather than four.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    dt, u, d, p, n_steps = _tree_params(method, T, r, q, sigma, n_steps, S0, K)
    disc = np.exp(-r * dt)

    j = np.arange(n_steps + 1)
    S_T = S0 * u ** (n_steps - j) * d**j
    V = _payoff(S_T, K, option)
    nodes = (S_T, V.copy()) if capture_step == n_steps else None

    for step in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american or step == capture_step:
            j = np.arange(step + 1)
            S_node = S0 * u ** (step - j) * d**j
            if american:
                V = np.maximum(V, _payoff(S_node, K, option))
            if step == capture_step:
                nodes = (S_node, V.copy())
    return float(V[0]), dt, nodes


def binomial_european(S0, K, T, r, sigma, n_steps: int = 500, option: str = "call",
                      method: str = "crr", q: float = 0.0) -> float:
    """European option by backward induction on a binomial lattice.

    Every parameterisation converges to Black-Scholes as `n_steps` grows, but not
    at the same rate and not in the same manner. With the default `method='crr'`
    the error oscillates, depending on whether the strike sits near a lattice
    node; averaging the n-step and (n+1)-step prices is the standard trick to damp
    it. With `method='lr'` the error is monotone and O(1/n^2), which is both more
    accurate (~1000x at n = 801 on the module's reference contract) and more
    honest, since successive prices then bracket the answer from one side and
    their difference actually estimates the remaining error.

    Parameters
    ----------
    S0, K, T, r, sigma  spot, strike, years to expiry, risk-free rate, volatility
    n_steps : int       time steps; 'lr' rounds this UP to the next odd integer
    option : str        'call' or 'put'
    method : str        'crr' (default), 'jr', 'tian' or 'lr'
    q : float           continuous dividend yield; drift is (r - q), discounting at r
    """
    return _rollback(S0, K, T, r, sigma, n_steps, option, False, method, q)[0]


def binomial_american(S0, K, T, r, sigma, n_steps: int = 500, option: str = "put",
                      method: str = "crr", q: float = 0.0) -> float:
    """American option: the same backward induction, plus an early-exercise test.

    At each node the holder takes the better of continuing or exercising:

        V = max(discounted expected value, intrinsic value)

    For a non-dividend-paying underlying an American CALL is worth exactly what
    its European twin is worth, since exercising early throws away both time
    value and the interest earned on the strike, so it is never optimal. The
    American PUT is genuinely worth more, because exercising frees up the strike
    in cash early. Both facts are asserted in the tests, which makes this a real
    check on the implementation rather than a comment. Once q > 0 the call's
    immunity goes away: the dividend stream is worth having, and with q = 0.08 an
    American call struck at 90 is worth 11.92 against the European 10.93.

    A caveat on `method`, since the module docstring's convergence table does NOT
    carry over here: Leisen-Reimer centres the tree on the terminal payoff kink,
    but an American option's error is dominated by the early-exercise boundary,
    which the construction says nothing about. Measured against a high-resolution
    reference on the standard put (S0=100, K=110, T=1, r=0.05, sigma=0.20), LR's
    error at n = 51 is 2.6e-2 against CRR's 9.6e-3; by n = 801 the two are level
    (9.4e-4 and 9.8e-4). Use 'lr' for European contracts, and do not expect it to
    pay for itself on American ones.
    """
    return _rollback(S0, K, T, r, sigma, n_steps, option, True, method, q)[0]


def binomial_richardson(S0, K, T, r, sigma, n_steps: int = 250, option: str = "call",
                        american: bool = False, method: str = "lr", q: float = 0.0) -> float:
    """Two-point Richardson extrapolation, 2*P(2n) - P(n), over a pair of lattices.

    If a scheme's error admits the expansion P(n) = V + C/n + O(1/n^2), then
    doubling the step count halves the leading term, and the combination
    2*P(2n) - P(n) cancels it outright. That is the classical argument, and it is
    only as good as its premise: the error must be a smooth function of 1/n with
    a stable sign.

    Where it works, and where it does not
    -------------------------------------
    On the module's reference contract (European call, S0=100, K=110, T=1,
    r=0.05, sigma=0.20), comparing the extrapolant against the n-step price it is
    built from:

        lr     n = 51, 101, 201, 401: error improves by 1.99x, 2.00x, 2.00x, 2.00x
        crr    n = 51, 101, 201, 401: 0.08x, 0.41x, 0.68x, 5.53x

    Specifically, LR gains a clean factor of two at every n, whilst CRR is made
    12x WORSE at n = 51 and is worse than doing nothing in five of the eight step
    counts tested. CRR's error alternates in sign as the strike drifts across the
    terminal nodes, so differencing two step counts amplifies the oscillation
    instead of cancelling a trend. Therefore we default to `method='lr'`, and
    extrapolating a CRR tree should be treated as unsupported rather than merely
    inadvisable.

    Why the gain on LR is 2x and not a whole order: LR's leading error is
    O(1/n^2), not O(1/n), so the weights above are matched to the wrong power and
    halve the term rather than annihilating it. The order-matched combination
    (4*P(2n) - P(n))/3 reaches 1.9e-7 at n = 51 where this one reaches 7.1e-5.
    We implement the classical two-point form because it is the one that is
    method-agnostic and the one the literature quotes; the caller who knows their
    scheme is second-order can build the sharper combination from two calls to
    `binomial_european`.

    American contracts are extrapolated at the caller's risk: the free boundary
    reintroduces an oscillatory component that the smooth expansion above does not
    describe.
    """
    price_n = _rollback(S0, K, T, r, sigma, n_steps, option, american, method, q)[0]
    price_2n = _rollback(S0, K, T, r, sigma, 2 * n_steps, option, american, method, q)[0]
    return 2.0 * price_2n - price_n


def binomial_greeks(S0, K, T, r, sigma, n_steps: int = 500, option: str = "call",
                    american: bool = False, method: str = "crr",
                    h_sigma: float = 1e-4, h_rate: float = 1e-4, q: float = 0.0) -> dict:
    """Delta, gamma, vega, theta and rho from a lattice (Hull's construction).

    Returns the same five keys as `black_scholes_greeks`, so the two can be
    compared directly, which the tests do.

    Cost: delta, gamma and theta are essentially FREE, whilst vega and rho are
    not. The first three are read off nodes the backward induction has already
    computed, so they add three subtractions to a tree we had to build anyway.
    Vega and rho have no such shortcut here and are central differences over
    re-priced trees, i.e. four extra lattices, making the full set of five Greeks
    about 5x the cost of the price alone. (Pathwise or adjoint derivatives would
    recover them cheaply; see Glasserman (2004), Ch. 7, for the simulation
    analogue.)

    The construction
    ----------------
    The step-2 nodes span the spot in both directions, so with S_uu, S_ud, S_dd
    and their values:

        delta = (V_uu - V_dd) / (S_uu - S_dd)
        gamma = [ (V_uu - V_ud)/(S_uu - S_ud) - (V_ud - V_dd)/(S_ud - S_dd) ]
                / [ (S_uu - S_dd)/2 ]
        theta = [ V_ud - delta*(S_ud - S0) - V_0 ] / (2*dt)

    Theta compares the central step-2 node with the root: same (approximate)
    spot, two steps less time. The `delta*(S_ud - S0)` term corrects for the fact
    that S_ud equals S0 only in a CRR tree, where d = 1/u; JR, Tian and LR all
    drift the middle node. That correction is not cosmetic. Dropping it leaves
    theta at -1.62 under 'lr' and -4.56 under 'jr' against a true -5.90, since a
    displacement of order dt divided by 2*dt contaminates theta at O(1).

    Accuracy, and why `method` matters more here than for the price
    ---------------------------------------------------------------
    On the reference contract (S0=100, K=110, T=1, r=0.05, sigma=0.20, n=501,
    call) the absolute errors against `black_scholes_greeks` are:

        method='lr'    delta 3.5e-4, gamma 1.2e-5, vega 9.1e-6, theta 4.5e-3, rho 5.5e-6
        method='crr'   delta 2.9e-4, gamma 1.8e-5, vega 5.7e-1, theta 2.7e-3, rho 1.7e-2

    Delta, gamma and theta are equally good either way (their error is the O(dt)
    truncation of the node differences, common to all trees), but CRR's vega is
    out by 1.4% relative and its rho by 4.3e-4. The reason is that a finite
    difference in sigma or r differences the lattice's OWN error as well as the
    price, and CRR's error swings with sigma; LR's does not, which is what its
    monotone convergence buys. Use `method='lr'` whenever vega or rho is wanted.

    The bump sizes default to 1e-4 in sigma and in r. Vega moves by 2e-8 relative
    across h in [1e-6, 1e-4] and picks up 6.8e-5 of O(h^2) truncation by h = 1e-2,
    so there is no cancellation cliff to fall off at the default: the tree price
    is smooth in both arguments at this resolution.
    """
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2: delta, gamma and theta are read off step 2")

    price, dt, nodes = _rollback(S0, K, T, r, sigma, n_steps, option, american, method, q,
                                 capture_step=2)
    S_nodes, V_nodes = nodes
    S_uu, S_ud, S_dd = S_nodes[0], S_nodes[1], S_nodes[2]
    V_uu, V_ud, V_dd = V_nodes[0], V_nodes[1], V_nodes[2]

    delta = (V_uu - V_dd) / (S_uu - S_dd)
    slope_up = (V_uu - V_ud) / (S_uu - S_ud)
    slope_down = (V_ud - V_dd) / (S_ud - S_dd)
    gamma = (slope_up - slope_down) / (0.5 * (S_uu - S_dd))
    theta = (V_ud - delta * (S_ud - S0) - price) / (2.0 * dt)

    def repriced(sigma_bumped, r_bumped):
        return _rollback(S0, K, T, r_bumped, sigma_bumped, n_steps, option, american,
                         method, q)[0]

    vega = (repriced(sigma + h_sigma, r) - repriced(sigma - h_sigma, r)) / (2.0 * h_sigma)
    rho = (repriced(sigma, r + h_rate) - repriced(sigma, r - h_rate)) / (2.0 * h_rate)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def binomial_tree_full(S0, K, T, r, sigma, n_steps: int = 6,
                       option: str = "put", american: bool = True, q: float = 0.0):
    """Return the full lattice, for plotting or inspection.

    Returns
    -------
    stock    : (n+1, n+1) upper-triangular stock price lattice
    value    : (n+1, n+1) option value lattice
    exercise : (n+1, n+1) bool, True where early exercise beats holding
               (all False when american=False)

    Intended for small `n_steps`: this is the teaching and diagnostic view. Use
    `binomial_european` / `binomial_american` for actual pricing. CRR only, since
    the point of the picture is the recombining d = 1/u geometry.
    """
    dt, u, d, p = _crr_params(T, r, sigma, n_steps, q)
    disc = np.exp(-r * dt)
    n = n_steps

    stock = np.zeros((n + 1, n + 1))
    for step in range(n + 1):
        j = np.arange(step + 1)
        stock[: step + 1, step] = S0 * u ** (step - j) * d**j

    value = np.zeros_like(stock)
    exercise = np.zeros_like(stock, dtype=bool)
    value[: n + 1, n] = _payoff(stock[: n + 1, n], K, option)

    for step in range(n - 1, -1, -1):
        cont = disc * (p * value[: step + 1, step + 1] + (1 - p) * value[1 : step + 2, step + 1])
        if american:
            intrinsic = _payoff(stock[: step + 1, step], K, option)
            value[: step + 1, step] = np.maximum(cont, intrinsic)
            exercise[: step + 1, step] = intrinsic > cont
        else:
            value[: step + 1, step] = cont
    return stock, value, exercise


# Broadie-Glasserman-Kou continuity-correction constant, equal to
# -zeta(1/2)/sqrt(2*pi), where zeta is the Riemann zeta function.
BGK_BETA = 0.5825971579390106


def trinomial_up_and_out(S0, K, B, T, r, sigma, n_steps: int = 500, q: float = 0.0) -> float:
    """Up-and-out barrier call on a Ritchken (1995) stretched trinomial tree.

    THIS is the lattice method to use for barriers; `binomial_up_and_out` is
    kept for comparison, but it converges badly (see its docstring).

    The problem with a binomial tree
    --------------------------------
    A CRR lattice can only represent prices S0*u^k for integer k. The barrier
    almost never lands on one, so the barrier the lattice actually enforces is
    displaced by up to half a level. Barrier prices are acutely sensitive to
    that displacement: a 0.3% error in barrier location moved the price by ~2%
    in testing.

    The alignment can be fixed by choosing u = (B/S0)^(1/m), but then u no
    longer equals exp(sigma*sqrt(dt)) and the lattice's effective volatility
    drifts by ~1%, which is just as damaging. With only two branches there are
    not enough free parameters to match the mean, the variance, AND the barrier
    position at once.

    Ritchken's solution
    -------------------
    A third branch supplies the missing degree of freedom. We introduce a
    stretch parameter lambda >= 1 with up-move exp(lambda*sigma*sqrt(dt)), and
    choose lambda so that exactly m up-moves land on the barrier:

        m = floor( ln(B/S0) / (sigma*sqrt(dt)) )
        lambda = ln(B/S0) / (m*sigma*sqrt(dt))

    The branch probabilities then still match the mean and variance of the
    log-price exactly:

        pu = 1/(2*lambda^2) + (r - q - sigma^2/2)*sqrt(dt) / (2*lambda*sigma)
        pd = 1/(2*lambda^2) - (r - q - sigma^2/2)*sqrt(dt) / (2*lambda*sigma)
        pm = 1 - 1/lambda^2

    Barrier aligned exactly, volatility matched exactly; convergence becomes
    smooth and roughly O(1/n).

    Analogy: a binomial tree is a ruler with fixed markings, so to measure to a
    line falling between them you must either move the line or stretch the
    ruler, and both distort something. The trinomial adds an adjustable marking.

    The dividend yield enters the drift only (the probabilities above), never the
    discounting; the knockout condition itself is a statement about the path of S,
    so it is untouched by q.
    """
    if B <= K or S0 >= B:
        return 0.0
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    log_ratio = np.log(B / S0)

    m = int(np.floor(log_ratio / (sigma * sqrt_dt)))
    if m < 1:
        # Barrier sits within a single up-move of spot: the grid is too coarse
        # to resolve it, so we decline to quote rather than return noise.
        return 0.0
    lam = log_ratio / (m * sigma * sqrt_dt)

    drift_term = (r - q - 0.5 * sigma**2) * sqrt_dt / (2.0 * lam * sigma)
    pu = 1.0 / (2.0 * lam**2) + drift_term
    pd = 1.0 / (2.0 * lam**2) - drift_term
    pm = 1.0 - 1.0 / lam**2
    if min(pu, pd, pm) < -1e-12:
        raise ValueError(
            f"Negative trinomial probability (pu={pu:.4f}, pm={pm:.4f}, pd={pd:.4f}). "
            "Increase n_steps."
        )

    disc = np.exp(-r * dt)
    dx = lam * sigma * sqrt_dt

    # Levels reachable at the final step, capped at the barrier level m: anything
    # at or above level m is knocked out, so we never need to carry it.
    max_level = m                     # knocked out at exactly this level
    min_level = -n_steps
    levels = np.arange(min_level, max_level + 1)
    S_nodes = S0 * np.exp(levels * dx)

    V = np.maximum(S_nodes - K, 0.0)
    V[levels >= m] = 0.0              # knocked out

    for _ in range(n_steps):
        # A node at level k branches to k+1, k, and k-1; the rolls line those
        # three destinations up so the step is one vectorised combination.
        up = np.roll(V, -1)
        up[-1] = 0.0
        down = np.roll(V, 1)
        down[0] = 0.0
        V = disc * (pu * up + pm * V + pd * down)
        V[levels >= m] = 0.0          # barrier enforced at every step

    return float(V[np.searchsorted(levels, 0)])


def binomial_up_and_out(S0, K, B, T, r, sigma, n_steps: int = 500,
                        bgk_adjust: bool = True, q: float = 0.0) -> float:
    """Up-and-out barrier call on a CRR binomial lattice.

    Prefer `trinomial_up_and_out` for real use. This function is retained
    because comparing the two is instructive, and because the original study
    this code came from benchmarked exactly these methods against each other.

    KNOWN LIMITATION (this is not a bug)
    ------------------------------------
    The binomial price for a barrier option converges slowly and
    NON-MONOTONICALLY. Measured against the closed form, the error moved
    1.2e-1 -> 1.4e-2 -> 2.7e-2 as steps went 500 -> 2000 -> 5000: more work,
    worse answer.

    The cause is structural, not a coding error. A CRR lattice can only take the
    values S0*u^k, and the barrier generally falls between two of them, so the
    barrier actually enforced is displaced by up to half a level; changing
    n_steps changes u, which moves that displacement erratically. Fixing it by
    choosing u = (B/S0)^(1/m) aligns the barrier but throws the lattice's
    effective volatility off by ~1%, which does comparable damage. Two branches
    simply do not provide enough freedom to satisfy both constraints; Ritchken's
    trinomial adds a third and does.

    Two refinements ARE applied here, and both help:

    1. Node prices are built from integer levels (``S0 * u**k``) rather than
       ``u**(step-j) * d**j``. The two are mathematically identical but not
       bitwise identical, so without this a node sitting exactly on the barrier
       is knocked out or not depending on floating-point rounding.

    2. `bgk_adjust` applies the Broadie-Glasserman-Kou continuity correction,
       shifting the lattice barrier to ``B*exp(-beta*sigma*sqrt(dt))`` with
       beta = -zeta(1/2)/sqrt(2*pi) ~ 0.5826. A lattice monitors the barrier
       only at its own timesteps, which makes knockout harder than under
       continuous monitoring and biases the price upward; the shift compensates
       for that, and it vanishes as dt -> 0, so the limit is unchanged.

    """
    if B <= K or S0 >= B:
        return 0.0

    dt_tmp = T / n_steps
    B_eff = B * np.exp(-BGK_BETA * sigma * np.sqrt(dt_tmp)) if bgk_adjust else B
    if B_eff <= S0:
        return 0.0

    dt, u, d, p = _crr_params(T, r, sigma, n_steps, q)
    disc = np.exp(-r * dt)

    def node_prices(step: int) -> np.ndarray:
        """Prices at `step`, built from integer levels so comparisons are exact."""
        levels = step - 2 * np.arange(step + 1)
        return S0 * u**levels

    # A relative tolerance, so a node mathematically ON the barrier is knocked
    # out consistently rather than flickering with floating-point rounding.
    def alive(S_node: np.ndarray) -> np.ndarray:
        return S_node < B_eff * (1.0 - 1e-9)

    S_T = node_prices(n_steps)
    V = np.where(alive(S_T), np.maximum(S_T - K, 0.0), 0.0)

    for step in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        V = np.where(alive(node_prices(step)), V, 0.0)
    return float(V[0])
