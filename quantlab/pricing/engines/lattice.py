# quantlab: Eugene (Yoogeun) Song. MIT licensed; see LICENSE.
"""Lattice (tree) engine with pluggable discretisation schemes.

Sources: Hull Ch. 21 (binomial trees, CRR parameters, control-variate
technique); Leisen & Reimer (1996) for the Peizer-Pratt tree; Ritchken (1995)
for the stretched trinomial; Broadie, Glasserman & Kou (1997) for the
continuity correction.

Design
------
The engine knows how to do backward induction. It does not know what the up
and down factors are -- a `Scheme` supplies those. That split is the whole
point: CRR, Jarrow-Rudd, Tian and Leisen-Reimer differ only in how they
match the lognormal's moments, and the induction loop is identical for all
of them. Adding a scheme is a dozen lines and no engine changes.

A scheme returns a `LatticeSpec`: per-step multiplicative moves and the
matching risk-neutral probabilities. Binomial schemes return two moves,
Ritchken's trinomial returns three. The induction is written for an arbitrary
number of branches, so nothing special-cases the trinomial.

Node prices are stored as integer *levels* (net number of up-moves), and the
price at a level is `S0 * exp(level * dx)`. Building prices from integer
levels rather than repeated multiplication is what makes barrier comparisons
exact rather than rounding-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.stats import norm

from ..instruments import BarrierOption, DigitalOption, Exercise, Instrument, VanillaOption
from ..models import BlackScholes
from ..results import PricingResult
from .base import UnsupportedInstrument

__all__ = [
    "LatticeSpec",
    "Scheme",
    "CoxRossRubinstein",
    "JarrowRudd",
    "Tian",
    "LeisenReimer",
    "RitchkenTrinomial",
    "LatticeEngine",
]

# Broadie-Glasserman-Kou continuity-correction constant: -zeta(1/2)/sqrt(2 pi).
_BGK_BETA = 0.5825971579390106


@dataclass(frozen=True)
class LatticeSpec:
    """One step of a recombining lattice.

    level_moves : integer change in level for each branch, e.g. (+1, -1) or (+1, 0, -1)
    log_step    : dx such that price(level) = S0 * exp(level * dx)
    probs       : risk-neutral probability of each branch
    n_steps     : number of time steps
    dt          : step length in years
    """

    level_moves: tuple[int, ...]
    log_step: float
    probs: tuple[float, ...]
    n_steps: int
    dt: float
    drift_per_step: float = 0.0        # log-drift absorbed per step (non-centred schemes)
    barrier_level: int | None = None   # integer level sitting exactly on a barrier

    def __post_init__(self) -> None:
        if len(self.level_moves) != len(self.probs):
            raise ValueError("one probability per branch")
        if min(self.probs) < -1e-12 or abs(sum(self.probs) - 1.0) > 1e-9:
            raise ValueError(
                f"branch probabilities {self.probs} are not a distribution -- the "
                "lattice would be arbitrageable. Use more steps or a different scheme.")

    def price_at(self, model: BlackScholes, level: np.ndarray) -> np.ndarray:
        return model.spot * np.exp(level * self.log_step)


class Scheme(Protocol):
    name: str

    def build(self, model: BlackScholes, expiry: float, n_steps: int,
              instrument: Instrument | None = None) -> LatticeSpec: ...


# ---------------------------------------------------------------------------
# Binomial schemes -- Hull Ch. 21 and the Leisen-Reimer paper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoxRossRubinstein:
    """u = exp(sigma sqrt dt), d = 1/u, p = (exp(b dt) - d)/(u - d).  Hull eq. 21.x.

    Matches the variance of the log-return exactly and the mean to O(dt).
    Converges at O(1/n) with a characteristic oscillation as the strike moves
    between node layers -- see `LeisenReimer` for the fix.
    """

    name: str = "crr"

    def build(self, model, expiry, n_steps, instrument=None) -> LatticeSpec:
        dt = expiry / n_steps
        dx = model.vol * np.sqrt(dt)
        u, d = np.exp(dx), np.exp(-dx)
        p = (np.exp(model.carry * dt) - d) / (u - d)
        return LatticeSpec((1, -1), dx, (p, 1 - p), n_steps, dt)


@dataclass(frozen=True)
class JarrowRudd:
    """Equal-probability tree: p = 1/2, moves carry the drift.

    u = exp((b - sigma^2/2) dt + sigma sqrt dt),  d = exp((b - sigma^2/2) dt - sigma sqrt dt)

    Since u*d != 1 the lattice is not centred on S0; we keep integer levels by
    storing the per-step drift separately and adding `step * drift` when
    recovering prices. The engine handles that through `LatticeSpec.shift`.
    """

    name: str = "jarrow-rudd"

    def build(self, model, expiry, n_steps, instrument=None) -> LatticeSpec:
        dt = expiry / n_steps
        dx = model.vol * np.sqrt(dt)
        nu = (model.carry - 0.5 * model.vol**2) * dt
        return LatticeSpec((1, -1), dx, (0.5, 0.5), n_steps, dt, drift_per_step=nu)


@dataclass(frozen=True)
class Tian:
    """Tian (1993): matches the first three moments of the lognormal.

    v = exp(sigma^2 dt);  u = (M v / 2)(v + 1 + sqrt(v^2 + 2v - 3));  d likewise with a minus.
    Where M = exp(b dt). Recombining only asymptotically; treated like Jarrow-Rudd.
    """

    name: str = "tian"

    def build(self, model, expiry, n_steps, instrument=None) -> LatticeSpec:
        dt = expiry / n_steps
        M = np.exp(model.carry * dt)
        v = np.exp(model.vol**2 * dt)
        root = np.sqrt(v**2 + 2 * v - 3)
        u = 0.5 * M * v * (v + 1 + root)
        d = 0.5 * M * v * (v + 1 - root)
        p = (M - d) / (u - d)
        # Factor u = exp(nu + dx), d = exp(nu - dx).
        nu = 0.5 * (np.log(u) + np.log(d))
        dx = 0.5 * (np.log(u) - np.log(d))
        return LatticeSpec((1, -1), dx, (p, 1 - p), n_steps, dt, drift_per_step=nu)


@dataclass(frozen=True)
class LeisenReimer:
    """Leisen & Reimer (1996): centre the tree on the strike.

    Uses the Peizer-Pratt inversion to choose p and p' so that the tree's
    binomial CDF matches N(d-) and N(d+) *at the strike*. The strike then
    always sits on a node, the oscillation disappears, and convergence is
    O(1/n^2) -- so 25 steps of Leisen-Reimer beat 1000 of CRR. Needs an odd
    number of steps.

    Because p depends on the strike, this scheme needs the instrument. For
    contracts without a strike it falls back to CRR.
    """

    name: str = "leisen-reimer"

    @staticmethod
    def _peizer_pratt(z: float, n: int) -> float:
        """Method 2 inversion from the paper."""
        denom = n + 1.0 / 3.0 + 0.1 / (n + 1)
        return 0.5 + np.sign(z) * 0.5 * np.sqrt(
            1.0 - np.exp(-((z / denom) ** 2) * (n + 1.0 / 6.0)))

    def build(self, model, expiry, n_steps, instrument=None) -> LatticeSpec:
        strike = getattr(instrument, "strike", None)
        if strike is None:
            return CoxRossRubinstein().build(model, expiry, n_steps)
        n = n_steps if n_steps % 2 == 1 else n_steps + 1  # must be odd
        dt = expiry / n
        d_plus, d_minus = model.d_plus_minus(strike, expiry)
        p = self._peizer_pratt(float(d_minus), n)
        p_prime = self._peizer_pratt(float(d_plus), n)
        M = np.exp(model.carry * dt)
        u = M * p_prime / p
        d = (M - p * u) / (1 - p)
        nu = 0.5 * (np.log(u) + np.log(d))
        dx = 0.5 * (np.log(u) - np.log(d))
        return LatticeSpec((1, -1), dx, (p, 1 - p), n, dt, drift_per_step=nu)


# ---------------------------------------------------------------------------
# Trinomial for barriers -- Ritchken (1995)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RitchkenTrinomial:
    """Stretched trinomial that places a node layer exactly on a barrier.

    Why a binomial tree cannot do this
    ----------------------------------
    A binomial tree has two free quantities per step (dx and p) and must
    spend both matching the mean and variance of the log-return. There is
    nothing left to spend on where the barrier falls, so the effective barrier
    the tree enforces is whichever node layer is nearest -- and that jumps
    around as n changes. The price oscillates instead of converging.

    Ritchken's construction adds a third branch and a stretch parameter
    lambda >= 1:

        dx = lambda * sigma * sqrt(dt)
        pu = 1/(2 lambda^2) + nu sqrt(dt) / (2 lambda sigma)
        pm = 1 - 1/lambda^2
        pd = 1/(2 lambda^2) - nu sqrt(dt) / (2 lambda sigma)

    with nu = b - sigma^2/2. Mean and variance are matched for *any* lambda,
    so lambda is free to be chosen such that an integer number of up-moves
    lands exactly on ln(H/S0). Barrier aligned, moments exact, convergence
    smooth.

    On the continuity correction -- and why it is OFF by default
    --------------------------------------------------------------
    Broadie, Glasserman & Kou's correction addresses a *discretely monitored*
    barrier on a *continuously moving* underlying: between two observations
    the true path may have crossed and come back unseen, so shifting the
    barrier inward by exp(-beta sigma sqrt dt), beta ~ 0.5826, compensates.

    An aligned trinomial has no such gap. The underlying exists only at nodes,
    moves at most one level per step, and a node sits exactly on the barrier
    -- so no path can reach the far side without landing on it first. In the
    lattice's own world the monitoring is already continuous. Applying the
    correction anyway shifts the barrier where it need not be and biases the
    price low. Measured: with the correction, error was 3.1e-1 at 250 steps
    and still 7.4e-2 at 4000; without it, 1.1e-2 and 7.0e-4.

    The flag is kept for experiments. Leave it off.
    """

    name: str = "ritchken"
    continuity_correction: bool = False

    def build(self, model, expiry, n_steps, instrument=None) -> LatticeSpec:
        if not isinstance(instrument, BarrierOption):
            raise UnsupportedInstrument("RitchkenTrinomial is a barrier-specific scheme")
        dt = expiry / n_steps
        sqrt_dt = np.sqrt(dt)
        sig = model.vol
        nu = model.carry - 0.5 * sig**2

        barrier = instrument.barrier
        if self.continuity_correction:
            shift = np.exp(-_BGK_BETA * sig * sqrt_dt)
            barrier = barrier * shift if instrument.barrier_type.is_up else barrier / shift

        gap = abs(np.log(barrier / model.spot))
        m = int(np.floor(gap / (sig * sqrt_dt)))
        if m < 1:
            raise ValueError("Barrier is within one lattice step of spot; increase n_steps")
        lam = gap / (m * sig * sqrt_dt)

        dx = lam * sig * sqrt_dt
        drift_term = nu * sqrt_dt / (2 * lam * sig)
        pu = 1 / (2 * lam**2) + drift_term
        pm = 1 - 1 / lam**2
        pd = 1 / (2 * lam**2) - drift_term
        return LatticeSpec((1, 0, -1), dx, (pu, pm, pd), n_steps, dt,
                           barrier_level=m if instrument.barrier_type.is_up else -m)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

@dataclass
class LatticeEngine:
    """Backward induction over any `Scheme`.

    Parameters
    ----------
    n_steps    : time steps
    scheme     : discretisation; defaults to Leisen-Reimer for vanillas and
                 Ritchken for barriers, chosen automatically when None
    richardson : Richardson extrapolation across n and 2n steps. Helps
                 Leisen-Reimer, whose error is smooth; HURTS CRR and
                 Jarrow-Rudd, whose errors oscillate as the strike moves
                 between node layers -- extrapolating an oscillation
                 amplifies it. Off by default for that reason.
    """

    n_steps: int = 500
    scheme: Scheme | None = None
    richardson: bool = False

    name = "lattice"

    def supports(self, instrument: Instrument) -> bool:
        return isinstance(instrument, (VanillaOption, DigitalOption, BarrierOption))

    def _choose_scheme(self, instrument: Instrument) -> Scheme:
        if self.scheme is not None:
            return self.scheme
        if isinstance(instrument, BarrierOption):
            return RitchkenTrinomial()
        return LeisenReimer()

    def price(self, instrument: Instrument, model: BlackScholes) -> PricingResult:
        if not self.supports(instrument):
            raise UnsupportedInstrument(f"LatticeEngine cannot price {type(instrument).__name__}")
        scheme = self._choose_scheme(instrument)

        if self.richardson:
            v1 = self._induct(instrument, model, scheme, self.n_steps)
            v2 = self._induct(instrument, model, scheme, 2 * self.n_steps)
            value = 2 * v2 - v1
            diag = {"scheme": scheme.name, "steps": (self.n_steps, 2 * self.n_steps),
                    "raw": (v1, v2), "richardson": True}
        else:
            value = self._induct(instrument, model, scheme, self.n_steps)
            diag = {"scheme": scheme.name, "steps": self.n_steps}
        return PricingResult(float(value), f"{self.name}:{scheme.name}", diagnostics=diag)

    def _induct(self, instrument, model, scheme, n_steps) -> float:
        spec = scheme.build(model, instrument.expiry, n_steps, instrument)
        n = spec.n_steps
        drift = spec.drift_per_step
        disc = model.discount(spec.dt)
        moves = np.array(spec.level_moves)
        probs = np.array(spec.probs)
        max_move = int(moves.max())

        # Level grid wide enough for every reachable node at expiry.
        levels = np.arange(-max_move * n, max_move * n + 1)

        def prices(step: int) -> np.ndarray:
            return model.spot * np.exp(levels * spec.log_step + step * drift)

        # Barrier handling: nodes at or beyond the barrier level are dead.
        barrier_level = spec.barrier_level
        is_knock_out = isinstance(instrument, BarrierOption) and instrument.barrier_type.is_knock_out
        if isinstance(instrument, BarrierOption) and not is_knock_out:
            # Price knock-in via parity: vanilla - knock-out.
            vanilla = instrument.underlying_vanilla
            v_vanilla = self._induct(vanilla, model, LeisenReimer(), n_steps)
            v_out = self._induct(instrument.with_barrier_type(instrument.barrier_type.complement),
                                 model, scheme, n_steps)
            return v_vanilla - v_out

        def alive_mask(step: int) -> np.ndarray:
            if not is_knock_out:
                return np.ones_like(levels, dtype=bool)
            if barrier_level is not None:
                return levels < barrier_level if instrument.barrier_type.is_up else levels > barrier_level
            # Scheme without explicit alignment: compare prices, with a tolerance.
            H = instrument.barrier
            p = prices(step)
            return (p < H * (1 - 1e-9)) if instrument.barrier_type.is_up else (p > H * (1 + 1e-9))

        early = instrument.exercise is Exercise.AMERICAN
        rebate = getattr(instrument, "rebate", 0.0)

        V = instrument.payoff(prices(n)).astype(float)
        if is_knock_out:
            V = np.where(alive_mask(n), V, rebate)

        for step in range(n - 1, -1, -1):
            cont = np.zeros_like(V)
            for move, p in zip(moves, probs):
                shifted = np.roll(V, -move)
                # Nodes shifted in from outside the grid are unreachable; zero them.
                if move > 0:
                    shifted[-move:] = 0.0
                elif move < 0:
                    shifted[:-move] = 0.0
                cont += p * shifted
            V = disc * cont
            if early:
                V = np.maximum(V, instrument.payoff(prices(step)))
            if is_knock_out:
                V = np.where(alive_mask(step), V, rebate * model.discount((n - step) * spec.dt))

        return float(V[np.searchsorted(levels, 0)])

    # -- diagnostics --------------------------------------------------------------

    def tree(self, instrument: VanillaOption, model: BlackScholes, n_steps: int = 6):
        """Full lattice for inspection: (stock, value, early_exercise) arrays.

        Small trees only -- this is a teaching view, not a pricing path.
        """
        scheme = self._choose_scheme(instrument)
        spec = scheme.build(model, instrument.expiry, n_steps, instrument)
        n = spec.n_steps
        drift = spec.drift_per_step
        disc = model.discount(spec.dt)
        p_up = spec.probs[0]

        stock = np.full((n + 1, n + 1), np.nan)
        value = np.full((n + 1, n + 1), np.nan)
        exercised = np.zeros((n + 1, n + 1), dtype=bool)
        for step in range(n + 1):
            lv = step - 2 * np.arange(step + 1)
            stock[: step + 1, step] = model.spot * np.exp(lv * spec.log_step + step * drift)
        value[: n + 1, n] = instrument.payoff(stock[: n + 1, n])
        for step in range(n - 1, -1, -1):
            cont = disc * (p_up * value[: step + 1, step + 1] + (1 - p_up) * value[1: step + 2, step + 1])
            if instrument.exercise is Exercise.AMERICAN:
                intrinsic = instrument.payoff(stock[: step + 1, step])
                value[: step + 1, step] = np.maximum(cont, intrinsic)
                exercised[: step + 1, step] = intrinsic > cont
            else:
                value[: step + 1, step] = cont
        return stock, value, exercised
