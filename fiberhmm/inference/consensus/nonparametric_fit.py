"""Nonparametric native family geometry: penalized maximum likelihood over the projection grid.

The fitted object is a probability mass q over the same projection classes the Gaussian
model is evaluated on. The objective is the Gaussian's conditional log-likelihood

    L(q) = sum_u [ log sum_p q_p e^{ll_up} 1[allowed_up] - log sum_p q_p 1[allowed_up] ]

plus a smoothing term alpha * sum_p g_p log q_p toward the Gaussian seed's mass g conditioned
on the cells some source unit admits, i.e. ``alpha`` pseudo-units drawn from the seed where the
real units are observed (default 4: the held-out optimum on Hia5, and
within 0.01 nats per unit of the best value on the largest DddA families). Two structural
facts drive the procedure:

* L is scale-free in q and its two halves are concave and convex respectively (a truncated
  likelihood), so it is not concave in general; the smoothing term is strictly concave. The
  fit therefore starts from the Gaussian seed and reports the seed and the smoothed optimum
  side by side; uniqueness is checked empirically from several starts, not assumed.
* The minorize-maximize step with the convex half linearized,

      q_p <- (q_p G_p + alpha g_p) / (E_p + alpha),   G_p = sum_u SL_up / z_u,  E_p = sum_u 1[allowed_up] / z0_u,

  never decreases the penalized objective (Turnbull-type argument), costs exactly one forward
  and one backward pass of the separable kernels per iteration, and is accelerated with a
  SQUAREM extrapolation in the log domain that falls back to the plain step whenever the
  extrapolated point is not at least as good. Cells no source unit could have produced carry
  the seed's mass exactly and the admissible cells share the seed's admissible mass (the
  stationarity condition gives q_p = g_p off the data's reach), so a transferred geometry never
  assigns zero probability to a projection class merely because the source units' windows did
  not cover it; the split is fixed up front rather than iterated, because iterating it is a
  plateau whenever the seed sits mostly outside the admissible cells.

Measured on real fits (plan document section 10.2): held-out conditional log-likelihood per
source unit improves by about +0.66 nats on Hia5 and +0.96 nats on the largest DddA families
against the 5-parameter Gaussian (5-fold, 510 and 1,401 held-out units). Transfer to another grid uses the tabulated-density path
(exact refinement of source cells), never a Gaussian evaluation of the nonparametric fit.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .separable_fit import SeparableObjective, _forward, _backward


class _DenseAccumulator:
    """Same z, z0, G, E as the separable kernels, for small dense fits."""

    def __init__(self, ll, mask):
        self.mask = mask
        self.offset = np.where(mask, ll, -np.inf).max(1)
        self.SL = np.exp(np.where(mask, ll-self.offset[:, None], -np.inf))
        self.E = mask.astype(float)

    def __call__(self, q):
        z = self.SL @ q; z0 = self.E @ q
        if np.any(z <= 0.) or np.any(z0 <= 0.) or not (np.isfinite(z).all() and np.isfinite(z0).all()):
            return None
        G = self.SL.T @ (1./z); E = self.E.T @ (1./z0)
        return z, z0, G, E


class _SeparableAccumulator:
    def __init__(self, sep):
        self.sep = sep; self.offset = sep.offset

    def __call__(self, q):
        s = self.sep; K1 = s.K1
        q_tri = np.zeros((K1, K1)); q_tri[s.a, s.b] = q
        z, z0 = _forward(q_tri, s.alo, s.ahi, s.blo, s.bhi, s.A, s.B)
        if np.any(z <= 0.) or np.any(z0 <= 0.) or not (np.isfinite(z).all() and np.isfinite(z0).all()):
            return None
        G, E = _backward(q_tri, s.alo, s.ahi, s.blo, s.bhi, s.A, s.B, 1./z, 1./z0, K1)
        return z, z0, G[s.a, s.b], E[s.a, s.b]


def _normalized_mass(log_mass):
    log_mass = np.asarray(log_mass, float)
    finite = np.isfinite(log_mass)
    out = np.zeros(log_mass.shape)
    if finite.any():
        out[finite] = np.exp(log_mass[finite]-log_mass[finite].max())
        out /= out.sum()
    return out


def fit_nonparametric(likelihood, allowed, areas, gaussian_log_mass, *, alpha=4.,
                      max_iterations=1000, tolerance=1e-9, stationarity=1e-6, floor=1e-300,
                      initial_log_mass=None, history=None):
    """Return dict(log_mass, objective, penalized_objective, iterations, converged, ...).

    ``objective`` is the conditional log-likelihood L(q) of the returned mass, directly
    comparable with the Gaussian fit's objective; ``penalized_objective`` adds the
    smoothing term. Convergence needs both a small relative change of the penalized
    objective and the first-order optimality condition on the simplex: for every cell
    with mass, (q_p G_p + alpha g_p) / (q_p (E_p + alpha)) is within ``stationarity`` of 1
    (Lindsay's gradient test; the ratio is exactly the MM step's multiplicative factor).
    The objective alone is not enough: a cell the data want but the seed gave almost no
    mass grows geometrically for many iterations while the objective barely moves, and
    stopping on that plateau returns a saddle. The start is half the seed and half a
    uniform mass over the admissible cells: cells the seed likes begin with real mass,
    cells it dislikes begin at 1/(2 x admissible cells) instead of underflow, so the
    plateau never forms (measured: 80/80 Hia5 fits converge, none more than 5e-6 nats
    below the best of three starts, against 79/80 and 2.5e-2 from the seed alone).
    ``initial_log_mass`` overrides the start (diagnostics: uniqueness from several
    starts); ``history`` collects the penalized objective per accepted iterate.
    """
    ll = np.asarray(likelihood, float); mask = np.asarray(allowed, bool)
    U, P = ll.shape
    alpha = float(alpha)
    prior = _normalized_mass(gaussian_log_mass)
    try:
        acc = _SeparableAccumulator(SeparableObjective(ll, mask, minimum_elements=1 << 16))
    except ValueError:
        acc = _DenseAccumulator(ll, mask)
    offset = acc.offset
    admissible = mask.any(0)
    if not admissible.any():
        raise ValueError('Nonparametric fit needs at least one admissible projection class')
    prior_inadmissible = float(prior[~admissible].sum())
    share = max(0., 1.-prior_inadmissible)
    # Declared split: cells no unit admits keep the seed's mass; the admissible cells share
    # the seed's admissible mass. The iteration runs on the admissible simplex r, smoothed
    # with ``alpha`` pseudo-units toward the seed conditioned on the admissible cells (the
    # pseudo-units are observed where the real units are). Iterating the split instead
    # contracts at rate alpha x share / U per step, a plateau whenever the seed sits mostly
    # outside the data's reach.
    seed_conditioning = 'seed_on_admissible_cells'
    if alpha > 0.:
        g = np.where(admissible, prior, 0.)
        if g.sum() > 0.:
            g = g/g.sum()
        else:
            g = admissible/admissible.sum(); seed_conditioning = 'uniform_fallback_seed_underflow'
    else:
        g = np.zeros(P)

    def project(rr):
        rr = np.where(admissible, np.maximum(rr, floor), 0.)
        total = rr.sum()
        if not np.isfinite(total) or total <= 0.:
            return None
        return rr/total

    def evaluate(rr):
        out = acc(rr)
        if out is None:
            return None
        z, z0, G, E = out
        value = float((np.log(z)-np.log(z0)+offset).sum())
        penalized = value+alpha*float(g[g > 0.] @ np.log(np.maximum(rr[g > 0.], floor))) if alpha > 0. else value
        if not np.isfinite(penalized):
            return None
        return value, penalized, G, E

    def optimality_gap(rr, G, E):
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = (rr*G+alpha*g)/(rr*(E+alpha))
        ratio = ratio[admissible & (rr > 0.)]
        return float(np.max(ratio)-1.) if ratio.size else 0.

    def mm_step(rr, G, E):
        with np.errstate(divide='ignore', invalid='ignore'):
            nxt = np.where(admissible, (rr*G+alpha*g)/(E+alpha), 0.)
        return project(nxt)

    if initial_log_mass is None:
        start = 0.5*g+0.5*admissible/admissible.sum() if alpha > 0. else admissible/admissible.sum()
    else:
        start = _normalized_mass(initial_log_mass)
    q = project(start)
    state = evaluate(q) if q is not None else None
    if state is None:
        raise FloatingPointError('Nonparametric fit cannot evaluate the initial mass')
    if history is not None:
        history.append(state[1])
    converged = False; it = 0
    for it in range(1, max_iterations+1):
        value, penalized, G, E = state
        q1 = mm_step(q, G, E); s1 = evaluate(q1) if q1 is not None else None
        if s1 is None:
            break
        best_q, best = q1, s1
        q2 = mm_step(q1, s1[2], s1[3]); s2 = evaluate(q2) if q2 is not None else None
        if s2 is not None:
            if s2[1] >= best[1]:
                best_q, best = q2, s2
            # SQUAREM (S3) extrapolation in the log domain, stabilized by one MM step.
            sup = admissible
            x0 = np.log(np.maximum(q[sup], floor)); x1 = np.log(np.maximum(q1[sup], floor)); x2 = np.log(np.maximum(q2[sup], floor))
            r = x1-x0; v = (x2-x1)-r
            vv = float(v @ v); rr = float(r @ r)
            if vv > 0. and rr > 0.:
                step = -max(1., np.sqrt(rr/vv))
                x = x0-2.*step*r+step*step*v
                cand = np.zeros(P); cand[sup] = np.exp(x-x.max()); cand = project(cand)
                sc = evaluate(cand) if cand is not None else None
                if sc is not None:
                    cand3 = mm_step(cand, sc[2], sc[3]); s3 = evaluate(cand3) if cand3 is not None else None
                    if s3 is not None and s3[1] >= best[1]:
                        best_q, best = cand3, s3
        if best[1] < penalized-1e-9*max(1., abs(penalized)):
            # The MM step is monotone in exact arithmetic; a decrease is round-off at a
            # stationary point. Keep the current iterate and stop.
            converged = optimality_gap(q, G, E) <= stationarity
            break
        improvement = best[1]-penalized
        q, state = best_q, best
        if history is not None:
            history.append(state[1])
        if abs(improvement) <= tolerance*max(1., abs(state[1])) and \
                optimality_gap(q, state[2], state[3]) <= stationarity:
            converged = True
            break
    value, penalized = state[0], state[1]
    gap = optimality_gap(q, state[2], state[3])
    # Assemble the full mass: admissible cells carry the seed's admissible share, the rest the seed.
    if alpha > 0.:
        q = np.where(admissible, q*max(share, 1e-12), prior)
        q = q/q.sum()
    # Finite everywhere: exported records must be JSON; a cell with no mass carries the
    # double-precision floor, which every consumer treats as zero.
    log_mass = np.log(np.maximum(q, floor))
    log_mass -= logsumexp(log_mass)
    return dict(log_mass=log_mass, objective=float(value), penalized_objective=float(penalized),
                iterations=int(it), converged=bool(converged), smoothing_pseudo_units=alpha,
                optimality_gap=gap, prior_inadmissible_mass=prior_inadmissible,
                seed_conditioning=seed_conditioning)
