"""Separable native boundary-distribution objective: same algebra, cache-resident work.

Every native profile likelihood is a prefix difference, ll[u,(a,b)] = pre_u[b]-pre_u[a],
over projections (a,b) enumerated in upper-triangular order, and every unit's
admissible set is a rectangle [alo,ahi) x [blo,bhi) intersected with a<b (the
call-overlap and frozen-neighbour constraints are each one-sided in a or in b).
So exp(ll-offset) factorizes as A_u[a]*B_u[b] and the dense (units x projections)
matrix never needs to exist: the objective and its exact gradient are sums over
per-unit rectangles of a shared (K+1)x(K+1) table of geometry mass. The dense
matrix is 400 MB for the largest family here and its matvecs are memory-bandwidth
bound; this form touches a few megabytes and is compute bound.

This is NOT bit-identical to the dense path: the same terms are summed in a
different order. It is a declared, versioned fit backend. Any unit that does not
satisfy the structure exactly, or whose prefix swing would overflow the shifted
factors, makes the whole fit fall back to the dense reference objective.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.special import logsumexp


@njit(cache=True, nogil=True)
def _forward(q_tri, alo, ahi, blo, bhi, A, B):
    """z_u = sum_{rect, a<b} A[a] q[a,b] B[b]; z0_u = sum_{rect, a<b} q[a,b].

    Both are direct sums over the unit's cells. A prefix-table rectangle sum
    for z0 would subtract near-equal totals and lose the value entirely when
    the mass lies outside the rectangle (measured relative error 0.8 on a
    z0 of 1e-16), which is exactly the regime the optimizer explores.
    """
    U = alo.shape[0]
    z = np.zeros(U); z0 = np.zeros(U)
    for u in range(U):
        s = 0.; s0 = 0.
        for a in range(alo[u], ahi[u]):
            Aa = A[u, a]
            b0 = a+1 if a+1 > blo[u] else blo[u]
            t = 0.; t0 = 0.
            for b in range(b0, bhi[u]):
                qab = q_tri[a, b]
                t += qab*B[u, b]
                t0 += qab
            s += Aa*t
            s0 += t0
        z[u] = s
        z0[u] = s0
    return z, z0


@njit(cache=True, nogil=True)
def _backward(q_tri, alo, ahi, blo, bhi, A, B, inv_z, inv_z0, K1):
    """G[a,b] = sum_u A[a]B[b]/z_u and E[a,b] = sum_u 1/z0_u over each unit's cells.

    E is accumulated directly rather than through a 2-D difference array:
    1/z0 spans many orders of magnitude across units, and prefix sums of such
    signed corner terms cancel catastrophically at cells the large term does
    not belong to.
    """
    G = np.zeros((K1, K1)); E = np.zeros((K1, K1))
    U = alo.shape[0]
    for u in range(U):
        w = inv_z[u]; v = inv_z0[u]
        for a in range(alo[u], ahi[u]):
            Aa = A[u, a]*w
            b0 = a+1 if a+1 > blo[u] else blo[u]
            for b in range(b0, bhi[u]):
                G[a, b] += Aa*B[u, b]
                E[a, b] += v
    return G, E


MINIMUM_DENSE_ELEMENTS = 1 << 19   # below ~4 MB the dense matvec is already cache resident


class SeparableObjective:
    """Prepared once per fit from the same (likelihood, allowed) the dense path uses."""

    def __init__(self, likelihood, allowed, *, maximum_swing=600., minimum_elements=MINIMUM_DENSE_ELEMENTS):
        ll = np.asarray(likelihood, float); mask = np.asarray(allowed, bool)
        U, P = ll.shape
        if U*P < minimum_elements:
            raise ValueError('dense matrices are cache resident; separable form not worthwhile')
        K1 = int(round((1.+np.sqrt(1.+8.*P))/2.))
        if K1*(K1-1)//2 != P:
            raise ValueError('projection count is not triangular')
        a, b = np.triu_indices(K1, 1)
        alo = np.empty(U, np.int64); ahi = np.empty(U, np.int64)
        blo = np.empty(U, np.int64); bhi = np.empty(U, np.int64)
        A = np.zeros((U, K1)); B = np.zeros((U, K1)); offset = np.empty(U)
        for u in range(U):
            m = mask[u]; idx = np.flatnonzero(m)
            if not len(idx):
                raise ValueError('unit without admissible projections')
            au, bu = a[idx], b[idx]
            alo[u], ahi[u] = au.min(), au.max()+1
            blo[u], bhi[u] = bu.min(), bu.max()+1
            rect = (a >= alo[u]) & (a < ahi[u]) & (b >= blo[u]) & (b < bhi[u]) & (a < b)
            if not np.array_equal(rect, m):
                raise ValueError('admissible set is not a rectangle')
            pre = np.full(K1, np.nan)
            a0 = alo[u]; pre[a0] = 0.
            first = idx[au == a0]; pre[b[first]] = ll[u, first]
            bref = bhi[u]-1
            last = idx[bu == bref]; pre[a[last]] = pre[bref]-ll[u, last]
            recon = pre[bu]-pre[au]
            scale = 1.+float(np.abs(ll[u, idx]).max())
            if np.isnan(recon).any() or float(np.abs(recon-ll[u, idx]).max()) > 1e-9*scale:
                raise ValueError('likelihood is not a prefix difference')
            offset[u] = float(ll[u, idx].max())
            rb = slice(blo[u], bhi[u]); ra = slice(alo[u], ahi[u])
            swing = float(np.nanmax(pre[rb]))-float(np.nanmin(pre[rb]))
            if not np.isfinite(swing) or swing > maximum_swing:
                raise ValueError('prefix swing too large for shifted factors')
            mB = float(np.nanmax(pre[rb]))
            B[u, rb] = np.exp(pre[rb]-mB)
            A[u, ra] = np.exp(mB-pre[ra]-offset[u])
            if not np.isfinite(A[u, ra]).all():
                raise ValueError('shifted factor overflow')
        self.K1, self.a, self.b = K1, a, b
        self.alo, self.ahi, self.blo, self.bhi = alo, ahi, blo, bhi
        self.A, self.B, self.offset = A, B, offset
        self.evaluations = 0; self.fallbacks = 0

    def __call__(self, parameters, xy, log_area, fallback):
        from .measurement_distribution import _density
        density, derivative = _density(parameters, xy)
        log_q = density+log_area
        log_q -= logsumexp(log_q)
        q = np.exp(log_q)
        K1 = self.K1
        q_tri = np.zeros((K1, K1)); q_tri[self.a, self.b] = q
        z, z0 = _forward(q_tri, self.alo, self.ahi, self.blo, self.bhi, self.A, self.B)
        self.evaluations += 1
        if np.any(z < 1e-180) or np.any(z0 < 1e-180) or not (np.isfinite(z).all() and np.isfinite(z0).all()):
            self.fallbacks += 1
            return fallback(parameters)
        G, E = _backward(q_tri, self.alo, self.ahi, self.blo, self.bhi, self.A, self.B, 1./z, 1./z0, K1)
        weights = (q_tri*(G-E))[self.a, self.b]
        return -float((np.log(z)-np.log(z0)+self.offset).sum()), -(weights @ derivative)
