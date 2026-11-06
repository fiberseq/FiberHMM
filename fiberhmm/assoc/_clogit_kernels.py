"""Numba kernels for :mod:`fiberhmm.assoc.clogit`: the stratum moments of sum_{i in cases} x_i under the conditional
distribution given the number of cases k (e_k, its gradient and Hessian pieces, and the case probabilities pi_i).

Two exact routes, chosen per stratum:

- **DP**: the elementary symmetric polynomial recurrence of the original implementation (Gail, Lubin and Rubinstein
  1981), one stratum at a time with its own n and k (no padding to the largest stratum), same update order and
  rescaling. O(n k p^2) per stratum: used for small strata.
- **DFT**: for strata with many cases. With weights r_i tilted by z so that sum_i t_i / (1 + t_i) = k (t_i = z r_i,
  p_i = t_i / (1 + t_i)), e_k(r) = z^-k prod_i (1 + t_i) P(K = k), where K is the Poisson-binomial count with
  success probabilities p_i. P(K = k) is the inverse DFT of the characteristic function phi(theta) = prod_i (1 - p_i
  + p_i e^{i theta}) on the N = n + 1 points theta_m = 2 pi m / N, which is exact (the polynomial has degree n).
  First and second moments of sum_{cases} x come from the same sums with q_im = p_i e^{i theta_m} / (1 - p_i + p_i
  e^{i theta_m}): pi_i = Re sum_m c_m q_im / C, E[(sum x)(sum x)'] = Re[sum_m c_m g_m g_m' + sum_i x_i x_i' sum_m
  c_m q_im (1 - q_im)] / C with g_m = sum_i x_i q_im, c_m = e^{-i k theta_m} phi(theta_m) and C = sum_m c_m. Every
  term at frequency m is bounded by exp(-(V - 1/2)(1 - cos theta_m)) (V = sum_i p_i (1 - p_i)), so frequencies past
  the cut where that bound is below exp(-T) are left out: their total is far below double precision. O(n M p) with
  M ~ sqrt(T / V) N, against O(n k p^2) for the DP. Used only when the kept band stays within |theta| <= 2 pi / 3,
  where |1 - p + p e^{i theta}| >= 1/2, so every q is bounded.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True, nogil=True, error_model='numpy')
def dp_strata(Xf, eta, offs, kk, which, need_q, loge, mean, cov):
    """DP moments for the strata listed in ``which``; writes loge[s], mean[s] and (need_q) cov[s]."""
    p = Xf.shape[1]
    for w in range(len(which)):
        t = which[w]
        a0, a1 = offs[t], offs[t + 1]
        n = a1 - a0
        K = kk[t]
        m = -np.inf
        for i in range(a0, a1):
            if eta[i] > m:
                m = eta[i]
        if not np.isfinite(m):
            m = 0.0
        e = np.zeros(K + 1)
        e[0] = 1.0
        s = np.zeros((K + 1, p))
        Q = np.zeros((K + 1, p, p)) if need_q else np.zeros((1, 1, 1))
        logsc = 0.0
        for j in range(n):
            rj = math.exp(eta[a0 + j] - m)
            if rj == 0.0:
                continue
            x = Xf[a0 + j]
            for l in range(K, 0, -1):
                el = e[l - 1]
                if need_q:
                    for u in range(p):
                        xu = x[u]
                        su = s[l - 1, u]
                        for v in range(u, p):
                            Q[l, u, v] = Q[l, u, v] + rj * (Q[l - 1, u, v] + xu * s[l - 1, v] + su * x[v]
                                                            + xu * x[v] * el)
                for u in range(p):
                    s[l, u] = s[l, u] + rj * (s[l - 1, u] + x[u] * el)
                e[l] = e[l] + rj * el
            sc = 0.0
            for l in range(K + 1):
                if e[l] > sc:
                    sc = e[l]
            if not sc > 0:
                sc = 1.0
            for l in range(K + 1):
                e[l] /= sc
                for u in range(p):
                    s[l, u] /= sc
                if need_q:
                    for u in range(p):
                        for v in range(u, p):
                            Q[l, u, v] /= sc
            logsc += math.log(sc)
        ek = e[K]
        loge[t] = math.log(ek) + logsc + K * m
        for u in range(p):
            mean[t, u] = s[K, u] / ek
        if need_q:
            for u in range(p):
                for v in range(u, p):
                    c = Q[K, u, v] / ek - mean[t, u] * mean[t, v]
                    cov[t, u, v] = c
                    cov[t, v, u] = c


@njit(cache=True, nogil=True, error_model='numpy')
def dp_case_probabilities(eta, offs, kk, which, pi):
    """pi_i = r_i e_{k-1}(r without i) / e_k(r) by forward and backward polynomials (each row rescaled, scales in
    log), for the strata in ``which``; the original padded computation one stratum at a time."""
    for w in range(len(which)):
        t = which[w]
        a0, a1 = offs[t], offs[t + 1]
        n = a1 - a0
        K = kk[t]
        m = -np.inf
        for i in range(a0, a1):
            if eta[i] > m:
                m = eta[i]
        if not np.isfinite(m):
            m = 0.0
        r = np.empty(n)
        for j in range(n):
            r[j] = math.exp(eta[a0 + j] - m)
        F = np.zeros((n + 1, K + 1))
        lF = np.zeros(n + 1)
        F[0, 0] = 1.0
        row = np.empty(K + 1)
        for j in range(n):
            for l in range(K + 1):
                row[l] = F[j, l]
            for l in range(1, K + 1):
                row[l] += r[j] * F[j, l - 1]
            sc = 0.0
            for l in range(K + 1):
                if row[l] > sc:
                    sc = row[l]
            if not sc > 0:
                sc = 1.0
            for l in range(K + 1):
                F[j + 1, l] = row[l] / sc
            lF[j + 1] = lF[j] + math.log(sc)
        B = np.zeros((n + 2, K + 1))
        lB = np.zeros(n + 2)
        B[n + 1, 0] = 1.0
        B[n, 0] = 1.0
        for j in range(n - 1, -1, -1):
            for l in range(K + 1):
                row[l] = B[j + 2, l]
            for l in range(1, K + 1):
                row[l] += r[j] * B[j + 2, l - 1]
            sc = 0.0
            for l in range(K + 1):
                if row[l] > sc:
                    sc = row[l]
            if not sc > 0:
                sc = 1.0
            for l in range(K + 1):
                B[j + 1, l] = row[l] / sc
            lB[j + 1] = lB[j + 2] + math.log(sc)
        log_ek = math.log(F[n, K]) + lF[n]
        for j in range(n):
            tot = 0.0
            for l in range(K):
                tot += F[j, l] * B[j + 2, K - 1 - l]
            if tot > 0 and r[j] > 0:
                pi[a0 + j] = math.exp(math.log(r[j]) + math.log(tot) + lF[j] + lB[j + 2] - log_ek)
            else:
                pi[a0 + j] = 0.0


@njit(cache=True, nogil=True, error_model='numpy')
def _softplus(v):
    return v + math.log1p(math.exp(-v)) if v > 0 else math.log1p(math.exp(v))


@njit(cache=True, nogil=True, error_model='numpy')
def _sigmoid(v):
    if v >= 0:
        return 1.0 / (1.0 + math.exp(-v))
    ev = math.exp(v)
    return ev / (1.0 + ev)


@njit(cache=True, nogil=True, error_model='numpy')
def tilt(eta, offs, kk, T0, T1, out_a, out_M):
    """Per stratum: the log tilt a with sum_i sigmoid(eta_i + a) = k, and the number of DFT frequencies M to keep
    (-1: the band would pass 2 pi / 3, so the stratum takes the DP route). Cut: (V - 1/2)(1 - cos theta) >= T with
    T = T0 + T1 log N."""
    S = len(kk)
    for t in range(S):
        a0, a1 = offs[t], offs[t + 1]
        n = a1 - a0
        k = kk[t]
        lo_e = np.inf
        hi_e = -np.inf
        for i in range(a0, a1):
            if eta[i] < lo_e:
                lo_e = eta[i]
            if eta[i] > hi_e:
                hi_e = eta[i]
        base = math.log(k / (n - k))
        lo = base - hi_e          # every sigmoid <= k / n: f <= 0
        hi = base - lo_e          # every sigmoid >= k / n: f >= 0
        a = base - 0.5 * (lo_e + hi_e)
        V = 0.0
        for it in range(200):
            f = -float(k)
            V = 0.0
            for i in range(a0, a1):
                q = _sigmoid(eta[i] + a)
                f += q
                V += q * (1.0 - q)
            if f > 0:
                hi = a
            else:
                lo = a
            if abs(f) < 1e-10 * max(1.0, float(k)) or hi - lo < 1e-14:
                break
            step = a - f / V if V > 0 else 0.5 * (lo + hi)
            a = step if (lo < step < hi) else 0.5 * (lo + hi)
        out_a[t] = a
        N = n + 1
        Vp = V - 0.5
        T = T0 + T1 * math.log(N)
        if Vp <= 0 or T / Vp > 1.5:
            out_M[t] = -1
            continue
        theta = math.acos(1.0 - T / Vp)
        M = int(math.ceil(theta * N / (2.0 * math.pi)))
        if 3 * M > N:
            out_M[t] = -1
        else:
            out_M[t] = M


@njit(cache=True, nogil=True, error_model='numpy')
def dft_stratum(eta, a0, a1, k, a, M, need_pi, Qr, Qi):
    """One stratum by the tilted DFT, frequencies m = 0..M (the conjugate m -> -m folded in as weight 2).
    Returns (log e_k, pi, wv, c): with need_pi, pi_i, wv_i = Re sum_m c_m q_im (1 - q_im) / C, c[m] = weighted
    c_m / C, and Qr[i, m] + 1j Qi[i, m] = q_im."""
    n = a1 - a0
    N = n + 1
    pr = np.empty(n)
    lsum = 0.0
    for i in range(n):
        v = eta[a0 + i] + a
        pr[i] = _sigmoid(v)
        lsum += _softplus(v)
    c = np.empty(M + 1, dtype=np.complex128)
    w = np.empty(M + 1, dtype=np.complex128)
    for m in range(M + 1):
        th = 2.0 * math.pi * m / N
        w[m] = complex(math.cos(th), math.sin(th))
        phr = 1.0
        phi = 0.0
        wr, wi = w[m].real, w[m].imag
        for i in range(n):
            p = pr[i]
            dr = 1.0 - p + p * wr
            di = p * wi
            nr = phr * dr - phi * di
            phi = phr * di + phi * dr
            phr = nr
        ang = -k * th
        cr, ci = math.cos(ang), math.sin(ang)
        wt = 1.0 if m == 0 else 2.0
        c[m] = wt * complex(phr * cr - phi * ci, phr * ci + phi * cr)
    C = 0.0
    for m in range(M + 1):
        C += c[m].real
    loge = -k * a + lsum + math.log(C / N)
    pi = np.zeros(n if need_pi else 0)
    wv = np.zeros(n if need_pi else 0)
    if need_pi:
        for m in range(M + 1):
            c[m] = c[m] / C
        for i in range(n):
            p = pr[i]
            sp = 0.0
            sw = 0.0
            for m in range(M + 1):
                q = p * w[m] / (1.0 - p + p * w[m])
                cq = c[m] * q
                Qr[i, m] = q.real
                Qi[i, m] = q.imag
                sp += cq.real
                sw += (cq * (1.0 - q)).real
            pi[i] = sp
            wv[i] = sw
    return loge, pi, wv, c
