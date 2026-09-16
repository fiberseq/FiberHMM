# Numerical kernel promoted from the validated September 2026 consensus experiments.
"""Exact whole-region bounded-geometry competition, without a run-count cap.

One configuration contains zero or one geometry of each nominated family.
Protected runs are disjoint, with at least one accessible UNION opportunity
between them (the same convention as the local bounded-family experiment).

P(C | eta) = exp(sum_f eta_f + sum_f log q_f(geometry_f)) / Z_prior(eta).
L_m(eta) / accessible_base_m = Z_m(eta) / Z_prior(eta).

This is a *factorized activity prior conditioned on physical compatibility*,
not the local prototype's unrestricted learned configuration-frequency vector.
It does not model extra long-range co-occupancy interactions. Both partitions
and all inclusion/geometry marginals are exact for this stated model.

A sweep DAG remembers a used family only while another geometry of that same
family could still start. This prevents repeated tiny families when +/-X lets
two of their possible geometries be disjoint, without a global 2**F state space.
No raw-call geometry reward enters the likelihood. Missing observations are 0 LR.
"""
from __future__ import annotations

import math
import numpy as np
from numba import njit, prange
from numba.extending import register_jitable
from scipy.optimize import minimize

from .projection import bounded_projection


@njit(cache=True)
def _add(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b-a))


@register_jitable(inline='always')
def _evaluate_body(prefix, eta, allowed, geometry_allowed, geometry_adjustment, offsets, dest, edge_geo, ga, gb, gf, logq,
              n_nodes, export_geometry):
    n, nf, ng = len(prefix), len(eta), len(ga)
    z = np.empty(n)
    inclusion = np.zeros((n, nf))
    geometry = np.zeros((n, ng if export_geometry else 0))
    for m in prange(n):
        if not allowed[m].any():
            # Only the accessible path exists. Observations outside any
            # admitted family cancel, regardless of their measured values.
            z[m] = 0.
            continue
        weights = np.empty(ng)
        for g in range(ng):
            weights[g] = eta[gf[g]] + logq[g] + prefix[m, gb[g]] - prefix[m, ga[g]]
            if geometry_adjustment.shape[1]:
                weights[g] += geometry_adjustment[m,g]
        fw = np.full(n_nodes, -np.inf)
        bw = np.full(n_nodes, -np.inf)
        fw[0] = 0.
        for v in range(n_nodes-1):
            if fw[v] == -np.inf:
                continue
            for e in range(offsets[v], offsets[v+1]):
                g = edge_geo[e]
                if g >= 0 and (not allowed[m,gf[g]] or
                              (geometry_allowed.shape[1] and not geometry_allowed[m,g])):
                    continue
                w = 0. if g < 0 else weights[g]
                u = dest[e]
                fw[u] = _add(fw[u], fw[v] + w)
        z[m] = fw[n_nodes-1]
        bw[n_nodes-1] = 0.
        for v in range(n_nodes-2, -1, -1):
            # No allowed path from the source reaches this state. Its family
            # and geometry masses are exactly zero, and no reachable parent
            # can need its backward value via an allowed finite-weight edge.
            # In broad catalogs most per-unit states are unreachable; skip
            # this work without pruning any possible configuration.
            if fw[v] == -np.inf:
                continue
            for e in range(offsets[v], offsets[v+1]):
                g = edge_geo[e]
                if g >= 0 and (not allowed[m,gf[g]] or
                              (geometry_allowed.shape[1] and not geometry_allowed[m,g])):
                    continue
                w = 0. if g < 0 else weights[g]
                tail = w + bw[dest[e]]
                bw[v] = _add(bw[v], tail)
                if g >= 0:
                    p = math.exp(fw[v] + tail - z[m])
                    inclusion[m, gf[g]] += p
                    if export_geometry:
                        geometry[m, g] += p
    return z, inclusion, geometry


# Distinct named wrappers have distinct on-disk cache identities. Compiling the
# same Python function twice with different parallel flags can collide in Numba
# caches. Inlining shares arithmetic/traversal, not the dispatcher's cache key.
@njit(cache=True, parallel=True)
def _evaluate(prefix, eta, allowed, geometry_allowed, geometry_adjustment, offsets, dest, edge_geo,
              ga, gb, gf, logq, n_nodes, export_geometry):
    return _evaluate_body(prefix, eta, allowed, geometry_allowed, geometry_adjustment,
        offsets, dest, edge_geo, ga, gb, gf, logq, n_nodes, export_geometry)


@njit(cache=True)
def _evaluate_serial(prefix, eta, allowed, geometry_allowed, geometry_adjustment, offsets, dest, edge_geo,
                     ga, gb, gf, logq, n_nodes, export_geometry):
    return _evaluate_body(prefix, eta, allowed, geometry_allowed, geometry_adjustment,
        offsets, dest, edge_geo, ga, gb, gf, logq, n_nodes, export_geometry)


class RegionFamilyLattice:
    def __init__(self, positions, centers, ambiguity_bp=10, *,
                 maximum_nodes=100_000, maximum_edges=2_000_000,
                 family_geometry_overrides=None):
        self.positions = np.asarray(positions, dtype=np.int64)
        self.centers = np.asarray(centers, dtype=np.int64).reshape(-1, 2)
        self.ambiguity_bp = ambiguity_bp
        self.k, self.f = len(self.positions), len(self.centers)
        if not self.f:
            raise ValueError('At least one nominated family is required')
        overrides=family_geometry_overrides or {}
        self.families = [overrides[f] if f in overrides else bounded_projection(self.positions, c, ambiguity_bp)
                         for f,c in enumerate(self.centers)]
        # Counterfactual-only explicit opportunity projections. Native discovery
        # does not use this extension. It permits exact transfer of a source
        # boundary cell without clipping it to the recipient's nominal bp box.
        for f,geometry in overrides.items():
            if not isinstance(f,(int,np.integer)) or not 0<=f<self.f:raise ValueError('Invalid family override index')
            g={**geometry};aa=np.asarray(g['starts']);bb=np.asarray(g['ends']);q=np.asarray(g['q'],float)
            if aa.ndim!=1 or not len(aa) or bb.shape!=aa.shape or q.shape!=aa.shape or np.any(~np.isfinite(aa)) or np.any(~np.isfinite(bb)) or np.any(aa!=np.rint(aa)) or np.any(bb!=np.rint(bb)) or np.any(aa<0) or np.any(bb>self.k) or np.any(aa>=bb):
                raise ValueError('Valid nonempty opportunity projection overrides required')
            if np.any(~np.isfinite(q)) or np.any(q<=0) or not np.isclose(q.sum(),1.,atol=1e-12):raise ValueError('Normalized positive override masses required')
            if len(np.unique(np.c_[aa,bb],axis=0))!=len(aa):raise ValueError('Duplicate projection overrides must be aggregated')
            g.update(starts=aa.astype(np.int64),ends=bb.astype(np.int64),q=q/q.sum())
            self.families[f]=g
        self.ga = np.concatenate([q['starts'] for q in self.families])
        self.gb = np.concatenate([q['ends'] for q in self.families])
        self.logq = np.log(np.concatenate([q['q'] for q in self.families]))
        self.gf = np.concatenate([np.full(len(q['q']), f, dtype=np.int64) for f,q in enumerate(self.families)])
        starts = [[] for _ in range(self.k+2)]
        for g,a in enumerate(self.ga):
            starts[int(a)].append(g)
        last = [int(q['starts'].max()) for q in self.families]
        alive = [sum(1 << f for f in range(self.f) if last[f] >= p) for p in range(self.k+2)]
        states = [set() for _ in range(self.k+2)]
        states[0].add(0)
        edges = []
        total_nodes = 1
        for p in range(self.k+1):
            for used in sorted(states[p]):
                src = (p, used)
                targets = [(p+1, used & alive[p+1], -1)]
                for g in starts[p]:
                    f = int(self.gf[g])
                    if used & (1 << f):
                        continue
                    # End b is exclusive; starting again at b+1 leaves one
                    # accessible opportunity at b. Terminal is k+1.
                    nxt = int(self.gb[g]) + 1
                    targets.append((nxt, (used | (1 << f)) & alive[nxt], g))
                for nxt, mask, g in targets:
                    if mask not in states[nxt]:
                        states[nxt].add(mask)
                        total_nodes += 1
                    edges.append((src, (nxt,mask), g))
                if total_nodes > maximum_nodes or len(edges) > maximum_edges:
                    raise MemoryError(f'Exact frontier budget exceeded: {total_nodes} nodes, {len(edges)} edges; no candidate silently removed')
        nodes = [(p, used) for p in range(self.k+2) for used in sorted(states[p])]
        index = {node:i for i,node in enumerate(nodes)}
        if nodes[0] != (0,0) or nodes[-1] != (self.k+1,0):
            raise AssertionError('Sweep did not produce unique source/terminal')
        offsets = np.zeros(len(nodes)+1, dtype=np.int64)
        for src,_,_ in edges:
            offsets[index[src]+1] += 1
        self.offsets = np.cumsum(offsets)
        self.dest = np.asarray([index[t] for _,t,_ in edges],dtype=np.int64)
        self.edge_geo = np.asarray([g for _,_,g in edges],dtype=np.int64)
        self.n_nodes = len(nodes)
        if any(index[s] >= index[t] for s,t,_ in edges):
            raise AssertionError('Sweep is not a DAG')
        self.max_frontier_states = max(map(len,states))

    def evaluate(self, values, eta, *, allowed=None, export_geometry=False,
                 geometry_allowed=None, geometry_log_adjustment=None):
        """Exact partition/marginals, optionally on a restricted geometry model.

        Unlike map_configuration's action-only mask, geometry_allowed HERE is
        an inference-model restriction. A caller must use the identical
        outcome-free exposure mask in its prior partition. Outcome-dependent
        vetoes belong only in the data likelihood, never the prior. Existing
        CR callers omit this argument and reproduce their original model.
        """
        eta = np.asarray(eta,dtype=float)
        if eta.shape != (self.f,) or np.any(~np.isfinite(eta)):
            raise ValueError('One finite log activity per family required')
        inputs = self._evaluation_inputs(values, allowed, geometry_allowed, geometry_log_adjustment)
        return self._evaluate_inputs(inputs, eta, export_geometry)

    def _evaluation_inputs(self, values, allowed, geometry_allowed, geometry_log_adjustment):
        """Validate fixed context once; reuse the exact prefix sums during fit."""
        values = np.asarray(values,dtype=float)
        if values.ndim != 2 or values.shape[1] != self.k or np.any(~np.isfinite(values)):
            raise ValueError('Finite full-domain native logLR matrix required')
        if allowed is None:
            allowed=np.ones((len(values),self.f),dtype=np.bool_)
        allowed=np.asarray(allowed,dtype=np.bool_)
        if allowed.shape != (len(values),self.f):
            raise ValueError('One candidate-availability flag per unit/family required')
        if geometry_allowed is None:
            geometry_allowed=np.empty((len(values),0),dtype=np.bool_)
        else:
            geometry_allowed=np.asarray(geometry_allowed,dtype=np.bool_)
            if geometry_allowed.shape != (len(values),len(self.ga)):
                raise ValueError('One geometry-availability flag per unit/projection required')
        if geometry_log_adjustment is None:
            geometry_log_adjustment=np.empty((len(values),0),dtype=float)
        else:
            geometry_log_adjustment=np.asarray(geometry_log_adjustment,dtype=float)
            if (geometry_log_adjustment.shape != (len(values),len(self.ga)) or
                    np.any(~np.isfinite(geometry_log_adjustment))):
                raise ValueError('Finite per-unit geometry log adjustments required')
        prefix = np.c_[np.zeros(len(values)), np.cumsum(values,axis=1)]
        return prefix, allowed, geometry_allowed, geometry_log_adjustment

    def _evaluate_inputs(self, inputs, eta, export_geometry=False):
        prefix, allowed, geometry_allowed, geometry_log_adjustment = inputs
        evaluate = _evaluate_serial if len(prefix) < 8 else _evaluate
        z, inclusion, geometry = evaluate(prefix,eta,allowed,geometry_allowed,geometry_log_adjustment,self.offsets,self.dest,self.edge_geo,
            self.ga,self.gb,self.gf,self.logq,self.n_nodes,export_geometry)
        if np.any(inclusion > 1+1e-7) or np.any(~np.isfinite(z)):
            raise FloatingPointError('Invalid whole-region partition/inclusion')
        return {'log_partition':z,'family_inclusion':inclusion,'geometry_mass':geometry}

    @staticmethod
    def _compress_inputs(inputs):
        """Memoize identical numerical evaluations, NOT evidence-unit collapse.

        Include the complete prefix and every availability/adjustment field in
        the key. Expand results to their original row order before every mean:
        multiplicity, fold membership, and floating-point reduction order stay
        unchanged. Byte equality only; no rounding or approximate grouping.
        Bound key storage for wide whole-region fits by simply skipping this
        optional optimization (never drop rows or hypotheses).
        """
        if sum(a.nbytes for a in inputs) > 64*1024**2:
            return inputs, None
        prefix, allowed, geometry_allowed, adjustment = inputs
        seen = {}; rows = []; inverse = np.empty(len(prefix), dtype=np.int64)
        for m in range(len(prefix)):
            key = tuple(a[m].tobytes() for a in inputs) if allowed[m].any() else ()
            if key not in seen:
                seen[key] = len(rows); rows.append(m)
            inverse[m] = seen[key]
        if len(rows) == len(prefix):
            return inputs, None
        return tuple(a[rows] for a in inputs), inverse

    def any_family_inclusion(self, values, eta, families, *, allowed=None,
                             geometry_allowed=None, geometry_log_adjustment=None,
                             log_partition=None):
        """Exact P(at least one listed family), not a sum of marginals.

        The excluded partition uses the SAME unnormalized configuration weights
        and base measure as the unrestricted partition. It is not renormalized
        under a new prior. This works for both mutually exclusive siblings and
        families which can co-occur. Physical/admission restrictions, if used,
        must be identical in the numerator and denominator.
        """
        values=np.asarray(values,dtype=float)
        indices=np.asarray(families)
        if indices.ndim!=1 or (len(indices) and indices.dtype.kind not in 'iu'):
            raise ValueError('A one-dimensional integer family index list is required')
        indices=np.unique(indices.astype(int))
        if np.any(indices<0) or np.any(indices>=self.f):
            raise ValueError('Family event index outside this catalog')
        if not len(indices):return np.zeros(len(values))
        base_allowed=np.ones((len(values),self.f),bool) if allowed is None else np.asarray(allowed,bool)
        kw=dict(geometry_allowed=geometry_allowed,geometry_log_adjustment=geometry_log_adjustment)
        base=(self.evaluate(values,eta,allowed=base_allowed,**kw)['log_partition']
              if log_partition is None else np.asarray(log_partition,float))
        if base.shape!=(len(values),) or np.any(~np.isfinite(base)):
            raise ValueError('Finite same-model log partition required per unit')
        excluded=base_allowed.copy();excluded[:,indices]=False
        z0=self.evaluate(values,eta,allowed=excluded,**kw)['log_partition']
        delta=z0-base
        if np.any(delta>1e-7):raise FloatingPointError('Excluded partition exceeds full partition')
        return -np.expm1(np.minimum(delta,0.))

    def prior_recipe(self,allowed,n):
        if allowed is None:
            return np.ones((1,self.f),bool),np.array([n]),np.zeros(n,dtype=int)
        unique,inverse,counts=np.unique(np.asarray(allowed,dtype=bool),axis=0,return_inverse=True,return_counts=True)
        return unique,counts,inverse

    def geometry_event_inclusion(self,values,eta,event_geometries,*,allowed=None,
                                 geometry_allowed=None,geometry_log_adjustment=None,log_partition=None):
        """P(any selected geometry satisfies a predeclared observable feature).

        Unlike an exact family identity, this can describe protection of a fixed
        site irrespective of which native size subclass accounts for it. The
        event does not nominate or alter classes. Its coordinate rule must be
        declared independently of cross-assay rate agreement.
        """
        values=np.asarray(values,float);n=len(values);ng=len(self.ga)
        event=np.asarray(event_geometries,float)
        if event.shape==(ng,):event=np.broadcast_to(event,(n,ng))
        if event.shape!=(n,ng):raise ValueError('One event-membership flag per geometry (or unit/geometry) required')
        if np.any(~np.isfinite(event)) or np.any((event<0)|(event>1)):
            raise ValueError('Event alias fractions must be finite and in [0,1]')
        physical=np.ones((n,ng),bool) if geometry_allowed is None else np.asarray(geometry_allowed,bool)
        if physical.shape!=(n,ng):raise ValueError('Incorrect physical geometry mask')
        kw=dict(allowed=allowed,geometry_log_adjustment=geometry_log_adjustment)
        base=self.evaluate(values,eta,geometry_allowed=physical,**kw)['log_partition'] if log_partition is None else np.asarray(log_partition,float)
        if base.shape!=(n,) or np.any(~np.isfinite(base)):raise ValueError('Finite same-model partition required')
        # Partial projection classes retain the original integer-alias mass:
        # only their non-event aliases enter the absent partition. Never use a
        # representative coordinate to resolve an unobserved boundary.
        absent_adjustment=np.log(np.maximum(1-event,1e-300))
        if geometry_log_adjustment is not None:absent_adjustment=absent_adjustment+geometry_log_adjustment
        absent=self.evaluate(values,eta,allowed=allowed,geometry_allowed=physical&(event<1),
            geometry_log_adjustment=absent_adjustment)['log_partition']
        delta=absent-base
        if np.any(delta>1e-7):raise FloatingPointError('Feature-excluded partition exceeds full partition')
        return -np.expm1(np.minimum(delta,0.))

    def map_configuration(self, values, eta, *, allowed=None, geometry_allowed=None,
                          geometry_log_adjustment=None):
        """Exact joint action decoder on the SAME configuration DAG.

        Optional geometry_allowed restricts displayable actions, not the fitted
        likelihood or its marginal probabilities. For example a grouping layer
        can require an actual upstream-call overlap and avoid frozen obstacles.
        The returned score is an unnormalized configuration log weight, not a
        per-family confidence. Each family occurs at most once; no greedy
        small-first assignment or marginal-probability threshold selects edges.
        """
        values = np.asarray(values, dtype=float)
        eta = np.asarray(eta, dtype=float)
        n = len(values)
        if values.ndim != 2 or values.shape[1] != self.k or np.any(~np.isfinite(values)):
            raise ValueError('Finite full-domain native logLR matrix required')
        if eta.shape != (self.f,) or np.any(~np.isfinite(eta)):
            raise ValueError('One finite log activity per family required')
        if allowed is None:
            allowed = np.ones((n, self.f), dtype=bool)
        if geometry_allowed is None:
            geometry_allowed = np.ones((n, len(self.ga)), dtype=bool)
        allowed = np.asarray(allowed, dtype=bool)
        geometry_allowed = np.asarray(geometry_allowed, dtype=bool)
        if allowed.shape != (n, self.f) or geometry_allowed.shape != (n, len(self.ga)):
            raise ValueError('Incorrect family/geometry action-mask shape')
        if geometry_log_adjustment is None:
            geometry_log_adjustment=np.empty((n,0),dtype=float)
        else:
            geometry_log_adjustment=np.asarray(geometry_log_adjustment,dtype=float)
            if geometry_log_adjustment.shape != (n,len(self.ga)) or np.any(~np.isfinite(geometry_log_adjustment)):
                raise ValueError('Finite per-unit geometry log adjustments required')
        prefix = np.c_[np.zeros(n), np.cumsum(values, axis=1)]
        score, geometries = _map_decode(prefix, eta, allowed, geometry_allowed, geometry_log_adjustment,
            self.offsets, self.dest, self.edge_geo, self.ga, self.gb, self.gf,
            self.logq, self.n_nodes)
        return {'log_weight': score, 'geometry_by_family': geometries}

    def objective(self, values, eta, *, allowed=None, prior_recipe=None,
                  geometry_allowed=None, geometry_log_adjustment=None):
        # A restricted configuration model must use exactly the same geometry
        # domain and integer-alias prior masses on both sides of Z_data/Z_prior.
        # These arguments are fixed context, never a data-dependent LR veto.
        kw=dict(geometry_allowed=geometry_allowed,geometry_log_adjustment=geometry_log_adjustment)
        out = self.evaluate(values,eta,allowed=allowed,**kw)
        if geometry_allowed is not None or geometry_log_adjustment is not None:
            prior=self.evaluate(np.zeros_like(values),eta,allowed=allowed,**kw)
            return (float(np.mean(prior['log_partition']-out['log_partition'])),
                    np.mean(prior['family_inclusion']-out['family_inclusion'],axis=0))
        unique,counts,_=self.prior_recipe(allowed,len(values)) if prior_recipe is None else prior_recipe
        prior = self.evaluate(np.zeros((len(unique),self.k)),eta,allowed=unique)
        # Average objective: every evidence unit enters once, irrespective of
        # its number of raw calls. Empty-exposure rows cancel exactly.
        return (float(prior['log_partition'] @ counts / len(values)-out['log_partition'].mean()),
                counts @ prior['family_inclusion'] / len(values)-out['family_inclusion'].mean(axis=0))

    def fit(self, values, *, allowed=None, max_iter=120, starts=(-2.,0.), check=None, report=None,
            geometry_allowed=None, geometry_log_adjustment=None):
        fits=[]
        recipe=self.prior_recipe(allowed,len(values))
        fixed = self._evaluation_inputs(values, allowed, geometry_allowed, geometry_log_adjustment)
        fixed, inverse = self._compress_inputs(fixed)
        restricted = geometry_allowed is not None or geometry_log_adjustment is not None
        if restricted:
            prior_fixed = (np.zeros_like(fixed[0]), *fixed[1:])
        else:
            prior_fixed = self._evaluation_inputs(np.zeros((len(recipe[0]), self.k)), recipe[0], None, None)

        def objective(eta):
            out = self._evaluate_inputs(fixed, eta)
            z, inclusion = out['log_partition'], out['family_inclusion']
            prior = self._evaluate_inputs(prior_fixed, eta)
            if restricted:
                dz = prior['log_partition']-z
                di = prior['family_inclusion']-inclusion
                if inverse is not None: dz, di = dz[inverse], di[inverse]
                return float(dz.mean()), di.mean(axis=0)
            if inverse is not None: z, inclusion = z[inverse], inclusion[inverse]
            return (float(prior['log_partition'] @ recipe[1] / len(values)-z.mean()),
                    recipe[1] @ prior['family_inclusion'] / len(values)-inclusion.mean(axis=0))

        for start in starts:
            iteration=0
            if report is not None:report(start,iteration)
            if check is not None:
                check()
            def callback(_eta):
                nonlocal iteration
                iteration+=1
                if check is not None:check()
                if report is not None:report(start,iteration)
            initial=np.full(self.f,start) if np.ndim(start)==0 else np.asarray(start,dtype=float).copy()
            if initial.shape!=(self.f,) or np.any(~np.isfinite(initial)):
                raise ValueError('Each fit start must be scalar or one finite activity per family')
            result=minimize(objective,initial,jac=True,
                method='L-BFGS-B',bounds=[(-12.,12.)]*self.f,
                callback=callback if check is not None or report is not None else None,
                options={'maxiter':max_iter,'ftol':1e-10,'gtol':1e-5,'maxls':30})
            gradient=np.asarray(result.jac).copy()
            gradient[(result.x <= -12+1e-7)&(gradient>0)] = 0
            gradient[(result.x >= 12-1e-7)&(gradient<0)] = 0
            fits.append({'initial_log_activity':float(start) if np.ndim(start)==0 else initial.tolist(),'eta':result.x,
                         'objective':float(result.fun),'success':bool(result.success),
                         'iterations':int(result.nit),'evaluations':int(result.nfev),
                         'message':str(result.message),'max_projected_gradient':float(np.max(np.abs(gradient)))})
        chosen=min(fits,key=lambda f:f['objective'])
        return chosen, fits

    def metadata(self):
        return {'families':self.f,'opportunities':self.k,'geometries':len(self.ga),
                'dag_nodes':self.n_nodes,'dag_edges':len(self.dest),
                'maximum_frontier_states':self.max_frontier_states,
                'global_family_or_run_cap':None,'one_occurrence_per_family':True,
                'minimum_accessible_union_opportunities_between_runs':1,
                'ambiguity_bp':self.ambiguity_bp,
                'prior':'factorized family activities times geometry masses, conditioned on nonoverlap; explicit Z_prior',
                'extra_cooccupancy_interactions':False,
                'unit_specific_candidate_availability_normalized_in_both_partitions':True,
                'conditional_on_visible_geometry':True}


@njit(cache=True, parallel=True)
def _map_decode(prefix, eta, allowed, geometry_allowed, geometry_adjustment, offsets, dest, edge_geo,
                ga, gb, gf, logq, n_nodes):
    n, nf = len(prefix), len(eta)
    scores = np.empty(n)
    selected = np.full((n, nf), -1, dtype=np.int64)
    for m in prange(n):
        best = np.full(n_nodes, -np.inf)
        previous = np.full(n_nodes, -1, dtype=np.int64)
        choice = np.full(n_nodes, -1, dtype=np.int64)
        best[0] = 0.
        for v in range(n_nodes - 1):
            if best[v] == -np.inf:
                continue
            for e in range(offsets[v], offsets[v + 1]):
                g = edge_geo[e]
                if g >= 0 and (not allowed[m, gf[g]] or not geometry_allowed[m, g]):
                    continue
                w = 0. if g < 0 else eta[gf[g]] + logq[g] + prefix[m, gb[g]] - prefix[m, ga[g]]
                if g >= 0 and geometry_adjustment.shape[1]:
                    w += geometry_adjustment[m,g]
                u = dest[e]
                if best[v] + w > best[u]:
                    best[u] = best[v] + w
                    previous[u] = v
                    choice[u] = g
        scores[m] = best[n_nodes - 1]
        v = n_nodes - 1
        while v > 0:
            g = choice[v]
            if g >= 0:
                selected[m, gf[g]] = g
            v = previous[v]
    return scores, selected
