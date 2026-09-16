"""Joint population activities on the exact configuration DAG.

Native emissions, opportunities, competing families and geometry mass are never
approximated here. A reference backward pass turns DAG edge weights into a
normalized Markov path measure. Activity changes then reweight these SAME paths
using multiplication, rather than re-exponentiating the native data on every
optimizer step. This is an algebraic change of measure, not a new state model.

Activity uncertainty uses a diagnosed Laplace proposal with exact training
likelihood importance correction. It is finite numerical posterior integration,
not calibrated accuracy/FDR. Geometry and the ancestral catalog remain fixed
empirical nuisances, and their provenance must accompany the output.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numba import njit, prange
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc


@njit(cache=True)
def _logadd(a, b):
    if a == -np.inf: return b
    if b == -np.inf: return a
    if b > a: a, b = b, a
    return a + math.log1p(math.exp(b-a))


@njit(cache=True, parallel=True)
def _reference(prefix, eta, physical, adjustment, offsets, dest, edge_geo,
               ga, gb, gf, logq, n_nodes):
    n = len(prefix)
    transition = np.zeros((n, len(dest)))
    z = np.zeros(n)
    for m in prange(n):
        bw = np.full(n_nodes, -np.inf)
        bw[-1] = 0.
        for v in range(n_nodes-2, -1, -1):
            for e in range(offsets[v], offsets[v+1]):
                g = edge_geo[e]
                if g >= 0 and not physical[m, g]: continue
                w = 0. if g < 0 else (eta[gf[g]] + logq[g] + adjustment[m,g]
                                      + prefix[m,gb[g]] - prefix[m,ga[g]])
                bw[v] = _logadd(bw[v], w+bw[dest[e]])
        z[m] = bw[0]
        for v in range(n_nodes-1):
            if bw[v] == -np.inf: continue
            for e in range(offsets[v], offsets[v+1]):
                g = edge_geo[e]
                if g >= 0 and not physical[m,g]: continue
                w = 0. if g < 0 else (eta[gf[g]] + logq[g] + adjustment[m,g]
                                      + prefix[m,gb[g]] - prefix[m,ga[g]])
                transition[m,e] = math.exp(w+bw[dest[e]]-bw[v])
    return transition, z


@njit(cache=True, parallel=True)
def _reweight(transition, multipliers, offsets, dest, edge_geo, gf, n_nodes,
              nf, ng, marginals, geometry):
    n = len(transition)
    z = np.zeros(n)
    inclusion = np.zeros((n, nf if marginals else 0))
    mass = np.zeros((n, ng if geometry else 0))
    for m in prange(n):
        fw = np.zeros(n_nodes)
        fw[0] = 1.
        for v in range(n_nodes-1):
            for e in range(offsets[v], offsets[v+1]):
                g = edge_geo[e]
                w = transition[m,e] * (1. if g < 0 else multipliers[gf[g]])
                fw[dest[e]] += fw[v]*w
        total = fw[-1]
        z[m] = math.log(total) if total > 0. else -np.inf
        if not marginals: continue
        bw = np.zeros(n_nodes)
        bw[-1] = 1.
        for v in range(n_nodes-2, -1, -1):
            for e in range(offsets[v], offsets[v+1]):
                g = edge_geo[e]
                w = transition[m,e] * (1. if g < 0 else multipliers[gf[g]])
                tail = w*bw[dest[e]]
                bw[v] += tail
                if g >= 0 and total > 0.:
                    p = fw[v]*tail/total
                    inclusion[m,gf[g]] += p
                    if geometry: mass[m,g] += p
    return z, inclusion, mass


@njit(cache=True, parallel=True)
def _reweight_log(transition, delta, excluded, offsets, dest, edge_geo, gf,
                  n_nodes,nf,ng,marginals,geometry):
    """Stable fallback for extreme optimizer proposals; same cached path measure."""
    n=len(transition);z=np.zeros(n)
    inc=np.zeros((n,nf if marginals else 0));mass=np.zeros((n,ng if geometry else 0))
    for m in prange(n):
        weights=np.full(len(dest),-np.inf)
        for e in range(len(dest)):
            g=edge_geo[e]
            if transition[m,e]>0 and (g<0 or gf[g]!=excluded):
                weights[e]=math.log(transition[m,e])+(0. if g<0 else delta[gf[g]])
        fw=np.full(n_nodes,-np.inf);fw[0]=0.
        for v in range(n_nodes-1):
            for e in range(offsets[v],offsets[v+1]):
                fw[dest[e]]=_logadd(fw[dest[e]],fw[v]+weights[e])
        z[m]=fw[-1]
        if not marginals:continue
        bw=np.full(n_nodes,-np.inf);bw[-1]=0.
        for v in range(n_nodes-2,-1,-1):
            for e in range(offsets[v],offsets[v+1]):
                g=edge_geo[e];tail=weights[e]+bw[dest[e]]
                bw[v]=_logadd(bw[v],tail)
                if g>=0:
                    p=math.exp(fw[v]+tail-z[m]);inc[m,gf[g]]+=p
                    if geometry:mass[m,g]+=p
    return z,inc,mass


@dataclass
class SparseTransitions:
    indptr: np.ndarray
    edge_ids: np.ndarray
    weights: np.ndarray

    def __len__(self):return len(self.indptr)-1

    @property
    def nbytes(self):return self.indptr.nbytes+self.edge_ids.nbytes+self.weights.nbytes

    @classmethod
    def from_dense(cls,t):
        row,edge=np.nonzero(t)
        counts=np.bincount(row,minlength=len(t))
        return cls(np.r_[0,np.cumsum(counts)].astype(np.int64),edge.astype(np.int32),t[row,edge])

    @classmethod
    def concatenate(cls,parts):
        lengths=np.concatenate([np.diff(p.indptr) for p in parts])
        return cls(np.r_[0,np.cumsum(lengths)].astype(np.int64),
                   np.concatenate([p.edge_ids for p in parts]),np.concatenate([p.weights for p in parts]))


@njit(cache=True,parallel=True)
def _sparse_reweight(ptr,edge_ids,weights,multipliers,src,dest,edge_geo,gf,n_nodes,nf,ng,marginals,geometry):
    n=len(ptr)-1;z=np.zeros(n)
    inclusion=np.zeros((n,nf if marginals else 0));mass=np.zeros((n,ng if geometry else 0))
    for m in prange(n):
        fw=np.zeros(n_nodes);fw[0]=1.
        for j in range(ptr[m],ptr[m+1]):
            e=edge_ids[j];g=edge_geo[e]
            w=weights[j]*(1. if g<0 else multipliers[gf[g]])
            fw[dest[e]]+=fw[src[e]]*w
        total=fw[-1];z[m]=math.log(total) if total>0 else -np.inf
        if not marginals:continue
        bw=np.zeros(n_nodes);bw[-1]=1.
        for j in range(ptr[m+1]-1,ptr[m]-1,-1):
            e=edge_ids[j];g=edge_geo[e]
            w=weights[j]*(1. if g<0 else multipliers[gf[g]])
            tail=w*bw[dest[e]];bw[src[e]]+=tail
            if g>=0 and total>0:
                p=fw[src[e]]*tail/total;inclusion[m,gf[g]]+=p
                if geometry:mass[m,g]+=p
    return z,inclusion,mass


@njit(cache=True,parallel=True)
def _sparse_reweight_log(ptr,edge_ids,weights,delta,excluded,src,dest,edge_geo,gf,n_nodes,nf,ng,marginals,geometry):
    n=len(ptr)-1;z=np.zeros(n)
    inclusion=np.zeros((n,nf if marginals else 0));mass=np.zeros((n,ng if geometry else 0))
    for m in prange(n):
        lw=np.full(ptr[m+1]-ptr[m],-np.inf)
        fw=np.full(n_nodes,-np.inf);fw[0]=0.
        for j in range(ptr[m],ptr[m+1]):
            e=edge_ids[j];g=edge_geo[e]
            if g>=0 and gf[g]==excluded:continue
            w=math.log(weights[j])+(0. if g<0 else delta[gf[g]])
            lw[j-ptr[m]]=w
            fw[dest[e]]=_logadd(fw[dest[e]],fw[src[e]]+w)
        z[m]=fw[-1]
        if not marginals:continue
        bw=np.full(n_nodes,-np.inf);bw[-1]=0.
        for j in range(ptr[m+1]-1,ptr[m]-1,-1):
            e=edge_ids[j];g=edge_geo[e];tail=lw[j-ptr[m]]+bw[dest[e]]
            bw[src[e]]=_logadd(bw[src[e]],tail)
            if g>=0:
                p=math.exp(fw[src[e]]+tail-z[m]);inclusion[m,gf[g]]+=p
                if geometry:mass[m,g]+=p
    return z,inclusion,mass


@dataclass
class PathCache:
    kernel: object
    reference_eta: np.ndarray
    transition: object
    reference_log_z: np.ndarray

    @classmethod
    def build(cls, kernel, values, eta, physical=None, adjustment=None):
        values = np.asarray(values, float)
        n, ng = len(values), len(kernel.ga)
        physical = np.ones((n,ng),bool) if physical is None else np.asarray(physical,bool)
        adjustment = np.zeros((n,ng)) if adjustment is None else np.asarray(adjustment,float)
        if values.shape != (n,kernel.k) or physical.shape != (n,ng) or adjustment.shape != (n,ng):
            raise ValueError('Full observation domain and one context value per geometry required')
        if not np.isfinite(values).all() or not np.isfinite(adjustment).all():
            raise ValueError('Finite native evidence/context required')
        eta = np.asarray(eta,float)
        if eta.shape != (kernel.f,) or not np.isfinite(eta).all():
            raise ValueError('One finite reference activity per family required')
        prefix = np.c_[np.zeros(n),np.cumsum(values,axis=1)]
        t,z = _reference(prefix,eta,physical,adjustment,kernel.offsets,kernel.dest,
                         kernel.edge_geo,kernel.ga,kernel.gb,kernel.gf,kernel.logq,kernel.n_nodes)
        if not np.isfinite(t).all() or not np.isfinite(z).all():
            raise FloatingPointError('Invalid reference path measure')
        return cls(kernel,eta.copy(),SparseTransitions.from_dense(t),z)

    @classmethod
    def concatenate(cls,caches):
        first=caches[0]
        if any(c.kernel is not first.kernel or not np.array_equal(c.reference_eta,first.reference_eta) for c in caches):
            raise ValueError('Matching reference models required to concatenate path caches')
        return cls(first.kernel,first.reference_eta.copy(),SparseTransitions.concatenate([c.transition for c in caches]),
                   np.concatenate([c.reference_log_z for c in caches]))

    def evaluate(self, eta, *, marginals=True, geometry=False, exclude_family=None):
        eta = np.asarray(eta,float)
        if eta.shape != self.reference_eta.shape or not np.isfinite(eta).all():
            raise ValueError('One finite activity per family required')
        k = self.kernel
        delta = eta-self.reference_eta
        if np.max(np.abs(delta)) > 60:
            raise FloatingPointError('Activity change exceeds cached numerical range; recenter the cache')
        multipliers=np.exp(delta)
        if exclude_family is not None:
            if not 0<=exclude_family<k.f:raise ValueError('Invalid excluded family')
            multipliers[exclude_family]=0.
        if isinstance(self.transition,SparseTransitions):
            if not hasattr(k,'_population_edge_sources'):
                k._population_edge_sources=np.repeat(np.arange(k.n_nodes,dtype=np.int64),np.diff(k.offsets))
            t=self.transition
            args=(t.indptr,t.edge_ids,t.weights,multipliers,k._population_edge_sources,k.dest,k.edge_geo,k.gf,
                  k.n_nodes,k.f,len(k.ga),marginals or geometry,geometry)
            z,p,g=_sparse_reweight(*args)
            if not np.isfinite(z).all() or not np.isfinite(p).all() or np.any(p>1+1e-7):
                z,p,g=_sparse_reweight_log(t.indptr,t.edge_ids,t.weights,delta,-1 if exclude_family is None else exclude_family,
                    k._population_edge_sources,k.dest,k.edge_geo,k.gf,k.n_nodes,k.f,len(k.ga),marginals or geometry,geometry)
        else:
            z,p,g = _reweight(self.transition,multipliers,k.offsets,k.dest,k.edge_geo,
                               k.gf,k.n_nodes,k.f,len(k.ga),marginals or geometry,geometry)
            if not np.isfinite(z).all() or not np.isfinite(p).all() or np.any(p>1+1e-7):
                z,p,g=_reweight_log(self.transition,delta,-1 if exclude_family is None else exclude_family,
                    k.offsets,k.dest,k.edge_geo,k.gf,k.n_nodes,k.f,len(k.ga),marginals or geometry,geometry)
        if not np.isfinite(z).all() or not np.isfinite(p).all() or np.any(p>1+1e-7):
            raise FloatingPointError('Invalid cached path reweighting even in log space')
        return dict(log_partition=z+self.reference_log_z,family_inclusion=p,geometry_mass=g)


class JointActivityPosterior:
    """Proper joint posterior, conditional on the supplied geometry and context."""
    def __init__(self, observed, prior, prior_mean=-2., prior_sd=2.):
        if observed.kernel is not prior.kernel or len(observed.transition)!=len(prior.transition):
            raise ValueError('Same exact model/evidence units in the two partitions required')
        if not np.isfinite(prior_mean) or not np.isfinite(prior_sd) or prior_sd<=0:
            raise ValueError('Proper finite Gaussian activity prior required')
        self.observed,self.prior=observed,prior
        self.mean=float(prior_mean);self.sd=float(prior_sd)

    def objective(self, eta, gradient=True):
        a=self.observed.evaluate(eta,marginals=gradient)
        b=self.prior.evaluate(eta,marginals=gradient)
        delta=(eta-self.mean)/self.sd
        value=float((b['log_partition']-a['log_partition']).sum()+.5*(delta@delta))
        if not gradient:return value
        grad=(b['family_inclusion']-a['family_inclusion']).sum(0)+(eta-self.mean)/self.sd**2
        return value,grad

    def fit(self, *, max_iterations=250, report=None):
        fits=[]
        for start in (self.mean,0.):
            iteration=[0]
            def callback(x):
                iteration[0]+=1
                if report:report('fit',dict(start=start,iteration=iteration[0]))
            # Wide numerical bounds are diagnosed; a boundary MAP is not treated
            # as an interior Gaussian posterior. These do not cap call counts.
            fit=minimize(self.objective,np.full(self.observed.kernel.f,start),jac=True,
                         method='L-BFGS-B',bounds=[(-24.,24.)]*self.observed.kernel.f,
                         callback=callback,options=dict(maxiter=max_iterations,ftol=1e-12,gtol=1e-5,maxls=35))
            fits.append(dict(eta=fit.x,objective=float(fit.fun),success=bool(fit.success),
                             iterations=int(fit.nit),message=str(fit.message),
                             max_abs_gradient=float(np.max(np.abs(fit.jac))),start=start))
        best=min(fits,key=lambda v:v['objective'])
        if np.any(np.abs(best['eta'])>=23.99):
            raise FloatingPointError('Activity MAP reached numerical bounds; no Laplace confidence issued')
        if not best['success'] and best['max_abs_gradient']>1e-3:
            raise RuntimeError('Joint activity fit did not converge; no confidence issued')
        return best,fits

    def hessian(self, eta, *, step=2e-4, report=None):
        """Observed information: prior covariance MINUS data covariance + precision.

        For binary family inclusion, Cov(I_f,I_j) = P(not j) *
        [P(f)-P(f|not j)]. Exact exclusion partitions obtain all entries of
        one covariance column together. No diagonal/Fisher approximation or
        finite-difference noise is used; finite differences are a test oracle.
        """
        n=len(eta);h=np.eye(n)/self.sd**2
        bases=[(cache,cache.evaluate(eta),sign) for cache,sign in [(self.prior,1.),(self.observed,-1.)]]
        for j in range(n):
            for cache,base,sign in bases:
                excluded=cache.evaluate(eta,exclude_family=j)
                absent=np.exp(np.minimum(excluded['log_partition']-base['log_partition'],0.))
                column=(absent[:,None]*(base['family_inclusion']-excluded['family_inclusion'])).sum(0)
                h[:,j]+=sign*column
            if report and (j%8==0 or j==n-1):report('hessian',dict(column=j+1,total=n))
        asymmetry=float(np.max(np.abs(h-h.T)));h=(h+h.T)/2
        eigenvalues,eigenvectors=np.linalg.eigh(h)
        if eigenvalues[0]<=1e-7:
            raise FloatingPointError(f'Nonpositive/degenerate observed information: {eigenvalues[0]:g}; no Gaussian confidence issued')
        covariance=(eigenvectors/eigenvalues)@eigenvectors.T
        return h,covariance,dict(minimum_eigenvalue=float(eigenvalues[0]),
                                maximum_eigenvalue=float(eigenvalues[-1]),maximum_asymmetry=asymmetry,
                                method='exact_binary_inclusion_exclusion_covariance')

    def draws(self, eta, covariance, *, number=128, seed=20260907, report=None):
        if number<8 or number&(number-1):raise ValueError('Power-of-two quadrature size >=8 required')
        l=np.linalg.cholesky(covariance)
        u=qmc.Sobol(len(eta),scramble=True,seed=seed).random_base2(int(math.log2(number)))
        normals=ndtri(np.clip(u,1e-12,1-1e-12));draws=eta+normals@l.T
        lw=np.empty(number)
        # Proposal normalizing constants cancel because all draws share it.
        for j,point in enumerate(draws):
            lw[j]=-self.objective(point,gradient=False)+.5*float(normals[j]@normals[j])
            if report and (j%16==0 or j==number-1):report('importance',dict(draw=j+1,total=number))
        lw-=logsumexp(lw);weights=np.exp(lw)
        return draws,lw,dict(draws=number,effective_sample_size=float(1/(weights@weights)),
                             maximum_weight=float(weights.max()),
                             method='Sobol_Laplace_proposal_exact_training_likelihood_importance_correction',
                             covariance_approximation='full_observed_information',seed=seed)

    def corrected_chain_draws(self,eta,covariance,*,number=128,seed=20260907,report=None):
        """Exact-target, full-Hessian-preconditioned Hamiltonian Monte Carlo.

        The failed high-dimensional importance and pCN pilots are not reused.
        Leapfrog proposals use the exact joint score AND gradient, with an
        energy-error Metropolis correction. Adaptation stops after warm-up.
        Correlated draws have uniform weights; convergence is measured separately.
        """
        from scipy.signal import correlate
        if number<128 or number&(number-1):
            raise ValueError('HMC requires a power-of-two draw count >=128')
        l=np.linalg.cholesky(covariance);rng=np.random.default_rng(seed)
        chain_count=4;kept=number//chain_count;warmup=192;thin=1
        chains=[];acceptance=[]
        for chain in range(chain_count):
            z=rng.normal(size=len(eta))*.25
            def potential(x):
                value,gradient=self.objective(eta+l@x)
                return value,l.T@gradient
            value,gradient=potential(z);epsilon=.3;accepted=0;block=[];samples=[];divergences=0
            for step in range(warmup+kept*thin):
                momentum=rng.normal(size=len(eta));initial_energy=value+.5*float(momentum@momentum)
                proposal=z.copy();p=momentum-.5*epsilon*gradient
                leapfrogs=int(rng.integers(3,8));valid=True
                for leap in range(leapfrogs):
                    proposal+=epsilon*p
                    try:updated,g=potential(proposal)
                    except FloatingPointError:
                        valid=False;break
                    if not np.isfinite(updated) or not np.isfinite(g).all():
                        valid=False;break
                    p-=epsilon*g*(.5 if leap==leapfrogs-1 else 1.)
                difference=initial_energy-updated-.5*float(p@p) if valid else -np.inf
                probability=math.exp(min(0.,difference))
                take=math.log(rng.random())<difference
                if take:z,value,gradient=proposal,updated,g
                accepted+=int(take);block.append(probability)
                if step>=warmup and (not valid or abs(difference)>100):divergences+=1
                if step<warmup and (step+1)%16==0:
                    epsilon=float(np.clip(epsilon*math.exp((np.mean(block)-.8)*1.5),.015,1.));block=[]
                if step>=warmup and (step-warmup+1)%thin==0:samples.append((eta+l@z).copy())
                if report and (step%32==0 or step+1==warmup+kept*thin):
                    report('posterior_chain',dict(chain=chain+1,step=step+1,total=warmup+kept*thin,epsilon=round(epsilon,3)))
            chains.append(samples);acceptance.append(dict(epsilon=epsilon,rate=accepted/(warmup+kept*thin),divergences=divergences))
        chains=np.asarray(chains)
        # Split-chain convergence and autocorrelation diagnostics; not a claim
        # that this short pilot is equilibrated. Failure remains visible.
        halves=np.concatenate([chains[:,:kept//2],chains[:,kept//2:2*(kept//2)]],axis=0)
        length=halves.shape[1]
        within=halves.var(axis=1,ddof=1).mean(0);between=length*halves.mean(1).var(axis=0,ddof=1)
        rhat=np.sqrt(np.maximum(((length-1)/length*within+between/length)/np.maximum(within,1e-15),1.))
        ess=np.zeros(len(eta))
        for f in range(len(eta)):
            ac=[]
            for c in chains[:,:,f]:
                centered=c-c.mean();corr=correlate(centered,centered,mode='full',method='fft')[len(c)-1:]
                ac.append(corr/max(corr[0],1e-15))
            rho=np.mean(ac,axis=0);total=0.
            for j in range(1,len(rho)-1,2):
                pair=rho[j]+rho[j+1]
                if pair<=0:break
                total+=pair
            ess[f]=min(chain_count*kept,chain_count*kept/max(1,1+2*total))
        draws=chains.reshape(-1,len(eta))
        diagnostic=dict(draws=len(draws),method='exact_target_full_Hessian_preconditioned_HMC',
            effective_sample_size=float(np.min(ess)),median_ESS=float(np.median(ess)),ESS_by_activity=ess.tolist(),
            maximum_split_Rhat=float(np.max(rhat)),median_split_Rhat=float(np.median(rhat)),split_Rhat_by_activity=rhat.tolist(),
            chains=chain_count,warmup=warmup,thin=thin,acceptance=acceptance,seed=seed,
            convergence_claim=False,laplace_is_proposal_only=True)
        return draws,np.full(len(draws),-math.log(len(draws))),diagnostic


def _integrate_recipient_dense(prior, observed, draws, log_weights, *, export_geometry=True):
    """Ratio of integrated joint likelihoods, never an unweighted posterior mean."""
    n=len(observed.transition);f=observed.kernel.f;ng=len(observed.kernel.ga)
    log_weights=np.asarray(log_weights,float)-logsumexp(log_weights)
    prior_mass=np.zeros((n,f));post_mass=np.zeros((n,f))
    geometry_mass=np.zeros((n,ng if export_geometry else 0))
    scale=np.full(n,-np.inf);den=np.zeros(n);squares=np.zeros(n)
    for point,lw in zip(draws,log_weights):
        p0=prior.evaluate(point)
        p=observed.evaluate(point,geometry=export_geometry)
        prior_mass+=math.exp(lw)*p0['family_inclusion']
        logw=lw+p['log_partition']-p0['log_partition']
        newscale=np.maximum(scale,logw)
        old=np.exp(scale-newscale);new=np.exp(logw-newscale)
        den=den*old+new;squares=squares*old**2+new**2
        post_mass=post_mass*old[:,None]+new[:,None]*p['family_inclusion']
        if export_geometry:geometry_mass=geometry_mass*old[:,None]+new[:,None]*p['geometry_mass']
        scale=newscale
    post_mass/=den[:,None]
    if export_geometry:geometry_mass/=den[:,None]
    eps=1e-12
    logit=lambda p:np.log(np.clip(p,eps,1-eps))-np.log1p(-np.clip(p,eps,1-eps))
    return dict(prior_inclusion=prior_mass,family_inclusion=post_mass,
                geometry_mass=geometry_mass,recipient_log_bf=logit(post_mass)-logit(prior_mass),
                log_predictive=scale+np.log(den),effective_sample_size=den**2/squares)


@njit(cache=True,parallel=True)
def _accumulate_sparse_geometry(indptr,edge_ids,transition,multiplier,sources,dest,edge_geo,gf,
                                n_nodes,weights,output):
    """Accumulate only nonzero edge marginals, without a dense G array per draw."""
    valid=np.ones(len(weights),dtype=np.bool_)
    for m in prange(len(weights)):
        if weights[m]==0:continue
        first,last=indptr[m],indptr[m+1]
        forward=np.zeros(n_nodes);backward=np.zeros(n_nodes)
        forward[0]=1.;backward[n_nodes-1]=1.
        for z in range(first,last):
            e=edge_ids[z];g=edge_geo[e]
            weight=transition[z]*(1. if g<0 else multiplier[gf[g]])
            forward[dest[e]]+=forward[sources[e]]*weight
        total=forward[n_nodes-1]
        if total<=0 or not np.isfinite(total):valid[m]=False;continue
        for z in range(last-1,first-1,-1):
            e=edge_ids[z];g=edge_geo[e]
            weight=transition[z]*(1. if g<0 else multiplier[gf[g]])
            backward[sources[e]]+=weight*backward[dest[e]]
        probabilities=np.zeros(last-first)
        for z in range(first,last):
            e=edge_ids[z];g=edge_geo[e]
            if g<0:continue
            p=forward[sources[e]]*transition[z]*multiplier[gf[g]]*backward[dest[e]]/total
            if not np.isfinite(p) or p>1+1e-7:valid[m]=False;break
            probabilities[z-first]=p
        if not valid[m]:continue
        for z in range(first,last):
            g=edge_geo[edge_ids[z]]
            if g>=0:output[m,g]+=weights[m]*probabilities[z-first]
    return valid


def integrate_recipient(prior, observed, draws, log_weights, *, export_geometry=True):
    """Exact same ratio estimator, with sparse geometry accumulation.

    First determine normalized recipient parameter weights without allocating a
    full geometry matrix for every draw. Then accumulate geometry contributions
    only along nonzero cached edges. No probability/candidate is thresholded or
    discarded. The dense implementation remains a regression oracle/fallback.
    """
    if not export_geometry or not isinstance(observed.transition,SparseTransitions):
        return _integrate_recipient_dense(prior,observed,draws,log_weights,export_geometry=export_geometry)
    n=len(observed.transition);k=observed.kernel
    log_weights=np.asarray(log_weights,float)-logsumexp(log_weights)
    prior_mass=np.zeros((n,k.f));post_mass=np.zeros_like(prior_mass)
    logw=np.empty((len(draws),n));scale=np.full(n,-np.inf);den=np.zeros(n);squares=np.zeros(n)
    for s,(point,lw) in enumerate(zip(draws,log_weights)):
        p0=prior.evaluate(point);p=observed.evaluate(point)
        prior_mass+=math.exp(lw)*p0['family_inclusion']
        logw[s]=lw+p['log_partition']-p0['log_partition']
        newscale=np.maximum(scale,logw[s]);old=np.exp(scale-newscale);new=np.exp(logw[s]-newscale)
        den=den*old+new;squares=squares*old**2+new**2
        post_mass=post_mass*old[:,None]+new[:,None]*p['family_inclusion'];scale=newscale
    post_mass/=den[:,None];log_predictive=scale+np.log(den)
    masses=np.zeros((n,len(k.ga)));t=observed.transition
    for s,point in enumerate(draws):
        weights=np.exp(logw[s]-log_predictive)
        valid=_accumulate_sparse_geometry(t.indptr,t.edge_ids,t.weights,np.exp(point-observed.reference_eta),
            k._population_edge_sources,k.dest,k.edge_geo,k.gf,k.n_nodes,weights,masses)
        if not valid.all():
            # Same log-space fallback used by PathCache; invalid rows were not
            # partially accumulated, so every contribution enters exactly once.
            dense=observed.evaluate(point,geometry=True)['geometry_mass']
            masses[~valid]+=weights[~valid,None]*dense[~valid]
    eps=1e-12
    logit=lambda p:np.log(np.clip(p,eps,1-eps))-np.log1p(-np.clip(p,eps,1-eps))
    return dict(prior_inclusion=prior_mass,family_inclusion=post_mass,geometry_mass=masses,
        recipient_log_bf=logit(post_mass)-logit(prior_mass),log_predictive=log_predictive,
        effective_sample_size=den**2/squares)
