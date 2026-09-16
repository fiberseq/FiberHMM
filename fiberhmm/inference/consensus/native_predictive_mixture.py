"""Full finite-mixture distinguishability on one actual native lattice.

Models are given distributions over protected masks, including arbitrary
unions. This does not fit, match, merge, recall, or change a native call. Shared
independent columns cancel, but equal marginal modification probabilities do
NOT erase correlations induced by the latent geometry mixture.

Every positive-weight geometry contributes to marginal likelihoods. Identical
observed masks are exactly quotiented by adding their weights, not sampled or
culled. References enumerate small outcome spaces or use a fixed-budget,
seeded Monte Carlo TV estimate with a bounded-variable Hoeffding interval.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Mapping

import numpy as np
from scipy.special import logsumexp

from .native_resolution import _read, _intervals, _membership


def _positive_integer(value, name, *, zero=False):
    if isinstance(value, (bool,np.bool_)) or not isinstance(value, Integral) or value < (0 if zero else 1):
        raise ValueError(name+' must be '+('a nonnegative' if zero else 'a positive')+' integer')
    return int(value)


def interval_mixture(positions, protected_interval_sets, *, weights=None, log_weights=None):
    """Convert specified interval unions to masks WITHOUT changing a lattice."""
    p=np.asarray(positions)
    if p.ndim!=1 or (p.size and p.dtype.kind not in 'iu') or np.any(p<0) or np.any(p[1:]<=p[:-1]):
        raise ValueError('Increasing actual integer opportunity positions required')
    masks=np.asarray([_membership(p,_intervals(spans)) for spans in protected_interval_sets],bool)
    if len(masks)==0:
        raise ValueError('At least one geometry is required')
    result=dict(protected_masks=masks)
    if weights is not None:result['weights']=weights
    if log_weights is not None:result['log_weights']=log_weights
    return result


def _parse_mixture(model, n_positions):
    raw=np.asarray(model['protected_masks'])
    if (raw.ndim!=2 or raw.shape[1]!=n_positions or len(raw)==0 or
            raw.dtype.kind not in 'biu' or np.any((raw!=0)&(raw!=1))):
        raise ValueError('One binary protected mask per geometry on the complete supplied lattice required')
    if ('weights' in model)==('log_weights' in model):
        raise ValueError('Provide exactly one of normalized weights or normalized log_weights')
    if 'weights' in model:
        weights=np.asarray(model['weights'],float)
        if weights.shape!=(len(raw),) or np.any(~np.isfinite(weights)) or np.any(weights<0) or not np.any(weights>0):
            raise ValueError('Finite nonnegative geometry weights with positive total required')
        total=math.fsum(map(float,weights))
        if not math.isfinite(total) or abs(total-1)>1e-9:
            raise ValueError('Geometry weights must be normalized to one')
        log_weights=np.full(len(raw),-np.inf)
        positive=weights>0
        log_weights[positive]=np.log(weights[positive])
    else:
        log_weights=np.asarray(model['log_weights'],float)
        if (log_weights.shape!=(len(raw),) or np.any(np.isnan(log_weights)|np.isposinf(log_weights)) or
                not np.isfinite(log_weights).any()):
            raise ValueError('Normalized finite or negative-infinite log weights required')
        positive=np.isfinite(log_weights)
    normalization=float(logsumexp(np.sort(log_weights[positive])))
    if abs(normalization)>1e-9:
        raise ValueError('Geometry log weights must be normalized to log total zero')
    log_weights=log_weights[positive]-normalization
    masks=raw[positive].astype(bool,copy=True)
    return dict(masks=masks,log_weights=log_weights,summary=dict(
        input_geometries=len(raw),positive_weight_geometries=len(masks),zero_weight_geometries=int((~positive).sum()),
        input_log_normalization_error=normalization,
        finite_log_weights_underflowing_as_probabilities=int((np.exp(log_weights)==0).sum()),
        all_positive_weight_geometries_retained=True))


def _quotient(model, columns):
    masks=model['masks'][:,columns]
    packed=np.packbits(masks,axis=1) if masks.shape[1] else np.zeros((len(masks),1),np.uint8)
    _,first,inverse=np.unique(packed,axis=0,return_index=True,return_inverse=True)
    # Canonical summation order makes input permutation irrelevant. Geometry
    # aliases add mass; they never add a likelihood or count bonus.
    order=np.lexsort((model['log_weights'],inverse))
    combined=np.full(len(first),-np.inf)
    np.logaddexp.at(combined,inverse[order],model['log_weights'][order])
    combined-=logsumexp(np.sort(combined))
    result=dict(masks=masks[first].copy(),log_weights=combined,
                summary=model['summary']|dict(unique_induced_masks=len(first),
                    duplicate_induced_masks_quotiented=len(masks)-len(first)))
    result['mask_matrix']=result['masks'].astype(float).T.copy()
    return result


class NativeMixtureComparison:
    """Reusable fixed-lattice comparison; references never inspect actual hits.

    The supplied unit's positions/emissions and both mixtures are frozen by
    copying. ``observed`` can then evaluate any binary pattern on this SAME
    lattice. No per-pattern geometry or emission fitting occurs.
    """
    def __init__(self, unit: Mapping, mixture_a: Mapping, mixture_b: Mapping, *, maximum_working_bytes=64*1024**2):
        self.maximum_working_bytes=_positive_integer(maximum_working_bytes,'maximum_working_bytes')
        positions,hits,pa,pp,_=_read(unit)
        self.positions=positions.copy();self.default_hits=hits.copy();self.pa=pa.copy();self.pp=pp.copy()
        self.unit_id=unit.get('unit_id')
        original_a=_parse_mixture(mixture_a,len(positions));original_b=_parse_mixture(mixture_b,len(positions))
        all_a=original_a['masks'].all(0);none_a=~original_a['masks'].any(0)
        all_b=original_b['masks'].all(0);none_b=~original_b['masks'].any(0)
        common_constant=(all_a&all_b)|(none_a&none_b)
        self.columns=(pa!=pp)&~common_constant
        self.shared_protected=(pa!=pp)&all_a&all_b
        self.a=_quotient(original_a,self.columns);self.b=_quotient(original_b,self.columns)
        self.variable_pa=self.pa[self.columns];self.variable_pp=self.pp[self.columns]
        self.same_coefficients=(np.array_equal(self.a['masks'],self.b['masks']) and
                                np.array_equal(self.a['log_weights'],self.b['log_weights']))
        k=int(self.columns.sum());g=max(len(self.a['masks']),len(self.b['masks']))
        # Chunking changes memory/work layout only, never the geometry catalog.
        bytes_per_outcome=8*(4*g+5*k+32)
        self.batch_size=self.maximum_working_bytes//max(1,bytes_per_outcome)
        if self.batch_size<1:
            raise MemoryError('One complete mixture evaluation exceeds working budget; no geometry was truncated')
        self.batch_size=int(min(4096,self.batch_size))

    def metadata(self):
        return dict(contract_version='native_predictive_mixture_v1',observed_opportunities=len(self.positions),
            variable_informative_positions=self.positions[self.columns].tolist(),
            variable_informative_opportunities=int(self.columns.sum()),
            cancelled_common_independent_positions=self.positions[~self.columns].tolist(),
            mixture_a=self.a['summary'],mixture_b=self.b['summary'],
            algebraically_identical_induced_mixtures=self.same_coefficients,
            geometry_masks_and_weights_not_fitted_or_culled=True,
            common_accessible_base=True,latent_geometry_correlations_preserved=True,
            same_actual_observations_for_both_models=True,
            equal_marginal_probabilities_not_used_to_cancel_correlated_columns=True,
            native_conditional_independence_given_geometry=True,
            maximum_working_bytes=self.maximum_working_bytes,evaluation_batch_size=self.batch_size,
            budget_bounds_evaluation_temporaries_not_input_catalog_storage=True,
            scores_are_not_posteriors_or_FDR=True,not_automatic_equivalence_or_matching=True,
            no_shared_identity_or_prevalence_inferred=True,
            floating_point_not_interval_arithmetic_certified=True)

    def _relative(self, patterns, model):
        patterns=np.asarray(patterns,np.int8)
        if patterns.ndim!=2 or patterns.shape[1]!=int(self.columns.sum()):
            raise ValueError('Pattern matrix does not match fixed variable lattice')
        if not patterns.shape[1]:return np.zeros(len(patterns))
        miss=np.log1p(-self.variable_pp)-np.log1p(-self.variable_pa)
        hit=np.log(self.variable_pp)-np.log(self.variable_pa)
        result=np.empty(len(patterns))
        for first in range(0,len(patterns),self.batch_size):
            block=patterns[first:first+self.batch_size]
            values=miss+block*(hit-miss)
            likelihood=values @ model['mask_matrix']
            likelihood+=model['log_weights']
            result[first:first+len(block)]=logsumexp(likelihood,axis=1)
        return result

    def observed(self, hits=None, *, unit_id=None):
        h=self.default_hits if hits is None else np.asarray(hits)
        if (h.shape!=self.positions.shape or (h.size and h.dtype.kind not in 'biu') or
                np.any((h!=0)&(h!=1))):
            raise ValueError('One actual binary observation per original opportunity required')
        base=float(np.where(h,np.log(self.pa),np.log1p(-self.pa)).sum())
        native=np.where(h,np.log(self.pp)-np.log(self.pa),np.log1p(-self.pp)-np.log1p(-self.pa))
        shared=float(native[self.shared_protected].sum())
        pattern=h[self.columns].reshape(1,-1)
        a=float(self._relative(pattern,self.a)[0])+shared
        b=a if self.same_coefficients else float(self._relative(pattern,self.b)[0])+shared
        return dict(unit_id=self.unit_id if unit_id is None else unit_id,
            observed_log_marginal_a_over_accessible=a,observed_log_marginal_b_over_accessible=b,
            log_lr_b_over_a=0. if self.same_coefficients else b-a,
            common_accessible_log_likelihood=base,
            observed_log_likelihood_a=base+a,observed_log_likelihood_b=base+b,
            actual_hits=int(np.asarray(h).sum()),actual_misses=int(len(h)-np.asarray(h).sum()),
            common_independent_observation_log_ratio=shared,
            actual_pattern_did_not_change_mixture_or_reference=True)

    def reference(self, *, max_exact_outcomes=4096, monte_carlo_samples=0, seed=0, monte_carlo_confidence=.95,
                  include_outcome_table=False):
        budget=_positive_integer(max_exact_outcomes,'max_exact_outcomes')
        samples=_positive_integer(monte_carlo_samples,'monte_carlo_samples',zero=True)
        seed=_positive_integer(seed,'seed',zero=True)
        if (isinstance(monte_carlo_confidence,(bool,np.bool_)) or not isinstance(monte_carlo_confidence,Real) or
                not math.isfinite(monte_carlo_confidence) or not 0<monte_carlo_confidence<1):
            raise ValueError('Monte Carlo confidence must lie strictly between zero and one')
        k=int(self.columns.sum());within=k<63 and k<=budget.bit_length()-1
        if self.same_coefficients:
            exact=dict(available=True,method='identical_induced_mask_coefficients',outcomes_enumerated=0,
                total_variation=0.,optimal_equal_prior_accuracy=.5,KL_a_vs_b_nats=0.,KL_b_vs_a_nats=0.,outcome_table=None)
        elif within:
            count=1<<k;codes=np.arange(count,dtype=np.uint64)
            patterns=((codes[:,None]>>np.arange(k,dtype=np.uint64))&np.uint64(1)).astype(np.int8)
            base=np.where(patterns,np.log(self.variable_pa),np.log1p(-self.variable_pa)).sum(1)
            loga=base+self._relative(patterns,self.a);logb=base+self._relative(patterns,self.b)
            za,zb=float(logsumexp(loga)),float(logsumexp(logb))
            if max(abs(za),abs(zb))>1e-8:
                raise ArithmeticError('Full finite-mixture outcome probabilities failed normalization')
            loga-=za;logb-=zb;pa=np.exp(loga);pb=np.exp(logb)
            tv=float(.5*np.abs(pa-pb).sum())
            exact=dict(available=True,method='complete_joint_binary_enumeration',outcomes_enumerated=count,
                total_variation=tv,optimal_equal_prior_accuracy=.5+.5*tv,
                KL_a_vs_b_nats=max(0.,float(pa @ (loga-logb))),
                KL_b_vs_a_nats=max(0.,float(pb @ (logb-loga))),
                probability_log_normalization_errors=[za,zb],outcome_table=None,
                numerical_zero_not_proof_of_identical_distributions=True)
            if include_outcome_table:
                exact['outcome_table']=[dict(bit_code=int(code),probability_a=float(a),probability_b=float(b),
                    log_lr_b_over_a=float(lb-la)) for code,a,b,la,lb in zip(codes,pa,pb,loga,logb)]
                exact['outcome_bit_order']='bit i is variable_informative_positions[i]'
        else:
            exact=dict(available=False,method=None,reason='exact_joint_outcome_budget_exceeded',
                outcomes_enumerated=0,total_variation=None,optimal_equal_prior_accuracy=None,
                KL_a_vs_b_nats=None,KL_b_vs_a_nats=None,outcome_table=None)
        mc=dict(available=False,requested_samples=samples,samples=0,
                reason='algebraic_identity_already_exact' if self.same_coefficients else 'not_requested')
        if samples and not self.same_coefficients:
            streams=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
            probabilities=[];cdfs=[];underflows=[]
            for model in (self.a,self.b):
                weights=np.exp(model['log_weights']);weights/=weights.sum()
                probabilities.append(weights);cdf=np.cumsum(weights);cdf[-1]=1.;cdfs.append(cdf)
                underflows.append(int((np.isfinite(model['log_weights'])&(weights==0)).sum()))
            total=0.;square_total=0.
            for first in range(0,samples,self.batch_size):
                n=min(self.batch_size,samples-first)
                model_b=streams[0].random(n)<.5;uniform=streams[1].random(n)
                masks=np.empty((n,k),bool)
                for choose,model,cdf in ((~model_b,self.a,cdfs[0]),(model_b,self.b,cdfs[1])):
                    indices=np.searchsorted(cdf,uniform[choose],side='right')
                    masks[choose]=model['masks'][indices]
                emission=np.where(masks,self.variable_pp,self.variable_pa)
                patterns=(streams[2].random((n,k))<emission).astype(np.int8)
                llr=self._relative(patterns,self.b)-self._relative(patterns,self.a)
                values=np.abs(np.tanh(.5*llr))
                total+=math.fsum(map(float,values));square_total+=float(values@values)
            estimate=total/samples
            half=math.sqrt(math.log(2/(1-float(monte_carlo_confidence)))/(2*samples))
            interval=[max(0.,estimate-half),min(1.,estimate+half)]
            mc=dict(available=True,method='equal_mixture_mean_absolute_tanh_half_logLR',
                requested_samples=samples,samples=samples,seed=seed,seed_streams=3,
                sampling_distribution='0.5*P_A + 0.5*P_B; latent geometry then native independent observations',
                total_variation_estimate=estimate,total_variation_interval=interval,
                optimal_equal_prior_accuracy_estimate=.5+.5*estimate,
                optimal_equal_prior_accuracy_interval=[.5+.5*v for v in interval],
                confidence=float(monte_carlo_confidence),uncertainty_method='fixed-N bounded-variable Hoeffding',
                untruncated_half_width=half,summand_range=[0.,1.],
                variance_estimate=max(0.,square_total/samples-estimate*estimate),
                interval_describes_sampling_error_not_model_fit_uncertainty=True,
                finite_log_mask_weights_underflowing_in_float64_sampler=underflows,
                all_finite_log_mask_weights_retained_in_likelihood=True,
                finite_precision_sampler_not_interval_arithmetic_certified=True)
        return dict(exact=exact,monte_carlo=mc,max_exact_outcomes=budget,
            exact_required_log2_outcomes=k,exact_required_outcomes=(1<<k) if k<63 else None,
            no_geometry_or_observation_truncation=True,reference_independent_of_observed_hit_pattern=True,
            numerical_zero_is_not_an_automatic_equivalence_claim=True)


def compare_native_mixtures(unit, mixture_a, mixture_b, *, max_exact_outcomes=4096,
                            monte_carlo_samples=0,seed=0,monte_carlo_confidence=.95,
                            maximum_working_bytes=64*1024**2,include_outcome_table=False):
    comparison=NativeMixtureComparison(unit,mixture_a,mixture_b,maximum_working_bytes=maximum_working_bytes)
    return comparison.metadata()|comparison.observed()|comparison.reference(
        max_exact_outcomes=max_exact_outcomes,monte_carlo_samples=monte_carlo_samples,
        seed=seed,monte_carlo_confidence=monte_carlo_confidence,include_outcome_table=include_outcome_table)
