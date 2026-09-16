"""Independent native-family validation of auxiliary recall/split proposals.

The fitted native catalog supplies identities and the original source-call
indices. It never receives recalled calls as training data. Source geometry is
refit from native observations after excluding the recipient's entire grouped
fold and restricting the requested biochemical donor cohort. No extra CR bp
allowance enters this validator. This remains caller-conditioned inference,
not out-of-fold proposal discovery or an occupancy/FDR calibration.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np

from .artifacts import digest
from .measurement_distribution import fit_native_distribution
from .native_cross import (
    boundary_grid, frozen_model_geometry, _recipient_observations, transferred_call,
)


def grouped_fold(group):
    return int(hashlib.sha256(('native-auxiliary-fold-v1|' + str(group)).encode()).hexdigest()[:12], 16) % 2


def donor_mask(strands, recipient_strand, mode):
    strands = np.asarray(strands)
    if mode == 'opposite_strand':
        if recipient_strand not in ('CT', 'GA'):
            return np.zeros(len(strands), bool)
        return strands == ('GA' if recipient_strand == 'CT' else 'CT')
    if mode == 'same_strand':
        return strands == recipient_strand
    if mode == 'pooled':
        return np.ones(len(strands), bool)
    raise ValueError(f'Unknown auxiliary source mode {mode!r}')


class NativeAuxiliaryShape:
    """Cached donor-mode/group-excluded fits on a frozen native geometry grid."""

    def __init__(self, stratum, native, *, mode='opposite_strand',
                 maximum_matrix_bytes=512*1024**2, max_iterations=100,
                 predictive_replicates=4095, reference_percent=99.9,
                 minimum_source_units=3, progress=None):
        self.stratum, self.native, self.mode = stratum, native, mode
        self.units = stratum['units']
        for call in native['calls']:
            index = call['unit_index']
            if not 0 <= index < len(self.units) or self.units[index]['unit_id'] != call['unit_id']:
                raise ValueError('Frozen native call indices do not match the supplied evidence-unit identities')
        self.models = {m['family']: m for m in native['family_models']
                       if m.get('status') == 'fitted' and 'fold_models' in m}
        self.maximum_matrix_bytes = maximum_matrix_bytes
        self.max_iterations = max_iterations
        self.replicates, self.reference_percent = predictive_replicates, reference_percent
        self.minimum_source_units = minimum_source_units
        self.progress = progress or (lambda *_: None)
        self.frozen, self.fits = {}, {}

    def fit(self, family, recipient_strand, held_fold):
        key = (family, recipient_strand, int(held_fold), self.mode)
        if key in self.fits:
            return self.fits[key]
        if family not in self.models:
            return dict(status='native_family_not_fitted', training_evidence_groups=[])
        model = self.models[family]
        if family not in self.frozen:
            self.frozen[family] = frozen_model_geometry(model, self.native['calls'], self.units)
        original = self.frozen[family]
        grid = original['grid']
        candidates = [self.native['calls'][i] for i in model['source_call_indices']]
        permitted = donor_mask([c['strand'] for c in candidates], recipient_strand, self.mode)
        # Native source indices already have one source per evidence group;
        # retain the assertion at this interface rather than assuming it.
        sources = [c for c, ok in zip(candidates, permitted) if ok
                   and grouped_fold(c.get('evidence_group_id', c['unit_id'])) != held_fold]
        groups = [str(c.get('evidence_group_id', c['unit_id'])) for c in sources]
        if len(groups) != len(set(groups)):
            raise ValueError('A physical evidence group cannot train an auxiliary family twice')
        estimate = len(grid['starts']) * (len(sources) * 17 + 160)
        if estimate > self.maximum_matrix_bytes:
            raise MemoryError(f'{family}: independent auxiliary shape fit needs {estimate} bytes; no source sampling')
        observations = [(c, _recipient_observations(self.units[c['unit_index']], c, grid)) for c in sources]
        observations = [(c, o) for c, o in observations if o['observed'].any() and o['allowed'].any()]
        groups = [str(c.get('evidence_group_id', c['unit_id'])) for c, _ in observations]
        base = dict(family=family, held_fold=int(held_fold), source_mode=self.mode,
                    recipient_strand=recipient_strand, training_evidence_groups=groups,
                    training_unit_ids=[c['unit_id'] for c, _ in observations],
                    training_digest=digest([family, self.mode, held_fold, sorted(groups)]),
                    source_units=len(observations), extra_boundary_tolerance_bp=0,
                    source_calls_added=False, native_catalog_refit=False,
                    proposal_discovery_out_of_fold=False)
        if len(observations) < self.minimum_source_units:
            value = base | dict(status='insufficient_independent_native_shape_sources')
        else:
            self.progress('auxiliary_shape', f'{family}: {self.mode} {recipient_strand} fold {held_fold}, {len(observations)} native sources')
            fitted = fit_native_distribution(
                np.asarray([o['likelihood'] for _, o in observations]),
                np.asarray([o['allowed'] for _, o in observations]),
                grid['coordinates'], grid['areas'], reference=model['reference_interval'],
                max_iterations=self.max_iterations)
            # This private frozen object holds ONLY the excluded-fold refit.
            # No full-data parameters initialize or score a recipient.
            descriptor = deepcopy(model)
            descriptor.update(fitted_distribution_center=fitted['center'].tolist(),
                fitted_distribution_covariance=fitted['covariance'].tolist(),
                fold_models={'full': dict(parameters=fitted['parameters'].tolist(),
                    parameter_reference=model['reference_interval'], parameter_coordinate_scale_bp=10.)})
            value = base | dict(status='fitted' if fitted['converged'] else 'native_shape_fit_not_converged',
                fit_diagnostics={k: fitted[k] for k in ('objective', 'converged', 'iterations', 'message')},
                frozen=dict(model=descriptor, grid=grid))
        self.fits[key] = value
        return value

    def validate(self, unit, interval, family, *, ordinal=0):
        group = str(unit.get('fold_group_id', unit['unit_id']))
        held_fold = grouped_fold(group)
        fitted = self.fit(family, unit['strand'], held_fold)
        record = {k: v for k, v in fitted.items() if k != 'frozen'}
        record.update(unit_id=unit['unit_id'], interval=list(map(int, interval)),
                      reference_percent=self.reference_percent,
                      evidence_group_excluded=group not in fitted['training_evidence_groups'],
                      evidence_semantics='independently fitted native latent-distribution shape; not protection existence or FDR')
        if not record['evidence_group_excluded']:
            raise AssertionError('Recipient evidence group leaked into auxiliary shape training')
        if fitted['status'] != 'fitted':
            return record | dict(accepted=False)
        positions = np.union1d(fitted['frozen']['grid']['positions'], unit['positions'])
        domain = fitted['frozen']['grid']['domain']
        in_domain = positions[(positions >= domain[0]) & (positions < domain[1])]
        estimate = (len(in_domain)+1)*len(in_domain)//2 * 160
        if estimate > self.maximum_matrix_bytes:
            raise MemoryError(f'{family}: auxiliary recipient shape grid needs {estimate} bytes; no opportunity thinning')
        grid = boundary_grid(positions, domain)
        call = dict(unit_id=unit['unit_id'], ordinal=ordinal, strand=unit['strand'],
                    start=int(interval[0]), end=int(interval[1]))
        score = transferred_call(fitted['frozen'], grid, unit, call, floor_bp=0,
                                 replicates=self.replicates)
        # The exact zero-loss case needs no Monte Carlo reference. Otherwise
        # an under-resolved simulation cannot make every shape compatible merely
        # because its zero-exceedance confidence interval is too broad.
        if (score['status'] == 'scored' and score['native_loss'] > 1e-10 and
                score['predictive_tail_interval'][0] <= 1.-self.reference_percent/100. and
                3.841458820694124/(self.replicates+3.841458820694124) >= 1.-self.reference_percent/100.):
            return record | dict(status='native_shape_reference_underresolved', accepted=False,
                                 native_shape_evidence=score)
        accepted = (score['status'] == 'scored'
                    and score['predictive_tail_interval'][1] >= 1.-self.reference_percent/100.)
        return record | dict(status=score['status'], accepted=bool(accepted), native_shape_evidence=score)

    def fit_manifest(self):
        return [{k: v for k, v in fit.items() if k != 'frozen'} for _, fit in sorted(self.fits.items())]
