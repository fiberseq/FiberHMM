"""Association callbacks and dataset calibration use synthetic observations."""
import json
from dataclasses import replace

import numpy as np
import pytest

from fiberhmm.assoc import AssociationCancelled
from fiberhmm.assoc.simulate import simulate_m1, simulate_m2, simulate_m3, simulate_nuc


def _analysis(mode, **callbacks):
    if mode == 'M1':
        from fiberhmm.assoc import cooccurrence
        s = simulate_m1(7, n_a=30, n_b=45, chrom_len=200000)
        return cooccurrence(s['a'], s['b'], s['molecules'], n_boot=150, per_pair=False, **callbacks)
    if mode == 'M2':
        from fiberhmm.assoc.condition import condition
        s = simulate_m2(7, n_anchors=12)
        return condition(s['app'], s['loci'], s['metric'], n_perm=250, n_role=250, n_boot=150, **callbacks)
    if mode in ('M3', 'edges'):
        from fiberhmm.assoc import rate_ratio, edge_association
        s = simulate_m3(7, n_molecules=15)
        if mode == 'edges':
            return edge_association(s, n_shift=150, **callbacks)
        return rate_ratio(s, n_shift=150, n_boot=150, **callbacks)
    if mode in ('M4', 'contrast'):
        from fiberhmm.assoc.overlap import simulate_overlap, overlap_enrichment, overlap_contrast
        s = simulate_overlap(7, n_a=30, n_b=100, n_groups=2)
        if mode == 'contrast':
            return overlap_contrast(s['a'], s['b_by_group'], s['territory'], s['strata'],
                                    n_relocations=150, n_boot=150, **callbacks)
        return overlap_enrichment(s['a'], s['b'], s['territory'], s['strata'], n_relocations=150, **callbacks)
    from fiberhmm.assoc import condition_stats, association_stats
    from nuc_synth import flat_calibration
    s = simulate_nuc(7, n_anchors=4, coverage=8)
    if mode == 'nuc_condition':
        return condition_stats(s['molecules'], s['anchors'], backgrounds=None, n_perm=250, n_role=0,
                               n_boot=0, frames=('anchor',), **callbacks)
    return association_stats(s['molecules'], cal=flat_calibration(), n_shift=150, n_boot=150, **callbacks)


@pytest.mark.parametrize('mode', ['M1', 'M2', 'M3', 'edges', 'M4', 'contrast', 'nuc_condition', 'nuc_rate'])
def test_cancelled_at_entry_and_during_analysis(mode):
    with pytest.raises(AssociationCancelled):
        _analysis(mode, cancel=lambda: True)
    events = []
    stop = [False]
    def progress(stage, done, total):
        events.append((stage, done, total))
        if stage in ('bootstrap', 'permutations', 'shifts', 'relocations', 'molecules'):
            stop[0] = True
    with pytest.raises(AssociationCancelled):
        _analysis(mode, cancel=lambda: stop[0], progress=progress)
    assert events


def test_callbacks_preserve_statistics_and_do_not_leak_after_cancellation():
    expected = _analysis('M2')
    events = []
    got = _analysis('M2', cancel=lambda: False, progress=lambda *event: events.append(event))
    assert json.dumps(got, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert {'permutations', 'roles', 'bootstrap'} <= {e[0] for e in events}
    # Context-local callbacks must be reset on both return and exceptions.
    with pytest.raises(AssociationCancelled):
        _analysis('M2', cancel=lambda: True)
    assert json.dumps(_analysis('M2'), sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_pooled_counts_sum_observations():
    from fiberhmm.assoc import FeatureStates, MoleculeTable, cooccurrence
    a = FeatureStates('a.x', ['a'], ['chrS'], [0], [1], [0], [0]*4, [0,1,2,3], [1,1,0,0])
    b = FeatureStates('b.x', ['b'], ['chrS'], [2], [3], [2], [0]*4, [0,1,2,3], [1,0,1,0])
    result = cooccurrence(a, b, MoleculeTable(np.array(['a','b','c','d'])), n_boot=0,
                          min_pool_molecules=1, effect=False)
    assert [result['pooled'][k] for k in ('a','b','c','d')] == [1,1,1,1]


def test_condition_stats_dataset_period_prior_and_size_calibration():
    from fiberhmm.assoc.nucleosome import anchor_appearances, condition_stats
    from fiberhmm.assoc.condition import LocusTable
    from fiberhmm.nuc.density import BetaPrior
    from nuc_synth import flat_calibration
    sim = simulate_nuc(12, n_anchors=8, coverage=12, n_datasets=2)
    cals = {'d0': flat_calibration(), 'd1': replace(flat_calibration(), nrl_pooled=210., prior=BetaPrior(2., 5.))}
    label = lambda m: m.meta['dataset']
    loci = LocusTable.from_sets(sim['anchors'], [], [[i, []] for i in range(len(sim['anchors']))])
    together, _, _ = anchor_appearances(sim['molecules'], loci, labels={'dataset': label},
                                       calibration_of=lambda m: cals[label(m)])
    for key, cal in cals.items():
        ms = [m for m in sim['molecules'] if label(m) == key]
        app, _, _ = anchor_appearances(ms, loci, calibration_of=lambda m: cal)
        select = together.labels['dataset'] == key
        np.testing.assert_allclose(together.stats[select], app.stats)
        np.testing.assert_allclose(together.density[select], app.density)
        np.testing.assert_array_equal(together.mask[select], app.mask)
    floor_cals = dict(cals, d0=replace(cals['d0'], lambda_floor=1.))
    floor_app, _, _ = anchor_appearances(sim['molecules'], loci, labels={'dataset': label},
                                       calibration_of=lambda m: floor_cals[label(m)])
    assert not np.any(floor_app.labels['dataset'] == 'd0')
    assert np.any(floor_app.labels['dataset'] == 'd1')
    out = condition_stats(sim['molecules'], sim['anchors'], cal_by_dataset=cals, labels={'dataset': label},
                          group='dataset', backgrounds=None, n_perm=0, n_role=0, n_boot=80, frames=['anchor'])
    assert out['calibration_by_dataset']['d1']['P'] == 210.
    assert out['calibration_by_dataset']['d0']['bins'] != out['calibration_by_dataset']['d1']['bins']
    assert 'groups' in out['frames']['anchor']['metrics']['R']['bins'][-1]
    with pytest.raises(ValueError, match='requires dataset_of'):
        condition_stats(sim['molecules'], sim['anchors'], cal_by_dataset=cals)
    with pytest.raises(KeyError):
        condition_stats(sim['molecules'], sim['anchors'], cal_by_dataset={}, dataset_of=label)
