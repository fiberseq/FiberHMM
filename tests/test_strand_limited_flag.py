"""Per-strand core informativeness flags for DAF recurrent states."""
import numpy as np

from fiberhmm.inference.consensus.harmonized_families.presentation import (
    core_informativeness, flag_strand_limits, presentation_context)


def _unit(uid, strand, positions, pa=.6, pp=.05):
    n = len(positions)
    return dict(unit_id=uid, strand=strand, positions=list(positions), p_accessible=[pa]*n,
                p_protected=[pp]*n, reference_start=0, reference_end=200, raw_nuc_intervals=[],
                msp_intervals=[[0, 200]], aligned_blocks=[[0, 200]])


def test_core_ceiling_is_the_all_miss_protection_llr():
    source = dict(dataset_id='d', chemistry='ddda', model_manifest=dict(native_minimum_llr=5.),
                  units=[_unit('a', 'CT', [10, 20, 30, 40, 90]), _unit('b', 'CT', [10, 35, 90])])
    context = presentation_context([source])
    step = np.log1p(-.05)-np.log1p(-.6)
    got = core_informativeness(context, {'d::a', 'd::b'}, 15, 45)
    assert got['core_opportunities'] == 2.
    assert np.isclose(got['core_protection_ceiling_llr'], 2*step)
    assert context['native_floors'] == {'d': 5.}


def test_hia5_sources_are_not_scored():
    source = dict(dataset_id='h', chemistry='hia5-pacbio', units=[_unit('a', 'pooled', [1, 2])])
    assert presentation_context([source])['lattices'] == {}


def test_only_the_asymmetric_strand_is_limited():
    by_strand = dict(CT=dict(core_protection_ceiling_llr=1.5), GA=dict(core_protection_ceiling_llr=9.))
    verdict = flag_strand_limits(by_strand, 5.)
    assert verdict['trusted_strand'] == 'GA' and verdict['core_resolution'] == 'resolved'
    assert by_strand['CT']['strand_limited'] and by_strand['CT']['core_below_native_floor']
    assert not by_strand['GA']['strand_limited'] and not by_strand['GA']['core_below_native_floor']
    both = dict(CT=dict(core_protection_ceiling_llr=3.), GA=dict(core_protection_ceiling_llr=4.))
    verdict = flag_strand_limits(both, 5.)
    assert verdict['trusted_strand'] == 'both' and verdict['core_resolution'] == 'below_native_floor'
    assert not any(v['strand_limited'] for v in both.values())
    assert all(v['core_below_native_floor'] for v in both.values())


def test_weak_asymmetric_core_trusts_the_better_strand_but_says_unresolved():
    verdict = flag_strand_limits(dict(CT=dict(core_protection_ceiling_llr=.8),
                                      GA=dict(core_protection_ceiling_llr=3.)), 5.)
    assert verdict['trusted_strand'] == 'GA' and verdict['core_resolution'] == 'below_native_floor'
