"""Cross-chemistry (XCR) resolution units of the staged engine: complete-link grouping."""
import numpy as np

from fiberhmm.inference.consensus.harmonized_families.presentation import resolution_units


def test_three_class_bridge_does_not_chain():
    """A~B and B~C are each closer than the resolution rule allows, A~C is not: A and C must not share a unit.
    (Single-link union-find merged all three; audit 2026-09-29, codex probe_integration.py.)"""
    shared = {f: dict(consensus_start=100.+i*10, consensus_end=140.+i*10) for i, f in enumerate('ABC')}
    context = dict(eligible={}, coverage={}, xcr_edge_sd_bp={'coarse': [10., 10.]})
    units, unit_of, provenance = resolution_units({'coarse': {'cr': {'catalog': [], 'records': []}}}, shared, set(shared), context)
    from scipy.stats import norm
    assert np.hypot(1., 1.) < 2*norm.ppf(.8) < np.hypot(2., 2.)      # the pair distances straddle the merge threshold
    assert unit_of['A'] != unit_of['C']
    assert unit_of['A'] == unit_of['B']
    assert len(units) == 2 and provenance['merged_pairs'] == 1
    assert sorted(len(u['members']) for u in units) == [1, 2]


def test_complete_link_still_merges_a_mutually_close_group():
    shared = {f: dict(consensus_start=100.+i*3, consensus_end=140.+i*3) for i, f in enumerate('ABC')}
    context = dict(eligible={}, coverage={}, xcr_edge_sd_bp={'coarse': [10., 10.]})
    units, unit_of, _ = resolution_units({'coarse': {'cr': {'catalog': [], 'records': []}}}, shared, set(shared), context)
    assert len(units) == 1 and unit_of['A'] == unit_of['B'] == unit_of['C']
