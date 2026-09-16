import numpy as np
from fiberhmm.inference.consensus.measurement_nomination import augment_catalog


def test_no_missing_source_hypothesis_cap_and_exact_lattice_aliases():
    calls = [dict(unit_id=f'u{i}', ordinal=0, start=a, end=b)
             for i, (a, b) in enumerate([(11, 29), (12, 28), (50, 70), (81, 89)])]
    result = dict(calls=calls, call_family_evidence=[[], [], [], []])
    initial = [dict(family='d:old', consensus_start=1, consensus_end=10)]
    out, ledger = augment_catalog(initial, result, np.arange(0, 101, 5), dataset_id='d', region=(0, 100))
    assert ledger['added_proposals'] == 3
    shared = next(p for p in out if p.get('nomination_source_units') == 2)
    assert len(shared['source_aliases']) == 2
    assert [shared['consensus_start'], shared['consensus_end']] in [[11, 29], [12, 28]]
    assert ledger['raw_footprints_added'] == 0


def test_compatible_call_does_not_create_redundant_residual_class():
    calls = [dict(unit_id='a', ordinal=0, start=10, end=30)]
    result = dict(calls=calls, call_family_evidence=[[dict(status='scored', predictive_tail_interval=[.1, .2])]])
    out, ledger = augment_catalog([], result, np.arange(0, 40, 2), dataset_id='d', region=(0, 40))
    assert not out and ledger['added_proposals'] == 0
