"""Tie-set membership: opt-in, primary-preserving, byte-identical when off."""
import math

import pytest

from fiberhmm.inference.consensus.native_presentation import classify_proposal, member_families
from fiberhmm.inference.consensus.native_cross import _reference_members


def _score(family, loss, distance, upper=1.):
    return dict(family=family, status='scored', floor_adjusted_loss=loss, geometry_distance_sq=distance,
                predictive_tail_interval=[0., upper])


EVIDENCE = [_score('near_but_weak', 9.0, 0), _score('best', 5.0, 1), _score('edge', 5.0+math.log(10.), 9),
            _score('outside', 5.0+math.log(10.)+1e-6, 4), _score('rejected', 0.1, 0, upper=0.)]


def test_default_membership_is_the_primary_only_and_adds_no_fields():
    out = classify_proposal(dict(candidate_evidence=EVIDENCE, unresolved_family='U'), 99.9)
    assert out['family'] == 'near_but_weak'
    assert 'member_families' not in out and 'membership_loss_odds' not in out
    same = classify_proposal(dict(candidate_evidence=EVIDENCE, unresolved_family='U'), 99.9, 1.0)
    assert same == out


def test_tie_set_anchors_on_the_best_loss_and_keeps_the_primary_first():
    out = classify_proposal(dict(candidate_evidence=EVIDENCE, unresolved_family='U'), 99.9, 10.0)
    assert out['family'] == 'near_but_weak'
    assert out['member_families'] == ['near_but_weak', 'best', 'edge']
    assert out['membership_loss_odds'] == 10.0
    assert 'rejected' not in out['member_families'] and 'outside' not in out['member_families']


def test_unresolved_calls_have_no_membership():
    out = classify_proposal(dict(candidate_evidence=[_score('x', 1., 0, upper=0.)], unresolved_family='U'), 99.9, 10.0)
    assert out['family'] == 'U' and out['member_families'] == []
    assert member_families([], 10.0) == []


@pytest.mark.parametrize('bad', [0.5, 0, -1, True])
def test_membership_odds_below_one_rejected(bad):
    with pytest.raises(ValueError):
        classify_proposal(dict(candidate_evidence=EVIDENCE, unresolved_family='U'), 99.9, bad)


def test_reference_members_widen_cohorts_without_changing_primaries():
    result = dict(calls=[dict(unit_id='u1'), dict(unit_id='u2')],
                  call_family_evidence=[EVIDENCE, [_score('best', 2.0, 0)]])
    narrow = _reference_members(result, 99.9)
    wide = _reference_members(result, 99.9, 10.0)
    assert dict(narrow) == {'near_but_weak': [0], 'best': [1]}
    assert dict(wide) == {'near_but_weak': [0], 'best': [0, 1], 'edge': [0]}
