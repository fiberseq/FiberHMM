"""Class support (q0): the assigned class's share of the scored class evidence."""
import math

import pytest

from fiberhmm.inference.consensus.native_presentation import class_shares, classify_proposal, q0_byte


def score(family, loss, tail=.5, optimum=0., distance=0, status='scored'):
    return dict(family=family, status=status, floor_adjusted_loss=loss, recipient_optimum=optimum,
                predictive_tail_interval=[tail/2, tail], geometry_distance_sq=distance)


def proposal(*scores):
    return dict(candidate_evidence=list(scores), unresolved_family='unresolved')


def test_single_candidate_is_255_and_unresolved_is_0():
    assert classify_proposal(proposal(score('a', 2.)), 99.9)['q0'] == 255
    unresolved = classify_proposal(proposal(score('a', 2., tail=1e-6)), 99.9)
    assert unresolved['classification_status'] == 'provisional_unresolved' and unresolved['q0'] == 0
    assert unresolved['member_q0'] == {}


def test_two_equal_candidates_are_half():
    out = classify_proposal(proposal(score('a', 1.), score('b', 1., distance=5)), 99.9)
    assert out['family'] == 'a' and out['q0'] in (127, 128)
    assert out['member_q0'] == {'a': out['q0'], 'b': out['q0']}


@pytest.mark.parametrize('reference', [50., 90., 99.9])
def test_q0_uses_every_scored_candidate_and_ignores_stringency(reference):
    # 'b' is predictive-rejected at strict references but still counts as evidence.
    p = proposal(score('a', 0.), score('b', math.log(3.), tail=.02, distance=9),
                 score('c', 0., status='core_contradicted'))
    out = classify_proposal(p, reference)
    assert out['family'] == 'a' and out['q0'] == round(255*.75)


def test_shares_use_a_common_optimum_across_grids():
    # Same constrained maximum (optimum - loss) on different grids -> equal shares.
    shares = class_shares([score('a', 1., optimum=10.), score('b', 3., optimum=12.)])
    assert shares['a'] == pytest.approx(.5) and shares['b'] == pytest.approx(.5)
    # Without optima everywhere, losses alone are compared.
    shares = class_shares([score('a', 1., optimum=None), score('b', 3., optimum=12.)])
    assert shares['a'] > .8
    assert q0_byte(0.) == 0 and q0_byte(1.) == 255
