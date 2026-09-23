"""Decision stopping ends a predictive run early without changing any gate decision."""
import numpy as np
import pytest

from fiberhmm.inference.consensus.measurement_distribution import (
    _predictive_exceedance_run, _predictive_exceedances, complete_predictive_reference,
    decision_stop_count, predictive_count_record, reset_predictive_decision_stop,
    set_predictive_decision_stop)
from fiberhmm.inference.consensus.parameters import parse_options


def _requests(number, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(number):
        n = int(rng.integers(3, 30)); k = int(rng.integers(2, 10))
        starts = np.sort(rng.integers(0, n-1, k)); ends = np.minimum(n, starts+rng.integers(1, 8, k))
        cdf = np.cumsum(rng.dirichlet(np.ones(k))); cdf[-1] = 1.
        yield (rng.uniform(.2, .8, n), rng.uniform(.01, .2, n), starts, ends, -rng.exponential(1., k),
               cdf, float(rng.exponential(2.)), 4095, int(rng.integers(0, 2**32)))


def test_stop_counts_match_the_wilson_gate():
    assert decision_stop_count(.001, 4095) == 1
    assert decision_stop_count(.01, 4095) == 29
    assert decision_stop_count(.05, 4095) == 178


@pytest.mark.parametrize('stop', [1, 29])
def test_stopped_run_is_a_prefix_of_the_full_experiment(stop):
    for request in _requests(60):
        full = _predictive_exceedances(*request)
        count, used = _predictive_exceedance_run(*request, stop)
        if used < request[-2]:
            assert count == stop <= full
        else:
            assert count == full and full <= stop
        assert _predictive_exceedance_run(*request, 0) == (full, request[-2])


def test_stopped_record_keeps_every_decision_and_bounds_the_full_record():
    stopped = 0
    for request in _requests(80, seed=1):
        full = predictive_count_record({}, _predictive_exceedances(*request), request[-2])
        token = set_predictive_decision_stop(1)
        try:
            record = complete_predictive_reference(dict(_native_predictive_request=request))
        finally:
            reset_predictive_decision_stop(token)
        assert (record['predictive_tail_interval'][1] >= .001) == (full['predictive_tail_interval'][1] >= .001)
        assert record['predictive_tail_interval'][1] <= full['predictive_tail_interval'][1]
        assert record['predictive_tail'] <= full['predictive_tail']
        if record.get('predictive_stopping'):
            stopped += 1
            assert record['simulations'] < record['planned_simulations'] == 4095
            assert record['exact_for_tail_cuts_at_most'] >= .001
        else:
            assert record == full
    assert stopped


def test_default_runs_every_draw():
    request = next(_requests(1, seed=2))
    record = complete_predictive_reference(dict(_native_predictive_request=request))
    assert record['simulations'] == 4095 and 'predictive_stopping' not in record


def test_stopping_is_staged_only():
    with pytest.raises(ValueError, match='staged'):
        parse_options(dict(compute=dict(predictive_stopping='decision')))
