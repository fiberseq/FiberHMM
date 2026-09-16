"""Vectorized predictive kernel: reference interval when untilted, finite bounds always."""
import numpy as np

from fiberhmm.inference.consensus.measurement_distribution import predictive_count_record
from fiberhmm.inference.consensus.vectorized_predictive import vectorized_predictive


def _request(k=24, projections=6, replicates=2047, seed=7):
    rs = np.random.RandomState(seed)
    pa = np.full(k, .3); pp = np.full(k, .05)
    starts = np.arange(projections)*2; ends = starts+10
    log_penalty = -rs.uniform(0., 3., projections)
    q = rs.dirichlet(np.ones(projections)); cdf = np.cumsum(q); cdf[-1] = 1.
    return (pa, pp, starts, ends, log_penalty, cdf, 1.5, replicates, seed)


def test_untilted_kernel_reports_the_reference_interval():
    request = _request()
    tail, lower, upper, replicates, info = vectorized_predictive(*request)
    assert replicates == request[-2]
    reference = predictive_count_record({}, info['raw_events'], replicates)
    assert tail == reference['predictive_tail']
    assert [lower, upper] == reference['predictive_tail_interval']
    assert 0. <= lower <= upper <= 1. and info['weighted_events'] == info['raw_events']
    again = vectorized_predictive(*request)
    assert again[0] == tail   # counter-based draws are reproducible from the seed


def test_bounds_stay_valid_for_tilts_and_degenerate_mass():
    request = _request()
    for tilt in (.3, 'threshold:4'):
        tail, lower, upper, _, info = vectorized_predictive(*request, tilt=tilt)
        assert np.isfinite([tail, lower, upper]).all() and 0. <= lower <= tail <= upper <= 1.
        assert info['tilt'] == tilt
    empty = list(request); empty[5] = np.zeros_like(request[5])
    for tilt in (0., .3):
        tail, lower, upper, _, _ = vectorized_predictive(*empty, tilt=tilt)
        assert np.isfinite([tail, lower, upper]).all() and 0. <= lower <= upper <= 1.
