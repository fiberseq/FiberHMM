"""Exactness, lattice, and separation regressions for TF configuration decoding."""
from itertools import product

import numpy as np
import pytest

from fiberhmm.inference.tf_recaller import (
    N_CTX, UNMETH_OFFSET, _call_tf_configurations_numba,
    call_tfs_in_interval, call_single_excursion_intervals,
)


def run_steps(steps, penalty=7., min_opps=3, positions=None, decoder="multi_interval"):
    steps = np.asarray(steps, dtype=float)
    if positions is None:
        positions = np.arange(len(steps))
    positions = np.asarray(positions)
    obs = np.full(int(positions[-1])+1 if len(positions) else 0, N_CTX, dtype=np.int32)
    hit, miss = np.full(N_CTX, -1.), np.full(N_CTX, 1.)
    for i,(p,z) in enumerate(zip(positions,steps)):
        assert i < N_CTX
        if z <= 0:
            obs[p], hit[i] = i, z
        else:
            obs[p], miss[i] = UNMETH_OFFSET+i, z
    return call_tfs_in_interval(obs,0,len(obs),hit,miss,penalty,min_opps,decoder=decoder)


def coords(calls):
    return [(c.start,c.start+c.length) for c in calls]


def objective(calls, penalty):
    return sum(c.llr-penalty for c in calls)


def exhaustive_oracle(steps, penalty, min_opps):
    """Independent enumeration of every protected/accessibile configuration.

    The decoder's positive-endpoint and minimum-opportunity domain is explicit;
    contiguous protected positions form one interval. The optimal configuration
    never needs adjacent intervals with no excluded opportunity between them.
    """
    best = (0.,0,0)
    for states in product([False,True], repeat=len(steps)):
        intervals, i = [], 0
        while i < len(states):
            if not states[i]:
                i += 1
                continue
            j = i + 1
            while j < len(states) and states[j]:
                j += 1
            intervals.append((i,j))
            i = j
        if any(b-a < min_opps or steps[a] <= 0 or steps[b-1] <= 0 for a,b in intervals):
            continue
        score = sum(sum(steps[a:b])-penalty for a,b in intervals)
        key = (score,-len(intervals),-sum(b-a for a,b in intervals))
        if key > best:
            best = key
    return best


def test_second_supported_peak_is_not_swallowed_by_first():
    z = [3.]*5 + [-4.]*3 + [3.]*3
    old = run_steps(z, decoder="single_excursion")
    new = run_steps(z)
    assert coords(old) == [(0,5)]
    assert coords(new) == [(0,5),(8,11)]
    assert [c.llr for c in new] == pytest.approx([15.,9.])
    assert objective(new,7) == pytest.approx(10.)
    assert objective(new,7) > objective(old,7)


def test_same_observation_can_be_one_or_two_intervals_by_gap_evidence():
    joined = run_steps([3.]*5 + [-2.] + [3.]*3)
    separated = run_steps([3.]*5 + [-8.] + [3.]*3)
    assert coords(joined) == [(0,9)]
    assert coords(separated) == [(0,5),(6,9)]


def test_subthreshold_state_is_not_promoted_by_a_neighbor():
    calls = run_steps([3.]*5 + [-4.]*3 + [2.]*3)
    assert coords(calls) == [(0,5)]


def test_napa_ga_actual_context_steps_recover_skipped_right_core():
    # Frozen unit_00eb268cc8856f04003ae135: first footprint at 077..110,
    # valley at 133, and four unmodified GA opportunities at 136..153.
    positions = [77,78,85,87,92,94,97,98,106,108,109,114,119,122,124,128,133,136,137,142,152]
    steps = [2.91297979648283,2.23577578924894,1.98230307943754,2.90785234682954,
             -3.03768398256702,2.057923,1.609341,-3.002353,2.830006,1.669707,3.493144,
             -1.762384,-3.187399,2.423096,-3.549168,-3.610195,-3.268096,
             1.863654,3.187587,1.872054,2.386687]
    old = run_steps(steps,positions=positions,decoder="single_excursion")
    new = run_steps(steps,positions=positions)
    assert coords(old) == [(77,110)]
    assert any(a <= 136 and b >= 153 for a,b in coords(new))
    assert objective(new,7) > objective(old,7)


@pytest.mark.parametrize("min_opps",[1,2,3,4])
@pytest.mark.parametrize("penalty",[0.,1.,3.])
def test_matches_exhaustive_configuration_enumeration(min_opps,penalty):
    rng = np.random.default_rng(8821+min_opps)
    for _ in range(35):
        steps = rng.integers(-5,6,8).astype(float)
        calls = run_steps(steps,penalty=penalty,min_opps=min_opps)
        expected = exhaustive_oracle(steps,penalty,min_opps)
        assert objective(calls,penalty) == pytest.approx(expected[0],abs=1e-9)
        assert len(calls) == -expected[1]
        assert sum(c.length for c in calls) == -expected[2]
        assert all(c.n_opps >= min_opps for c in calls)
        assert all(a[1] <= b[0] for a,b in zip(coords(calls),coords(calls)[1:]))


def test_reversal_preserves_optimal_calls_when_geometry_has_no_ties():
    rng = np.random.default_rng(7781)
    for _ in range(80):
        steps = rng.normal(.4,3.0,40)
        a = run_steps(steps,penalty=3)
        b = run_steps(steps[::-1],penalty=3)
        mirrored = sorted((len(steps)-e,len(steps)-s) for s,e in coords(b))
        assert coords(a) == mirrored
        assert objective(a,3) == pytest.approx(objective(b,3))


def test_non_target_padding_does_not_add_opportunities():
    assert run_steps([4.,4.],positions=[0,1000]) == []
    a = run_steps([3.,3.,3.],positions=[5,15,25])
    assert coords(a) == [(5,26)]
    assert a[0].n_opps == 3


def test_no_fixed_call_count_cap():
    calls = run_steps(([3.,3.,3.,-12.] * 700), min_opps=3)
    assert len(calls) == 700


def test_penalty_equality_tie_prefers_empty_configuration():
    assert run_steps([2.,2.,3.], penalty=7) == []


def test_nucleosome_consumers_keep_single_excursion_behavior():
    from fiberhmm.inference import nuc_recaller
    assert nuc_recaller.call_tfs_in_interval is call_single_excursion_intervals
    obs = np.array([UNMETH_OFFSET]*5 + [0]*3 + [UNMETH_OFFSET]*3)
    hit,miss = np.full(N_CTX,-4.),np.full(N_CTX,3.)
    old = nuc_recaller.call_tfs_in_interval(obs,0,len(obs),hit,miss,7.,3)
    new = call_tfs_in_interval(obs,0,len(obs),hit,miss,7.,3)
    assert coords(old) == [(0,5)]
    assert coords(new) == [(0,5),(8,11)]


def test_compiled_kernel_matches_python_kernel():
    if not hasattr(_call_tf_configurations_numba,"py_func"):
        pytest.skip("Numba not installed")
    obs = np.array([UNMETH_OFFSET]*5 + [0]*3 + [UNMETH_OFFSET]*3,dtype=np.int32)
    hit,miss = np.full(N_CTX,-4.),np.full(N_CTX,3.)
    args = (obs,0,len(obs),hit,miss,7.,3,False,np.zeros(1,dtype=bool),hit,miss)
    a = _call_tf_configurations_numba(*args)
    b = _call_tf_configurations_numba.py_func(*args)
    for x,y in zip(a,b):
        np.testing.assert_allclose(x,y)


@pytest.mark.parametrize("options",[{"decoder":"unknown"},{"min_opps":0},{"min_llr":-1},{"min_llr":float("nan")}])
def test_invalid_configuration_options_rejected(options):
    args = dict(min_llr=7.,min_opps=3)
    args.update(options)
    with pytest.raises(ValueError):
        call_tfs_in_interval(np.array([UNMETH_OFFSET]*4),0,4,np.zeros(N_CTX),np.ones(N_CTX),**args)
