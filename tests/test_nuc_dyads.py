"""Observed dyads use nucleosome length, independently of coarse size resolution."""
from dataclasses import replace

import numpy as np
import pytest

from nuc_synth import complement_msps, flat_calibration
from fiberhmm.nuc.disrupted import MISSING, missing_nucleosomes, rule_runs
from fiberhmm.nuc.molecule import footprint_calls, molecule_outputs
from fiberhmm.nuc.params import DEFAULT_PARAMS
from fiberhmm.nuc.records import make_molecule
from fiberhmm.nuc.segment import REGULAR, Segment, build_steps, emissions, state_space
from fiberhmm.nuc.sizes import CANT_TELL, EDGE_TRUNCATED, MONO, MULTI2
from fiberhmm.nuc.spacing import eligible_dyads, footprint_dyad_mask


def sparse_array():
    # Two informative sites per 20 bp edge fail the unchanged size-class gate.
    length = 5000
    nucs = [(0, 147)] + [(300 + k * 190, 447 + k * 190) for k in range(20)]
    tfs = [(4500, 4647)]  # nucleosome-length TF calls still supply no dyads
    informative = np.arange(0, length, 10)
    occupied = np.zeros(length, bool)
    for a, b in nucs + tfs:
        occupied[a:b] = True
    return make_molecule(length, nucs=nucs, tfs=tfs,
                         msps=complement_msps(length, nucs + tfs),
                         informative=informative, hits=informative[~occupied[informative]])


def test_cant_tell_nucleosomes_supply_dyads_but_tf_and_truncated_do_not():
    m = sparse_array()
    cal = flat_calibration()
    fp = footprint_calls(m, cal=cal)
    assert (fp['coarse'][~fp['truncated']] == CANT_TELL).all()
    assert fp['coarse'][0] == EDGE_TRUNCATED
    mask = footprint_dyad_mask(fp, cal.nrl_pooled)
    assert mask.sum() == 20
    assert not mask[fp['is_tf']].any() and not mask[fp['truncated']].any()
    d = eligible_dyads(m, cal.nrl_pooled, coarse_codes=np.full(len(m.nucs), CANT_TELL))
    assert len(d.centres) == 20 and len(d.gaps) == 19
    r = molecule_outputs(m, cal=cal)
    assert r.scalars['n_dyads'] == 20 and r.scalars['nrl'] == 190
    assert not r.flags['low_information']
    assert rule_runs(m, fp, cal.nrl_pooled)[1]
    assert any(s.state == REGULAR for s in r.segments)
    # Tightening only the dyad window removes spacing evidence, preserving all
    # per-footprint annotations and their confidence scores.
    narrow = replace(DEFAULT_PARAMS, mono_frac=(0.9, 1.25))
    other = molecule_outputs(m, narrow, cal)
    assert other.scalars['n_dyads'] == 0
    for key in ('coarse', 'posterior', 'named', 'z', 'n_left', 'n_right'):
        np.testing.assert_array_equal(r.footprints[key], other.footprints[key])


@pytest.mark.parametrize('nrl,lengths,expected', [
    (200, [89, 90, 250, 251], [False, True, True, False]),
    (None, [89, 90, 230, 231], [False, True, True, False]),
])
def test_dyad_window_boundaries_and_bootstrap_ignore_coarse_codes(nrl, lengths, expected):
    starts = np.arange(len(lengths)) * 400 + 100
    m = make_molecule(2500, nucs=list(zip(starts, starts + lengths)), hits=[10, 2490])
    d = eligible_dyads(m, nrl, coarse_codes=np.full(len(lengths), MONO))
    assert d.eligible_mask.tolist() == expected


def test_cant_tell_steps_are_observed_and_multimers_remain_virtual():
    starts = np.array([100, 290, 480, 900])
    ends = starts + [147, 147, 337, 147]
    codes = np.array([CANT_TELL, MONO, MULTI2, CANT_TELL])
    steps = build_steps(starts, ends, codes, np.ones(4), np.zeros(4, bool),
                        np.full(4, .5), np.zeros(4, bool), (0, 1300), 190,
                        lambda lam: 147, None)
    assert steps[0].observed_left and steps[0].observed_right
    assert steps[0].gap == 190
    assert sum(st.observed_left and st.observed_right for st in steps) == 1
    # The existing emission gate excludes all gaps touching virtual dyads.
    _, parts = emissions(steps, state_space(), flat_calibration())
    assert np.any(parts['gR'][0] != 0)
    assert np.all(parts['gR'][1:] == 0)
    assert np.all(parts['gI'][1:] == 0)



def test_missing_event_accepts_cant_tell_dyad_boundaries():
    m = sparse_array()
    nucs = np.delete(m.nucs, 10, axis=0)
    occupied = np.zeros(m.length, bool)
    for a, b in nucs:
        occupied[a:b] = True
    hits = m.informative & ~occupied
    m = make_molecule(m.length, nucs=nucs, informative=m.informative,
                      hits=hits, msps=complement_msps(m.length, nucs))
    cal = flat_calibration()
    fp = footprint_calls(m, cal=cal)
    seg = Segment(200, 4300, REGULAR, False, 1.0, 10.0, 190.0, 19, 18)
    d = eligible_dyads(m, cal.nrl_pooled)
    events = missing_nucleosomes(m, fp, [seg], d.centres, 190.0, cal)
    assert len(events) == 1 and events[0].kind == MISSING
