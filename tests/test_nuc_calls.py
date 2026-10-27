"""fiberhmm.nuc calls (NUC P-C): coarse size calls, named classes with an
absolute control, factorial segmentation, molecule outputs. Planted signals."""
from __future__ import annotations

import numpy as np
import pytest

from nuc_synth import array_molecule, complement_msps, flat_calibration

from fiberhmm.nuc.molecule import SCALARS, calls_digest, footprint_calls, molecule_outputs
from fiberhmm.nuc.params import NucParams
from fiberhmm.nuc.records import make_molecule
from fiberhmm.nuc.disrupted import MISSING, UNDERSIZED
from fiberhmm.nuc.segment import DISORDERED, IRREGULAR, LOWINFO, REGULAR, UNDETERMINED, state_space
from fiberhmm.nuc.sizes import (CANT_TELL, EDGE_TRUNCATED, LONG, MONO, MULTI2, MULTI3, SHORT,
                                AbsoluteClasses, ReferenceCurve, call_coarse, call_named)
from fiberhmm.nuc.thinning import calibrate_absolute

CURVE = ReferenceCurve([np.log(0.5)], [147.0], [15.0])


def _calls(lengths, sigma_loc=0.0, n_edge=10, trunc=None, supported=None):
    L = np.asarray(lengths, float)
    n = np.full(len(L), n_edge)
    t = np.zeros(len(L), bool) if trunc is None else np.asarray(trunc)
    return call_coarse(L, np.full(len(L), 0.5), CURVE, 190.0, 12.0,
                       lambda lam: np.full(np.shape(lam), sigma_loc), n, n, t,
                       lambda_supported=supported)


def test_coarse_regions_without_edge_noise():
    # mono 147+/-30; multi2 at 337 +/- 2*sqrt(15^2+12^2)=38; multi3 >= 527-2*sqrt(15^2+2*12^2)
    c = _calls([90, 140, 190, 250, 340, 440, 560, 900])
    assert c.code.tolist() == [SHORT, MONO, LONG, LONG, MULTI2, LONG, MULTI3, MULTI3]
    np.testing.assert_allclose(c.z[:2], [(90 - 147) / 15, (140 - 147) / 15])


def test_edge_noise_makes_boundary_footprints_cant_tell():
    c = _calls([118, 150, 176], sigma_loc=8.0)
    # 118 and 176 sit near the mono boundaries (117, 177): can't tell; 150 stays mono
    assert c.code.tolist() == [CANT_TELL, MONO, CANT_TELL]


def test_edge_sites_truncation_and_support():
    c = _calls([150, 150], n_edge=2)
    assert c.code.tolist() == [CANT_TELL, CANT_TELL]
    c = _calls([150, 150], trunc=[True, False])
    assert c.code.tolist() == [EDGE_TRUNCATED, MONO]
    c = _calls([150], supported=lambda lam: np.zeros(np.shape(lam), bool))
    assert c.code.tolist() == [CANT_TELL]


def test_named_classes_need_an_absolute_model():
    codes, post = call_named([147, 110], [0.5, 0.5], None, [10, 10], [10, 10])
    assert codes.tolist() == [0, 0]
    model = AbsoluteClasses(T={'hexasome': 110.0, 'octasome': 147.0}, b_log_lambda=[np.log(0.5)],
                            b=[0.0], sigma_b=[3.0], sigma_struct={'hexasome': 5.0, 'octasome': 5.0})
    codes, post = call_named([147, 110, 128], [0.5] * 3, model, [10] * 3, [10] * 3)
    assert codes.tolist() == [3, 2, 0]           # octasome, hexasome, can't tell


def test_calibrate_absolute_recovers_density_bias():
    rng = np.random.default_rng(1)
    base = flat_calibration()
    mols = []
    for i in range(60):
        dense = i % 2 == 0
        rate = 0.6 if dense else 0.15
        bias = 5 if dense else 35            # sparse marking: longer footprints
        nucs, pos = [], 300
        L = 6000
        while pos < L - 400:
            n = int(rng.normal(147 + bias, 4))
            nucs.append((pos, pos + n))
            pos += n + 40
        occ = np.zeros(L, bool)
        for s, e in nucs:
            occ[s:e] = True
        hits = np.flatnonzero(~occ & (rng.random(L) < rate))
        mols.append(make_molecule(L, nucs=nucs, msps=complement_msps(L, nucs), hits=hits, name=f'c{i}'))
    cal = calibrate_absolute(base, mols)
    assert cal.source == 'absolute' and cal.named_available
    b = dict(zip(np.round(np.exp(cal.absolute.b_log_lambda), 2), cal.absolute.b))
    lams = sorted(b)
    assert b[lams[0]] > 25 and b[lams[-1]] < 12
    rt = type(cal).from_dict(cal.to_dict())
    assert rt.absolute.T == cal.absolute.T


# --- segmentation ---------------------------------------------------------------------

def _segments(mol, cal=None):
    cal = cal or flat_calibration()
    return molecule_outputs(mol, cal=cal)


def test_segmentation_recovers_planted_arrays_disorder_and_undersized_runs():
    rng = np.random.default_rng(1)
    m = array_molecule([('reg', 30, 190), ('irr', 20, 0), ('reg', 25, 170), ('sub', 30, 60),
                        ('reg', 3, 190), ('irr', 10, 0)], rng)
    r = _segments(m)
    states = [(s.state, s.q_hat) for s in r.segments]
    assert states[0] == (REGULAR, 190.0)
    # the irregular block: disordered where its spacing evidence holds the
    # margin, undetermined where the neighbouring array's rule run trims it
    assert states[1][0] in (DISORDERED, UNDETERMINED)
    assert states[2] == (REGULAR, 170.0)
    und = [e for e in r.events if e.kind == UNDERSIZED]
    assert len(und) == 1 and und[0].end - und[0].start >= 800 and und[0].count >= 10
    # the 3-dyad regular stretch is below the minimum: never a regular segment there
    tail_start = und[0].end
    assert all(s.state != REGULAR for s in r.segments if s.start >= tail_start)
    # the undersized stretch has no dyads: no spacing evidence, so undetermined
    mid = (und[0].start + und[0].end) // 2
    assert [s.state for s in r.segments if s.start <= mid < s.end] == [UNDETERMINED]
    assert r.segments[0].posterior > 0.9 and r.segments[0].n_dyads >= 25
    # the decoder's change probability at the array's end (absent when the
    # rule moved the end inside the next decoder segment)
    if r.segments[0].source == 'hmm':
        assert 0.6 < r.segments[0].boundary[1] <= 1.0
    assert r.segments[0].source in ('hmm', 'hmm+rule')
    assert r.segments[0].regularity > 0.3
    assert 0.3 < r.scalars['reg_fraction'] < 0.8
    assert 0.0 < r.scalars['undersized_fraction'] < 0.5
    assert 0.0 < r.scalars['disordered_fraction'] + r.scalars['undetermined_fraction'] < 0.7


def test_minimum_regular_segment_is_five_dyads_and_800bp():
    rng = np.random.default_rng(2)
    m = array_molecule([('irr', 12, 0), ('reg', 4, 190), ('irr', 12, 0)], rng)
    assert all(s.state != REGULAR for s in _segments(m).segments)
    m = array_molecule([('irr', 12, 0), ('reg', 9, 190), ('irr', 12, 0)], rng)
    regs = [s for s in _segments(m).segments if s.state == REGULAR]
    assert len(regs) == 1 and regs[0].n_dyads >= 5 and regs[0].end - regs[0].start >= 800


def test_far_repeat_change_passes_through_a_boundary():
    rng = np.random.default_rng(3)
    m = array_molecule([('reg', 25, 165), ('reg', 25, 215)], rng)
    r = _segments(m)
    regs = [s for s in r.segments if s.state == REGULAR]
    assert {s.q_hat for s in regs} == {165.0, 215.0}


def test_long_msps_pre_split_and_short_molecules():
    rng = np.random.default_rng(4)
    m = array_molecule([('reg', 15, 190)], rng)
    # open a 600 bp NFR in the middle by dropping three nucleosomes
    keep = [iv for k, iv in enumerate(m.nucs.tolist()) if k not in (6, 7, 8)]
    m2 = make_molecule(m.length, nucs=keep, msps=complement_msps(m.length, keep),
                       hits=np.flatnonzero(m.hits))
    r = _segments(m2)
    nfr = (keep[5][1], keep[6][0])
    assert all(not (s.start < nfr[1] and s.end > nfr[0]) for s in r.segments)
    short = make_molecule(1500, nucs=[(100, 247), (290, 437)], msps=[(0, 100), (247, 290), (437, 1500)],
                          hits=[99, 250, 440])
    rs = _segments(short)
    assert rs.segments == [] and rs.flags['too_short_segments']


def test_low_information_state_on_cant_tell_runs():
    rng = np.random.default_rng(5)
    m = array_molecule([('reg', 20, 190)], rng)
    # a long stretch of unresolved 700 bp footprints, then the array again
    nucs = m.nucs.tolist()
    pos = nucs[-1][1] + 43
    extra = []
    for _ in range(8):
        extra.append((pos, pos + 700))
        pos += 743
    tail = [(pos + k * 190, pos + k * 190 + 147) for k in range(20)]
    allc = nucs + extra + tail
    L = tail[-1][1] + 300
    occ = np.zeros(L, bool)
    for s, e in allc:
        occ[s:e] = True
    hits = np.flatnonzero(~occ & (rng.random(L) < 0.5))
    mm = make_molecule(L, nucs=allc, msps=complement_msps(L, allc), hits=hits)
    cal = flat_calibration(sigma_loc={'log_lambda': [np.log(0.5)], 'sigma': [30.0]})
    r = molecule_outputs(mm, cal=cal)
    mid = [s for s in r.segments if s.start <= extra[3][0] < s.end]
    assert mid and mid[0].state in (LOWINFO, UNDETERMINED)
    assert mid[0].state != REGULAR


def test_state_space_shape_and_durations():
    sp = state_space(NucParams())
    # 11 q x 2 tightness x 2 comps x 5 + I (3 + 5) + X
    assert len(sp.exp_base) == 11 * 2 * 2 * 5 + 3 + 5 + 1
    assert len(sp.base_order) == 47
    one = state_space(NucParams(seg_sigma_tight=None))
    assert len(one.base_order) == 25


# --- molecule outputs --------------------------------------------------------------------------

def test_molecule_outputs_are_pure_and_deterministic():
    rng = np.random.default_rng(6)
    m = array_molecule([('reg', 20, 188), ('irr', 10, 0)], rng)
    cal = flat_calibration()
    a = molecule_outputs(m, cal=cal)
    b = molecule_outputs(m, cal=cal)
    assert repr(a.scalar_vector()) == repr(b.scalar_vector())
    assert repr([s.as_dict() for s in a.segments]) == repr([s.as_dict() for s in b.segments])
    assert set(a.scalars) == set(SCALARS)
    assert a.digest == calls_digest(m) and len(a.digest) == 8
    assert a.calibration_digest == cal.digest()
    rows = a.marks_rows()
    assert sum(r['type'] == 'nuc.size.coarse' for r in rows) == len(m.nucs) + len(m.tfs)
    assert sum(r['type'] == 'nuc.seg' for r in rows) == len(a.segments)


def test_tf_calls_are_footprints_with_tf_overlap():
    rng = np.random.default_rng(7)
    m = array_molecule([('reg', 10, 190), ('sub', 10, 60)], rng)
    fp = footprint_calls(m, cal=flat_calibration())
    assert fp['tf_overlap'][fp['is_tf']].all()
    assert (fp['coarse'][fp['is_tf']] == SHORT).all()


def test_calls_digest_is_shift_invariant():
    rng = np.random.default_rng(8)
    m = array_molecule([('reg', 10, 190)], rng)
    shifted = make_molecule(m.length + 500, nucs=m.nucs + 500, tfs=m.tfs + 500, msps=m.msps + 500)
    assert calls_digest(m) == calls_digest(shifted)
    moved = m.nucs.copy()
    moved[3] += 5
    other = make_molecule(m.length, nucs=moved, msps=m.msps)
    assert calls_digest(other) != calls_digest(m)


def test_virtual_dyads_give_continuity_not_regularity():
    rng = np.random.default_rng(9)
    L = 20_000
    # a molecule of unresolved 6-nucleosome footprints only: no regular segment
    nucs, pos = [], 300
    while pos + 1150 < L - 300:
        nucs.append((pos, pos + 1100))
        pos += 1150
    occ = np.zeros(L, bool)
    for s_, e_ in nucs:
        occ[s_:e_] = True
    m = make_molecule(L, nucs=nucs, msps=complement_msps(L, nucs),
                      hits=np.flatnonzero(~occ & (rng.random(L) < 0.6)))
    r = molecule_outputs(m, cal=flat_calibration())
    assert (r.footprints['coarse'][~r.footprints['truncated']] == MULTI3).all()
    assert all(s.state != REGULAR for s in r.segments)
    # a regular array interrupted by one multi2 call stays one regular segment
    m2 = array_molecule([('reg', 15, 190)], rng)
    iv = m2.nucs.tolist()
    merged = iv[:7] + [(iv[7][0], iv[8][1])] + iv[9:]
    occ = np.zeros(m2.length, bool)
    for s_, e_ in merged:
        occ[s_:e_] = True
    m3 = make_molecule(m2.length, nucs=merged, msps=complement_msps(m2.length, merged),
                       hits=np.flatnonzero(~occ & (rng.random(m2.length) < 0.5)))
    r3 = molecule_outputs(m3, cal=flat_calibration())
    regs = [s for s in r3.segments if s.state == REGULAR]
    assert len(regs) == 1 and regs[0].start < merged[7][0] and regs[0].end > merged[7][1]


def test_segments_partition_the_molecule_around_composition_runs():
    rng = np.random.default_rng(1)
    m = array_molecule([('reg', 30, 190), ('sub', 60, 90), ('reg', 30, 190)], rng)
    segs = molecule_outputs(m, cal=flat_calibration()).segments
    for a, b in zip(segs, segs[1:]):
        assert a.end <= b.start, (a.as_dict(), b.as_dict())
    from fiberhmm.nuc.window import _seg_fracs
    f = _seg_fracs(segs, 6500, 7500)
    assert sum(f[k] for k in ('regular', 'disordered', 'undetermined', 'low_information')) <= 1.0 + 1e-9


def test_relabel_compares_orders_with_their_composition_terms():
    from fiberhmm.nuc.segment import LOWINFO, Step, segment_steps
    cal = flat_calibration()
    cal.composition = {'cats': ['mono', 'short', 'long', 'multi', 'tf'],
                       'canonical': [0.9, 0.025, 0.025, 0.025, 0.025],
                       'sub_rich': [0.05, 0.45, 0.025, 0.025, 0.45],
                       'rate_canonical': 3 / 230, 'rate_sub_rich': 3 / 230, 'weight_sub_rich': 0.5}
    steps = [Step(230 * i, 230 * (i + 1), 230.0, 1.0, np.array([1.0, 1, 0, 0, 1]), True, True, True)
             for i in range(20)]
    segs = segment_steps(steps, cal)
    assert segs and all(s.state == LOWINFO for s in segs)
