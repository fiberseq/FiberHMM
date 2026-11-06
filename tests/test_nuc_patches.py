"""Regularity patches (fiberhmm.nuc.patches, MA nucpatch) and shared geometries (fiberhmm.nuc.geometries)."""
from __future__ import annotations

import json

import numpy as np
import pysam
import pytest

from nuc_synth import flat_calibration, header, make_record, molecule_sequence, revcomp

from fiberhmm.nuc.geometries import (ABSENT, CARRIES, NO_CALL_STATE, NOT_SPANNED, GeometryParams, MoleculeLayout,
                                     RefPatch, assign_states, density_matched_shares, find_geometries,
                                     geometry_metrics, molecule_layout, nested_near_duplicates, support)
from fiberhmm.nuc.molecule import molecule_outputs
from fiberhmm.nuc.patches import (IN_PATCH, IRREGULAR, NO_CALL, PatchParams, break_linkers, call_patches,
                                  decode_patch_rows, patch_digest, patch_ref_interval, patch_rows, snap_edge)
from fiberhmm.nuc.records import DecodeContext, decode_record, make_molecule
from fiberhmm.nuc.regularity import RegularityParams, RegularityTrack
from fiberhmm.nuc.tags import NUCPATCH, annotate_header, annotate_record, group_rows, read_nuc_groups

NOSNAP = PatchParams(snap=False)


def _track(scores, step=250, first=1000, periods=None):
    s = np.asarray([np.nan if x is None else x for x in scores], dtype=float)
    c = first + step * np.arange(len(s))
    per = np.full(len(s), 190) if periods is None else np.asarray(periods)
    return RegularityTrack(int(c[-1] + 1000), c, c - step // 2, c + step // 2, s, per, np.zeros(len(s)),
                           params_digest='abcdef12')


def _reference_call(v, enter=2.0, extend=1.25, min_w=2):
    """An independent, loop-by-loop statement of the calling rule (no snapping)."""
    n = len(v)
    w = [x if x is not None else ((v[j - 1] + v[j + 1]) / 2 if 0 < j < n - 1 and v[j - 1] is not None
                                  and v[j + 1] is not None else None) for j, x in enumerate(v)]
    st = [255] * n
    out = []
    for j, x in enumerate(w):
        if x is not None:
            st[j] = 0
    j = 0
    while j < n:
        if w[j] is None or w[j] < extend:
            j += 1
            continue
        e = j
        while e + 1 < n and w[e + 1] is not None and w[e + 1] >= extend:
            e += 1
        hit, s, c = False, 0.0, 0
        for q in range(j, e + 1):
            if w[q] >= enter:
                hit = True
            if v[q] is not None:
                s += v[q]
                c += 1
        if hit and e - j + 1 >= min_w:
            for q in range(j, e + 1):
                st[q] = 1
            out.append((j, e, s / c if c else 0.0, j == 0, e == n - 1))
        j = e + 1
    return st, out


# --- calling ------------------------------------------------------------------------------------------
def test_hysteresis_gaps_and_open_ends():
    #        0    1    2    3     4    5    6     7     8    9    10   11   12
    v = [2.5, 1.5, 0.2, 1.3, 2.1, None, 1.4, 0.0, None, None, 2.2, 2.3, 1.0]
    mp = call_patches(_track([x + 3.0 if x is not None else None for x in v]), ref_median=3.0, params=NOSNAP)
    got = [(p.first_tile, p.last_tile, p.open_start, p.open_end) for p in mp.patches]
    # 0-1: open at the molecule start; 3-6: tile 5 bridged (mean of 2.1 and 1.4 >= 1.25); 10-11 closed
    assert got == [(0, 1, True, False), (3, 6, False, False), (10, 11, False, False)]
    p = mp.patches[1]
    assert p.mean_score == pytest.approx(3.0 + np.mean([1.3, 2.1, 1.4]))     # the bridged tile does not count
    assert (p.grid_start, p.grid_end) == (1000 + 3 * 250 - 125, 1000 + 6 * 250 + 125)
    assert p.edges() == p.edges(snapped=False) and p.period == 190
    st = mp.tile_state
    assert st[5] == IN_PATCH and st[8] == NO_CALL and st[9] == NO_CALL and st[7] == IRREGULAR
    # two no-call tiles end a patch; below enter everywhere -> no patch; min_tiles
    assert not call_patches(_track([4.5, 4.5, None, None, 4.5]), ref_median=3.0,
                            params=PatchParams(snap=False, min_tiles=2)).patches[1:]
    assert not call_patches(_track([4.9, 4.9, 4.9]), ref_median=3.0, params=NOSNAP).patches
    one = call_patches(_track([1.0, 5.5, 1.0]), ref_median=3.0, params=NOSNAP).patches
    assert one == [] and len(call_patches(_track([1.0, 5.5, 1.0]), ref_median=3.0,
                                          params=PatchParams(snap=False, min_tiles=1)).patches) == 1


def test_calls_match_an_independent_statement_of_the_rule_on_random_tracks():
    rng = np.random.default_rng(5)
    for _ in range(300):
        n = int(rng.integers(3, 40))
        v = [None if rng.random() < 0.15 else float(np.round(rng.normal(1.2, 1.2), 1)) for _ in range(n)]
        st, want = _reference_call(v)
        mp = call_patches(_track(v), ref_median=0.0, params=NOSNAP)       # exact thresholds: no offset
        got = [(p.first_tile, p.last_tile, p.mean_score, p.open_start, p.open_end) for p in mp.patches]
        assert [g[:2] + g[3:] for g in got] == [w[:2] + w[3:] for w in want]
        assert np.allclose([g[2] for g in got], [w[2] for w in want])
        assert [{255: NO_CALL, 0: IRREGULAR, 1: IN_PATCH}[x] for x in st] == mp.tile_state.tolist()


def test_params_validation_and_digest():
    assert PatchParams.from_dict(PatchParams(open_bp=200).to_dict()) == PatchParams(open_bp=200)
    for bad in ({'enter': 1.0, 'extend': 2.0}, {'snap_rule': 'x'}, {'tol_frac': 0}, {'min_tiles': 0}):
        with pytest.raises(ValueError):
            PatchParams(**bad)
    with pytest.raises(ValueError):
        PatchParams.from_dict({'nope': 1})
    p = PatchParams()
    assert patch_digest('aaaaaaaa', p, 3.0) != patch_digest('aaaaaaaa', p, 3.1)
    assert patch_digest('aaaaaaaa', p, 3.0) != patch_digest('bbbbbbbb', p, 3.0)
    assert patch_digest('aaaaaaaa', p, 3.0) != patch_digest('aaaaaaaa', PatchParams(open_bp=250), 3.0)


def _aligned(L, ref_start, name, reverse=False):
    return make_molecule(L, nucs=[(1000, 1147)], name=name, is_reverse=reverse, ref_start=ref_start,
                         ref_end=ref_start + L, blocks=np.array([[0, ref_start, L]]))


# --- snapping -----------------------------------------------------------------------------------------
def _nucs(starts, core=147):
    return np.array([(s, s + core) for s in starts])


def test_open_rule_snaps_to_the_nucleosome_next_to_the_open_linker():
    # array 2000..5800 every 190 bp, an open stretch, then a second array
    a1 = list(range(2000, 5800, 190))
    a2 = list(range(a1[-1] + 147 + 400, 9000, 190))
    fp = _nucs(a1 + a2)
    last1_end = a1[-1] + 147
    # end edge 600 bp past the array's end: the open linker is the nearest break
    assert snap_edge(last1_end + 600, 'end', fp, 190) == last1_end
    # a start edge 500 bp before a2: snaps to a2's first nucleosome
    assert snap_edge(a2[0] - 500, 'start', fp, 190) == a2[0]
    # nothing open within search_bp: no snap
    assert snap_edge(3500, 'end', fp, 190) is None
    # beyond the outermost footprint counts as a break
    assert snap_edge(a2[-1] + 147 + 300, 'end', fp, 190) == a2[-1] + 147
    with pytest.raises(ValueError):
        snap_edge(0, 'middle', fp, 190)


def test_period_rule_breaks_where_spacing_is_off_and_spares_merged_footprints():
    starts = [1000, 1190, 1380, 1570, 1850, 2040]          # 1570 -> 1850: dyad distance 280
    fp = _nucs(starts)
    pp = PatchParams(snap_rule='period')
    b = break_linkers(fp, 190, pp)
    assert b.tolist() == [False, False, False, True, False]
    # a merged footprint (two nucleosomes in one call) next to a short linker is not a break
    merged = np.array([(1000, 1147), (1190, 1527), (1570, 1717)])
    assert not break_linkers(merged, 190, pp).any()
    assert snap_edge(1900, 'end', fp, 190, pp) == 1717
    # the open rule needs >= open_bp: no break here
    assert not break_linkers(fp, 190, PatchParams()).any()


def test_patch_edges_snap_and_keep_the_grid_edges():
    a1 = list(range(500, 6000, 190))
    nucs = _nucs(a1)
    # a patch over tiles 4..12 of a track starting at 1000 (grid 1875..4125), period 190
    v = [0.0] * 4 + [3.0] * 9 + [0.0] * 4
    tr = _track([x + 3 for x in v])
    snapped = call_patches(tr, nucs, ref_median=3.0).patches[0]
    plain = call_patches(tr, nucs, ref_median=3.0, params=NOSNAP).patches[0]
    assert (plain.start, plain.end) == (plain.grid_start, plain.grid_end) == (1875, 4125)
    # no open linker within 1 kb of either grid edge inside one long regular array: grid edges stay
    assert (snapped.start, snapped.end) == (1875, 4125) and not (snapped.snapped_start or snapped.snapped_end)
    # open stretch right after 3800: the end snaps to the last nucleosome before it
    cut = [s for s in a1 if s < 3700] + [s for s in a1 if s > 4300]
    p = call_patches(tr, _nucs(cut), ref_median=3.0).patches[0]
    last = max(s for s in cut if s < 3700) + 147
    assert p.end == last and p.snapped_end and (p.grid_start, p.grid_end) == (1875, 4125)
    assert patch_rows(call_patches(tr, _nucs(cut), ref_median=3.0))[0][2][2] & 8
    # open-ended edges never snap
    v2 = [3.0] * 6 + [0.0] * 3
    q = call_patches(_track([x + 3 for x in v2]), _nucs([800, 1300]), ref_median=3.0).patches[0]
    assert q.open_start and not q.snapped_start and q.start == q.grid_start


def test_projection_flips_reverse_records_and_codec_round_trips():
    L = 8000
    fwd = _aligned(L, 10_000, 'f')
    rev = _aligned(L, 10_000, 'r', reverse=True)
    tr = _track([5.0, 5.0, 5.0, 0.0])
    mp = call_patches(tr, None, ref_median=3.0, params=NOSNAP)
    p = mp.patches[0]
    assert p.open_start and not p.open_end
    a, b, ol, orr = patch_ref_interval(fwd, p)
    assert (a, b, ol, orr) == (10_000 + p.start, 10_000 + p.end, True, False)
    a, b, ol, orr = patch_ref_interval(rev, p)
    assert (a, b) == (10_000 + L - p.end, 10_000 + L - p.start) and (ol, orr) == (False, True)
    rows = patch_rows(mp)
    d = decode_patch_rows([(s, n) for s, n, _q in rows], [q for _s, _n, q in rows])
    assert d[0]['start'] == p.start and d[0]['end'] == p.end and d[0]['period'] == 190
    assert d[0]['open_start'] and not d[0]['open_end'] and abs(d[0]['mean_score'] - 5.0) < 0.05


# --- geometries ---------------------------------------------------------------------------------------
def _lay(uid, ds, spans, lo=0, hi=10_000, dens=50.0, nocall=()):
    t_lo = np.arange(lo, hi, 250)
    st = np.full(len(t_lo), IRREGULAR, np.int8)
    for a, b in nocall:
        st[(t_lo >= a) & (t_lo < b)] = NO_CALL
    ps = [RefPatch(a, b, False, False, 5.0, 190) for a, b in spans]
    return MoleculeLayout(uid, ds, ps, t_lo, t_lo + 250, st, dens)


def test_geometries_group_by_both_ends_and_count_nested_duplicates():
    rng = np.random.default_rng(3)
    lays = []
    for i in range(60):           # geometry A: 1000-4000
        lays.append(_lay(f'a{i}', 'x' if i % 2 else 'y', [(1000 + int(rng.integers(-200, 200)),
                                                          4000 + int(rng.integers(-200, 200)))]))
    for i in range(30):           # same start, end 1 kb short: a nested near-duplicate of A
        lays.append(_lay(f'b{i}', 'x', [(1000 + int(rng.integers(-100, 100)), 3000 + int(rng.integers(-100, 100)))]))
    for i in range(20):           # geometry C elsewhere
        lays.append(_lay(f'c{i}', 'y', [(6000, 8500)]))
    lays.append(_lay('lonely', 'x', [(5000, 5600)]))
    lays.append(_lay('none', 'y', [], nocall=[(0, 10_000)]))
    lays.append(_lay('short', 'y', [], lo=0, hi=2000))
    G = find_geometries(lays, GeometryParams(min_share=0.03))
    assert [g.n_molecules for g in G] == [60, 30, 20]
    assert abs(G[0].start - 1000) <= 60 and abs(G[0].end - 4000) <= 60 and abs(G[1].end - 3000) <= 40
    assert (G[2].start, G[2].end, G[2].start_mad, G[2].end_mad) == (6000, 8500, 0.0, 0.0)
    assert nested_near_duplicates(G) == [(G[0].geometry_id, G[1].geometry_id)]
    m = geometry_metrics(lays, G)
    assert m['geometries'] == 3 and m['patches'] == 111 and m['assigned'] == 110 and m['nested_pairs'] == 1
    # need = max(3, ceil(share x molecules)): 25% of 113 molecules keeps only A and B
    assert [g.n_molecules for g in find_geometries(lays, GeometryParams(min_share=0.25))] == [60, 30]
    assert len(find_geometries(lays, GeometryParams(max_geometries=1))) == 1
    states = assign_states(lays, G)
    gid = G[2].geometry_id
    names = [l.uid for l in lays]
    assert states[names.index('c0')][gid] == CARRIES and states[names.index('a0')][gid] == ABSENT
    assert states[names.index('none')][gid] == NO_CALL_STATE and states[names.index('short')][gid] == NOT_SPANNED
    s = support(lays, states, gid, 'y')
    assert s['carriers'] == 20 and s['molecules'] == sum(l.dataset == 'y' for l in lays)
    assert s['callable'] == s['molecules'] - 2
    sh = density_matched_shares(lays, states, G[0].geometry_id, ['x', 'y'], n_bins=1)
    assert sh['x'] == pytest.approx(30 / (30 + 30 + 1)) and sh['y'] == pytest.approx(30 / (30 + 20))


def test_molecule_layout_clips_to_a_view_and_marks_open_sides():
    L = 9000
    mol = _aligned(L, 20_000, 'm')
    v = [0.0] * 4 + [3.0] * 20 + [0.0] * 4
    tr = _track([x + 3 for x in v], first=1000)
    mp = call_patches(tr, None, ref_median=3.0, params=NOSNAP)
    full = molecule_layout(mol, tr, mp, 'd')
    p = full.patches[0]
    assert (p.start, p.end, p.open_left, p.open_right) == (20_000 + 1875, 20_000 + 6875, False, False)
    clip = molecule_layout(mol, tr, mp, 'd', view=(20_000 + 3000, 20_000 + 5000))
    q = clip.patches[0]
    assert (q.start, q.end, q.open_left, q.open_right) == (23_000, 25_000, True, True)
    assert clip.span == (23_000, 25_000) and full.uid == 'm'


# --- MA group -----------------------------------------------------------------------------------------
def _record(tmp_path, reverse=False):
    rng = np.random.default_rng(12)
    n_nuc = 50
    L = 300 + n_nuc * 190 + 300
    seq = molecule_sequence(L, rng)
    nucs = [(300 + 190 * k, 300 + 190 * k + 147) for k in range(n_nuc) if not 20 <= k < 24]
    occ = np.zeros(L, bool)
    for a, b in nucs:
        occ[a:b] = True
    marks = [i for i in range(12, L - 12) if seq[i] == 'A' and not occ[i] and rng.random() < 0.7]
    msps, p = [], 0
    for a, b in nucs:
        msps.append((p, a))
        p = b
    msps.append((p, L))
    path = str(tmp_path / 'r.bam')
    hdr = annotate_header(pysam.AlignmentHeader.from_dict(header()), regularity=RegularityParams(),
                          patches=PatchParams(), reference=3.0)
    with pysam.AlignmentFile(path, 'wb', header=hdr) as out:
        out.write(make_record(out, 'r1', seq, 1000, nucs, msps, (), marks, reverse=reverse))
    bam = pysam.AlignmentFile(path)
    read = next(bam.fetch(until_eof=True))
    mol, _ = decode_record(read, DecodeContext(bam.header))
    return bam, read, mol


@pytest.mark.parametrize('reverse', [False, True])
def test_nucpatch_tags_decode_back_and_go_stale(tmp_path, reverse):
    bam, read, mol = _record(tmp_path, reverse)
    reg, pp = RegularityParams(), PatchParams()
    with pytest.raises(ValueError):
        molecule_outputs(mol, cal=flat_calibration(), regularity=reg, patches=pp)
    res = molecule_outputs(mol, cal=flat_calibration(), regularity=reg, patches=pp, ref_median=3.0)
    assert molecule_outputs(mol, cal=flat_calibration(), regularity=reg).patches is None
    assert res.patches.patches, 'the planted arrays give patches'
    annotate_record(read, res)
    dg = patch_digest(res.regularity.params_digest, pp, 3.0)
    names = [n for n in read.get_tag('AN').split(',') if n.startswith('nucpatch@')]
    assert names and all(n.startswith(f'nucpatch@{res.digest}.{dg}_') for n in names)
    g = read_nuc_groups(read, patches_digest=dg)[NUCPATCH]
    assert not g['stale'] and not g['params_stale'] and g['params_digest'] == dg
    d = decode_patch_rows(g['intervals'], g['q'])
    assert [(x['start'], x['end']) for x in d] == [(p.start, p.end) for p in res.patches.patches]
    other = read_nuc_groups(read, patches_digest=patch_digest(res.regularity.params_digest, pp, 3.2))[NUCPATCH]
    assert other['stale'] and other['params_stale']
    contracts = {json.loads(c.split(':', 2)[2])['group']: json.loads(c.split(':', 2)[2])
                 for c in bam.header.to_dict()['CO'] if c.startswith('FIBERHMM-SCAN-MA')}
    assert contracts[NUCPATCH]['params'] == pp.to_dict() and contracts[NUCPATCH]['reference'] == 3.0
    assert NUCPATCH not in group_rows(molecule_outputs(mol, cal=flat_calibration(), regularity=reg))


# --- scan ---------------------------------------------------------------------------------------------
def _scan_bam(tmp_path):
    rng = np.random.default_rng(13)
    length = 40_000
    genome = molecule_sequence(length, rng)
    dyads, pos = [], 1500.0
    while pos < length - 1500:
        pos += 190 + rng.normal(0, 3) if 8000 <= pos < 30_000 and not 17_000 <= pos < 17_600 else \
            rng.uniform(150, 300) if not 17_000 <= pos < 17_600 else 600
        dyads.append(pos)
    dyads = np.array(dyads)
    path = str(tmp_path / 'scan.bam')
    with pysam.AlignmentFile(path, 'wb', header=header(contig='chrS', length=length)) as out:
        for k, start in enumerate(range(4000, length - 12_000, 400)):
            L = int(min(rng.integers(9000, 12_000), length - start))
            ref_seq = genome[start:start + L]
            dd = dyads[(dyads > start + 80) & (dyads < start + L - 80)] - start
            nucs_ref = [(int(d - 73), int(d + 74)) for d in dd]
            rev = k % 3 == 1
            mol_seq = revcomp(ref_seq) if rev else ref_seq
            nucs = sorted([(L - e, L - s) for s, e in nucs_ref] if rev else nucs_ref)
            occ = np.zeros(L, bool)
            for s, e in nucs:
                occ[s:e] = True
            a = np.array([b == 'A' for b in mol_seq])
            marks = np.flatnonzero(a & ~occ & (rng.random(L) < 0.55))
            marks = marks[(marks >= 12) & (marks < L - 12)]
            msps, p = [], 0
            for s, e in nucs:
                if s > p:
                    msps.append((p, s))
                p = e
            if p < L:
                msps.append((p, L))
            out.write(make_record(out, f'm1/{100 + k}/ccs', mol_seq, start, nucs, msps, (), marks, reverse=rev))
    pysam.index(path)
    return path, length


def test_scan_writes_nucpatch_marks_geometries_and_states(tmp_path):
    from fiberhmm.nuc.calibration import NucCalibration
    from fiberhmm.scan import run_units
    from fiberhmm.scan.tables import read_all
    from fiberhmm.scan.units import plan
    bam, length = _scan_bam(tmp_path)
    units, _info = plan({'chrS': length}, regions=[('chrS', 14_000, 20_000)])
    assert units[0].mode == 'bed'
    base = {'calibration': 'self', 'calibration_population': 300}
    ds = [{'dataset_id': 'synth', 'paths': [bam]}]
    params = {'nuc': dict(base, nucpatch={}, nucgeo={'min_share': 0.05})}
    run_units(ds, units, modules=['nuc'], params=params, out=tmp_path / 'tsv', annotate=False)
    rec = json.loads((tmp_path / 'tsv' / 'nuc' / 'extra' / 'calibration.json').read_text())
    ref = rec['nucreg_reference']['synth']
    assert ref['n'] > 100 and np.isfinite(ref['median'])
    cal = NucCalibration.from_dict(rec['datasets']['synth'])
    reg, pp = RegularityParams(), PatchParams()
    direct = {}
    with pysam.AlignmentFile(bam) as b:
        ctx = DecodeContext(b.header)
        for r in b.fetch(until_eof=True):
            mol, _ = decode_record(r, ctx)
            direct[r.query_name] = molecule_outputs(mol, cal=cal, regularity=reg, patches=pp,
                                                    ref_median=ref['median'])
    marks = [m for m in read_all(tmp_path / 'tsv' / 'nuc' / 'marks.tsv.bgz') if m['group'] == NUCPATCH]
    assert marks
    got = sorted((m['qname'], int(m['mol_start']), int(m['mol_len']), m['q'], m['name']) for m in marks)
    want = sorted((q, s, n, ','.join(map(str, qb)), a) for q, res in direct.items()
                  for s, n, qb, a in group_rows(res)[NUCPATCH][1] if q in {m['qname'] for m in marks})
    assert got == want
    hits = [h for h in read_all(tmp_path / 'tsv' / 'nuc' / 'hits.tsv.bgz') if h['type'] == 'nuc.geometry']
    assert hits
    states = [s for s in read_all(tmp_path / 'tsv' / 'nuc' / 'states.tsv.bgz')]
    by_hit = {}
    for s in states:
        by_hit.setdefault(s['hit_id'], []).append(s)
    for h in hits:
        rows = by_hit[h['hit_id']]
        carriers = [s for s in rows if s['state'] == '1']
        assert len({s['qname'] for s in carriers}) == int(h['molecules'])
        assert all(s['reason'] == 'carries' for s in carriers)
        assert {s['reason'] for s in rows if s['state'] == '.'} <= {'no_call', 'not_spanned'}
        assert float(h['score']) == pytest.approx(len(carriers) / len(rows))
    vals = [v for v in read_all(tmp_path / 'tsv' / 'nuc' / 'values.tsv.bgz')
            if v['hit_id'] == hits[0]['hit_id'] and v['value_name'] == 'share']
    assert {v['group'] for v in vals} == {'all', 'synth'}
    # annotated: nucpatch marks and nucgeo states on the records
    run_units(ds, units, modules=['nuc'], params=params, out=tmp_path / 'ann', annotate=True)
    m = json.loads((tmp_path / 'ann' / 'manifest.json').read_text())
    n_geo = 0
    with pysam.AlignmentFile(str(tmp_path / 'ann' / m['outputs']['annotated'][0]['path'])) as out:
        for r in out.fetch(until_eof=True):
            ma = r.get_tag('MA') if r.has_tag('MA') else ''
            n_geo += 'nucgeo' in ma
            g = read_nuc_groups(r)
            if NUCPATCH in g and r.query_name in direct:
                assert not g[NUCPATCH]['stale']
    assert n_geo > 0
    # off: no reference in the calibration, no rows, no geometries
    run_units(ds, units, modules=['nuc'], params={'nuc': base}, out=tmp_path / 'off', annotate=False)
    assert 'nucreg_reference' not in json.loads((tmp_path / 'off' / 'nuc' / 'extra' / 'calibration.json').read_text())
    assert not [m for m in read_all(tmp_path / 'off' / 'nuc' / 'marks.tsv.bgz') if m['group'] == NUCPATCH]
    assert not [h for h in read_all(tmp_path / 'off' / 'nuc' / 'hits.tsv.bgz') if h['type'] == 'nuc.geometry']


def test_settings_validation():
    from fiberhmm.scan.modules.nuc import NucModule
    mod = NucModule()
    s = mod.resolve({'nucpatch': {'open_bp': 250, 'ref_median': 3}})
    assert s['nucpatch']['open_bp'] == 250 and s['nucpatch']['ref_median'] == 3.0
    assert mod.resolve({}).get('nucpatch') is None and 'nucgeo' not in mod.resolve({})
    for bad in ({'nucpatch': {}, 'nucreg': None}, {'nucpatch': {'bogus': 1}}, {'nucpatch': {}, 'nucgeo': {'cohort': 'x'}},
                {'nucpatch': {'reference_windows': 5}}):
        with pytest.raises(ValueError):
            mod.resolve(bad)


def test_cli_flags_fold_into_nuc_parameters():
    from fiberhmm.scan.cli import _nucpatch_params, _parser
    p = _parser()

    def fold(argv, modules=('nuc',), params=None):
        params = {} if params is None else params
        _nucpatch_params(p, p.parse_args(argv), list(modules), params)
        return params
    assert fold([]) == {}
    got = fold(['--nucpatch-enter', '2.5', '--nucpatch-snap', 'period', '--nucpatch-ref', '3.1',
                '--nucgeo-tol', '400', '--nucgeo-edges', 'grid', '--nucgeo-cohort', 'overlapping'])
    assert got['nuc']['nucpatch']['enter'] == 2.5 and got['nuc']['nucpatch']['snap_rule'] == 'period'
    assert got['nuc']['nucpatch']['ref_median'] == 3.1
    assert got['nuc']['nucgeo'] == dict(GeometryParams(tol_bp=400, snapped=False).to_dict(), cohort='overlapping')
    assert fold(['--nucpatch', '--nucpatch-snap', 'off'])['nuc']['nucpatch']['snap'] is False
    assert fold(['--nucpatch', '--no-nucgeo'])['nuc']['nucgeo'] is None
    for argv, modules in ((['--nucpatch'], ('fp',)), (['--nucgeo-tol', '300'], ('nuc',)),
                          (['--nucpatch', '--no-nucreg'], ('nuc',)),
                          (['--nucpatch', '--no-nucgeo', '--nucgeo-tol', '3'], ('nuc',)),
                          (['--nucpatch-enter', '1', '--nucpatch-extend', '2'], ('nuc',))):
        with pytest.raises(SystemExit):
            fold(argv, modules)
