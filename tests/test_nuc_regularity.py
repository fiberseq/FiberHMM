"""Single-fibre nucleosome regularity (fiberhmm.nuc.regularity, MA nucreg).

- the window score reproduces the exploration prototype's spectral() formula;
- regular arrays score above random packing, at the planted repeat length;
- min_calls, open-stretch masking, read-name thinning and the tile grid;
- nucreg bytes round-trip, the random-packing reference is seeded;
- a scan writes nucreg marks and annotated-BAM tags that decode back, and a
  reader flags rows made with other parameters as stale.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pysam
import pytest

from nuc_synth import flat_calibration, header, make_record, molecule_sequence, revcomp

from fiberhmm.nuc.molecule import molecule_outputs
from fiberhmm.nuc.records import DecodeContext, decode_record, make_molecule
from fiberhmm.nuc.regularity import (RegularityParams, decode_period, decode_rows, decode_score, encode_period,
                                     encode_score, masked_mark_acf, open_mask, reference_scores,
                                     sliding_regularity, spectral_scores, thinned_hits, track_rows,
                                     track_summary)
from fiberhmm.nuc.tags import NUCREG, annotate_header, annotate_record, group_rows, read_nuc_groups


# --- the prototype formulas, inlined (exploration scripts, not shipped) -----------------------------
def _prototype_spectral(X, WIN=2000, LO=140, HI=260, NFFT=8192):
    freq = np.fft.rfftfreq(NFFT)
    with np.errstate(divide='ignore'):
        period = np.where(freq > 0, 1 / freq, np.inf)
    BAND = (period >= LO) & (period <= HI)
    REF = (period >= 50) & (period <= 1000) & ~BAND
    taper = np.hanning(WIN)
    X = (X - X.mean(axis=1, keepdims=True)) * taper
    P = np.abs(np.fft.rfft(X, NFFT, axis=1)) ** 2
    ref = np.median(P[:, REF], axis=1)
    bandP = P[:, BAND]
    j = np.argmax(bandP, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(ref > 0, bandP[np.arange(len(P)), j] / np.where(ref > 0, ref, 1), np.nan)
        return np.log2(snr), np.round(period[BAND][j]).astype(int)


def _prototype_masked_acf(x, w, MAXLAG=500):
    n = len(x)
    m = 1 << (2 * n - 1).bit_length()
    mu = (x * w).sum() / w.sum()
    y = (x - mu) * w
    fy, fw = np.fft.rfft(y, m), np.fft.rfft(w, m)
    num = np.fft.irfft(fy * np.conj(fy), m)[:MAXLAG + 1]
    den = np.fft.irfft(fw * np.conj(fw), m)[:MAXLAG + 1]
    c = num / np.maximum(den, 1)
    return c / c[0]


# --- synthetic molecules ----------------------------------------------------------------------------
def _array(length, rng, repeat=190, core=147, rate=0.5, regular=True, name='mol', open_iv=()):
    """Nucleosome calls regular at ``repeat`` (or randomly packed at the same
    count), marks in linkers at ``rate``; ``open_iv`` stretches hold no calls."""
    if regular:
        starts = np.arange(100, length - core - 100, repeat)
    else:
        n = len(np.arange(100, length - core - 100, repeat))
        free = (length - 200) - n * core
        cuts = np.sort(rng.integers(0, free + 1, n))
        starts = 100 + cuts + np.arange(n) * core
    nucs = [(int(s), int(s + core)) for s in starts
            if not any(a < s + core and s < b for a, b in open_iv)]
    occ = np.zeros(length, bool)
    for a, b in nucs:
        occ[a:b] = True
    hits = ~occ & (rng.random(length) < rate)
    return make_molecule(length, nucs=nucs, hits=hits, name=name)


def test_score_reproduces_prototype_spectral():
    rng = np.random.default_rng(0)
    X = (rng.random((40, 2000)) < 0.08).astype(float)                    # random marks
    t = np.arange(2000)
    X[:10] = (np.sin(2 * np.pi * t / 190) > 0.3) * (rng.random((10, 2000)) < 0.4)   # periodic
    X[10] = 0.0                                                            # flat: no reference power
    X[11] = rng.random(2000)                                               # continuous
    want_s, want_p = _prototype_spectral(X)
    got_s, got_p = spectral_scores(X, RegularityParams())
    assert np.array_equal(np.isfinite(got_s), np.isfinite(want_s))
    ok = np.isfinite(want_s)
    assert np.allclose(got_s[ok], want_s[ok], rtol=0, atol=1e-9)
    assert np.array_equal(got_p[ok], want_p[ok])
    assert (got_p[~ok] == -1).all()


def test_regular_arrays_score_above_random_packing_at_the_planted_repeat():
    rng = np.random.default_rng(1)
    for repeat in (172, 190, 205):
        reg = sliding_regularity(_array(12_000, rng, repeat=repeat))
        rnd = sliding_regularity(_array(12_000, rng, repeat=repeat, regular=False))
        assert np.isfinite(reg.score).all() and np.isfinite(rnd.score).all()
        assert np.median(reg.score) > np.median(rnd.score) + 1.5
        assert abs(np.median(reg.period) - repeat) <= 4


def test_occupancy_signal_scores_the_calls():
    rng = np.random.default_rng(2)
    p = RegularityParams(signal='occupancy')
    reg = sliding_regularity(_array(10_000, rng), p)
    rnd = sliding_regularity(_array(10_000, rng, regular=False), p)
    assert np.median(reg.score) > np.median(rnd.score) + 1.5
    assert abs(np.median(reg.period) - 190) <= 4
    # no calls: nothing to score
    empty = sliding_regularity(make_molecule(5000, hits=np.arange(0, 5000, 7)), p)
    assert len(empty) and not np.isfinite(empty.score).any() and (empty.period == -1).all()


def test_windows_below_min_calls_have_no_score():
    rng = np.random.default_rng(3)
    mol = _array(8000, rng, rate=0.012)              # about 10 calls per 2 kb window
    tr = sliding_regularity(mol)
    assert (tr.n_calls < 15).all()
    assert not np.isfinite(tr.score).any() and (tr.period == -1).all()
    loose = sliding_regularity(mol, RegularityParams(min_calls=3))
    assert np.isfinite(loose.score[loose.n_calls >= 3]).all()
    assert track_summary(tr, 3.0) == {'n_tiles': 0, 'median': None, 'max': None, 'frac_above_ref': None,
                                      'repeat_length': None}
    short = sliding_regularity(make_molecule(1500, hits=np.arange(0, 1500, 9)))
    assert len(short) == 0 and track_rows(short) == []


def test_mask_removes_an_inserted_open_stretch():
    """With masking, what an open stretch holds (dense marks, none, or an
    array-like pattern) no longer changes any score; without it, it does."""
    rng = np.random.default_rng(4)
    base = _array(12_000, np.random.default_rng(40), rate=0.3)
    nfr = (5800, 6400)
    nucs = [(a, b) for a, b in base.nucs.tolist() if not (a < nfr[1] and nfr[0] < b)]
    inside = np.zeros(12_000, bool)
    inside[nfr[0]:nfr[1]] = True
    empty = base.hits & ~inside
    fillings = (empty, empty | (inside & (rng.random(12_000) < 0.6)),
                empty | (inside & (np.sin(2 * np.pi * np.arange(12_000) / 190) > 0.5)))
    mols = [make_molecule(12_000, nucs=nucs, hits=h, name='mol') for h in fillings]
    keep = open_mask(mols[0], 300)
    assert not keep[nfr[0] + 50:nfr[1] - 50].any() and keep[2000:5000].all()
    assert not keep[:base.nucs[0, 0]].any() and not keep[base.nucs[-1, 1]:].any()
    on = RegularityParams(mask_open_bp=300)
    masked = [sliding_regularity(m, on).score for m in mols]
    plain = [sliding_regularity(m).score for m in mols]
    centres = sliding_regularity(mols[0], on).centres
    near = (centres - 1000 < nfr[1]) & (centres + 1000 > nfr[0])          # windows meeting the stretch
    assert near.sum() >= 5 and np.isfinite(masked[0][near]).all()
    for k in (1, 2):
        assert np.allclose(masked[k], masked[0], rtol=0, atol=1e-9)
        assert np.abs(plain[k][near] - plain[0][near]).max() > 0.5
        assert np.allclose(plain[k][~near], plain[0][~near], rtol=0, atol=1e-9)
    # a window more than half open has no score
    wide = make_molecule(12_000, nucs=[(a, b) for a, b in base.nucs.tolist() if not (3000 < a < 7000)],
                         hits=base.hits, name='mol')
    tr = sliding_regularity(wide, on)
    inside = (tr.centres > 4200) & (tr.centres < 5800)
    assert inside.any() and not np.isfinite(tr.score[inside]).any()


def test_thinning_is_seeded_by_read_name():
    rng = np.random.default_rng(5)
    mol = _array(10_000, rng, rate=0.5, name='read/a')
    twin = make_molecule(10_000, nucs=mol.nucs, hits=mol.hits, name='read/a')
    other = make_molecule(10_000, nucs=mol.nucs, hits=mol.hits, name='read/b')
    p = RegularityParams(thin_per_kb=25)
    a, b, c = (sliding_regularity(m, p) for m in (mol, twin, other))
    assert a.thinned and np.array_equal(a.n_calls, b.n_calls) and np.array_equal(a.score, b.score, equal_nan=True)
    assert not np.array_equal(a.n_calls, c.n_calls)
    h, thinned, density = thinned_hits(mol, 25.0)
    assert thinned and density > 25 and abs(1000 * h.sum() / mol.length - 25) < 4
    assert (h <= mol.hits).all()
    sparse = _array(10_000, rng, rate=0.03)
    h2, thinned2, _d = thinned_hits(sparse, 25.0)
    assert not thinned2 and np.array_equal(h2, sparse.hits)
    assert not sliding_regularity(sparse, p).thinned


def test_tiles_abut_and_own_each_position_once():
    rng = np.random.default_rng(6)
    for step in (250, 200, 333):
        tr = sliding_regularity(_array(9_731, rng), RegularityParams(step_bp=step))
        assert len(tr) == (9_731 - 2000) // step + 1
        assert ((tr.tile_ends - tr.tile_starts) == step).all()
        assert np.array_equal(tr.tile_ends[:-1], tr.tile_starts[1:])
        assert ((tr.tile_starts <= tr.centres) & (tr.centres < tr.tile_ends)).all()
        assert tr.tile_starts[0] >= 0 and tr.tile_ends[-1] <= 9_731
        pos = np.arange(tr.tile_starts[0], tr.tile_ends[-1])
        owner = np.searchsorted(tr.tile_starts, pos, side='right') - 1
        assert ((tr.tile_starts[owner] <= pos) & (pos < tr.tile_ends[owner])).all()


def test_nucreg_bytes_round_trip():
    assert encode_score(float('nan')) == 0 and encode_score(None) == 0
    assert encode_score(-5.0) == 1 and encode_score(-2.0) == 1 and encode_score(8.0) == 255
    assert encode_score(99.0) == 255
    for s in np.linspace(-2, 8, 37):
        assert abs(decode_score(encode_score(s)) - s) <= 10 / 254 / 2 + 1e-12
    assert np.isnan(decode_score(0))
    assert encode_period(-1) == 0 and decode_period(0) == -1
    for p in (140, 187, 260):
        assert decode_period(encode_period(p)) == p
    rng = np.random.default_rng(7)
    tr = sliding_regularity(_array(10_000, rng))
    rows = track_rows(tr)
    assert len(rows) == int(np.isfinite(tr.score).sum())
    d = decode_rows([(s, n) for s, n, _q in rows], [q for _s, _n, q in rows])
    ok = np.isfinite(tr.score)
    assert np.array_equal(d['tile_starts'], tr.tile_starts[ok]) and np.array_equal(d['period'], tr.period[ok])
    assert np.array_equal(d['centres'], tr.centres[ok])
    assert np.abs(d['score'] - np.clip(tr.score[ok], -2, 8)).max() <= 10 / 254 / 2 + 1e-12


def test_reference_scores_are_seeded_and_below_regular_arrays():
    rng = np.random.default_rng(8)
    mols = [_array(int(rng.integers(6000, 12_000)), rng, name=f'm{i}') for i in range(30)]
    p = RegularityParams()
    a = reference_scores(mols, p, n_windows=300, seed=11)
    b = reference_scores(mols, p, n_windows=300, seed=11)
    c = reference_scores(mols, p, n_windows=300, seed=12)
    assert a.n > 250 and np.array_equal(a.scores, b.scores) and a.median == b.median
    assert not np.array_equal(a.scores, c.scores)
    observed = np.concatenate([sliding_regularity(m, p).score for m in mols])
    assert np.nanmedian(observed) > a.median + 1.5
    assert sorted(a.quantiles) == [0.05, 0.25, 0.5, 0.75, 0.95]
    assert a.quantiles[0.05] <= a.median <= a.quantiles[0.95]
    json.dumps(a.to_dict())
    occ = reference_scores(mols, RegularityParams(signal='occupancy'), n_windows=200, seed=1)
    assert occ.n > 150 and np.isfinite(occ.median)
    assert reference_scores([], p).n == 0
    s = track_summary(sliding_regularity(mols[0], p), a.median)
    assert s['frac_above_ref'] > 0.8 and s['max'] >= s['median'] > a.median


def test_masked_mark_acf_matches_the_prototype():
    rng = np.random.default_rng(9)
    mol = _array(8000, rng, rate=0.4, open_iv=[(3000, 3700)])
    lags = np.arange(0, 501, 4)
    keep = open_mask(mol, 300).astype(float)
    want = _prototype_masked_acf(mol.hits.astype(float), keep)[lags]
    got = masked_mark_acf(mol, lags, mask_open_bp=300, thin_per_kb=None)
    assert np.allclose(got, want, rtol=0, atol=1e-9)
    unmasked = masked_mark_acf(mol, lags, mask_open_bp=None, thin_per_kb=None)
    assert np.allclose(unmasked, _prototype_masked_acf(mol.hits.astype(float), np.ones(8000))[lags], atol=1e-9)
    k = int(np.argmax(got[(lags >= 140) & (lags <= 260)]))
    assert abs(lags[(lags >= 140) & (lags <= 260)][k] - 190) <= 8
    thinned = masked_mark_acf(mol, lags)                      # 300 bp mask, 25/kb, seeded by the name
    assert np.array_equal(thinned, masked_mark_acf(mol, lags))
    assert masked_mark_acf(make_molecule(3000, hits=[5, 9]), lags) is None      # no calls: all masked


def test_params_round_trip_and_validation():
    p = RegularityParams(window_bp=1500, band_bp=(150, 250), mask_open_bp=300, thin_per_kb=25.0)
    assert RegularityParams.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    assert p.digest() != RegularityParams().digest()
    with pytest.raises(ValueError):
        RegularityParams.from_dict({'bogus': 1})
    with pytest.raises(ValueError):
        RegularityParams(signal='dna')
    with pytest.raises(ValueError):
        RegularityParams(band_bp=(260, 140))


def test_a_20kb_molecule_is_fast():
    rng = np.random.default_rng(10)
    mol = _array(20_000, rng)
    sliding_regularity(mol)
    t0 = time.perf_counter()
    for _ in range(5):
        sliding_regularity(mol)
    assert (time.perf_counter() - t0) / 5 < 0.25            # about 3 ms here; loose for slow CI


# --- tags, staleness and the scan ---------------------------------------------------------------------
def _record(tmp_path, n_nuc=40, reverse=False):
    rng = np.random.default_rng(12)
    L = 300 + n_nuc * 190 + 300
    seq = molecule_sequence(L, rng)
    nucs = [(300 + 190 * k, 300 + 190 * k + 147) for k in range(n_nuc)]
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
    with pysam.AlignmentFile(path, 'wb', header=annotate_header(pysam.AlignmentHeader.from_dict(
            header()), regularity=RegularityParams())) as out:
        out.write(make_record(out, 'r1', seq, 1000, nucs, msps, (), marks, reverse=reverse))
    bam = pysam.AlignmentFile(path)
    read = next(bam.fetch(until_eof=True))
    mol, _ = decode_record(read, DecodeContext(bam.header))
    return bam, read, mol


@pytest.mark.parametrize('reverse', [False, True])
def test_nucreg_tags_decode_back_and_go_stale_when_params_change(tmp_path, reverse):
    bam, read, mol = _record(tmp_path, reverse=reverse)
    p = RegularityParams()
    res = molecule_outputs(mol, cal=flat_calibration(), regularity=p)
    assert molecule_outputs(mol, cal=flat_calibration()).regularity is None
    annotate_record(read, res)
    names = [n for n in read.get_tag('AN').split(',') if n.startswith('nucreg@')]
    assert names and all(n.startswith(f'nucreg@{res.digest}.{p.digest()[:8]}_') for n in names)
    g = read_nuc_groups(read, regularity=p)[NUCREG]
    assert not g['stale'] and not g['params_stale'] and g['digest'] == res.digest
    assert g['params_digest'] == p.digest()[:8]
    d = decode_rows(g['intervals'], g['q'])
    ok = np.isfinite(res.regularity.score)
    assert np.array_equal(d['tile_starts'], res.regularity.tile_starts[ok])
    assert np.array_equal(d['period'], res.regularity.period[ok])
    assert abs(np.median(d['period']) - 190) <= 4
    other = read_nuc_groups(read, regularity=RegularityParams(window_bp=1500))[NUCREG]
    assert other['stale'] and other['params_stale']
    assert not read_nuc_groups(read)[NUCREG]['stale']                # no parameters asked: calls only
    # new calls: the calls digest no longer matches
    contracts = {json.loads(c.split(':', 2)[2])['group']: json.loads(c.split(':', 2)[2])
                 for c in bam.header.to_dict()['CO'] if c.startswith('FIBERHMM-SCAN-MA')}
    assert contracts[NUCREG]['params'] == p.to_dict() and contracts[NUCREG]['params_digest'] == p.digest()[:8]
    moved = mol.nucs.copy()
    moved[3] += 20
    from fiberhmm.nuc.records import make_molecule as mm
    stale_mol = mm(mol.length, nucs=moved, msps=mol.msps, hits=mol.hits)
    assert read_nuc_groups(read, mol=stale_mol, regularity=p)[NUCREG]['stale']


def _scan_bam(tmp_path):
    rng = np.random.default_rng(13)
    length = 60_000
    genome = molecule_sequence(length, rng)
    dyads, pos = [], 1500.0
    while pos < length - 1500:
        pos += 190 + rng.normal(0, 3) if 10_000 <= pos < 40_000 else rng.uniform(150, 300)
        dyads.append(pos)
    dyads = np.array(dyads)
    path = str(tmp_path / 'scan.bam')
    with pysam.AlignmentFile(path, 'wb', header=header(contig='chrS', length=length)) as out:
        for k, start in enumerate(range(0, length - 9000, 700)):
            L = int(min(rng.integers(8000, 11_000), length - start))
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


def test_scan_writes_nucreg_marks_and_annotated_tags(tmp_path):
    from fiberhmm.nuc.calibration import NucCalibration
    from fiberhmm.scan import run_units
    from fiberhmm.scan.tables import read_all
    from fiberhmm.scan.units import plan
    bam, length = _scan_bam(tmp_path)
    units = plan({'chrS': length}, genome=True, contigs=['chrS'])
    reg = {'mask_open_bp': 300}
    p = RegularityParams(mask_open_bp=300)
    params = {'nuc': {'calibration': 'self', 'calibration_population': 400, 'nucreg': reg}}
    ds = [{'dataset_id': 'synth', 'paths': [bam]}]
    run_units(ds, units, modules=['nuc'], params=params, out=tmp_path / 'tsv', annotate=False)
    run_units(ds, units, modules=['nuc'], params=params, out=tmp_path / 'ann', annotate=True)
    cal = NucCalibration.from_dict(json.loads((tmp_path / 'tsv' / 'nuc' / 'extra' / 'calibration.json')
                                              .read_text())['datasets']['synth'])
    direct = {}
    with pysam.AlignmentFile(bam) as b:
        ctx = DecodeContext(b.header)
        for r in b.fetch(until_eof=True):
            mol, _ = decode_record(r, ctx)
            direct[r.query_name] = molecule_outputs(mol, cal=cal, regularity=p)
    marks = [m for m in read_all(tmp_path / 'tsv' / 'nuc' / 'marks.tsv.bgz') if m['group'] == NUCREG]
    assert marks
    got = sorted((m['qname'], int(m['mol_start']), int(m['mol_len']), m['q'], m['name']) for m in marks)
    want = sorted((q, s, n, ','.join(map(str, qb)), a) for q, res in direct.items()
                  for s, n, qb, a in group_rows(res)[NUCREG][1])
    assert got == want
    m = json.loads((tmp_path / 'ann' / 'manifest.json').read_text())
    path = tmp_path / 'ann' / m['outputs']['annotated'][0]['path']
    seen = 0
    with pysam.AlignmentFile(str(path)) as out:
        co = out.header.to_dict()['CO']
        line = next(json.loads(c.split(':', 2)[2]) for c in co
                    if c.startswith('FIBERHMM-SCAN-MA') and '"group":"nucreg"' in c)
        assert line['params'] == p.to_dict() and line['params_digest'] == p.digest()[:8]
        for r in out.fetch(until_eof=True):
            g = read_nuc_groups(r, regularity=p)
            res = direct[r.query_name]
            want_rows = sorted((s, n, list(q)) for s, n, q, _a in group_rows(res)[NUCREG][1])
            got_rows = sorted((s, n, list(q)) for (s, n), q in zip(g[NUCREG]['intervals'], g[NUCREG]['q']))
            assert got_rows == want_rows
            assert not g[NUCREG]['stale']
            d = decode_rows(g[NUCREG]['intervals'], g[NUCREG]['q'])
            ok = np.isfinite(res.regularity.score)
            assert np.array_equal(d['period'], res.regularity.period[ok])
            seen += 1
    assert seen == len(direct)
    # nucreg off: no rows, and the contract says so
    off = dict(params, nuc=dict(params['nuc'], nucreg=None))
    run_units(ds, units, modules=['nuc'], params=off, out=tmp_path / 'off', annotate=False)
    assert not [m for m in read_all(tmp_path / 'off' / 'nuc' / 'marks.tsv.bgz') if m['group'] == NUCREG]


def test_cli_flags_fold_into_nuc_parameters():
    from fiberhmm.scan.cli import _nucreg_params, _parser
    p = _parser()

    def fold(argv, modules=('nuc',), params=None):
        params = {} if params is None else params
        _nucreg_params(p, p.parse_args(argv), list(modules), params)
        return params
    assert fold([]) == {}
    got = fold(['--nucreg-window', '1500', '--nucreg-band', '150-250', '--nucreg-mask-open', '300',
                '--nucreg-thin', '25', '--nucreg-signal', 'occupancy', '--nucreg-step', '200'])
    assert RegularityParams.from_dict(got['nuc']['nucreg']) == RegularityParams(
        window_bp=1500, step_bp=200, signal='occupancy', band_bp=(150, 250), mask_open_bp=300, thin_per_kb=25.0)
    assert fold(['--nucreg-mask-open', '0'], params={'nuc': {'nucreg': {'mask_open_bp': 300}}}) == \
        {'nuc': {'nucreg': RegularityParams().to_dict()}}
    assert fold(['--no-nucreg']) == {'nuc': {'nucreg': None}}
    for argv, modules in ((['--no-nucreg', '--nucreg-window', '1500'], ('nuc',)),
                          (['--nucreg-window', '1500'], ('fp',)), (['--nucreg-band', 'x'], ('nuc',))):
        with pytest.raises(SystemExit):
            fold(argv, modules)
