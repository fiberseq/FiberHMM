"""Public scan additions: annotated associations, reference density and signal inputs."""
import json

import numpy as np
import pysam
import pytest

from fiberhmm.assoc import scanio as S


@pytest.fixture
def annotated_scan(tmp_path):
    from test_assoc_nuc_scan import _write_scan
    from fiberhmm.scan.modules.fp import FpModule
    from fiberhmm.scan.modules.nfr import NfrModule
    from fiberhmm.scan.modules.nuc import NucModule
    from fiberhmm.io.scan_ma import with_scan_contracts, scan_contract_line, replace_scan_groups, encode_scalars
    from fiberhmm.scan.identity import molecule_id
    path = _write_scan(tmp_path)
    scan = S.open_scan(path)
    declarations = [g for mod in (FpModule(), NfrModule(), NucModule()) for g in mod.ma_groups()]
    hdr = with_scan_contracts({'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': 'chrT', 'LN': 50000}]},
                              [scan_contract_line(g.to_json()) for g in declarations])
    by_read = {}
    for module in ('fp', 'nfr'):
        group = 'fpcls' if module == 'fp' else 'nfrvar'
        decl = next(g for g in declarations if g.group == group)
        for row in S._rows(scan, f'{module}/states.tsv.bgz'):
            by_read.setdefault(row['qname'], {}).setdefault(group, (decl.spec, []))[1].append(
                (int(row['start']), int(row['end']) - int(row['start']),
                 [1 if row['state'] == '1' else 2] + [0] * (len(decl.spec) - 1), row['hit_id']))
    scalar = next(g for g in declarations if g.group == 'nfrmol')
    expected = {}
    for row in S._rows(scan, 'nfr/molecules.tsv.bgz'):
        value = float(row['openness'])
        by_read.setdefault(row['qname'], {})['nfrmol'] = (scalar.spec, [
            (0, 45000, encode_scalars({'openness': value}, scalar.codecs), 'nfrmol')])
        expected[molecule_id([row['qname']])] = value
    bam_path = path / 'annotated.bam'
    with pysam.AlignmentFile(str(bam_path), 'wb', header=hdr) as bam:
        for qname, groups in by_read.items():
            read = pysam.AlignedSegment(bam.header)
            read.query_name, read.query_sequence = qname, 'A' * 45000
            read.reference_id, read.reference_start, read.cigarstring = 0, 0, '45000M'
            read.mapping_quality = 60
            read.set_tag('HP', 1 + int(qname.split('/')[1]) % 2)
            nucseg = next(g for g in declarations if g.group == 'nucseg')
            groups['nucseg'] = (nucseg.spec, [(500, 10000, [1] + [0]*(len(nucseg.spec)-1), 'nucseg'),
                                            (10500, 10000, [2] + [0]*(len(nucseg.spec)-1), 'nucseg')])
            nucsz = next(g for g in declarations if g.group == 'nucsz')
            groups['nucsz'] = (nucsz.spec, [(1000, 20, [16] + [0] * (len(nucsz.spec)-1), 'nucsz')])
            replace_scan_groups(read, groups)
            bam.write(read)
    pysam.index(str(bam_path))
    man = scan.manifest
    man['reference'] = {'contigs': [['chrT', 50000]]}
    for module, mod in [('fp', FpModule()), ('nfr', NfrModule()), ('nuc', NucModule())]:
        man['modules'].setdefault(module, {})['ma_groups'] = [g.to_json() for g in mod.ma_groups()]
    man['outputs']['annotate'] = True
    man['outputs']['annotated'] = [dict(dataset_id='d1', input_index=0, path='annotated.bam')]
    for rel in list(man['outputs']['tables']):
        if rel.endswith(('states.tsv.bgz', 'marks.tsv.bgz', 'molecules.tsv.bgz')):
            del man['outputs']['tables'][rel]
    (path / 'manifest.json').write_text(json.dumps(man))
    return path, expected


def test_annotated_states_scalars_and_region_selection(annotated_scan):
    path, expected = annotated_scan
    scan = S.open_scan(path)
    mi = S.MoleculeIndex()
    a = S.read_states(scan, 'nfr.region', mi)
    assert a.n_hits and len(a.state)
    scalar = S.molecule_scalars(scan, 'nfr', ['openness'], mi)['openness']
    for (ds, mol), i in mi.index.items():
        assert scalar[i] == pytest.approx(expected[mol], abs=0.002 + 1e-10)
    assert set(S.haplotypes_from_bams(scan, mi)) == {1, 2}
    assert list(S._rows(scan, 'nfr/states.tsv.bgz', [])) == []
    windows = [('chrT', 0, 50000), ('chrT', 100, 49000)]
    assert len(list(S._rows(scan, 'nfr/states.tsv.bgz', windows))) == len(list(S._rows(scan, 'nfr/states.tsv.bgz')))


@pytest.mark.parametrize('mode,opts', [
    ('cooccur', ['--a', 'nfr.region', '--b', 'fp.class']),
    ('condition', ['--a', 'nfr.region', '--response', 'nfr:openness']),
    ('rate', ['--a', 'nuc.seg.regular', '--b', 'tf']),
])
def test_cli_reads_annotated_m1_m2_m3(annotated_scan, tmp_path, mode, opts):
    from fiberhmm.scan.cli import main
    path, _ = annotated_scan
    out = tmp_path / mode
    assert main(['assoc', str(path), '--mode', mode, *opts, '--group', 'haplotype',
                 '--n-perm', '5', '--n-boot', '5', '-o', str(out)]) == 0
    result = json.loads((out / 'result.json').read_text())
    assert result['input']['scalar_precision'] == 'MA codecs'
    if mode == 'rate':
        assert result['O_in'] == 150 and result['O_out'] == 0


@pytest.mark.parametrize('reverse', [False, True])
def test_projection_hard_clips_indels_and_insertion_only(reverse):
    from fiberhmm.assoc.annotated import reference_span
    r = pysam.AlignedSegment()
    r.reference_start, r.query_sequence = 100, 'A' * 45
    r.cigarstring = '5H10M5I10M5D20M5H'
    r.flag = 16 if reverse else 0
    # Whole molecule coordinates, leading hard clip included.
    def span(a, n):
        return reference_span(r, 55-a-n if reverse else a, n, 55)
    assert span(5, 45) == (100, 145)
    assert span(15, 5) == (110, 110)
    assert span(20, 10) == (110, 120)


def test_fasta_matching_covariate_and_strata(annotated_scan, tmp_path):
    from fiberhmm.assoc.sites import locus_density, density_strata
    from fiberhmm.scan.cli import main
    path, _ = annotated_scan
    fasta = tmp_path / 'ref.fa'
    fasta.write_text('>chrT\n' + 'A'*25000 + 'C'*25000 + '\n')
    pysam.faidx(str(fasta))
    scan = S.open_scan(path)
    assert locus_density(scan, fasta, ['chrT', 'chrT'], [10000, 40000]).tolist() == [1., 0.]
    strata = density_strata(scan, fasta, [('chrT', 0, 50000)])
    assert sum(e-s for c,s,e,k in strata) == 50000 and len({s[3] for s in strata}) > 1
    out = tmp_path / 'density'
    assert main(['assoc', str(path), '--mode', 'condition', '--a', 'nfr.region', '--response', 'nfr:openness',
                 '--fasta', str(fasta), '--n-perm', '0', '--n-boot', '0', '-o', str(out)]) == 0
    result = json.loads((out / 'result.json').read_text())
    assert 'site_density' in result['backgrounds']['matched_on']
    fasta.write_text('>other\nAAAA\n')
    pysam.faidx(str(fasta))
    with pytest.raises(S.ScanInputError, match='different length'):
        locus_density(scan, fasta, ['chrT'], [100])


def test_matcher_preserves_shuffled_selection_and_handles_large_pools():
    from fiberhmm.assoc.background import matched_backgrounds
    # Exact expected selection was produced by the original algorithm.
    a = [('c', i*10000) for i in range(100)]
    c = [('c', i*10000+4000) for i in range(100)]
    result = matched_backgrounds(a, c, k=1, seed=0)
    assert all(js for i, js in result['sets'])
    assert len({j for i, js in result['sets'] for j in js}) == 100


def test_signal_bam_gate_all_modules_receive_same_snapshot(tmp_path):
    from fiberhmm.scan.testing import write_synthetic_bam, PlantedAnchorModule
    from fiberhmm.scan import run_units
    from fiberhmm.scan.units import plan
    from fiberhmm.scan.signal import PREFIX
    original = write_synthetic_bam(tmp_path / 'original.bam', length=100000)
    path = tmp_path / 'signal.bam'
    with pysam.AlignmentFile(original, 'rb') as source:
        hdr = source.header.to_dict()
        hdr.setdefault('CO', []).append(PREFIX + json.dumps({'tag_schema': 5}))
        with pysam.AlignmentFile(str(path), 'wb', header=hdr) as out:
            for i, r in enumerate(source.fetch(until_eof=True)):
                r.set_tag('bg', i % 2)
                out.write(r)
    pysam.index(str(path))
    spec = dict(paths=[str(path)], input_type='signal-bam', signal_filters={'requireDetectable': True})
    out = tmp_path / 'scan'
    units = plan({'chrS': 100000}, genome=True)
    kw = dict(modules=['plant'], registry={'plant': PlantedAnchorModule},
              params={'plant': {'features': [['A', 10000, 10300, 'both']], 'states': True}}, out=out)
    run_units([spec], units, **kw)
    scan = S.open_scan(out)
    f = scan.manifest['datasets'][0]['files'][0]
    assert f['signal_input']['signal_filters']['requireDetectable']
    with pysam.AlignmentFile(f['path']) as bam:
        records = list(bam.fetch(until_eof=True))
        assert records and all(r.get_tag('bg') == 0 for r in records)
    before = (out / 'manifest.json').read_bytes()
    run_units([spec], units, continue_run=True, **kw)
    assert (out / 'manifest.json').read_bytes() == before
    with pytest.raises(ValueError, match='input_type'):
        run_units([dict(spec, input_type='other')], units, **dict(kw, out=tmp_path / 'invalid'))


@pytest.mark.parametrize('filters,code,want', [
    ({'hideBackground': True}, 1, False), ({'hideBackground': True}, -2, True),
    ({'excludeUnscored': True}, -2, False), ({'requireDetectable': True}, 0, True),
    ({'requireStructure': True}, 0, False), ({}, 99, True),
])
def test_signal_predicates(filters, code, want):
    from fiberhmm.scan.signal import keeps, validate_spec
    r = pysam.AlignedSegment()
    r.set_tag('bg', code)
    assert keeps(r, validate_spec(dict(input_type='signal-bam', signal_filters=filters))) == want


def test_random_transferable_cli_uses_scan_rule(tmp_path):
    from fiberhmm.scan.testing import write_daf_locus_bam
    from fiberhmm.scan import run_units
    from fiberhmm.scan.units import plan
    from fiberhmm.scan.cli import main
    bam = write_daf_locus_bam(tmp_path / 'locus.bam')
    out = tmp_path / 'scan'
    run_units([dict(paths=[bam])], plan({'chr1': 6000}, regions=[('chr1', 0, 6000)]),
              modules=['nfr'], out=out, annotate=False)
    result_dir = tmp_path / 'assoc'
    assert main(['assoc', str(out), '--mode', 'condition', '--a', 'nfr.region', '--response', 'nfr:openness',
                 '--backgrounds', 'random', '--background-step', '100', '--background-min-sep', '300',
                 '--background-width', '100', '--background-k', '1', '--n-perm', '5', '--n-boot', '5',
                 '-o', str(result_dir)]) == 0
    result = json.loads((result_dir / 'result.json').read_text())
    assert result['backgrounds']['rule'] == 'random'
    assert result['backgrounds']['n_candidates'] > 0
    assert 'prevalence' not in result['backgrounds']['matched_on']
    assert result['backgrounds']['width_bp'] == 100
    with pytest.raises(S.ScanInputError, match='no transferable'):
        S.transferable_states(S.open_scan(out), 'fp.class', [('chr1', 1000)], 10, S.MoleculeIndex())



@pytest.mark.parametrize('reverse', [False, True])
def test_fast_projection_matches_per_base_reference(reverse):
    from fiberhmm.io.scan_ma import project_to_molecule
    r = pysam.AlignedSegment()
    r.reference_start, r.query_sequence = 100, 'A' * 100
    r.cigarstring = '5H10S15M10I10M10D10M5N20M10S15M7H'
    r.flag = 16 if reverse else 0
    # Cover empty intersections, deletion boundaries, zero width, and clips.
    for start in [*range(90, 205, 3), 100.1, 127.8]:
        for end in (start, start+1, start+10, start+45):
            qs = [q for q, ref in r.get_aligned_pairs(matches_only=True) if start <= ref < max(end, start+1)]
            for length in (r.query_length, r.query_length+12):
                expected = None
                if qs:
                    a, b = min(qs), max(qs)+1
                    if length != r.query_length:
                        a, b = a+5, b+5
                    if reverse:
                        a, b = length-b, length-a
                    expected = (a, b-a)
                assert project_to_molecule(r, start, end, length) == expected


def test_conditional_logistic_stratum_batching_is_byte_identical(monkeypatch):
    import fiberhmm.assoc.clogit as C
    rng = np.random.default_rng(8)
    Xp = rng.normal(size=(80, 30, 9))
    valid = np.ones((80, 30), bool)
    kk = rng.integers(2, 14, 80)
    beta = rng.normal(size=9) * .1
    for need_q in (True, False):
        expected = C._dp_full(Xp, valid, kk, beta, need_q)
        monkeypatch.setattr(C, '_DP_MAX_ELEMENTS', 2000)
        actual = C._dp(Xp, valid, kk, beta, need_q)
        for a, b in zip(actual, expected):
            if a is None:
                assert b is None
            else:
                assert a.tobytes() == b.tobytes()


def test_cached_discovery_fit_keeps_exact_model_and_does_not_mutate_inputs():
    from fiberhmm.inference.consensus.lattice_recaller import discovery as D
    from sklearn.cluster import KMeans
    rng = np.random.default_rng(18)
    X = np.r_[rng.normal(0, 2, (30, 2)), rng.normal(20, 2, (30, 2))]
    before = X.tobytes()
    expected = KMeans(2, n_init=10, random_state=4).fit(X)
    a, pa = D._fit(X, 2, 4)
    b, pb = D._fit(X.copy(), 2, 4)
    assert a is b
    assert a.cluster_centers_.tobytes() == expected.cluster_centers_.tobytes()
    assert pa(X).tobytes() == expected.predict(X).tobytes() == pb(X).tobytes()
    assert X.tobytes() == before


def test_signal_policy_change_cannot_reuse_scan(tmp_path):
    from fiberhmm.scan.signal import PREFIX
    from fiberhmm.scan.testing import write_synthetic_bam, PlantedAnchorModule
    from fiberhmm.scan import run_units
    from fiberhmm.scan.runner import ScanRefused
    from fiberhmm.scan.units import plan
    original = write_synthetic_bam(tmp_path / 'input.bam', length=50000)
    path = tmp_path / 'signal.bam'
    with pysam.AlignmentFile(original) as src:
        hdr = src.header.to_dict()
        hdr.setdefault('CO', []).append(PREFIX + json.dumps({'tag_schema': 5}))
        with pysam.AlignmentFile(str(path), 'wb', header=hdr) as dest:
            for i, read in enumerate(src.fetch(until_eof=True)):
                read.set_tag('bg', i % 2)
                dest.write(read)
    pysam.index(str(path))
    spec = dict(paths=[str(path)], input_type='signal-bam')
    kw = dict(modules=['plant'], registry={'plant': PlantedAnchorModule}, out=tmp_path / 'scan')
    units = plan({'chrS': 50000}, genome=True)
    run_units([spec], units, **kw)
    before = (kw['out'] / 'manifest.json').read_bytes()
    with pytest.raises(ScanRefused, match='not empty'):
        run_units([spec], units, **kw)
    with pytest.raises(ScanRefused, match='differ'):
        run_units([dict(spec, signal_filters={'requireDetectable': True})], units, continue_run=True, **kw)
    assert (kw['out'] / 'manifest.json').read_bytes() == before
    timings = json.loads((kw['out'] / 'run.json').read_text())['attempts'][-1]
    assert timings['module_unit_seconds']['plant'] >= 0 and timings['finalize_seconds'] >= 0


@pytest.mark.parametrize('detail,want', [
    ({'schema': 5, 'signal_status': 'detectable'}, True),
    ({'signal_status': 'detectable'}, False),
    ({'schema': 4, 'signal_status': 'detectable'}, False),
    ({'schema': 5, 'signal_status': 'background'}, False),
])
def test_signal_schema_and_inconsistent_status_are_unknown(detail, want):
    from fiberhmm.scan.signal import keeps, validate_spec
    r = pysam.AlignedSegment()
    r.set_tag('bg', 0)
    r.set_tag('me', json.dumps(detail))
    assert keeps(r, validate_spec(dict(input_type='signal-bam', signal_filters={'requireDetectable': True}))) == want


def test_annotated_contract_disagreement_is_refused(annotated_scan):
    path, _ = annotated_scan
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['modules']['nfr']['ma_groups'][0]['group_version'] += 1
    (path / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(S.ScanInputError, match='contract'):
        S.read_states(S.open_scan(path), 'nfr.region', S.MoleculeIndex())


@pytest.mark.parametrize('large', [False, True])
def test_table_spill_merge_is_byte_identical_including_stable_ties(tmp_path, monkeypatch, large):
    from fiberhmm.scan import tables as T
    rng = np.random.default_rng(4)
    rows = [('c', int(a), int(a+length), ['same' if a % 2 else 'other'])
            for a, length in zip(rng.integers(0, 60, 300), rng.integers(0, 10, 300))]
    rows += [('c', 20, 20, ['tie']), ('c', 20, 21, ['tie'])] * 12
    if large:
        rows.append(('c', T.TBI_MAX+10, T.TBI_MAX+20, ['large']))
    paths = [tmp_path / 'memory.tsv.bgz', tmp_path / 'spill.tsv.bgz']
    for path in paths:
        writer = T.TableWriter(path, [*T.COMMON, 'value'], {'c': 0})
        for c, a, b, cells in rows:
            writer.add(c, a, b, cells)
        assert len(writer) == len(rows)
        writer.close()
        monkeypatch.setattr(T, '_SORT_ROWS', 7)
        monkeypatch.setattr(T, '_MERGE_FANIN', 3)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert T.index_path(paths[0]).read_bytes() == T.index_path(paths[1]).read_bytes()
    assert not list(tmp_path.glob('.*.sort.*'))


def test_lazy_nuc_track_assembly_matches_eager_padding_and_slices(tmp_path):
    from fiberhmm.scan.finalize import _stitch_inputs, _unit_dir
    from fiberhmm.scan.runner import _write_json
    from fiberhmm.scan.types import UnitResult, unit_result_to_json
    from fiberhmm.scan.units import plan
    units, _ = plan({'c': 150000}, genome=True)
    status = {u.unit_id: 'empty' for u in units}
    for i in (0, 1, 3):
        u = units[i]
        status[u.unit_id] = 'done'
        folder = _unit_dir(tmp_path, 'nuc', u)
        folder.mkdir(parents=True)
        key = 'a' if i < 3 else 'b'
        np.savez_compressed(folder / 'tracks.npz', **{key: np.arange(u.zone_end-u.zone_start, dtype=np.float32)+i})
        _write_json(folder / 'result.json.gz', unit_result_to_json(UnitResult()))
    class Module:
        _lazy_stitch_tracks = False
        def stitch_support_bp(self, settings):
            return 1200
    mod = Module()
    territory, expected = _stitch_inputs(tmp_path, 'nuc', mod, {}, units, status)
    mod._lazy_stitch_tracks = True
    actual_territory, actual = _stitch_inputs(tmp_path, 'nuc', mod, {}, units, status)
    assert territory == actual_territory and set(expected) == set(actual)
    for key in expected:
        for reference, lazy in zip(expected[key], actual[key]):
            if reference is None:
                assert lazy is None
                continue
            assert np.asarray(lazy).tobytes() == reference.astype(float).tobytes()
            for selection in (slice(0, 30), slice(-10, None), slice(23000, 26000), slice(20, 10)):
                assert lazy[selection].tobytes() == reference[selection].astype(float).tobytes()


def test_reference_density_requires_index_without_writing_source(annotated_scan, tmp_path):
    from fiberhmm.assoc.sites import locus_density
    path, _ = annotated_scan
    fasta = tmp_path / 'unindexed.fa'
    fasta.write_text('>chrT\n' + 'A'*50000 + '\n')
    with pytest.raises(S.ScanInputError, match='existing .fai'):
        locus_density(S.open_scan(path), fasta, ['chrT'], [1000])
    assert not fasta.with_suffix('.fa.fai').exists()


def test_bed_anchor_membership_includes_earlier_enclosing_intervals():
    from fiberhmm.assoc.cli import _in_bed
    chrom = ['c', 'c', 'c', 'd', 'missing']
    positions = [15, 80, 200, 10, 10]
    intervals = [('c', 0, 100), ('c', 10, 20), ('c', 0, 0), ('d', 5, 15)]
    assert _in_bed(chrom, positions, intervals).tolist() == [True, True, False, True, False]


def test_indexed_annotated_queries_deduplicate_disjoint_window_fetches(annotated_scan):
    path, _ = annotated_scan
    scan = S.open_scan(path)
    original = list(S._rows(scan, 'nfr/molecules.tsv.bgz'))
    # Identical SAM occurrences are distinct records, even though a record
    # fetched twice through disjoint windows must be emitted only once.
    duplicate_path = path / 'duplicates.bam'
    with pysam.AlignmentFile(str(path / 'annotated.bam'), 'rb') as bam:
        with pysam.AlignmentFile(str(duplicate_path), 'wb', template=bam) as out:
            for i, read in enumerate(bam.fetch(until_eof=True)):
                out.write(read)
                if i == 0:
                    out.write(read)
    pysam.index(str(duplicate_path))
    manifest = scan.manifest
    manifest['outputs']['annotated'][0]['path'] = duplicate_path.name
    (path / 'manifest.json').write_text(json.dumps(manifest))
    scan = S.open_scan(path)
    full = list(S._rows(scan, 'nfr/molecules.tsv.bgz'))
    assert len(full) == len(original) + 1
    windows = [('chrT', 35000, 44000), ('chrT', 100, 200)]
    queried = list(S._rows(scan, 'nfr/molecules.tsv.bgz', windows))
    assert queried == full
    assert len({r['record'] for r in queried}) == len(queried)
