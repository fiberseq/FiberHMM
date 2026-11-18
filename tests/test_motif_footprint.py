"""fiberhmm-footprint: motif-anchored footprint scoring from FiberHMM calls."""
from __future__ import annotations

import array

import pysam
import pytest

from fiberhmm.inference import motif_footprint as mf
from fiberhmm.io.ma_tags import ambiguity_to_edge, parse_an_tag, parse_aq_array, parse_ma_tag

L = 1000            # read length
POS = 1000          # alignment start: query i <-> reference POS + i
HEADER = {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': 'chr1', 'LN': 10000}],
          'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '3.0.6',
                  'DS': 'FiberHMM call; coord=molecular'}]}


def ma_tag(nucs, msps, tfs, reverse):
    """MA/AQ in molecular frame from SEQ-frame intervals.

    tfs: [(start, end, tq, amb_left, amb_right)] (SEQ frame, ambiguities in bp)."""
    def mol(s, e):
        return (L - e, e - s) if reverse else (s, e - s)
    parts, aq = [str(L)], []
    if nucs:
        items = sorted(mol(s, e) for s, e in nucs)
        parts.append('nuc.Q:' + ','.join(f'{s + 1}-{n}' for s, n in items))
        aq += [200] * len(items)
    if msps:
        parts.append('msp.:' + ','.join(f'{s + 1}-{n}' for s, n in sorted(mol(s, e) for s, e in msps)))
    if tfs:
        items = []
        for s, e, tq, al, ar in tfs:
            el, er = ambiguity_to_edge(al), ambiguity_to_edge(ar)
            if reverse:
                el, er = er, el
            items.append((*mol(s, e), tq, el, er))
        items.sort()
        parts.append('tf.QQQ:' + ','.join(f'{s + 1}-{n}' for s, n, *_ in items))
        for *_, tq, el, er in items:
            aq += [tq, el, er]
    return ';'.join(parts), aq


def make_read(header, name, *, nucs=(), msps=(), tfs=(), reverse=False, pos=POS, length=L, m6a=(), annotate=True):
    r = pysam.AlignedSegment(header)
    r.query_name = name
    r.query_sequence = 'A' * length
    r.flag = 16 if reverse else 0
    r.reference_id = 0
    r.reference_start = pos
    r.mapping_quality = 60
    r.cigar = [(0, length)]
    r.query_qualities = pysam.qualitystring_to_array('I' * length)
    if annotate:
        ma, aq = ma_tag(nucs, msps, tfs, reverse)
        r.set_tag('MA', ma, 'Z')
        if aq:
            r.set_tag('AQ', array.array('B', aq))
    if m6a:
        # MM is in original-read orientation; with an all-A SEQ the reverse read's
        # original strand is all T, so use the A+a call on the SEQ strand (T-a) only for forward reads.
        assert not reverse
        pos_sorted = sorted(m6a)
        deltas = [pos_sorted[0]] + [b - a - 1 for a, b in zip(pos_sorted, pos_sorted[1:])]
        r.set_tag('MM', 'A+a,' + ','.join(map(str, deltas)) + ';', 'Z')
        r.set_tag('ML', array.array('B', [250] * len(pos_sorted)))
    return r


def write_bam(path, reads):
    header = pysam.AlignmentHeader.from_dict(HEADER)
    built = [f(header) for f in reads]
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        for r in sorted(built, key=lambda r: r.reference_start):
            out.write(r)
    pysam.index(str(path))
    return path


def bed(tmp_path, rows, name='motifs.bed'):
    p = tmp_path / name
    p.write_text(''.join('\t'.join(map(str, r)) + '\n' for r in rows))
    return p


def run(bam, motifs, **opts):
    options = mf.FootprintOptions(**opts)
    return mf.run_footprint(bam, motifs, options), options


def one(by_motif, motif_index=0, read='r1'):
    return next(o for o in by_motif[motif_index] if o.read_name == read)


# A call with conservative SEQ bounds [200, 230), ambiguity 5 bp left and 3 bp right:
# outer (exclusive of the terminating hits) = [195, 233) -> reference [1195, 1233).
CALL = (200, 230, 80, 5, 3)
MSP = (100, 400)


def basic(reverse=False, tfs=(CALL,), msps=(MSP,), **kw):
    return lambda h: make_read(h, 'r1', msps=msps, tfs=tfs, reverse=reverse, **kw)


def motif(start, end, strand='+', name='CTCF', index=0):
    return mf.Motif(index, 'chr1', POS + start, POS + end, strand, name, mf.sanitize_label(name))


@pytest.mark.parametrize('reverse', [False, True])
def test_outer_vs_conservative_edges(tmp_path, reverse):
    bam = write_bam(tmp_path / 'a.bam', [basic(reverse)])
    m = [motif(196, 232)]             # inside outer [195,233), not inside conservative [200,230)
    o = one(run(bam, m)[0])
    assert (o.code, o.status, o.edge) == (3, mf.FOOTPRINTED, 'outer')
    assert o.call[:4] == (POS + 200, POS + 230, POS + 195, POS + 233)
    o = one(run(bam, m, edges='conservative')[0])
    assert (o.code, o.status, o.reason) == (1, mf.NOT_FOOTPRINTED, 'partial_overlap')
    o = one(run(bam, [motif(202, 228)], edges='conservative')[0])
    assert (o.code, o.edge) == (3, 'conservative')


@pytest.mark.parametrize('reverse', [False, True])
def test_outer_bound_exclusive_vs_inclusive(tmp_path, reverse):
    bam = write_bam(tmp_path / 'a.bam', [basic(reverse)])
    # [195, 234) needs the terminating position 233 on the right
    assert one(run(bam, [motif(195, 234)])[0]).code == 1
    assert one(run(bam, [motif(195, 234)], outer_bound='inclusive')[0]).code == 3
    # [194, 233) needs the terminating position 194 on the left
    assert one(run(bam, [motif(194, 233)])[0]).code == 1
    assert one(run(bam, [motif(194, 233)], outer_bound='inclusive')[0]).code == 3


def test_width_cap(tmp_path):
    wide = (180, 260, 90, 0, 0)       # 80 bp
    bam = write_bam(tmp_path / 'a.bam', [basic(tfs=(wide,))])
    m = [motif(200, 220)]
    o = one(run(bam, m, max_width=68)[0])
    assert (o.code, o.reason) == (1, 'too_wide')
    assert one(run(bam, m, max_width=80)[0]).code == 3
    assert one(run(bam, m, max_width=0)[0]).code == 3
    assert mf.FootprintOptions().max_width == mf.DEFAULT_MAX_WIDTH == 74


def test_min_width(tmp_path):
    bam = write_bam(tmp_path / 'a.bam', [basic(tfs=((200, 225, 90, 0, 0), (180, 260, 90, 0, 0)))])   # 25 and 80 bp
    m = [motif(205, 220)]
    assert one(run(bam, m, max_width=0, min_width=30)[0]).call[:2] == (POS + 180, POS + 260)
    o = one(run(bam, m, max_width=74, min_width=30)[0])
    assert (o.code, o.reason) == (1, 'too_wide')
    o = one(run(bam, m, max_width=0, min_width=90)[0])
    assert (o.code, o.reason) == (1, 'too_narrow') and mf.REASON_CODE['too_narrow'] == 5
    assert one(run(bam, m, max_width=25, min_width=25)[0]).code == 3       # bounds are inclusive
    with pytest.raises(ValueError):
        mf.FootprintOptions(min_width=80, max_width=74)


def test_narrowest_qualifying_call_is_reported(tmp_path):
    bam = write_bam(tmp_path / 'a.bam', [basic(tfs=((150, 300, 120, 0, 0), (195, 225, 60, 0, 0)))])
    o = one(run(bam, [motif(200, 220)], max_width=0)[0])
    assert o.call[:2] == (POS + 195, POS + 225) and o.call[4] == 60


def test_min_llr(tmp_path):
    bam = write_bam(tmp_path / 'a.bam', [basic(tfs=((195, 225, 49, 0, 0),))])
    assert one(run(bam, [motif(200, 220)])[0]).reason == 'no_call'
    assert one(run(bam, [motif(200, 220)], min_llr=4.9)[0]).code == 3


def test_motif_at_msp_edge_uses_ft_spans_rule(tmp_path):
    # ft: MSP start < motif start and MSP end >= motif end (reference)
    bam = write_bam(tmp_path / 'a.bam', [basic(tfs=((95, 140, 80, 0, 0), (370, 405, 80, 0, 0)))])
    at_start = one(run(bam, [motif(100, 120)])[0])
    assert (at_start.code, at_start.status) == (2, mf.NO_MSP)      # footprinted but not eligible
    assert one(run(bam, [motif(101, 121)])[0]).code == 3
    assert one(run(bam, [motif(380, 400)])[0]).code == 3            # MSP end == motif end
    assert one(run(bam, [motif(381, 401)])[0]).code == 2


def test_not_spanning_and_unannotated(tmp_path):
    bam = write_bam(tmp_path / 'a.bam', [
        basic(),
        lambda h: make_read(h, 'r2', msps=[(0, 300)], pos=POS + 205),   # starts inside the motif
        lambda h: make_read(h, 'r3', annotate=False)])
    by = run(bam, [motif(200, 220)])[0]
    assert one(by, read='r2').code == mf.SPAN_NOT and one(by, read='r2').status == mf.NOT_SPANNING
    assert one(by, read='r3').status == mf.UNANNOTATED and one(by, read='r3').code == 0
    row = mf.motif_row(motif(200, 220), by[0])
    cols = dict(zip(mf.TABLE_COLUMNS, row))
    assert (cols['n_spanning_fibers'], cols['n_spanning_msps'], cols['n_footprinted'], cols['n_not_spanning']) == \
        ('2', '1', '1', '1')
    assert cols['fraction_footprinted'] == '1'


def test_strand_and_read_orientation_agree(tmp_path):
    fwd = write_bam(tmp_path / 'f.bam', [basic(False, nucs=[(500, 650)])])
    rev = write_bam(tmp_path / 'r.bam', [basic(True, nucs=[(500, 650)])])
    for strand in '+-':
        ms = [motif(196, 232, strand), motif(520, 540, strand, index=1)]
        a, b = run(fwd, ms)[0], run(rev, ms)[0]
        for i in (0, 1):
            oa, ob = one(a, i), one(b, i)
            assert (oa.code, oa.status, oa.call, oa.nuc_overlap) == (ob.code, ob.status, ob.call, ob.nuc_overlap)
        assert one(a, 1).nuc_overlap and one(a, 1).code == 0      # inside a nucleosome: no MSP


def test_require_unmodified(tmp_path):
    bam = write_bam(tmp_path / 'a.bam', [basic(m6a=(210,))])
    assert one(run(bam, [motif(200, 220)])[0]).code == 3
    o = one(run(bam, [motif(200, 220)], require_unmodified=True)[0])
    assert (o.code, o.reason) == (1, 'modified')
    assert one(run(bam, [motif(212, 222)], require_unmodified=True)[0]).code == 3


def test_parallel_matches_serial(tmp_path):
    bam = write_bam(tmp_path / 'a.bam', [basic(), lambda h: make_read(h, 'r2', msps=[MSP], tfs=[CALL], pos=POS + 5000)])
    ms = [motif(196, 232), motif(5196, 5232, index=1), motif(300, 320, index=2)]
    s = mf.run_footprint(bam, ms, mf.FootprintOptions())
    p = mf.run_footprint(bam, ms, mf.FootprintOptions(), cores=2)
    key = lambda by: {i: sorted((o.read_name, o.code, o.call) for o in v) for i, v in by.items()}
    assert key(s) == key(p)


def test_sanitize_and_bed_grouping(tmp_path):
    assert mf.sanitize_label('CTCF') == 'CTCF'
    assert mf.sanitize_label('MA0139.1 CTCF+core-1') == 'MA0139_1_CTCF_core_1'
    assert mf.sanitize_label('fp_REST') == 'REST'
    assert mf.sanitize_label('...') == 'motif'
    p = bed(tmp_path, [('chr1', 10, 20, 'CTCF', 0, '+'), ('chr1', 30, 40, 'REST.v2', 0, '-'), ('chr1', 50, 60)])
    ms = mf.load_motif_bed(p)
    assert [(m.label, m.strand) for m in ms] == [('CTCF', '+'), ('REST_v2', '-'), ('motif', '+')]
    assert {m.label for m in mf.load_motif_bed(p, layer_name='set A')} == {'set_A'}
    with pytest.raises(ValueError):
        mf.load_motif_bed(bed(tmp_path, [('chr1', 20, 10)], 'bad.bed'))


# ------------------------------------------------------------------ MA output

def layers_of(read):
    p = parse_ma_tag(read.get_tag('MA'))
    q = parse_aq_array(read.get_tag('AQ') if read.has_tag('AQ') else None,
                       [t[2] for t in p['raw_types']], [len(t[3]) for t in p['raw_types']])
    names = parse_an_tag(read.get_tag('AN')) if read.has_tag('AN') else [''] * len(q)
    out, k = {}, 0
    for name, strand, spec, ivs in p['raw_types']:
        for s, n in ivs:
            out.setdefault(name, []).append((s, n, q[k], names[k]))
            k += 1
    return out


def footprint_bam(tmp_path, src, rows, out, layer_name=None, **opts):
    motifs = mf.load_motif_bed(bed(tmp_path, rows, f'{out}.bed'), layer_name)
    options = mf.FootprintOptions(**opts)
    by = mf.run_footprint(src, motifs, options, want_spans=True)
    return mf.write_annotated_bam(src, tmp_path / out, motifs, by, options, 'x.bed')


def reads_by_name(path):
    with pysam.AlignmentFile(str(path)) as bam:
        return {r.query_name: r for r in bam.fetch(until_eof=True)}, bam.header.to_dict()


@pytest.mark.parametrize('reverse', [False, True])
def test_ma_annotation_round_trip_and_stacking(tmp_path, reverse):
    src = write_bam(tmp_path / 'src.bam', [
        lambda h: make_read(h, 'r1', nucs=[(500, 650)], msps=[MSP], tfs=[CALL, (300, 330, 70, 0, 0)], reverse=reverse)])
    orig, _ = reads_by_name(src)
    ctcf = [('chr1', POS + 196, POS + 232, 'CTCF', 0, '+'), ('chr1', POS + 340, POS + 360, 'CTCF', 0, '-')]
    rest = [('chr1', POS + 302, POS + 328, 'REST', 0, '+')]
    footprint_bam(tmp_path, src, ctcf, 'one.bam')
    footprint_bam(tmp_path, tmp_path / 'one.bam', rest, 'two.bam')
    reads, header = reads_by_name(tmp_path / 'two.bam')
    r = reads['r1']
    lay = layers_of(r)
    assert set(lay) == {'nuc', 'msp', 'tf', 'fp_CTCF', 'fpneg_CTCF', 'fp_REST'}
    # original sections and AQ bytes are byte-identical
    o = orig['r1']
    assert r.get_tag('MA').startswith(o.get_tag('MA') + ';')
    assert list(r.get_tag('AQ'))[:len(o.get_tag('AQ'))] == list(o.get_tag('AQ'))
    # round trip: interval maps back to the motif's reference coordinates
    def ref_span(s, n):
        if reverse:
            s = L - s - n
        return POS + s, POS + s + n
    (s, n, q, name), = lay['fp_CTCF']
    assert ref_span(s, n) == (POS + 196, POS + 232) and name == f'chr1:{POS + 196}-{POS + 232}:+'
    assert q == [80, 30, 2]                                    # tq, conservative width, outer-only edge flag
    (s, n, q, name), = lay['fpneg_CTCF']
    assert ref_span(s, n) == (POS + 340, POS + 360) and q == [mf.REASON_CODE['no_call']]
    (s, n, q, _), = lay['fp_REST']
    assert ref_span(s, n) == (POS + 302, POS + 328) and q == [70, 30, 1]
    co = '\n'.join(header['CO'])
    assert 'fp_CTCF' in co and 'fp_REST' in co and mf.CONTRACT in co


def test_rerun_replaces_layer(tmp_path):
    src = write_bam(tmp_path / 'src.bam', [basic(nucs=[(500, 650)])])
    footprint_bam(tmp_path, src, [('chr1', POS + 196, POS + 232, 'CTCF', 0, '+')], 'one.bam')
    footprint_bam(tmp_path, tmp_path / 'one.bam', [('chr1', POS + 202, POS + 228, 'CTCF', 0, '+')], 'two.bam',
                  edges='conservative')
    reads, header = reads_by_name(tmp_path / 'two.bam')
    lay = layers_of(reads['r1'])
    assert len(lay['fp_CTCF']) == 1 and lay['fp_CTCF'][0][2][2] == 1    # replaced: conservative edge flag
    assert reads['r1'].get_tag('MA').count('fp_CTCF.') == 1
    assert sum(c.startswith(mf.CONTRACT) for c in header['CO']) == 1
    # a re-run whose motifs no longer footprint the read removes the stale layer
    footprint_bam(tmp_path, tmp_path / 'two.bam', [('chr1', POS + 600, POS + 620, 'CTCF', 0, '+')], 'three.bam')
    lay = layers_of(reads_by_name(tmp_path / 'three.bam')[0]['r1'])
    assert 'fp_CTCF' not in lay and set(lay) == {'nuc', 'msp', 'tf'}
    orig = reads_by_name(src)[0]['r1']
    three = reads_by_name(tmp_path / 'three.bam')[0]['r1']
    assert three.get_tag('MA') == orig.get_tag('MA') and list(three.get_tag('AQ')) == list(orig.get_tag('AQ'))


def test_layer_name_override_and_header_types(tmp_path):
    src = write_bam(tmp_path / 'src.bam', [basic()])
    footprint_bam(tmp_path, src, [('chr1', POS + 196, POS + 232, 'CTCF', 0, '+')], 'o.bam', layer_name='my set')
    reads, header = reads_by_name(tmp_path / 'o.bam')
    assert 'fp_my_set' in layers_of(reads['r1'])
    from fiberhmm.io.bam_header import declared_ma_types
    assert {'fp_my_set', 'fpneg_my_set'} <= set(declared_ma_types(pysam.AlignmentHeader.from_dict(header)))


def test_cli_writes_tables(tmp_path):
    from fiberhmm.cli.motif_footprint import main
    src = write_bam(tmp_path / 'src.bam', [basic()])
    b = bed(tmp_path, [('chr1', POS + 196, POS + 232, 'CTCF', 0, '+')])
    main(['--bam', str(src), '--bed', str(b), '-o', str(tmp_path / 't.tsv'),
          '--per-molecule', str(tmp_path / 'm.tsv'), '--out-bam', str(tmp_path / 'o.bam')])
    t = (tmp_path / 't.tsv').read_text().splitlines()
    assert t[0].split('\t') == mf.TABLE_COLUMNS
    assert t[1].split('\t')[6:10] == ['1', '1', '1', '1']
    m = (tmp_path / 'm.tsv').read_text().splitlines()
    assert m[0].split('\t') == mf.MOLECULE_COLUMNS and m[1].split('\t')[7:9] == ['footprinted', '3']
    with pytest.raises(SystemExit):
        main(['--bam', str(src), '--bed', str(b), '--max-width', '-1'])


# ------------------------------------------------------------------ fiberhmm-utils ma-list / ma-drop

def _stacked(tmp_path):
    src = write_bam(tmp_path / 'src.bam', [
        basic(nucs=[(500, 650)]),
        lambda h: make_read(h, 'r2', nucs=[(500, 650)], msps=[MSP], tfs=[CALL], reverse=True),
        lambda h: make_read(h, 'r3', annotate=False)])
    footprint_bam(tmp_path, src, [('chr1', POS + 196, POS + 232, 'CTCF', 0, '+'),
                                  ('chr1', POS + 340, POS + 360, 'CTCF', 0, '+')], 'one.bam')
    footprint_bam(tmp_path, tmp_path / 'one.bam', [('chr1', POS + 202, POS + 228, 'REST', 0, '-')], 'two.bam')
    return src, tmp_path / 'two.bam'


def _declared(path):
    from fiberhmm.io.bam_header import declared_ma_types
    with pysam.AlignmentFile(str(path)) as bam:
        return declared_ma_types(bam.header)


def test_ma_list_counts_stacked_layers(tmp_path):
    from fiberhmm.cli.utils import ma_layer_counts
    _, stacked = _stacked(tmp_path)
    counts, records, declared, exhaustive = ma_layer_counts(str(stacked), sample=0)
    assert records == 3 and exhaustive
    assert {k: tuple(v) for k, v in counts.items()} == {
        'nuc.Q': ('nuc', 2, 2), 'msp.': ('msp', 2, 2), 'tf.QQQ': ('tf', 2, 2),
        'fp_CTCF.QQQ': ('fp_CTCF', 2, 2), 'fpneg_CTCF.Q': ('fpneg_CTCF', 2, 2), 'fp_REST.QQQ': ('fp_REST', 2, 2)}
    assert {'fp_CTCF', 'fpneg_CTCF', 'fp_REST', 'fpneg_REST'} <= set(declared)
    counts, records, _, exhaustive = ma_layer_counts(str(stacked), sample=1)
    assert records == 1 and not exhaustive


def _strings(path):
    with pysam.AlignmentFile(str(path)) as bam:
        return {r.query_name: r.to_string() for r in bam.fetch(until_eof=True)}


def test_ma_drop_one_layer_and_glob(tmp_path):
    from fiberhmm.cli.utils import _resolve_drop_patterns, drop_ma_layers_bam
    from fiberhmm.cli.utils import ma_layer_counts
    src, stacked = _stacked(tmp_path)
    _, m = _resolve_drop_patterns(['fp_REST'])
    s = drop_ma_layers_bam(str(stacked), str(tmp_path / 'norest.bam'), m)
    assert s['layers_dropped'] == {'fp_REST': 2}
    names = {v[0] for v in ma_layer_counts(str(tmp_path / 'norest.bam'), sample=0)[0].values()}
    assert names == {'nuc', 'msp', 'tf', 'fp_CTCF', 'fpneg_CTCF'}
    assert 'fp_REST' not in _declared(tmp_path / 'norest.bam') and 'fp_CTCF' in _declared(tmp_path / 'norest.bam')
    _, m = _resolve_drop_patterns(['fp_*,fpneg_*'])
    drop_ma_layers_bam(str(stacked), str(tmp_path / 'nofp.bam'), m)
    # all footprint layers gone: reads are byte-identical to the source, header declares only the core layers
    assert _strings(tmp_path / 'nofp.bam') == _strings(src)
    assert _declared(tmp_path / 'nofp.bam') == _declared(src)


def test_ma_drop_refuses_core_layers(tmp_path):
    from fiberhmm.cli.utils import _resolve_drop_patterns, drop_ma_layers_bam
    src, _ = _stacked(tmp_path)
    _, m = _resolve_drop_patterns(['tf'])
    with pytest.raises(ValueError, match='core layer'):
        drop_ma_layers_bam(str(src), str(tmp_path / 'x.bam'), m)
    _, m = _resolve_drop_patterns(['*'])
    with pytest.raises(ValueError, match='core layer'):
        drop_ma_layers_bam(str(src), str(tmp_path / 'x.bam'), m)
    drop_ma_layers_bam(str(src), str(tmp_path / 'x.bam'), _resolve_drop_patterns(['tf'])[1], force=True)
    assert not any('tf.' in s for s in _strings(tmp_path / 'x.bam').values())
    with pytest.raises(SystemExit):
        _resolve_drop_patterns(['tf.QQQ'])


def test_round_trip_add_then_drop_is_byte_identical(tmp_path):
    from fiberhmm.cli.utils import _resolve_drop_patterns, drop_ma_layers_bam
    src = write_bam(tmp_path / 'src.bam', [basic(nucs=[(500, 650)]),
                                           lambda h: make_read(h, 'r2', msps=[MSP], tfs=[CALL], reverse=True)])
    footprint_bam(tmp_path, src, [('chr1', POS + 196, POS + 232, 'CTCF', 0, '+')], 'fp.bam')
    assert _strings(tmp_path / 'fp.bam') != _strings(src)
    assert 'fp_CTCF' in _declared(tmp_path / 'fp.bam')
    drop_ma_layers_bam(str(tmp_path / 'fp.bam'), str(tmp_path / 'back.bam'), _resolve_drop_patterns(['fp_CTCF', 'fpneg_CTCF'])[1])
    assert _strings(tmp_path / 'back.bam') == _strings(src)
    assert _declared(tmp_path / 'back.bam') == _declared(src)


def test_ma_drop_in_place_cli(tmp_path, monkeypatch, capsys):
    import sys
    from fiberhmm.cli import utils
    _, stacked = _stacked(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['fiberhmm-utils', 'ma-drop', str(stacked), '--layers', 'fp_*', '--in-place'])
    utils.main()
    capsys.readouterr()
    assert pysam.AlignmentFile(str(stacked)).check_index()
    monkeypatch.setattr(sys, 'argv', ['fiberhmm-utils', 'ma-list', str(stacked), '--scan'])
    utils.main()
    out = capsys.readouterr().out
    assert '\nfp_CTCF' not in out and 'fpneg_CTCF.Q' in out


def test_streamed_bam_pass_matches_stored_observations(tmp_path):
    src = write_bam(tmp_path / 'src.bam', [
        basic(nucs=[(500, 650)]),
        lambda h: make_read(h, 'r2', msps=[MSP], tfs=[CALL, (300, 330, 70, 0, 0)], reverse=True),
        lambda h: make_read(h, 'r3', msps=[(0, 900)], tfs=[(195, 240, 90, 2, 2)], pos=POS + 50)])
    motifs = mf.load_motif_bed(bed(tmp_path, [('chr1', POS + 196, POS + 232, 'CTCF', 0, '+'),
                                              ('chr1', POS + 302, POS + 328, 'REST', 0, '-'),
                                              ('chr1', POS + 340, POS + 360, 'CTCF', 0, '+')]))
    options = mf.FootprintOptions(max_width=0)
    stored = mf.run_footprint(src, motifs, options, want_spans=True)
    mf.write_annotated_bam(src, tmp_path / 'a.bam', motifs, stored, options, 'x.bed')
    mf.write_annotated_bam(src, tmp_path / 'b.bam', motifs, None, options, 'x.bed', msp_bam_path=None)
    assert _strings(tmp_path / 'a.bam') == _strings(tmp_path / 'b.bam')
    # the same with MSPs from a second BAM
    mf.write_annotated_bam(src, tmp_path / 'c.bam', motifs, mf.run_footprint(src, motifs, options, want_spans=True,
                           msp_bam_path=src), options, 'x.bed', msp_bam_path=src)
    mf.write_annotated_bam(src, tmp_path / 'd.bam', motifs, None, options, 'x.bed', msp_bam_path=src)
    assert _strings(tmp_path / 'c.bam') == _strings(tmp_path / 'd.bam') == _strings(tmp_path / 'a.bam')


# ------------------------------------------------------------------ bigBed motif input

import shutil
import subprocess

AUTOSQL = '''table jasparLike
"JASPAR-like TFBS"
(
string chrom; "Chromosome"
uint chromStart; "Start"
uint chromEnd; "End"
string name; "JASPAR matrix ID"
uint score; "Score"
char[1] strand; "Strand"
string TFName; "Transcription factor name"
)
'''


def _bigbed(tmp_path, rows):
    if not shutil.which('bedToBigBed'):
        pytest.skip('bedToBigBed not on PATH')
    bedp = tmp_path / 'm.bed'
    bedp.write_text(''.join('\t'.join(map(str, r)) + '\n' for r in sorted(rows, key=lambda r: (r[0], r[1]))))
    (tmp_path / 'sizes').write_text('chr1\t10000\n')
    (tmp_path / 'm.as').write_text(AUTOSQL)
    out = tmp_path / 'motifs.data'          # no .bb extension: detection is by magic number
    subprocess.run(['bedToBigBed', '-type=bed6+1', f'-as={tmp_path / "m.as"}', str(bedp), str(tmp_path / 'sizes'),
                    str(out)], check=True, capture_output=True)
    return out


JROWS = [('chr1', POS + 196, POS + 232, 'MA0139.2', 500, '+', 'CTCF'),
         ('chr1', POS + 340, POS + 360, 'MA0139.2', 300, '+', 'CTCF'),
         ('chr1', POS + 302, POS + 328, 'MA0138.3', 450, '-', 'REST'),
         ('chr1', POS + 5196, POS + 5232, 'MA0139.2', 900, '+', 'CTCF')]


def _backends():
    out = []
    try:
        import pyBigWig  # noqa: F401
        out.append('pybigwig')
    except ImportError:
        pass
    if shutil.which('bigBedToBed'):
        out.append('bigbedtobed')
    return out or ['none']


def _cli_table(tmp_path, bed_path, *extra, name='t.tsv'):
    from fiberhmm.cli.motif_footprint import main
    src = tmp_path / 'src.bam'
    if not src.exists():
        write_bam(src, [basic(nucs=[(500, 650)]),
                        lambda h: make_read(h, 'r2', msps=[MSP], tfs=[CALL, (300, 330, 70, 0, 0)], reverse=True),
                        lambda h: make_read(h, 'r4', msps=[MSP], tfs=[CALL], pos=POS + 5000)])
    main(['--bam', str(src), '--bed', str(bed_path), '-o', str(tmp_path / name), '--max-width', '0', *extra])
    return [l.split('\t') for l in (tmp_path / name).read_text().splitlines()[1:]]


@pytest.mark.parametrize('backend', _backends())
def test_bigbed_matches_bed(tmp_path, backend, monkeypatch):
    if backend == 'none':
        pytest.skip('no bigBed reader available')
    from fiberhmm.io import bigbed_motifs
    monkeypatch.setattr(bigbed_motifs.BigBedReader, '_choose', staticmethod(lambda: backend))
    bb = _bigbed(tmp_path, JROWS)
    assert bigbed_motifs.is_bigbed(bb) and not bigbed_motifs.is_bigbed(tmp_path / 'm.bed')
    plain = bed(tmp_path, [(c, s, e, tf, sc, st) for c, s, e, _id, sc, st, tf in JROWS], 'plain.bed')
    a = _cli_table(tmp_path, bb, name='a.tsv')
    b = _cli_table(tmp_path, plain, name='b.tsv')
    key = lambda rows: sorted(tuple(r[:4] + r[5:]) for r in rows)     # name column: matrix ID vs TF name
    assert key(a) == key(b) and len(a) == 4
    assert {r[5] for r in a} == {'fp_CTCF', 'fp_REST'}             # layers from the TFName field
    # name filter: TF name, matrix ID with or without version, case-insensitive
    for flt in ('ctcf', 'MA0139.2', 'ma0139'):
        rows = _cli_table(tmp_path, bb, '--motif-name', flt, name='f.tsv')
        assert len(rows) == 3 and {r[4] for r in rows} == {'MA0139.2'}
    assert len(_cli_table(tmp_path, bb, '--min-score', '450', name='s.tsv')) == 3
    # region-restricted query
    rows = _cli_table(tmp_path, bb, '--region', f'chr1:{POS + 5000}-{POS + 6000}', name='r.tsv')
    assert [int(r[1]) for r in rows] == [POS + 5196]
    # annotated BAM equals the BED run's
    _cli_table(tmp_path, bb, '--out-bam', str(tmp_path / 'bb.bam'), name='x.tsv')
    _cli_table(tmp_path, plain, '--out-bam', str(tmp_path / 'bed.bam'), name='y.tsv')
    assert _strings(tmp_path / 'bb.bam') == _strings(tmp_path / 'bed.bam')


def test_many_layers_need_confirmation(tmp_path):
    rows = [('chr1', POS + 100 + 20 * i, POS + 110 + 20 * i, f'TF{i}', 0, '+') for i in range(25)]
    b = bed(tmp_path, rows, 'many.bed')
    with pytest.raises(SystemExit):
        _cli_table(tmp_path, b)
    assert len(_cli_table(tmp_path, b, '--allow-many-layers', name='ok.tsv')) == 25
    assert {r[5] for r in _cli_table(tmp_path, b, '--layer-name', 'all', name='one.tsv')} == {'fp_all'}


def test_missing_bigbed_reader_message(tmp_path, monkeypatch):
    from fiberhmm.io import bigbed_motifs
    import builtins
    real_import = builtins.__import__

    def no_pybigwig(name, *a, **k):
        if name == 'pyBigWig':
            raise ImportError
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, '__import__', no_pybigwig)
    monkeypatch.setattr(bigbed_motifs.shutil, 'which', lambda name: None)
    with pytest.raises(ValueError, match='pyBigWig'):
        bigbed_motifs.BigBedReader._choose()


# ------------------------------------------------------------------ FiberHMM call bigBeds as input

def _extract(tmp_path, bam, monkeypatch, *flags):
    import sys
    if not shutil.which('bedToBigBed'):
        pytest.skip('bedToBigBed not on PATH')
    from fiberhmm.cli import extract_tags
    out = tmp_path / 'bb'
    out.mkdir(exist_ok=True)
    monkeypatch.setattr(sys, 'argv', ['fiberhmm-extract', '-i', str(bam), '-o', str(out), '--tf', '--msp',
                                      '--nucleosome', '--m6a', '-p', '125', *flags])
    extract_tags.main()
    return out / bam.stem


def _bb_fixture(tmp_path):
    return write_bam(tmp_path / 'src.bam', [
        basic(nucs=[(500, 650)], m6a=(210, 700)),
        lambda h: make_read(h, 'r2', nucs=[(500, 650)], msps=[MSP], tfs=[CALL, (300, 330, 70, 0, 0)], reverse=True),
        lambda h: make_read(h, 'r3', msps=[(0, 900)], tfs=[(195, 240, 90, 2, 2), (600, 700, 60, 3, 3)],
                            pos=POS + 50)])


BB_MOTIFS = [('chr1', POS + 196, POS + 232, 'CTCF', 0, '+'), ('chr1', POS + 302, POS + 328, 'REST', 0, '-'),
             ('chr1', POS + 340, POS + 360, 'CTCF', 0, '+'), ('chr1', POS + 520, POS + 540, 'CTCF', 0, '+')]


def _mol_rows(path):
    rows = [l.split('\t') for l in path.read_text().splitlines()[1:]]
    cols = mf.MOLECULE_COLUMNS
    keep = [cols.index(c) for c in ('motif_id', 'read_name', 'status', 'code', 'reason', 'call_start', 'call_end',
                                    'outer_start', 'outer_end', 'llr', 'edge', 'nuc_overlap')]
    return sorted(tuple(r[i] for i in keep) for r in rows)


@pytest.mark.parametrize('extra', [[], ['--edges', 'conservative'], ['--require-unmodified']])
def test_call_bigbeds_match_bam(tmp_path, monkeypatch, extra):
    from fiberhmm.cli.motif_footprint import main
    src = _bb_fixture(tmp_path)
    prefix = _extract(tmp_path, src, monkeypatch, '--block-scores')
    b = bed(tmp_path, BB_MOTIFS)
    common = ['--bed', str(b), '--max-width', '0', *extra]
    main(['--bam', str(src), *common, '-o', str(tmp_path / 'a.tsv'), '--per-molecule', str(tmp_path / 'a.mol')])
    main(['--calls-bb', str(prefix), *common, '-o', str(tmp_path / 'b.tsv'), '--per-molecule', str(tmp_path / 'b.mol'),
          '--out-bb', str(tmp_path / 'out' )])
    assert _mol_rows(tmp_path / 'a.mol') == _mol_rows(tmp_path / 'b.mol')
    ta = [l.split('\t') for l in (tmp_path / 'a.tsv').read_text().splitlines()]
    tb = [l.split('\t') for l in (tmp_path / 'b.tsv').read_text().splitlines()]
    assert [r[:12] for r in ta] == [r[:12] for r in tb]
    # footprinted instances as a per-read BED12 bigBed
    pyBigWig = pytest.importorskip('pyBigWig')
    written = sorted(tmp_path.glob('out_fp_*.bb'))
    assert written
    bb = pyBigWig.open(str(written[0]))
    rows = bb.entries('chr1', 0, 10000)
    assert rows and all(len(r[2].split('\t')) == 13 for r in rows)
    sql = bb.SQL().decode() if isinstance(bb.SQL(), bytes) else bb.SQL()
    assert 'fiberhmm_fp' in sql and 'blockTq' in sql and 'Sample: out' in sql


def test_call_bigbeds_without_block_scores_warn_and_use_conservative(tmp_path, monkeypatch, capsys):
    from fiberhmm.cli.motif_footprint import main
    src = _bb_fixture(tmp_path)
    prefix = _extract(tmp_path, src, monkeypatch)
    b = bed(tmp_path, BB_MOTIFS)
    capsys.readouterr()
    main(['--calls-bb', str(prefix), '--bed', str(b), '--max-width', '0', '-o', str(tmp_path / 'b.tsv'),
          '--per-molecule', str(tmp_path / 'b.mol')])
    err = capsys.readouterr().err
    assert 'outer edges are unavailable' in err and 'blockTq' in err
    main(['--bam', str(src), '--bed', str(b), '--max-width', '0', '--edges', 'conservative',
          '-o', str(tmp_path / 'a.tsv'), '--per-molecule', str(tmp_path / 'a.mol')])
    # outer bounds and LLR are unknown without block scores
    strip = lambda rows: [r[:7] + r[10:] for r in rows]
    assert strip(_mol_rows(tmp_path / 'a.mol')) == strip(_mol_rows(tmp_path / 'b.mol'))
