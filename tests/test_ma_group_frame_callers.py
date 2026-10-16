"""Every reader of MA groups resolves their frame per group through
annotation_frame.MAGroupFrame (docs/reference/footprint-tag-frame.md).

Reads are reverse 400M alignments at chr1:1000 unless stated otherwise: a
query interval [s, e) maps to the reference [1000+s, 1000+e).
"""
import array
from pathlib import Path

import numpy as np
import pysam
import pytest

FX = Path(__file__).parent / 'fixtures' / 'fibertools_frame'
OLD_CALL = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '2.12.0',
            'CL': 'fiberhmm-call -i in.bam -o out.bam'}


def _header(pg=(), co=()):
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': 'chr1', 'LN': 10_000}]}
    if pg:
        header['PG'] = [dict(p) for p in pg]
    if co:
        header['CO'] = list(co)
    return pysam.AlignmentHeader.from_dict(header)


def _read(header, tags, reverse=True):
    read = pysam.AlignedSegment(header)
    read.query_name, read.flag, read.reference_id = 'rev', 16 if reverse else 0, 0
    read.reference_start, read.mapping_quality = 1000, 60
    read.query_sequence, read.cigarstring = 'A' * 400, '400M'
    read.query_qualities = pysam.qualitystring_to_array('I' * 400)
    for tag, value in tags.items():
        read.set_tag(tag, value)
    return read


# --- 1. fiberhmm.daf.m5c.ma_strand_intervals ---------------------------------

@pytest.mark.parametrize('pg, co, nuc, msp', [
    ([OLD_CALL], (), (10, 110), (120, 190)),                          # unmarked: SEQ
    ([OLD_CALL], ('fiberhmm:coord=molecular',), (290, 390), (210, 280)),
    ((), (), (10, 110), (120, 190)),                                  # no evidence, '+' groups
])
def test_m5c_reader_follows_the_group_frame(pg, co, nuc, msp):
    from fiberhmm.daf.m5c import ma_intervals, ma_strand_intervals

    read = _read(_header(pg, co), {'MA': '400;nuc+:11-100;msp+:121-70;ddda_mcg.:291-100'})
    assert ma_intervals(read, 'nuc') == [nuc]
    assert ma_intervals(read, 'msp') == [msp]
    assert ma_intervals(read, 'ddda_mcg') == [(10, 110)]      # m5C: always molecular
    assert ma_strand_intervals(read, 'nuc') == {'+': [nuc]}


@pytest.mark.parametrize('co', [(), ('fiberhmm:coord=molecular',)])
def test_m5c_deam_reader_agrees_with_the_duplex_mask(co):
    from fiberhmm.crossstrand.recall import deam_regime_masks
    from fiberhmm.daf.m5c import ma_strand_intervals

    read = _read(_header([OLD_CALL], co), {'MA': '400;deam+:11-100;deam-:51-100'})
    plus, minus = deam_regime_masks(read)
    coverage = ma_strand_intervals(read, 'deam')
    for strand, mask in (('+', plus), ('-', minus)):
        (start, end), = coverage[strand]
        assert list(np.flatnonzero(mask)) == list(range(start, end))


def test_consensus_hia5_replay_accepts_an_unmarked_scaffold(tmp_path):
    """The replay re-reads the scaffold with ma_intervals; on an unmarked
    FiberHMM header it must read MA as the loader does (SEQ frame)."""
    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.consensus.adapter import evidence_unit
    from fiberhmm.inference.consensus.upstream_recall import recall_hia5_alignment
    from fiberhmm.inference.strand_rescue import load_region_evidence
    from fiberhmm.inference.tf_recaller import build_llr_tables

    source = FX / 'fiberhmm3.0.0_call.bam'
    path = tmp_path / 'unmarked.bam'
    with pysam.AlignmentFile(str(source)) as bam:
        hd = bam.header.to_dict()
        hd['PG'] = [OLD_CALL]
        hd['CO'] = [c for c in hd.get('CO', []) if 'coord=molecular' not in c]
        header = pysam.AlignmentHeader.from_dict(hd)
        with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
            for read in bam:
                out.write(pysam.AlignedSegment.from_dict(read.to_dict(), header))
    pysam.index(str(path))
    model, k, mode = load_model_with_metadata(
        str(Path(__file__).resolve().parents[1] / 'fiberhmm' / 'models' / 'hia5_pacbio.json'))
    hit, miss = build_llr_tables(model)
    reads = load_region_evidence(
        str(path), 'chrT', 0, 8000, strand_mode='alignment', mode=mode, context_size=k,
        prob_threshold=125, llr_hit=hit, llr_miss=miss, min_mapq=0,
        ma_annotation_frame='auto', legacy_annotation_frame='seq')
    by_name = {r.name: r for r in reads}
    with pysam.AlignmentFile(str(path)) as bam:
        for alignment in bam:
            unit = evidence_unit(by_name[alignment.query_name], model, 'd', [], 0, 8000)
            result = recall_hia5_alignment(alignment, unit, model, 'alignment', mode, k,
                                           125, 5.0, legacy_annotation_frame='seq')
            assert result['status'] == 'recalled', alignment.query_name


# --- 2. consensus export of fibertools Ma ------------------------------------

def test_export_keeps_ma_groups_where_the_output_header_reads_them():
    from fiberhmm.cli.extract_tags import _parse_all_ma_annotations
    from fiberhmm.inference.consensus.bam_export import EXPORT_DS, _append_annotations
    from fiberhmm.io.annotation_frame import MAGroupFrame

    contract = ('FIBERHMM-CONSENSUS-MA:v1:{"coordinates": '
                '"original_source_call_in_molecular_frame", "layers": ["fam"]}')
    for pg, molecular in (((), True),                     # no evidence: '.' molecular
                          ([OLD_CALL], False)):           # unmarked writer: SEQ
        header = _header([*pg, {'ID': 'fiberhmm-consensus', 'PN': 'fiberhmm-consensus',
                                'DS': EXPORT_DS}], [contract])
        read = _read(header, {'Ma': '400;nuc.:11-100'})
        frame = MAGroupFrame(header)
        before = _parse_all_ma_annotations(read)['nuc'][0]          # Ma: molecular
        assert (before['start'], before['length']) == (290, 100)
        rows = [dict(chrom='chr1', interval=[1100, 1120], layer='fam', token='t1',
                     tq=50, fi=1, op=2)]
        assert _append_annotations(read, rows, frame) == 1
        assert read.get_tag('MA').startswith('400;nuc.:11-100' if molecular
                                              else '400;nuc.:291-100')
        parsed = _parse_all_ma_annotations(read, annotation_frame=frame.header_frame,
                                           group_frame=frame)
        nuc, fam = parsed['nuc'][0], parsed['fam'][0]
        assert (nuc['start'], nuc['length']) == (290, 100)          # still chr1:1290-1390
        assert (fam['start'], fam['start'] + fam['length']) == (100, 120)


# --- 3. consensus / strand-rescue loader -------------------------------------

def test_loader_resolves_mixed_qualifiers_per_group(tmp_path):
    from fiberhmm.inference.strand_rescue import N_CTX, load_region_evidence

    path = tmp_path / 'mixed.bam'
    header = _header()
    seq = list('A' * 400)
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        read = _read(out.header, {'MA': '400;tf.:11-20;tf+:101-20'})
        read.query_sequence = ''.join(seq)
        read.query_qualities = pysam.qualitystring_to_array('I' * 400)
        read.set_tag('MM', 'A+a.;')
        read.set_tag('ML', array.array('B', []))
        out.write(read)
    pysam.index(str(path))

    def load(window):
        return load_region_evidence(
            str(path), 'chr1', 900, 1500, strand_mode='alignment', mode='pacbio-fiber',
            context_size=3, prob_threshold=125, llr_hit=np.zeros(N_CTX),
            llr_miss=np.zeros(N_CTX), min_mapq=0, ma_annotation_frame='auto',
            required_annotation_overlap=('tf', *window))

    assert len(load((1370, 1390))) == 1       # the dotted TF (molecular)
    assert len(load((1100, 1120))) == 1       # the '+' TF (SEQ)
    assert len(load((1010, 1030))) == 0       # where a per-name rule put the dotted TF


# --- other MA readers --------------------------------------------------------

@pytest.mark.parametrize('pg, expected', [([OLD_CALL], 1060), ([], 1340)])
def test_pair_dyads_follow_the_group_frame(pg, expected):
    from fiberhmm.crossstrand.pairing import nuc_dyads_ref

    strand = '+' if pg else '.'
    read = _read(_header(pg), {'MA': f'400;nuc{strand}:11-100'})
    assert list(nuc_dyads_ref(read)) == [expected]


@pytest.mark.parametrize('pg, start', [([OLD_CALL], 10), ([], 10),
                                       ([dict(OLD_CALL, VN='3.0.0', DS='coord=molecular')], 290)])
def test_utils_reference_bam_reads_ns_in_its_frame(tmp_path, monkeypatch, pg, start):
    """fiberhmm-utils transfer --reference-bam reads ns/nl by the provenance rule."""
    from types import SimpleNamespace

    from fiberhmm.cli import utils

    path = tmp_path / 'ref.bam'
    with pysam.AlignmentFile(str(path), 'wb', header=_header(pg)) as out:
        out.write(_read(out.header, {'ns': array.array('I', [10]), 'nl': array.array('I', [100])}))
    pysam.index(str(path))
    seen = []

    class Counter:
        def __init__(self, *args):
            pass

        def process_read_with_footprints(self, sequence, mask, edge_trim):
            seen.append(np.flatnonzero(mask))

    monkeypatch.setattr(utils, 'AccessibilityCounter', Counter)
    utils._process_reference_bam(str(path), 3, SimpleNamespace(
        min_mapq=0, edge_trim=0, max_reads=0))
    got = seen[0]
    assert (got[0], got[-1] + 1) == (start, start + 100)


# --- round 4 ------------------------------------------------------------------

NEW_APPLY = {'ID': 'fiberhmm-apply', 'PN': 'fiberhmm-apply', 'VN': '3.0.1',
             'DS': 'FiberHMM apply; coord=molecular'}


def test_apply_no_msps_keeps_retained_msps_in_the_declared_frame(tmp_path):
    """apply --no-msps keeps the input's as/al/aq, while its output declares
    coord=molecular: SEQ-frame MSPs are converted, so they stay in place."""
    from fiberhmm.cli.extract_tags import extract_tags_parallel
    from fiberhmm.inference.streaming_drain import strip_for_apply

    read = _read(_header([OLD_CALL]), {'as': array.array('I', [120, 300]),
                                       'al': array.array('I', [70, 20]),
                                       'aq': array.array('B', [5, 9]),
                                       'ns': array.array('I', [10]), 'nl': array.array('I', [50])})
    strip_for_apply(read, write_msps=False)
    assert not read.has_tag('ns')
    # Molecular: SEQ 300-320 -> 80-100, SEQ 120-190 -> 210-280 (sorted, aq follows).
    assert list(read.get_tag('as')) == [80, 210] and list(read.get_tag('al')) == [20, 70]
    assert list(read.get_tag('aq')) == [9, 5]
    path = tmp_path / 'apply_out.bam'
    with pysam.AlignmentFile(str(path), 'wb', header=_header([OLD_CALL, dict(NEW_APPLY, PP='fiberhmm-call')])) as out:
        out.write(pysam.AlignedSegment.from_dict(read.to_dict(), out.header))
    pysam.index(str(path))
    bed = tmp_path / 'msp.bed'
    extract_tags_parallel(str(path), {'msp': str(bed)}, ['msp'], with_scores=False)
    fields = bed.read_text().split('\t')
    starts = [int(fields[1]) + int(x) for x in fields[11].split(',') if x]
    sizes = [int(x) for x in fields[10].split(',') if x]
    assert list(zip(starts, sizes)) == [(1120, 70), (1300, 20)]
    # Molecular input MSPs are kept as they are; undecidable ones are dropped.
    kept = _read(_header([NEW_APPLY]), {'as': array.array('I', [120]), 'al': array.array('I', [70])})
    strip_for_apply(kept, write_msps=False)
    assert list(kept.get_tag('as')) == [120]


@pytest.mark.parametrize('pg, flipped', [([OLD_CALL], False), ([NEW_APPLY], True)])
def test_apply_stats_read_legacy_tags_in_their_frame(tmp_path, pg, flipped):
    from fiberhmm.inference.stats import collect_stats_from_bam

    path = tmp_path / 'in.bam'
    with pysam.AlignmentFile(str(path), 'wb', header=_header(pg)) as out:
        out.write(_read(out.header, {'ns': array.array('I', [10]), 'nl': array.array('I', [100])}))
    import fiberhmm.inference.stats as stats_module
    seen = []
    original = stats_module.flip_intervals_to_seq
    stats_module.flip_intervals_to_seq = lambda *a, **k: seen.append(a) or original(*a, **k)
    try:
        collect_stats_from_bam(str(path), n_samples=10)
    finally:
        stats_module.flip_intervals_to_seq = original
    assert bool(seen) == flipped          # only molecular tags are mirrored


@pytest.mark.parametrize('pg, tf, msp', [
    ([OLD_CALL], (1010, 1030), (1050, 1150)),
    ([NEW_APPLY], (1370, 1390), (1250, 1350)),
])
def test_footprint_model_loader_follows_the_group_frame(tmp_path, pg, tf, msp):
    from fiberhmm.io.footprint_bam import load_footprint_molecules_from_bam

    path = tmp_path / 'fp.bam'
    with pysam.AlignmentFile(str(path), 'wb', header=_header(pg)) as out:
        out.write(_read(out.header, {'MA': '400;tf+:11-20;msp+:51-100'}))
    pysam.index(str(path))
    molecule, = load_footprint_molecules_from_bam(str(path)).molecules
    assert [(t.start, t.end) for t in molecule.tfs] == [tf]
    assert [tuple(m) for m in molecule.msps] == [msp]


@pytest.mark.parametrize('pg', [(), [OLD_CALL]])
def test_strand_rescue_baseline_layers_stay_in_place(pg):
    from fiberhmm.cli.extract_tags import _parse_all_ma_annotations
    from fiberhmm.cli.strand_rescue_annotate import add_strand_rescue_groups
    from fiberhmm.io.annotation_frame import MAGroupFrame

    header = _header(pg)
    read = _read(header, {'MA': '400;tf+QQQ:11-20;nuc+QQQ:101-100',
                          'AQ': array.array('B', [100, 1, 2, 50, 3, 4])})
    add_strand_rescue_groups(read, [])
    frame = MAGroupFrame(header)
    parsed = _parse_all_ma_annotations(read, annotation_frame=frame.header_frame,
                                       group_frame=frame)

    def spans(layer):
        return [(1000 + a['start'], 1000 + a['start'] + a['length']) for a in parsed[layer]]

    assert spans('tf') == spans('tf_sr') == [(1010, 1030)]
    assert spans('nuc') == spans('nuc_sr') == [(1100, 1200)]


def test_export_native_tq_comes_from_the_tf_at_the_family_position():
    from fiberhmm.inference.consensus.bam_export import _append_annotations
    from fiberhmm.io.annotation_frame import MAGroupFrame

    for ma, aq, expected in (('400;tf+Q:11-20,371-20', [80, 190], 80),
                             ('400;tf+Q:11-20', [80], 80)):
        header = _header([OLD_CALL])
        read = _read(header, {'MA': ma, 'AQ': array.array('B', aq)})
        rows = [dict(chrom='chr1', interval=[1010, 1030], layer='fam', token='t1',
                     tq=None, fi=1, op=2)]
        assert _append_annotations(read, rows, MAGroupFrame(header)) == 1
        assert list(read.get_tag('AQ'))[len(aq)] == expected


@pytest.mark.parametrize('pg, span', [([OLD_CALL], (1010, 1110)), ([NEW_APPLY], (1290, 1390))])
def test_legacy_bed_helper_reads_ns_in_its_frame(tmp_path, pg, span):
    from fiberhmm.inference.bam_output import extract_bed_from_tagged_bam

    path = tmp_path / 'in.bam'
    with pysam.AlignmentFile(str(path), 'wb', header=_header(pg)) as out:
        out.write(_read(out.header, {'ns': array.array('I', [10]), 'nl': array.array('I', [100])}))
    bed = tmp_path / 'out.bed'
    assert extract_bed_from_tagged_bam(str(path), str(bed), with_scores=False) == 1
    fields = bed.read_text().split('\t')
    assert (int(fields[1]), int(fields[2])) == span


@pytest.mark.parametrize('pg, strand, span', [
    ([OLD_CALL], '.', (1010, 1030)),    # unmarked: '.' groups SEQ
    ((), '.', (1370, 1390)),            # no evidence: '.' molecular
    ((), '+', (1010, 1030)),
])
def test_strand_rescue_audit_projects_groups_in_their_frame(pg, strand, span):
    from fiberhmm.cli.strand_rescue_audit import _reference_interval

    read = _read(_header(pg), {'MA': f'400;tf_sr{strand}QQQ:11-20'})
    assert _reference_interval(read, {'type': 'tf_sr', 'strand': strand,
                                      'interval': (10, 20)}, 400) == span


def test_stream_action_validation_compares_molecular_intervals():
    import hashlib

    from fiberhmm.cli.strand_rescue_annotate import (
        StreamActionRecord,
        StreamEdgeAction,
        _validate_stream_action_against_read,
    )

    read = _read(_header([OLD_CALL]), {'MA': '400;tf+QQQ:11-20',
                                       'AQ': array.array('B', [100, 1, 2])})
    sha = hashlib.sha256(read.to_string().encode('utf-8')).hexdigest()

    def record(source):
        return StreamActionRecord(0, 'rev', sha, (), (StreamEdgeAction(
            'tok', 'tf', 0, source, (368, 24), (1, 2, 3)),))

    _validate_stream_action_against_read(read, record((370, 20)))   # molecular: accepted
    with pytest.raises(ValueError, match='differs from MA'):
        _validate_stream_action_against_read(read, record((10, 20)))


def test_strand_rescue_report_loads_ma_by_the_header_rule(tmp_path, monkeypatch):
    import fiberhmm.cli.strand_rescue as cli

    path = tmp_path / 'in.bam'
    with pysam.AlignmentFile(str(path), 'wb', header=_header([OLD_CALL])) as out:
        out.write(_read(out.header, {'MA': '400;tf+:11-20'}))
    pysam.index(str(path))
    seen = {}

    class Stop(Exception):
        pass

    def fake_load(*args, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(cli, 'load_region_evidence', fake_load)
    with pytest.raises(Stop):
        cli.main(['-i', str(path), '--preset', 'dddb', '--region', 'chr1:1000-1400',
                  '-o', str(tmp_path / 'sr.json')])
    assert seen['ma_annotation_frame'] == 'auto'


def _extract_spans(path, tmp, label):
    from fiberhmm.cli.extract_tags import extract_tags_parallel

    out = {}
    for kind in ('nucleosome', 'msp'):
        bed = tmp / f'{label}_{kind}.bed'
        extract_tags_parallel(str(path), {kind: str(bed)}, [kind], with_scores=True,
                              block_scores=True)
        for line in bed.read_text().splitlines():
            f = line.split('\t')
            starts = [int(f[1]) + int(x) for x in f[11].split(',') if x]
            sizes = [int(x) for x in f[10].split(',') if x]
            out[(f[3], kind)] = (list(zip(starts, sizes)), f[12])
    return out


def test_call_m5c_locus_tagging_keeps_native_fibertools_footprints(tmp_path):
    """call-m5c --tag-mode locus on fibertools output, through the CLI's
    writer (its output header declares the carried frame): extracted with
    the actual output header, every native nucleosome/MSP keeps its position
    and quality, and the m5C span is added."""
    from types import SimpleNamespace

    from fiberhmm.cli.call_m5c import _write_outputs
    from fiberhmm.daf.m5c import M5CDomain

    source = FX / 'ft0.13_addnuc_fire.bam'
    output = tmp_path / 'locus.bam'
    args = SimpleNamespace(output=str(tmp_path / 'd.bed'), tag_bam=str(source),
                           tag_output=str(output), tag_mode='locus', tag_input_frame='auto',
                           io_threads=1, reference=None, cpg_islands=None,
                           write_cpg_islands=None, read_posterior=0.99, min_other=0,
                           read_min_island_cpg=1)
    staged = {args.output: args.output, args.tag_output: args.tag_output}
    _write_outputs(args, staged, [M5CDomain('chrT', 100, 200, True, 0.99)],
                   np.ones(4), 0, 0, 0)
    pysam.index(str(output))
    with pysam.AlignmentFile(str(output)) as bam:
        reads = {r.query_name: r for r in bam}
        assert all(r.has_tag('MA') and 'ddda_mcg' in r.get_tag('MA') for r in reads.values())
    before = _extract_spans(source, tmp_path, 'before')
    after = _extract_spans(output, tmp_path, 'after')
    assert before and before == after


def test_tag_m5c_island_writer_seeds_native_groups_in_the_output_frame():
    """The island writer (tag-m5c, call-m5c island mode) passes its output
    header's frame the same way."""
    from fiberhmm.daf.m5c import add_island_m5c_ma_tags
    from fiberhmm.io.annotation_frame import MAGroupFrame, append_coord_to_ds, pass_through_frame
    from fiberhmm.io.bam_header import maybe_append_pg
    from fiberhmm.cli.extract_tags import _ma_parse_options, _parse_all_ma_annotations, resolve_extract_frames

    with pysam.AlignmentFile(str(FX / 'ft0.13_addnuc_fire.bam')) as bam:
        out_header = maybe_append_pg(bam.header, {
            'PN': 'fiberhmm-tag-m5c', 'VN': 'x', 'CL': 'x',
            'DS': append_coord_to_ds('DddA mCG', pass_through_frame(bam.header, None))})
        in_frames, out_frames = resolve_extract_frames(bam.header), resolve_extract_frames(out_header)
        for read in bam:
            before = _parse_all_ma_annotations(read, **_ma_parse_options(in_frames))
            add_island_m5c_ma_tags(read, [(100, 200)], [], output_frame=MAGroupFrame(out_header))
            after = _parse_all_ma_annotations(read, **_ma_parse_options(out_frames))
            for name in ('nuc', 'msp', 'fire'):
                key = lambda a: (a['start'], a['length'], tuple(a['quals']))
                assert sorted(map(key, after[name])) == sorted(map(key, before[name])), (read.query_name, name)

