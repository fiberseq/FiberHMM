"""Footprint-frame resolution: compatibility with the rules before the
stripped-header extension, and the extension itself.

Whenever the header gives frame evidence (a footprint writer on a @PG
chain, or coord=molecular), every reader must resolve exactly what the
pre-extension code resolved; ``fixtures/frames/pre_c1ede9e_frames.json``
holds those results (QC MSPs, recall-tfs payload frame and tags, m5C
MSP/nucleosome intervals) for historical header variants x read shapes,
computed by the pre-c1ede9e code with two intended changes applied: QC uses
a read's legacy tags when its MA holds no nuc/msp (as at 88cecd0), and (3.0.1)
the m5C readers' MA footprints follow the header's MA frame
(annotation_frame.MAGroupFrame), as QC and recall do, instead of always
molecular.
"""
import array
import json
from pathlib import Path
from types import SimpleNamespace

import pysam
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "frames" / "pre_c1ede9e_frames.json"

L = 1000
SQ = [{'SN': 'chr1', 'LN': 100_000}]
HIST_CALL_212 = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '2.12.0',
                 'DS': 'FiberHMM fused apply+recall; mode=daf recall_nucs=True'}
CALL_2131 = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '2.13.1',
             'DS': 'FiberHMM fused apply+recall; coord=molecular (ns/nl/as/al/MA in molecular original-fiber coordinates); mode=daf'}
CALL_2132 = dict(CALL_2131, VN='2.13.2')
APPLY_2x = {'ID': 'fiberhmm-apply', 'PN': 'fiberhmm-apply', 'VN': '2.16.0',
            'DS': 'FiberHMM apply; coord=molecular (ns/nl/as/al in molecular original-fiber coordinates); mode=daf'}
FT = {'ID': 'ft', 'PN': 'fibertools-rs', 'VN': '0.6.2', 'CL': 'fibertools-rs add-nucleosomes in.bam out.bam'}
OTHER = {'ID': 'samtools', 'PN': 'samtools', 'VN': '1.21', 'CL': 'samtools view -b in.bam'}
CHEM = ('FIBERHMM-CHEMISTRY:v1:assay=daf;enzyme=ddda;platform=pacbio;mode=daf;'
        'model=ddda_TF;pg=fiberhmm-merge')


def merge_header(base):
    from fiberhmm.cli.merge import _merge_output_header
    h = _merge_output_header(base, recall=True, enzyme='ddda', prob_threshold=128,
                             pairs_only=False, nuc_recall_policy='conservative',
                             phase_nrl=0, cpg_mask_policy='unmethylated-only')
    return h.to_dict()


def tag_m5c_header(base):
    from fiberhmm.io.bam_header import maybe_append_pg
    rec = {'PN': 'fiberhmm-tag-m5c', 'VN': '3.0.0',
           'DS': 'DddA molecule-specific mCG; mode=whole_cpg_island'}
    return maybe_append_pg(pysam.AlignmentHeader.from_dict(base), rec).to_dict()


def variants():
    hd = {'HD': {'VN': '1.6'}, 'SQ': SQ}
    v = {
        'v212_seq_plus': dict(hd, PG=[HIST_CALL_212]),
        'v2131_plus_molecular': dict(hd, PG=[CALL_2131]),
        'v2132_dot': dict(hd, PG=[CALL_2132]),
        'apply_2x_legacy': dict(hd, PG=[APPLY_2x]),
        'fibertools': dict(hd, PG=[FT]),
        'v212_with_co_marker': dict(hd, PG=[HIST_CALL_212], CO=['fiberhmm:coord=molecular']),
        'v212_with_chemistry': dict(hd, PG=[HIST_CALL_212], CO=[CHEM]),
        'merged_212_and_2132': dict(hd, PG=[HIST_CALL_212, dict(CALL_2132, ID='fiberhmm-call.1')]),
        'NOEVIDENCE_stripped': dict(hd),
        'NOEVIDENCE_stripped_chemistry': dict(hd, CO=[CHEM]),
        'NOEVIDENCE_samtools_only': dict(hd, PG=[OTHER]),
    }
    try:
        v['merge_recall_over_212'] = merge_header(dict(hd, PG=[HIST_CALL_212]))
    except Exception as exc:
        v['merge_recall_over_212'] = {'error': repr(exc)}
    v['tag_m5c_over_212'] = tag_m5c_header(dict(hd, PG=[HIST_CALL_212]))
    v['tag_m5c_over_2132'] = tag_m5c_header(dict(hd, PG=[CALL_2132]))
    return v


READS = {
    'ma_plus': {'MA': f'{L};nuc+Q:101-50;msp+:301-200', 'AQ': [200]},
    'ma_dot': {'MA': f'{L};nuc.Q:101-50;msp.:301-200', 'AQ': [200]},
    'ma_plus_m5c_dot': {'MA': f'{L};nuc+Q:101-50;msp+:301-200;ddda_mcg.:351-50', 'AQ': [200]},
    'legacy_only': {'legacy': True},
    'legacy_plus_m5c_dot': {'legacy': True, 'MA': f'{L};ddda_ucg.:351-50'},
    'legacy_and_ma_dot': {'legacy': True, 'MA': f'{L};nuc.Q:101-50;msp.:301-200', 'AQ': [200]},
}


def make_read(header, spec, reverse):
    read = pysam.AlignedSegment(pysam.AlignmentHeader.from_dict(header))
    read.query_name = 'r'
    read.flag = 16 if reverse else 0
    read.reference_id = 0
    read.reference_start = 100
    read.query_sequence = 'A' * L
    read.cigarstring = f'{L}M'
    if spec.get('legacy'):
        read.set_tag('ns', array.array('I', [100]))
        read.set_tag('nl', array.array('I', [50]))
        read.set_tag('as', array.array('I', [300]))
        read.set_tag('al', array.array('I', [200]))
    if 'MA' in spec:
        read.set_tag('MA', spec['MA'], value_type='Z')
    if 'AQ' in spec:
        read.set_tag('AQ', array.array('B', spec['AQ']))
    return read


def probe(header, spec, reverse):
    out = {}
    import io, contextlib
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        from fiberhmm.qc import states
        frames = states.tag_frames(header)
        read = make_read(header, spec, reverse)
        try:
            out['qc_msps'] = states.msp_intervals_from_tags(read, frames)
        except Exception as exc:
            out['qc_msps'] = 'ERR ' + type(exc).__name__
        from fiberhmm.cli import recall_tfs
        from fiberhmm.io.annotation_frame import ma_annotation_frame, MOLECULAR
        recall_tfs._MA_FRAME['molecular'] = ma_annotation_frame(header) == MOLECULAR
        try:
            from fiberhmm.io.annotation_frame import header_has_frame_evidence
            recall_tfs._MA_FRAME['evidence'] = header_has_frame_evidence(header)
        except ImportError:
            pass
        frame = recall_tfs._resolve_input_molecular_frame(SimpleNamespace(input_frame='auto'), header)
        try:
            payload = recall_tfs._make_payload(make_read(header, spec, reverse), 'daf', frame)
            tags = payload['tags']
            out['recall'] = {
                'frame': payload.get('_input_molecular_frame', frame),
                'tags': {k: list(map(int, tags[k])) for k in ('ns', 'nl', 'as', 'al') if k in tags},
            }
        except SystemExit as exc:
            out['recall'] = 'EXIT ' + str(exc)[:40]
        # m5c: the CLI's auto resolution + its MSP/nucleosome extraction
        from fiberhmm.daf import m5c
        try:
            from fiberhmm.io.annotation_frame import auto_legacy_frame
            mframe = auto_legacy_frame(header)
        except ImportError:
            from fiberhmm.io.annotation_frame import legacy_tag_frame
            resolved = legacy_tag_frame(header)[0]
            mframe = None if resolved is None else resolved == MOLECULAR
        if mframe is None:
            from fiberhmm.io.bam_header import header_has_coord_marker
            mframe = header_has_coord_marker(header)
        read = make_read(header, spec, reverse)
        out['m5c_msps'] = m5c._initial_msp_intervals_seq(read, mframe)
        out['m5c_nucs'] = m5c._nucleosome_intervals_seq(read, mframe)
    return out




def _cases():
    return {name: header for name, header in variants().items()}


EXPECTED = json.loads(FIXTURE.read_text())


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_frames_with_header_evidence_are_unchanged(key, monkeypatch):
    from fiberhmm.cli import recall_tfs

    monkeypatch.setattr(recall_tfs, "_MA_FRAME", dict(recall_tfs._MA_FRAME))
    monkeypatch.setattr(recall_tfs, "_UNDECIDED_FRAME", dict(recall_tfs._UNDECIDED_FRAME))
    vname, rname, orient = key.split("|")
    got = probe(_cases()[vname], READS[rname], orient == "rev")
    assert json.loads(json.dumps(got, default=str)) == EXPECTED[key]


def _no_evidence(name, read, reverse=True):
    hd = {"HD": {"VN": "1.6"}, "SQ": SQ}
    headers = {"stripped": hd, "chemistry": dict(hd, CO=[CHEM]),
               "samtools": dict(hd, PG=[{"ID": "samtools", "PN": "samtools",
                                          "CL": "samtools view -b x.bam"}])}
    return json.loads(json.dumps(probe(headers[name], READS[read], reverse), default=str))


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    from fiberhmm.cli import recall_tfs

    monkeypatch.setattr(recall_tfs, "_MA_FRAME", dict(recall_tfs._MA_FRAME))
    monkeypatch.setattr(recall_tfs, "_UNDECIDED_FRAME", dict(recall_tfs._UNDECIDED_FRAME))


def test_without_evidence_each_ma_group_decides_its_own_frame():
    # '.' footprint groups are molecular (reverse read: molecule 300-499 is SEQ 500-699).
    got = _no_evidence("stripped", "ma_dot")
    assert got["qc_msps"] == [[500, 200, 200]]
    assert got["recall"]["frame"] is True
    # An appended '.' m5C group decides nothing for the '+' footprint groups.
    got = _no_evidence("stripped", "ma_plus_m5c_dot")
    assert got["qc_msps"] == [[300, 200, 200]]
    assert got["recall"] == {"frame": False,
                             "tags": {"ns": [100], "nl": [50], "as": [300], "al": [200]}}


def test_without_evidence_legacy_tags_follow_dot_footprints_only():
    # A chemistry declaration decides nothing: merge/pair add it while
    # carrying historical tags unchanged.
    got = _no_evidence("chemistry", "legacy_only")
    assert got["qc_msps"] == [[300, 200, 200]]
    assert got["recall"]["frame"] is False
    assert got["m5c_msps"] == [[300, 500]] and got["m5c_nucs"] == [[100, 150]]
    # A '.' footprint group on the same read makes them molecular...
    got = _no_evidence("stripped", "legacy_and_ma_dot")
    assert got["recall"]["frame"] is True
    # ...but a '.' m5C group alone does not.
    got = _no_evidence("samtools", "legacy_plus_m5c_dot")
    assert got["recall"]["frame"] is False
    assert got["m5c_msps"] == [[300, 500]]
    assert got["qc_msps"] == [[300, 200, 200]]   # QC keeps the legacy calls
    got = _no_evidence("stripped", "legacy_only")
    assert got["qc_msps"] == [[300, 200, 200]] and got["recall"]["frame"] is False


def test_no_evidence_warns(capsys, tmp_path, monkeypatch):
    """Without header evidence the per-read rule applies, and the shared
    per-file warning (annotation_frame.warn_if_frames_unverifiable) is
    printed once for an input that carries footprints."""
    from types import SimpleNamespace

    from fiberhmm.cli import recall_tfs
    from fiberhmm.io import annotation_frame as af
    from fiberhmm.io.annotation_frame import LegacyFrame, auto_legacy_frame

    monkeypatch.setattr(af, "_WARNED_PATHS", set())
    monkeypatch.setattr(af, "_CHECKED_PATHS", set())
    hd = {"HD": {"VN": "1.6"}, "SQ": SQ}
    path = tmp_path / "stripped.bam"
    with pysam.AlignmentFile(str(path), "wb", header=hd) as out:
        out.write(make_read(hd, READS["legacy_only"], True))
    recall_tfs._resolve_input_molecular_frame(
        SimpleNamespace(input_frame="auto", in_bam=str(path)), hd)
    assert "header has no FiberHMM/fibertools provenance" in capsys.readouterr().err
    assert isinstance(auto_legacy_frame(hd, "fiberhmm-tag-m5c"), LegacyFrame)
    # Evidence: no warning, and m5C readers keep a plain bool.
    with_chain = dict(hd, PG=[HIST_CALL_212])
    assert auto_legacy_frame(with_chain, "fiberhmm-tag-m5c") is False
    recall_tfs._resolve_input_molecular_frame(SimpleNamespace(input_frame="auto"), with_chain)
    assert "warning" not in capsys.readouterr().err


def test_qc_keeps_legacy_calls_after_an_m5c_only_layer():
    """The real add_m5c_ma_tag on a called read without MA footprints."""
    from fiberhmm.daf.m5c import add_m5c_ma_tag
    from fiberhmm.qc.states import msp_intervals_from_tags, tag_frames

    header = {"HD": {"VN": "1.6"}, "SQ": SQ, "PG": [APPLY_2x]}
    read = make_read(header, READS["legacy_only"], False)
    frames = tag_frames(header)
    before = msp_intervals_from_tags(read, frames)
    add_m5c_ma_tag(read, [(350, 400)])
    assert read.get_tag("MA").endswith("ddda_mcg.:351-50")
    assert msp_intervals_from_tags(read, frames) == before == [(300, 200, 200)]
