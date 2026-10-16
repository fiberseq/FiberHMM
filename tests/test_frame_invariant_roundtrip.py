"""Frame invariant of the calling writers (docs/reference/footprint-tag-frame.md).

fiberhmm-apply, -call and -recall-tfs/-nucs declare coord=molecular and
carry the annotations they do not rewrite into that frame, resolved from the
input header and the read's original tags. So, read with fiberhmm-extract's
parser, every annotation present both before and after sits at the same
reference positions. (Pass-through tools keep the 3.0.0 header behaviour and
are supported on 3.0 output; they are covered by the 3.0-dataset check.)

Each case is one writer x one input header state; the BAM holds one read per
annotation type x orientation (400M at chr1:1000, so query position q is
reference 1000 + q in either orientation).
"""
import array
import sys
from pathlib import Path

import pysam
import pytest

ROOT = Path(__file__).resolve().parents[1]

OLD_CALL = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '2.12.0',
            'CL': 'fiberhmm-call -i in.bam -o out.bam'}
NEW_CALL = {'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '3.0.1',
            'DS': 'FiberHMM fused apply+recall; coord=molecular'}
FT = {'ID': 'ft', 'PN': 'fibertools-rs', 'VN': '0.6.2', 'CL': 'ft add-nucleosomes a.bam b.bam'}
HEADER_STATES = {
    'stripped': {},
    'unmarked': {'PG': [OLD_CALL]},
    'declared': {'PG': [NEW_CALL], 'CO': ['fiberhmm:coord=molecular']},
    'fibertools': {'PG': [FT]},
}
CASES = {
    'nuc_plus': {'MA': '400;nuc+Q:11-100', 'AQ': [200]},
    'msp_dot': {'MA': '400;msp.:121-70'},
    'tf_dot': {'MA': '400;tf.QQQ:11-20', 'AQ': [90, 10, 20]},
    'tf_plus': {'MA': '400;tf+QQQ:301-20', 'AQ': [90, 10, 20]},
    'tf_sr_dot': {'MA': '400;tf_sr.QQQ:11-20', 'AQ': [255, 0, 0]},
    'deam': {'MA': '400;deam+:51-100;deam-:101-100'},
    'm5c': {'MA': '400;ddda_mcg.:291-100'},
    'legacy_as': {'as': [120], 'al': [70], 'aq': [9]},
    'legacy_ns': {'ns': [10], 'nl': [100]},
    'dot_ma_plus_legacy_as': {'MA': '400;nuc.:11-100;msp.:121-70', 'as': [120], 'al': [70]},
}


def _write_input(path, state):
    header = {'HD': {'VN': '1.6', 'SO': 'coordinate'}, 'SQ': [{'SN': 'chr1', 'LN': 10_000}],
              **{k: [dict(v) if isinstance(v, dict) else v for v in vals]
                 for k, vals in HEADER_STATES[state].items()}}
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        for name, tags in sorted(CASES.items()):
            for reverse in (False, True):
                read = pysam.AlignedSegment(out.header)
                read.query_name = f"{name}|{'rev' if reverse else 'fwd'}"
                read.flag = 16 if reverse else 0
                read.reference_id, read.reference_start, read.mapping_quality = 0, 1000, 60
                read.query_sequence, read.cigarstring = 'A' * 400, '400M'
                read.query_qualities = pysam.qualitystring_to_array('I' * 400)
                for tag, value in tags.items():
                    if isinstance(value, list):
                        kind = 'B' if tag in ('AQ', 'aq', 'nq') else 'I'
                        read.set_tag(tag, array.array(kind, value))
                    else:
                        read.set_tag(tag, value)
                out.write(read)
    pysam.index(str(path))
    return str(path)


def _positions(path, input_frame='auto'):
    """{read: {annotation: reference spans}} by fiberhmm-extract's parser."""
    from fiberhmm.cli.extract_tags import (
        _legacy_tags_to_seq,
        _ma_parse_options,
        _parse_all_ma_annotations,
        resolve_extract_frames,
    )

    out = {}
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        frames = resolve_extract_frames(bam.header, input_frame)
        for read in bam.fetch(until_eof=True):
            spans = {}
            for name, values in (_parse_all_ma_annotations(read, **_ma_parse_options(frames))
                                 or {}).items():
                spans['MA:' + name] = sorted((1000 + a['start'], 1000 + a['start'] + a['length'])
                                             for a in values)
            for start_tag, length_tag in (('ns', 'nl'), ('as', 'al')):
                if read.has_tag(start_tag) and read.has_tag(length_tag):
                    try:
                        starts, lengths = _legacy_tags_to_seq(
                            read.get_tag(start_tag), read.get_tag(length_tag), read, frames)
                    except ValueError:
                        continue                      # undecidable: no positions
                    spans[start_tag] = sorted((1000 + s, 1000 + s + n)
                                              for s, n in zip(starts, lengths))
            out[read.query_name] = spans
    return out


# --- writers -------------------------------------------------------------------

def _run_cli(monkeypatch, module, argv, entry='main'):
    import importlib
    monkeypatch.setattr(sys, 'argv', argv)
    try:
        getattr(importlib.import_module(module), entry)()
    except SystemExit as stop:
        assert stop.code in (0, None), stop


def _apply(source, out, monkeypatch, *, region_parallel):
    from fiberhmm.inference.parallel import process_bam_for_footprints
    from fiberhmm.models import get_model_path

    pg = {'PN': 'fiberhmm-apply', 'VN': '3.0.1', 'CL': 'fiberhmm-apply',
          'DS': 'FiberHMM apply; coord=molecular'}
    process_bam_for_footprints(
        source, out, get_model_path('hia5', tool='apply', seq='pacbio'), set(), 10, False,
        'pacbio-fiber', 3, 0, n_cores=1, region_parallel=region_parallel,
        streaming_pipeline=not region_parallel, write_msps=False, pg_record=pg, io_threads=1)


def _recall(source, out, monkeypatch, *, nucs, legacy=True):
    from fiberhmm.cli import recall_tfs
    # Run in-process: keep the module's run state from leaking into other tests.
    for name in ('_WORKER', '_MA_FRAME', '_UNDECIDED_FRAME'):
        monkeypatch.setattr(recall_tfs, name, dict(getattr(recall_tfs, name)))
    _run_cli(monkeypatch, 'fiberhmm.cli.recall_tfs',
             ['fiberhmm-recall-tfs', '-i', source, '-o', out, '--enzyme', 'hia5',
              '--seq', 'pacbio', '-c', '1', '--force-seq',
              *(['--recall-nucs'] if nucs else []), *([] if legacy else ['--no-legacy-tags'])])


def _call_no_legacy(source, out, monkeypatch):
    _run_cli(monkeypatch, 'fiberhmm.cli.call',
             ['fiberhmm-call', '-i', source, '-o', out, '--enzyme', 'hia5', '--seq', 'pacbio',
              '-c', '1', '--io-threads', '1', '--no-qc', '--no-dedup', '--no-legacy-tags',
              '--force-seq'])


WRITERS = {
    'apply-no-msps-streaming': lambda s, o, m: _apply(s, o, m, region_parallel=False),
    'apply-no-msps-region': lambda s, o, m: _apply(s, o, m, region_parallel=True),
    'call-no-legacy-tags': _call_no_legacy,
    'recall-tfs': lambda s, o, m: _recall(s, o, m, nucs=False),
    'recall-nucs': lambda s, o, m: _recall(s, o, m, nucs=True),
    'recall-tfs-no-legacy-tags': lambda s, o, m: _recall(s, o, m, nucs=False, legacy=False),
}


# What each calling writer regenerates (not carried through): its own
# footprint groups, and the legacy tags it rewrites.
FOOTPRINTS = {'MA:nuc', 'MA:msp', 'MA:tf'}
REGENERATED = {
    'apply-no-msps-streaming': FOOTPRINTS | {'ns'},
    'apply-no-msps-region': FOOTPRINTS | {'ns'},
    'call-no-legacy-tags': FOOTPRINTS,
    'recall-tfs': FOOTPRINTS | {'ns', 'as'},
    'recall-nucs': FOOTPRINTS | {'ns', 'as'},
    'recall-tfs-no-legacy-tags': FOOTPRINTS,
}


@pytest.mark.parametrize('state', sorted(HEADER_STATES))
@pytest.mark.parametrize('writer', sorted(WRITERS))
def test_writer_keeps_carried_annotations_in_place(tmp_path, monkeypatch, writer, state):
    source = _write_input(tmp_path / 'in.bam', state)
    out = str(tmp_path / 'out.bam')
    WRITERS[writer](source, out, monkeypatch)
    before = _positions(source)
    after = _positions(out)
    compared, moved = 0, []
    for read, spans in before.items():
        for key, value in spans.items():
            if key in REGENERATED.get(writer, ()):
                continue
            if key in after.get(read, {}):
                compared += 1
                if after[read][key] != value:
                    moved.append((read, key, value, after[read][key]))
    assert compared, 'no carried annotation to compare'
    assert not moved, moved
