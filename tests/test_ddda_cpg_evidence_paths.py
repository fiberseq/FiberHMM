"""DddA CpG islands reach every calling path, whatever tag carries them.

``ft fire`` (fibertools >= 0.13) moves every ``MA`` group, DddA
``ddda_ucg``/``ddda_mcg`` included, into ``Ma``. Region calling strips stale
calls (adopting ``Ma`` as ``MA``) before building its payload; streaming
builds the payload first; recall ships a slim payload to its workers. All of
them must see the same SEQ-frame islands, read through
``tf_recaller.read_cpg_intervals`` (``MA``, else ``Ma``, molecular frame).
"""
from __future__ import annotations

import numpy as np
import pysam
import pytest

from fiberhmm.inference.engine import make_apply_payload, read_cpg_ma_intervals
from fiberhmm.inference.read_filters import strip_stale_call_tags
from fiberhmm.inference.tf_recaller import build_cpg_mask, read_cpg_intervals

L = 1000


def _read(reverse: bool, tag: str):
    header = pysam.AlignmentHeader.from_dict(
        {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chrT', 'LN': 10_000}]})
    r = pysam.AlignedSegment(header)
    r.query_name = 'r1'
    r.flag = 16 if reverse else 0
    r.reference_id = 0
    r.reference_start = 100
    r.mapping_quality = 60
    r.cigarstring = f'{L}M'
    r.query_sequence = ('ACGTY' * (L // 5))
    # molecular [100, 300) unmethylated island, [600, 650) methylated
    r.set_tag(tag, f'{L};nuc.:401-150;msp.:1-100,251-150;'
                   'ddda_ucg.:101-200;ddda_mcg.:601-50')
    return r


def _expected(reverse: bool):
    if reverse:
        return {'ucg': [(L - 300, L - 100)], 'mcg': [(L - 650, L - 600)]}
    return {'ucg': [(100, 300)], 'mcg': [(600, 650)]}


def _norm(iv):
    return {k: [tuple(map(int, x)) for x in iv.get(k, ())] for k in ('ucg', 'mcg')}


@pytest.mark.parametrize('reverse', [False, True], ids=['fwd', 'rev'])
@pytest.mark.parametrize('tag', ['MA', 'Ma'])
def test_every_calling_path_sees_the_same_cpg_islands(reverse, tag):
    want = _expected(reverse)

    # region calling, mask inputs read before the strip (region_workers)
    assert _norm(read_cpg_ma_intervals(_read(reverse, tag))) == want

    # region calling, strip (Ma adopted as MA) then payload
    stripped = _read(reverse, tag)
    strip_stale_call_tags(stripped)
    region_payload = make_apply_payload(stripped, mode='daf')
    assert _norm(region_payload['_cpg_ma_intervals']) == want

    # streaming: payload built from the raw record, before any strip
    stream_payload = make_apply_payload(_read(reverse, tag), mode='daf')
    assert _norm(stream_payload['_cpg_ma_intervals']) == want

    # recall: the slim payload and the worker stub's CpG mask
    from fiberhmm.cli.recall_tfs import _PayloadRead, _make_payload
    payload = _make_payload(_read(reverse, tag), 'daf', True)
    stub = _PayloadRead(payload['seq'], payload['is_reverse'], payload['tags'],
                        cpg_intervals=payload.get('_cpg_intervals'))
    assert _norm(read_cpg_intervals(stub)) == want
    mask = build_cpg_mask(stub, L)
    lo, hi = want['ucg'][0]
    assert not mask[lo:hi].any() and mask[:lo].all() and mask[hi:].all()


def test_ma_wins_over_fibertools_ma():
    r = _read(False, 'Ma')
    r.set_tag('MA', f'{L};ddda_ucg.:501-10')
    assert read_cpg_intervals(r) == {'ucg': [(500, 510)], 'mcg': []}


def test_no_annotation_tags_no_islands():
    r = _read(False, 'MA')
    r.set_tag('MA', None)
    assert read_cpg_intervals(r) == {'ucg': [], 'mcg': []}
    assert 'cpg_intervals' not in make_apply_payload(r, mode='daf')
    assert np.asarray(build_cpg_mask(r, L)).all()
