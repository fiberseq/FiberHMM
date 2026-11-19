"""Caller evidence survives a BAM round trip without any prepass files."""
import array

import numpy as np
import pysam
import pytest

from fiberhmm.daf.insert_consensus import ReadInsertEvidence, record_key
from fiberhmm.daf.replay import evidence_token
from fiberhmm.inference import engine
from fiberhmm.inference.streaming_drain import strip_for_fused_call
from fiberhmm.nuc.calibration import calibrate_self
from fiberhmm.nuc.molecule import molecule_outputs
from fiberhmm.nuc.recall import CallerSettings, fiber_read_from_record
from fiberhmm.nuc.records import DecodeContext, decode_record


def _header(portable=True, mask=None):
    ds = ('coord=molecular mode=daf cpg_mask=off '
          'daf_snp_mask=on/2sites daf_insert_consensus=on/1of1/min20')
    if portable:
        ds += ' ' + evidence_token(mask or {'c': {1040, 1060}})
    return pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'c', 'LN': 10000}],
        'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call',
                'CL': 'fiberhmm-call --enzyme dddb --daf-mask-runs 0', 'DS': ds}]})


def _read(h, reverse=False, ga=False):
    r = pysam.AlignedSegment(h)
    r.query_name = 'carrier'
    seq = list('ACGT' * 60)
    seq[40], seq[60], seq[80] = 'Y', 'C', 'Y'
    seq[100:120] = 'C' * 20
    seq[110] = 'T'
    if ga:
        seq = [b.translate(str.maketrans('ACGTY', 'TGCAR')) for b in seq]
    r.query_sequence = ''.join(seq)
    r.flag = 16 if reverse else 0
    r.reference_id, r.reference_start = 0, 1000
    r.cigartuples = [(0, 100), (1, 20), (0, 120)]
    r.set_tag('st', 'GA' if ga else 'CT')
    return r


def _calls(r):
    r.set_tag('ns', array.array('i', [20, 150]))
    r.set_tag('nl', array.array('i', [50, 50]))


@pytest.mark.parametrize('reverse,ga', [(False, False), (True, False), (False, True), (True, True)])
def test_portable_masks_and_insertions_match_caller_encoder(tmp_path, monkeypatch, reverse, ga):
    from fiberhmm.core.bam_reader import daf_run_mask_scope, encode_from_query_sequence
    h = _header()
    r = _read(h, reverse, ga)
    ev = ReadInsertEvidence(known=set(range(100, 120)), mods={110}, strand=int(ga))
    mask = {'c': {1040, 1060}}
    with engine.daf_snp_mask_scope(mask), engine.daf_insert_evidence_scope({record_key(r): ev}):
        fr = engine._extract_fiber_read_from_pysam(r, 'daf', 128)
        with daf_run_mask_scope(0, 'keep-one'):
            codes = encode_from_query_sequence(
                fr['query_sequence'], fr['m6a_query_positions'], 10, mode='daf',
                strand=fr['_daf_strand'], context_size=3, is_reverse=reverse,
                unknown_positions=fr.get('unknown_query_positions'))
        # The common streaming/region writer persists the active evidence.
        strip_for_fused_call(r)
    _calls(r)
    path = str(tmp_path / 'portable.bam')
    with pysam.AlignmentFile(path, 'wb', header=h) as b:
        b.write(r)
    # Deliberately poison both process globals; neither nuc loader may use them.
    monkeypatch.setattr(engine, '_DAF_SNP_MASK', {'c': {1080}})
    monkeypatch.setattr(engine, '_DAF_INSERT_EVIDENCE', {})
    with pysam.AlignmentFile(path) as b:
        rr = next(b)
        ctx = DecodeContext(b.header)
        m, reason = decode_record(rr, ctx)
        recalled = fiber_read_from_record(rr, CallerSettings.from_header(b.header))
    assert reason == 'ok' and m is not None
    expected_hits = codes < 4096
    expected_inf = expected_hits | ((codes >= 4097) & (codes < 8193))
    if reverse:
        expected_hits, expected_inf = expected_hits[::-1], expected_inf[::-1]
    np.testing.assert_array_equal(m.hits, expected_hits)
    np.testing.assert_array_equal(m.informative, expected_inf)
    seq_inf = m.informative[::-1] if reverse else m.informative
    seq_hits = m.hits[::-1] if reverse else m.hits
    assert not seq_inf[40] and not seq_inf[60]  # masked hit AND masked miss
    assert seq_hits[80] and seq_hits[110] and seq_inf[111]
    assert recalled['m6a_query_positions'] == fr['m6a_query_positions'] == {80, 110}
    assert recalled['unknown_query_positions'] == fr['unknown_query_positions']
    assert not ctx.evidence.warnings
    assert engine._DAF_SNP_MASK == {'c': {1080}}
    assert engine._DAF_INSERT_EVIDENCE == {}


def test_older_bam_fallback_records_warning_in_results_and_calibration(monkeypatch):
    r = _read(_header(portable=False))
    _calls(r)
    monkeypatch.setattr(engine, '_DAF_SNP_MASK', {'c': {1040}})
    ctx = DecodeContext(r.header)
    m, why = decode_record(r, ctx)
    assert why == 'ok' and m.informative[40]
    assert not m.informative[111] and not m.hits[110]
    cal = calibrate_self([m])
    result = molecule_outputs(m, cal=cal)
    assert any('legacy BAM lacks portable' in w for w in cal.notes)
    assert result.warnings == ctx.evidence.warnings
    assert cal.notes == result.warnings


def test_header_contexts_do_not_leak_masks_or_stale_insertion_tags(monkeypatch):
    a = _read(_header())
    b = _read(_header(mask={'c': {1080}}))
    _calls(a)
    _calls(b)
    ca, cb = DecodeContext(a.header), DecodeContext(b.header)
    ma, _ = decode_record(a, ca)
    mb, _ = decode_record(b, cb)
    assert not ma.hits[40] and ma.hits[80]
    assert mb.hits[40] and not mb.hits[80]
    with engine.daf_insert_evidence_scope({}):
        a.set_tag('ei', 'stale')
        strip_for_fused_call(a)
    assert not a.has_tag('ei')


def test_portable_insertion_evidence_rescues_an_otherwise_empty_record():
    r = _read(_header())
    seq = list(r.query_sequence)
    seq[80] = 'C'  # the remaining aligned conversion at 40 is SNP-masked
    r.query_sequence = ''.join(seq)
    ctx = DecodeContext(r.header)
    _calls(r)
    assert decode_record(r, ctx) == (None, 'no_evidence')
    ev = ReadInsertEvidence(known=set(range(100, 120)), mods={110}, strand=0)
    with engine.daf_insert_evidence_scope({record_key(r): ev}):
        strip_for_fused_call(r)
    _calls(r)
    m, why = decode_record(r, ctx)
    assert why == 'ok' and m.hits[110] and m.n_hits == 1
    fr = fiber_read_from_record(r, CallerSettings.from_header(r.header))
    assert fr['m6a_query_positions'] == {110}


@pytest.mark.parametrize('prior_qcfail', [False, True])
def test_replay_insertion_key_survives_chimera_cleanup(prior_qcfail):
    from fiberhmm.inference.chimeras import clear_chimera_mark, mark_chimera
    r = _read(_header())
    _calls(r)
    r.is_qcfail = prior_qcfail
    ev = ReadInsertEvidence(known=set(range(100, 120)), mods={110}, strand=0)
    with engine.daf_insert_evidence_scope({record_key(r): ev}):
        strip_for_fused_call(r)
    _calls(r)
    mark_chimera(r)
    assert decode_record(r, None) == (None, 'chimera')
    clear_chimera_mark(r)
    assert r.is_qcfail == prior_qcfail
    mol, reason = decode_record(r, DecodeContext(r.header))
    assert reason == 'ok' and mol.hits[110] and mol.informative[111]
    # Foreign xc values and QC failures are not FiberHMM chimera marks.
    r.set_tag('xc', 'foreign')
    r.is_qcfail = True
    mol, reason = decode_record(r, DecodeContext(r.header))
    assert reason == 'ok' and mol.hits[110]


@pytest.mark.parametrize('mode', ['flag', 'remove', 'keep'])
def test_nuc_replays_published_decisions_for_unnamed_records(tmp_path, mode, monkeypatch):
    from fiberhmm.inference.chimeras import mark_chimera
    from fiberhmm.nuc.records import iter_molecules
    h = _header()
    hd = h.to_dict()
    hd['PG'][-1]['DS'] += ' chimeras=' + mode
    h = pysam.AlignmentHeader.from_dict(hd)
    swap, clean = _read(h), _read(h)
    swap.query_sequence = 'ACGY' * 25 + 'ACRT' * 25 + 'ACGY' * 10
    swap.cigarstring = '240M'
    clean.cigarstring = '240M'
    for r in (swap, clean):
        r.query_name = '*'
        _calls(r)
    if mode == 'flag':
        mark_chimera(swap)
    current = [clean] if mode == 'remove' else [swap, clean]
    path = tmp_path / 'decided.bam'
    with pysam.AlignmentFile(path, 'wb', header=h) as bam:
        for r in current * 2:
            bam.write(r)
    # A reader may coexist with a fresh caller whose default detection is on.
    monkeypatch.setattr(engine, '_DAF_CHIMERA_CFG', {'filter': True, 'min_seg': 5, 'purity': 0.8})
    with pysam.AlignmentFile(path, 'rb') as bam:
        ctx = DecodeContext(bam.header)
        settings = CallerSettings.from_header(bam.header)
        for r in bam:
            mol, reason = decode_record(r, ctx)
            recalled = fiber_read_from_record(r, settings)
            if r.is_qcfail:
                assert reason == 'chimera' and mol is None and recalled is None
            else:
                assert reason == 'ok' and mol is not None and recalled is not None
    stats = {}
    molecules = list(iter_molecules(str(path), stats=stats))
    assert len(molecules) == (4 if mode == 'keep' else 2)
    assert stats.get('chimera', 0) == (2 if mode == 'flag' else 0)
    assert engine._DAF_CHIMERA_CFG['filter']  # replay does not change caller policy
