"""``--chimeras split`` with the standard (default) chimera detector.

The standard detector flags the reads (same names and counts as flag mode);
the strict segmentation, run on the standard detector's events and
thresholds, finds the breakpoints of the flagged records only. Synthetic
data (helpers shared with tests/test_chimera_strict_qc_split.py).
"""
import json
import random
import subprocess
import sys

import pysam
import pytest

from fiberhmm.daf.encoder import (
    daf_chimera_segments, get_daf_positions, is_daf_chimera, is_daf_chimera_strict,
    md_matches_cigar,
)
from fiberhmm.inference.chimeras import (
    STRICT_MIN_SPAN, ChimeraDecisions, is_marked_chimera, prepare_chimeras,
    standard_breakpoints,
)
from test_chimera_strict_qc_split import (
    _calls, _expected_calls, _split_fixture, _with_modifications, clean, header,
    make_read, records, reference, swap_at, write_bam,
)


@pytest.fixture(autouse=True)
def standard_policy():
    from fiberhmm.inference import engine
    engine.configure_daf_chimera_filter()
    engine.configure_daf_snp_mask(None)
    engine.configure_daf_insert_evidence(None)
    yield
    engine.configure_daf_chimera_filter()
    engine.configure_daf_snp_mask(None)
    engine.configure_daf_insert_evidence(None)


@pytest.mark.parametrize('min_seg,purity', [(5, 0.8), (3, 0.7), (8, 0.95)])
def test_breakpoint_finder_agrees_with_the_standard_decision(min_seg, purity):
    """With the standard thresholds, min_span 0 and single-switch cuts, the
    segmentation finds a breakpoint exactly when the standard test flags."""
    rng = random.Random(min_seg * 100 + int(purity * 100))
    agree = flagged = 0
    for _ in range(3000):
        n = rng.randrange(0, 60)
        positions = rng.sample(range(5000), n)
        if rng.random() < 0.5:
            cut = rng.randrange(5000)
            signs = [(1 if p < cut else -1) if rng.random() < 0.9 else rng.choice((1, -1))
                     for p in positions]
        else:
            signs = [rng.choice((1, -1)) for _ in positions]
        ct = [p for p, s in zip(positions, signs) if s == 1]
        ga = [p for p, s in zip(positions, signs) if s == -1]
        hit = is_daf_chimera(ct, ga, min_seg_events=min_seg, purity=purity)
        segments = daf_chimera_segments(ct, ga, min_seg, purity, min_span=0,
                                        multi_switch=False)
        assert hit == (segments is not None)
        agree += 1
        flagged += hit
    assert flagged > 100


def short_swap_read(hdr, ref, name='short_swap', start=8000, length=3000):
    """Six CT events within 150 bp, then six GA events within 150 bp: the
    standard detector (5 / 0.8) flags it, the strict defaults do not."""
    left = [r for r in range(start + 500, start + 650) if ref[r] == 'C'][:6]
    right = [r for r in range(start + 2000, start + 2150) if ref[r] == 'G'][:6]
    events = set(left) | set(right)
    no_events = set(range(start, start + length)) - events
    strand = swap_at(start + 1500)
    return make_read(hdr, ref, name, start, [(0, length)], strand, rate=1.0,
                     no_events=no_events)


def test_standard_split_cuts_reads_the_strict_detector_misses(tmp_path):
    ref = reference()
    hdr = header()
    read = short_swap_read(hdr, ref)
    ct, ga, _ = get_daf_positions(read, force_strand='CT')
    assert len(ct) == len(ga) == 6 and is_daf_chimera(ct, ga)
    assert not is_daf_chimera_strict(ct, ga, min_span=STRICT_MIN_SPAN, multi_switch=True)
    assert standard_breakpoints(read, None, 5, 0.8)['strands'] == ['CT', 'GA']
    reads = [read] + [make_read(hdr, ref, f'clean{i}', 7000 + 100 * i, [(0, 3000)],
                                clean('GA' if i % 2 else 'CT'), seed=i) for i in range(4)]
    bam = write_bam(tmp_path / 'in.bam', reads)
    spool = tmp_path / 'spool.bam'
    stats = prepare_chimeras(bam, mode='split', spool_bam=str(spool), min_read_length=100)
    assert stats['n_chimeric'] == 1 and stats['split']['n_records_split'] == 1
    pieces = sorted((r for r in records(spool) if r.has_tag('xn')),
                    key=lambda r: r.get_tag('xi'))
    assert [p.query_name for p in pieces] == ['short_swap/c0', 'short_swap/c1']
    for piece in pieces:
        p_ct, p_ga, _ = get_daf_positions(piece, force_strand='CT')
        assert not is_daf_chimera(p_ct, p_ga) and (p_ct or p_ga)
        assert md_matches_cigar(piece)


def test_standard_split_writes_valid_pieces_and_leaves_other_reads_unchanged(tmp_path):
    ref = reference()
    bam = _split_fixture(tmp_path, ref)
    flag_names = ChimeraDecisions()
    flag = prepare_chimeras(bam, chimeric_names=flag_names)
    names = ChimeraDecisions()
    spool = tmp_path / 'spool.bam'
    stats = prepare_chimeras(bam, mode='split', min_read_length=500,
                             spool_bam=str(spool), chimeric_names=names)
    # Same flagging decision and counts as flag mode.
    assert set(names) == set(flag_names) == {'swap', 'shortright'}
    for key in ('n_reads_checked', 'n_chimeric', 'fraction', 'n_records_affected',
                'min_seg', 'purity'):
        assert stats[key] == flag[key]
    assert stats['mode'] == 'split' and 'detector' not in stats and 'grade' not in stats
    split = stats['split']
    assert split['flagged_by'] == 'standard'
    assert 'strict segmentation' in split['breakpoints']
    assert (split['n_records_split'], split['n_pieces_written'],
            split['n_pieces_too_short']) == (2, 3, 1)
    out = records(spool)
    assert [r.reference_start for r in out] == sorted(r.reference_start for r in out)
    assert pysam.AlignmentFile(str(spool)).check_index()
    # Records the detector does not flag are written unchanged.
    source = records(bam)
    unflagged = [r.to_string() for r in source if not r.query_name.startswith(('swap', 'shortright'))]
    assert unflagged == [r.to_string() for r in out if not r.has_tag('xn')
                         and r.query_name not in ('swap', 'shortright')]
    # The non-chimeric supplementary of a chimeric name stays whole (flagged
    # downstream by name); the pieces are new, unflagged records.
    assert [r.is_supplementary for r in out if r.query_name == 'swap'] == [True]
    pieces = sorted((r for r in out if r.has_tag('xn') and r.get_tag('xn') == 'swap'),
                    key=lambda r: r.get_tag('xi'))
    assert [p.query_name for p in pieces] == ['swap/c0', 'swap/c1']
    original = next(r for r in source if r.query_name == 'swap' and not r.is_supplementary)
    o_ct, o_ga, _ = get_daf_positions(original, force_strand='CT')
    o_ref = original.get_reference_positions(full_length=True)
    for piece in pieces:
        assert piece.get_tag('xc') == 'split' and not piece.is_qcfail
        assert md_matches_cigar(piece)
        assert len(piece.query_sequence) == len(piece.query_qualities)
        piece.get_aligned_pairs(with_seq=True)
        p_ct, p_ga, _ = get_daf_positions(piece, force_strand='CT')
        assert not is_daf_chimera(p_ct, p_ga)
        p_ref = piece.get_reference_positions(full_length=True)
        lo, hi = piece.reference_start, piece.reference_end
        for p_events, o_events in ((p_ct, o_ct), (p_ga, o_ga)):
            assert {p_ref[q] for q in p_events} == {
                o_ref[q] for q in o_events if lo <= o_ref[q] < hi}
        assert tuple(piece.get_tag('xb')) == tuple(pieces[0].get_tag('xb'))
    assert 'swap/c0' not in names


def test_standard_split_reindexes_modifications(tmp_path):
    ref = reference()
    hdr = header()
    reads = [make_read(hdr, ref, f'clean{i}', 3000, [(0, 3000)], clean('CT'), seed=i)
             for i in range(4)]
    kept = _with_modifications(make_read(hdr, ref, 'kept', 3000, [(0, 3000)],
                                         swap_at(4500), seed=50, flag=16))
    bam = write_bam(tmp_path / 'mods.bam', reads + [kept])
    stats = prepare_chimeras(bam, mode='split', spool_bam=str(tmp_path / 'spool.bam'),
                             min_read_length=100)
    assert stats['split']['n_pieces_modifications_kept'] == 2
    out = {r.query_name: r for r in records(tmp_path / 'spool.bam')}
    original = next(r for r in records(bam) if r.query_name == 'kept')
    for name in ('kept/c0', 'kept/c1'):
        piece = out[name]
        offset = original.query_sequence.find(piece.query_sequence)
        assert offset >= 0 and piece.is_reverse and piece.get_tag('MN') == piece.query_length
        expected = _expected_calls(original, offset, offset + piece.query_length)
        assert ({k: v for k, v in _calls(piece).items() if v}
                == {k: v for k, v in expected.items() if v})


def _call(source, target, *extra):
    command = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', str(source),
               '-o', str(target), '--enzyme', 'dddb', '-c', '1', '--io-threads', '1',
               '--no-dedup', '--no-daf-call-snps', '--daf-insert-consensus', 'off',
               '--qc-sample-reads', '400', *extra]
    run = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    return run


def test_call_split_with_the_standard_detector_end_to_end(tmp_path):
    ref = reference()
    bam = _split_fixture(tmp_path, ref, gap_without_events=400)
    flagged_bam = tmp_path / 'flag.bam'
    _call(bam, flagged_bam, '--min-read-length', '500', '--no-qc')
    target = tmp_path / 'calls.bam'
    run = _call(bam, target, '--chimeras', 'split', '--min-read-length', '500')
    assert 'split: 2 records cut into 3 pieces' in run.stderr
    out = records(target)
    pieces = [r for r in out if r.has_tag('xi')]
    assert len(pieces) == 3 and not any(r.is_qcfail for r in pieces)
    assert all(r.has_tag('MA') or r.has_tag('ns') for r in pieces)
    flagged = [r for r in out if is_marked_chimera(r)]
    assert [r.query_name for r in flagged] == ['swap'] and flagged[0].is_supplementary
    # Reads the standard detector does not flag: the same records as flag mode.
    flag_out = records(flagged_bam)
    untouched = lambda rs: sorted(r.to_string() for r in rs
                                  if r.query_name.startswith('clean'))
    assert untouched(out) == untouched(flag_out) and untouched(out)
    ds = pysam.AlignmentFile(str(target)).header.to_dict()['PG'][-1]['DS']
    assert 'chimeras=split' in ds and 'chimera_split_breakpoints=strict_segmentation' in ds
    assert 'chimera_detector=strict' not in ds
    flag_ds = pysam.AlignmentFile(str(flagged_bam)).header.to_dict()['PG'][-1]['DS']
    assert 'chimera_split_breakpoints' not in flag_ds
    sidecar = json.loads((tmp_path / 'qc/calls.chimeras.json').read_text())['statistics']
    assert sidecar['split']['n_pieces_written'] == 3
    assert sidecar['split']['flagged_by'] == 'standard'
    qc = json.loads((tmp_path / 'qc/calls.qc.json').read_text())
    assert qc['chimeras']['mode'] == 'split' and qc['chimeras']['split']
    region_parallel = tmp_path / 'rp.bam'
    _call(bam, region_parallel, '--chimeras', 'split', '--min-read-length', '500',
          '--region-parallel', '--no-qc')
    assert [r.to_string() for r in records(region_parallel)] == [r.to_string() for r in out]
