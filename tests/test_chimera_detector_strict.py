"""The strict (opt-in, ``--chimera-detector strict``) DAF strand-swap
detector: base quality, span, several switches, the chimera-only error mask,
duplicates and provenance.

All data are synthetic: a random reference and reads whose C->T / G->A
conversions and base qualities are placed by the test.
"""
import functools
import json
import random
import subprocess
import sys

import pysam
import pytest

from fiberhmm.daf.encoder import daf_chimera_segments, is_daf_chimera_strict
from fiberhmm.inference.chimeras import (
    STRICT_MIN_BASEQ, STRICT_MIN_SPAN, ERROR_MASK_MAX_PVALUE, ERROR_MASK_MIN_DEPTH,
    detector_config, filter_chimera_events, prepare_chimeras, STRICT_MIN_SEG, STRICT_PURITY,
)

# The strict detector is opt-in (--chimera-detector strict).
strict_prepare = functools.partial(prepare_chimeras, detector='strict')

CONTIG = 'chrS'
REF_LEN = 12_000
START = 1000
LENGTH = 5000


@pytest.fixture(autouse=True)
def extraction_policy():
    from fiberhmm.inference import engine
    engine.configure_daf_chimera_filter(True, STRICT_MIN_SEG, STRICT_PURITY, detector='strict')
    engine.configure_daf_snp_mask(None)
    engine.configure_daf_insert_evidence(None)
    yield
    engine.configure_daf_chimera_filter()
    engine.configure_daf_snp_mask(None)
    engine.configure_daf_insert_evidence(None)


def reference(seed=5):
    rng = random.Random(seed)
    return ''.join(rng.choice('ACGT') for _ in range(REF_LEN))


REF = reference()


def header():
    return pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate'},
        'SQ': [{'SN': CONTIG, 'LN': REF_LEN}]})


HDR = header()


def build(name, strand_at, *, start=START, length=LENGTH, rate=0.35, seed=0,
          force=(), skip=(), low_q=(), flag=0, iupac=False):
    """An all-M read over REF with MD/NM and QUAL 40 (``low_q``: Q 5).

    ``strand_at(ref_pos)`` is 'CT', 'GA' or None; target bases convert with
    probability ``rate``. Reference positions in ``force`` always convert
    (C->T at a C, G->A at a G), those in ``skip`` never do. With ``iupac``
    the dominant CT conversions are written as Y (R/Y input).
    """
    rng = random.Random(seed)
    seq, md, run, nm = [], [], 0, 0
    for r in range(start, start + length):
        base = REF[r]
        out = base
        strand = strand_at(r)
        convert = r in force or (r not in skip and rng.random() < rate)
        if convert and base == 'C' and (strand == 'CT' or r in force):
            out = 'T'
        elif convert and base == 'G' and (strand == 'GA' or r in force):
            out = 'A'
        if out != base:
            md += [str(run), base]
            run = 0
            nm += 1
        else:
            run += 1
        seq.append('Y' if iupac and out == 'T' and base == 'C' else out)
    md.append(str(run))
    read = pysam.AlignedSegment(HDR)
    read.query_name = name
    read.flag = flag
    read.reference_id = 0
    read.reference_start = start
    read.mapping_quality = 60
    read.cigartuples = [(0, length)]
    read.query_sequence = ''.join(seq)
    read.query_qualities = pysam.qualitystring_to_array(''.join(
        chr(33 + (5 if start + q in low_q else 40)) for q in range(length)))
    read.set_tag('NM', nm)
    read.set_tag('MD', ''.join(md))
    return read


def write_bam(path, reads):
    reads = sorted(reads, key=lambda x: (x.reference_start, x.query_name))
    with pysam.AlignmentFile(str(path), 'wb', header=HDR) as out:
        for read in reads:
            out.write(read)
    pysam.index(str(path))
    return str(path)


def write_fasta(path):
    path.write_text(f'>{CONTIG}\n' + '\n'.join(
        REF[i:i + 80] for i in range(0, REF_LEN, 80)) + '\n')
    pysam.faidx(str(path))
    return str(path)


def records(path):
    with pysam.AlignmentFile(str(path), 'rb', check_sq=False) as bam:
        return list(bam.fetch(until_eof=True))


def sites(base, lo, hi, step=1):
    return [r for r in range(lo, hi) if REF[r] == base][::step]


def ct_with_ga_tail(tail_sites, **kw):
    """A CT read whose only GA events are at ``tail_sites`` (forced)."""
    def make(name, force=(), **extra):
        return build(name, lambda r: 'CT' if r < START + 4000 else None,
                     force=set(tail_sites) | set(force), **{**kw, **extra})
    return make


TAIL = sites('G', START + 4100, START + 4900, step=4)[:15]   # spans > 200 bp


# --- base quality -----------------------------------------------------------

def test_low_quality_events_do_not_make_a_chimera(tmp_path):
    make = ct_with_ga_tail(TAIL)
    bam = write_bam(tmp_path / 'q.bam', [make('lowq', low_q=set(TAIL)),
                                         make('highq', seed=1)])
    stats = strict_prepare(bam, mode='flag')
    assert (stats['n_reads_checked'], stats['n_chimeric']) == (2, 1)
    off = strict_prepare(bam, mode='flag', min_baseq=0)
    assert off['n_chimeric'] == 2 and off['min_baseq'] == 0


def test_base_quality_filter_on_the_iupac_path(tmp_path):
    make = ct_with_ga_tail(TAIL, iupac=True)
    bam = write_bam(tmp_path / 'ry.bam', [make('lowq', low_q=set(TAIL)),
                                          make('highq', seed=1)])
    assert 'Y' in records(bam)[0].query_sequence
    assert strict_prepare(bam, mode='flag')['n_chimeric'] == 1
    assert strict_prepare(bam, mode='flag', min_baseq=0)['n_chimeric'] == 2


@pytest.mark.parametrize('iupac', [False, True])
def test_engine_filter_applies_base_quality(iupac):
    from fiberhmm.inference import engine
    make = ct_with_ga_tail(TAIL, iupac=iupac)
    low, high = make('lowq', low_q=set(TAIL)), make('highq', seed=1)
    engine.configure_daf_chimera_filter(True, STRICT_MIN_SEG, STRICT_PURITY, detector='strict')
    assert engine._extract_fiber_read_from_pysam(high, 'daf', 128) is engine.CHIMERA_SKIP
    assert engine._extract_fiber_read_from_pysam(low, 'daf', 128) is not engine.CHIMERA_SKIP
    engine.configure_daf_chimera_filter(True, STRICT_MIN_SEG, STRICT_PURITY, detector='strict', min_baseq=0)
    assert engine._extract_fiber_read_from_pysam(low, 'daf', 128) is engine.CHIMERA_SKIP
    # A slim worker payload carries the live record's decision.
    engine.configure_daf_chimera_filter(True, STRICT_MIN_SEG, STRICT_PURITY, detector='strict')
    for read, expected in ((low, False), (high, True)):
        payload = engine.make_apply_payload(read, 'daf')
        assert payload['_daf_chimera'] is expected
        got = engine.extract_fiber_read_from_payload(payload, 'daf', 128)
        assert (got is engine.CHIMERA_SKIP) is expected


def test_no_quality_filter_without_qual():
    read = ct_with_ga_tail(TAIL)('noqual', low_q=set(TAIL))
    read.query_qualities = None
    cfg = detector_config()
    ct, ga = [10, 20], [30]
    assert filter_chimera_events(read, ct, ga, cfg) == (ct, ga)


def test_split_breakpoint_ignores_low_quality_events(tmp_path):
    # One stray low-quality GA event in the CT segment, right before the
    # switch: with the filter the right piece starts at the first good GA.
    stray = sites('G', START + 3900, START + 4000)[-1]
    make = ct_with_ga_tail(TAIL)
    bam = write_bam(tmp_path / 's.bam', [make('r', low_q={stray}, force={stray})])
    for min_baseq, first in ((STRICT_MIN_BASEQ, TAIL[0]), (0, None)):
        stats = strict_prepare(bam, mode='split', min_baseq=min_baseq,
                                 spool_bam=str(tmp_path / f'sp{min_baseq}.bam'))
        assert stats['split']['n_records_split'] == 1
        pieces = records(tmp_path / f'sp{min_baseq}.bam')
        right = next(p for p in pieces if p.get_tag('xi') == 1)
        if first is not None:
            assert right.reference_start == first
        else:
            assert right.reference_start <= stray


# --- span -------------------------------------------------------------------

def test_minimum_span():
    ct = list(range(0, 3000, 25))
    dense = [3100 + 4 * i for i in range(12)]         # 12 GA events in 44 bp
    assert is_daf_chimera_strict(ct, dense, 10, 0.9)
    assert not is_daf_chimera_strict(ct, dense, 10, 0.9, min_span=STRICT_MIN_SPAN)
    spread = [3100 + 25 * i for i in range(12)]       # 275 bp
    assert is_daf_chimera_strict(ct, spread, 10, 0.9, min_span=STRICT_MIN_SPAN)
    # The span is measured on the dominant events of each segment.
    short_ct = list(range(0, 150, 10))
    assert not is_daf_chimera_strict(short_ct, spread, 10, 0.9, min_span=STRICT_MIN_SPAN)


# --- several switches -------------------------------------------------------

def test_multi_switch_detection_and_segments():
    ct = list(range(0, 1000, 10)) + list(range(2000, 3000, 10))
    island = list(range(1100, 1900, 20))
    assert not is_daf_chimera_strict(ct, island, 10, 0.9)
    assert is_daf_chimera_strict(ct, island, 10, 0.9, multi_switch=True,
                          min_span=STRICT_MIN_SPAN)
    seg = daf_chimera_segments(ct, island, 10, 0.9, multi_switch=True)
    assert seg == {'pieces': [(None, 990), (1100, 1880), (2000, None)],
                   'strands': ['CT', 'GA', 'CT']}
    # A small island is not a segment.
    assert not is_daf_chimera_strict(ct, island[:5], 10, 0.9, multi_switch=True)
    # Four segments: every piece is found.
    ga = island + list(range(3100, 4000, 10))
    seg = daf_chimera_segments(ct, ga, 10, 0.9, multi_switch=True)
    assert seg['strands'] == ['CT', 'GA', 'CT', 'GA']
    # One switch keeps the single best breakpoint of daf_chimera_split.
    seg = daf_chimera_segments(list(range(0, 1000, 10)), island, 10, 0.9,
                               multi_switch=True)
    assert seg['pieces'] == [(None, 990), (1100, None)]


def _island_read(name='island', **kw):
    lo, hi = START + 2000, START + 3000
    return build(name, lambda r: 'GA' if lo <= r < hi else 'CT', **kw)


def test_multi_switch_flag_and_recursive_split(tmp_path):
    bam = write_bam(tmp_path / 'i.bam', [_island_read()])
    assert strict_prepare(bam, mode='flag')['n_chimeric'] == 1
    assert strict_prepare(bam, mode='flag', multi_switch=False)['n_chimeric'] == 0
    spool = tmp_path / 'spool.bam'
    stats = strict_prepare(bam, mode='split', spool_bam=str(spool))
    split = stats['split']
    assert (split['n_records_split'], split['n_pieces_written']) == (1, 3)
    assert split['n_records_several_switches'] == 1
    assert split['n_pieces_still_chimeric'] == 0
    pieces = sorted(records(spool), key=lambda p: p.get_tag('xi'))
    assert [p.query_name for p in pieces] == ['island/c0', 'island/c1', 'island/c2']
    assert {p.get_tag('xn') for p in pieces} == {'island'}
    gaps = [list(p.get_tag('xb')) for p in pieces]
    assert all(g == gaps[0] for g in gaps) and len(gaps[0]) == 4
    g0s, g0e, g1s, g1e = gaps[0]
    assert pieces[0].reference_end == g0s and pieces[1].reference_start == g0e
    assert pieces[1].reference_end == g1s and pieces[2].reference_start == g1e
    assert g0s <= START + 2000 <= g0e and g1s <= START + 3000 <= g1e
    for piece in pieces:
        ct, ga = filter_chimera_events(piece, *_events(piece), detector_config())
        assert not is_daf_chimera_strict(ct, ga, 10, 0.9, multi_switch=True)
        assert (len(ga) > len(ct)) == (piece.get_tag('xi') == 1)


def _events(read):
    from fiberhmm.daf.encoder import get_daf_positions
    ct, ga, _ = get_daf_positions(read, force_strand='CT')
    return ct, ga


# --- duplicates ---------------------------------------------------------------

def test_duplicates_are_marked_but_not_counted(tmp_path):
    make = ct_with_ga_tail(TAIL)
    reads = [build(f'clean{i}', lambda r: 'CT', seed=i) for i in range(6)]
    reads += [make(f'chim{i}', seed=10 + i) for i in range(2)]
    reads += [make(f'dup{i}', seed=20 + i, flag=0x400) for i in range(3)]
    bam = write_bam(tmp_path / 'd.bam', reads)
    out = tmp_path / 'out.bam'
    stats = strict_prepare(bam, str(out), mode='flag')
    assert (stats['n_reads_checked'], stats['n_chimeric']) == (8, 2)
    assert stats['n_duplicates_excluded'] == 3
    assert stats['fraction'] == pytest.approx(2 / 8)
    assert 'duplicate' in stats['duplicates']
    marked = {r.query_name for r in records(out) if r.has_tag('xc')}
    assert marked == {'chim0', 'chim1', 'dup0', 'dup1', 'dup2'}


def test_sample_qc_leaves_duplicates_out():
    from fiberhmm.inference.chimeras import mark_chimera
    from fiberhmm.qc.core import analyze_sample
    make = ct_with_ga_tail(TAIL)
    reads = [build(f'clean{i}', lambda r: 'CT', seed=i) for i in range(4)]
    reads += [make('chim', seed=10), make('dup', seed=11, flag=0x400)]
    for read in reads[-2:]:
        mark_chimera(read)
    chim = analyze_sample(reads, 'daf', None, strict_chimeras=True)[0]['chimeras']
    assert (chim['n_reads_checked'], chim['n_chimeric']) == (5, 1)
    assert chim['n_duplicates_excluded'] == 1
    # A standard-detector BAM is sampled as in 3.2.1: duplicates count.
    standard = analyze_sample(reads, 'daf', None)[0]['chimeras']
    assert (standard['n_reads_checked'], standard['n_chimeric']) == (6, 2)
    assert 'n_duplicates_excluded' not in standard and 'note' not in standard


# --- chimera error mask -------------------------------------------------------

def test_error_mask_binomial_gate():
    from collections import Counter
    from fiberhmm.daf.snps import _error_mask_sites

    def stats(depth, k):
        return Counter(opposite_depth=depth, opposite_mismatches=k)
    table = {('c', 1, 'G'): stats(20, 4),     # p ~ 4.5e-5: masked
             ('c', 2, 'G'): stats(19, 4),     # below the depth floor
             ('c', 3, 'G'): stats(60, 4),     # p ~ 3e-3: random error
             ('c', 4, 'C'): stats(100, 5),    # fraction 0.05 but p ~ 3e-3
             ('c', 5, 'C'): stats(200, 9),    # fraction < 0.05
             ('c', 6, 'C'): stats(40, 8),     # masked
             ('c', 7, 'G'): stats(40, 8)}     # a called SNP: not repeated
    masked = _error_mask_sites(table, set(), {('c', 7, 'G')})
    assert [s['position_0based'] for s in masked] == [1, 6]
    assert all(s['p_value'] < ERROR_MASK_MAX_PVALUE for s in masked)
    assert masked[0]['masked_event'] == 'G>A' and masked[1]['masked_event'] == 'C>T'


def _error_library(tmp_path):
    """Clean CT voters, recurrent G>A errors, and chimeras at a hotspot."""
    region = sites('G', START + 500, START + 3500, step=40)
    error, low_error, hotspot = region[0], region[1], sites('G', START + 4750, START + 4800)[0]
    reads = []
    for i in range(25):
        force = {error} if i < 6 else set()
        force |= {low_error} if i < 6 else set()
        reads.append(build(f'v{i}', lambda r: 'CT', seed=i, force=force,
                           skip={error, low_error, hotspot} - force,
                           low_q={low_error}))
    # Chimeras: CT then a GA tail (purity > 0.92) with a real hotspot that
    # converts on 12 of 30; they must not vote.
    for i in range(30):
        force = {hotspot} if i < 12 else set()
        reads.append(build(f'c{i}', lambda r: 'CT' if r < START + 4700 else 'GA',
                           seed=100 + i, force=force, skip={hotspot} - force))
    return write_bam(tmp_path / 'lib.bam', reads), error, low_error, hotspot


def test_error_mask_voters_and_separate_outputs(tmp_path):
    from fiberhmm.daf.snps import (
        _error_mask_voter, _read_arrays_with_query, call_opposite_conversion_snps,
        write_snp_outputs,
    )
    bam, error, low_error, hotspot = _error_library(tmp_path)
    cfg = detector_config()
    by_name = {r.query_name: r for r in records(bam)}
    chimera = by_name['c0']
    assert _error_mask_voter(chimera, _read_arrays_with_query(chimera), cfg)[0] is None
    voter = by_name['v0']
    assert _error_mask_voter(voter, _read_arrays_with_query(voter), cfg)[0] == 'CT'

    payload = call_opposite_conversion_snps(bam, chimera_error_mask=True,
                                            chimera_detector=cfg)
    mask = payload['chimera_error_mask']
    masked = {s['position_0based'] for s in mask['sites']}
    assert error in masked                      # recurrent, high quality
    assert low_error not in masked              # its events are Q5
    assert hotspot not in masked                # chimeras do not vote
    assert mask['accounting']['ct_voters'] == 25
    site = next(s for s in mask['sites'] if s['position_0based'] == error)
    assert (site['voter_depth'], site['voter_mismatches']) == (25, 6)
    assert site['voter_depth'] >= ERROR_MASK_MIN_DEPTH
    # Without the option nothing is computed.
    assert 'chimera_error_mask' not in call_opposite_conversion_snps(bam)

    written = write_snp_outputs(payload, str(tmp_path / 'qc' / 'x.daf_snps'))
    snp_bed = (tmp_path / 'qc' / 'x.daf_snps.bed').read_text()
    mask_bed = (tmp_path / 'qc' / 'x.daf_snps.chimera_error_mask.bed').read_text()
    assert f'\t{error}\t' not in snp_bed
    assert f'{CONTIG}\t{error}\t{error + 1}\tG>A' in mask_bed
    report = json.loads((tmp_path / 'qc' / 'x.daf_snps.json').read_text())
    assert 'sites' not in report['chimera_error_mask']
    assert report['chimera_error_mask']['n_sites'] == len(masked)
    side = json.loads((tmp_path / 'qc' / 'x.daf_snps.chimera_error_mask.json').read_text())
    assert {s['position_0based'] for s in side['sites']} == masked
    assert written['outputs']['chimera_error_mask_bed'].endswith('.chimera_error_mask.bed')


def test_error_mask_removes_events_from_the_chimera_test_only(tmp_path):
    from fiberhmm.inference import engine
    make = ct_with_ga_tail(TAIL)
    bam = write_bam(tmp_path / 'm.bam', [make('r')])
    bed = tmp_path / 'mask.bed'
    bed.write_text(''.join(f'{CONTIG}\t{s}\t{s + 1}\tG>A\n' for s in TAIL))
    assert strict_prepare(bam, mode='flag')['n_chimeric'] == 1
    stats = strict_prepare(bam, mode='flag', error_mask=str(bed))
    assert stats['n_chimeric'] == 0
    assert stats['error_mask']['applied'] and stats['error_mask']['n_sites'] == len(TAIL)
    # Calling's own mask is untouched: the GA events are still evidence.
    assert not engine._DAF_SNP_MASK
    ct, ga, _ = __import__('fiberhmm.daf.encoder', fromlist=['x']).get_daf_positions(
        records(bam)[0], force_strand='CT',
        excluded_reference_positions=engine._daf_reference_mask(records(bam)[0]))
    assert len(ga) == len(TAIL)


# --- call CLI: provenance, separation, reasons --------------------------------

def _call(source, target, *extra):
    command = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', str(source),
               '-o', str(target), '--enzyme', 'dddb', '-c', '1', '--io-threads', '1',
               '--no-dedup', '--daf-insert-consensus', 'off', '--no-qc',
               '--chimera-detector', 'strict', *extra]
    run = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    return run


def _ds(path):
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        pg = [p for p in bam.header.to_dict()['PG'] if p['PN'] == 'fiberhmm-call'][-1]
    return dict(item.split('=', 1) for item in pg['DS'].split() if '=' in item)


def test_call_records_detector_provenance_and_keeps_masks_apart(tmp_path):
    bam, error, _low, _hot = _error_library(tmp_path)
    _call(bam, tmp_path / 'calls.bam', '--daf-call-snps')
    side = json.loads((tmp_path / 'qc/calls.chimeras.json').read_text())['statistics']
    mask = side['error_mask']
    assert mask['applied'] and mask['n_sites'] >= 1
    assert mask['bed'].endswith('calls.daf_snps.chimera_error_mask.bed')
    assert (side['min_seg'], side['purity'], side['min_baseq'], side['min_span'],
            side['multi_switch']) == (10, 0.9, 15, 200, True)
    assert side['n_duplicates_excluded'] == 0
    ds = _ds(tmp_path / 'calls.bam')
    assert ds['chimera_min_baseq'] == '15' and ds['chimera_min_span'] == '200'
    assert ds['chimera_multi_switch'] == 'on'
    assert ds['chimera_error_mask'] == f"on/{mask['n_sites']}sites"
    from fiberhmm.io.run_state import sha256_file
    digest = sha256_file(tmp_path / 'qc/calls.daf_snps.chimera_error_mask.bed')
    assert mask['sha256'] == digest and ds['chimera_error_mask_sha256'] == digest
    assert ds['chimera_duplicates_excluded'] == '0'
    snp_bed = (tmp_path / 'qc/calls.daf_snps.bed').read_text()
    assert f'\t{error}\t' not in snp_bed
    assert f'\t{error}\t' in (tmp_path / 'qc/calls.daf_snps.chimera_error_mask.bed').read_text()
    assert ds['daf_snp_mask'] == f"on/{len(snp_bed.splitlines()) - 1}sites"


@pytest.mark.parametrize('extra, reason, state', [
    (('--no-daf-call-snps',), '--no-daf-call-snps', 'unavailable'),
    (('--no-chimera-error-mask', '--daf-call-snps'), 'disabled', 'off'),
])
def test_call_without_error_mask_says_why(tmp_path, extra, reason, state):
    make = ct_with_ga_tail(TAIL)
    bam = write_bam(tmp_path / 'in.bam', [make(f'r{i}', seed=i) for i in range(3)])
    _call(bam, tmp_path / 'calls.bam', *extra)
    side = json.loads((tmp_path / 'qc/calls.chimeras.json').read_text())['statistics']
    assert side['error_mask']['applied'] is False
    assert reason in side['error_mask']['reason']
    assert side['min_baseq'] == 15             # the quality filter still applies
    assert _ds(tmp_path / 'calls.bam')['chimera_error_mask'] == state
    assert 'chimera_error_mask_sha256' not in _ds(tmp_path / 'calls.bam')
    assert side['error_mask']['sha256'] is None
    assert not (tmp_path / 'qc/calls.daf_snps.chimera_error_mask.bed').exists()


@pytest.mark.parametrize('min_baseq', [0, 15, 30])
def test_error_mask_voters_use_fixed_q15(tmp_path, min_baseq):
    """Voter quality is 15 whatever --chimera-min-baseq is."""
    from fiberhmm.daf.snps import (
        _error_mask_voter, _read_arrays_with_query, call_opposite_conversion_snps,
    )
    region = range(START, START + LENGTH)
    all_low = build('lowq', lambda r: 'CT', seed=3, low_q=set(region))
    cfg = detector_config(min_baseq=min_baseq)
    assert _error_mask_voter(all_low, _read_arrays_with_query(all_low), cfg) == (None, None)
    good = build('good', lambda r: 'CT', seed=4)
    assert _error_mask_voter(good, _read_arrays_with_query(good), cfg)[0] == 'CT'
    # Q5 recurrent errors never make a mask site, at any detector setting.
    bam, error, low_error, _hot = _error_library(tmp_path)
    payload = call_opposite_conversion_snps(bam, chimera_error_mask=True,
                                            chimera_detector=cfg)
    masked = {s['position_0based'] for s in payload['chimera_error_mask']['sites']}
    assert error in masked and low_error not in masked
    # Provenance records the effective voter quality, and the detector's apart.
    params = payload['chimera_error_mask']['parameters']
    assert params['voter_min_baseq'] == 15
    assert params['voter_detector']['min_baseq'] == 15
    assert params['detector_min_baseq'] == min_baseq


# --- QC with state processing on ----------------------------------------------

def _call_header(ds):
    return pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate'},
        'SQ': [{'SN': CONTIG, 'LN': REF_LEN}],
        'PG': [{'ID': 'fiberhmm-call', 'PN': 'fiberhmm-call', 'VN': '3.3.0',
                'DS': ds}]})


_DS_DEFAULT = ('mode=daf enzyme=dddb chimera_filter=on chimeras=flag chimera_min_seg=10 '
               'chimera_detector=strict '
               'chimera_purity=0.9 chimera_min_baseq=15 chimera_min_span=200 '
               'chimera_multi_switch=on chimera_error_mask={mask} '
               'daf_run_mask=off')


def _write_with_header(path, reads, hdr):
    reads = sorted(reads, key=lambda x: (x.reference_start, x.query_name))
    with pysam.AlignmentFile(str(path), 'wb', header=hdr) as out:
        for read in reads:
            out.write(pysam.AlignedSegment.fromstring(read.to_string(), hdr))
    pysam.index(str(path))
    return str(path)


def _qc(bam, tmp_path, name):
    from fiberhmm.qc.core import run_qc
    return run_qc(bam, output_prefix=str(tmp_path / 'out' / name), mode='daf',
                  enzyme='dddb', reference_profile='none', sample_reads=100,
                  state_source='light-call', stream=None)


def test_state_qc_leaves_duplicate_chimeras_out(tmp_path):
    from fiberhmm.inference.chimeras import mark_chimera
    make = ct_with_ga_tail(TAIL)
    reads = [build('clean', lambda r: 'CT', seed=1)]
    for i in range(3):
        dup = make(f'dup{i}', seed=20 + i, flag=0x400)
        mark_chimera(dup)
        reads.append(dup)
    # One chimera with two marked records counts once.
    chim = make('chim', seed=30)
    supp = make('chim', seed=30, flag=0x800)
    for read in (chim, supp):
        mark_chimera(read)
    reads += [chim, supp]
    hdr = _call_header(_DS_DEFAULT.format(mask='off'))
    bam = _write_with_header(tmp_path / 'dups.bam', reads, hdr)
    result = _qc(bam, tmp_path, 'dups')
    states = result['state_rates']
    assert states['source'] == 'light_call'
    assert states['reads']['reads_chimera_skipped'] == 1
    chimeras = result['chimeras']
    assert (chimeras['n_chimeric'], chimeras['n_reads_checked']) == (1, 2)
    assert chimeras['n_duplicates_excluded'] == 3


def _mask_bed(sites):
    return ''.join(f'{CONTIG}\t{s}\t{s + 1}\tG>A\n' for s in sites)


def _sha256(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


# The call's mask, and another run's mask with the same number of sites.
_CALL_MASK = _mask_bed(TAIL)
_OTHER_MASK = _mask_bed([s + 1 for s in TAIL])


def _masked_ds(digest=_sha256(_CALL_MASK)):
    ds = _DS_DEFAULT.format(mask=f'on/{len(TAIL)}sites')
    return ds + (f' chimera_error_mask_sha256={digest}' if digest else '')


@pytest.mark.parametrize('bed_text, digest, state', [
    (_CALL_MASK, _sha256(_CALL_MASK), 'restored'),
    (None, _sha256(_CALL_MASK), 'unavailable'),
    (_OTHER_MASK, _sha256(_CALL_MASK), 'mismatch'),      # same site count
    (_CALL_MASK, None, 'unverified'),                    # no digest recorded
])
def test_standalone_state_qc_restores_the_calls_error_mask(tmp_path, bed_text, digest,
                                                           state):
    from fiberhmm.inference import engine
    make = ct_with_ga_tail(TAIL)
    # Chimeric under the default detector, rescued by the call's error mask,
    # so the call kept it unmarked.
    reads = [make('rescued', seed=3), build('clean', lambda r: 'CT', seed=4)]
    hdr = _call_header(_masked_ds(digest))
    bam = _write_with_header(tmp_path / 'calls.bam', reads, hdr)
    with_bed = state == 'restored'
    if bed_text is not None:
        bed = tmp_path / 'qc' / 'calls.daf_snps.chimera_error_mask.bed'
        bed.parent.mkdir()
        bed.write_text(bed_text)
    before = dict(engine._DAF_CHIMERA_CFG)
    result = _qc(bam, tmp_path, 'calls')
    assert engine._DAF_CHIMERA_CFG == before            # restored after QC
    states = result['state_rates']
    assert states['source'] == 'light_call'
    detector = states['definition']['chimera_detector']
    assert (detector['min_baseq'], detector['min_span'], detector['multi_switch']) == (
        15, 200, True)
    if with_bed:
        assert states['definition']['chimera_error_mask'] == 'restored'
        assert result['chimeras']['error_mask_restored'] == 'restored'
        assert detector['error_mask'].endswith('calls.daf_snps.chimera_error_mask.bed')
        assert states['reads']['reads_chimera_skipped'] == 0
    else:
        # Not this call's mask (or none, or unverifiable): not restored.
        assert states['definition']['chimera_error_mask'] == state
        assert result['chimeras']['error_mask_restored'] == state
        assert result['chimeras']['error_mask_not_restored'] == state
        assert detector['error_mask'] is None
        assert states['reads']['reads_chimera_skipped'] == 1
        from fiberhmm.qc.core import format_terminal
        assert f'error mask not restored ({state}' in format_terminal(result)


def test_standalone_state_qc_restores_the_calls_detector_settings(tmp_path):
    make = ct_with_ga_tail(TAIL)
    # Its GA tail is Q5: a chimera only for a call made with base quality 0.
    reads = [make('lowq', low_q=set(TAIL)), build('clean', lambda r: 'CT', seed=4)]
    ds = _DS_DEFAULT.format(mask='off').replace('chimera_min_baseq=15',
                                                'chimera_min_baseq=0')
    bam = _write_with_header(tmp_path / 'q0.bam', reads, _call_header(ds))
    states = _qc(bam, tmp_path, 'q0')['state_rates']
    assert states['definition']['chimera_detector']['min_baseq'] == 0
    assert states['reads']['reads_chimera_skipped'] == 1
    # A standard-detector call (no chimera_detector=strict in DS) keeps the
    # process settings and is reported as in 3.2.1: no detector restored,
    # no grade, no note.
    standard = ('mode=daf enzyme=dddb chimera_filter=on chimeras=flag chimera_min_seg=5 '
                'chimera_purity=0.8 daf_run_mask=off')
    bam = _write_with_header(tmp_path / 'old.bam', reads, _call_header(standard))
    result = _qc(bam, tmp_path, 'old')
    assert 'chimera_detector' not in result['state_rates']['definition']
    assert 'grade' not in result['chimeras'] and 'note' not in result['chimeras']
    assert not (tmp_path / 'out' / 'old.qc.chimera_regions.tsv').exists()


@pytest.mark.parametrize('valid, replaced', [(True, False), (False, False),
                                             (True, True)])
def test_standalone_state_qc_finds_a_custom_prefix_mask_via_the_sidecar(tmp_path, valid,
                                                                       replaced):
    """--daf-snp-output-prefix elsewhere, no --snp-report: the BAM-validated
    chimeras sidecar names the mask; an unvalidated one is not trusted, nor
    a BED another run wrote over it (same site count, other content)."""
    from fiberhmm.io.run_state import sha256_file
    make = ct_with_ga_tail(TAIL)
    reads = [make('rescued', seed=3), build('clean', lambda r: 'CT', seed=4)]
    hdr = _call_header(_DS_DEFAULT.format(mask=f'on/{len(TAIL)}sites'))
    bam = _write_with_header(tmp_path / 'calls.bam', reads, hdr)
    bed = tmp_path / 'custom' / 'screen.chimera_error_mask.bed'
    bed.parent.mkdir()
    bed.write_text(''.join(f'{CONTIG}\t{s}\t{s + 1}\tG>A\n' for s in TAIL))
    stats = strict_prepare(bam, mode='flag', error_mask=str(bed))
    assert stats['error_mask']['bed'] == str(bed) and stats['n_chimeric'] == 0
    assert stats['error_mask']['sha256'] == sha256_file(bed)
    if replaced:
        bed.write_text(_OTHER_MASK)
    (tmp_path / 'qc').mkdir()
    (tmp_path / 'qc' / 'calls.chimeras.json').write_text(json.dumps({
        'assay': {'mode': 'daf', 'enzyme': 'dddb'},
        'output_sha256': sha256_file(bam) if valid else '0' * 64,
        'statistics': stats}))
    states = _qc(bam, tmp_path, 'calls')['state_rates']
    if replaced:
        assert states['definition']['chimera_error_mask'] == 'mismatch'
        assert states['definition']['chimera_detector']['error_mask'] is None
        assert states['reads']['reads_chimera_skipped'] == 1
    elif valid:
        assert states['definition']['chimera_error_mask'] == 'restored'
        assert states['definition']['chimera_detector']['error_mask'] == str(bed)
        assert states['reads']['reads_chimera_skipped'] == 0
    else:
        assert states['definition']['chimera_error_mask'] == 'unavailable'
        assert states['reads']['reads_chimera_skipped'] == 1


# --- the option: standard by default, strict opt-in ---------------------------

def _flags_fixture(path):
    from test_chimera_flags import fixture_bam
    return fixture_bam(path)


def _call_args(monkeypatch, *argv):
    from fiberhmm.cli import call
    monkeypatch.setattr(sys, 'argv', ['fiberhmm-call', '-i', 'in.bam', '-o', 'out.bam',
                                      '--enzyme', 'dddb', *argv])
    return call.parse_args()


def test_standard_is_the_default_and_strict_fills_its_defaults(monkeypatch):
    args = _call_args(monkeypatch)
    assert args.chimera_detector == 'standard'
    assert (args.chimera_min_seg, args.chimera_purity) == (5, 0.8)
    assert args.chimera_min_baseq is None and args.chimera_error_mask is None
    args = _call_args(monkeypatch, '--chimera-detector', 'strict')
    assert (args.chimera_min_seg, args.chimera_purity, args.chimera_min_baseq,
            args.chimera_min_span, args.chimera_multi_switch,
            args.chimera_error_mask) == (10, 0.9, 15, 200, True, True)
    args = _call_args(monkeypatch, '--chimera-detector', 'strict', '--chimera-min-seg', '7',
                      '--no-chimera-error-mask')
    assert args.chimera_min_seg == 7 and args.chimera_error_mask is False


@pytest.mark.parametrize('argv', [
    ('--chimera-min-baseq', '20'), ('--chimera-min-span', '0'),
    ('--chimera-multi-switch',), ('--no-chimera-error-mask',),
    ('--chimeras', 'split'),
])
def test_strict_only_options_need_the_strict_detector(monkeypatch, capsys, argv):
    with pytest.raises(SystemExit):
        _call_args(monkeypatch, *argv)
    assert '--chimera-detector strict' in capsys.readouterr().err


def test_prepare_chimeras_standard_refuses_strict_options(tmp_path):
    source = str(_flags_fixture(tmp_path / 'source.bam'))
    with pytest.raises(ValueError, match='strict chimera detector only'):
        prepare_chimeras(source, min_baseq=15)
    with pytest.raises(ValueError, match='requires the strict detector'):
        prepare_chimeras(source, mode='split', spool_bam=str(tmp_path / 's.bam'))
    stats = prepare_chimeras(source)
    assert (stats['min_seg'], stats['purity']) == (5, 0.8)
    assert 'detector' not in stats and 'grade' not in stats
    strict = strict_prepare(source, min_span=0, min_baseq=0)
    assert strict['detector'] == 'strict' and strict['min_seg'] == 10


def test_standard_and_strict_detectors_differ_on_a_short_swap():
    """The standard single-switch test (5 / 0.8) calls a swap of five events
    per segment; the strict defaults (10 / 0.9, 200 bp span) do not."""
    from fiberhmm.daf.encoder import is_daf_chimera
    ct = [10, 20, 30, 40, 50]
    ga = [200, 210, 220, 230, 240]
    assert is_daf_chimera(ct, ga)
    assert not is_daf_chimera_strict(ct, ga, min_span=STRICT_MIN_SPAN, multi_switch=True)
    assert is_daf_chimera_strict(ct, ga, 5, 0.8)


@pytest.mark.parametrize('reverse', [False, True])
def test_strict_balanced_raw_chimera(tmp_path, reverse):
    """A 20-bp balanced raw chimera: the strict detector with its minimum
    span off (the read is shorter than 200 bp) tests the strand logic."""
    from fiberhmm.daf.encoder import get_daf_positions
    from fiberhmm.inference.chimeras import _is_chimeric_strict
    from test_chimera_flags import fixture_bam, records as records303
    read = records303(fixture_bam(tmp_path / 'source.bam'))[-1]
    read.query_sequence = 'T' * 10 + 'A' * 10
    read.cigarstring = '20M'
    read.flag = 16 if reverse else 0
    read.set_tag('MD', '0' + 'C0' * 10 + 'G0' * 10)
    ct, ga, _ = get_daf_positions(read, force_strand='CT')
    assert is_daf_chimera_strict(ct, ga, 10, 0.9)
    assert _is_chimeric_strict(read, None, detector_config(min_span=0))
    source, target = tmp_path / 'raw.bam', tmp_path / 'out.bam'
    with pysam.AlignmentFile(source, 'wb', header=read.header) as out:
        out.write(read)
    stats = strict_prepare(str(source), str(target), min_span=0)
    assert stats['n_chimeric'] == 1
    assert records303(target)[0].is_qcfail
    strict_prepare(str(source), str(target), mode='remove', min_span=0)
    assert records303(target) == []


def test_standard_ds_has_no_strict_fields_and_strict_records_itself(tmp_path, monkeypatch):
    from fiberhmm.cli import call
    source = _flags_fixture(tmp_path / 'source.bam')
    seen = {}

    def stop(**kwargs):
        seen.update(kwargs)
        raise SystemExit(7)
    monkeypatch.setattr(call, '_process_bam_streaming_pipeline_fused', stop)
    base = ['fiberhmm-call', '-i', str(source), '-o', str(tmp_path / 'calls.bam'),
            '--enzyme', 'dddb', '--seq', 'nanopore', '-c', '1', '--io-threads', '1',
            '--no-qc', '--no-dedup', '--no-daf-call-snps', '--daf-insert-consensus', 'off']
    from fiberhmm.inference import engine
    try:
        for extra in ([], ['--chimera-detector', 'strict']):
            seen.clear()
            monkeypatch.setattr(sys, 'argv', base + extra)
            with pytest.raises(SystemExit):
                call._main(call.parse_args())
            ds = seen['pg_record']['DS']
            if extra:
                assert 'chimera_detector=strict' in ds and 'chimera_min_baseq=15' in ds
                assert seen['chimera_detector'] == 'strict'
            else:
                assert 'chimera_detector' not in ds and 'chimera_min_baseq' not in ds
                assert 'chimera_detector' not in seen and 'chimera_min_baseq' not in seen
    finally:
        engine.configure_daf_chimera_filter()


# --- every strict setting reaches the workers and is restored -----------------

def test_in_process_call_restores_every_detector_setting(tmp_path, monkeypatch):
    from fiberhmm.cli import call
    from fiberhmm.inference import engine
    source = _flags_fixture(tmp_path / 'source.bam')
    mask = tmp_path / 'prior_mask.bed'
    mask.write_text('chr1\t10\t11\tG>A\n')
    engine.configure_daf_chimera_filter(True, 7, 0.85, detector='strict', min_baseq=30,
                                        min_span=0, multi_switch=False,
                                        error_mask=str(mask))
    policy = dict(engine._DAF_CHIMERA_CFG)
    try:
        monkeypatch.setattr(call, '_process_bam_streaming_pipeline_fused',
                            lambda **kwargs: (0, 0))
        for extra in ([], ['--chimera-detector', 'strict']):
            monkeypatch.setattr(sys, 'argv', [
                'fiberhmm-call', '-i', str(source), '-o', str(tmp_path / 'calls.bam'),
                '--enzyme', 'dddb', '--seq', 'nanopore', '-c', '1', '--io-threads', '1',
                '--no-qc', '--no-dedup', '--no-daf-call-snps',
                '--daf-insert-consensus', 'off', *extra])
            call._main(call.parse_args())
            assert engine._DAF_CHIMERA_CFG == policy
    finally:
        engine.configure_daf_chimera_filter()


_NON_DEFAULT_DETECTOR = dict(min_seg=7, purity=0.85, min_baseq=30, min_span=0,
                             multi_switch=False)


@pytest.mark.parametrize('kind', ['streaming', 'region', 'fused_region'])
def test_worker_init_sees_every_detector_setting(tmp_path, kind):
    from fiberhmm.inference import engine, region_workers, streaming_workers
    from fiberhmm.models import get_model_path
    mask = tmp_path / 'mask.bed'
    mask.write_text('chr1\t10\t11\tG>A\n')
    model = get_model_path('dddb', tool='apply')
    d = _NON_DEFAULT_DETECTOR
    params = {'filter_chimeras': True, 'chimera_detector': 'strict',
              'chimera_min_seg': d['min_seg'],
              'chimera_purity': d['purity'], 'chimera_min_baseq': d['min_baseq'],
              'chimera_min_span': d['min_span'],
              'chimera_multi_switch': d['multi_switch'],
              'chimera_error_mask_path': str(mask)}
    try:
        if kind == 'streaming':
            streaming_workers._init_fused_worker(
                model, None, filter_chimeras=True, chimera_min_seg=d['min_seg'],
                chimera_purity=d['purity'], chimera_min_baseq=d['min_baseq'],
                chimera_min_span=d['min_span'], chimera_multi_switch=d['multi_switch'],
                chimera_error_mask_path=str(mask), chimera_detector='strict')
        elif kind == 'region':
            region_workers._init_region_worker(model, params)
        else:
            region_workers._init_fused_region_worker(model, None, 1.0, params)
        assert engine._DAF_CHIMERA_CFG == {'filter': True, 'detector': 'strict', **d,
                                           'error_mask': str(mask)}
    finally:
        engine.configure_daf_chimera_filter()


@pytest.mark.parametrize('region_parallel', [False, True])
def test_call_passes_every_detector_setting_to_its_workers(tmp_path, monkeypatch,
                                                           region_parallel):
    from fiberhmm.cli import call
    from fiberhmm.inference import region_pipeline, streaming_pipeline
    source = _flags_fixture(tmp_path / 'source.bam')
    seen = {}

    class Stop(Exception):
        pass

    def executor(*args, initializer=None, initargs=(), **kwargs):
        seen['initializer'], seen['initargs'] = initializer, initargs
        raise Stop
    monkeypatch.setattr(streaming_pipeline, 'ProcessPoolExecutor', executor)
    monkeypatch.setattr(region_pipeline, 'ProcessPoolExecutor', executor)
    monkeypatch.setattr(sys, 'argv', [
        'fiberhmm-call', '-i', str(source), '-o', str(tmp_path / 'calls.bam'),
        '--enzyme', 'dddb', '--seq', 'nanopore', '-c', '1', '--io-threads', '1',
        '--no-qc', '--no-dedup', '--no-daf-call-snps', '--daf-insert-consensus', 'off',
        '--chimera-detector', 'strict',
        '--chimera-min-seg', '7', '--chimera-purity', '0.85',
        '--chimera-min-baseq', '30', '--chimera-min-span', '0',
        '--no-chimera-multi-switch', *(['--region-parallel'] if region_parallel else [])])
    try:
        with pytest.raises(Stop):
            call._main(call.parse_args())
    finally:
        from fiberhmm.inference import engine
        engine.configure_daf_chimera_filter()
    d = _NON_DEFAULT_DETECTOR
    if region_parallel:
        initargs = seen['initargs']
        params = next(a for a in initargs if isinstance(a, dict))
        got = {k: params[f'chimera_{k}'] for k in d}
        assert 'chimera_error_mask_path' in params
        assert params['chimera_detector'] == 'strict'
    else:
        import inspect
        bound = inspect.signature(seen['initializer']).bind(*seen['initargs']).arguments
        got = {k: bound[f'chimera_{k}'] for k in d}
        assert 'chimera_error_mask_path' in bound
        assert bound['chimera_detector'] == 'strict'
    assert got == d


def test_export_posteriors_output_cannot_be_the_chimera_error_mask(tmp_path, monkeypatch,
                                                                  capsys):
    from test_export_posteriors_robustness import _bam, _run_cli
    bam = _bam(tmp_path / "in.bam", [(0, 1_000)])
    mask = tmp_path / "calls.daf_snps.chimera_error_mask.bed"
    mask.write_text("chr1\t10\t11\tG>A\n")
    code = _run_cli(monkeypatch, "-i", bam, "--enzyme", "hia5", "--seq", "pacbio",
                    "--chimera-detector", "strict",
                    "--chimera-error-mask", mask, "-o", mask, "--format", "tsv")
    assert code == 2 and "same file as --chimera-error-mask" in capsys.readouterr().err
    assert mask.read_text() == "chr1\t10\t11\tG>A\n"
    code = _run_cli(monkeypatch, "-i", bam, "--enzyme", "hia5", "--seq", "pacbio",
                    "--chimera-error-mask", mask, "-o", tmp_path / "x.tsv",
                    "--format", "tsv")
    assert code == 2 and "need --chimera-detector strict" in capsys.readouterr().err


@pytest.mark.parametrize('bed_text', [None, _OTHER_MASK])
def test_qc_without_the_calls_mask_grades_the_calls_marks(tmp_path, bed_text):
    """No sidecar and the call's error mask missing (or replaced by another
    run's BED): extraction re-detects mask-rescued reads, but they are
    reported apart and the grade stays on the call's flagged fraction (0)."""
    from fiberhmm.qc.core import run_qc
    make = ct_with_ga_tail(TAIL)
    reads = [make(f'rescued{i}', seed=100 + i) for i in range(30)]
    reads += [build(f'clean{i}', lambda r: 'CT', seed=200 + i) for i in range(90)]
    bam = _write_with_header(tmp_path / 'calls.bam', reads, _call_header(_masked_ds()))
    if bed_text is not None:
        bed = tmp_path / 'qc' / 'calls.daf_snps.chimera_error_mask.bed'
        bed.parent.mkdir()
        bed.write_text(bed_text)
    result = run_qc(bam, output_prefix=str(tmp_path / 'out' / 'calls'), mode='daf',
                    enzyme='dddb', reference_profile='none', sample_reads=200,
                    state_source='light-call', light_call_reads=10, stream=None)
    chimeras = result['chimeras']
    assert chimeras['error_mask_restored'] == ('unavailable' if bed_text is None
                                               else 'mismatch')
    assert result['state_rates']['reads']['reads_chimera_detected_at_extraction'] > 0
    assert chimeras['sampled_extraction_detected'] > 0
    assert (chimeras['n_chimeric'], chimeras['sampled_chimeric']) == (0, 0)
    assert chimeras['n_reads_checked'] == 120 and chimeras['fraction'] == 0
    grade = chimeras['grade']
    assert grade['source'] == 'bounded sample' and grade['status'] == 'PASS'
    assert 'warned_by' not in result['overall']


@pytest.mark.parametrize('mask', ['unavailable', 'off'])
def test_qc_does_not_report_a_missing_mask_the_call_never_applied(tmp_path, mask):
    from fiberhmm.qc.core import format_terminal
    make = ct_with_ga_tail(TAIL)
    reads = [make('chim', seed=3), build('clean', lambda r: 'CT', seed=4)]
    bam = _write_with_header(tmp_path / 'calls.bam', reads,
                             _call_header(_DS_DEFAULT.format(mask=mask)))
    result = _qc(bam, tmp_path, 'calls')
    assert result['chimeras']['error_mask_restored'] == mask
    assert 'error_mask_not_restored' not in result['chimeras']
    assert 'error mask not restored' not in format_terminal(result)
