"""DAF chimera QC (strict count, grade, regions, warnings) and split mode (3.0.4).

All data are synthetic: a random reference and reads whose C->T / G->A
conversions are placed by the test, with MD tags computed from the CIGAR.
"""
import json
import random
import subprocess
import sys

import pysam
import pytest

from fiberhmm.daf.encoder import (
    daf_chimera_split, get_daf_positions, is_daf_chimera, md_matches_cigar,
)
from fiberhmm.inference.chimera_split import make_piece
from fiberhmm.inference.chimeras import (
    CHIMERA_NOTE, ChimeraDecisions, chimera_warning, grade_chimeras,
    is_marked_chimera, prepare_chimeras,
)

CONTIG = 'chrS'
REF_LEN = 40_000


@pytest.fixture(autouse=True)
def extraction_policy():
    from fiberhmm.inference import engine
    engine.configure_daf_chimera_filter()
    engine.configure_daf_snp_mask(None)
    engine.configure_daf_insert_evidence(None)
    yield
    engine.configure_daf_chimera_filter()
    engine.configure_daf_snp_mask(None)
    engine.configure_daf_insert_evidence(None)


def reference(seed=11):
    rng = random.Random(seed)
    return ''.join(rng.choice('ACGT') for _ in range(REF_LEN))


def write_fasta(path, ref):
    path.write_text(f'>{CONTIG}\n' + '\n'.join(ref[i:i + 80] for i in range(0, len(ref), 80)) + '\n')
    pysam.faidx(str(path))
    return str(path)


def header():
    return pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate'},
        'SQ': [{'SN': CONTIG, 'LN': REF_LEN}]})


def make_read(hdr, ref, name, start, cigar, strand_at, rate=0.35, seed=0,
              flag=0, no_events=()):
    """A read over ``ref`` from ``start`` with ``cigar`` ([(op, len)]).

    ``strand_at(ref_pos)`` gives 'CT', 'GA' or None for each aligned base;
    target bases convert with probability ``rate``. Reference positions in
    ``no_events`` never convert. MD/NM are computed from the alignment.
    """
    rng = random.Random(seed)
    seq, md, run, nm = [], [], 0, 0
    r = start
    for op, length in cigar:
        if op == 4 or op == 1:          # S / I: random bases
            seq.extend(rng.choice('AT') for _ in range(length))
            nm += length if op == 1 else 0
        elif op == 0:
            for _ in range(length):
                base = ref[r]
                strand = strand_at(r)
                out = base
                if r not in no_events and rng.random() < rate:
                    if strand == 'CT' and base == 'C':
                        out = 'T'
                    elif strand == 'GA' and base == 'G':
                        out = 'A'
                if out != base:
                    md.append(str(run))
                    md.append(base)
                    run = 0
                    nm += 1
                else:
                    run += 1
                seq.append(out)
                r += 1
        elif op == 2:
            md.append(str(run))
            md.append('^' + ref[r:r + length])
            run = 0
            nm += length
            r += length
    md.append(str(run))
    read = pysam.AlignedSegment(hdr)
    read.query_name = name
    read.flag = flag
    read.reference_id = 0
    read.reference_start = start
    read.mapping_quality = 60
    read.cigartuples = cigar
    read.query_sequence = ''.join(seq)
    read.query_qualities = pysam.qualitystring_to_array(
        ''.join(chr(33 + 10 + (i % 30)) for i in range(len(seq))))
    read.set_tag('NM', nm)
    read.set_tag('MD', ''.join(md))
    return read


def swap_at(breakpoint, left='CT', right='GA'):
    return lambda r: left if r < breakpoint else right


def clean(strand):
    return lambda r: strand


def write_bam(path, reads):
    hdr = reads[0].header
    reads = sorted(reads, key=lambda x: x.reference_start)
    with pysam.AlignmentFile(str(path), 'wb', header=hdr) as out:
        for read in reads:
            out.write(read)
    pysam.index(str(path))
    return str(path)


def records(path):
    with pysam.AlignmentFile(str(path), 'rb', check_sq=False) as bam:
        return list(bam.fetch(until_eof=True))


# --- detector -------------------------------------------------------------

def _is_daf_chimera_303(ct_positions, ga_positions, min_seg_events=5, purity=0.8):
    """3.0.3 is_daf_chimera, verbatim: the decision must not change."""
    nct, nga = len(ct_positions), len(ga_positions)
    if min(nct, nga) < min_seg_events:
        return False
    events = sorted([(int(p), 1) for p in ct_positions] + [(int(p), -1) for p in ga_positions])
    n = len(events)
    if n < 2 * min_seg_events:
        return False
    ct_left = 0
    for k in range(1, n):
        if events[k - 1][1] == 1:
            ct_left += 1
        left_n, right_n = k, n - k
        if left_n < min_seg_events or right_n < min_seg_events:
            continue
        ga_left = left_n - ct_left
        ct_right, ga_right = nct - ct_left, nga - ga_left
        if (ct_left >= min_seg_events and ga_right >= min_seg_events
                and ct_left >= purity * left_n and ga_right >= purity * right_n):
            return True
        if (ga_left >= min_seg_events and ct_right >= min_seg_events
                and ga_left >= purity * left_n and ct_right >= purity * right_n):
            return True
    return False


def test_decision_unchanged_and_split_agrees():
    rng = random.Random(3)
    for _ in range(3000):
        positions = rng.sample(range(2000), rng.randrange(0, 60))
        cut = rng.randrange(0, 2000)
        ct, ga = [], []
        for p in positions:
            noisy = rng.random() < rng.choice([0.0, 0.05, 0.3])
            (ct if (p < cut) != noisy else ga).append(p)
        for min_seg, purity in [(5, 0.8), (10, 0.9), (3, 0.6)]:
            old = _is_daf_chimera_303(ct, ga, min_seg, purity)
            assert is_daf_chimera(ct, ga, min_seg, purity) == old
            assert (daf_chimera_split(ct, ga, min_seg, purity) is not None) == old


def test_split_picks_best_breakpoint_and_gap_edges():
    # A stray GA inside the CT segment must not move the split there.
    ct = [10, 20, 30, 40, 50, 60, 70, 90]
    ga = [55, 300, 310, 320, 330, 340]
    split = daf_chimera_split(ct, ga)
    assert split == {'left_strand': 'CT', 'right_strand': 'GA',
                     'left_last': 90, 'right_first': 300}
    split = daf_chimera_split([300, 310, 320, 330, 340], [10, 20, 30, 40, 50])
    assert split['left_strand'] == 'GA' and split['left_last'] == 50
    assert split['right_first'] == 300


def test_grade_thresholds():
    assert grade_chimeras(99, 50)['status'] == 'INSUFFICIENT'
    assert grade_chimeras(1000, None)['status'] == 'INSUFFICIENT'
    assert grade_chimeras(1000, 49)['status'] == 'PASS'
    moderate = grade_chimeras(1000, 50)
    assert (moderate['status'], moderate['severity']) == ('WARN', 'moderate')
    high = grade_chimeras(1000, 150)
    assert (high['status'], high['severity']) == ('WARN', 'high')
    assert 'FAIL' not in {grade_chimeras(100, n)['status'] for n in range(101)}
    assert chimera_warning({'grade': grade_chimeras(1000, 49)}) is None
    assert chimera_warning({'grade': moderate, 'fraction': 0.1}).startswith('WARNING:')
    assert 'very high' in chimera_warning({'grade': high, 'fraction': 0.2})


# --- strict count, SNP mask, regions ---------------------------------------

def library(tmp_path, ref, n_clean=150, n_strong=20, n_weak=10, far=False):
    """Clean reads, strong swaps (strict + default) and weak swaps (default only)."""
    hdr = header()
    reads = []
    for i in range(n_clean):
        start = 2000 + (i % 7) * 30
        reads.append(make_read(hdr, ref, f'clean{i}', start, [(0, 3000)],
                               clean('CT' if i % 2 else 'GA'), seed=i))
    for i in range(n_strong):
        start = (20_000 if far else 2000) + (i % 5) * 20
        reads.append(make_read(hdr, ref, f'strong{i}', start, [(0, 3000)],
                               swap_at(start + 1500), seed=1000 + i))
    for i in range(n_weak):
        # Seven GA events on the right: default (min_seg 5) only.
        start = 2000 + i * 10
        bp = start + 2900
        def strand(r, bp=bp):
            return 'CT' if r < bp else 'GA'
        read = make_read(hdr, ref, f'weak{i}', start, [(0, 3000)], strand,
                         rate=0.0, seed=2000 + i)
        seq = list(read.query_sequence)
        md_ref = list(ref[start:start + 3000])
        cs = [q for q in range(0, 2800) if md_ref[q] == 'C'][::40][:20]
        gs = [q for q in range(2905, 3000) if md_ref[q] == 'G'][:7]
        for q in cs:
            seq[q] = 'T'
        for q in gs:
            seq[q] = 'A'
        reads.append(_with_seq(hdr, read, ref, ''.join(seq)))
    return write_bam(tmp_path / 'lib.bam', reads)


def _with_seq(hdr, read, ref, seq):
    """Replace an all-M read's sequence and recompute MD/NM."""
    out = pysam.AlignedSegment(hdr)
    out.query_name = read.query_name
    out.flag = read.flag
    out.reference_id = 0
    out.reference_start = read.reference_start
    out.mapping_quality = 60
    out.cigartuples = read.cigartuples
    out.query_sequence = seq
    md, run, nm = [], 0, 0
    for q, base in enumerate(seq):
        r = ref[read.reference_start + q]
        if r != base:
            md += [str(run), r]
            run = 0
            nm += 1
        else:
            run += 1
    md.append(str(run))
    out.set_tag('NM', nm)
    out.set_tag('MD', ''.join(md))
    return out


def test_strict_and_default_counts_in_one_pass(tmp_path):
    ref = reference()
    bam = library(tmp_path, ref, far=True)
    stats = prepare_chimeras(bam, mode='flag')
    assert stats['n_reads_checked'] == 180
    assert stats['n_chimeric'] == 30
    assert stats['n_chimeric_strict'] == 20
    assert stats['fraction_strict'] == pytest.approx(20 / 180)
    assert (stats['strict_min_seg'], stats['strict_purity']) == (10, 0.9)
    assert stats['grade']['status'] == 'WARN' and stats['grade']['severity'] == 'moderate'
    assert stats['note'] == CHIMERA_NOTE
    # 3.0.3 keys keep their order; new keys follow.
    assert list(stats)[:11] == ['detected', 'mode', 'n_reads_checked', 'n_chimeric',
                                'fraction', 'min_seg', 'purity', 'n_records_affected',
                                'n_stale_marks_cleared', 'full_run', 'count_unit']
    rows = stats['regions']['rows']
    assert len(rows) == 2
    near = next(r for r in rows if r['start'] < 10_000)
    distant = next(r for r in rows if r['start'] >= 20_000)
    assert distant['n_reads_checked'] == 20 and distant['fraction_strict'] == 1.0
    assert near['n_chimeric'] == 10 and near['n_chimeric_strict'] == 0


def test_detection_follows_the_snp_mask(tmp_path):
    ref = reference()
    hdr = header()
    # Clean CT reads plus one site-specific G->A "variant" run on the right:
    # the GA events are only at SNP positions.
    start = 5000
    snp_sites = [start + q for q in range(2500, 2990) if ref[start + q] == 'G'][:12]
    reads = []
    for i in range(3):
        read = make_read(hdr, ref, f'r{i}', start, [(0, 3000)],
                         lambda r: 'CT' if r < start + 2400 else None, seed=i)
        seq = list(read.query_sequence)
        for site in snp_sites:
            seq[site - start] = 'A'
        reads.append(_with_seq(hdr, read, ref, ''.join(seq)))
    bam = write_bam(tmp_path / 'snp.bam', reads)
    mask = tmp_path / 'mask.bed'
    mask.write_text(''.join(f'{CONTIG}\t{s}\t{s + 1}\n' for s in snp_sites))
    assert prepare_chimeras(bam, mode='flag')['n_chimeric'] == 3
    masked = prepare_chimeras(bam, mode='flag', snp_mask=str(mask))
    assert masked['n_chimeric'] == 0 and masked['n_chimeric_strict'] == 0
    # QC's strict re-test of sampled flagged reads uses QC's mask too.
    from fiberhmm.qc.core import _strict_chimeras
    pairs = [(r.query_name, r) for r in records(bam)]
    assert len(_strict_chimeras(pairs, None, None)) == 3
    from fiberhmm.daf.snps import load_snp_mask
    assert _strict_chimeras(pairs, None, load_snp_mask(str(mask))) == set()


# --- verdict --------------------------------------------------------------

def _analysis(signal='PASS', periodicity='PASS', states=False):
    graded = {'status': 'PASS', 'score': 90.0}
    absent = {'status': 'INSUFFICIENT', 'score': None}
    return {'signal': {'status': signal, 'score': 80.0},
            'periodicity': {'status': periodicity, 'score': 70.0 if periodicity != 'FAIL' else 10.0},
            'efficiency': graded if states else absent,
            'background': graded if states else absent}


@pytest.mark.parametrize('states', [False, True])
def test_chimera_grade_only_demotes_pass(states):
    from fiberhmm.qc.core import _overall_verdict
    for periodicity in ('PASS', 'WARN', 'FAIL'):
        analysis = _analysis(periodicity=periodicity, states=states)
        base = _overall_verdict(analysis)
        assert set(base) == {'score', 'status', 'components', 'verdict_basis'}
        for grade in (None, grade_chimeras(1000, 10), grade_chimeras(10, 5)):
            assert _overall_verdict(analysis, grade) == base
        for warn in (grade_chimeras(1000, 60), grade_chimeras(1000, 600)):
            verdict = _overall_verdict(analysis, warn)
            assert verdict['score'] == base['score']
            assert verdict['components'] == base['components']
            if base['status'] == 'PASS':
                assert verdict['status'] == 'WARN' and verdict['warned_by'] == ['chimeras']
            else:
                assert verdict == base


# --- call CLI and QC report -----------------------------------------------

def _call(source, target, *extra, reference_fasta=None):
    command = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', str(source),
               '-o', str(target), '--enzyme', 'dddb', '-c', '1', '--io-threads', '1',
               '--no-dedup', '--no-daf-call-snps', '--daf-insert-consensus', 'off',
               '--qc-sample-reads', '400', *extra]
    if reference_fasta:
        command += ['--reference', reference_fasta]
    run = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    return run


def test_call_reports_both_counts_warns_and_grades(tmp_path):
    ref = reference()
    bam = library(tmp_path, ref, far=True)
    run = _call(bam, tmp_path / 'calls.bam')
    assert 'strict detector (min_seg=10, purity=0.9): 20' in run.stderr
    assert 'WARNING: high DAF strand-swap chimera rate' in run.stderr
    assert CHIMERA_NOTE in run.stderr
    sidecar = json.loads((tmp_path / 'qc/calls.chimeras.json').read_text())['statistics']
    assert sidecar['n_chimeric'] == 30 and sidecar['n_chimeric_strict'] == 20
    assert sidecar['note'] == CHIMERA_NOTE and sidecar['split'] is None
    qc = json.loads((tmp_path / 'qc/calls.qc.json').read_text())
    chim = qc['chimeras']
    assert chim['grade']['status'] == 'WARN' and chim['grade']['source'] == 'full run'
    assert chim['fraction_strict'] == pytest.approx(20 / 180)
    assert len(chim['regions']['rows']) == 2
    if qc['overall'].get('warned_by'):
        assert qc['overall']['status'] == 'WARN'
    tsv = (tmp_path / 'qc/calls.qc.tsv').read_text().splitlines()
    row = dict(zip(tsv[0].split('\t'), tsv[1].split('\t')))
    assert row['chimeras_status'] == 'WARN' and row['chimeras_severity'] == 'moderate'
    assert row['chimeras_n_chimeric_strict'] == '20'
    regions = (tmp_path / 'qc/calls.qc.chimera_regions.tsv').read_text().splitlines()
    assert regions[0].startswith('region\tchrom\tstart\tend') and len(regions) == 3
    from fiberhmm.qc.core import format_terminal
    text = format_terminal(qc)
    assert 'DAF strand-swap chimeras  [WARN (moderate)]' in text
    assert 'strict detector:  20' in text and CHIMERA_NOTE in text


def test_standalone_qc_grades_flag_mode_from_the_sample(tmp_path):
    ref = reference()
    bam = library(tmp_path, ref, n_clean=100, n_strong=40, n_weak=0)
    _call(bam, tmp_path / 'calls.bam', '--no-qc')
    elsewhere = tmp_path / 'copy'
    elsewhere.mkdir()
    for suffix in ('', '.bai'):
        (elsewhere / f'calls.bam{suffix}').write_bytes(
            (tmp_path / f'calls.bam{suffix}').read_bytes())
    from fiberhmm.qc.core import run_qc
    result = run_qc(str(elsewhere / 'calls.bam'), output_prefix=str(tmp_path / 'sa'),
                    mode='daf', enzyme='dddb', sample_reads=500,
                    state_source='none', stream=None)
    chim = result['chimeras']
    assert chim['full_run'] is False
    assert chim['n_chimeric'] == 40 and chim['n_chimeric_strict'] == 40
    assert chim['grade']['source'] == 'bounded sample'
    assert (chim['grade']['status'], chim['grade']['severity']) == ('WARN', 'high')


def test_qc_with_a_303_sidecar_grades_from_the_sample(tmp_path):
    """A 3.0.3 sidecar has no strict count: grade on the sample, not a mix."""
    ref = reference()
    bam = library(tmp_path, ref, n_clean=100, n_strong=40, n_weak=0)
    _call(bam, tmp_path / 'calls.bam', '--no-qc')
    sidecar = tmp_path / 'qc/calls.chimeras.json'
    payload = json.loads(sidecar.read_text())
    keys_303 = ('detected', 'mode', 'n_reads_checked', 'n_chimeric', 'fraction', 'min_seg',
                'purity', 'n_records_affected', 'n_stale_marks_cleared', 'full_run',
                'count_unit')
    payload['statistics'] = {k: payload['statistics'][k] for k in keys_303}
    payload['statistics']['n_reads_checked'] = 10_000   # distinguishable from the sample
    sidecar.write_text(json.dumps(payload))
    from fiberhmm.qc.core import run_qc
    result = run_qc(str(tmp_path / 'calls.bam'), output_prefix=str(tmp_path / 'sa'),
                    mode='daf', enzyme='dddb', sample_reads=500,
                    state_source='none', stream=None)
    chim = result['chimeras']
    assert chim['full_run'] is True and chim['n_reads_checked'] == 10_000
    assert chim['grade']['source'] == 'bounded sample'
    assert chim['grade']['fraction'] == pytest.approx(40 / 140)
    assert chim['fraction_strict'] is None and chim['n_chimeric_strict'] is None


def test_unmapped_records_share_one_region(tmp_path):
    ref = reference()
    hdr = header()
    reads = [make_read(hdr, ref, f'm{i}', 2000, [(0, 3000)], clean('CT'), seed=i)
             for i in range(25)]
    for i in range(25):
        read = pysam.AlignedSegment(hdr)
        read.query_name = f'u{i}'
        read.flag = 4
        read.reference_id = -1
        read.reference_start = -1
        read.query_sequence = ref[100:2100]
        reads.append(read)
    path = tmp_path / 'unmapped.bam'
    with pysam.AlignmentFile(str(path), 'wb', header=hdr) as out:
        for read in reads:
            out.write(read)
    stats = prepare_chimeras(str(path), mode='flag', process_unmapped=True)
    regions = {row['region']: row['n_reads_checked'] for row in stats['regions']['rows']}
    assert regions['unmapped'] == 25 and sum(regions.values()) == 50


def test_non_daf_qc_has_no_chimera_grade(tmp_path):
    from fiberhmm.qc.core import _grade_chimera_block
    block = {'mode': 'n/a', 'n_reads_checked': 0}
    _grade_chimera_block(block, False)
    assert block['grade'] is None


# --- split mode -----------------------------------------------------------

def _split_fixture(tmp_path, ref, gap_without_events=0):
    hdr = header()
    reads = []
    for i in range(6):
        reads.append(make_read(hdr, ref, f'clean{i}', 3000 + 50 * i, [(0, 4000)],
                               clean('CT' if i % 2 else 'GA'), seed=i))
    # Chimera with soft clips, an insertion and deletions on both sides.
    cigar = [(4, 30), (0, 1200), (1, 4), (0, 900), (2, 3), (0, 1500), (2, 6),
             (0, 1100), (4, 20)]
    start = 3100
    bp = start + 2500
    quiet = set(range(bp - gap_without_events // 2, bp + gap_without_events // 2))
    reads.append(make_read(hdr, ref, 'swap', start, cigar, swap_at(bp), seed=99,
                           no_events=quiet))
    # A supplementary of the same read that is not itself chimeric.
    reads.append(make_read(hdr, ref, 'swap', 20_000, [(0, 1500)], clean('CT'),
                           seed=98, flag=2048))
    # A chimera whose right side is too short to keep.
    reads.append(make_read(hdr, ref, 'shortright', 3300, [(0, 3000)],
                           swap_at(3300 + 2700), seed=97, rate=0.5))
    return write_bam(tmp_path / 'split_in.bam', reads)


def test_split_pieces_are_valid_single_strand_alignments(tmp_path):
    ref = reference()
    bam = _split_fixture(tmp_path, ref)
    spool = tmp_path / 'spool.bam'
    names = ChimeraDecisions()
    stats = prepare_chimeras(bam, mode='split', min_read_length=500,
                             spool_bam=str(spool), chimeric_names=names)
    assert stats['n_chimeric'] == 2 and stats['mode'] == 'split'
    split = stats['split']
    assert split['n_records_split'] == 2 and split['n_reads_split'] == 2
    assert split['n_pieces_written'] == 3 and split['n_pieces_too_short'] == 1
    assert 0 < split['fraction_bases_recovered'] < 1
    assert split['gap_reference_bp']['n'] == 2
    out = records(spool)
    assert [r.reference_start for r in out] == sorted(r.reference_start for r in out)
    original = next(r for r in records(bam) if r.query_name == 'swap' and not r.is_supplementary)
    pieces = sorted((r for r in out if r.has_tag('xn') and r.get_tag('xn') == 'swap'),
                    key=lambda r: r.get_tag('xi'))
    assert [p.query_name for p in pieces] == ['swap/c0', 'swap/c1']
    o_ct, o_ga, _ = get_daf_positions(original, force_strand='CT')
    o_ref = original.get_reference_positions(full_length=True)
    best = daf_chimera_split(o_ct, o_ga)
    gap = (o_ref[best['left_last']] + 1, o_ref[best['right_first']])
    for piece in pieces:
        assert piece.get_tag('xc') == 'split' and not piece.is_qcfail
        assert tuple(piece.get_tag('xb')) == gap
        assert not piece.has_tag('MM') and not piece.has_tag('ML')
        assert md_matches_cigar(piece)
        assert len(piece.query_sequence) == len(piece.query_qualities)
        piece.get_aligned_pairs(with_seq=True)   # MD parses against CIGAR
        # Same events as the original at the same reference positions.
        p_ct, p_ga, _ = get_daf_positions(piece, force_strand='CT')
        p_ref = piece.get_reference_positions(full_length=True)
        lo, hi = piece.reference_start, piece.reference_end
        for p_events, o_events in ((p_ct, o_ct), (p_ga, o_ga)):
            assert {p_ref[q] for q in p_events} == {
                o_ref[q] for q in o_events if lo <= o_ref[q] < hi}
        # One strand only, and it ends/starts on its own last/first event.
        assert not is_daf_chimera(p_ct, p_ga)
    left, right = pieces
    l_ct, l_ga, _ = get_daf_positions(left, force_strand='CT')
    r_ct, r_ga, _ = get_daf_positions(right, force_strand='CT')
    assert max(l_ct) == len(left.query_sequence) - 1    # CT piece ends on an event
    assert min(r_ga) == 0                               # GA piece starts on one
    assert left.reference_end == gap[0] and right.reference_start == gap[1]
    assert left.cigartuples[0] == (4, 30) and right.cigartuples[-1] == (4, 20)
    nm = sum(1 for _, _, b in left.get_aligned_pairs(with_seq=True)
             if b is not None and b.islower())
    nm += sum(n for op, n in left.cigartuples if op in (1, 2))
    assert left.get_tag('NM') == nm
    # The non-chimeric supplementary stays and is flagged by name; the
    # pieces are not in the decision set.
    supplementary = [r for r in out if r.query_name == 'swap']
    assert len(supplementary) == 1 and supplementary[0].is_supplementary
    assert 'swap' in names and 'swap/c0' not in names


def _inner_edge_footprints(piece, within=25):
    """Nucleosome calls that reach within ``within`` bp of a piece's cut edge."""
    length = len(piece.query_sequence)
    starts = list(piece.get_tag('ns')) if piece.has_tag('ns') else []
    lengths = list(piece.get_tag('nl')) if piece.has_tag('nl') else []
    if piece.get_tag('xi') == 0:
        return [(s, n) for s, n in zip(starts, lengths) if s + n >= length - within]
    return [(s, n) for s, n in zip(starts, lengths) if s <= within]


def test_split_call_end_to_end_and_no_edge_footprint(tmp_path):
    ref = reference()
    # 400 bp without events around the switch: a naive cut there would leave
    # an undeaminated stretch at each piece's inner edge.
    bam = _split_fixture(tmp_path, ref, gap_without_events=400)
    target = tmp_path / 'calls.bam'
    run = _call(bam, target, '--chimeras', 'split', '--min-read-length', '500')
    assert 'split: 2 records cut into 3 pieces' in run.stderr
    out = records(target)
    pieces = [r for r in out if r.has_tag('xi')]
    assert len(pieces) == 3 and all(not r.is_qcfail for r in pieces)
    assert all(r.has_tag('MA') or r.has_tag('ns') for r in pieces)
    flagged = [r for r in out if is_marked_chimera(r)]
    assert [r.query_name for r in flagged] == ['swap'] and flagged[0].is_supplementary
    edge = {r.query_name: _inner_edge_footprints(r) for r in pieces
            if r.query_name.startswith('swap/')}
    assert edge == {'swap/c0': [], 'swap/c1': []}
    # Control: cutting inside the event-free gap instead leaves footprints
    # at the inner edges, which split mode must not create.
    reads = records(bam)
    naive = []
    for read in reads:
        if read.query_name == 'swap' and not read.is_supplementary:
            q = next(q for q, r in read.get_aligned_pairs(matches_only=True)
                     if r == 3100 + 2500)
            naive += [make_piece(read, read.header, 0, q, 0, None),
                      make_piece(read, read.header, q, read.query_length, 1, None)]
        elif read.query_name != 'swap':
            naive.append(read)
    naive_bam = write_bam(tmp_path / 'naive.bam', naive)
    _call(naive_bam, tmp_path / 'naive_calls.bam', '--chimeras', 'keep',
          '--min-read-length', '500', '--no-qc')
    naive_edges = [_inner_edge_footprints(r) for r in records(tmp_path / 'naive_calls.bam')
                   if r.query_name.startswith('swap/')]
    assert any(naive_edges)
    with pysam.AlignmentFile(str(target)) as handle:
        assert 'chimeras=split' in handle.header.to_dict()['PG'][-1]['DS']
    sidecar = json.loads((tmp_path / 'qc/calls.chimeras.json').read_text())['statistics']
    assert sidecar['split']['n_pieces_written'] == 3
    qc = json.loads((tmp_path / 'qc/calls.qc.json').read_text())
    assert qc['chimeras']['mode'] == 'split' and qc['chimeras']['split']
    region_parallel = tmp_path / 'rp.bam'
    _call(bam, region_parallel, '--chimeras', 'split', '--min-read-length', '500',
          '--region-parallel', '--no-qc')
    assert [r.to_string() for r in records(region_parallel)] == [r.to_string() for r in out]


def test_pipeline_accepts_split_and_repeats_the_warning(tmp_path):
    from fiberhmm.cli.pipeline import config_from_args, parse_args
    cfg = config_from_args(parse_args(['a.bam', '--reference', 'ref.fa', '--enzyme',
                                       'dddb', '-o', str(tmp_path), '--chimeras', 'split']))
    assert cfg.resolved_chimeras() == 'split'
    from fiberhmm.pipeline.progress import ProgressReporter
    from fiberhmm.pipeline.runner import Pipeline
    pipeline = Pipeline(cfg, ProgressReporter())
    pipeline.called_bam = str(tmp_path / 'called.bam')
    (tmp_path / 'qc').mkdir()
    (tmp_path / 'qc/called.chimeras.json').write_text(json.dumps(
        {'statistics': {'fraction': 0.3, 'grade': grade_chimeras(1000, 200)}}))
    logged = []
    pipeline.log = lambda message, level='info': logged.append((message, level))
    pipeline._chimera_warning()
    assert logged and logged[0][1] == 'warning' and 'WARNING: very high' in logged[0][0]
