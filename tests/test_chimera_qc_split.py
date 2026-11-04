"""DAF chimera QC (one strict detector, grade, regions, warnings) and split mode (3.0.4).

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
from fiberhmm.inference.chimera_split import cut_modifications, make_piece
from fiberhmm.inference.chimeras import (
    CHIMERA_NOTE, DEFAULT_MIN_SEG, DEFAULT_MIN_SPAN, DEFAULT_PURITY, HIGH_FRACTION, LEGACY_MIN_SEG,
    LEGACY_PURITY, WARN_FRACTION, ChimeraDecisions, chimera_warning, grade_chimeras,
    is_marked_chimera, prepare_chimeras,
)

# The full 3.0.3 detector: no base-quality or span limit, one switch, no mask.
LEGACY = dict(min_seg=LEGACY_MIN_SEG, purity=LEGACY_PURITY, min_baseq=0,
              min_span=0, multi_switch=False)
LEGACY_ARGS = ('--chimera-min-seg', str(LEGACY_MIN_SEG),
               '--chimera-purity', str(LEGACY_PURITY), '--chimera-min-baseq', '0',
               '--chimera-min-span', '0', '--no-chimera-multi-switch',
               '--no-chimera-error-mask')

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
    split = daf_chimera_split(ct, ga, 5, 0.8)
    assert split == {'left_strand': 'CT', 'right_strand': 'GA',
                     'left_last': 90, 'right_first': 300}
    split = daf_chimera_split([300, 310, 320, 330, 340], [10, 20, 30, 40, 50], 5, 0.8)
    assert split['left_strand'] == 'GA' and split['left_last'] == 50
    assert split['right_first'] == 300


def test_grade_thresholds():
    warn, high = round(1000 * WARN_FRACTION), round(1000 * HIGH_FRACTION)
    assert grade_chimeras(99, 50)['status'] == 'INSUFFICIENT'
    assert grade_chimeras(1000, None)['status'] == 'INSUFFICIENT'
    assert grade_chimeras(1000, warn - 1)['status'] == 'PASS'
    moderate = grade_chimeras(1000, warn)
    assert (moderate['status'], moderate['severity']) == ('WARN', 'moderate')
    assert grade_chimeras(1000, high - 1)['severity'] == 'moderate'
    very = grade_chimeras(1000, high)
    assert (very['status'], very['severity']) == ('WARN', 'high')
    assert 'FAIL' not in {grade_chimeras(100, n)['status'] for n in range(101)}
    assert moderate['basis'] == 'default' and 'basis_note' not in moderate
    assert (moderate['min_seg'], moderate['purity']) == (DEFAULT_MIN_SEG, DEFAULT_PURITY)
    assert chimera_warning({'grade': grade_chimeras(1000, warn - 1)}) is None
    assert chimera_warning({'grade': moderate}).startswith('WARNING: high')
    assert 'very high' in chimera_warning({'grade': very})


def test_grade_basis_of_older_counts():
    """3.0.3 counts (lenient detector) are graded as recorded and labelled."""
    lenient = grade_chimeras(1000, 150, LEGACY_MIN_SEG, LEGACY_PURITY)
    assert (lenient['status'], lenient['severity']) == ('WARN', 'moderate')
    assert lenient['basis'] == 'lenient' and '3.0.3' in lenient['basis_note']
    assert lenient['basis_note'] in chimera_warning({'grade': lenient})
    assert grade_chimeras(1000, 10, 7, 0.85)['basis'] == 'lenient'
    assert grade_chimeras(1000, 10, 20, 0.95)['basis'] == 'custom'
    assert grade_chimeras(1000, 10, None, None)['basis'] == 'unknown'


# --- count, SNP mask, regions ----------------------------------------------

def library(tmp_path, ref, n_clean=150, n_strong=20, n_weak=10, far=False):
    """Clean reads, strong swaps (called) and weak swaps (called only at 5 / 0.8)."""
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
        # Seven GA events on the right: called only by the 3.0.3 setting.
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


def test_one_detector_count_and_regions(tmp_path):
    ref = reference()
    bam = library(tmp_path, ref, far=True)
    stats = prepare_chimeras(bam, mode='flag')
    assert stats['n_reads_checked'] == 180
    assert stats['n_chimeric'] == 20               # the weak swaps are not called
    assert stats['fraction'] == pytest.approx(20 / 180)
    assert (stats['min_seg'], stats['purity']) == (DEFAULT_MIN_SEG, DEFAULT_PURITY)
    assert not [key for key in stats if 'strict' in key]
    assert stats['grade']['status'] == 'WARN' and stats['grade']['severity'] == 'moderate'
    assert stats['grade']['basis'] == 'default'
    assert stats['note'] == CHIMERA_NOTE
    # 3.0.3 keys keep their order; new keys follow.
    assert list(stats) == ['detected', 'mode', 'n_reads_checked', 'n_chimeric',
                           'fraction', 'min_seg', 'purity', 'n_records_affected',
                           'n_stale_marks_cleared', 'full_run', 'count_unit',
                           'grade', 'regions', 'split', 'note', 'min_baseq',
                           'min_span', 'span_unit', 'multi_switch', 'error_mask',
                           'n_duplicates_excluded', 'duplicates']
    assert (stats['min_baseq'], stats['min_span'], stats['multi_switch']) == (15, 200, True)
    assert stats['error_mask']['applied'] is False and stats['n_duplicates_excluded'] == 0
    rows = stats['regions']['rows']
    assert len(rows) == 2
    near = next(r for r in rows if r['start'] < 10_000)
    distant = next(r for r in rows if r['start'] >= 20_000)
    assert distant['n_reads_checked'] == 20 and distant['fraction'] == 1.0
    assert near['n_chimeric'] == 0
    assert set(near) == {'chrom', 'start', 'end', 'region', 'n_reads_checked',
                         'n_chimeric', 'fraction'}
    # The 3.0.3 setting is still available and also calls the weak swaps.
    legacy = prepare_chimeras(bam, mode='flag', **LEGACY)
    assert legacy['n_chimeric'] == 30 and legacy['grade']['basis'] == 'lenient'


def test_detection_follows_the_snp_mask(tmp_path):
    ref = reference()
    hdr = header()
    # Clean CT reads plus one site-specific G->A "variant" run on the right:
    # the GA events are only at SNP positions.
    start = 5000
    # Spread over > DEFAULT_MIN_SPAN bp, so only the mask stops the call.
    snp_sites = [start + q for q in range(2500, 2990) if ref[start + q] == 'G'][::8][:12]
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
    assert len(snp_sites) >= DEFAULT_MIN_SEG
    assert snp_sites[-1] - snp_sites[0] >= DEFAULT_MIN_SPAN
    assert prepare_chimeras(bam, mode='flag')['n_chimeric'] == 3
    masked = prepare_chimeras(bam, mode='flag', snp_mask=str(mask))
    assert masked['n_chimeric'] == 0


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
        for warn in (grade_chimeras(1000, round(1000 * WARN_FRACTION)),
                     grade_chimeras(1000, 600)):
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


def test_call_reports_one_count_warns_and_grades(tmp_path):
    ref = reference()
    bam = library(tmp_path, ref, far=True)
    run = _call(bam, tmp_path / 'calls.bam')
    assert ('20 / 180 checked reads' in run.stderr
            and f'min_seg={DEFAULT_MIN_SEG}, purity={DEFAULT_PURITY}' in run.stderr)
    assert 'strict detector' not in run.stderr
    assert 'WARNING: high DAF strand-swap chimera rate: 11.1% of reads;' in run.stderr
    assert CHIMERA_NOTE in run.stderr
    sidecar = json.loads((tmp_path / 'qc/calls.chimeras.json').read_text())['statistics']
    assert sidecar['n_chimeric'] == 20 and not [k for k in sidecar if 'strict' in k]
    assert sidecar['note'] == CHIMERA_NOTE and sidecar['split'] is None
    qc = json.loads((tmp_path / 'qc/calls.qc.json').read_text())
    chim = qc['chimeras']
    assert chim['grade']['status'] == 'WARN' and chim['grade']['source'] == 'full run'
    assert chim['grade']['basis'] == 'default'
    assert chim['fraction'] == pytest.approx(20 / 180)
    assert not [k for k in chim if 'strict' in k]
    assert len(chim['regions']['rows']) == 2
    if qc['overall'].get('warned_by'):
        assert qc['overall']['status'] == 'WARN'
    tsv = (tmp_path / 'qc/calls.qc.tsv').read_text().splitlines()
    row = dict(zip(tsv[0].split('\t'), tsv[1].split('\t')))
    assert row['chimeras_status'] == 'WARN' and row['chimeras_severity'] == 'moderate'
    assert row['chimeras_n_chimeric'] == '20' and row['chimeras_basis'] == 'default'
    assert not [k for k in row if 'strict' in k]
    regions = (tmp_path / 'qc/calls.qc.chimera_regions.tsv').read_text().splitlines()
    assert regions[0] == 'region\tchrom\tstart\tend\tn_reads_checked\tn_chimeric\tfraction'
    assert len(regions) == 3
    from fiberhmm.qc.core import format_terminal
    text = format_terminal(qc)
    assert 'DAF strand-swap chimeras  [WARN (moderate)]' in text
    assert 'chimeric: 20 of 180 reads' in text and CHIMERA_NOTE in text


def test_standalone_qc_grades_flag_mode_from_the_sample(tmp_path):
    ref = reference()
    bam = library(tmp_path, ref, n_clean=100, n_strong=40, n_weak=0)
    from fiberhmm.qc.core import run_qc
    for name, extra, basis in (('calls', (), 'default'), ('legacy', LEGACY_ARGS, 'lenient')):
        _call(bam, tmp_path / f'{name}.bam', '--no-qc', *extra)
        elsewhere = tmp_path / f'copy_{name}'
        elsewhere.mkdir()
        for suffix in ('', '.bai'):
            (elsewhere / f'{name}.bam{suffix}').write_bytes(
                (tmp_path / f'{name}.bam{suffix}').read_bytes())
        result = run_qc(str(elsewhere / f'{name}.bam'), output_prefix=str(tmp_path / name),
                        mode='daf', enzyme='dddb', sample_reads=500,
                        state_source='none', stream=None)
        chim = result['chimeras']
        assert chim['full_run'] is False and chim['n_chimeric'] == 40
        grade = chim['grade']
        assert grade['source'] == 'bounded sample' and grade['basis'] == basis
        assert (grade['status'], grade['severity']) == ('WARN', 'high')
    assert (grade['min_seg'], grade['purity']) == (LEGACY_MIN_SEG, LEGACY_PURITY)
    assert '3.0.3' in grade['basis_note']


def test_qc_with_a_303_sidecar_grades_its_lenient_count(tmp_path):
    """A 3.0.3 sidecar (5 / 0.8, no grade) is graded on its count, labelled lenient."""
    ref = reference()
    bam = library(tmp_path, ref, n_clean=100, n_strong=40, n_weak=0)
    _call(bam, tmp_path / 'calls.bam', '--no-qc')
    sidecar = tmp_path / 'qc/calls.chimeras.json'
    payload = json.loads(sidecar.read_text())
    keys_303 = ('detected', 'mode', 'n_reads_checked', 'n_chimeric', 'fraction', 'min_seg',
                'purity', 'n_records_affected', 'n_stale_marks_cleared', 'full_run',
                'count_unit')
    payload['statistics'] = {k: payload['statistics'][k] for k in keys_303}
    payload['statistics'].update(n_reads_checked=10_000, n_chimeric=1_500, fraction=0.15,
                                 min_seg=LEGACY_MIN_SEG, purity=LEGACY_PURITY)
    sidecar.write_text(json.dumps(payload))
    from fiberhmm.qc.core import format_terminal, run_qc
    result = run_qc(str(tmp_path / 'calls.bam'), output_prefix=str(tmp_path / 'sa'),
                    mode='daf', enzyme='dddb', sample_reads=500,
                    state_source='none', stream=None)
    chim = result['chimeras']
    assert chim['full_run'] is True and chim['n_reads_checked'] == 10_000
    grade = chim['grade']
    assert grade['source'] == 'full run' and grade['fraction'] == pytest.approx(0.15)
    assert (grade['status'], grade['severity']) == ('WARN', 'moderate')
    assert grade['basis'] == 'lenient' and '3.0.3' in grade['basis_note']
    assert f"basis: {grade['basis_note']}" in format_terminal(result)


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


# --- split pieces keep MM/ML ------------------------------------------------

_MOD_ENTRIES = (('C+mh?', 'C', 2), ('G-m.', 'G', 1), ('A+a', 'A', 1),
                ('N+n?', 'N', 1), ('C+76792.', 'C', 1))


def _with_modifications(read, seed=5, names=('MM', 'ML'), mn=True):
    """Random MM/ML over the read's original orientation (see the SAM spec)."""
    rng = random.Random(seed)
    seq = read.query_sequence
    original = seq if not read.is_reverse else seq.translate(
        str.maketrans('ACGT', 'TGCA'))[::-1]
    mm, ml = [], []
    for head, base, n_codes in _MOD_ENTRIES:
        occurrences = [i for i, b in enumerate(original) if base in ('N', b)]
        chosen = sorted(rng.sample(range(len(occurrences)), len(occurrences) // 4))
        skips, previous = [], -1
        for index in chosen:
            skips.append(index - previous - 1)
            previous = index
        mm.append(head + ''.join(f',{k}' for k in skips) + ';')
        ml += [rng.randrange(256) for _ in range(len(chosen) * n_codes)]
    from array import array
    read.set_tag(names[0], ''.join(mm))
    read.set_tag(names[1], array('B', ml))
    if mn:
        read.set_tag('MN', len(seq))
    return read


def _expected_calls(read, qs, qe):
    return {key: sorted((pos - qs, qual) for pos, qual in calls if qs <= pos < qe)
            for key, calls in read.modified_bases.items()}


def _calls(read):
    return {key: sorted(calls) for key, calls in (read.modified_bases or {}).items()}


@pytest.mark.parametrize('reverse', [False, True])
def test_pieces_keep_modifications_reindexed(reverse):
    ref = reference()
    hdr = header()
    read = make_read(hdr, ref, 'm', 1000, [(4, 25), (0, 800), (1, 3), (0, 900), (4, 15)],
                     clean('CT'), seed=3, flag=16 if reverse else 0)
    read = _with_modifications(read)
    length = read.query_length
    original = _calls(read)
    assert {key[2] for key in original} >= {'m', 'h', 'a', 'n', 76792}
    assert {key[1] for key in original} == {0, 1}           # '+' and '-' entries
    for qs, qe in ((0, 400), (37, 1211), (900, length), (0, length)):
        piece = make_piece(read, hdr, qs, qe, 0, None)
        assert piece.get_tag('MN') == qe - qs
        got = _calls(piece)
        expected = _expected_calls(read, qs, qe)
        assert {k: v for k, v in got.items() if v} == {k: v for k, v in expected.items() if v}
        # Each entry, its codes and its '.'/'?' flag survive, also without calls.
        heads = [entry.split(',')[0] for entry in piece.get_tag('MM').split(';') if entry]
        assert heads == [head for head, _, _ in _MOD_ENTRIES]
        assert len(piece.get_tag('ML')) == sum(len(v) for v in got.values())
    whole = make_piece(read, hdr, 0, length, 0, None)
    assert whole.get_tag('MM') == read.get_tag('MM')
    assert list(whole.get_tag('ML')) == list(read.get_tag('ML'))


def test_piece_modifications_legacy_names_and_drops():
    ref = reference()
    hdr = header()
    read = make_read(hdr, ref, 'm', 1000, [(0, 1500)], clean('GA'), seed=4)
    legacy = _with_modifications(read, names=('Mm', 'Ml'), mn=False)
    piece = make_piece(legacy, hdr, 200, 900, 1, None)
    assert piece.has_tag('Mm') and piece.has_tag('Ml') and not piece.has_tag('MM')
    assert not piece.has_tag('MN')
    assert cut_modifications(legacy, 200, 900, 'Mm', 'Ml')[0] == piece.get_tag('Mm')

    def dropped(record):
        return not make_piece(record, hdr, 200, 900, 1, None).has_tag('MM')

    good = _with_modifications(make_read(hdr, ref, 'g', 1000, [(0, 1500)], clean('GA'), seed=4))
    assert not dropped(good)
    clipped = make_read(hdr, ref, 'h', 1000, [(5, 40), (0, 1500)], clean('GA'), seed=4)
    assert dropped(_with_modifications(clipped))             # hard clip
    from array import array
    short_ml = _with_modifications(make_read(hdr, ref, 's', 1000, [(0, 1500)], clean('GA')))
    short_ml.set_tag('ML', array('B', list(short_ml.get_tag('ML'))[:-1]))
    assert dropped(short_ml)                                 # ML shorter than MM
    past_end = make_read(hdr, ref, 'p', 1000, [(0, 1500)], clean('GA'))
    past_end.set_tag('MM', 'C+m,100000;')
    past_end.set_tag('ML', array('B', [200]))
    assert dropped(past_end)                                 # skip runs past SEQ
    garbled = make_read(hdr, ref, 'q', 1000, [(0, 1500)], clean('GA'))
    garbled.set_tag('MM', 'C+m,x;')
    garbled.set_tag('ML', array('B', [200]))
    assert dropped(garbled)
    wrong_mn = _with_modifications(make_read(hdr, ref, 'n', 1000, [(0, 1500)], clean('GA')))
    wrong_mn.set_tag('MN', 99)
    assert dropped(wrong_mn)                                 # MN disagrees with SEQ


def test_split_mode_counts_kept_and_dropped_modifications(tmp_path):
    ref = reference()
    hdr = header()
    reads = [make_read(hdr, ref, f'clean{i}', 3000, [(0, 3000)], clean('CT'), seed=i)
             for i in range(4)]
    kept = _with_modifications(make_read(hdr, ref, 'kept', 3000, [(0, 3000)],
                                         swap_at(4500), seed=50, flag=16))
    clipped = _with_modifications(make_read(hdr, ref, 'clip', 3200, [(5, 30), (0, 3000)],
                                            swap_at(4700), seed=51))
    bam = write_bam(tmp_path / 'mods.bam', reads + [kept, clipped])
    stats = prepare_chimeras(bam, mode='split', spool_bam=str(tmp_path / 'spool.bam'),
                             min_read_length=100)
    split = stats['split']
    assert split['n_pieces_written'] == 4
    assert split['n_pieces_modifications_kept'] == 2
    assert split['n_pieces_modifications_dropped'] == 2
    out = {r.query_name: r for r in records(tmp_path / 'spool.bam')}
    original = next(r for r in records(bam) if r.query_name == 'kept')
    for name in ('kept/c0', 'kept/c1'):
        piece = out[name]
        offset = original.query_sequence.find(piece.query_sequence)
        assert offset >= 0 and piece.is_reverse
        expected = _expected_calls(original, offset, offset + piece.query_length)
        assert ({k: v for k, v in _calls(piece).items() if v}
                == {k: v for k, v in expected.items() if v})
    assert not out['clip/c0'].has_tag('MM') and not out['clip/c1'].has_tag('ML')


def _unmapped_ry_chimera(hdr, ref, name='u_chim', seed=3):
    """An unmapped R/Y read: Y (C->T) in its first half, R (G->A) after."""
    rng = random.Random(seed)
    bases = ref[5000:9000]
    seq = []
    for i, base in enumerate(bases):
        if i < 2000 and base == 'C' and rng.random() < 0.35:
            seq.append('Y')
        elif i >= 2000 and base == 'G' and rng.random() < 0.35:
            seq.append('R')
        else:
            seq.append(base)
    read = pysam.AlignedSegment(hdr)
    read.query_name = name
    read.flag = 4
    read.reference_id = -1
    read.reference_start = -1
    read.query_sequence = ''.join(seq)
    read.query_qualities = pysam.qualitystring_to_array('I' * len(seq))
    return read


def test_split_keeps_unmapped_chimeras_whole_and_flagged(tmp_path):
    from fiberhmm.inference.chimera_split import split_record
    from fiberhmm.inference.chimeras import ChimeraDecisions
    ref = reference()
    hdr = header()
    unmapped = _unmapped_ry_chimera(hdr, ref)
    reads = [make_read(hdr, ref, 'mapped_chim', 2000, [(0, 4000)], swap_at(4000), seed=1),
             make_read(hdr, ref, 'clean', 2000, [(0, 3000)], clean('CT'), seed=2)]
    path = tmp_path / 'in.bam'
    with pysam.AlignmentFile(str(path), 'wb', header=hdr) as out:
        for read in reads + [unmapped]:
            out.write(read)
    # The public cutter refuses an unmapped record instead of crashing.
    pieces, gaps = split_record(unmapped, hdr, {'left_last': 100, 'right_first': 3000})
    assert pieces == [None, None] and gaps == [None]

    spool = tmp_path / 'spool.bam'
    names = ChimeraDecisions()
    stats = prepare_chimeras(str(path), mode='split', spool_bam=str(spool),
                             process_unmapped=True, chimeric_names=names,
                             min_baseq=0)
    assert stats['n_chimeric'] == 2                    # mapped + unmapped
    split = stats['split']
    assert split['n_records_unmapped_kept_whole'] == 1
    assert split['n_records_split'] == 1
    assert 'u_chim' in names and 'mapped_chim' in names
    spooled = records(spool)
    whole = [r for r in spooled if r.query_name == 'u_chim']
    assert len(whole) == 1 and whole[0].is_unmapped
    assert whole[0].query_sequence == unmapped.query_sequence
    assert {r.query_name for r in spooled if r.query_name.startswith('mapped_chim/')} \
        == {'mapped_chim/c0', 'mapped_chim/c1'}
    # The flag-mode writer marks the whole unmapped chimera.
    out = tmp_path / 'out.bam'
    prepare_chimeras(str(path), str(out), mode='flag', process_unmapped=True, min_baseq=0)
    marked = {r.query_name for r in records(out) if r.has_tag('xc')}
    assert 'u_chim' in marked
