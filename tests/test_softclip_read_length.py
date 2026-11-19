"""--min-read-length measures what the HMM models: SEQ length for Hia5, aligned length for masked DAF."""
from __future__ import annotations

import os

import numpy as np
import pysam
import pytest

from fiberhmm.inference.read_filters import (
    ReadFilterConfig,
    count_skip,
    modeled_read_length,
    new_skip_counts,
    printable_skip_reasons,
    streaming_skip_reason,
    too_short_hint,
)

CLIP, ALIGNED = 1800, 400            # 4,000 bp molecule, 400 bp aligned
LENGTH = 2 * CLIP + ALIGNED


def _revcomp(seq):
    return seq.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def _mm_entry(sequence, positions, base, strand):
    index, previous, skips = {}, -1, []
    count = 0
    for k, b in enumerate(sequence):
        if b == base:
            index[k] = count
            count += 1
    for pos in sorted(positions):
        skips.append(index[pos] - previous - 1)
        previous = index[pos]
    return f'{base}{strand}a.' + ''.join(f',{s}' for s in skips) + ';'


def _softclipped_bam(path, n_reads=6, seed=3):
    """Hia5 PacBio molecules over a phased nucleosome array; each is 4 kb, soft-clipped 1.8 kb on both sides."""
    rng = np.random.default_rng(seed)
    contig = 20_000
    header = pysam.AlignmentHeader.from_dict({'HD': {'VN': '1.6', 'SO': 'coordinate'},
                                               'SQ': [{'SN': 'chr1', 'LN': contig}]})
    reads = []
    for i in range(n_reads):
        seq = ''.join(rng.choice(list('ACGT'), size=LENGTH))
        protected = np.zeros(LENGTH, bool)
        pos = int(rng.integers(0, 60))
        while pos < LENGTH:
            protected[pos:pos + 147] = True
            pos += 195
        rate = np.where(protected, 0.01, 0.7)
        entries, ml = [], []
        for base, strand in (('A', '+'), ('T', '-')):
            hits = [k for k, b in enumerate(seq) if b == base and rng.random() < rate[k]]
            entries.append(_mm_entry(seq, hits, base, strand))
            ml.extend([250] * len(hits))
        r = pysam.AlignedSegment(header)
        r.query_name = f'clip{i}'
        r.query_sequence = seq
        r.flag = 0
        r.reference_id = 0
        r.reference_start = 1000 + 2000 * i
        r.mapping_quality = 60
        r.cigartuples = [(4, CLIP), (0, ALIGNED), (4, CLIP)]
        r.query_qualities = pysam.qualitystring_to_array('I' * LENGTH)
        r.set_tag('MM', ''.join(entries))
        r.set_tag('ML', ml)
        reads.append(r)
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        for r in reads:
            out.write(r)
    pysam.index(str(path))
    return str(path)


def _one_read(tmp_path):
    with pysam.AlignmentFile(_softclipped_bam(tmp_path / 'one.bam', n_reads=1)) as bam:
        return next(bam.fetch(until_eof=True))


def test_modeled_length_by_mode(tmp_path):
    from fiberhmm.inference.engine import configure_daf_unaligned_mask
    read = _one_read(tmp_path)
    assert read.query_alignment_length == ALIGNED and read.query_length == LENGTH
    assert modeled_read_length(read, 'pacbio-fiber') == LENGTH
    assert modeled_read_length(read, 'nanopore-fiber') == LENGTH
    assert modeled_read_length(read, None) == LENGTH
    configure_daf_unaligned_mask(True)
    assert modeled_read_length(read, 'daf') == ALIGNED
    configure_daf_unaligned_mask(False)
    assert modeled_read_length(read, 'daf') == LENGTH


def test_streaming_filter_keeps_softclipped_hia5_and_masks_daf(tmp_path):
    from fiberhmm.inference.engine import configure_daf_unaligned_mask
    read = _one_read(tmp_path)
    hia5 = ReadFilterConfig(min_read_length=1000, mode='pacbio-fiber')
    assert streaming_skip_reason(read, hia5) is None                     # was 'too_short' (400 bp aligned)
    assert streaming_skip_reason(read, ReadFilterConfig(min_read_length=5000, mode='pacbio-fiber')) == 'too_short'
    configure_daf_unaligned_mask(True)
    assert streaming_skip_reason(read, ReadFilterConfig(min_read_length=1000, mode='daf')) == 'too_short'
    configure_daf_unaligned_mask(False)
    assert streaming_skip_reason(read, ReadFilterConfig(min_read_length=1000, mode='daf')) is None


def test_too_short_hint_counts_mostly_soft_clipped_reads(tmp_path):
    read = _one_read(tmp_path)
    counts = new_skip_counts()
    for _ in range(40):
        count_skip(counts, 'too_short', read)
    assert counts['too_short'] == 40 and counts['too_short_soft_clipped'] == 40
    assert 'too_short_soft_clipped' not in dict(printable_skip_reasons(counts))
    hint = too_short_hint(counts, total=100, mode='daf', min_read_length=1000)
    assert hint and 'soft-clipped' in hint and '--min-read-length' in hint
    assert too_short_hint(counts, total=10_000) is None                  # a small share: no note


def _benchmark_model():
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'hia5_pacbio.json')
    return os.path.abspath(path)


def _called(path):
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        return {r.query_name: r for r in bam.fetch(until_eof=True)}


@pytest.mark.parametrize('strategy', ['streaming', 'region'])
def test_hia5_call_annotates_softclipped_molecules(tmp_path, strategy):
    from fiberhmm.inference.region_pipeline import _process_bam_region_parallel_fused
    from fiberhmm.inference.streaming_pipeline import _process_bam_streaming_pipeline_fused
    from fiberhmm.io.ma_tags import parse_ma_tag
    src = _softclipped_bam(tmp_path / 'clip.bam')
    out = tmp_path / f'{strategy}.bam'
    common = dict(output_bam=str(out), recall_model_path=None, train_rids=set(), edge_trim=10, circular=False,
                  mode='pacbio-fiber', context_size=3, msp_min_size=0, nuc_min_size=85, min_mapq=0,
                  prob_threshold=0, min_read_length=1000, with_scores=False, min_llr=5.0, min_opps=3,
                  unify_threshold=90, emission_uplift=1.0, also_write_legacy=True, downstream_compat=False,
                  n_cores=1, io_threads=1)
    if strategy == 'streaming':
        _process_bam_streaming_pipeline_fused(input_bam=src, model_path=_benchmark_model(), max_reads=0,
                                              chunk_size=10, **common)
    else:
        _process_bam_region_parallel_fused(input_bam=src, apply_model_path=_benchmark_model(),
                                           region_size=10_000_000, skip_scaffolds=False, chroms=None, **common)
    reads = _called(out)
    assert len(reads) == 6
    for name, r in reads.items():
        assert r.has_tag('MA'), f'{name} skipped as too short'
        nucs = parse_ma_tag(r.get_tag('MA'))['nuc']
        # the HMM models the soft clips too: nucleosomes are called inside them
        assert any(s + n <= CLIP for s, n in nucs) and any(s >= CLIP + ALIGNED for s, n in nucs)


def test_read_bam_keeps_softclipped_hia5_reads(tmp_path):
    from fiberhmm.core.bam_reader import read_bam
    src = _softclipped_bam(tmp_path / 'clip.bam', n_reads=2)
    assert len(list(read_bam(src, min_mapq=0, min_read_length=1000, mode='pacbio-fiber'))) == 2
    assert len(list(read_bam(src, min_mapq=0, min_read_length=5000, mode='pacbio-fiber'))) == 0


def test_modeled_length_tolerates_minimal_records():
    """Stand-ins without query_length (or without a CIGAR) still get a length."""
    from types import SimpleNamespace
    seq_only = SimpleNamespace(is_unmapped=False, query_sequence='A' * 1500, query_alignment_length=300)
    assert modeled_read_length(seq_only, 'pacbio-fiber') == 1500
    span_only = SimpleNamespace(is_unmapped=False, reference_start=100, reference_end=1300)
    assert modeled_read_length(span_only, 'pacbio-fiber') == 1200
    assert modeled_read_length(span_only, 'daf') == 1200
    unmapped = SimpleNamespace(is_unmapped=True, query_sequence='ACGT')
    assert modeled_read_length(unmapped) == 4
    counts = new_skip_counts()
    count_skip(counts, 'too_short', span_only)              # no CIGAR: not counted as soft-clipped
    assert counts['too_short'] == 1 and counts['too_short_soft_clipped'] == 0
