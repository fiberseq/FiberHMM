"""DAF strand-swap marks and molecule-wide call preprocessing.

``xc:Z:strand_swap`` is a local SAM tag (lowercase namespace). Together
with 0x200 it identifies FiberHMM chimeras; unrelated QC failures are untouched.
Detection uses the caller's extraction and masking policy, not a second test.
"""
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import pysam

CHIMERA_TAG = 'xc'
CHIMERA_REASON = 'strand_swap'
CHIMERA_QC_TAG = 'xq'  # original QC-fail state before our mark (0 or 1)

# The strict detector: a conservative floor for the chimera rate, counted in
# the same pass as the default (upper) estimate. Flag/remove/split act on the
# default decision only.
STRICT_MIN_SEG = 10
STRICT_PURITY = 0.9
# Grades of the strict fraction (QC row and the call's stderr warning).
WARN_FRACTION = 0.05
HIGH_FRACTION = 0.15
GRADE_MIN_READS = 100
CHIMERA_NOTE = ('Same-strand chimeras cannot be detected, so the true chimera '
                'rate is higher than these counts. Read-count and occupancy '
                'quantification from a library with many chimeras should be '
                'taken with caution.')
REGION_BIN_BP = 10_000
REGION_MIN_READS = 20
REGION_MAX_ROWS = 50


def _has_chimera_tag(read):
    return read.has_tag(CHIMERA_TAG) and read.get_tag(CHIMERA_TAG) == CHIMERA_REASON


def is_marked_chimera(read):
    return (bool(getattr(read, 'is_qcfail', False))
            and _has_chimera_tag(read))


def clear_chimera_mark(read):
    """Restore QC-fail only when our exact mark records flag ownership.

    Older marks without ownership metadata cannot prove who set 0x200;
    conservatively preserve the flag in that case.
    """
    if _has_chimera_tag(read):
        if read.has_tag(CHIMERA_QC_TAG):
            if read.get_tag(CHIMERA_QC_TAG) == 0:
                read.is_qcfail = False
            read.set_tag(CHIMERA_QC_TAG, None)
        read.set_tag(CHIMERA_TAG, None)


def mark_chimera(read):
    # Repeated marking must not turn our own QC failure into a prior one.
    if not _has_chimera_tag(read):
        read.set_tag(CHIMERA_QC_TAG, int(read.is_qcfail), value_type='i')
    read.is_qcfail = True
    read.set_tag(CHIMERA_TAG, CHIMERA_REASON, value_type='Z')


def seekable_input(path):
    """Regular local inputs can be reopened; stdin and FIFOs need a spool."""
    return path != '-' and Path(path).is_file()


def _chimera_events(read, ref, *, min_seg=None):
    """(CT, GA) event query positions the caller's detector sees, or None.

    SNP-masked reference positions are excluded (the caller's mask), and on
    R/Y input also the unaligned bases, exactly as calling does.
    """
    from . import engine
    from fiberhmm.daf.encoder import get_daf_positions

    min_seg = engine._DAF_CHIMERA_CFG['min_seg'] if min_seg is None else min_seg
    seq = read.query_sequence
    if not seq:
        return None
    if engine.has_iupac_encoding(seq):
        return engine._iupac_chimera_events(
            read, seq, engine._stub_or_live_excluded(read), ref, min_seg=min_seg)
    result = get_daf_positions(
        # Both arrays are populated regardless of the chosen strand. Force
        # collection past the dominant-strand tie check; the spatial test
        # evaluates both CT->GA and GA->CT swaps.
        read, force_strand='CT', ref_fasta=ref,
        excluded_reference_positions=engine._daf_reference_mask(read))
    return None if result is None else (result[0], result[1])


def _is_chimeric(read, ref, *, min_seg=None, purity=None):
    """Only the caller's chimera decision, without extracting call evidence."""
    from . import engine
    from fiberhmm.daf.encoder import is_daf_chimera

    min_seg = engine._DAF_CHIMERA_CFG['min_seg'] if min_seg is None else min_seg
    purity = engine._DAF_CHIMERA_CFG['purity'] if purity is None else purity
    events = _chimera_events(read, ref, min_seg=min_seg)
    return events is not None and is_daf_chimera(
        events[0], events[1], min_seg_events=min_seg, purity=purity)


def grade_chimeras(n_checked, n_strict):
    """QC grade of the strict chimera fraction.

    PASS below 5%, WARN from 5% (severity 'moderate') and from 15% (severity
    'high'); INSUFFICIENT with fewer than 100 checked reads or no strict
    count. Never FAIL: flagged chimeras are excluded from analysis and the
    remaining reads are usable.
    """
    if not n_checked or n_strict is None or n_checked < GRADE_MIN_READS:
        return {'status': 'INSUFFICIENT', 'severity': None, 'basis': 'strict',
                'fraction': (n_strict / n_checked
                             if n_checked and n_strict is not None else None),
                'warn_fraction': WARN_FRACTION, 'high_fraction': HIGH_FRACTION,
                'min_reads': GRADE_MIN_READS}
    fraction = n_strict / n_checked
    if fraction >= HIGH_FRACTION:
        status, severity = 'WARN', 'high'
    elif fraction >= WARN_FRACTION:
        status, severity = 'WARN', 'moderate'
    else:
        status, severity = 'PASS', None
    return {'status': status, 'severity': severity, 'basis': 'strict',
            'fraction': fraction, 'warn_fraction': WARN_FRACTION,
            'high_fraction': HIGH_FRACTION, 'min_reads': GRADE_MIN_READS}


def chimera_warning(stats):
    """A prominent stderr warning for a high strict chimera fraction, or None."""
    grade = (stats or {}).get('grade') or {}
    if grade.get('status') != 'WARN':
        return None
    strict = 100 * grade['fraction']
    default = stats.get('fraction')
    default_text = '' if default is None else f' (default detector: {100 * default:.1f}%)'
    if grade.get('severity') == 'high':
        return (f"WARNING: very high DAF strand-swap chimera rate: at least "
                f"{strict:.1f}% of reads{default_text}, and same-strand chimeras "
                "are not detectable. Treat read counts and occupancy from this "
                "library with strong caution.")
    return (f"WARNING: high DAF strand-swap chimera rate: at least {strict:.1f}% "
            f"of reads{default_text}; same-strand chimeras are not detectable. "
            "Take read counts and occupancy from this library with caution.")


def chimera_summary_lines(stats, indent='  '):
    """Human-readable lines for a chimera statistics block (stderr, QC text)."""
    if not stats or not stats.get('detected') or stats.get('mode') in ('keep', 'n/a'):
        return []
    checked = stats.get('n_reads_checked') or 0
    lines = []
    fraction = stats.get('fraction')
    strict = stats.get('fraction_strict')
    pct = (lambda v: 'NA' if v is None else f'{100 * v:.2f}%')
    lines.append(f"{indent}DAF strand-swap chimeras ({stats.get('mode')}): "
                 f"{stats.get('n_chimeric', 0):,} / {checked:,} checked reads "
                 f"({pct(fraction)}, default detector min_seg={stats.get('min_seg')}, "
                 f"purity={stats.get('purity')}; upper estimate)")
    if stats.get('n_chimeric_strict') is not None:
        lines.append(f"{indent}  strict detector (min_seg={stats.get('strict_min_seg')}, "
                     f"purity={stats.get('strict_purity')}): "
                     f"{stats['n_chimeric_strict']:,} ({pct(strict)}; conservative floor)")
    grade = stats.get('grade') or {}
    if grade.get('status'):
        severity = f" ({grade['severity']})" if grade.get('severity') else ''
        lines.append(f"{indent}  grade: {grade['status']}{severity} "
                     f"[strict fraction; WARN >= {100 * WARN_FRACTION:.0f}%, "
                     f"high >= {100 * HIGH_FRACTION:.0f}%]")
    split = stats.get('split')
    if split:
        lines.append(f"{indent}  split: {split['n_records_split']:,} records cut into "
                     f"{split['n_pieces_written']:,} pieces "
                     f"({split['n_pieces_too_short']:,} too short, dropped); "
                     f"{pct(split.get('fraction_bases_recovered'))} of chimeric "
                     f"bases kept")
    lines.append(f"{indent}  note: {CHIMERA_NOTE}")
    return lines


class ChimeraDecisions(set):
    """Named molecule decisions plus the policy for independent unnamed records.

    Unavailable QNAMEs are never stored in the set. Re-evaluate those records
    with the prepass policy during replay (also after region partitioning).
    The policy is pickleable so region workers and NRL sampling use it too.
    """
    anonymous_policy = None


def is_precalled_chimera(read, decisions):
    if read.query_name not in (None, '*', ''):
        return read.query_name in decisions
    policy = getattr(decisions, 'anonymous_policy', None)
    if policy is None:  # e.g. an authoritative empty keep-mode decision
        return False
    from .read_filters import streaming_skip_reason
    config, reference, min_seg, purity = policy
    if streaming_skip_reason(read, config) is not None:
        return False
    with ExitStack() as stack:
        ref = stack.enter_context(pysam.FastaFile(reference)) if reference else None
        return _is_chimeric(read, ref, min_seg=min_seg, purity=purity)


def _new_stats_fields(mode, n_checked=0, n_strict=None, regions=None, split=None):
    """3.0.4 fields, appended after the 3.0.3 ones (key order is kept)."""
    detected = mode not in ('keep', 'n/a')
    return dict(
        n_chimeric_strict=n_strict if detected else None,
        fraction_strict=(n_strict / n_checked
                         if detected and n_checked and n_strict is not None else None),
        strict_min_seg=STRICT_MIN_SEG if detected else None,
        strict_purity=STRICT_PURITY if detected else None,
        grade=grade_chimeras(n_checked, n_strict) if detected else None,
        regions=regions if detected else None,
        split=split,
        note=CHIMERA_NOTE,
    )


def empty_chimera_stats(mode, min_seg=5, purity=0.8):
    return dict(detected=False, mode=mode, n_reads_checked=0, n_chimeric=0,
                fraction=None, min_seg=min_seg, purity=purity,
                n_records_affected=0, n_stale_marks_cleared=0,
                full_run=True, count_unit='unique read names before removal',
                **_new_stats_fields(mode))


def _quantiles(values):
    import numpy as np
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {'n': int(array.size), 'mean': float(array.mean()),
            **{f'q{int(100 * q):02d}': float(np.quantile(array, q))
               for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}


def _region_rows(bin_of, bin_spans, anonymous_bins, chimeric, strict):
    """Per-region (adjacent occupied 10-kb start bins) chimera fractions."""
    counts = {}
    for name, key in bin_of.items():
        row = counts.setdefault(key, [0, 0, 0])
        row[0] += 1
        row[1] += name in chimeric
        row[2] += name in strict
    for key, (checked, default, strict_n) in anonymous_bins.items():
        row = counts.setdefault(key, [0, 0, 0])
        row[0] += checked
        row[1] += default
        row[2] += strict_n
    by_chrom = {}
    for (chrom, index) in counts:
        by_chrom.setdefault(chrom, []).append(index)
    regions = []
    for chrom, indices in by_chrom.items():
        group = []
        for index in sorted(indices) + [None]:
            if group and (index is None or index > group[-1] + 1):
                checked = sum(counts[(chrom, i)][0] for i in group)
                default = sum(counts[(chrom, i)][1] for i in group)
                strict_n = sum(counts[(chrom, i)][2] for i in group)
                start = min(bin_spans[(chrom, i)][0] for i in group)
                end = max(bin_spans[(chrom, i)][1] for i in group)
                regions.append(dict(
                    chrom=chrom, start=int(start), end=int(end),
                    region='unmapped' if chrom == '*' else f'{chrom}:{start + 1}-{end}',
                    n_reads_checked=checked, n_chimeric=default,
                    fraction=default / checked if checked else None,
                    n_chimeric_strict=strict_n,
                    fraction_strict=strict_n / checked if checked else None))
                group = []
            if index is not None:
                group.append(index)
    regions.sort(key=lambda r: (-r['n_reads_checked'], r['chrom'], r['start']))
    shown = [r for r in regions if r['n_reads_checked'] >= REGION_MIN_READS]
    return {'n_regions': len(shown), 'min_reads': REGION_MIN_READS,
            'bin_bp': REGION_BIN_BP, 'max_rows': REGION_MAX_ROWS,
            'unit': 'reads by the start bin of their primary (else first eligible) record; '
                    'regions are runs of adjacent occupied bins',
            'rows': shown[:REGION_MAX_ROWS]}


def prepare_chimeras(input_bam, output_bam=None, *, mode='flag', daf=True,
                     min_seg=5, purity=0.8, prob_threshold=128,
                     reference=None, snp_mask=None, min_mapq=0,
                     min_read_length=0, process_unmapped=False,
                     io_threads=1, chimeric_names=None, spool_bam=None,
                     alignments='primary-supplementary'):
    """Decide QNAMEs in a read-only pass; spool only non-seekable inputs.

    The CLI passes a mutable ``ChimeraDecisions`` set and applies marks/removal
    in the caller, so file input needs no intermediate BAM. A plain set gets
    only named decisions; ChimeraDecisions also carries the unnamed replay
    policy. For streams,
    ``spool_bam`` supplies the replay path. ``output_bam`` optionally writes
    a marked/filtered BAM for standalone use. Counts precede removal.

    The default and the strict detector (:data:`STRICT_MIN_SEG`,
    :data:`STRICT_PURITY`) are evaluated on the same events in this pass.
    Mode ``split`` requires ``spool_bam``: each eligible named record that is
    itself chimeric is written there as two pieces
    (:mod:`fiberhmm.inference.chimera_split`); pieces shorter than
    ``min_read_length`` are dropped, and a record with no piece left is kept
    whole. Every other record of a chimeric name is flagged as in flag mode.
    A coordinate-sorted input gives a sorted, indexed spool.
    """
    from . import engine
    from .read_filters import ReadFilterConfig, streaming_skip_reason
    from fiberhmm.daf.encoder import daf_chimera_split, is_daf_chimera

    if mode not in ('flag', 'remove', 'keep', 'split'):
        raise ValueError(f'Invalid chimera mode: {mode}')
    if min_seg < 1 or not 0.5 < purity <= 1:
        raise ValueError('chimera min_seg must be >= 1 and purity in (0.5, 1]')
    splitting = bool(daf and mode == 'split')
    if splitting and spool_bam is None:
        raise ValueError('chimera split mode requires spool_bam')
    engine.configure_daf_chimera_filter(daf and mode != 'keep', min_seg, purity)
    engine.configure_daf_snp_mask(snp_mask)
    config = ReadFilterConfig(min_mapq=min_mapq, min_read_length=min_read_length,
                              primary_only=alignments,
                              process_unmapped=process_unmapped, mode='daf',
                              has_reference=reference is not None)
    chimeric, strict = ChimeraDecisions(), set()
    if daf and mode != 'keep':
        chimeric.anonymous_policy = (config, reference, min_seg, purity)
    anonymous_checked = anonymous_chimeric = anonymous_strict = 0
    counts = Counter()
    stale = 0
    # Regions: one start bin per read name (its primary record if eligible).
    bin_of, bin_spans, anonymous_bins = {}, {}, {}
    event_floor = min(min_seg, STRICT_MIN_SEG)
    split_info = dict(n_records_split=0, n_pieces_written=0, n_pieces_too_short=0,
                      n_records_kept_whole=0, n_pieces_still_chimeric=0,
                      chimeric_bases=0, recovered_bases=0)
    gaps, piece_lengths, still_chimeric = [], [], set()

    def decide(read):
        events = _chimera_events(read, ref, min_seg=event_floor)
        if events is None:
            return None, False
        ct, ga = events
        hit = (daf_chimera_split(ct, ga, min_seg, purity) if splitting
               else is_daf_chimera(ct, ga, min_seg_events=min_seg, purity=purity))
        return hit or None, bool(is_daf_chimera(
            ct, ga, min_seg_events=STRICT_MIN_SEG, purity=STRICT_PURITY))

    # The optional standalone writer owns its private spool. The CLI owns
    # its stream spool's lifetime, which extends through calling.
    private_spool = False
    if not seekable_input(input_bam) and spool_bam is None:
        if output_bam is None:
            raise ValueError('Non-seekable input requires spool_bam')
        spool_bam = str(Path(output_bam).with_suffix('.source.bam'))
        private_spool = True
    replay = spool_bam or input_bam
    try:
        with ExitStack() as stack:
            ref = stack.enter_context(pysam.FastaFile(reference)) if reference else None
            bam = stack.enter_context(pysam.AlignmentFile(
                input_bam, 'rb', check_sq=False, threads=io_threads))
            sort_spool = bool(splitting and (bam.header.to_dict().get('HD') or {})
                              .get('SO') == 'coordinate')
            spool_target = (str(spool_bam) + '.unsorted.bam') if sort_spool else spool_bam
            out = (stack.enter_context(pysam.AlignmentFile(
                spool_target, 'wb', header=bam.header, threads=io_threads))
                   if spool_bam else None)
            for read in bam.fetch(until_eof=True):
                named = read.query_name not in (None, '*', '')
                if named:
                    counts[read.query_name] += 1
                stale += int(_has_chimera_tag(read))
                clear_chimera_mark(read)
                pieces = None
                if daf and mode != 'keep' and streaming_skip_reason(read, config) is None:
                    # Unmapped records (--process-unmapped) share one region.
                    if read.is_unmapped or read.reference_name is None:
                        key = ('*', 0)
                        start = end = 0
                    else:
                        key = (read.reference_name,
                               int(read.reference_start) // REGION_BIN_BP)
                        start = int(read.reference_start)
                        end = int(read.reference_end or start)
                    span = bin_spans.get(key)
                    bin_spans[key] = ((start, end) if span is None else
                                      (min(span[0], start), max(span[1], end)))
                    if named:
                        name = read.query_name
                        if (name not in bin_of
                                or not (read.is_supplementary or read.is_secondary)):
                            bin_of[name] = key
                        if splitting or name not in chimeric or name not in strict:
                            hit, strict_hit = decide(read)
                            if hit:
                                chimeric.add(name)
                            if strict_hit:
                                strict.add(name)
                            if splitting and hit:
                                pieces = _split_pieces(
                                    read, bam.header, hit, min_read_length, ref,
                                    decide, split_info, gaps, piece_lengths,
                                    still_chimeric)
                    else:
                        anonymous_checked += 1
                        hit, strict_hit = decide(read)
                        anonymous_chimeric += int(bool(hit))
                        anonymous_strict += int(strict_hit)
                        row = anonymous_bins.setdefault(key, [0, 0, 0])
                        row[0] += 1
                        row[1] += int(bool(hit))
                        row[2] += int(strict_hit)
                if out is not None:
                    if pieces:
                        for piece in pieces:
                            out.write(piece)
                    else:
                        out.write(read)
        if sort_spool:
            pysam.sort('-o', str(spool_bam), spool_target)
            Path(spool_target).unlink(missing_ok=True)
            pysam.index(str(spool_bam))
        if chimeric_names is not None:
            chimeric_names.update(chimeric)
            # Pieces that still show a swap (reads with several switches) are
            # flagged like any chimera; they are not counted as new reads.
            chimeric_names.update(still_chimeric)
            if isinstance(chimeric_names, ChimeraDecisions):
                chimeric_names.anonymous_policy = chimeric.anonymous_policy
        if output_bam is not None:
            decisions = ChimeraDecisions(chimeric | still_chimeric)
            decisions.anonymous_policy = chimeric.anonymous_policy
            with pysam.AlignmentFile(replay, 'rb', check_sq=False) as bam, \
                    pysam.AlignmentFile(output_bam, 'wb', header=bam.header,
                                        threads=io_threads) as out:
                for read in bam.fetch(until_eof=True):
                    clear_chimera_mark(read)
                    if is_precalled_chimera(read, decisions):
                        if mode == 'remove':
                            continue
                        mark_chimera(read)
                    out.write(read)
    finally:
        if private_spool:
            Path(spool_bam).unlink(missing_ok=True)
    n_checked = len(bin_of) + anonymous_checked
    n_chimeric = len(chimeric) + anonymous_chimeric
    n_strict = len(strict) + anonymous_strict
    split = None
    if splitting:
        split = dict(split_info)
        split['n_reads_split'] = len(split.pop('_names', ()))
        split['min_piece_length'] = int(min_read_length)
        split['fraction_bases_recovered'] = (
            split['recovered_bases'] / split['chimeric_bases']
            if split['chimeric_bases'] else None)
        split['gap_reference_bp'] = _quantiles(gaps)
        split['piece_aligned_bp'] = _quantiles(piece_lengths)
        split['tags'] = {'xc': 'Z:split', 'xi': 'i piece index (0 = left on the reference)',
                         'xn': 'Z original read name',
                         'xb': 'B:i breakpoint gap [start, end), 0-based on the reference'}
        split['dropped_tags'] = 'MM/ML and every tag not in RG, ds, di, qs, du, ch, rn, fn; ' \
                                'MD/NM recomputed'
    stats = dict(detected=bool(daf and mode != 'keep'), mode=mode if daf else 'n/a',
                 n_reads_checked=n_checked, n_chimeric=n_chimeric,
                 fraction=n_chimeric / n_checked if n_checked else None,
                 min_seg=min_seg if daf else None, purity=purity if daf else None,
                 n_records_affected=sum(counts[name] for name in chimeric) + anonymous_chimeric,
                 n_stale_marks_cleared=stale, full_run=True,
                 count_unit='unique read names before removal; unnamed records counted individually')
    stats.update(_new_stats_fields(
        stats['mode'], n_checked, n_strict,
        _region_rows(bin_of, bin_spans, anonymous_bins, chimeric, strict)
        if daf and mode != 'keep' else None, split))
    return stats


def _split_pieces(read, header, split, min_read_length, ref, decide, info,
                  gaps, piece_lengths, still_chimeric):
    """Pieces to write in place of ``read`` (None: keep the record whole)."""
    from .chimera_split import split_record

    pieces, gap = split_record(read, header, split)
    kept = []
    for piece in pieces:
        if piece is None:
            continue
        length = piece.query_alignment_length or 0
        if length < max(1, min_read_length):
            info['n_pieces_too_short'] += 1
            continue
        kept.append(piece)
    info['chimeric_bases'] += read.query_alignment_length or 0
    if not kept:
        info['n_records_kept_whole'] += 1
        return None
    info['n_records_split'] += 1
    info.setdefault('_names', set()).add(read.query_name)
    info['n_pieces_written'] += len(kept)
    if gap is not None:
        gaps.append(gap[1] - gap[0])
    for piece in kept:
        length = piece.query_alignment_length or 0
        info['recovered_bases'] += length
        piece_lengths.append(length)
        if decide(piece)[0]:
            info['n_pieces_still_chimeric'] += 1
            still_chimeric.add(piece.query_name)
    return kept
