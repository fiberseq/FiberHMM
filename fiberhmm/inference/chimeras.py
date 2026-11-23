"""DAF strand-swap marks and molecule-wide call preprocessing.

``xc:Z:strand_swap`` is a local SAM tag (lowercase namespace). Together
with 0x200 it identifies FiberHMM chimeras; unrelated QC failures are untouched.
Detection uses the caller's extraction and masking policy, not a second test.
"""
import copy
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import pysam

CHIMERA_TAG = 'xc'
CHIMERA_REASON = 'strand_swap'
CHIMERA_QC_TAG = 'xq'  # original QC-fail state before our mark (0 or 1)

# Two DAF strand-swap detectors (``--chimera-detector``):
#   standard (default): the 3.2.1 detector, one CT/GA switch with min_seg
#     events per segment at purity; every event counts.
#   strict (opt-in): base-quality filter, minimum span, several switches,
#     the chimera-only error mask, duplicates not counted and a QC grade.
# Split mode (--chimeras split) works with either: the run's detector flags
# the reads, and the strict segmentation (daf_chimera_segments) finds the
# breakpoints of the flagged records (:func:`standard_breakpoints` for the
# standard detector).
# The standard detector's code path is the 3.2.1 one (prepare_chimeras ->
# _prepare_chimeras_standard, _is_chimeric, encoder.is_daf_chimera); the
# strict detector has its own functions (``*_strict``).
DETECTORS = ('standard', 'strict')
DEFAULT_DETECTOR = 'standard'
from fiberhmm.defaults import CHIMERA_MIN_SEG as STANDARD_MIN_SEG    # 5: dominant events per segment
from fiberhmm.defaults import CHIMERA_PURITY as STANDARD_PURITY      # 0.8: dominant fraction of a segment's events
STRICT_MIN_SEG = 10
STRICT_PURITY = 0.9
STRICT_MIN_BASEQ = 15       # events need base quality >= this (when QUAL is present)
STRICT_MIN_SPAN = 200       # query bp from a segment's first to last dominant event
STRICT_MULTI_SWITCH = True  # also call reads with several switches (CT->GA->CT)
STRICT_ERROR_MASK = True    # chimera-only recurrent error mask from the SNP screen
# Chimera-only recurrent error mask (computed in the DAF SNP screen's passes,
# fiberhmm.daf.snps). Voters are clearly single-strand, non-chimeric fibers;
# a reference site is masked from chimera events when its opposite-direction
# mismatches among voters are recurrent and improbable as random error.
ERROR_MASK_VOTER_MIN_EVENTS = 10
ERROR_MASK_VOTER_MIN_PURITY = 0.92
ERROR_MASK_VOTER_MIN_BASEQ = 15     # fixed; independent of --chimera-min-baseq
ERROR_MASK_MIN_DEPTH = 20           # voter fibers covering the site
ERROR_MASK_MIN_MISMATCHES = 4       # voter fibers with the opposite event
ERROR_MASK_MIN_FRACTION = 0.05
ERROR_MASK_BACKGROUND_RATE = 0.01   # binomial null per-fiber error rate
ERROR_MASK_MAX_PVALUE = 1e-4        # P(X >= k | depth, background) must be below
# Grades of the chimera fraction (QC row and the call's stderr warning).
WARN_FRACTION = 0.10
HIGH_FRACTION = 0.20
GRADE_MIN_READS = 100
CHIMERA_NOTE = ('Only opposite-strand (CT/GA) chimeras are detectable; '
                'same-strand chimeras are invisible. With balanced strands, '
                'the unflagged reads still contain roughly as many undetected '
                'chimeras as were flagged. Read-count, occupancy and '
                'long-range (co-accessibility, phasing) results from a library '
                'with many chimeras should be taken with caution.')
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


def detector_config(min_seg=STRICT_MIN_SEG, purity=STRICT_PURITY,
                    min_baseq=STRICT_MIN_BASEQ, min_span=STRICT_MIN_SPAN,
                    multi_switch=STRICT_MULTI_SWITCH, error_mask=None):
    """The detector settings as one picklable dict (``error_mask``: BED path)."""
    return {'min_seg': int(min_seg), 'purity': float(purity),
            'min_baseq': int(min_baseq), 'min_span': int(min_span),
            'multi_switch': bool(multi_switch),
            'error_mask': str(error_mask) if error_mask else None}


def _active_config(overrides=None):
    from . import engine
    cfg = {key: engine._DAF_CHIMERA_CFG[key]
           for key in ('min_seg', 'purity', 'min_baseq', 'min_span',
                       'multi_switch', 'error_mask')}
    for key, value in (overrides or {}).items():
        if value is not None or key == 'error_mask':
            cfg[key] = value
    return cfg


_ERROR_MASKS = {}


def load_error_mask(path):
    """``{contig: frozenset(positions)}`` of a chimera error-mask BED (cached)."""
    if not path:
        return {}
    mask = _ERROR_MASKS.get(path)
    if mask is None:
        from fiberhmm.daf.snps import load_snp_mask
        if len(_ERROR_MASKS) > 8:
            _ERROR_MASKS.clear()
        mask = _ERROR_MASKS[path] = load_snp_mask(path)
    return mask


def _error_masked_query_positions(read, error_mask):
    """Query positions of ``read`` aligned to chimera error-mask sites."""
    sites = load_error_mask(error_mask).get(getattr(read, 'reference_name', None))
    if not sites or not hasattr(read, 'get_aligned_pairs'):
        return set()
    import numpy as np
    from fiberhmm.daf.aligned_arrays import matched_base_arrays
    from fiberhmm.daf.snps import wrapped_reference_sites
    wrapped = wrapped_reference_sites(read, sites)
    if wrapped is sites:
        # Skip reads that cover no masked site (most reads of a large genome).
        ordered = _sorted_sites(error_mask, read.reference_name, sites)
        start, end = read.reference_start, read.reference_end or read.reference_start
        lo = int(np.searchsorted(ordered, start))
        if lo >= ordered.size or ordered[lo] >= end:
            return set()
    # Same query coordinates as the event scan (get_daf_positions strips CIGAR
    # 'P' on a copy; pysam's aligned pairs would shift every later base).
    from fiberhmm.daf.encoder import _without_cigar_padding
    read = _without_cigar_padding(read)
    arrays = matched_base_arrays(read)
    if arrays is not None:
        qpos, rpos = arrays[0], arrays[1]
        ordered = np.fromiter(wrapped, dtype=np.int64, count=len(wrapped))
        return set(qpos[np.isin(rpos, ordered)].tolist())
    try:
        return {int(q) for q, r in read.get_aligned_pairs(matches_only=True)
                if r in wrapped}
    except (ValueError, TypeError, IndexError):
        return set()


_SORTED_SITES = {}


def _sorted_sites(path, contig, sites):
    key = (path, contig)
    ordered = _SORTED_SITES.get(key)
    if ordered is None or ordered[0] is not sites:
        import numpy as np
        if len(_SORTED_SITES) > 4096:
            _SORTED_SITES.clear()
        ordered = (sites, np.array(sorted(sites), dtype=np.int64))
        _SORTED_SITES[key] = ordered
    return ordered[1]


def filter_chimera_events(read, ct, ga, cfg):
    """``(ct, ga)`` without low-quality and error-masked events.

    Events need base quality >= ``cfg['min_baseq']`` when the record has
    QUAL (no filter without QUAL, or at 0); events on chimera error-mask
    sites (``cfg['error_mask']``) are dropped. Chimera detection only:
    calling never sees this filter.
    """
    min_baseq = int(cfg.get('min_baseq') or 0)
    quals = None
    if min_baseq > 0:
        try:
            quals = getattr(read, 'query_qualities', None)
        except (ValueError, TypeError):
            quals = None
    masked = (_error_masked_query_positions(read, cfg['error_mask'])
              if cfg.get('error_mask') else ())
    if quals is None and not masked:
        return list(ct), list(ga)
    if quals is not None:
        import numpy as np
        quals = np.asarray(quals)
        size = quals.size

        def keep(p):
            return p not in masked and 0 <= p < size and quals[p] >= min_baseq
    else:
        def keep(p):
            return p not in masked
    return [p for p in ct if keep(p)], [p for p in ga if keep(p)]


def decide_chimera(ct, ga, cfg, *, segments=False):
    """The strict detector's decision on filtered events: bool, or (``segments``)
    the :func:`~fiberhmm.daf.encoder.daf_chimera_segments` result or None."""
    from fiberhmm.daf.encoder import daf_chimera_segments, is_daf_chimera_strict
    kwargs = dict(min_span=cfg['min_span'], multi_switch=cfg['multi_switch'])
    if segments:
        return daf_chimera_segments(ct, ga, cfg['min_seg'], cfg['purity'], **kwargs)
    return is_daf_chimera_strict(ct, ga, cfg['min_seg'], cfg['purity'], **kwargs)


def _chimera_events(read, ref, cfg=None):
    """(CT, GA) event query positions the caller's detector sees, or None.

    SNP-masked reference positions are excluded (the caller's mask), and on
    R/Y input also the unaligned bases, exactly as calling does. Then the
    chimera-only filters apply (:func:`filter_chimera_events`).
    """
    from . import engine
    from fiberhmm.daf.encoder import get_daf_positions

    cfg = _active_config() if cfg is None else cfg
    seq = read.query_sequence
    if not seq:
        return None
    if engine.has_iupac_encoding(seq):
        events = engine._iupac_chimera_events(
            read, seq, engine._stub_or_live_excluded(read), ref,
            min_seg=cfg['min_seg'])
    else:
        result = get_daf_positions(
            # Both arrays are populated regardless of the chosen strand. Force
            # collection past the dominant-strand tie check; the spatial test
            # evaluates both CT->GA and GA->CT swaps.
            read, force_strand='CT', ref_fasta=ref,
            excluded_reference_positions=engine._daf_reference_mask(read))
        events = None if result is None else (result[0], result[1])
    if events is None:
        return None
    return filter_chimera_events(read, events[0], events[1], cfg)


def _is_chimeric(read, ref, *, min_seg=None, purity=None):
    """Only the caller's chimera decision, without extracting call evidence."""
    from . import engine
    from fiberhmm.daf.encoder import get_daf_positions, is_daf_chimera

    min_seg = engine._DAF_CHIMERA_CFG['min_seg'] if min_seg is None else min_seg
    purity = engine._DAF_CHIMERA_CFG['purity'] if purity is None else purity
    seq = read.query_sequence
    if not seq:
        return False
    if engine.has_iupac_encoding(seq):
        return engine._is_iupac_daf_chimera(
            read, seq, engine._stub_or_live_excluded(read), ref,
            min_seg=min_seg, purity=purity)
    result = get_daf_positions(
        # Both arrays are populated regardless of the chosen strand. Force
        # collection past the dominant-strand tie check; the spatial test
        # below evaluates both CT->GA and GA->CT swaps.
        read, force_strand='CT', ref_fasta=ref,
        excluded_reference_positions=engine._daf_reference_mask(read))
    return result is not None and is_daf_chimera(
        result[0], result[1], min_seg_events=min_seg, purity=purity)


def _is_chimeric_strict(read, ref, cfg=None):
    """The strict detector's decision, without extracting call evidence."""
    cfg = _active_config() if cfg is None else cfg
    events = _chimera_events(read, ref, cfg)
    return events is not None and decide_chimera(events[0], events[1], cfg)


def _standard_events(read, ref, min_seg):
    """(CT, GA) event query positions the standard detector tests, or None.

    The events of :func:`_is_chimeric` (SNP-masked reference positions
    excluded; R/Y input as in :func:`engine._is_iupac_daf_chimera`), no
    chimera-only filters.
    """
    from . import engine
    from fiberhmm.daf.encoder import get_daf_positions

    seq = read.query_sequence
    if not seq:
        return None
    if engine.has_iupac_encoding(seq):
        return engine._iupac_chimera_events(
            read, seq, engine._stub_or_live_excluded(read), ref, min_seg=min_seg)
    result = get_daf_positions(
        read, force_strand='CT', ref_fasta=ref,
        excluded_reference_positions=engine._daf_reference_mask(read))
    return None if result is None else (result[0], result[1])


def standard_breakpoints(read, ref, min_seg, purity):
    """Split mode under the standard detector: the pieces of ``read``.

    The strict segmentation (:func:`~fiberhmm.daf.encoder.daf_chimera_segments`)
    applied to the standard detector's events and thresholds (``min_seg``,
    ``purity``; no minimum span, single-switch cuts, re-applied to each piece,
    no base-quality filter or error mask). With these settings its first cut
    is the standard test's own breakpoint scan, so every read the standard
    detector flags has a breakpoint (None otherwise).
    """
    from fiberhmm.daf.encoder import daf_chimera_segments
    events = _standard_events(read, ref, min_seg)
    if events is None:
        return None
    return daf_chimera_segments(events[0], events[1], min_seg, purity,
                                min_span=0, multi_switch=False)


# Split-mode provenance (statistics ``split`` block and @PG DS).
SPLIT_BREAKPOINTS = {
    'standard': ('strict segmentation (daf_chimera_segments) of the reads the '
                 'standard detector flags, on its events and thresholds '
                 '(min_seg, purity; min_span 0, single-switch cuts re-applied '
                 'to each piece, no base-quality filter or error mask)'),
    'strict': ('strict detector segmentation (daf_chimera_segments with the '
               "run's min_seg, purity, min_baseq, min_span, multi_switch and "
               'error mask)'),
}
SPLIT_BREAKPOINTS_DS = {'standard': 'strict_segmentation', 'strict': 'strict_detector'}


def _split_summary(split_info, min_read_length, gaps, piece_lengths, detector):
    """The statistics ``split`` block of a split-mode pass."""
    split = dict(split_info)
    split['n_reads_split'] = len(split.pop('_names', ()))
    split['min_piece_length'] = int(min_read_length)
    split['fraction_bases_recovered'] = (
        split['recovered_bases'] / split['chimeric_bases']
        if split['chimeric_bases'] else None)
    split['gap_reference_bp'] = _quantiles(gaps)
    split['piece_aligned_bp'] = _quantiles(piece_lengths)
    split['tags'] = {'xc': 'Z:split',
                     'xi': 'i piece index, 0..n-1 from left to right on the reference',
                     'xn': 'Z original read name',
                     'xb': 'B:i every breakpoint gap of the original record as '
                           '[start, end) pairs, 0-based on the reference, left to '
                           'right; piece xi lies between gap xi-1 and gap xi'}
    split['kept_tags'] = ('RG, ds, di, qs, du, ch, rn, fn; MM/ML (and Mm/Ml) '
                          're-indexed to the piece, MN set to its length; '
                          'MD/NM recomputed')
    split['dropped_tags'] = ('every other tag; MM/ML of a piece whose record is '
                             'hard-clipped or whose MM/ML do not parse or match '
                             'SEQ (n_pieces_modifications_dropped)')
    split['flagged_by'] = detector
    split['breakpoints'] = SPLIT_BREAKPOINTS[detector]
    return split


def _new_split_info():
    return dict(n_records_split=0, n_pieces_written=0, n_pieces_too_short=0,
                n_records_kept_whole=0, n_records_unmapped_kept_whole=0,
                n_records_paired_kept_whole=0,
                n_pieces_still_chimeric=0,
                n_pieces_modifications_kept=0, n_pieces_modifications_dropped=0,
                chimeric_bases=0, recovered_bases=0,
                n_records_several_switches=0)


def engine_chimera_hit(read, ct, ga, ref=None):
    """The strict engine filter's decision for ``read`` with raw events ``ct``/``ga``.

    A slim worker payload carries the decision made on the live record
    (``_daf_chimera``), because the stub has no QUAL or alignment.
    """
    decided = getattr(read, '_daf_chimera', None)
    if decided is not None:
        return bool(decided)
    cfg = _active_config()
    ct, ga = filter_chimera_events(read, ct, ga, cfg)
    return decide_chimera(ct, ga, cfg)


def detector_basis(min_seg, purity, min_baseq=None, min_span=None):
    """``(basis, note)`` of a count made with these detector settings.

    Strict detector only. ``basis`` is ``default`` for the strict
    detector's default settings, ``lenient`` for looser settings (for
    example min_seg 5 and purity 0.8, the standard detector's values),
    ``custom`` for any other setting and ``unknown`` when the parameters were
    not recorded. ``min_baseq``/``min_span`` of None are not compared.
    ``note`` explains a non-default basis (None for ``default``).
    """
    if min_seg is None or purity is None:
        return 'unknown', 'detector parameters not recorded'
    values = [(int(min_seg), STRICT_MIN_SEG), (float(purity), STRICT_PURITY)]
    names = ['min_seg', 'purity']
    for name, value, default in (('min_baseq', min_baseq, STRICT_MIN_BASEQ),
                                 ('min_span', min_span, STRICT_MIN_SPAN)):
        if value is not None:
            values.append((int(value), default))
            names.append(name)
    if all(value == default for value, default in values):
        return 'default', None
    setting = ', '.join(f'{name}={value:g}' for name, (value, _) in zip(names, values))
    default_setting = ', '.join(f'{name}={default:g}'
                                for name, (_, default) in zip(names, values))
    if all(value <= default for value, default in values):
        return 'lenient', (f'counted with lenient strict-detector settings {setting}; '
                           f'the strict defaults ({default_setting}) mark fewer reads')
    return 'custom', f'counted with non-default strict-detector settings ({setting})'


def grade_chimeras(n_checked, n_chimeric, min_seg=STRICT_MIN_SEG,
                   purity=STRICT_PURITY, min_baseq=None, min_span=None):
    """QC grade of the chimera fraction.

    PASS below :data:`WARN_FRACTION`, WARN from it (severity 'moderate') and
    from :data:`HIGH_FRACTION` (severity 'high'); INSUFFICIENT with fewer
    than :data:`GRADE_MIN_READS` checked reads or no count. Never FAIL:
    flagged chimeras are excluded from calls. The grade measures detectable
    (opposite-strand) chimeras only; the unflagged reads still hold about as
    many same-strand chimeras (:data:`CHIMERA_NOTE`). A count from a
    non-default detector is graded as recorded; its ``basis`` and
    ``basis_note`` say so (:func:`detector_basis`).
    """
    basis, note = detector_basis(min_seg, purity, min_baseq, min_span)
    grade = {'status': None, 'severity': None, 'basis': basis,
             'min_seg': None if min_seg is None else int(min_seg),
             'purity': None if purity is None else float(purity),
             'fraction': (n_chimeric / n_checked
                          if n_checked and n_chimeric is not None else None),
             'warn_fraction': WARN_FRACTION, 'high_fraction': HIGH_FRACTION,
             'min_reads': GRADE_MIN_READS}
    if note:
        grade['basis_note'] = note
    if not n_checked or n_chimeric is None or n_checked < GRADE_MIN_READS:
        grade['status'] = 'INSUFFICIENT'
        return grade
    fraction = grade['fraction']
    if fraction >= HIGH_FRACTION:
        grade['status'], grade['severity'] = 'WARN', 'high'
    elif fraction >= WARN_FRACTION:
        grade['status'], grade['severity'] = 'WARN', 'moderate'
    else:
        grade['status'] = 'PASS'
    return grade


def chimera_warning(stats):
    """A prominent stderr warning for a high chimera fraction, or None."""
    grade = (stats or {}).get('grade') or {}
    if grade.get('status') != 'WARN':
        return None
    percent = 100 * grade['fraction']
    basis = (f" ({grade['basis_note']})" if grade.get('basis_note') else '')
    if grade.get('severity') == 'high':
        return (f"WARNING: very high DAF strand-swap chimera rate: {percent:.1f}% "
                f"of reads{basis}. Same-strand chimeras are invisible, so the "
                "unflagged reads still hold about as many again. Treat read "
                "counts, occupancy and long-range results from this library "
                "with strong caution.")
    return (f"WARNING: high DAF strand-swap chimera rate: {percent:.1f}% of "
            f"reads{basis}; same-strand chimeras are invisible, so the unflagged "
            "reads still hold about as many again. Take read counts, occupancy "
            "and long-range results from this library with caution.")


def detector_details(stats):
    """One line on the strict detector settings of a statistics block ('' if absent)."""
    if not stats or stats.get('min_baseq') is None:
        return ''
    parts = [f"min_baseq={stats['min_baseq']}",
             f"min_span={stats.get('min_span')} bp",
             f"multi-switch {'on' if stats.get('multi_switch') else 'off'}"]
    mask = stats.get('error_mask') or {}
    if mask.get('applied'):
        parts.append(f"error mask {mask.get('n_sites', 0):,} sites")
    elif mask:
        parts.append(f"no error mask ({mask.get('reason')})")
    parts.append(f"{stats.get('n_duplicates_excluded', 0):,} duplicates not counted")
    return 'detector: ' + '; '.join(parts)


def chimera_summary_lines(stats, indent='  '):
    """Human-readable lines for a chimera statistics block (stderr, QC text)."""
    if not stats or not stats.get('detected') or stats.get('mode') in ('keep', 'n/a'):
        return []
    checked = stats.get('n_reads_checked') or 0
    pct = (lambda v: 'NA' if v is None else f'{100 * v:.2f}%')
    lines = [f"{indent}DAF strand-swap chimeras ({stats.get('mode')}): "
             f"{stats.get('n_chimeric', 0):,} / {checked:,} checked reads "
             f"({pct(stats.get('fraction'))}; detector min_seg={stats.get('min_seg')}, "
             f"purity={stats.get('purity')})"]
    details = detector_details(stats)
    if details:
        lines.append(f"{indent}  {details}")
    grade = stats.get('grade') or {}
    if grade.get('status'):
        severity = f" ({grade['severity']})" if grade.get('severity') else ''
        lines.append(f"{indent}  grade: {grade['status']}{severity} "
                     f"[WARN >= {100 * grade.get('warn_fraction', WARN_FRACTION):.0f}%, "
                     f"high >= {100 * grade.get('high_fraction', HIGH_FRACTION):.0f}%]")
        if grade.get('basis_note'):
            lines.append(f"{indent}  basis: {grade['basis_note']}")
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
    if len(policy) == 3:  # strict: (config, reference, detector settings)
        config, reference, cfg = policy
        if streaming_skip_reason(read, config) is not None:
            return False
        with ExitStack() as stack:
            ref = stack.enter_context(pysam.FastaFile(reference)) if reference else None
            return _is_chimeric_strict(read, ref, cfg)
    config, reference, min_seg, purity = policy
    if streaming_skip_reason(read, config) is not None:
        return False
    with ExitStack() as stack:
        ref = stack.enter_context(pysam.FastaFile(reference)) if reference else None
        return _is_chimeric(read, ref, min_seg=min_seg, purity=purity)


DUPLICATES_POLICY = ('duplicate-flagged (0x400) reads are tested and marked like '
                     'any read but left out of n_reads_checked, n_chimeric, the '
                     'fraction, the regions and the grade')


def error_mask_summary(enabled=STRICT_ERROR_MASK, bed=None, n_sites=0,
                       reason=None):
    """The ``error_mask`` provenance block of the chimera statistics.

    ``enabled``: requested (``--no-chimera-error-mask`` turns it off);
    ``applied``: a mask was computed and used; ``reason`` says why not.
    ``sha256``: the applied BED's content digest, so QC restores only this
    call's mask (:func:`fiberhmm.qc.core._call_chimera_detector`).
    """
    applied = bool(enabled and bed)
    if not enabled:
        reason = 'disabled (--no-chimera-error-mask)'
    elif not applied and reason is None:
        reason = 'no DAF SNP screen in this run'
    digest = None
    if applied and Path(bed).is_file():
        from fiberhmm.io.run_state import sha256_file
        digest = sha256_file(bed)
    return {'enabled': bool(enabled), 'applied': applied,
            'n_sites': int(n_sites) if applied else 0,
            'bed': str(bed) if applied else None,
            'reason': None if applied else reason,
            'sha256': digest}


def _new_stats_fields(mode, n_checked=0, n_chimeric=None, cfg=None,
                      regions=None, split=None, error_mask=None,
                      n_duplicates=0):
    """Strict-detector fields, appended after the standard ones (key order is kept)."""
    detected = mode not in ('keep', 'n/a')
    cfg = cfg or {}
    return dict(
        detector='strict',
        grade=(grade_chimeras(n_checked, n_chimeric, cfg.get('min_seg'),
                              cfg.get('purity'), cfg.get('min_baseq'),
                              cfg.get('min_span')) if detected else None),
        regions=regions if detected else None,
        split=split,
        note=CHIMERA_NOTE,
        min_baseq=cfg.get('min_baseq'),
        min_span=cfg.get('min_span'),
        span_unit='query bp, first to last dominant event of a segment',
        multi_switch=cfg.get('multi_switch'),
        error_mask=error_mask,
        n_duplicates_excluded=int(n_duplicates),
        duplicates=DUPLICATES_POLICY,
    )


def empty_chimera_stats(mode, min_seg=5, purity=0.8):
    return dict(detected=False, mode=mode, n_reads_checked=0, n_chimeric=0,
                fraction=None, min_seg=min_seg, purity=purity,
                n_records_affected=0, n_stale_marks_cleared=0,
                full_run=True, count_unit='unique read names before removal')


def empty_strict_chimera_stats(mode, min_seg=STRICT_MIN_SEG, purity=STRICT_PURITY,
                        min_baseq=STRICT_MIN_BASEQ, min_span=STRICT_MIN_SPAN,
                               multi_switch=STRICT_MULTI_SWITCH, error_mask=None):
    """Statistics of a strict-detector run without detection (keep mode)."""
    cfg = detector_config(min_seg, purity, min_baseq, min_span, multi_switch)
    return dict(detected=False, mode=mode, n_reads_checked=0, n_chimeric=0,
                fraction=None, min_seg=min_seg, purity=purity,
                n_records_affected=0, n_stale_marks_cleared=0,
                full_run=True, count_unit='unique read names before removal',
                **_new_stats_fields(mode, cfg=cfg, error_mask=error_mask))


def _quantiles(values):
    import numpy as np
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {'n': int(array.size), 'mean': float(array.mean()),
            **{f'q{int(100 * q):02d}': float(np.quantile(array, q))
               for q in (0.1, 0.25, 0.5, 0.75, 0.9)}}


def _region_rows(bin_of, bin_spans, anonymous_bins, chimeric):
    """Per-region (adjacent occupied 10-kb start bins) chimera fractions."""
    counts = {}
    for name, key in bin_of.items():
        row = counts.setdefault(key, [0, 0])
        row[0] += 1
        row[1] += name in chimeric
    for key, (checked, hits) in anonymous_bins.items():
        row = counts.setdefault(key, [0, 0])
        row[0] += checked
        row[1] += hits
    by_chrom = {}
    for (chrom, index) in counts:
        by_chrom.setdefault(chrom, []).append(index)
    regions = []
    for chrom, indices in by_chrom.items():
        group = []
        for index in sorted(indices) + [None]:
            if group and (index is None or index > group[-1] + 1):
                checked = sum(counts[(chrom, i)][0] for i in group)
                hits = sum(counts[(chrom, i)][1] for i in group)
                start = min(bin_spans[(chrom, i)][0] for i in group)
                end = max(bin_spans[(chrom, i)][1] for i in group)
                regions.append(dict(
                    chrom=chrom, start=int(start), end=int(end),
                    region='unmapped' if chrom == '*' else f'{chrom}:{start + 1}-{end}',
                    n_reads_checked=checked, n_chimeric=hits,
                    fraction=hits / checked if checked else None))
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
                     min_seg=None, purity=None, detector=DEFAULT_DETECTOR,
                     min_baseq=None, min_span=None, multi_switch=None,
                     error_mask=None, error_mask_info=None, **options):
    """Decide QNAMEs in a read-only pass with the chosen ``detector``.

    ``standard`` (default): the 3.2.1 detector (:func:`_prepare_chimeras_standard`,
    ``min_seg``/``purity`` default 5/0.8); ``min_baseq``, ``min_span``,
    ``multi_switch`` and ``error_mask`` belong to the strict detector and are
    refused. Mode ``split`` works with both (breakpoints of standard-flagged
    records: :func:`standard_breakpoints`). ``strict``: :func:`_prepare_chimeras_strict`
    (defaults: the ``STRICT_*`` constants). Other keywords go to both.
    """
    if detector not in DETECTORS:
        raise ValueError(f'Invalid chimera detector: {detector}')
    if detector == 'strict':
        return _prepare_chimeras_strict(
            input_bam, output_bam, mode=mode, daf=daf,
            min_seg=STRICT_MIN_SEG if min_seg is None else min_seg,
            purity=STRICT_PURITY if purity is None else purity,
            min_baseq=STRICT_MIN_BASEQ if min_baseq is None else min_baseq,
            min_span=STRICT_MIN_SPAN if min_span is None else min_span,
            multi_switch=STRICT_MULTI_SWITCH if multi_switch is None else multi_switch,
            error_mask=error_mask, error_mask_info=error_mask_info, **options)
    strict_only = dict(min_baseq=min_baseq, min_span=min_span,
                       multi_switch=multi_switch, error_mask=error_mask,
                       error_mask_info=error_mask_info)
    given = sorted(key for key, value in strict_only.items() if value is not None)
    if given:
        raise ValueError(f'{", ".join(given)}: strict chimera detector only')
    return _prepare_chimeras_standard(
        input_bam, output_bam, mode=mode, daf=daf,
        min_seg=STANDARD_MIN_SEG if min_seg is None else min_seg,
        purity=STANDARD_PURITY if purity is None else purity, **options)


def _prepare_chimeras_strict(input_bam, output_bam=None, *, mode='flag', daf=True,
                     min_seg=STRICT_MIN_SEG, purity=STRICT_PURITY,
                     min_baseq=STRICT_MIN_BASEQ, min_span=STRICT_MIN_SPAN,
                     multi_switch=STRICT_MULTI_SWITCH, error_mask=None,
                     error_mask_info=None,
                     prob_threshold=128,
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

    The strict detector. One decision drives flag, remove and split and
    gives the count and the grade: ``min_seg``/``purity`` per segment, events with base quality >=
    ``min_baseq`` (when QUAL is present), segments whose dominant events span
    >= ``min_span`` query bp, reads with several switches (``multi_switch``)
    and events on the chimera-only ``error_mask`` BED dropped (defaults: the
    ``DEFAULT_*`` constants; no mask). ``error_mask_info`` is recorded as the
    statistics' ``error_mask`` block (:func:`error_mask_summary`).

    Duplicate-flagged (0x400) reads are tested and marked like any read but
    are not counted (:data:`DUPLICATES_POLICY`); ``n_duplicates_excluded``
    reports how many read names were left out.

    Mode ``split`` requires ``spool_bam``: each eligible named record that
    is itself chimeric is written there as its single-strand pieces, two for
    one switch and more for several (:mod:`fiberhmm.inference.chimera_split`);
    pieces shorter than ``min_read_length`` are dropped, and a record with no
    piece left is kept whole, as is an unmapped record (no CIGAR or
    reference to cut; ``n_records_unmapped_kept_whole``) and a paired record
    (flag 0x1: its mate cannot be updated to match renamed pieces;
    ``n_records_paired_kept_whole``). Every other record
    of a chimeric name is
    flagged as in flag mode. A coordinate-sorted input gives a sorted,
    indexed spool.
    """
    from . import engine
    from .read_filters import ReadFilterConfig, streaming_skip_reason

    if mode not in ('flag', 'remove', 'keep', 'split'):
        raise ValueError(f'Invalid chimera mode: {mode}')
    if min_seg < 1 or not 0.5 < purity <= 1:
        raise ValueError('chimera min_seg must be >= 1 and purity in (0.5, 1]')
    if min_baseq < 0 or min_span < 0:
        raise ValueError('chimera min_baseq and min_span must be >= 0')
    splitting = bool(daf and mode == 'split')
    if splitting and spool_bam is None:
        raise ValueError('chimera split mode requires spool_bam')
    cfg = detector_config(min_seg, purity, min_baseq, min_span, multi_switch,
                          error_mask)
    engine.configure_daf_chimera_filter(daf and mode != 'keep', detector='strict', **cfg)
    engine.configure_daf_snp_mask(snp_mask)
    if error_mask_info is None:
        error_mask_info = error_mask_summary(
            True, error_mask,
            sum(len(v) for v in load_error_mask(error_mask).values()))
    config = ReadFilterConfig(min_mapq=min_mapq, min_read_length=min_read_length,
                              primary_only=alignments,
                              process_unmapped=process_unmapped, mode='daf',
                              has_reference=reference is not None)
    chimeric = ChimeraDecisions()
    if daf and mode != 'keep':
        chimeric.anonymous_policy = (config, reference, cfg)
    anonymous_checked = anonymous_chimeric = 0
    anonymous_duplicates = 0
    duplicate_of = {}
    counts = Counter()
    stale = 0
    # Regions: one start bin per read name (its primary record if eligible).
    bin_of, bin_spans, anonymous_bins = {}, {}, {}
    split_info = _new_split_info()
    gaps, piece_lengths, still_chimeric = [], [], set()

    def decide(read):
        """The detector's pieces (split mode) or True, else None."""
        events = _chimera_events(read, ref, cfg)
        if events is None:
            return None
        hit = decide_chimera(events[0], events[1], cfg, segments=splitting)
        return hit or None

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
                            duplicate_of[name] = bool(read.is_duplicate)
                        if splitting or name not in chimeric:
                            hit = decide(read)
                            if hit:
                                chimeric.add(name)
                            if splitting and hit:
                                pieces = _split_pieces(
                                    read, bam.header, hit, min_read_length, ref,
                                    decide, split_info, gaps, piece_lengths,
                                    still_chimeric)
                    else:
                        hit = decide(read)
                        if read.is_duplicate:
                            anonymous_duplicates += 1
                        else:
                            anonymous_checked += 1
                            anonymous_chimeric += int(bool(hit))
                            row = anonymous_bins.setdefault(key, [0, 0])
                            row[0] += 1
                            row[1] += int(bool(hit))
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
    duplicates = {name for name, dup in duplicate_of.items() if dup}
    counted_bins = ({name: key for name, key in bin_of.items()
                     if name not in duplicates} if duplicates else bin_of)
    n_checked = len(counted_bins) + anonymous_checked
    n_chimeric = len(chimeric - duplicates) + anonymous_chimeric
    split = None
    if splitting:
        split = _split_summary(split_info, min_read_length, gaps, piece_lengths,
                               'strict')
    stats = dict(detected=bool(daf and mode != 'keep'), mode=mode if daf else 'n/a',
                 n_reads_checked=n_checked, n_chimeric=n_chimeric,
                 fraction=n_chimeric / n_checked if n_checked else None,
                 min_seg=min_seg if daf else None, purity=purity if daf else None,
                 n_records_affected=sum(counts[name] for name in chimeric) + anonymous_chimeric,
                 n_stale_marks_cleared=stale, full_run=True,
                 count_unit='unique read names before removal; unnamed records counted individually')
    stats.update(_new_stats_fields(
        stats['mode'], n_checked, n_chimeric, cfg if daf else None,
        _region_rows(counted_bins, bin_spans, anonymous_bins, chimeric)
        if daf and mode != 'keep' else None, split,
        error_mask_info if daf and mode != 'keep' else None,
        len(duplicates) + anonymous_duplicates))
    return stats


def _has_iupac(read):
    """Whether ``read`` carries R/Y-encoded deaminations in SEQ."""
    from .engine import has_iupac_encoding
    seq = read.query_sequence
    return bool(seq) and has_iupac_encoding(seq)


def _reencode_piece(piece, ref):
    """``piece`` of an R/Y-encoded chimera, re-encoded for its own strand.

    The parent was encoded for one strand (``st``), so a piece from its
    other-strand segment keeps only the parent strand's minority Y (or R)
    marks, while its dominant deaminations are raw mismatches. Every piece
    of an R/Y record (also one left with no mark) is decoded on its aligned
    bases (Y -> T, R -> A: the raw read; marks on inserted or soft-clipped
    bases stay as cut) and encoded again as ``fiberhmm-daf-encode``
    encodes a read: the strand with more events
    (SNP-masked sites do not vote, as in calling an MD-only read) gets its
    events written as Y (C->T) or R (G->A), and ``st`` is set. A tie or no
    event leaves the decoded raw piece (calling then reads its MD, as for a
    piece of the equivalent MD-only read). Without MD or a reference to
    compare with, the piece is returned as cut.
    """
    from . import engine
    from fiberhmm.daf.encoder import (
        _daf_positions_result, encode_read_daf, get_daf_positions,
    )

    seq = piece.query_sequence
    if not seq:
        return piece
    quals = piece.query_qualities
    # Only aligned bases are decoded: a mark on an inserted or soft-clipped
    # base has no reference to re-derive it from, so it stays as cut.
    from .chimera_split import _aligned_query_positions
    decoded = list(seq)
    for position in _aligned_query_positions(piece):
        decoded[position] = decoded[position].translate(_DECODE_IUPAC)
    raw = copy.copy(piece)
    raw.query_sequence = ''.join(decoded)
    raw.query_qualities = quals
    events = get_daf_positions(raw, force_strand='CT', ref_fasta=ref,
                               excluded_reference_positions=engine._daf_reference_mask(raw))
    if events is None:
        return piece                  # no reference evidence: keep the marks
    chosen = _daf_positions_result(events[0], events[1], None)
    if chosen is None:
        return raw                    # tie or no event: an MD-only piece
    encoded = encode_read_daf(raw, force_strand=chosen[2], ref_fasta=ref)
    if encoded is None:
        return raw
    raw.query_sequence = encoded[0]
    raw.query_qualities = quals
    raw.set_tag('st', encoded[1], value_type='Z')
    return raw


_DECODE_IUPAC = str.maketrans({'Y': 'T', 'y': 't', 'R': 'A', 'r': 'a'})


def _split_pieces(read, header, split, min_read_length, ref, decide, info,
                  gaps, piece_lengths, still_chimeric):
    """Pieces to write in place of ``read`` (None: keep the record whole)."""
    from .chimera_split import is_splittable, split_record

    if not is_splittable(read):
        # Unmapped (no CIGAR or reference): kept whole and flagged.
        info['n_records_kept_whole'] += 1
        info['n_records_unmapped_kept_whole'] += 1
        return None
    if read.is_paired:
        # Paired (0x1): pieces would need new names and mate fields that the
        # mate record cannot be repaired to match in this streaming pass, so
        # the record is kept whole and flagged, with its pairing untouched.
        info['n_records_kept_whole'] += 1
        info['n_records_paired_kept_whole'] += 1
        return None
    pieces, record_gaps = split_record(read, header, split)
    if _has_iupac(read):
        pieces = [None if piece is None else _reencode_piece(piece, ref)
                  for piece in pieces]
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
    info['n_records_several_switches'] += int(len(pieces) > 2)
    info.setdefault('_names', set()).add(read.query_name)
    info['n_pieces_written'] += len(kept)
    for gap in record_gaps:
        if gap is not None:
            gaps.append(gap[1] - gap[0])
    for piece in kept:
        length = piece.query_alignment_length or 0
        info['recovered_bases'] += length
        piece_lengths.append(length)
        if read.has_tag('MM') or read.has_tag('Mm'):
            kept_mods = piece.has_tag('MM') or piece.has_tag('Mm')
            info['n_pieces_modifications_kept' if kept_mods
                 else 'n_pieces_modifications_dropped'] += 1
        if decide(piece):
            info['n_pieces_still_chimeric'] += 1
            still_chimeric.add(piece.query_name)
    return kept


def _prepare_chimeras_standard(input_bam, output_bam=None, *, mode='flag', daf=True,
                     min_seg=5, purity=0.8, prob_threshold=128,
                     reference=None, snp_mask=None, min_mapq=0,
                     min_read_length=0, process_unmapped=False,
                     io_threads=1, chimeric_names=None, spool_bam=None,
                     alignments='primary-supplementary'):
    """The standard (3.2.1) detector: decide QNAMEs in a read-only pass;
    spool only non-seekable inputs.

    The CLI passes a mutable ``ChimeraDecisions`` set and applies marks/removal
    in the caller, so file input needs no intermediate BAM. A plain set gets
    only named decisions; ChimeraDecisions also carries the unnamed replay
    policy. For streams,
    ``spool_bam`` supplies the replay path. ``output_bam`` optionally writes
    a marked/filtered BAM for standalone use. Counts precede removal.

    Mode ``split`` requires ``spool_bam``. The standard detector decides
    which read names are chimeric, as in flag mode (same names, same
    counts). Each eligible named record that the standard test flags itself
    is written to the spool as its single-strand pieces, cut at the
    breakpoints of :func:`standard_breakpoints`; everything else about the
    pieces (names, tags, dropped short pieces, unmapped and paired records
    kept whole and flagged, a sorted indexed spool for sorted input) is as
    in the strict detector's split mode (:func:`_prepare_chimeras_strict`).
    Records the detector does not flag are written unchanged; other records
    of a chimeric name are flagged as in flag mode. The statistics gain a
    ``split`` block (``flagged_by`` 'standard', ``breakpoints``); flag,
    remove and keep statistics are unchanged.
    """
    from . import engine
    from .read_filters import ReadFilterConfig, streaming_skip_reason

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
    checked, chimeric = set(), ChimeraDecisions()
    if daf and mode != 'keep':
        chimeric.anonymous_policy = (config, reference, min_seg, purity)
    anonymous_checked = anonymous_chimeric = 0
    counts = Counter()
    stale = 0
    split_info = _new_split_info()
    gaps, piece_lengths, still_chimeric = [], [], set()
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
                    if named:
                        checked.add(read.query_name)
                        if splitting:
                            # Every record is decided: each flagged one is cut.
                            if _is_chimeric(read, ref):
                                chimeric.add(read.query_name)
                                pieces = _standard_split_pieces(
                                    read, bam.header, ref, min_seg, purity,
                                    min_read_length, split_info, gaps,
                                    piece_lengths, still_chimeric)
                        elif read.query_name not in chimeric and _is_chimeric(read, ref):
                            chimeric.add(read.query_name)
                    else:
                        anonymous_checked += 1
                        anonymous_chimeric += int(_is_chimeric(read, ref))
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
            # Pieces that still show a swap (several switches) are flagged
            # like any chimera; they are not counted as new reads.
            chimeric_names.update(still_chimeric)
            if isinstance(chimeric_names, ChimeraDecisions):
                chimeric_names.anonymous_policy = chimeric.anonymous_policy
        if output_bam is not None:
            decisions = chimeric
            if still_chimeric:
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
    n_checked = len(checked) + anonymous_checked
    n_chimeric = len(chimeric) + anonymous_chimeric
    stats = dict(detected=bool(daf and mode != 'keep'), mode=mode if daf else 'n/a',
                 n_reads_checked=n_checked, n_chimeric=n_chimeric,
                 fraction=n_chimeric / n_checked if n_checked else None,
                 min_seg=min_seg if daf else None, purity=purity if daf else None,
                 n_records_affected=sum(counts[name] for name in chimeric) + anonymous_chimeric,
                 n_stale_marks_cleared=stale, full_run=True,
                 count_unit='unique read names before removal; unnamed records counted individually')
    if splitting:
        stats['split'] = _split_summary(split_info, min_read_length, gaps,
                                        piece_lengths, 'standard')
    return stats


def _standard_split_pieces(read, header, ref, min_seg, purity, min_read_length,
                           info, gaps, piece_lengths, still_chimeric):
    """Pieces of a record the standard detector flagged (None: keep it whole)."""
    segments = standard_breakpoints(read, ref, min_seg, purity)
    if segments is None:
        # Not reached for a flagged record (the first cut is the standard
        # test's own scan); kept whole and flagged if it ever is.
        info['n_records_kept_whole'] += 1
        return None
    return _split_pieces(read, header, segments, min_read_length, ref,
                         lambda piece: _is_chimeric(piece, ref), info, gaps,
                         piece_lengths, still_chimeric)
