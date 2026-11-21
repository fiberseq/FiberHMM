"""Shared read skip/filter policy for inference pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AbstractSet, Mapping, Optional

# CIGAR operation code for a hard clip (BAM_CHARD_CLIP).
_HARD_CLIP = 5

# Skip reason for alignments whose MM/ML cannot be trusted: a hard-clipped
# supplementary/secondary record carries the MM/ML of the full molecule
# (minimap2 without -Y), but SEQ holds only the aligned piece, so the MM skip
# counts walk the wrong bases.
HARD_CLIPPED_MM = "hard_clipped_mm"

# Every skip-reason key a pipeline may count. Pipelines build their tallies
# from this so a new reason can never raise KeyError at ``+= 1``.
SKIP_REASONS = (
    "unmapped",
    "secondary_supplementary",
    "low_mapq",
    "too_short",
    "training_excluded",
    HARD_CLIPPED_MM,
    "no_modifications",
    "extraction_failed",
    "no_footprints",
    "chimera",
    # Region-parallel only: record on a contig excluded by --chroms or
    # --skip-scaffolds, copied through unannotated.
    "contig_not_selected",
    # Not a skip reason of its own: the subset of ``too_short`` records that
    # are mostly soft-clipped (counted alongside, for the run-summary hint).
    "too_short_soft_clipped",
)

# Tally keys that annotate another reason instead of counting skips.
NOTE_ONLY_SKIP_KEYS = frozenset({"too_short_soft_clipped"})

# Legacy footprint tags written by FiberHMM (and fibertools) call passes.
LEGACY_CALL_TAGS = ("ns", "nl", "as", "al", "nq", "aq")
# Per-read marker of fiberhmm-apply --harmonize-legacy-tags: this read's ns/nl
# hold only the MA-view nucleosomes, not every HMM footprint recall needs.
# Every writer that rewrites ns/nl removes it.
HARMONIZED_LEGACY_TAG = "hl"

# Above this fraction of records skipped as unmapped, a call run produced
# essentially nothing and is treated as a configuration error.
MOSTLY_UNMAPPED_FRACTION = 0.9


def new_skip_counts() -> dict:
    """Fresh zeroed tally covering every known skip reason."""
    return {reason: 0 for reason in SKIP_REASONS}


# Which alignment records are called. ``primary-supplementary`` (the 3.0
# default) calls primary and supplementary records: a supplementary record is
# another part of the same read (a split alignment: the far side of a
# structural variant, a TE copy elsewhere), not a copy of it. Secondary records
# are alternative placements of the same bases and stay uncalled. ``primary``
# is the earlier 3.0 behaviour (--primary), ``all`` calls every record
# (--no-primary).
from fiberhmm import defaults as _shared_defaults  # noqa: E402

ALIGNMENT_SETS = ("primary", "primary-supplementary", "all")
DEFAULT_ALIGNMENTS = _shared_defaults.ALIGNMENTS


def resolve_alignments(value) -> str:
    """Normalise a ``primary_only``/``--alignments`` value to an ALIGNMENT_SETS
    name: True -> ``primary``, False -> ``all``, None -> the default."""
    if value is None:
        return DEFAULT_ALIGNMENTS
    if value is True:
        return "primary"
    if value is False:
        return "all"
    if value not in ALIGNMENT_SETS:
        raise ValueError(f"alignments must be one of {ALIGNMENT_SETS}, not {value!r}")
    return value


def alignment_skipped(read, alignments) -> bool:
    """True when ``read``'s record type is outside the called alignment set."""
    alignments = resolve_alignments(alignments)
    if alignments == "all":
        return False
    if read.is_secondary:
        return True
    return bool(read.is_supplementary) and alignments == "primary"


def add_alignment_args(parser) -> None:
    """``--alignments`` with ``--primary``/``--no-primary`` as aliases."""
    import argparse
    parser.add_argument(
        '--alignments', choices=list(ALIGNMENT_SETS), default=DEFAULT_ALIGNMENTS,
        help='Alignment records to call (default primary-supplementary): '
             'primary and supplementary records, the parts of a split read '
             '(e.g. both sides of a structural variant); secondary records, '
             'alternative placements of the same bases, pass through uncalled. '
             'On a supplementary record only its aligned bases are called. '
             '"primary" calls primary records only; "all" also calls secondary '
             'records. Hard-clipped records whose MM/ML cannot match SEQ are '
             'always skipped (hard_clipped_mm).')
    parser.add_argument('--primary', dest='alignments', action='store_const',
                        const='primary',
                        help='Same as --alignments primary.')
    parser.add_argument('--no-primary', dest='alignments', action='store_const',
                        const='all', help='Same as --alignments all.')
    del argparse


@dataclass(frozen=True)
class ReadFilterConfig:
    """Filtering options shared by streaming inference paths."""

    min_mapq: int = 0
    min_read_length: int = 0
    # Alignment set (see ALIGNMENT_SETS); True/False are the older
    # primary-only / all spellings.
    primary_only: object = False
    process_unmapped: bool = False
    train_rids: AbstractSet[str] = field(default_factory=frozenset)
    # Observation mode and reference availability decide whether a read's
    # calls would come from MM/ML (and therefore need the hard-clip guard).
    mode: Optional[str] = None
    has_reference: bool = False
    honor_chimera_marks: bool = True


def _local_mm_applicable(read) -> bool:
    """Whether ``read``'s MM/ML can be walked against its stored SEQ.

    Same contract as ``fiberhmm.core.bam_reader.mm_applicable`` (added by the
    MM-parser fixes on the fh-core-daf branch, commit 5430211): MN, when
    present, must equal len(SEQ); otherwise a hard clip (CIGAR op 5) makes
    MM/ML inapplicable. Records without MM are trivially applicable.
    TODO(integration): drop this copy once bam_reader.mm_applicable is on
    the base branch; the import below already prefers it.
    """
    has_tag = getattr(read, "has_tag", None)
    if has_tag is None or not (has_tag("MM") or has_tag("Mm")):
        return True
    seq = getattr(read, "query_sequence", None)
    if has_tag("MN"):
        try:
            return seq is not None and int(read.get_tag("MN")) == len(seq)
        except (KeyError, TypeError, ValueError):
            return False
    cigar = getattr(read, "cigartuples", None)
    if cigar:
        for op, _length in cigar:
            if op == _HARD_CLIP:
                return False
    return True


try:  # pragma: no cover - depends on which branch this is merged onto
    from fiberhmm.core.bam_reader import mm_applicable
except ImportError:  # pragma: no cover
    mm_applicable = _local_mm_applicable


def hard_clipped_mm_unreliable(read, mode: Optional[str] = None,
                               has_reference: bool = False) -> bool:
    """True when a read's calls would come from MM/ML that SEQ cannot support.

    MM skip counts index the bases of the complete original read. A record
    with hard clips stores only part of SEQ, so its MM/ML (copied from the
    primary by aligners such as minimap2 without ``-Y``) no longer lines up.
    The SAM ``MN`` tag records the SEQ length the MM/ML was written for and,
    when present, decides (see ``mm_applicable``).

    DAF reads whose deaminations come from R/Y encoding, the MD tag or a
    reference FASTA do not use MM/ML and are never guarded.
    """
    if not getattr(read, "query_sequence", None):
        return False
    if mm_applicable(read):
        return False
    if mode == "daf":
        seq = read.query_sequence
        if has_reference or read.has_tag("MD") or ("R" in seq or "Y" in seq):
            return False
    return True


_SOFT_CLIP = 4


def soft_clipped_bases(read) -> int:
    """Bases in the record's soft clips (CIGAR S); 0 when the record has no CIGAR."""
    try:
        cigar = getattr(read, 'cigartuples', None) or ()
    except (ValueError, TypeError):
        return 0
    return sum(n for op, n in cigar if op == _SOFT_CLIP)


def _seq_length(read) -> Optional[int]:
    """Length of the stored SEQ, from whatever the record offers."""
    for get in (lambda: getattr(read, 'query_length', None),
                lambda: len(read.query_sequence) if getattr(read, 'query_sequence', None) else None,
                lambda: read.infer_query_length() if callable(getattr(read, 'infer_query_length', None)) else None):
        try:
            value = get()
        except (AttributeError, TypeError, ValueError):
            value = None
        if value:
            return int(value)
    return None


def _aligned_length(read) -> Optional[int]:
    """Aligned query length, else the reference span (records without a CIGAR)."""
    value = getattr(read, 'query_alignment_length', None)
    if value is not None:
        return int(value)
    start, end = getattr(read, 'reference_start', None), getattr(read, 'reference_end', None)
    if start is not None and end is not None:
        return int(end) - int(start)
    return None


def modeled_read_length(read, mode: Optional[str] = None) -> Optional[int]:
    """The read length ``--min-read-length`` is compared with: what the HMM models.

    The HMM runs over the whole stored SEQ. For Hia5 (and other m6A) data the
    soft-clipped bases carry MM/ML marks like any other, so the length is the
    SEQ (query) length, soft clips included: a 20 kb molecule with a 300 bp
    alignment is still a 20 kb read to call. DAF marks come from comparing the
    read with the reference, so with the default unaligned-base mask
    (``--daf-mask-unaligned``) soft-clipped bases carry no evidence and the
    length is the aligned query length (``query_alignment_length``). Unmapped
    records use their SEQ length. Hard-clipped bases are not in SEQ and never
    count (such records are also screened by the hard-clip MM guard).
    Records that lack an attribute (lightweight stand-ins) fall back to the
    SEQ, then the aligned length, then the reference span.
    """
    if getattr(read, 'is_unmapped', False):
        return _seq_length(read) or 0
    if mode == 'daf':
        from fiberhmm.inference.engine import daf_unaligned_mask_enabled
        if daf_unaligned_mask_enabled():
            aligned = _aligned_length(read)
            return aligned if aligned is not None else _seq_length(read)
    seq = _seq_length(read)
    return seq if seq else _aligned_length(read)


def is_mostly_soft_clipped(read) -> bool:
    """Soft clips longer than the aligned part: the reads the too-short hint is about."""
    clipped = soft_clipped_bases(read)
    aligned = _aligned_length(read) or 0
    return clipped > 0 and clipped >= aligned


def count_skip(skip_reasons: dict, reason: str, read=None) -> None:
    """Tally ``reason``; a too-short read that is mostly soft-clipped is also
    counted under ``too_short_soft_clipped`` (for :func:`too_short_hint`)."""
    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    if reason == "too_short" and read is not None and not getattr(read, 'is_unmapped', False) \
            and is_mostly_soft_clipped(read):
        skip_reasons["too_short_soft_clipped"] = skip_reasons.get("too_short_soft_clipped", 0) + 1


def too_short_hint(skip_reasons: Mapping, total: int, mode: Optional[str] = None,
                   min_read_length: Optional[int] = None) -> Optional[str]:
    """One line for the run summary when many too-short skips are mostly soft-clipped reads."""
    n = int(skip_reasons.get("too_short_soft_clipped", 0) or 0)
    if not total or n < 10 or n < 0.05 * total:
        return None
    if mode == 'daf':
        why = ('their soft-clipped bases carry no DAF evidence (--daf-mask-unaligned), so their length is the '
               'aligned part')
    else:
        why = 'their SEQ is short even with the soft clips included'
    limit = f' (--min-read-length {min_read_length})' if min_read_length is not None else ''
    return (f'  Note: {n:,} reads ({100 * n / total:.1f}%) were skipped as too short{limit} although most of each '
            f'is soft-clipped: {why}. Check the alignment (e.g. adapters or a missing reference contig), or lower '
            '--min-read-length to call them.')


def printable_skip_reasons(skip_reasons: Mapping):
    """``(reason, count)`` pairs to list in a run summary (annotations excluded)."""
    return [(r, c) for r, c in skip_reasons.items() if r not in NOTE_ONLY_SKIP_KEYS]


def streaming_skip_reason(read, config: ReadFilterConfig) -> Optional[str]:
    """Return the skip reason for a streaming read, or None if processable."""
    if config.honor_chimera_marks:
        from .chimeras import is_marked_chimera
        if is_marked_chimera(read):
            return "chimera"

    if read.is_unmapped:
        if not config.process_unmapped or read.query_sequence is None:
            return "unmapped"

    if alignment_skipped(read, config.primary_only):
        return "secondary_supplementary"

    if not read.is_unmapped and read.mapping_quality < config.min_mapq:
        return "low_mapq"

    read_len = modeled_read_length(read, config.mode)
    if read_len is None or read_len < config.min_read_length:
        return "too_short"

    if config.train_rids and read.query_name in config.train_rids:
        return "training_excluded"

    if hard_clipped_mm_unreliable(read, config.mode, config.has_reference):
        return HARD_CLIPPED_MM

    return None


def _strip_tag(read, tag: str) -> None:
    if read.has_tag(tag):
        try:
            read.set_tag(tag, None)
        except Exception:
            pass


_AQ_AMBIGUITY_CACHE: dict = {}


def header_aq_may_be_fiberhmm(header) -> bool:
    """True when an ``aq`` in a file with this header may be FiberHMM's MSP
    posterior rather than fibertools FIRE: the header has a ``fiberhmm-apply``
    @PG (3.2.1 and older wrote ``aq`` with ``--scores``) and no
    ``FIBERHMM-ANNOTATION-CONTRACT`` line."""
    key = id(header)
    hit = _AQ_AMBIGUITY_CACHE.get(key)
    if hit is not None and hit[0] is header:
        return hit[1]
    from fiberhmm.io.bam_header import header_annotation_contract
    data = header.to_dict() if hasattr(header, 'to_dict') else dict(header)
    ambiguous = header_annotation_contract(data) is None and any(
        'fiberhmm-apply' in (str(pg.get('PN', '')), str(pg.get('ID', '')).split('.')[0])
        for pg in data.get('PG', []) or [])
    if len(_AQ_AMBIGUITY_CACHE) > 64:
        _AQ_AMBIGUITY_CACHE.clear()
    _AQ_AMBIGUITY_CACHE[key] = (header, ambiguous)
    return ambiguous


def strip_ambiguous_aq(read) -> None:
    """Remove an ``aq`` that may be an older FiberHMM's MSP posterior (see
    :func:`header_aq_may_be_fiberhmm`), so a record a v2-contract writer passes
    through unchanged carries no MSP score that is not fibertools FIRE."""
    if not hasattr(read, 'has_tag') or not read.has_tag('aq'):
        return
    header = getattr(read, 'header', None)
    if header is not None and header_aq_may_be_fiberhmm(header):
        _strip_tag(read, 'aq')


def _adopt_fibertools_ma(read) -> None:
    """Take a fibertools ``Ma``/``Aq``/``An`` as the read's ``MA``/``AQ``/``AN``."""
    read.set_tag("MA", str(read.get_tag("Ma")), value_type="Z")
    if read.has_tag("Aq"):
        read.set_tag("AQ", read.get_tag("Aq"))
    else:
        _strip_tag(read, "AQ")
    if read.has_tag("An"):
        read.set_tag("AN", str(read.get_tag("An")), value_type="Z")
    else:
        _strip_tag(read, "AN")


def strip_stale_call_tags(read, *, legacy_tags=LEGACY_CALL_TAGS,
                          keep_m5c_groups: bool = True) -> None:
    """Remove annotations a previous call pass left on ``read``.

    A re-call must not leave the old run's calls on reads it skips (or
    annotates with fewer groups): the new header declares every footprint tag
    as belonging to this run. ``legacy_tags`` lists the legacy tags this run
    rewrites (default ``ns/nl/as/al/nq/aq``; empty when it leaves them). MA is
    reduced exactly as the MA writer of the matching pipeline would reduce it:
    the fused call writer keeps DddA m5C groups (``keep_m5c_groups=True``),
    the legacy apply writer drops MA/AN/AQ wholesale (``False``).
    fibertools' (>= 0.13) ``Ma/Aq/An`` never survive: ``ft fire`` moves every
    group of an input ``MA`` into ``Ma`` (FiberHMM's nuc/msp/tf and any DddA
    m5C groups), so a read without ``MA`` first has its ``Ma`` taken as its
    ``MA`` (``Ma`` is molecular, the frame callers declare) and is then reduced
    like any ``MA``; FIRE built on the replaced MSPs is dropped with them.
    Groups derived from the old calls (fiberhmm-scan's
    ``nucsz``/``nucseg``/``nucmol``/``nucreg``, ``fpcls``, ``nfrvar``, ``nfrmol``) are
    always dropped: only m5C groups survive.
    """
    if not hasattr(read, "has_tag"):
        return
    for tag in legacy_tags or ():
        _strip_tag(read, tag)
    if keep_m5c_groups and not read.has_tag("MA") and read.has_tag("Ma"):
        _adopt_fibertools_ma(read)
    for tag in ("Ma", "Aq", "An"):
        _strip_tag(read, tag)
    if "ns" in (legacy_tags or ()):
        _strip_tag(read, HARMONIZED_LEGACY_TAG)
    if not read.has_tag("MA"):
        for tag in ("AQ", "AN"):
            _strip_tag(read, tag)
        return
    if not keep_m5c_groups:
        for tag in ("MA", "AQ", "AN"):
            _strip_tag(read, tag)
        return

    from fiberhmm.daf.m5c import ma_group_feature
    from fiberhmm.io.ma_tags import (
        DDDA_MCG_FEATURE,
        DDDA_MCG_HEMI_FEATURE,
        DDDA_UCG_FEATURE,
        parse_an_tag,
    )

    m5c_features = {DDDA_MCG_FEATURE, DDDA_MCG_HEMI_FEATURE, DDDA_UCG_FEATURE}
    tokens = str(read.get_tag("MA")).split(";")
    names = parse_an_tag(str(read.get_tag("AN"))) if read.has_tag("AN") else []
    kept_groups = []
    kept_names = []
    offset = 0
    for group in tokens[1:]:
        count = sum(bool(item) for item in group.partition(":")[2].split(","))
        group_names = names[offset:offset + count]
        group_names.extend([""] * (count - len(group_names)))
        offset += count
        if ma_group_feature(group) in m5c_features:
            kept_groups.append(group)
            kept_names.extend(group_names)
    # m5C groups carry no AQ bytes, so AQ always belongs to removed groups.
    _strip_tag(read, "AQ")
    if not kept_groups:
        _strip_tag(read, "MA")
        _strip_tag(read, "AN")
        return
    read.set_tag("MA", ";".join([tokens[0], *kept_groups]), value_type="Z")
    if read.has_tag("AN"):
        if any(kept_names):
            read.set_tag("AN", ",".join(name or "." for name in kept_names),
                         value_type="Z")
        else:
            _strip_tag(read, "AN")


def mostly_unmapped_message(skip_reasons: Mapping[str, int],
                            total_records: int) -> Optional[str]:
    """Explain a run that skipped nearly every record as unmapped, else None."""
    unmapped = int(skip_reasons.get("unmapped", 0))
    total_records = int(total_records)
    if total_records <= 0 or unmapped <= MOSTLY_UNMAPPED_FRACTION * total_records:
        return None
    return (
        f"{unmapped:,} of {total_records:,} records "
        f"({100.0 * unmapped / total_records:.1f}%) were skipped as unmapped, "
        "so this run produced essentially no calls. For unaligned (uBAM) or "
        "streamed input pass --process-unmapped (enabled automatically for "
        "stdin, unindexed and unaligned input); pass --no-process-unmapped to "
        "keep unmapped reads as untouched pass-through deliberately."
    )


class MostlyUnmappedError(RuntimeError):
    """A call run skipped nearly all records as unmapped."""
