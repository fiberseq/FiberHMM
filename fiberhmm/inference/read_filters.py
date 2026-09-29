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
)

# Legacy footprint tags written by FiberHMM (and fibertools) call passes.
LEGACY_CALL_TAGS = ("ns", "nl", "as", "al", "nq", "aq")

# Above this fraction of records skipped as unmapped, a call run produced
# essentially nothing and is treated as a configuration error.
MOSTLY_UNMAPPED_FRACTION = 0.9


def new_skip_counts() -> dict:
    """Fresh zeroed tally covering every known skip reason."""
    return {reason: 0 for reason in SKIP_REASONS}


@dataclass(frozen=True)
class ReadFilterConfig:
    """Filtering options shared by streaming inference paths."""

    min_mapq: int = 0
    min_read_length: int = 0
    primary_only: bool = False
    process_unmapped: bool = False
    train_rids: AbstractSet[str] = field(default_factory=frozenset)
    # Observation mode and reference availability decide whether a read's
    # calls would come from MM/ML (and therefore need the hard-clip guard).
    mode: Optional[str] = None
    has_reference: bool = False


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


def streaming_skip_reason(read, config: ReadFilterConfig) -> Optional[str]:
    """Return the skip reason for a streaming read, or None if processable."""
    if read.is_unmapped:
        if not config.process_unmapped or read.query_sequence is None:
            return "unmapped"

    if config.primary_only and (read.is_secondary or read.is_supplementary):
        return "secondary_supplementary"

    if not read.is_unmapped and read.mapping_quality < config.min_mapq:
        return "low_mapq"

    if read.is_unmapped:
        read_len = read.query_length or 0
    else:
        read_len = read.query_alignment_length
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
    """
    if not hasattr(read, "has_tag"):
        return
    for tag in legacy_tags or ():
        _strip_tag(read, tag)
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
