"""The array-pair rule for legacy footprint tags, and which MA groups are footprint calls.

Legacy footprint calls come as start/length tag pairs: ``ns``/``nl`` (nucleosome
starts/lengths, with optional ``nq`` scores) and ``as``/``al`` (MSP starts/lengths,
with optional ``aq``). A pair counts as calls only when BOTH tags are present, both
are integer arrays (pysam returns ``array.array`` for a B-array) and they have the
same length. Anything else -- a missing partner, a scalar (Dorado's ``ns:i`` is a
signal sample count), a string (``ns:Z:100,700``), a float array or unequal lengths
-- means the pair is absent: no calls, and no evidence that the read was called.
This is FiberBrowser's rule (``bam_tags.interval_tag_pair``).
"""
from __future__ import annotations

import array
from typing import List, Optional, Tuple

#: (start, length, score) tag names of the two legacy pairs.
LEGACY_PAIRS = (("ns", "nl", "nq"), ("as", "al", "aq"))

#: MA/Ma group types that are footprint calls. Other groups (``ddda_mcg.``,
#: ``fire.Q``...) annotate a read without calling its footprints.
FOOTPRINT_GROUP_TYPES = ("nuc", "msp", "tf")

_INT_TYPECODES = frozenset("bBhHiIlLqQ")


def int_array(value) -> Optional[List[int]]:
    """``value`` as a list of ints when it is an integer array, else None.

    Accepts an integer ``array.array`` (pysam's B-array) or a list/tuple of ints.
    Scalars, strings, bytes, float arrays and None give None."""
    if isinstance(value, array.array):
        return [int(v) for v in value] if value.typecode in _INT_TYPECODES else None
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, int) and not isinstance(v, bool) for v in value):
            return [int(v) for v in value]
        return None
    return None


def array_tag_pair(read, start_tag: str, length_tag: str) -> Optional[Tuple[List[int], List[int]]]:
    """``(starts, lengths)`` of a valid legacy pair on ``read``, else None (the pair is absent).

    A valid pair may be empty (``([], [])``): a call with nothing in it."""
    if not (read.has_tag(start_tag) and read.has_tag(length_tag)):
        return None
    starts = int_array(read.get_tag(start_tag))
    lengths = int_array(read.get_tag(length_tag))
    if starts is None or lengths is None or len(starts) != len(lengths):
        return None
    return starts, lengths


def has_array_tag_pair(read, start_tag: str, length_tag: str) -> bool:
    """True when ``read`` carries a valid (possibly empty) legacy pair."""
    return array_tag_pair(read, start_tag, length_tag) is not None


def has_legacy_pair(read) -> bool:
    """True when ``read`` carries a valid ns/nl or as/al pair (possibly empty)."""
    return any(has_array_tag_pair(read, s, l) for s, l, _q in LEGACY_PAIRS)


def has_legacy_calls(read) -> bool:
    """True when a valid ns/nl or as/al pair holds at least one interval: call evidence."""
    for start_tag, length_tag, _score in LEGACY_PAIRS:
        pair = array_tag_pair(read, start_tag, length_tag)
        if pair is not None and pair[0]:
            return True
    return False


def ma_has_footprint_groups(value) -> bool:
    """True when an MA/Ma tag value holds at least one nuc/msp/tf group.

    A tag carrying only other groups (an m5C layer such as ``ddda_mcg.``, or
    ``fire.Q``) does not call the read's footprints: its legacy ns/nl/as/al or
    the other MA/Ma tag do. An unparsable value gives False."""
    from fiberhmm.io.ma_tags import parse_ma_tag

    try:
        parsed = parse_ma_tag(str(value))
    except (ValueError, TypeError):
        return False
    return any(name in FOOTPRINT_GROUP_TYPES for name, *_rest in parsed["raw_types"])
