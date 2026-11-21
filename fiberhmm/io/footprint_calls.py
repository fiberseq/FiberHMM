"""Where a record's nucleosome and MSP calls come from: MA first, legacy fallback.

The rule (3.3.0), shared by every FiberHMM reader of nucleosome/MSP calls:

* The ``nuc`` and ``msp`` groups of the molecular-annotation tag are
  authoritative: FiberHMM's ``MA`` (what ``fiberhmm-call``,
  ``fiberhmm-recall-tfs`` and ``fiberhmm-recall-nucs`` write), else
  fibertools' ``Ma`` (fibertools-rs >= 0.13, always molecular frame).
* Legacy ``ns/nl`` and ``as/al`` (with ``nq``) are read only when the record
  has no ``nuc`` or ``msp`` group in either tag, through the array-pair rule
  of :mod:`fiberhmm.io.legacy_tags`.

A tag holding only other groups (an m5C layer such as ``ddda_mcg.``,
``fire.Q``, a lone ``tf`` layer) does not call the read's nucleosomes or
MSPs, so it does not hide the legacy pairs. An MA ``nuc``/``msp`` group that
is present is authoritative even when the legacy tags disagree with it (a
recall that refreshed MA only, or legacy tags another tool left behind).

:func:`resolve_footprint_tags` returns the calls as legacy-shaped
``ns/nl/as/al`` (+ ``nq``) arrays in the frame the caller reads legacy tags
in, so code written for legacy tags needs no other change.
"""
from __future__ import annotations

import array
from typing import Optional, Tuple

#: MA/Ma groups that are nucleosome or MSP calls: their presence makes the
#: annotation tag the read's source of these calls.
NUC_MSP_GROUPS = ('nuc', 'msp')

#: Annotation tags in precedence order, with their quality tag.
ANNOTATION_SOURCES = (('MA', 'AQ'), ('Ma', 'Aq'))


def ma_has_nuc_msp_groups(value) -> bool:
    """True when an MA/Ma tag value holds a ``nuc`` or ``msp`` group (even an
    empty one). An unparsable value gives False."""
    from fiberhmm.io.ma_tags import parse_ma_tag

    try:
        parsed = parse_ma_tag(str(value))
    except (ValueError, TypeError):
        return False
    return any(name in NUC_MSP_GROUPS for name, *_rest in parsed['raw_types'])


def footprint_annotation_tag(read) -> Optional[str]:
    """``'MA'`` or ``'Ma'``: the annotation tag that holds the read's
    nucleosome/MSP calls (FiberHMM's MA before fibertools' Ma), or None when
    neither has a ``nuc``/``msp`` group (legacy tags, if any, then hold them)."""
    for tag, _quality in ANNOTATION_SOURCES:
        try:
            if read.has_tag(tag) and ma_has_nuc_msp_groups(read.get_tag(tag)):
                return tag
        except (AttributeError, KeyError, TypeError):
            continue
    return None


def annotation_nuc_msp(read, tag: Optional[str] = None) -> Optional[dict]:
    """Nucleosomes, MSPs and nucleosome ``nq`` bytes from an annotation tag.

    ``{'nuc': [(start, length), ...], 'msp': [...], 'nq': [...] or None,
    'strands': {'nuc': [...], 'msp': [...]}, 'groups': {'nuc', 'msp'} present,
    'tag': 'MA'|'Ma'}``, 0-based, in the frame the tag was written in.
    ``tag`` defaults to :func:`footprint_annotation_tag`; None when that tag is
    missing or unparsable. ``nq`` is the first quality byte of each ``nuc``
    annotation (the ``Q`` of ``nuc.Q``/``nuc.QQQ``), or None unless every
    nucleosome has one.
    """
    from fiberhmm.io.ma_tags import parse_aq_array, parse_ma_tag

    if tag is None:
        tag = footprint_annotation_tag(read)
    if tag is None or not read.has_tag(tag):
        return None
    quality_tag = dict(ANNOTATION_SOURCES).get(tag, 'AQ')
    try:
        parsed = parse_ma_tag(str(read.get_tag(tag)))
    except (ValueError, TypeError):
        return None
    aq = read.get_tag(quality_tag) if read.has_tag(quality_tag) else None
    raw_types = parsed['raw_types']
    quals = parse_aq_array(aq, [rt[2] for rt in raw_types],
                           [len(rt[3]) for rt in raw_types])
    nucs, msps, nq = [], [], []
    strands = {'nuc': [], 'msp': []}
    groups = set()
    has_nq = aq is not None
    index = 0
    for name, strand, qual_spec, intervals in raw_types:
        if name in NUC_MSP_GROUPS:
            groups.add(name)
        for start, length in intervals:
            q = quals[index]
            index += 1
            if name == 'nuc':
                nucs.append((int(start), int(length)))
                strands['nuc'].append(strand)
                if q and qual_spec.startswith('Q'):
                    nq.append(int(q[0]))
                else:
                    has_nq = False
            elif name == 'msp':
                msps.append((int(start), int(length)))
                strands['msp'].append(strand)
    return {'nuc': nucs, 'msp': msps, 'nq': nq if has_nq and nucs else None,
            'strands': strands, 'groups': groups, 'tag': tag}


def header_ma_frame(header) -> Tuple[bool, bool]:
    """``(ma_molecular, ma_evidence)`` of FiberHMM ``MA`` intervals in a BAM
    with ``header``: molecular when the header declares coord=molecular, and
    whether the header has frame evidence at all (without it each group's
    strand decides: '.' molecular). ``(True, True)`` without a header."""
    if header is None:
        return True, True
    from fiberhmm.io.annotation_frame import MOLECULAR, ma_group_frame_of

    frame = ma_group_frame_of(header)
    return frame.header_frame == MOLECULAR, bool(frame.evidence)


def _int_array(values):
    """Compact signed int array (pickles small in worker payloads)."""
    return array.array('l', (int(v) for v in values))


def resolve_footprint_tags(read, want_molecular, *, ma_molecular: Optional[bool] = None,
                           ma_evidence: Optional[bool] = None) -> Optional[dict]:
    """The read's nucleosome/MSP calls as legacy-shaped tags, MA first.

    Returns ``{'source': 'MA'|'Ma'|'legacy', 'ns', 'nl', 'as', 'al', 'nq'}``
    (each pair only when the source has it) or None when the read has no
    calls in either form.

    * From MA/Ma (whenever :func:`footprint_annotation_tag` finds a nuc/msp
      group): intervals are expressed in the frame the caller reads legacy
      tags in -- molecular when ``want_molecular``, else SEQ -- so a reverse
      read's intervals are flipped when the tag's frame differs. fibertools'
      ``Ma`` is always molecular; FiberHMM's ``MA`` frame is
      ``ma_molecular``/``ma_evidence`` (default: from ``read.header``, see
      :func:`header_ma_frame`). ``nq`` is the MA nuc ``Q`` byte; legacy
      ``nq`` only when the legacy ns/nl list exactly the MA nucleosomes.
      Circular reads' MA pieces are used as linear pieces, like legacy tags.
    * From legacy ``ns/nl``/``as/al`` (only without MA/Ma nuc/msp groups):
      the tags' own values, in whatever frame they were written, by the
      array-pair rule; ``nq`` as stored.
    """
    from fiberhmm.io.legacy_tags import raw_array_tag_pair

    tag = footprint_annotation_tag(read)
    if tag is None:
        out = {}
        for start_tag, length_tag in (('ns', 'nl'), ('as', 'al')):
            pair = raw_array_tag_pair(read, start_tag, length_tag)
            if pair is not None:
                out[start_tag], out[length_tag] = pair
        if not out:
            return None
        if 'ns' in out and read.has_tag('nq'):
            out['nq'] = read.get_tag('nq')
        out['source'] = 'legacy'
        return out

    ma = annotation_nuc_msp(read, tag)
    if ma is None:
        return None
    from fiberhmm.io.annotation_frame import MOLECULAR, ma_group_is_molecular
    from fiberhmm.io.ma_tags import flip_intervals_to_seq

    if tag == 'Ma':
        ma_molecular, ma_evidence = True, True
    elif ma_molecular is None or ma_evidence is None:
        header_molecular, header_evidence = header_ma_frame(getattr(read, 'header', None))
        ma_molecular = header_molecular if ma_molecular is None else ma_molecular
        ma_evidence = header_evidence if ma_evidence is None else ma_evidence
    want_molecular = bool(want_molecular)
    frame_name = MOLECULAR if ma_molecular else 'seq'
    out = {'source': tag}
    for (start_tag, length_tag), feature in ((('ns', 'nl'), 'nuc'), (('as', 'al'), 'msp')):
        if feature not in ma['groups']:
            continue
        starts = [int(s) for s, _ in ma[feature]]
        lengths = [int(n) for _, n in ma[feature]]
        if ma_evidence:
            if bool(ma_molecular) != want_molecular:
                # flip_intervals_to_seq mirrors reverse reads' intervals; it is
                # its own inverse, so it also takes SEQ intervals to molecular.
                starts, lengths = flip_intervals_to_seq(starts, lengths, read)
        else:
            # No header evidence: each group's own strand decides its frame.
            converted = []
            for start, length, strand in zip(starts, lengths, ma['strands'][feature]):
                if ma_group_is_molecular(strand, frame_name, False) != want_molecular:
                    flipped = flip_intervals_to_seq([start], [length], read)
                    start, length = flipped[0][0], flipped[1][0]
                converted.append((int(start), int(length)))
            starts = [s for s, _ in converted]
            lengths = [n for _, n in converted]
        out[start_tag], out[length_tag] = _int_array(starts), _int_array(lengths)
    if 'ns' in out:
        nq = ma['nq']
        if nq is None:
            nq = _matching_legacy_nq(read, out['ns'], out['nl'])
        if nq is not None:
            out['nq'] = array.array('B', (max(0, min(255, int(q))) for q in nq))
    return out


def _matching_legacy_nq(read, ns, nl):
    """Legacy ``nq`` re-keyed onto the MA nucleosomes, when the legacy ns/nl
    (read in the same frame as ``ns``/``nl``) list exactly those nucleosomes;
    else None. Legacy scores never attach to nucleosomes they did not score."""
    from fiberhmm.io.legacy_tags import array_tag_pair, int_array

    if not read.has_tag('nq'):
        return None
    pair = array_tag_pair(read, 'ns', 'nl')
    scores = int_array(read.get_tag('nq'))
    if pair is None or scores is None or len(scores) != len(pair[0]):
        return None
    legacy = sorted(zip(pair[0], pair[1]))
    if legacy != sorted(zip((int(s) for s in ns), (int(n) for n in nl))):
        return None
    by_interval = dict(zip(zip(pair[0], pair[1]), scores))
    return [by_interval[(int(s), int(n))] for s, n in zip(ns, nl)]


def carry_annotation_msps_as_legacy(read, *, ma_molecular: Optional[bool] = None,
                                    ma_evidence: Optional[bool] = None) -> bool:
    """Write a read's MA/Ma ``msp`` calls as legacy ``as``/``al`` (molecular
    frame) when it has no legacy as/al pair, for a writer that keeps the
    record's own MSPs (``fiberhmm-apply --no-msps``) but drops the MA
    footprint groups first. A legacy pair already on the record is a carried
    annotation and is left in place. Returns whether the tags were written.
    ``ma_molecular``/``ma_evidence``: see :func:`resolve_footprint_tags` (a
    caller that already carried MA into molecular frame passes True, True)."""
    from fiberhmm.io.legacy_tags import has_array_tag_pair

    if has_array_tag_pair(read, 'as', 'al'):
        return False
    calls = resolve_footprint_tags(read, True, ma_molecular=ma_molecular, ma_evidence=ma_evidence)
    if not calls or calls['source'] == 'legacy' or not len(calls.get('as', ())):
        return False
    read.set_tag('as', array.array('I', [int(v) for v in calls['as']]))
    read.set_tag('al', array.array('I', [int(v) for v in calls['al']]))
    return True
