"""Coordinate-frame policy for footprint annotations read from a BAM.

Three tag families carry footprints, and each has its own frame rule:

``MA``/``AQ``/``AN`` (written only by FiberHMM)
    Historical unmarked FiberHMM BAMs use stored SEQ coordinates. Molecular MA
    producers declare ``coord=molecular`` in PG/CO; never infer it from
    alignment flags (:func:`ma_annotation_frame`).

``Ma``/``Aq``/``An`` (fibertools-rs >= 0.13)
    Always molecular (original read orientation), whatever the header says.

``ns/nl/as/al`` (fibertools, and FiberHMM >= 2.0)
    The tags carry no frame, so it comes from the @PG/@CO provenance
    (:func:`legacy_tag_frame`):

    1. follow the @PG history through PP links (the graph of
       :mod:`fiberhmm.advisories`: samtools-merge branches, renamed IDs,
       missing links); on each branch the last record bearing on footprint
       tags decides: a FiberHMM record (``fiberhmm-*``) with ``coord=`` in DS
       -> that frame; a fibertools-rs record (PN, ID or CL program) whose
       command writes nucleosomes (``ft predict-m6a``/``m6a``, ``ft
       add-nucleosomes``, ``ft fire``, ``ft fiber-hmm``, and the old ``ft
       predict``/``ft add`` names) -> molecular; a FiberHMM record with no
       coord token passed the tags through and is looked through;
    2. all branches agree -> that frame;
    3. otherwise the header's explicit coord= declarations decide if they
       agree; else unknown (``None``).

    FiberHMM <= 2.12 wrote ns/nl in SEQ frame and no @PG of its own; FiberHMM
    2.13.0 started writing molecular tags, its @PG and the ``coord=molecular``
    token in the same release. So no released FiberHMM wrote SEQ-frame tags
    under its own @PG, and an unmarked ``fiberhmm-*`` record is a tool that
    did not touch ns/nl. Unknown covers both an unmarked FiberHMM <= 2.12 BAM
    (SEQ) and a fibertools BAM whose @PG history was lost (molecular); callers
    must ask for an explicit frame rather than guess.

FiberBrowser's ``bam_legacy_tags_are_molecular`` applies the same evidence;
an unmarked ``fiberhmm-*`` record after fibertools carries the frame through
here (for the reason above).
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

MOLECULAR = 'molecular'
SEQ = 'seq'

# fibertools-rs subcommands that (re)write ns/nl (kept identical to
# FiberBrowser's browser/services/bam_tags.py so both tools agree):
# predict-m6a (aliases m6a/m6A; `ft predict` in fibertools-rs 0.1-0.3, still
# accepted as a prefix by 0.13), add-nucleosomes (`ft add` in 0.2, likewise),
# fire, fiber-hmm.
FIBERTOOLS_NUC_COMMAND_RE = re.compile(
    r"(?:^|[\s/])(?:ft|fibertools)(?:\s+\S+)*?\s+"
    r"(?:predict(?:-m6a)?|m6a|add(?:-nuc\w*)?|fire|fiber-hmm)(?:\s|$)")

_COORD_TOKEN_RE = re.compile(r"coord=(molecular|seq)\b", re.IGNORECASE)


def _header_dict(header) -> dict:
    return header.to_dict() if hasattr(header, 'to_dict') else dict(header or {})


def ma_annotation_frame(header):
    header = _header_dict(header)
    texts = [str(c) for c in header.get('CO', [])]
    texts += [' '.join(str(v) for v in pg.values()) for pg in header.get('PG', [])]
    return 'molecular' if any('coord=molecular' in text.lower() for text in texts) else 'seq'


def _is_fiberhmm_program(pg: dict) -> bool:
    """A FiberHMM program record: PN or ID ``fiberhmm-*`` (samtools may suffix
    the ID). A command line that merely mentions fiberhmm is not one."""
    return any(str(pg.get(key, '') or '').lower().startswith('fiberhmm')
               for key in ('PN', 'ID'))


def _is_fibertools_program(pg: dict) -> bool:
    """fibertools-rs by PN, ID (``ft``, ``ft.1``, ``fibertools...``, samtools-
    suffixed) or the CL executable, so records without PN count too."""
    if 'fibertools' in str(pg.get('PN', '') or '').lower():
        return True
    if re.match(r'(?:ft|fibertools)(?:$|[.\-_])', str(pg.get('ID', '') or '').lower()):
        return True
    words = str(pg.get('CL', '') or '').split()
    return bool(words) and words[0].rsplit('/', 1)[-1].lower() in ('ft', 'fibertools')


def _is_fibertools_nuc_program(pg: dict) -> bool:
    return (_is_fibertools_program(pg)
            and bool(FIBERTOOLS_NUC_COMMAND_RE.search(str(pg.get('CL', '') or '').lower())))


# How one @PG record bears on the frame of the footprint tags after it:
# (frame, kind) with kind 'fibertools' (wrote ns/nl molecular), 'caller'
# (FiberHMM wrote its own tags and declared their frame) or 'carried' (a
# FiberHMM pass-through tool recorded the frame it carried over); None when
# the record does not touch footprint tags.
def _record_decision(pg: dict):
    if _is_fiberhmm_program(pg):
        ds = str(pg.get('DS', '') or '')
        match = _COORD_TOKEN_RE.search(ds)
        if match:
            return (match.group(1).lower(), 'carried' if CARRIED_OVER in ds else 'caller')
        return None
    if _is_fibertools_nuc_program(pg):
        return (MOLECULAR, 'fibertools')
    return None


def _branch_decisions(d: dict, transparent=()) -> set:
    """The nearest deciding record on every ancestry branch of every leaf of
    the @PG history (the PP graph of :mod:`fiberhmm.advisories`: samtools
    merge joins, renamed IDs, missing, duplicated or forward PP links), over
    each plausible reading. Returns a set of ``(frame, kind)`` and ``None``
    for branches nothing decides. Records whose kind is in ``transparent``
    are looked through. A step that joined several inputs but kept one header
    (``samtools cat``, GatherBamFiles) adds an undecided branch: the other
    inputs' history is not in the header."""
    from fiberhmm.advisories import _drops_input_headers, _history_graphs, _pg_records
    records = _pg_records(d)
    if not records:
        return {None}
    graphs, node = _history_graphs(records)
    own, dropped = {}, set()
    for index, record in enumerate(records):
        decision = _record_decision(record)
        if decision is not None and decision[1] not in transparent:
            own.setdefault(node[index], decision)
        if _drops_input_headers(record):
            dropped.add(node[index])
    nodes = sorted(set(node))
    out = set()
    for graph in graphs:
        parents = graph.parents
        children = {p for n in nodes for p in parents[n]}
        memo = {}

        def decide(n, visiting):
            if n in memo:
                return memo[n]
            if n in own:
                result = {own[n]}
            elif not parents[n]:
                result = {None}
            else:
                result = set()
                for parent in parents[n]:
                    if parent not in visiting:  # a PP cycle adds nothing
                        result |= decide(parent, visiting | {n})
                result = result or {None}
                if n in dropped:
                    result = result | {None}
            memo[n] = result
            return result

        for leaf in (n for n in nodes if n not in children):
            out |= decide(leaf, frozenset())
    return out


def _explicit_frames(d: dict) -> set:
    """Every frame the header states outright: coord= tokens in FiberHMM @PG
    DS, and coord=molecular in any @PG or @CO text."""
    frames = set()
    for pg in d.get('PG', []) or []:
        if not isinstance(pg, dict):
            continue
        if _is_fiberhmm_program(pg):
            match = _COORD_TOKEN_RE.search(str(pg.get('DS', '') or ''))
            if match:
                frames.add(match.group(1).lower())
        if 'coord=molecular' in ' '.join(str(v) for v in pg.values()).lower():
            frames.add(MOLECULAR)
    if any('coord=molecular' in str(c).lower() for c in d.get('CO', []) or []):
        frames.add(MOLECULAR)
    return frames


def legacy_tag_frame(header) -> Tuple[Optional[str], str]:
    """Frame of a BAM's ns/nl/as/al tags from its provenance.

    Returns ``(frame, reason)``: ``frame`` is ``'molecular'``, ``'seq'`` or
    ``None`` when the header does not decide it; ``reason`` is a short
    human-readable account of the evidence. The @PG history is followed
    through its PP links: each branch (samtools merge keeps one per input)
    takes the frame of its last footprint-writing record. When the branches
    disagree or some are undecided, the header's explicit coord= declarations
    decide if they agree; otherwise the frame is unknown.
    """
    try:
        d = _header_dict(header)
        branches = _branch_decisions(d)
    except (TypeError, ValueError, AttributeError):
        return None, 'unreadable header'
    frames = {b[0] if b else None for b in branches}
    if len(frames) == 1 and None not in frames:
        frame = frames.pop()
        kinds = {b[1] for b in branches}
        how = ('a fibertools nucleosome command wrote the tags' if kinds == {'fibertools'}
               else f'the @PG history declares coord={frame}')
        return frame, how
    explicit = _explicit_frames(d)
    if len(explicit) == 1:
        frame = explicit.pop()
        return frame, f'the header declares coord={frame}'
    if len(frames - {None}) > 1 or len(explicit) > 1:
        return None, ('merged @PG histories disagree on the frame of the '
                      'footprint tags')
    return None, ('no coord=molecular declaration and no fibertools '
                  'nucleosome command in the @PG history')


def resolve_disabled_legacy_frame(header) -> Optional[Tuple[str, str]]:
    """``('molecular', reason)`` when fibertools wrote the ns/nl/as/al.

    For consensus's ``legacy_hia5_annotation_frame='disabled'``, which only
    applies to reads without MA/Ma. When, on every branch of the @PG history,
    fibertools' nucleosome command is the last program that wrote footprint
    tags, every read's legacy tags are fibertools' own (molecular). A FiberHMM
    caller (a coord= declaration on its own writes, or the @CO marker) does
    not vouch for them: the reads it left without MA are ones it skipped, and
    they keep whatever tags came before it (for example query-frame tags from
    FiberHMM <= 2.12). FiberHMM pass-through records are looked through.
    Anything else returns ``None`` and the explicit-frame error stands.
    """
    try:
        d = _header_dict(header)
        if any('coord=molecular' in str(c).lower() for c in d.get('CO', []) or []):
            return None
        branches = _branch_decisions(d, transparent=('carried',))
    except (TypeError, ValueError, AttributeError):
        return None
    if branches == {(MOLECULAR, 'fibertools')}:
        return MOLECULAR, 'a fibertools nucleosome command wrote the tags'
    return None


CARRIED_OVER = 'footprint tags carried over from the input'


def coord_ds_token(frame: Optional[str]) -> str:
    """``coord=<frame>`` for a @PG DS, or ``''`` when the frame is unknown."""
    return f'coord={frame}' if frame in (MOLECULAR, SEQ) else ''


def append_coord_to_ds(ds: str, frame: Optional[str]) -> str:
    """Record the frame of the ns/nl/as/al a pass-through tool carried over.

    Tools that copy footprint tags without rewriting them keep the input's
    frame; saying so in their own @PG lets readers that look at the latest
    record (and people reading the header) see it. Unknown frames are left
    unrecorded rather than guessed.
    """
    token = coord_ds_token(frame)
    if not token:
        return ds
    token = f'{token} ({CARRIED_OVER})'
    return f'{ds}; {token}' if ds else token


def pass_through_frame(header, explicit=None) -> Optional[str]:
    """Frame of the footprint tags a tool copies from ``header``'s BAM.

    ``explicit`` is the tool's own frame choice when it has one (``True`` /
    ``'molecular'``, ``False`` / ``'query'`` / ``'seq'``); ``None`` or
    ``'auto'`` uses the provenance rule (:func:`legacy_tag_frame`).
    """
    if explicit in (True, MOLECULAR):
        return MOLECULAR
    if explicit in (False, SEQ, 'query'):
        return SEQ
    return legacy_tag_frame(header)[0]


AMBIGUOUS_FRAME_HELP = (
    'FiberHMM cannot tell which coordinate frame this BAM\'s ns/nl/as/al tags '
    'use ({reason}). Reverse-strand footprints would be mirrored if it '
    'guessed wrong. Pass the frame explicitly: {query_flag} for output of '
    'FiberHMM 2.12 or earlier (SEQ frame); {molecular_flag} for fibertools '
    'output whose @PG history was lost, or any other molecular-frame BAM.'
)


def ambiguous_frame_message(reason: str, *, query_flag: str = '--input-frame query',
                            molecular_flag: str = '--input-frame molecular') -> str:
    return AMBIGUOUS_FRAME_HELP.format(reason=reason, query_flag=query_flag,
                                       molecular_flag=molecular_flag)
