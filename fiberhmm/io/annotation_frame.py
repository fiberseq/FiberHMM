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

    1. any header text containing ``coord=molecular`` -> molecular;
    2. otherwise walk @PG in order: a fibertools-rs record whose command writes
       nucleosomes (``ft predict-m6a``/``m6a``, ``ft add-nucleosomes``, ``ft
       fire``, ``ft fiber-hmm``, and the old ``ft predict``/``ft add`` names)
       -> molecular; a FiberHMM record (``fiberhmm-*``) whose DS says
       ``coord=seq`` -> SEQ; a FiberHMM record with no coord token passed the
       tags through unchanged and leaves the frame as it was;
    3. nothing decided it -> unknown (``None``).

    FiberHMM <= 2.12 wrote ns/nl in SEQ frame and no @PG of its own; FiberHMM
    2.13.0 started writing molecular tags, its @PG and the ``coord=molecular``
    token in the same release. So no released FiberHMM wrote SEQ-frame tags
    under its own @PG, and an unmarked ``fiberhmm-*`` record is a tool that
    did not touch ns/nl. Unknown covers both an unmarked FiberHMM <= 2.12 BAM
    (SEQ) and a fibertools BAM whose @PG history was lost (molecular); callers
    must ask for an explicit frame rather than guess.

The rule matches FiberBrowser's ``bam_legacy_tags_are_molecular`` wherever that
has positive evidence. The one difference: FiberBrowser shows a fibertools BAM
followed by an unmarked FiberHMM @PG as SEQ frame, here the frame carries
through that record (for the reason above).
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
    return any(str(pg.get(key, '') or '').lower().startswith('fiberhmm')
               for key in ('PN', 'ID'))


def _is_fibertools_nuc_program(pg: dict) -> bool:
    return ('fibertools' in str(pg.get('PN', '') or '').lower()
            and bool(FIBERTOOLS_NUC_COMMAND_RE.search(str(pg.get('CL', '') or '').lower())))


def legacy_tag_frame(header) -> Tuple[Optional[str], str]:
    """Frame of a BAM's ns/nl/as/al tags from its provenance.

    Returns ``(frame, reason)``: ``frame`` is ``'molecular'``, ``'seq'`` or
    ``None`` when the header does not decide it; ``reason`` is a short
    human-readable account of the evidence. See the module docstring.
    """
    try:
        d = _header_dict(header)
    except (TypeError, ValueError):
        return None, 'unreadable header'
    if ma_annotation_frame(d) == MOLECULAR:
        return MOLECULAR, 'header declares coord=molecular'
    frame, reason = None, ''
    for pg in d.get('PG', []) or []:
        if not isinstance(pg, dict):
            continue
        name = pg.get('ID') or pg.get('PN') or '?'
        if _is_fiberhmm_program(pg):
            match = _COORD_TOKEN_RE.search(str(pg.get('DS', '') or ''))
            if match:
                frame = match.group(1).lower()
                reason = f'@PG {name} declares coord={frame}'
        elif _is_fibertools_nuc_program(pg):
            frame = MOLECULAR
            reason = f'fibertools @PG {name} wrote the nucleosome tags'
    if frame is None:
        return None, ('no coord=molecular declaration and no fibertools '
                      'nucleosome command in the @PG history')
    return frame, reason


def resolve_disabled_legacy_frame(header) -> Optional[Tuple[str, str]]:
    """``('molecular', reason)`` when provenance shows molecular ns/nl/as/al.

    For consensus's ``legacy_hia5_annotation_frame='disabled'``: a fibertools
    nucleosome command or a coord=molecular declaration settles the frame, so
    the option need not be set by hand. Anything else (unknown, or a FiberHMM
    coord=seq record) returns ``None`` and the explicit-frame error stands.
    """
    frame, reason = legacy_tag_frame(header)
    return (frame, reason) if frame == MOLECULAR else None


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
    return f'{ds}; {token} (footprint tags carried over from the input)' if ds else token


def merged_input_frame(headers) -> Optional[str]:
    """One frame for several inputs, or ``None`` when any is unknown or they differ."""
    frames = {legacy_tag_frame(header)[0] for header in headers}
    if len(frames) == 1:
        return frames.pop()
    return None


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
