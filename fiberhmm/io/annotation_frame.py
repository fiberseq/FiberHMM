"""Coordinate-frame policy for footprint annotations read from a BAM.

Three tag families carry footprints, and each has its own frame rule
(docs/reference/footprint-tag-frame.md, shared with FiberBrowser's
``browser/services/bam_tags.py``):

``MA``/``AQ``/``AN`` (written only by FiberHMM)
    Historical unmarked FiberHMM BAMs use stored SEQ coordinates. Molecular MA
    producers declare ``coord=molecular`` in PG/CO; never infer it from
    alignment flags (:func:`ma_annotation_frame`).

``Ma``/``Aq``/``An`` (fibertools-rs >= 0.13)
    Always molecular (original read orientation), whatever the header says.

``ns/nl/as/al`` (fibertools, and FiberHMM >= 2.0)
    The tags carry no frame, so it comes from the @PG provenance
    (:func:`legacy_tag_frame_report`): every @PG leaf starts a chain walked up
    through PP; the chain votes with the frame of the last footprint writer
    that ran on it (fibertools nucleosome commands: molecular; FiberHMM
    apply/call/recall-tfs/recall-nucs: molecular when declared, else SEQ).
    FiberHMM pass-through tools do not vote. Agreeing votes decide; no votes
    mean molecular if coord=molecular is declared anywhere, else SEQ
    (unmarked FiberHMM <= 2.12). Disagreeing votes (merged files) mean
    molecular when declared; otherwise FiberBrowser shows the majority, while
    FiberHMM, where a wrong guess changes calls, refuses
    (:func:`legacy_tag_frame` returns ``None``).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

MOLECULAR = 'molecular'
SEQ = 'seq'
_COORD_MOLECULAR = 'coord=molecular'

# fibertools-rs subcommands that (re)write ns/nl: predict-m6a (m6a; `predict`
# in 0.1-0.3), add-nucleosomes (`add` in 0.2), fire, fiber-hmm. fibertools
# accepts any unambiguous subcommand prefix (`ft add-nuc` is in real headers);
# the shortest unambiguous prefix of each writer.
FIBERTOOLS_FOOTPRINT_SUBCOMMANDS = frozenset(
    {'predict', 'predict-m6a', 'm6a', 'add', 'add-nucleosomes', 'add-nucleosome', 'fire', 'fiber-hmm'})
_FIBERTOOLS_FOOTPRINT_PREFIXES = (('add-nucleosomes', 3), ('predict-m6a', 4), ('fire', 3), ('fiber-hmm', 5))
_FIBERTOOLS_PROGRAMS = frozenset({'ft', 'fibertools', 'fibertools-rs'})
# FiberHMM programs that write the legacy tags. Every other fiberhmm-* program
# (tag-m5c, call-m5c, dedup, merge, pair, strand-rescue-annotate, tag-consensus,
# pipeline, ...) passes them through unchanged.
FIBERHMM_FOOTPRINT_PROGRAMS = frozenset(
    {'fiberhmm-apply', 'fiberhmm-call', 'fiberhmm-recall-tfs', 'fiberhmm-recall-nucs'})
_FIBERHMM_NAME_RE = re.compile(r'^fiberhmm(?:-[a-z0-9-]+)?$')
# @PG ID collision suffixes added by samtools/pysam when headers are merged
# (`ID.1`, `ID-61483E4`); stripped for classification only, never for PP.
_PG_ID_SUFFIX_RE = re.compile(r'(?:\.\d+|-[0-9a-f]{6,8})$')

# Pass-through tools append this to their @PG DS with coord=molecular when the
# tags they carried are molecular (append_coord_to_ds).
CARRIED_OVER = 'footprint tags carried over from the input'


def _header_dict(header) -> Optional[dict]:
    try:
        return header.to_dict() if hasattr(header, 'to_dict') else dict(header or {})
    except (TypeError, ValueError):
        return None


def ma_annotation_frame(header):
    header = _header_dict(header) or {}
    texts = [str(c) for c in header.get('CO', [])]
    texts += [' '.join(str(v) for v in pg.values()) for pg in header.get('PG', [])]
    return 'molecular' if any('coord=molecular' in text.lower() for text in texts) else 'seq'


def _is_fibertools_footprint_subcommand(token: str) -> bool:
    if token in FIBERTOOLS_FOOTPRINT_SUBCOMMANDS:
        return True
    return any(len(token) >= n and full.startswith(token) for full, n in _FIBERTOOLS_FOOTPRINT_PREFIXES)


def _pg_id_base(ident: str) -> str:
    base = ident.lower()
    while True:
        stripped = _PG_ID_SUFFIX_RE.sub('', base)
        if stripped == base or not stripped:
            return base
        base = stripped


def _cl_tokens(cl) -> List[str]:
    import shlex
    try:
        return shlex.split(str(cl or '').lower())
    except ValueError:
        return str(cl or '').lower().split()


def _basename(token: str) -> str:
    return token.rstrip('/').rsplit('/', 1)[-1]


def pg_family(pg: dict) -> Optional[str]:
    """``'fibertools'``, ``'fiberhmm'`` or None (any other program).

    PN decides when present; else the ID with collision suffixes stripped; else
    the program named by the first CL token. A PN such as ``samtools`` is never
    overridden by an ID or CL that happens to mention a producer."""
    pn = str(pg.get('PN', '') or '').strip().lower()
    if pn:
        if pn in _FIBERTOOLS_PROGRAMS:
            return 'fibertools'
        if 'fibertools' in pn:
            return 'fibertools'
        return 'fiberhmm' if _FIBERHMM_NAME_RE.match(pn) else None
    base = _pg_id_base(str(pg.get('ID', '') or '').strip())
    if base in _FIBERTOOLS_PROGRAMS or base.startswith('fibertools'):
        return 'fibertools'
    if _FIBERHMM_NAME_RE.match(base):
        return 'fiberhmm'
    tokens = _cl_tokens(pg.get('CL', ''))
    program = _basename(tokens[0]) if tokens else ''
    if program in _FIBERTOOLS_PROGRAMS:
        return 'fibertools'
    if _FIBERHMM_NAME_RE.match(program):
        return 'fiberhmm'
    return None


def _pg_text(pg: dict) -> str:
    try:
        return ' '.join(str(v) for v in pg.values()).lower()
    except AttributeError:
        return str(pg).lower()


def _is_carried_over(pg: dict) -> bool:
    """A FiberHMM pass-through record that recorded the frame it carried."""
    return CARRIED_OVER in str(pg.get('DS', '') or '')


def pg_writer_frame(pg: dict, header_marker: bool, *, carried_votes: bool = True) -> Optional[str]:
    """Frame of the legacy tags this @PG record wrote, or None if it wrote none.

    ``carried_votes=False`` looks through pass-through records that declared
    ``coord=molecular`` for tags they carried over (the shared rule counts them
    as writers of that frame, which gives the same answer)."""
    family = pg_family(pg)
    if family == 'fibertools':
        tokens = _cl_tokens(pg.get('CL', ''))
        start = next((i + 1 for i, t in enumerate(tokens) if _basename(t) in _FIBERTOOLS_PROGRAMS), 1)
        if any(_is_fibertools_footprint_subcommand(t) for t in tokens[start:]):
            return MOLECULAR   # fibertools only ever writes molecular frame
        return None
    if family == 'fiberhmm':
        record_marker = _COORD_MOLECULAR in _pg_text(pg)
        tokens = _cl_tokens(pg.get('CL', ''))
        names = {
            str(pg.get('PN', '') or '').strip().lower(),
            _pg_id_base(str(pg.get('ID', '') or '').strip()),
            _basename(tokens[0]) if tokens else '',
        }
        writer = bool(names & FIBERHMM_FOOTPRINT_PROGRAMS)
        if not writer and record_marker and not carried_votes and _is_carried_over(pg):
            return None
        if record_marker or writer:
            return MOLECULAR if (record_marker or header_marker) else SEQ
    return None


def legacy_tag_frame_report(header, *, carried_votes: bool = True) -> dict:
    """How the legacy ns/nl/as/al tags' frame was decided (the shared rule;
    field-for-field FiberBrowser's ``legacy_tag_frame_report``).

    Each @PG leaf (a record no other record names as PP) starts a chain that is
    walked up through PP. The chain's frame is that of the first footprint
    writer met (the last one to run). Header order is used as the chain only
    when no record has a PP field. Chains without a writer do not vote.
    Agreeing votes decide; with no votes an explicit coord=molecular anywhere
    means molecular, else SEQ (unmarked legacy FiberHMM). Conflicting votes
    (merged files) resolve to molecular when coord=molecular is declared
    anywhere, else to the majority (ties: SEQ), and are reported as
    ambiguous; ``declared`` says whether a declaration exists. ``writers``
    lists each vote's record ID, frame and producer family.
    """
    report = {'frame': SEQ, 'source': 'default', 'ambiguous': False, 'declared': False,
              'votes': {MOLECULAR: 0, SEQ: 0}, 'writers': [], 'chains': 0}
    hdr = _header_dict(header)
    if hdr is None:
        return report
    pgs = [pg for pg in (hdr.get('PG', []) or []) if isinstance(pg, dict)]
    comments = [str(c).lower() for c in (hdr.get('CO', []) or [])]
    header_marker = any(_COORD_MOLECULAR in c for c in comments)
    declared = header_marker or any(_COORD_MOLECULAR in _pg_text(pg) for pg in pgs)
    report['declared'] = declared

    frames = [pg_writer_frame(pg, header_marker, carried_votes=carried_votes) for pg in pgs]
    ids = [str(pg.get('ID', '') or '') for pg in pgs]
    parent: List[Optional[int]] = [None] * len(pgs)
    if any('PP' in pg for pg in pgs):
        by_id: Dict[str, List[int]] = {}
        for i, ident in enumerate(ids):
            by_id.setdefault(ident, []).append(i)
        for i, pg in enumerate(pgs):
            pp = str(pg.get('PP', '') or '')
            candidates = [j for j in by_id.get(pp, []) if j != i] if pp else []
            if candidates:
                earlier = [j for j in candidates if j < i]
                parent[i] = earlier[-1] if earlier else candidates[-1]
    else:
        parent = [i - 1 if i > 0 else None for i in range(len(pgs))]
    referenced = {j for j in parent if j is not None}
    leaves = [i for i in range(len(pgs)) if i not in referenced]
    if pgs and not leaves:          # every record is someone's parent: a PP cycle
        leaves = [len(pgs) - 1]
    report['chains'] = len(leaves)

    for leaf in leaves:
        node, seen = leaf, set()
        while node is not None and node not in seen:
            seen.add(node)
            if frames[node] is not None:
                report['votes'][frames[node]] += 1
                report['writers'].append({'id': ids[node], 'frame': frames[node],
                                          'family': pg_family(pgs[node])})
                break
            node = parent[node]

    mol, seq = report['votes'][MOLECULAR], report['votes'][SEQ]
    if mol and seq:
        report['ambiguous'] = True
        report['source'] = 'declared' if declared else 'majority'
        report['frame'] = MOLECULAR if (declared or mol > seq) else SEQ
    elif mol or seq:
        report['source'] = 'provenance'
        report['frame'] = MOLECULAR if mol else SEQ
    elif declared:
        report['source'] = 'declared'
        report['frame'] = MOLECULAR
    return report


def legacy_tag_frame(header) -> Tuple[Optional[str], str]:
    """Frame of a BAM's ns/nl/as/al for FiberHMM: ``(frame, reason)``.

    The shared rule (:func:`legacy_tag_frame_report`), except that merged
    histories whose writers disagree with no coord=molecular declaration give
    ``None``: FiberBrowser may show the majority, but FiberHMM would change
    calls on a wrong guess, so its tools ask for an explicit frame instead.
    """
    report = legacy_tag_frame_report(header)
    votes = f"{report['votes'][MOLECULAR]} molecular / {report['votes'][SEQ]} SEQ"
    if report['ambiguous'] and not report['declared']:
        return None, (f'merged @PG histories disagree ({votes} footprint-writer votes) '
                      'and nothing declares coord=molecular')
    reasons = {
        'provenance': 'the last footprint writer on the @PG chains that have one',
        'declared': 'coord=molecular is declared in the header',
        'default': 'no footprint writer in the @PG history (unmarked FiberHMM <= 2.12 wrote SEQ)',
    }
    reason = reasons[report['source']]
    if report['ambiguous']:
        reason += f' (merged histories disagree: {votes})'
    return report['frame'], reason


CONSENSUS_CONTRACT_PREFIX = 'FIBERHMM-CONSENSUS-MA:v1:'
CONSENSUS_MOLECULAR_COORDINATES = 'original_source_call_in_molecular_frame'


def consensus_molecular_layers(header) -> frozenset:
    """MA layers a FIBERHMM-CONSENSUS-MA contract declares molecular.

    Consensus export writes its family layers in molecular frame and says so
    per layer (``coordinates``), whatever the header's MA frame is; readers
    honour it ahead of :func:`ma_annotation_frame` (FiberBrowser's
    ``bam_consensus_contracts`` does the same)."""
    import json
    hdr = _header_dict(header) or {}
    layers = set()
    for comment in hdr.get('CO', []) or []:
        text = str(comment)
        if not text.startswith(CONSENSUS_CONTRACT_PREFIX):
            continue
        try:
            value = json.loads(text[len(CONSENSUS_CONTRACT_PREFIX):])
        except ValueError:
            continue
        if isinstance(value, dict) and value.get('coordinates') == CONSENSUS_MOLECULAR_COORDINATES:
            layers.update(str(layer) for layer in value.get('layers', []) or [])
    return frozenset(layers)


def resolve_disabled_legacy_frame(header) -> Optional[Tuple[str, str]]:
    """``('molecular', reason)`` when fibertools wrote the ns/nl/as/al.

    For consensus's ``legacy_hia5_annotation_frame='disabled'``, which only
    applies to reads without MA/Ma. It is stricter than the shared rule: every
    @PG chain's last footprint writer must be fibertools. A FiberHMM writer
    (call/apply/recall, or the @CO marker of older ones) does not vouch for
    legacy tags: the reads it left without MA are ones it skipped, and they
    keep whatever tags came before it (for example query-frame tags from
    FiberHMM <= 2.12). Pass-through records are looked through. Anything else
    returns ``None`` and the explicit-frame error stands.
    """
    hdr = _header_dict(header)
    if hdr is None:
        return None
    if any(_COORD_MOLECULAR in str(c).lower() for c in hdr.get('CO', []) or []):
        return None
    report = legacy_tag_frame_report(hdr, carried_votes=False)
    writers = report['writers']
    # Every chain must vote fibertools: a chain with no footprint writer (an
    # independent aligner history merged in, say) is unknown, not molecular.
    if (report['source'] == 'provenance' and report['frame'] == MOLECULAR and writers
            and len(writers) == report['chains']
            and all(w['family'] == 'fibertools' for w in writers)):
        return MOLECULAR, 'a fibertools nucleosome command is the last footprint writer on every @PG chain'
    return None


def coord_ds_token(frame: Optional[str]) -> str:
    """``coord=molecular`` for a @PG DS, or ``''``. The shared rule has one
    declaration token; SEQ and unknown frames are not recorded."""
    return _COORD_MOLECULAR if frame == MOLECULAR else ''


def append_coord_to_ds(ds: str, frame: Optional[str]) -> str:
    """Record that a pass-through tool carried molecular-frame footprint tags.

    Tools that copy footprint tags without rewriting them keep the input's
    frame. Saying so in their own @PG makes the header state it outright
    (readers that only look for the declaration, and people reading the
    header). SEQ and unknown frames are left unrecorded rather than guessed.
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
    'use: {reason}. Reverse-strand footprints would be mirrored if it guessed '
    'wrong. Pass the frame explicitly: {query_flag} if they are SEQ frame '
    '(FiberHMM 2.12 or earlier), {molecular_flag} if they are molecular '
    '(fibertools, FiberHMM 2.13 or later). A merged file whose inputs used '
    'different frames cannot be read either way; split it by input.'
)


def ambiguous_frame_message(reason: str, *, query_flag: str = '--input-frame query',
                            molecular_flag: str = '--input-frame molecular') -> str:
    return AMBIGUOUS_FRAME_HELP.format(reason=reason, query_flag=query_flag,
                                       molecular_flag=molecular_flag)


# ---------------------------------------------------------------------------
# Headers that give no frame evidence (a pipeline stripped @PG)
# ---------------------------------------------------------------------------
#
# Every rule above is unchanged and decides whenever the header has any
# evidence (a footprint writer on a @PG chain, or coord=molecular anywhere).
# Only when it has none:
#
# * MA, per annotation group: a group written with the '.' strand is molecular
#   (FiberHMM >= 2.13.2 and fibertools write every type that way); '+'/'-'
#   groups follow the header rule (SEQ). A group never decides another
#   group's frame.
# * legacy ns/nl/as/al: a read whose MA has a '.' *footprint* group
#   (nuc/msp/tf, not an m5C layer) has molecular legacy tags (the same writer
#   wrote both); else SEQ is assumed, with a warning. A FIBERHMM-CHEMISTRY
#   declaration decides nothing: pass-through producers (merge, pair) add it
#   while carrying historical tags unchanged.

FOOTPRINT_GROUPS = frozenset({'nuc', 'msp', 'tf'})
_MA_GROUP_HEAD_RE = re.compile(r'^([A-Za-z0-9_]+?)([.+-])([A-Za-z]*)$')


def ma_group_strands(ma) -> List[Tuple[str, str]]:
    """``[(name, strand), ...]`` of an MA string's groups, in order."""
    out = []
    for group in str(ma or '').split(';')[1:]:
        head = group.partition(':')[0]
        match = _MA_GROUP_HEAD_RE.match(head)
        if match:
            out.append((match.group(1), match.group(2)))
    return out


def read_has_dot_footprint_group(read) -> bool:
    """Whether the read's MA has a '.'-strand nuc/msp/tf group."""
    try:
        if not read.has_tag('MA'):
            return False
        ma = read.get_tag('MA')
    except (AttributeError, KeyError, TypeError):
        return False
    return any(name in FOOTPRINT_GROUPS and strand == '.'
               for name, strand in ma_group_strands(ma))


def header_has_frame_evidence(header) -> bool:
    """Whether the header decides footprint frames by the rules above: a
    footprint writer on a @PG chain or a coord=molecular declaration."""
    return legacy_tag_frame_report(header)['source'] != 'default'


def ma_group_is_molecular(strand: str, header_ma_frame: str, header_evidence: bool) -> bool:
    """One MA group's frame: the header rule, except that without header
    evidence a '.'-strand group is molecular."""
    if header_ma_frame == MOLECULAR:
        return True
    return (not header_evidence) and strand == '.'


# DddA m5C groups. Their producers (tag-m5c, call-m5c, pair; @PG DS
# ddda_mcg_frame=molecular) write every qualifier ('.', '+', '-') molecular,
# whatever frame the footprint tags use, and fiberhmm.daf.m5c reads them so
# (the recall CpG mask); readers follow that in every header state.
M5C_GROUPS = frozenset({'ddda_mcg', 'ddda_ucg', 'ddda_mcg_hemi'})


class MAGroupFrame:
    """Frame of each MA group of a BAM, by the rules above: m5C groups,
    consensus family layers and fiberhmm-scan groups (declared by a
    FIBERHMM-SCAN-MA contract) molecular; otherwise the header rule, except
    that without header evidence a '.' group is molecular. Picklable."""

    def __init__(self, header):
        from fiberhmm.io.scan_ma import scan_molecular_groups
        self.header_frame = ma_annotation_frame(header)
        self.evidence = header_has_frame_evidence(header)
        hdr = _header_dict(header) or {}
        self.molecular_layers = frozenset(consensus_molecular_layers(header) | M5C_GROUPS
                                          | scan_molecular_groups(hdr))

    def is_molecular(self, name: str, strand: str) -> bool:
        return (name in self.molecular_layers
                or ma_group_is_molecular(strand, self.header_frame, self.evidence))


_GROUP_FRAMES: Dict[int, tuple] = {}


def _cached(kind, header, build):
    key = (kind, id(header))
    cached = _GROUP_FRAMES.get(key)
    if cached is not None and cached[0] is header:
        return cached[1]
    resolver = build(header)
    if len(_GROUP_FRAMES) > 32:
        _GROUP_FRAMES.clear()
    _GROUP_FRAMES[key] = (header, resolver)
    return resolver


def ma_group_frame_of(header) -> MAGroupFrame:
    """:class:`MAGroupFrame` of ``header``, cached per header object (pysam
    hands every record of one file the same header)."""
    return _cached('ma', header, MAGroupFrame)


def legacy_frame_of(header) -> "LegacyFrame":
    """:class:`LegacyFrame` (auto) of ``header``, cached like
    :func:`ma_group_frame_of`."""
    return _cached('legacy', header, LegacyFrame)


class LegacyFrame:
    """Frame of legacy ns/nl/as/al, per read.

    ``explicit`` (True/False) is a user's choice and always wins. Otherwise
    the header rule (:func:`legacy_tag_frame`) decides whenever the header has
    evidence (``frame`` may then be ``None``: undecided merged histories).
    Without evidence: a read with a '.' footprint MA group is molecular; else
    SEQ, assumed.
    """

    def __init__(self, header, explicit: Optional[bool] = None):
        self.explicit = explicit
        self.evidence = header_has_frame_evidence(header)
        frame, reason = legacy_tag_frame(header)
        if explicit is not None:
            self.frame, self.reason = (MOLECULAR if explicit else SEQ), 'explicit'
        else:
            self.frame, self.reason = frame, reason
        self.assumed = explicit is None and not self.evidence

    def for_read(self, read) -> Optional[str]:
        if self.assumed and read_has_dot_footprint_group(read):
            return MOLECULAR
        return self.frame

    def is_molecular(self, read) -> bool:
        return self.for_read(read) == MOLECULAR

    def warning(self, tool: str) -> Optional[str]:
        if not self.assumed:
            return None
        return (f"{tool}: warning: the header records no footprint writer, "
                "or coord=molecular declaration (stripped @PG?); legacy "
                "ns/nl/as/al are assumed SEQ frame, except on reads whose MA has a "
                "'.' nuc/msp/tf group (FiberHMM >= 2.13.2), taken as molecular. "
                "Pass --input-frame to decide it.")


def auto_legacy_frame(header, tool: Optional[str] = None):
    """``--input-frame auto`` for tools taking a per-read frame (tag-m5c,
    call-m5c): ``True``/``False`` when the header decides (exactly
    :func:`legacy_tag_frame`), ``None`` when it is undecided, or a per-read
    :class:`LegacyFrame` when the header has no evidence (the caller warns
    once per input file: :func:`warn_if_frames_unverifiable`)."""
    resolver = LegacyFrame(header)
    if resolver.assumed:
        return resolver
    if resolver.frame is None:
        return None
    return resolver.frame == MOLECULAR


# ---------------------------------------------------------------------------
# One warning per input file whose header gives no frame evidence
# ---------------------------------------------------------------------------

UNVERIFIABLE_FRAME_WARNING = (
    "warning: {path}: header has no FiberHMM/fibertools provenance, so "
    "footprint coordinates on reverse reads can't be confirmed; re-call with "
    "FiberHMM 3.0 (or pass --input-frame where available)")
_WARNED_PATHS: set = set()
_CHECKED_PATHS: set = set()


def read_has_frame_dependent_annotations(read) -> bool:
    """Whether ``read`` carries footprint annotations whose frame depends on
    the header: legacy ns/nl or as/al arrays, or MA groups other than m5C
    (always molecular). fibertools ``Ma`` is always molecular."""
    try:
        for start_tag, length_tag in (("ns", "nl"), ("as", "al")):
            if read.has_tag(start_tag) and read.has_tag(length_tag):
                value = read.get_tag(start_tag)
                if hasattr(value, "__len__") and len(value):
                    return True
        if read.has_tag("MA"):
            for group in str(read.get_tag("MA")).split(";")[1:]:
                head, _, body = group.partition(":")
                match = _MA_GROUP_HEAD_RE.match(head)
                if match and match.group(1) not in M5C_GROUPS and body.strip(","):
                    return True
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return False


def warn_if_frames_unverifiable(path, header=None, *, scan_records: int = 5000,
                                stream=None) -> bool:
    """Print :data:`UNVERIFIABLE_FRAME_WARNING` once per input file (per
    process) when its header gives no frame evidence and its first
    ``scan_records`` records carry frame-dependent footprint annotations.

    Only a warning: nothing is changed. A header with any frame evidence, a
    file without such annotations (raw aligned reads), stdin (``'-'``, which
    cannot be peeked) and unreadable paths are silent. Returns whether it
    warned."""
    import os
    import sys

    if path in (None, "-") or not isinstance(path, (str, os.PathLike)):
        return False
    try:
        key = os.path.realpath(os.fspath(path))
    except (TypeError, ValueError):
        return False
    if key in _WARNED_PATHS or key in _CHECKED_PATHS or not os.path.isfile(key):
        return False
    _CHECKED_PATHS.add(key)       # each file is scanned at most once per process
    try:
        from pysam.libcalignmentfile import AlignmentFile

        with AlignmentFile(key, check_sq=False) as bam:
            if header_has_frame_evidence(header if header is not None else bam.header):
                return False
            found = False
            for index, read in enumerate(bam.fetch(until_eof=True)):
                if index >= scan_records:
                    break
                if read_has_frame_dependent_annotations(read):
                    found = True
                    break
    except (OSError, ValueError):
        return False
    if not found:
        return False
    _WARNED_PATHS.add(key)
    print(UNVERIFIABLE_FRAME_WARNING.format(path=os.fspath(path)),
          file=stream if stream is not None else sys.stderr)
    return True
