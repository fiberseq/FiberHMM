"""Which FiberHMM outputs need re-running after a fix.

FiberHMM outputs record how they were made: BAM ``@PG`` records (program,
version, command line) and, since 3.0, a ``FIBERHMM-CHEMISTRY`` declaration per
run with the digests of the emission tables it read, the FiberHMM version and
commit (see :func:`fiberhmm.cli.provenance.chemistry_declaration`). QC
reports, posteriors files and consensus results carry their own metadata.
This module compares that record with ``fiberhmm/advisories.json`` — the list
of released changes that alter results, the digests of every historical
emission table, and which commits contain each fix — and reports what applies.

Public API (stable; used by FiberBrowser's Library)::

    from fiberhmm.advisories import check_path, check_bam, check_header, report

    advisories = check_bam("calls.bam")          # list[Advisory]
    payload = report("calls.bam")                 # JSON-ready dict

Evidence, strongest first: a recorded table digest (decides table advisories
exactly); a recorded commit or one named in the path of the program the
``@PG CL`` ran (e.g. a ``frozen_<sha>/fiberhmm/cli/call.py`` source tree; input,
output and model paths never count), checked against the commits that contain
the fix; the reported version. A version shared by builds on both sides of a fix
(development snapshots reported 2.16.8 while already containing 3.0 fixes)
gives ``possibly_affected`` with low confidence, never ``affected``.
"""
from __future__ import annotations

import functools
import gzip
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

INDEX_SCHEMA = "fiberhmm.advisories.v1"
REPORT_SCHEMA = "fiberhmm.advisory_report.v1"
SEVERITIES = ("rerun-required", "rerun-recommended", "info")
DEFAULT_SCAN_RECORDS = 5000

AFFECTED = "affected"
POSSIBLY = "possibly_affected"
_NOT = "not_affected"
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

# Programs whose output replaces all footprint calls, and later passes that
# re-call part of them on the same molecules.
_CALLING = {"fiberhmm-call", "fiberhmm-apply", "fiberhmm-run"}
_RECALLING = {"fiberhmm-recall-tfs", "fiberhmm-recall-nucs", "fiberhmm-merge"}
_CALL_PRODUCERS = _CALLING | _RECALLING | {"fiberhmm-pair"}
_PROGRAM_RE = re.compile(r"^(fiberhmm-[a-z0-9-]+?)(?:\.\d+)?(?:-[0-9A-F]{8})?$")
_HEX_TOKEN_RE = re.compile(r"(?<![0-9A-Za-z])([0-9a-f]{7,40})(?![0-9A-Za-z])")
_DS_FIELD_RE = r"(?:^|[\s;(]){key}=([A-Za-z0-9_.+/-]+)"
_BUNDLED_MODEL_TOOLS = {"ddda_nuc", "ddda_TF", "dddb_nanopore", "hia5_nanopore",
                        "hia5_pacbio", "ecogii_pacbio", "cpg_nanopore"}


@dataclass(frozen=True)
class Advisory:
    """One advisory that applies to one output.

    ``status`` is ``"affected"`` (the evidence shows the output was made
    before the fix) or ``"possibly_affected"`` (the evidence cannot tell);
    outputs the evidence clears are not reported. ``evidence`` says what was
    matched, in words. ``program`` is the ``@PG`` ID the advisory was matched
    on (None for file-level checks); ``path`` the file it concerns (a QC
    sidecar of a BAM, for instance), when known.
    """

    id: str
    severity: str
    status: str
    confidence: str
    title: str
    reason: str
    artifact: str
    fix: str
    fixed_in: str
    evidence: tuple[str, ...] = ()
    program: Optional[str] = None
    path: Optional[str] = None

    @property
    def needs_rerun(self) -> bool:
        return self.severity != "info"

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["evidence"] = list(self.evidence)
        out["needs_rerun"] = self.needs_rerun
        return out


class AdvisoryInputError(ValueError):
    """The path is missing, unreadable or not an output this module knows."""


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_index() -> dict:
    """The bundled advisory index (``fiberhmm/advisories.json``)."""
    path = Path(__file__).with_name("advisories.json")
    index = json.loads(path.read_text())
    if index.get("schema") != INDEX_SCHEMA:
        raise ValueError(f"{path}: unsupported advisory index schema {index.get('schema')!r}")
    return index


class _Index:
    def __init__(self, data: dict):
        self.data = data
        self.rules = {rule["id"]: rule for rule in data["advisories"]}
        self.commits: dict[str, dict] = data.get("commits", {})
        self.tables: list[dict] = data.get("tables", [])

    def commit(self, token: str) -> tuple[Optional[str], Optional[dict]]:
        token = token.lower()
        if token in self.commits:
            return token, self.commits[token]
        if len(token) >= 7:
            matches = [sha for sha in self.commits if sha.startswith(token)]
            if len(matches) == 1:
                return matches[0], self.commits[matches[0]]
        return None, None

    def contains(self, commit_entry: dict, fix: str) -> bool:
        return fix in commit_entry.get("contains", ())

    def versions_with_fix(self, fix: str) -> set[str]:
        return {entry["version"] for entry in self.commits.values()
                if entry.get("version") and fix in entry.get("contains", ())}

    def table(self, digest: str) -> Optional[dict]:
        for entry in self.tables:
            if entry["sha256"] == digest:
                return entry
        return None


def _index(index: Optional[dict]) -> _Index:
    return _Index(index if index is not None else load_index())


def _version_tuple(value) -> Optional[tuple[int, ...]]:
    match = re.match(r"^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?", str(value or ""))
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


def _short(sha: str) -> str:
    return sha[:7]


# ---------------------------------------------------------------------------
# Header model
# ---------------------------------------------------------------------------

@dataclass
class _Run:
    position: int
    id: str
    program: str
    version: Optional[str]
    cl: str
    ds: str
    declaration: Optional[dict] = None
    chemistry: dict = field(default_factory=dict)
    note: Optional[str] = None  # evidence for a stand-in run (unrecorded history)


def _header_dict(header) -> dict:
    if isinstance(header, str):
        import pysam

        return pysam.AlignmentHeader.from_text(header).to_dict()
    if hasattr(header, "to_dict"):
        return header.to_dict()
    return dict(header)


def _program_name(record: dict) -> Optional[str]:
    for candidate in (record.get("PN"), record.get("ID")):
        match = _PROGRAM_RE.match(str(candidate or ""))
        if match:
            return match.group(1)
    return None


def _ds_field(ds: str, key: str) -> Optional[str]:
    match = re.search(_DS_FIELD_RE.format(key=re.escape(key)), ds)
    return match.group(1) if match else None


def _cl_tokens(cl: str) -> list[str]:
    return cl.split()


def _cl_has(cl: str, flags: Iterable[str]) -> bool:
    tokens = _cl_tokens(cl)
    for flag in flags:
        for token in tokens:
            if token == flag or token.startswith(flag + "="):
                return True
    return False


def _cl_value(cl: str, flags: Iterable[str]) -> Optional[str]:
    """Value of the first of ``flags`` in a space-joined argv (paths may hold spaces)."""
    for flag in flags:
        match = re.search(
            rf"(?:^|\s){re.escape(flag)}(?:=|\s+)(.+?)(?=\s+--?[A-Za-z]|\s*$)", cl)
        if match:
            return match.group(1).strip()
    return None


_EXECUTABLE_RE = re.compile(r"(?:/fiberhmm/cli/[A-Za-z0-9_]+\.py|(?:^|/)fiberhmm-[a-z0-9-]+)$")


def _executable_path(cl: str) -> Optional[str]:
    """The program path a FiberHMM ``@PG CL`` starts with, or None.

    FiberHMM records ``' '.join(sys.argv)``: its first word is the script
    that ran (``.../fiberhmm/cli/call.py`` for ``python -m`` or a source
    tree, ``.../bin/fiberhmm-call`` for an installed command). Paths may hold
    spaces, so the program is the words before the first option that end in
    such a script name. Input, output and model paths are never taken as
    the program.
    """
    words = str(cl or "").split(" ")
    for end in range(1, len(words) + 1):
        if words[end - 1].startswith("-") and end > 1:
            return None
        candidate = " ".join(words[:end]).strip()
        if candidate and _EXECUTABLE_RE.search(candidate):
            return candidate
    return None


def _model_stem(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    name = os.path.basename(value.rstrip("/"))
    return re.sub(r"\.json$", "", name) or None


def _runs(header: dict, declarations: Optional[dict] = None) -> list[_Run]:
    """FiberHMM runs of a header, each with the declaration ``declarations``
    (``{@PG position: declaration}``) assigns it."""
    declarations = declarations or {}
    runs = []
    for position, record in enumerate(_pg_records(header)):
        program = _program_name(record)
        if program is None:
            continue
        version = str(record.get("VN") or "") or None
        if version in (None, "unknown"):
            version = None
        run = _Run(position=position, id=str(record.get("ID", program)), program=program,
                   version=version, cl=str(record.get("CL", "")), ds=str(record.get("DS", "")),
                   declaration=declarations.get(position))
        run.chemistry = _run_chemistry(run)
        runs.append(run)
    return runs


def _run_chemistry(run: _Run) -> dict:
    """Best-effort chemistry of one run: its linked declaration, else DS/CL."""
    declared = dict(run.declaration or {})
    chemistry: dict[str, Any] = {}
    mode = declared.get("mode") or _ds_field(run.ds, "mode")
    enzyme = declared.get("enzyme")
    if not enzyme or enzyme == "custom":
        enzyme = _ds_field(run.ds, "enzyme") or _cl_value(run.cl, ["--enzyme"]) or enzyme
    platform = declared.get("platform")
    if not platform or platform == "unknown":
        platform = _cl_value(run.cl, ["--seq"])
    if not platform and mode == "nanopore-fiber":
        platform = "nanopore"
    elif not platform and mode == "pacbio-fiber":
        platform = "pacbio"
    model = declared.get("model")
    custom = _cl_value(run.cl, ["-m", "--model", "--recall-model"])
    if not model and custom:
        model = _model_stem(custom)
    chemistry.update(mode=(mode or "").lower() or None,
                     enzyme=(enzyme or "").lower() or None,
                     platform=(platform or "").lower() or None,
                     model=model, custom_model=bool(custom))
    for key in ("apply_sha256", "recall_sha256", "fiberhmm_version", "fiberhmm_commit"):
        if declared.get(key):
            chemistry[key] = declared[key]
    return chemistry


# ---------------------------------------------------------------------------
# @PG history: which calls a file holds
# ---------------------------------------------------------------------------
#
# ``@PG`` records form chains through ``PP`` (previous program). ``samtools
# merge`` keeps every input's chain, renames IDs that clash with a
# ``-XXXXXXXX`` suffix, and appends one merge record per chain end (same
# PN/VN/CL, one PP each). FiberHMM links a new record to the last one only. A
# history is therefore a DAG: a merge record joins its chains, and a run is
# superseded only by a full call that descends from it. Every other branch's
# calls are still in the file.
#
# Chemistry declarations (``@CO FIBERHMM-CHEMISTRY``) name their run's @PG ID
# in ``pg``; samtools renames the @PG but not the comment, so after a merge
# several declarations can name one ID. When attribution or the history is
# ambiguous every plausible reading is checked: an advisory that holds in all
# of them stands; one that holds in some is reported as possibly affected.
# Where the plausible readings are too many to list, one upper-bound reading
# (every run with every declaration it could carry) decides, and nothing it
# finds can be more than possibly affected.
#
# Malformed links never prove that a call was replaced: PP links that point
# forward (to a later record; they are how a cycle arises) or to an ID that
# several records carry are ambiguous, so readings with and without them are
# both checked, and runs on a cycle never supersede one another.
#
# A step that drops its inputs' headers (``samtools cat`` of several files,
# Picard GatherBamFiles) leaves reads whose calling history is not in the
# header. Unless a later full call re-called them, each current calling run
# gets a stand-in with unrecorded provenance, which can only be possibly
# affected; the file is never clean on the strength of the first input alone.

_MERGE_SUFFIX_RE = re.compile(r"(?:-[0-9A-F]{8})+$")
_MAX_SCENARIOS = 64
_MAX_OPEN_DECLARATIONS = 10


def _pg_records(header: dict) -> list[dict]:
    records = header.get("PG", []) or []
    return [record for record in records if isinstance(record, dict)]


def _is_merge_record(record: dict) -> bool:
    command = str(record.get("CL", ""))
    return bool(re.search(r"(?:^|[\s/])samtools\s+merge(?:\s|$)", command))


# samtools cat's getopt (1.21): short options with a value, short flags, and
# long options (getopt_long: "--name=value" or "--name value", unique prefixes).
_CAT_SHORT_VALUE = frozenset("hob@rp")
_CAT_SHORT_FLAG = frozenset("fq")
_CAT_LONG = {"no-PG": False, "threads": True, "output-fmt": True,
             "output-fmt-option": True, "input-fmt-option": True, "reference": True,
             "verbosity": True, "write-index": False}
# fiberhmm-call joining its own region BAMs (bam_output._samtools_cat_bams);
# matched on the whole command so directory names may contain spaces.
_FIBERHMM_REGION_CAT_RE = re.compile(
    r"samtools\s+cat\s+-h\s+(?P<dir>.+?/\.[^/]*fiberhmm[^/]*)/region_\d+\.bam\s+"
    r"-b\s+(?P=dir)/[^/]*bam_list[^/]*\.txt\s+-o\s+\S.*$")


def _cat_arguments(tokens: list[str]) -> tuple[list[str], list[str], bool]:
    """``(inputs, list_files, unknown)`` of a ``samtools cat`` argument list,
    parsed the way its getopt does: attached (``-bFILE``) and separate
    (``-b FILE``) values, clustered flags (``-fq``, ``-fbFILE``), long options
    with ``=`` or a separate value and unique prefixes, ``--``, and ``-``
    (stdin) as an input. ``unknown`` is set for anything it cannot parse."""
    inputs: list[str] = []
    lists: list[str] = []
    unknown = False
    options_done = False
    i = 0
    while i < len(tokens):
        token = tokens[i]
        i += 1
        if options_done or token == "-" or not token.startswith("-"):
            inputs.append(token)
            continue
        if token == "--":
            options_done = True
            continue
        if token.startswith("--"):
            name, equals, _value = token[2:].partition("=")
            matches = [k for k in _CAT_LONG if k == name] or \
                      [k for k in _CAT_LONG if k.startswith(name)]
            if len(matches) != 1:
                unknown = True
                continue
            if _CAT_LONG[matches[0]] and not equals:
                i += 1  # its value is the next word
            continue
        j = 1
        while j < len(token):
            flag = token[j]
            if flag in _CAT_SHORT_VALUE:
                value = token[j + 1:]
                if not value:
                    value = tokens[i] if i < len(tokens) else ""
                    i += 1
                if flag == "b":
                    lists.append(value)
                break
            if flag not in _CAT_SHORT_FLAG:
                unknown = True
                break
            j += 1
    return inputs, lists, unknown


def _drops_input_headers(record: dict) -> bool:
    """Whether a @PG step may have joined several inputs but kept one header.

    ``samtools cat`` of two or more inputs, or of any file list (it can name
    any number of files), and Picard GatherBamFiles. A ``samtools cat``
    command line that cannot be parsed counts too; fiberhmm-call's own
    concatenation of its region BAMs does not.
    """
    command = str(record.get("CL", ""))
    name = str(record.get("PN") or record.get("ID") or "")
    if re.search(r"GatherBamFiles", f"{name} {command}", re.IGNORECASE):
        return True
    match = re.search(r"(?:^|[\s/])samtools\s+cat(?:\s|$)(.*)", command)
    if not match:
        return False
    if _FIBERHMM_REGION_CAT_RE.search(command):
        return False
    inputs, lists, unknown = _cat_arguments(match.group(1).split())
    return unknown or bool(lists) or len(inputs) >= 2


@dataclass
class _Graph:
    parents: list[set[int]]
    notes: list[str]


def _history_graphs(records: list[dict]) -> tuple[list[_Graph], list[int]]:
    """Parent sets per @PG node for the plausible readings of the history.

    Returns ``(graphs, node)``: ``node[i]`` is the node of record ``i``
    (records of one merge event share a node). The first graph keeps only
    unambiguous links; a second one, when anything is ambiguous, adds the
    ambiguous links (PP-less records continuing the record before them, PP
    links to a duplicated ID -- to every record carrying it -- and forward PP
    links), with notes saying what was ambiguous.
    """
    by_id: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        by_id.setdefault(str(record.get("ID", "")), []).append(index)
    # One node per htslib merge event: consecutive records with the same
    # program and command line, each linked to a different chain end.
    node = list(range(len(records)))
    for index in range(1, len(records)):
        record, before = records[index], records[index - 1]
        key = tuple(str(record.get(k, "")) for k in ("PN", "VN", "CL", "DS"))
        if (record.get("PP") and before.get("PP")
                and key == tuple(str(before.get(k, "")) for k in ("PN", "VN", "CL", "DS"))
                and record.get("PP") != before.get("ID")):
            node[index] = node[index - 1]
    strong: list[set[int]] = [set() for _ in records]
    weak: list[set[int]] = [set() for _ in records]
    notes: list[str] = []
    unlinked = []
    for index, record in enumerate(records):
        previous = record.get("PP")
        if previous is None:
            if index > 0 and node[index] == index:
                unlinked.append(index)
            continue
        targets = by_id.get(str(previous), [])
        if not targets:
            continue  # links to nothing recorded: a root of unknown history
        parents = {node[t] for t in targets} - {node[index]}
        if len(targets) > 1:
            weak[node[index]] |= parents
            notes.append(f"several @PG records carry the ID {previous}, so the parent of "
                         f"@PG {record.get('ID')} is ambiguous")
        elif targets[0] >= index:
            weak[node[index]] |= parents
            notes.append(f"@PG {record.get('ID')} links forward to {previous} (PP links "
                         "that point forward or form a cycle cannot order the history)")
        else:
            strong[node[index]] |= parents
    merged_later = [i for i, record in enumerate(records) if _is_merge_record(record)]
    ambiguous = [i for i in unlinked
                 if not _MERGE_SUFFIX_RE.search(str(records[i].get("ID", "")))
                 and not any(m > i for m in merged_later)]
    if ambiguous:
        for index in ambiguous:
            weak[index].add(node[index - 1])
        names = ", ".join(str(records[i].get("ID")) for i in ambiguous[:3])
        notes.append(f"@PG {names} has no PP link, so the header cannot tell whether it "
                     "continued the history before it or was merged in beside it")
    graphs = [_Graph(strong, [])]
    if any(weak):
        graphs.append(_Graph([a | b for a, b in zip(strong, weak)], []))
        graphs[0].notes = graphs[1].notes = list(dict.fromkeys(notes))
    return graphs, node


def _components(parents: list[set[int]]) -> list[int]:
    """Strongly connected component of every node (iterative Tarjan)."""
    n = len(parents)
    index_of = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    component = [-1] * n
    stack: list[int] = []
    counter = 0
    for root in range(n):
        if index_of[root] != -1:
            continue
        work = [(root, iter(parents[root]))]
        index_of[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack[root] = True
        while work:
            vertex, edges = work[-1]
            advanced = False
            for nxt in edges:
                if index_of[nxt] == -1:
                    index_of[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack[nxt] = True
                    work.append((nxt, iter(parents[nxt])))
                    advanced = True
                    break
                if on_stack[nxt]:
                    low[vertex] = min(low[vertex], index_of[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[vertex])
            if low[vertex] == index_of[vertex]:
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component[member] = vertex
                    if member == vertex:
                        break
    return component


def _ancestors(parents: list[set[int]], start: int) -> set[int]:
    seen: set[int] = set()
    stack = list(parents[start])
    while stack:
        item = stack.pop()
        if item in seen:
            continue
        seen.add(item)
        stack.extend(parents[item])
    return seen


def _current_runs(runs: list[_Run], parents: list[set[int]],
                  node: list[int]) -> list[_Run]:
    """The runs whose calls the file holds: every calling or recalling run
    that no later full call descends from (one per merged branch). Runs on
    one PP cycle do not supersede each other."""
    producers = [run for run in runs if run.program in _CALLING | _RECALLING]
    component = _components(parents)
    superseded: set[int] = set()
    for call in producers:
        if call.program not in _CALLING:
            continue
        own = component[node[call.position]]
        superseded |= {a for a in _ancestors(parents, node[call.position])
                       if component[a] != own}
    return [run for run in producers if node[run.position] not in superseded]


def _unrecorded_inputs(records: list[dict], runs: list[_Run], current: list[_Run],
                       parents: list[set[int]], node: list[int]) -> list[_Run]:
    """Stand-in runs for calls whose history a header-dropping step discarded."""
    calls = [run for run in runs if run.program in _CALLING]
    stand_ins: list[_Run] = []
    templates = [run for run in current if run.program in _CALLING | _RECALLING]
    for index, record in enumerate(records):
        if not _drops_input_headers(record):
            continue
        step = node[index]
        if any(step in _ancestors(parents, node[call.position]) for call in calls):
            continue  # a later full call re-called every read
        note = (f"@PG {record.get('ID')} ({str(record.get('CL', ''))[:80]}) joined several "
                "inputs but kept one header: the other inputs' calls are in the file with "
                "no recorded provenance, so they cannot be cleared")
        for template in templates:
            chemistry = {key: template.chemistry.get(key)
                         for key in ("mode", "enzyme", "platform")}
            chemistry.update(model=None, custom_model=False)
            stand_ins.append(_Run(
                position=template.position,
                id=f"{template.id} (unrecorded inputs of {record.get('ID')})",
                program=template.program, version=None, cl=template.cl, ds=template.ds,
                chemistry=chemistry, note=note))
    return stand_ins


def _declaration_candidates(runs: list[_Run], header: dict) -> tuple[list[dict], list[list[int]]]:
    """Every chemistry declaration naming a run (duplicates kept) and the
    @PG positions it could belong to (its ``pg`` ID, or that ID renamed by
    ``samtools merge``)."""
    from fiberhmm.io.bam_header import _parse_chemistry_comment

    declarations, candidates = [], []
    for comment in header.get("CO", []) or []:
        parsed = _parse_chemistry_comment(str(comment))
        if not parsed or not parsed.get("pg"):
            continue
        pg = parsed["pg"]
        matches = [run.position for run in runs
                   if run.id == pg or (run.id.startswith(pg)
                                       and _MERGE_SUFFIX_RE.fullmatch(run.id[len(pg):]))]
        if matches:
            declarations.append(parsed)
            candidates.append(matches)
    return declarations, candidates


@dataclass
class _Attribution:
    fixed: dict                      # position -> declaration, unambiguous
    options: dict                    # position -> [declaration or None], ambiguous runs
    assignments: list                # every plausible assignment, when complete
    complete: bool


def _assignments(declarations: list[dict], candidates: list[list[int]]) -> _Attribution:
    """Plausible ``{@PG position: declaration}`` assignments (each run has at
    most one declaration; as many declarations are placed as possible). When
    they are too many to list, ``complete`` is False and ``options`` gives
    every declaration each ambiguous run could carry."""
    fixed: dict[int, dict] = {}
    open_items = []
    claims: dict[int, int] = {}
    for options in candidates:
        for position in options:
            claims[position] = claims.get(position, 0) + 1
    for declaration, options in zip(declarations, candidates):
        if len(options) == 1 and claims[options[0]] == 1:
            fixed[options[0]] = declaration
        else:
            open_items.append((declaration, options))
    choice: dict[int, list] = {}
    for declaration, options in open_items:
        for position in options:
            choice.setdefault(position, [None])
            if declaration not in choice[position]:
                choice[position].append(declaration)
    if not open_items:
        return _Attribution(fixed, {}, [fixed], True)
    results: list[dict[int, dict]] = []
    seen: set = set()
    best = 0
    truncated = len(open_items) > _MAX_OPEN_DECLARATIONS
    budget = [_MAX_SCENARIOS * 50]

    def place(i: int, chosen: dict[int, dict]) -> None:
        nonlocal best, truncated
        if truncated:
            return
        budget[0] -= 1
        if budget[0] < 0:
            truncated = True
            return
        if i == len(open_items):
            size = len(chosen)
            key = tuple(sorted((pos, tuple(sorted(d.items()))) for pos, d in chosen.items()))
            if size < best or key in seen:
                return
            if size > best:
                best = size
                results.clear()
                seen.clear()
            seen.add(key)
            results.append(dict(chosen))
            if len(results) > _MAX_SCENARIOS:
                truncated = True
            return
        declaration, options = open_items[i]
        for position in options:
            if position in chosen or position in fixed:
                continue
            chosen[position] = declaration
            place(i + 1, chosen)
            del chosen[position]
        place(i + 1, chosen)

    if not truncated:
        place(0, {})
    if truncated:
        return _Attribution(fixed, choice, [], False)
    return _Attribution(fixed, choice, [{**fixed, **chosen} for chosen in results], True)


@dataclass
class _Reading:
    runs: list[_Run]
    current: list[_Run]
    notes: list[str]
    upper_bound: bool = False   # findings here are at most possibly affected


def _readings(header: dict) -> list[_Reading]:
    """Every plausible reading of a header's history and declarations."""
    records = _pg_records(header)
    graphs, node = _history_graphs(records)
    bare = _runs(header)
    declarations, candidates = _declaration_candidates(bare, header)
    attribution = _assignments(declarations, candidates)
    attribution_note = None
    if attribution.options:
        claims: dict[int, int] = {}
        for options in candidates:
            for position in options:
                claims[position] = claims.get(position, 0) + 1
        named = sorted({d["pg"] for d, options in zip(declarations, candidates)
                        if len(options) > 1 or any(claims[o] > 1 for o in options)})
        attribution_note = (
            "chemistry declarations for @PG " + ", ".join(named[:5]) + " cannot be matched "
            "to their runs (samtools merge renamed the @PG IDs but not the declarations)")
        if not attribution.complete:
            attribution_note += ("; too many pairings to check one by one, so every run "
                                 "was checked with every declaration it could carry")
    readings = []
    for graph in graphs:
        notes = [n for n in (*graph.notes, attribution_note) if n]
        if attribution.complete:
            for assignment in attribution.assignments:
                runs = _runs(header, assignment)
                current = _current_runs(runs, graph.parents, node)
                current = current + _unrecorded_inputs(records, runs, current,
                                                       graph.parents, node)
                readings.append(_Reading(runs, current, notes))
        else:
            # Upper bound: each ambiguous run once per declaration it could carry.
            base = _runs(header, attribution.fixed)
            runs = []
            for run in base:
                if run.position not in attribution.options:
                    runs.append(run)
                    continue
                for declaration in attribution.options[run.position]:
                    variant = _Run(run.position, run.id, run.program, run.version, run.cl,
                                   run.ds, declaration=declaration)
                    variant.chemistry = _run_chemistry(variant)
                    runs.append(variant)
            current = _current_runs(runs, graph.parents, node)
            current = current + _unrecorded_inputs(records, runs, current, graph.parents, node)
            readings.append(_Reading(runs, current, notes, upper_bound=True))
    return readings


@dataclass
class _Finding:
    status: str
    confidence: str
    evidence: list[str]


def _fix_commit(rule: dict, program: Optional[str] = None) -> Optional[str]:
    fixed = rule.get("fixed_in", {})
    return fixed.get("by_program", {}).get(program or "", fixed.get("commit"))


def _code_finding(index: _Index, rule: dict, run: _Run) -> _Finding:
    """Did the code of ``run`` contain the rule's fix?"""
    fix = _fix_commit(rule, run.program)
    fixed_version = rule.get("fixed_in", {}).get("version")
    where = f"@PG {run.id}"
    noun = "change" if rule.get("severity") == "info" else "fix"
    declared = run.chemistry.get("fiberhmm_commit")
    if declared and fix:
        sha, entry = index.commit(declared.split("+")[0])
        if entry is not None:
            dirty = declared.endswith("+dirty")
            has_fix = index.contains(entry, fix)
            note = " (with uncommitted changes)" if dirty else ""
            return _Finding(
                _NOT if has_fix else AFFECTED, "medium" if dirty else "high",
                [f"{where} declares commit {_short(sha)}{note}, which "
                 f"{'contains' if has_fix else 'predates'} the {noun} ({_short(fix)})"])
    executable = _executable_path(run.cl)
    if fix and executable:
        verdicts = {}
        for token in _HEX_TOKEN_RE.findall(os.path.dirname(executable)):
            sha, entry = index.commit(token)
            if entry is not None:
                verdicts[sha] = index.contains(entry, fix)
        if verdicts and len(set(verdicts.values())) == 1:
            has_fix = next(iter(verdicts.values()))
            named = ", ".join(_short(sha) for sha in verdicts)
            return _Finding(
                _NOT if has_fix else AFFECTED, "medium",
                [f"{where} command line runs code from a tree named after commit "
                 f"{named}, which {'contains' if has_fix else 'predates'} the {noun} "
                 f"({_short(fix)})"])
    version = run.version or run.chemistry.get("fiberhmm_version")
    reported = _version_tuple(version)
    if reported is None:
        return _Finding(POSSIBLY, "low", [f"{where} records no FiberHMM version"])
    if fixed_version and reported >= _version_tuple(fixed_version):
        return _Finding(_NOT, "medium", [f"{where} version {version} includes the {noun}"])
    if fix and version in index.versions_with_fix(fix):
        return _Finding(
            POSSIBLY, "low",
            [f"{where} reports version {version}, which both the release and later "
             f"development builds containing the {noun} ({_short(fix)}) reported; the "
             "header names no commit or table digest to tell them apart"])
    return _Finding(AFFECTED, "high",
                    [f"{where} version {version} predates the {noun} (in {fixed_version})"])


def _table_finding(index: _Index, rule: dict, run: _Run) -> Optional[_Finding]:
    table = rule["match"]["table"]
    where = f"@PG {run.id}"
    digests = {role: run.chemistry.get(f"{role}_sha256") for role in ("apply", "recall")}
    digests = {role: value for role, value in digests.items() if value}
    if digests:
        swapped = []
        for role, digest in digests.items():
            entry = index.table(digest)
            if entry and entry["table"] == table and entry["status"] == "gt_swapped":
                releases = entry.get("releases") or []
                shipped = (f"shipped in {releases[0]}-{releases[-1]}" if len(releases) > 1
                           else f"shipped in {releases[0]}" if releases else "a development copy")
                swapped.append(f"{where} {role} table sha256 {digest[:12]} is the "
                               f"context-swapped {table} table ({shipped})")
        if swapped:
            return _Finding(AFFECTED, "high", swapped)
        return _Finding(_NOT, "high", [f"{where} table digests are not a swapped {table} table"])
    model = run.chemistry.get("model")
    bundled = set(rule["match"].get("bundled_models", ()))
    if model and model.endswith("_gt_swapped_legacy"):
        return _Finding(AFFECTED, "high", [f"{where} used the legacy table {model}"])
    if model and run.chemistry.get("custom_model") and model not in bundled:
        return _Finding(POSSIBLY, "low", [
            f"{where} used a custom table ({model}) and records no table digest, so its "
            "context order cannot be checked"])
    return _code_finding(index, rule, run)


_ONT_PROGRAMS = ("dorado", "guppy", "basecaller", "minknow", "bonito")
_PACBIO_PROGRAMS = ("ccs", "pbmm2", "jasmine", "primrose", "fibertools-predict", "lima")


def sequencing_platform(header: dict) -> tuple[Optional[str], Optional[str]]:
    """(``nanopore``/``pacbio``, evidence) from @RG PL and upstream @PG records.

    Only unambiguous evidence counts: both platforms seen gives (None, None).
    """
    seen: dict[str, str] = {}
    for group in header.get("RG", []):
        platform = str(group.get("PL", "")).upper()
        if platform in ("ONT", "OXFORD_NANOPORE", "NANOPORE"):
            seen.setdefault("nanopore", f"@RG PL:{group.get('PL')}")
        elif platform == "PACBIO":
            seen.setdefault("pacbio", f"@RG PL:{group.get('PL')}")
    for program in header.get("PG", []):
        name = str(program.get("PN") or program.get("ID") or "").lower()
        command = str(program.get("CL", ""))
        if any(tool in name for tool in _ONT_PROGRAMS) or re.search(r"-x\s*map-ont|-ax\s*map-ont|lr:hq", command):
            seen.setdefault("nanopore", f"@PG {program.get('ID')} ({name or 'aligner'} "
                                        f"{'map-ont' if 'map-ont' in command else 'Nanopore'})")
        elif any(name.startswith(tool) for tool in _PACBIO_PROGRAMS) or re.search(r"map-pb|map-hifi", command):
            seen.setdefault("pacbio", f"@PG {program.get('ID')} ({name})")
    if len(seen) == 1:
        platform, evidence = next(iter(seen.items()))
        return platform, evidence
    return None, None


def _matches_run(rule: dict, run: _Run, runs: list[_Run],
                 header_platform: Optional[str] = None) -> bool:
    match = rule.get("match", {})
    if match.get("header_platform") and match["header_platform"] != header_platform:
        return False
    if match.get("programs") and run.program not in match["programs"]:
        return False
    chemistry = run.chemistry
    for key in ("mode", "enzyme", "platform"):
        wanted = match.get(key)
        if wanted:
            wanted = [wanted] if isinstance(wanted, str) else wanted
            if chemistry.get(key) not in wanted:
                return False
    if match.get("cl_has_any") and not _cl_has(run.cl, match["cl_has_any"]):
        return False
    if match.get("cl_lacks_all") and _cl_has(run.cl, match["cl_lacks_all"]):
        return False
    if any(token in run.ds for token in match.get("ds_lacks_all", ())):
        return False
    if match.get("inherited_enzyme"):
        if _inherited_enzyme(run, runs) not in match["inherited_enzyme"]:
            return False
    return True


def _inherited_enzyme(run: _Run, runs: list[_Run]) -> Optional[str]:
    """Enzyme the input of ``run`` declared (its own inherited one, else earlier runs')."""
    declared = (run.declaration or {}).get("enzyme")
    if declared and declared != "custom":
        return declared
    for earlier in reversed([r for r in runs if r.position < run.position]):
        enzyme = earlier.chemistry.get("enzyme")
        if enzyme and enzyme != "custom":
            return enzyme
    return None


def _make(rule: dict, finding: _Finding, *, program=None, path=None) -> Advisory:
    return Advisory(
        id=rule["id"], severity=rule["severity"], status=finding.status,
        confidence=finding.confidence, title=rule["title"], reason=rule["reason"],
        artifact=rule["artifact"], fix=rule["fix"],
        fixed_in=rule.get("fixed_in", {}).get("version", ""),
        evidence=tuple(finding.evidence), program=program, path=path)


def _combine(findings: list[_Finding]) -> Optional[_Finding]:
    """Worst finding across sources: affected > possibly; best confidence within."""
    for status in (AFFECTED, POSSIBLY):
        chosen = [f for f in findings if f.status == status]
        if chosen:
            confidence = max((f.confidence for f in chosen), key=_CONFIDENCE_RANK.get)
            evidence = [line for f in chosen for line in f.evidence]
            return _Finding(status, confidence, evidence)
    return None


# ---------------------------------------------------------------------------
# Header / BAM checks
# ---------------------------------------------------------------------------

@dataclass
class ReadScan:
    """What a bounded scan of the first records found (see :func:`scan_reads`)."""

    records: int = 0
    dedup_tags: int = 0
    pair_tags: int = 0
    paired_duplicates: int = 0
    call_tags: int = 0


def scan_reads(bam, max_records: int = DEFAULT_SCAN_RECORDS) -> ReadScan:
    """Scan up to ``max_records`` records of an open ``pysam.AlignmentFile``."""
    scan = ReadScan()
    if max_records <= 0:
        return scan
    for read in bam.fetch(until_eof=True):
        scan.records += 1
        if read.has_tag("di"):
            scan.dedup_tags += 1
        if read.has_tag("mt"):
            scan.pair_tags += 1
            if read.is_duplicate and read.has_tag("mp"):
                scan.paired_duplicates += 1
        if read.has_tag("ns") or read.has_tag("as") or (
                read.has_tag("MA") and re.search(r";(nuc|msp|tf)[.+-]", str(read.get_tag("MA")))):
            scan.call_tags += 1
        if scan.records >= max_records:
            break
    return scan


def check_header(header, *, scan: Optional[ReadScan] = None, index: Optional[dict] = None,
                 path: Optional[str] = None) -> list[Advisory]:
    """Advisories for a BAM header (pysam header, header dict, or SAM header text).

    ``scan`` adds evidence only reads carry (dedup/pair tags, calls without
    provenance); :func:`check_bam` supplies it.
    """
    idx = _index(index)
    data = _header_dict(header)
    platform = sequencing_platform(data)
    readings = _readings(data)
    results = [_check_reading(idx, data, reading, scan, path, platform) for reading in readings]
    if len(results) == 1 and not readings[0].upper_bound:
        return results[0]
    notes = list(dict.fromkeys(note for reading in readings for note in reading.notes))
    return _merge_readings(idx, results, notes,
                           upper_bound=any(r.upper_bound for r in readings))


def _check_reading(idx, data, reading: _Reading, scan, path, platform) -> list[Advisory]:
    runs, current = reading.runs, reading.current
    chemistries = {(run.chemistry.get("enzyme"), run.chemistry.get("platform"))
                   for run in current if run.program in _CALLING}
    mixed = len(chemistries) > 1
    found: list[Advisory] = []
    for rule in idx.data["advisories"]:
        detector = rule["detector"]
        per_run: list[Advisory] = []
        if detector == "table":
            per_run = _check_table(idx, rule, current, path)
        elif detector == "run":
            per_run = _check_run_rule(idx, rule, runs, current, path, platform)
        if mixed and rule["match"].get("scope") != "any":
            per_run = [_mixed_history(advisory) for advisory in per_run]
        found.extend(per_run)
        if detector in ("table", "run"):
            continue
        if detector == "dedup":
            found.extend(_check_dedup(idx, rule, runs, scan, path))
        elif detector == "pair":
            found.extend(_check_pair(idx, rule, runs, scan, path))
        elif detector == "untracked_calls":
            found.extend(_check_untracked(rule, data, runs, scan, path))
    notes = {run.id: run.note for run in current if run.note}
    if notes:
        from dataclasses import replace

        found = [replace(a, evidence=(notes[a.program],) + a.evidence)
                 if a.program in notes and notes[a.program] not in a.evidence else a
                 for a in found]
    return found


_STATUS_RANK = {POSSIBLY: 0, AFFECTED: 1}


def _merge_readings(idx, results: list[list[Advisory]], notes: list[str],
                    upper_bound: bool = False) -> list[Advisory]:
    """One advisory per rule across plausible readings of a header: as found
    when every reading agrees on the status, else possibly affected (always,
    when a reading is only an upper bound)."""
    from dataclasses import replace

    merged = []
    for rule in idx.data["advisories"]:
        found = [next((a for a in result if a.id == rule["id"]), None) for result in results]
        present = [a for a in found if a is not None]
        if not present:
            continue
        worst = max(present, key=lambda a: (_STATUS_RANK.get(a.status, 0),
                                            _CONFIDENCE_RANK.get(a.confidence, 0)))
        if (not upper_bound and len(present) == len(found)
                and len({a.status for a in present}) == 1):
            confidence = min((a.confidence for a in present), key=_CONFIDENCE_RANK.get)
            merged.append(replace(worst, confidence=confidence))
        else:
            merged.append(replace(worst, status=POSSIBLY, confidence="low",
                                  evidence=worst.evidence + tuple(notes)))
    return merged


def _mixed_history(advisory: Advisory) -> Advisory:
    """Name the branch: a merged file also holds calls of another chemistry."""
    from dataclasses import replace

    note = (f"the header merges FiberHMM call histories of different chemistries "
            f"(samtools merge); this applies to the reads of the @PG {advisory.program} "
            "history")
    return replace(advisory, evidence=advisory.evidence + (note,))


def _check_table(idx, rule, current, path):
    match = rule["match"]
    findings = []
    programs = []
    for run in current:
        chem = run.chemistry
        if chem.get("enzyme") != match["enzyme"]:
            # A digest match on a custom-chemistry run still counts.
            if not any(idx.table(chem.get(k, "")) and idx.table(chem[k])["table"] == match["table"]
                       for k in ("apply_sha256", "recall_sha256") if chem.get(k)):
                continue
        if match.get("platform") and chem.get("platform") not in (match["platform"], None):
            continue
        if match.get("platform") and chem.get("platform") is None and chem.get("mode") != "nanopore-fiber":
            continue
        finding = _table_finding(idx, rule, run)
        if finding is not None:
            findings.append(finding)
            programs.append(run.id)
    combined = _combine(findings)
    if combined is None:
        return []
    # Name the latest run with the reported status (merged branches differ).
    named = [program for program, finding in zip(programs, findings)
             if finding.status == combined.status]
    return [_make(rule, combined, program=named[-1] if named else None, path=path)]


def _check_run_rule(idx, rule, runs, current, path, platform=(None, None)):
    scope = runs if rule["match"].get("scope") == "any" else current
    out = []
    for run in scope:
        if not _matches_run(rule, run, runs, platform[0]):
            continue
        finding = _code_finding(idx, rule, run)
        if finding.status != _NOT:
            if rule["match"].get("header_platform"):
                finding.evidence.insert(0, f"@PG {run.id} ran in {run.chemistry.get('mode')} "
                                           f"mode without --seq; the reads are "
                                           f"{platform[0]} ({platform[1]})")
            out.append(_make(rule, finding, program=run.id, path=path))
    # One advisory per rule: the worst matching run speaks for the file (the
    # latest of equals); merged branches each hold their own reads.
    if not out:
        return []
    return [max(reversed(out), key=lambda a: (_STATUS_RANK.get(a.status, 0),
                                              _CONFIDENCE_RANK.get(a.confidence, 0)))]


def _check_dedup(idx, rule, runs, scan, path):
    findings = []
    recorded = False
    for run in runs:
        if run.program == "fiberhmm-dedup":
            recorded = True
            if "grouping=none" in run.ds:
                continue
            findings.append(_code_finding(idx, rule, run))
        elif run.program == "fiberhmm-call":
            state = _ds_field(run.ds, "dedup")
            if state and state != "off" and run.chemistry.get("mode") == "daf":
                recorded = True
                findings.append(_code_finding(idx, rule, run))
    if scan and scan.dedup_tags and not recorded:
        findings.append(_Finding(POSSIBLY, "medium", [
            f"{scan.dedup_tags} of the first {scan.records} records carry di/ds duplicate-"
            "cluster tags, but the header has no fiberhmm-dedup @PG (recorded since 3.0) "
            "or fiberhmm-call dedup setting"]))
    combined = _combine(findings)
    return [_make(rule, combined, path=path)] if combined else []


def _check_pair(idx, rule, runs, scan, path):
    findings = []
    pair_runs = [run for run in runs if run.program == "fiberhmm-pair"]
    for run in pair_runs:
        finding = _code_finding(idx, rule, run)
        if finding.status == _NOT:
            continue
        if scan and scan.paired_duplicates and finding.status == AFFECTED:
            findings.append(_Finding(AFFECTED, "high", finding.evidence + [
                f"{scan.paired_duplicates} of the first {scan.records} records are "
                "0x400 duplicates with a pairing partner"]))
        else:
            findings.append(_Finding(POSSIBLY, "low" if finding.status == POSSIBLY else "medium",
                                     finding.evidence + [
                "affects the output only if its input carried 0x400 duplicate flags"]))
    if scan and scan.pair_tags and not pair_runs:
        if scan.paired_duplicates:
            findings.append(_Finding(AFFECTED, "medium", [
                f"{scan.paired_duplicates} of the first {scan.records} records are 0x400 "
                "duplicates with a pairing partner, and no fiberhmm-pair @PG is recorded"]))
        else:
            findings.append(_Finding(POSSIBLY, "low", [
                "reads carry pairing tags (mt) but the header has no fiberhmm-pair @PG; "
                "affects the output only if its input carried 0x400 duplicate flags"]))
    combined = _combine(findings)
    return [_make(rule, combined, path=path)] if combined else []


def _check_untracked(rule, data, runs, scan, path):
    if any(run.program in _CALL_PRODUCERS for run in runs):
        return []
    ma_types = [c for c in data.get("CO", []) if str(c).startswith("MA-TYPES:v1:")]
    header_calls = any(re.search(r"(^|[:,])(nuc|msp|tf)(,|$)", str(c)) for c in ma_types)
    if scan and scan.call_tags:
        evidence = [f"{scan.call_tags} of the first {scan.records} records carry footprint "
                    "calls; no FiberHMM calling @PG is recorded"]
    elif header_calls:
        evidence = ["the header advertises nuc/msp/tf annotations (MA-TYPES) but records "
                    "no FiberHMM calling @PG"]
    else:
        return []
    return [_make(rule, _Finding(POSSIBLY, "low", evidence), path=path)]


def check_bam(path, *, scan_records: int = DEFAULT_SCAN_RECORDS,
              sidecars: bool = True, index: Optional[dict] = None) -> list[Advisory]:
    """Advisories for a BAM/CRAM/SAM: its header, the first ``scan_records``
    records (0: header only) and, with ``sidecars``, the QC report
    ``fiberhmm-call`` writes beside it (``qc/<name>.qc.json``)."""
    import pysam

    path = str(path)
    if "://" not in path and not os.path.exists(path):
        raise AdvisoryInputError(f"{path}: no such file")
    try:
        with pysam.AlignmentFile(path, "r" if path.endswith(".sam") else "rb",
                                 check_sq=False) as bam:
            header = bam.header.to_dict()
            scan = scan_reads(bam, scan_records)
    except (OSError, ValueError, UnicodeDecodeError) as error:
        raise AdvisoryInputError(f"{path}: cannot read as a BAM ({error})") from error
    found = check_header(header, scan=scan, index=index, path=path)
    if sidecars and "://" not in path:
        stem = Path(path)
        name = stem.name
        for suffix in (".bam", ".cram", ".sam"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        for candidate in (stem.parent / "qc" / f"{name}.qc.json", stem.parent / f"{name}.qc.json"):
            if candidate.is_file():
                found.extend(check_qc(candidate, index=index))
    return found


# ---------------------------------------------------------------------------
# File checks: QC, posteriors, consensus results
# ---------------------------------------------------------------------------

def _file_finding(idx: _Index, rule: dict, version: Optional[str], where: str,
                  missing: str) -> Optional[_Finding]:
    if version:
        run = _Run(position=0, id=where, program="", version=version, cl="", ds="")
        finding = _code_finding(idx, rule, run)
        finding.evidence = [line.replace(f"@PG {where}", where) for line in finding.evidence]
        return None if finding.status == _NOT else finding
    return _Finding(POSSIBLY, "medium", [missing])


def check_qc(path, *, index: Optional[dict] = None) -> list[Advisory]:
    """Advisories for a ``fiberhmm-qc`` report (``<prefix>.qc.json`` or combined)."""
    idx = _index(index)
    path = str(path)
    payload = _read_json(path)
    samples = payload.get("samples") if isinstance(payload.get("samples"), list) else [payload]
    rule = idx.rules["qc-nanopore-opportunities"]
    findings = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise AdvisoryInputError(f"{path}: malformed QC report (a sample is not a JSON object)")
        assay = sample.get("assay") or {}
        if not isinstance(assay, dict):
            raise AdvisoryInputError(f"{path}: malformed QC report (assay is not a JSON object)")
        mode = str(assay.get("mode") or "").lower()
        profile = str(assay.get("reference_profile") or "").lower()
        if mode != "nanopore-fiber" and "nanopore" not in profile:
            continue
        finding = _file_finding(
            idx, rule, sample.get("fiberhmm_version"), "QC report",
            "the QC report has no fiberhmm_version (written since 3.0), and its "
            "reads are Nanopore Fiber-seq")
        if finding:
            findings.append(finding)
    combined = _combine(findings)
    return [_make(rule, combined, path=path)] if combined else []


def check_posteriors(path, *, index: Optional[dict] = None) -> list[Advisory]:
    """Advisories for a ``fiberhmm-posteriors`` TSV(.gz) or HDF5 file."""
    idx = _index(index)
    path = str(path)
    metadata = _posteriors_metadata(path)
    rule = idx.rules["posteriors-reverse-frame"]
    finding = _file_finding(
        idx, rule, metadata.get("fiberhmm_version"), "posteriors file",
        "the posteriors file records no fiberhmm_version (written since 3.0)")
    return [_make(rule, finding, path=path)] if finding else []


def _posteriors_metadata(path: str) -> dict:
    if path.endswith((".h5", ".hdf5")):
        try:
            import h5py
        except ImportError as error:
            raise AdvisoryInputError(f"{path}: h5py is needed to read HDF5 posteriors") from error
        try:
            with h5py.File(path, "r") as handle:
                attrs = {key: handle.attrs[key] for key in handle.attrs}
        except (OSError, ValueError, KeyError) as error:
            raise AdvisoryInputError(f"{path}: cannot read as HDF5 ({error})") from error
        if "format_version" not in attrs:
            raise AdvisoryInputError(f"{path}: not a fiberhmm-posteriors HDF5 file")
        try:
            return {key: (value.decode() if isinstance(value, bytes) else value)
                    for key, value in attrs.items()}
        except UnicodeDecodeError as error:
            raise AdvisoryInputError(f"{path}: malformed HDF5 metadata ({error})") from error
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            first = handle.readline()
    except (OSError, EOFError, UnicodeDecodeError) as error:
        raise AdvisoryInputError(f"{path}: {error}") from error
    if not first.startswith("#metadata:"):
        raise AdvisoryInputError(f"{path}: not a fiberhmm-posteriors TSV (no #metadata line)")
    try:
        metadata = json.loads(first[len("#metadata:"):])
    except ValueError as error:
        raise AdvisoryInputError(f"{path}: malformed #metadata line ({error})") from error
    if not isinstance(metadata, dict):
        raise AdvisoryInputError(f"{path}: malformed #metadata line (not a JSON object)")
    return metadata


def check_consensus(path, *, index: Optional[dict] = None) -> list[Advisory]:
    """Advisories for a ``fiberhmm-consensus`` result directory (or its manifest.json)."""
    idx = _index(index)
    root = Path(path)
    if root.is_file():
        root = root.parent
    manifests = []
    if (root / "manifest.json").is_file():
        manifests.append(root / "manifest.json")
    manifests.extend(sorted(root.glob("window_*/manifest.json")))
    run_manifest = root / "consensus_run.json"
    if not manifests and not run_manifest.is_file():
        raise AdvisoryInputError(f"{path}: no consensus manifest.json or consensus_run.json")
    rule = idx.rules["recaller-tier-double-count"]
    findings = []
    versions = []
    if run_manifest.is_file():
        attempts = _read_json(str(run_manifest)).get("attempts", [])
        if not isinstance(attempts, list) or not all(isinstance(a, dict) for a in attempts):
            raise AdvisoryInputError(f"{run_manifest}: malformed attempts (not a list of objects)")
        versions = [a.get("fiberhmm_version") for a in attempts]
    for manifest_path in manifests:
        manifest = _read_json(str(manifest_path))
        if manifest.get("cr_mode") != "lattice_recaller":
            continue
        recaller = manifest.get("recaller") or {}
        if not isinstance(recaller, dict):
            raise AdvisoryInputError(f"{manifest_path}: malformed recaller (not a JSON object)")
        if "unscored_classes" in recaller:
            continue  # written by code with the fix (same commit added the field)
        classes = manifest_path.parent / "classes.tsv"
        tiers = None
        if classes.is_file():
            try:
                with open(classes, encoding="utf-8") as handle:
                    tiers = "prevalence_edge" in handle.readline()
            except (OSError, UnicodeDecodeError) as error:
                raise AdvisoryInputError(f"{classes}: {error}") from error
        if tiers is False:
            continue  # predates prevalence tiers
        if versions and all(v and _version_tuple(v) >= _version_tuple(rule["fixed_in"]["version"])
                            for v in versions):
            continue
        where = manifest_path.parent.name if manifest_path.parent != root else root.name
        if tiers:
            findings.append(_Finding(AFFECTED, "medium", [
                f"{where}: lattice-recaller manifest has prevalence tiers but no "
                "recaller.unscored_classes, which the same fix added"]))
        else:
            findings.append(_Finding(POSSIBLY, "low", [
                f"{where}: lattice-recaller manifest predates the fix; classes.tsv is "
                "missing, so whether it has prevalence tiers is unknown"]))
    combined = _combine(findings)
    return [_make(rule, combined, path=str(root))] if combined else []


def _read_json(path: str) -> dict:
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, EOFError, ValueError) as error:
        raise AdvisoryInputError(f"{path}: cannot read JSON ({error})") from error
    if not isinstance(data, dict):
        raise AdvisoryInputError(f"{path}: not a JSON object")
    return data


# ---------------------------------------------------------------------------
# Dispatch and report
# ---------------------------------------------------------------------------

def output_kind(path) -> str:
    """``bam``, ``qc``, ``posteriors`` or ``consensus`` for a path, by name and shape."""
    text = str(path)
    lower = text.lower()
    if lower.endswith((".bam", ".cram", ".sam")) or "://" in text:
        return "bam"
    candidate = Path(text)
    if candidate.is_dir() or candidate.name in ("manifest.json", "consensus_run.json"):
        return "consensus"
    if lower.endswith(".qc.json"):
        return "qc"
    if lower.endswith((".tsv", ".tsv.gz", ".h5", ".hdf5")):
        return "posteriors"
    if lower.endswith(".json"):
        data = _read_json(text)
        if "assay" in data and "sampling" in data or data.get("report_type") == "fiberhmm_multi_sample_qc":
            return "qc"
        if data.get("cr_mode") or str(data.get("schema", "")).startswith("fiberhmm.consensus"):
            return "consensus"
    raise AdvisoryInputError(f"{text}: not a BAM, QC report, posteriors file or consensus result")


def check_path(path, *, scan_records: int = DEFAULT_SCAN_RECORDS, sidecars: bool = True,
               index: Optional[dict] = None) -> list[Advisory]:
    """Advisories for any FiberHMM output (see :func:`output_kind`)."""
    kind = output_kind(path)
    if kind != "bam" and not Path(str(path)).exists():
        raise AdvisoryInputError(f"{path}: no such file or directory")
    if kind == "bam":
        return check_bam(path, scan_records=scan_records, sidecars=sidecars, index=index)
    if kind == "qc":
        return check_qc(path, index=index)
    if kind == "posteriors":
        return check_posteriors(path, index=index)
    return check_consensus(path, index=index)


def overall_status(advisories: Iterable[Advisory]) -> str:
    """``clean``, ``info``, ``rerun-recommended`` or ``rerun-required``: the worst severity."""
    severities = {advisory.severity for advisory in advisories}
    for severity in SEVERITIES:
        if severity in severities:
            return severity
    return "clean"


def report(path, *, scan_records: int = DEFAULT_SCAN_RECORDS, sidecars: bool = True,
           index: Optional[dict] = None) -> dict:
    """JSON-ready report for one output (``fiberhmm.advisory_report.v1``).

    Never raises for a bad input: missing, unreadable, malformed or unknown
    paths (and any other failure while checking one) give ``status: "error"``
    with ``error`` set. Interrupts (``KeyboardInterrupt``, ``SystemExit``)
    still propagate.
    """
    import fiberhmm

    base = {
        "schema": REPORT_SCHEMA,
        "path": str(path),
        "checked_with": {"fiberhmm_version": fiberhmm.__version__,
                         "advisories_revision": None},
    }

    def error_report(message: str) -> dict:
        return {**base, "kind": None, "status": "error", "needs_rerun": None,
                "confirmed": None, "error": message, "advisories": []}

    try:
        data = index if index is not None else load_index()
        base["checked_with"]["advisories_revision"] = data.get("revision")
        kind = output_kind(path)
        advisories = check_path(path, scan_records=scan_records, sidecars=sidecars, index=data)
    except AdvisoryInputError as error:
        return error_report(str(error))
    except Exception as error:  # noqa: BLE001 - one bad input must not abort a batch
        return error_report(f"{path}: cannot check ({type(error).__name__}: {error})")
    return {
        **base,
        "kind": kind,
        "status": overall_status(advisories),
        "needs_rerun": any(a.needs_rerun for a in advisories),
        "confirmed": any(a.needs_rerun and a.status == AFFECTED for a in advisories),
        "error": None,
        "advisories": [a.to_dict() for a in advisories],
    }
