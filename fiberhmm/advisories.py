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


def _runs(header: dict) -> list[_Run]:
    from fiberhmm.io.bam_header import declared_chemistries

    declarations = declared_chemistries(header)
    by_pg = {d["pg"]: d for d in declarations if d.get("pg")}
    runs = []
    for position, record in enumerate(header.get("PG", [])):
        program = _program_name(record)
        if program is None:
            continue
        version = str(record.get("VN") or "") or None
        if version in (None, "unknown"):
            version = None
        run = _Run(position=position, id=str(record.get("ID", program)), program=program,
                   version=version, cl=str(record.get("CL", "")), ds=str(record.get("DS", "")),
                   declaration=by_pg.get(str(record.get("ID", ""))))
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


_MERGED_ID_RE = re.compile(r"-[0-9A-F]{8}$")


def _current_runs(runs: list[_Run]) -> tuple[list[_Run], bool]:
    """The runs whose calls the file holds, and whether they mix chemistries.

    The last full call and the recalls after it. ``samtools merge`` appends
    the other inputs' @PG records after the first input's, renaming clashing
    IDs with a ``-XXXXXXXX`` suffix; calls in those merged-in histories hold
    their own inputs' reads, so they count too. A FiberHMM run after the merge
    re-calls every read and supersedes them all.
    """
    producers = [run for run in runs if run.program in _CALLING | _RECALLING]
    calls = [i for i, run in enumerate(producers) if run.program in _CALLING]
    if not calls:
        return producers, False
    own = [i for i in calls if not _MERGED_ID_RE.search(producers[i].id)]
    current = producers[(own or calls)[-1]:]
    chemistries = {(run.chemistry.get("enzyme"), run.chemistry.get("platform"))
                   for run in current if run.program in _CALLING}
    return current, len(chemistries) > 1


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

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
    runs = _runs(data)
    current, mixed = _current_runs(runs)
    platform = sequencing_platform(data)
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
    return found


def _mixed_history(advisory: Advisory) -> Advisory:
    """A header merging call histories of different chemistries cannot say which reads."""
    from dataclasses import replace

    note = ("the header merges FiberHMM call histories of different chemistries "
            "(samtools merge), so it cannot tell which reads this applies to")
    status = POSSIBLY if advisory.status == AFFECTED else advisory.status
    return replace(advisory, status=status, confidence="low",
                   evidence=advisory.evidence + (note,))


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
    return [_make(rule, combined, program=programs[-1] if programs else None, path=path)]


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
    # One advisory per rule: the latest matching run speaks for the file.
    return out[-1:]


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
