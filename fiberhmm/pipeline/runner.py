"""The ordered steps of ``fiberhmm-pipeline``.

``prepare_reference -> index -> align -> call -> qc [-> tracks]``

Each step reports ``running``/``done``/``skipped`` through a
:class:`~fiberhmm.pipeline.progress.ProgressReporter` (the FiberBrowser
contract, ``fiberhmm.pipeline.outputs.v1``) and the long steps write a
completion marker, so re-running the same command continues where an earlier
run stopped. ``Pipeline(config, progress).run()`` can be driven in-process
(with a progress callback) or through the command line.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import pysam

from fiberhmm import __version__
from fiberhmm.io.run_state import DirectoryBusy, DirectoryLock, load_memo_file, save_memo_file
from fiberhmm.pipeline import aligner as mm2
from fiberhmm.pipeline.circular import hard_clip, merge_origin_pieces
from fiberhmm.pipeline.progress import (
    STATE_DIR,
    ProgressReporter,
    clear_marker,
    file_fingerprint,
    fingerprint_changes,
    outputs_valid,
    read_marker,
    write_json_atomic,
    write_marker,
)
from fiberhmm.pipeline.reference import (
    ReferenceInfo,
    declared_references,
    decorate_header,
    header_matches_reference,
    prepare_reference,
)

OUTPUTS_SCHEMA = "fiberhmm.pipeline.outputs.v1"
BASE_STEPS = ("prepare_reference", "index", "align", "call", "qc")
DAF_ENZYMES = ("ddda", "dddb")
# fiberhmm-call --region-parallel's default region size, and the smallest run
# for which its resumable mode is used (see Pipeline.resumable_call).
CALL_REGION_SIZE = 10_000_000
RESUMABLE_MIN_RECORDS = 20_000
READ_PATTERNS = (".fastq", ".fastq.gz", ".fq", ".fq.gz", ".bam")


class PipelineError(RuntimeError):
    """A failure with a message meant for the user (no traceback)."""

    def __init__(self, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.hint = hint


class PipelineCancelled(PipelineError):
    def __init__(self, message: str = "cancelled", signum: int = signal.SIGTERM):
        super().__init__(message, hint="Run the same command again to continue; "
                                       "completed steps are kept.")
        self.exit_code = 128 + int(signum)


@dataclass
class PipelineConfig:
    reads: list[str]
    reference: str
    enzyme: str
    outdir: str
    sample: Optional[str] = None
    cores: int = 4
    seq: Optional[str] = None
    topology: str = "auto"
    regions: list[str] = field(default_factory=list)
    min_mapq: int = 20
    min_read_length: Optional[int] = None
    # None: DAF reads on circular contigs only (concatemer arms beyond one
    # full circle); linear-contig soft clips are kept (DAF calling treats them
    # as no evidence, and they are the read's own sequence for SV views).
    hard_clip: Optional[bool] = None
    origin_merge: bool = True
    origin_tolerance: int = 100
    dedup: str = "auto"  # auto | on | off (fiberhmm-call: automatic for DAF files)
    dedup_mode: str = "flag"  # flag (mark 0x400, keep reads) | collapse
    snp_screen: str = "auto"  # auto | on | off
    snp_mask: Optional[str] = None  # user BED of SNP sites to mask
    chimera_filter: bool = True
    daf_mask_unaligned: bool = True  # DAF: insertion/soft-clip bases are no evidence
    # Records kept by the aligner step and called: see read_filters.ALIGNMENT_SETS.
    alignments: str = "primary-supplementary"
    prob_threshold: Optional[int] = None
    use_m5c: Optional[bool] = None  # None: fiberhmm-call default (on for ddda)
    cpg_mask_policy: Optional[str] = None
    qc: bool = True
    call_args: list[str] = field(default_factory=list)
    call_mode: str = "auto"  # auto | streaming | resumable
    tracks: bool = False
    aligner: str = "auto"
    force_chemistry: bool = False  # run although the reads contradict --enzyme/--seq
    redo: Optional[str] = None  # "all" or a step: redo it and the later ones
    verbose: bool = False
    quiet: bool = False

    def resolved_seq(self) -> Optional[str]:
        """The platform the run states: ``--seq``, or what the reads record.

        None for DAF reads with no platform record that are called as given
        (fiberhmm-call then applies its own default); DAF reads aligned here
        without a record are aligned and declared as Nanopore (the
        Pipeline sets ``seq`` when it decides to align).
        """
        return self.seq or None

    def resolved_hard_clip(self) -> str:
        """``on``, ``off``, or ``circular`` (DAF default: hard-clip records on
        circular contigs only)."""
        if self.hard_clip is not None:
            return "on" if self.hard_clip else "off"
        return "circular" if self.enzyme in DAF_ENZYMES else "off"

    def hard_clips(self, circular_contig: bool) -> bool:
        mode = self.resolved_hard_clip()
        return mode == "on" or (mode == "circular" and circular_contig)

    def preset(self) -> str:
        return mm2.PRESET_FOR_PLATFORM[self.resolved_seq() or "nanopore"]

    def steps(self) -> list[str]:
        return list(BASE_STEPS) + (["tracks"] if self.tracks else [])

    def calling_settings(self) -> dict:
        """Effective calling settings (``None`` = the fiberhmm-call default)."""
        daf = self.enzyme in DAF_ENZYMES
        return {
            "enzyme": self.enzyme,
            "seq": self.resolved_seq() or "auto",
            "min_mapq": self.min_mapq,
            "min_read_length": self.min_read_length or 1000,
            "dedup": self.dedup if daf else "off",
            "dedup_mode": self.dedup_mode if daf else None,
            "snp_screen": self.snp_screen if daf else "off",
            "snp_mask": os.path.abspath(self.snp_mask) if self.snp_mask else None,
            "chimera_filter": self.chimera_filter if daf else None,
            "daf_unaligned_mask": self.daf_mask_unaligned if daf else None,
            # fiberhmm-call's default; --call-args "--daf-insert-consensus off"
            # turns it off.
            "daf_insert_consensus": (
                ("off" if "off" in _call_arg_value(self.call_args, "--daf-insert-consensus")
                 else "auto") if daf and self.daf_mask_unaligned else None),
            "primary_only": self.alignments == "primary",
            "alignments": self.alignments,
            "prob_threshold": self.prob_threshold if self.prob_threshold is not None
            else "auto",
            "use_m5c": ("auto" if self.use_m5c is None else self.use_m5c)
            if self.enzyme == "ddda" else None,
            "cpg_mask_policy": (self.cpg_mask_policy or "unmethylated-only")
            if self.enzyme == "ddda" else None,
            "hard_clip": self.resolved_hard_clip(),
            "origin_merge": self.origin_merge,
            "qc": self.qc,
            "tracks": self.tracks,
            "call_args": list(self.call_args),
        }


def expand_read_inputs(paths: list[str]) -> list[str]:
    """Files as given; a directory contributes its FASTQ and BAM files (sorted)."""
    out: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            found = sorted(
                os.path.join(path, name) for name in os.listdir(path)
                if name.lower().endswith(READ_PATTERNS) and not name.startswith("."))
            if not found:
                raise PipelineError(f"no FASTQ or BAM files in {path}",
                                    hint="Expected *.fastq, *.fastq.gz, *.fq(.gz) or *.bam")
            out.extend(found)
        else:
            out.append(path)
    return out


def default_sample_name(path: str) -> str:
    name = os.path.basename(os.path.normpath(path))
    for suffix in (".gz", ".fastq", ".fq", ".bam", ".cram", ".sam", ".sorted",
                   ".unaligned", ".aligned", ".fiberhmm"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name).strip("_") or "sample"


def validate_sample_name(name: str) -> str:
    """An explicit --sample must be one plain file-name component.

    It names files inside OUTDIR and the read group, so path separators,
    ``.``/``..``, a leading ``.`` or ``-``, whitespace and control characters
    are refused rather than allowed to place outputs elsewhere.
    """
    text = str(name)
    bad = (not text or text in (".", "..") or text[0] in ".-"
           or any(sep in text for sep in ("/", "\\", os.sep, os.altsep or "/"))
           or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in text))
    if bad:
        raise PipelineError(
            f"--sample {text!r} is not a plain file name",
            hint="Use one name without path separators, spaces or a leading '.'/'-', "
                 "e.g. --sample run1 (letters, digits, '_', '-' and '.').")
    return text


def infer_daf_platform(paths: list[str]) -> tuple[Optional[str], str]:
    """``(platform or None, evidence)`` for DAF reads, from header records only.

    The same evidence ``fiberhmm-call`` accepts for a DAF enzyme: a FIBERHMM
    chemistry declaration, ``@RG PL`` or aligner/basecaller ``@PG`` records
    (DAF reads carry no platform-specific MM pattern). FASTQ files carry
    none. Raises :class:`PipelineError` when the files disagree.
    """
    from fiberhmm.cli.common import sniff_sequencing_platform
    votes: dict[str, list[str]] = {}
    for path in paths:
        if not path.lower().endswith((".bam", ".cram", ".sam")):
            continue
        name = os.path.basename(path)
        evidence = sniff_sequencing_platform(path, inspect_reads=False)
        if evidence.conflict:
            raise PipelineError(f"cannot tell the sequencing platform of {name}: "
                                f"{evidence.conflict}",
                                hint="Pass --seq pacbio or --seq nanopore.")
        if evidence.platform:
            votes.setdefault(evidence.platform, []).append(f"{name} ({evidence.source})")
    if len(votes) > 1:
        detail = "; ".join(f"{platform}: {', '.join(src)}" for platform, src in sorted(votes.items()))
        raise PipelineError(f"the read files disagree about the sequencing platform ({detail})",
                            hint="Pass --seq pacbio or --seq nanopore.")
    if not votes:
        return None, "no platform record in the read files"
    platform, sources = next(iter(votes.items()))
    return platform, "; ".join(sources)


def chemistry_problems(paths: list[str], enzyme: str, seq: Optional[str]) -> list[str]:
    """Why the reads do not look like ``enzyme`` (and ``--seq``) data.

    Checked on the inputs themselves, before alignment, from strong evidence
    only: a FIBERHMM-CHEMISTRY declaration naming another enzyme (or, with an
    explicit ``seq``, another platform); for a DAF enzyme, reads that mostly
    carry m6A calls (MM A+a/T-a) and no R/Y-encoded deaminations; for Hia5,
    reads that mostly carry R/Y-encoded deaminations, or no m6A call at all.
    """
    from fiberhmm.io.bam_header import declared_chemistries
    problems: list[str] = []
    for path in paths:
        name = os.path.basename(path)
        if path.lower().endswith((".bam", ".cram", ".sam")):
            with pysam.AlignmentFile(path, check_sq=False) as bam:
                declarations = declared_chemistries(bam.header)
            enzymes = {str(d.get("enzyme", "")).lower() for d in declarations} - {"", "custom"}
            if enzymes and enzymes != {enzyme}:
                problems.append(f"{name} declares {'/'.join(sorted(enzymes))} chemistry "
                                f"(FIBERHMM-CHEMISTRY), not {enzyme}")
            platforms = {("nanopore" if str(d.get("platform", "")).lower() == "ont"
                          else str(d.get("platform", "")).lower())
                         for d in declarations} & {"pacbio", "nanopore"}
            if seq and platforms and platforms != {seq}:
                problems.append(f"{name} declares {'/'.join(sorted(platforms))} reads "
                                f"(FIBERHMM-CHEMISTRY), not --seq {seq}")
        found = mm2.read_chemistry_evidence(path)
        n, m6a, iupac = found["records"], found["m6a"], found["iupac"]
        if not n:
            continue
        if enzyme in DAF_ENZYMES and m6a * 2 > n and not iupac:
            problems.append(
                f"{m6a} of the first {n} reads of {name} carry m6A calls (MM A+a/T-a) and "
                "none carries R/Y-encoded deaminations: these look like Fiber-seq (Hia5) "
                "reads, not DAF-seq")
        elif enzyme == "hia5" and iupac * 2 > n and not m6a:
            problems.append(
                f"{iupac} of the first {n} reads of {name} carry R/Y-encoded deaminations "
                "and none an m6A call: these look like DAF-seq reads, not Hia5 Fiber-seq")
        elif enzyme == "hia5" and not m6a:
            problems.append(
                f"none of the first {n} reads of {name} carries an m6A call (MM/ML A+a or "
                "T-a): Hia5 calling would find no m6A")
    return problems


def explicit_seq_warnings(paths: list[str], enzyme: str, seq: str) -> list[str]:
    """Warnings for an explicit ``--seq`` the reads' own records disagree with.

    An explicit ``--seq`` is authoritative (as in ``fiberhmm-call``); a
    disagreeing chemistry declaration is refused by :func:`chemistry_problems`,
    other evidence (MM pattern for Hia5, @RG/@PG records) only warns.
    """
    from fiberhmm.cli.common import sniff_sequencing_platform
    warnings = []
    for path in paths:
        name = os.path.basename(path)
        if path.lower().endswith((".bam", ".cram", ".sam")):
            evidence = sniff_sequencing_platform(path, inspect_reads=enzyme == "hia5")
            found, source = evidence.platform, evidence.source
        elif enzyme == "hia5":
            counts = mm2.fastq_platform_votes(path)
            found = max(counts, key=counts.get) if any(counts.values()) else None
            source = f"MM tags of {sum(counts.values())} read(s)"
        else:
            continue
        if found and found != seq and "declaration" not in str(source):
            warnings.append(f"platform: --seq {seq} was given, but {name} looks like "
                            f"{found} ({source}); using --seq {seq} as requested")
    return warnings


def infer_read_platform(paths: list[str]) -> tuple[Optional[str], str]:
    """``(platform, evidence)`` for Hia5 reads, before alignment.

    BAM inputs use ``fiberhmm-call``'s detection (FIBERHMM chemistry
    declaration, MM specs of the first reads, @RG/@PG); FASTQ inputs the MM
    tags in their header comments (``T-a`` present: PacBio; ``A+a`` only:
    Nanopore). Raises :class:`PipelineError` when the inputs disagree or
    carry no evidence: the aligner preset and read group depend on it.
    """
    from fiberhmm.cli.common import sniff_sequencing_platform
    votes: dict[str, list[str]] = {}
    for path in paths:
        name = os.path.basename(path)
        if path.lower().endswith((".bam", ".cram", ".sam")):
            evidence = sniff_sequencing_platform(path)
            if evidence.conflict:
                raise PipelineError(f"cannot tell the sequencing platform of {name}: "
                                    f"{evidence.conflict}",
                                    hint="Pass --seq pacbio or --seq nanopore.")
            if evidence.platform:
                votes.setdefault(evidence.platform, []).append(f"{name} ({evidence.source})")
        else:
            counts = mm2.fastq_platform_votes(path)
            informative = counts["pacbio"] + counts["nanopore"]
            if not informative:
                continue
            majority = max(counts, key=counts.get)
            if informative - counts[majority] > 0.1 * informative:
                raise PipelineError(
                    f"cannot tell the sequencing platform of {name}: its MM tags are mixed "
                    f"({counts['pacbio']} read(s) with PacBio T-a calls, "
                    f"{counts['nanopore']} with Nanopore-style A+a calls only)",
                    hint="Pass --seq pacbio or --seq nanopore.")
            votes.setdefault(majority, []).append(
                f"{name} (MM tags of {informative} read(s))")
    if len(votes) > 1:
        detail = "; ".join(f"{platform}: {', '.join(src)}" for platform, src in sorted(votes.items()))
        raise PipelineError(f"the read files disagree about the sequencing platform ({detail})",
                            hint="Pass --seq pacbio or --seq nanopore.")
    if not votes:
        raise PipelineError(
            "cannot tell the sequencing platform of the Hia5 reads: no MM tags or "
            "platform records in the first reads",
            hint="Pass --seq pacbio or --seq nanopore (it selects the minimap2 preset "
                 "and the calling model).")
    platform, sources = next(iter(votes.items()))
    return platform, "; ".join(sources)


_REGION = re.compile(r"^(?P<chrom>[^:\s]+):(?P<start>[\d,]+)-(?P<end>[\d,]+)$")


def parse_region(text: str) -> tuple[str, int, int]:
    """``chr:start-end`` (1-based, inclusive; commas allowed) -> 0-based half-open."""
    match = _REGION.match(str(text).strip())
    if not match:
        raise PipelineError(f"bad region {text!r}",
                            hint="Use CHROM:START-END, e.g. chr2L:1000-5000")
    start = int(match.group("start").replace(",", ""))
    end = int(match.group("end").replace(",", ""))
    if start < 1 or end < start:
        raise PipelineError(f"bad region {text!r}: start must be >= 1 and <= end")
    return match.group("chrom"), start - 1, end


def format_region(chrom: str, start0: int, end: int) -> str:
    return f"{chrom}:{start0 + 1}-{end}"


# ---------------------------------------------------------------------------
# Child processes and cancellation
# ---------------------------------------------------------------------------

_CHILDREN: list[subprocess.Popen] = []
# Re-entrant: the main thread may be interrupted by a signal while it holds the
# lock, and the cleanup that follows (terminate_children) runs on that thread.
_CHILDREN_LOCK = threading.RLock()


def _register(proc: subprocess.Popen) -> subprocess.Popen:
    with _CHILDREN_LOCK:
        _CHILDREN.append(proc)
    return proc


def _unregister(proc: subprocess.Popen) -> None:
    with _CHILDREN_LOCK:
        if proc in _CHILDREN:
            _CHILDREN.remove(proc)


def _signal_group(proc: subprocess.Popen, sig) -> None:
    try:
        if os.getpgid(proc.pid) == proc.pid:
            os.killpg(proc.pid, sig)  # the child and its worker processes
            return
    except (OSError, AttributeError):
        pass
    with contextlib.suppress(OSError):
        proc.send_signal(sig)


def terminate_children() -> None:
    """Stop every running child process (with its workers)."""
    with _CHILDREN_LOCK:
        children = list(_CHILDREN)
    for proc in children:
        if proc.poll() is None:
            _signal_group(proc, signal.SIGTERM)
    deadline = time.time() + 10
    for proc in children:
        with contextlib.suppress(Exception):
            proc.wait(timeout=max(0.1, deadline - time.time()))
        if proc.poll() is None:
            _signal_group(proc, signal.SIGKILL)


def install_cancel_handlers() -> None:
    """SIGTERM (and SIGINT) raise :class:`PipelineCancelled` in the main thread.

    The handler only raises: the children are stopped by the normal cleanup
    paths the exception unwinds through (``Pipeline.run``, the step's
    ``finally``), never from inside the handler, which could interrupt a
    section that holds the child registry lock.
    """

    def handler(signum, frame):
        raise PipelineCancelled(signum=signum)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# ---------------------------------------------------------------------------
# fiberhmm-call capabilities and progress
# ---------------------------------------------------------------------------

def call_supported_options() -> set[str]:
    """Option strings ``fiberhmm-call`` accepts (``--progress-json``, ``--resume``...)."""
    parser = _call_parser()
    if parser is None:
        return set()
    return {flag for action in parser._actions for flag in action.option_strings}


class _CallArgsError(Exception):
    pass


def call_arg_file_paths(call_args: list[str]) -> list[str]:
    """Existing files that ``--call-args`` hands to ``fiberhmm-call``.

    ``call_args`` is parsed with fiberhmm-call's own parser, so every option
    spelling it accepts (``-m PATH``, ``-mPATH``, ``--model=PATH``, unique
    prefixes...) resolves to its value; every resolved string value that
    names an existing file is a dependency. Arguments the parser does not
    know (or all of them, when parsing fails -- fiberhmm-call itself will
    then refuse them) fall back to the whole-token check.
    """
    parser = _call_parser()
    leftover = list(call_args)
    values: list[str] = []
    if parser is not None:
        for action in parser._actions:
            action.required = False

        def error(message):
            raise _CallArgsError(message)

        parser.error = error
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                namespace, leftover = parser.parse_known_args(list(call_args))
        except (_CallArgsError, SystemExit, ValueError, TypeError):
            leftover = list(call_args)
        else:
            for value in vars(namespace).values():
                items = value if isinstance(value, (list, tuple)) else [value]
                values.extend(item for item in items if isinstance(item, str))
    for token in leftover:
        values.append(token)
        if token.startswith("-") and "=" in token:
            values.append(token.split("=", 1)[1])
    return [value for value in values if value and os.path.isfile(value)]


def _call_parser() -> Optional[argparse.ArgumentParser]:
    """A fresh fiberhmm-call argument parser, captured without running the
    command (the way ``tools/gen_cli_reference.py`` documents it)."""
    from fiberhmm.cli import call

    class _Captured(Exception):
        def __init__(self, parser):
            self.parser = parser

    original = argparse.ArgumentParser.parse_known_args

    def intercept(self, args=None, namespace=None):
        raise _Captured(self)

    argparse.ArgumentParser.parse_known_args = intercept
    saved = sys.argv
    sys.argv = ["fiberhmm-call"]
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            call.parse_args()
    except _Captured as captured:
        return captured.parser
    finally:
        argparse.ArgumentParser.parse_known_args = original
        sys.argv = saved
    return None


_CALL_PROGRESS = re.compile(
    r"Fused:\s*([\d,]+)\s*\|\s*Skipped:\s*([\d,]+).*?\|\s*([\d.]+)\s*r/s")


def parse_call_progress(line: str) -> Optional[tuple[int, float]]:
    """``(records processed, reads/s)`` from a fiberhmm-call status line."""
    match = _CALL_PROGRESS.search(line)
    if not match:
        return None
    done = int(match.group(1).replace(",", "")) + int(match.group(2).replace(",", ""))
    return done, float(match.group(3))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, config: PipelineConfig,
                 progress: Optional[ProgressReporter] = None):
        self.config = config
        self.progress = progress or ProgressReporter()
        self.outdir = os.path.abspath(config.outdir)
        self.sample = config.sample
        self.steps = config.steps()
        self._log_handle = None
        self.reference: Optional[ReferenceInfo] = None
        self.read_files: list[mm2.ReadFile] = []
        self.use_aligned_input: Optional[str] = None
        self.aligner: Optional[mm2.Aligner] = None
        self.index_path: Optional[str] = None
        self.regions: list[tuple[str, int, int]] = []
        self.stats: dict = {}
        self.track_files: list[str] = []
        self.qc_outputs: Optional[dict] = None
        self.memo = None
        self._lock: Optional[DirectoryLock] = None
        self._ran: set[str] = set()
        self._notes: list = []  # logged right after the start event (text or (text, level))
        self._replace_chemistry = False
        self._redo_from = None
        if config.redo:
            self._redo_from = (0 if config.redo == "all" else
                               self.steps.index(config.redo) if config.redo in self.steps
                               else len(self.steps))

    def _setup(self) -> None:
        cfg = self.config
        cfg.reads = expand_read_inputs(cfg.reads)
        self.regions = [parse_region(r) for r in cfg.regions]
        self.sample = (validate_sample_name(cfg.sample) if cfg.sample
                       else default_sample_name(cfg.reads[0]))
        state = os.path.join(self.outdir, STATE_DIR)
        os.makedirs(state, exist_ok=True)
        # One run owns OUTDIR at a time (released by close(), or by the kernel
        # when this process ends): markers, work files and outputs are never
        # written by two runs at once.
        try:
            self._lock = DirectoryLock(os.path.join(state, "lock"),
                                       "fiberhmm-pipeline output directory").acquire()
        except DirectoryBusy as exc:
            raise PipelineError(str(exc), hint="Use another output directory (-o) for a "
                                               "second run at the same time.") from None
        self.memo = load_memo_file(os.path.join(state, "digests.json"))
        os.makedirs(os.path.join(self.outdir, "logs"), exist_ok=True)
        self._log_handle = open(os.path.join(self.outdir, "logs", "pipeline.log"), "a",
                                encoding="utf-8")
        self.aligned_bam = os.path.join(self.outdir, f"{self.sample}.aligned.bam")
        self.called_bam = os.path.join(self.outdir, f"{self.sample}.fiberhmm.bam")
        self.qc_prefix = os.path.join(self.outdir, "qc", f"{self.sample}.fiberhmm")
        self.call_progress_path = os.path.join(state, "call.progress.jsonl")
        for path in (self.aligned_bam, self.called_bam, self.qc_prefix):
            if os.path.commonpath([self.outdir, os.path.abspath(path)]) != self.outdir:
                raise PipelineError(f"output {path} would be outside {self.outdir}")
        self._check_chemistry()
        if cfg.seq:
            self._notes.extend((warning, "warning") for warning in
                               explicit_seq_warnings(cfg.reads, cfg.enzyme, cfg.seq))
        if cfg.enzyme == "hia5" and not cfg.seq:
            # The minimap2 preset, the read group's PL and fiberhmm-call's
            # model all follow the platform: decide it once, from the reads.
            platform, evidence = infer_read_platform(cfg.reads)
            cfg.seq = platform
            self._notes.append(f"platform: {platform} (detected from {evidence})")
        elif cfg.enzyme in DAF_ENZYMES and not cfg.seq:
            platform, evidence = infer_daf_platform(cfg.reads)
            if platform:
                cfg.seq = platform
                self._notes.append(f"platform: {platform} (detected from {evidence})")

    def _check_chemistry(self) -> None:
        """Refuse reads that contradict --enzyme/--seq (unless --force-chemistry)."""
        cfg = self.config
        try:
            problems = chemistry_problems(cfg.reads, cfg.enzyme, cfg.seq)
        except (OSError, ValueError) as exc:
            raise PipelineError(f"cannot read the input: {exc}")
        if not problems:
            return
        # A forced run over a contradicting FIBERHMM-CHEMISTRY declaration must
        # let fiberhmm-call replace it (it refuses otherwise).
        self._replace_chemistry = cfg.force_chemistry and any(
            "FIBERHMM-CHEMISTRY" in problem for problem in problems)
        if cfg.force_chemistry:
            for problem in problems:
                self._notes.append((f"chemistry: {problem}; continuing (--force-chemistry)",
                                    "warning"))
            return
        raise PipelineError(
            f"the reads do not look like --enzyme {cfg.enzyme}"
            + (f" --seq {cfg.seq}" if cfg.seq else "") + " data: " + "; ".join(problems),
            hint="Check --enzyme and --seq. If the data really are what you said, add "
                 "--force-chemistry.")

    # -- messages ------------------------------------------------------------
    def log(self, message: str, level: str = "info") -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        if self._log_handle is not None:
            self._log_handle.write(line + "\n")
            self._log_handle.flush()
        if not self.config.quiet:
            print(line, file=sys.stderr, flush=True)
        self.progress.log(message, level)

    def _forced(self, step: str) -> bool:
        return self._redo_from is not None and self.steps.index(step) >= self._redo_from

    def _refuse(self, step: str, changed: list[str], what: str = "result") -> None:
        raise PipelineError(
            f"{self.outdir} already holds a '{step}' {what} made with different "
            f"settings or inputs ({', '.join(changed) or 'setup'})",
            hint=f"Use a new output directory (-o), or add --redo {step} to "
                 "replace the earlier result.")

    def _upstream_reran(self, changed: list[str], upstream: dict) -> bool:
        """Every changed key is the output of an earlier step rerun in this run."""
        return bool(changed) and all(upstream.get(k) in self._ran for k in changed)

    def _is_complete(self, step: str, fingerprint: dict,
                     upstream: Optional[dict] = None) -> bool:
        """True when the step can be skipped; refuse a changed setup.

        ``upstream`` maps fingerprint keys that hold an earlier step's output
        to that step: when only those changed because the step reran in this
        run (e.g. its output was damaged), this step reruns instead of being
        refused. Kept outputs must still have their recorded size and SHA-256.
        """
        if self._forced(step):
            return False
        marker = read_marker(self.outdir, step)
        if not marker:
            return False
        if marker.get("fingerprint") != fingerprint:
            recorded = marker.get("fingerprint") or {}
            changed = fingerprint_changes(recorded, fingerprint)
            if self._upstream_reran(changed, upstream or {}):
                return False
            if changed and all(key not in recorded for key in changed):
                # Settings an older FiberHMM did not record for this step: its
                # result may not reflect them, so make it again.
                self.log(f"{step}: the earlier result does not record "
                         f"{', '.join(changed)}; running it again")
                return False
            self._refuse(step, changed)
        ok, why = outputs_valid(marker, self.memo)
        if not ok:
            self.log(f"{step}: the earlier result cannot be reused ({why}); running "
                     "it again", "warning")
            return False
        self.stats[step] = marker.get("summary") or {}
        self.progress.step(step, "skipped", message="already complete")
        self.log(f"{step}: already complete, skipped")
        return True

    def _start_step(self, step: str) -> None:
        """A step is about to change OUTDIR: the previous outputs.json no longer
        describes it (it is written again when the run completes)."""
        self._ran.add(step)
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(self.outdir, "outputs.json"))

    def _drop_stale_markers(self) -> None:
        """Forget QC/track results of an earlier called BAM that this run did not
        redo (e.g. --redo call --no-qc): they describe a BAM that no longer
        exists, so a later run makes them again instead of refusing."""
        if not os.path.isfile(self.called_bam):
            return
        called = file_fingerprint(self.called_bam, self.memo)
        for step in ("qc", "tracks"):
            marker = read_marker(self.outdir, step)
            if marker and (marker.get("fingerprint") or {}).get("called") != called:
                clear_marker(self.outdir, step)

    def _write_marker(self, step: str, fingerprint: dict, outputs: dict,
                      summary: Optional[dict] = None) -> None:
        write_marker(self.outdir, step, fingerprint, outputs, summary, self.memo)
        self._save_memo()

    def _save_memo(self) -> None:
        if self.memo is not None:
            save_memo_file(os.path.join(self.outdir, STATE_DIR, "digests.json"), self.memo)

    # -- run -----------------------------------------------------------------
    def run(self) -> dict:
        try:
            self._setup()
            self.progress.emit("start", version=__version__, sample=self.sample,
                               steps=list(self.steps), outdir=self.outdir,
                               settings=self.config.calling_settings())
            for note in self._notes:
                message, level = note if isinstance(note, tuple) else (note, "info")
                self.log(message, level)
            self.step_prepare_reference()
            self.step_index()
            self.step_align()
            self.step_call()
            self.step_qc()
            if self.config.tracks:
                self.step_tracks()
            self._drop_stale_markers()
            outputs = self.write_outputs()
        except mm2.AlignerNotFound as exc:
            self.progress.emit("done", status="error",
                               error="minimap2 was not found", hint=str(exc))
            raise
        except PipelineError as exc:
            terminate_children()
            self.progress.emit("done", status="error", error=str(exc), hint=exc.hint)
            raise
        except BaseException as exc:
            terminate_children()
            self.progress.emit("done", status="error",
                               error=f"{type(exc).__name__}: {exc}")
            raise
        self.progress.emit("done", status="ok", outputs=outputs)
        return outputs

    def close(self) -> None:
        self._save_memo()
        if self._log_handle is not None:
            with contextlib.suppress(Exception):
                self._log_handle.close()
            self._log_handle = None
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    # -- prepare_reference ----------------------------------------------------
    def step_prepare_reference(self) -> None:
        cfg = self.config
        self.progress.step("prepare_reference", "running")
        # The reference is prepared in a private staging directory and checked
        # against the earlier steps' records before anything in OUTDIR changes:
        # a refused rerun leaves the earlier reference, BAMs and outputs as a
        # consistent set.
        stage = os.path.join(self.outdir, STATE_DIR, "staging")
        shutil.rmtree(stage, ignore_errors=True)
        try:
            staged = prepare_reference(cfg.reference, stage, cfg.topology,
                                       cache_dir=mm2.index_cache_dir())
        except (OSError, ValueError) as exc:
            raise PipelineError(f"reference: {exc}",
                                hint="Give a FASTA or a plasmid map (.dna, .gb/.gbk, .embl)")
        self._check_reference_reuse(staged)
        self.reference = self._publish_reference(staged, stage)
        ref = self.reference
        for warning in ref.warnings:
            self.log(f"reference: {warning}", "warning")
        names = {c.name for c in ref.contigs}
        for chrom, _, _ in self.regions:
            if chrom not in names:
                raise PipelineError(f"region contig {chrom!r} is not in the reference")
        try:
            self.read_files = [mm2.classify_read_file(os.path.abspath(p)) for p in cfg.reads]
        except (OSError, ValueError) as exc:
            raise PipelineError(str(exc))
        circular = [c.name for c in ref.contigs if c.circular]
        message = (f"{os.path.basename(ref.source)} -> {os.path.basename(ref.fasta)}, "
                   f"{len(ref.contigs)} contig{'s' if len(ref.contigs) != 1 else ''}"
                   + (f" (circular: {', '.join(circular[:5])})" if circular else ""))
        self.log(f"reference: {message}")
        self._decide_alignment()
        self.progress.step("prepare_reference", "done", message=message)

    def _reference_identity(self, ref: Optional[ReferenceInfo] = None) -> list[dict]:
        """What the aligned and called records depend on: every contig's name,
        length, sequence MD5 and topology."""
        return [asdict(c) for c in (ref or self.reference).contigs]

    def _check_reference_reuse(self, staged: ReferenceInfo) -> None:
        identity = self._reference_identity(staged)
        for step in ("align", "call"):
            if self._forced(step):
                continue
            marker = read_marker(self.outdir, step) or {}
            recorded = (marker.get("fingerprint") or {}).get("reference")
            if recorded is not None and recorded != identity:
                self._refuse(step, ["reference"])

    def _publish_reference(self, staged: ReferenceInfo, stage: str) -> ReferenceInfo:
        """Move the staged reference files into OUTDIR (unchanged files are left alone)."""
        import filecmp
        moved: dict[str, str] = {}
        # A FASTA index goes in after its FASTA, so it is never older than it.
        names = sorted(os.listdir(stage), key=lambda n: (n.endswith(".fai"), n))
        replaced: set[str] = set()
        for name in names:
            source = os.path.join(stage, name)
            target = os.path.join(self.outdir, name)
            if (name[:-len(".fai")] in replaced if name.endswith(".fai") else False) or not (
                    os.path.isfile(target) and filecmp.cmp(source, target, shallow=False)):
                self._start_step("prepare_reference")
                os.replace(source, target)
                replaced.add(name)
            moved[source] = target
        shutil.rmtree(stage, ignore_errors=True)
        staged.fasta = moved.get(staged.fasta, staged.fasta)
        if staged.plasmid_map:
            staged.plasmid_map = moved.get(staged.plasmid_map, staged.plasmid_map)
        return staged

    def _decide_alignment(self) -> None:
        self._decide_input_alignment()
        cfg = self.config
        if cfg.enzyme in DAF_ENZYMES and not cfg.seq and not self.use_aligned_input:
            # Reads aligned here need a preset and a read-group platform:
            # without a record in the reads, DAF-seq is taken as Nanopore.
            cfg.seq = "nanopore"
            self.log("platform: nanopore (the DAF-seq default; the reads record no "
                     "platform). Pass --seq pacbio for PacBio reads.")

    def _decide_input_alignment(self) -> None:
        cfg = self.config
        aligned = [rf for rf in self.read_files if rf.kind == "aligned"]
        if not aligned or len(aligned) != len(self.read_files):
            return
        if len(aligned) > 1:
            self.log("align: several aligned BAMs given; they are realigned together")
            return
        path = aligned[0].path
        with pysam.AlignmentFile(path) as bam:
            ok, why = header_matches_reference(bam.header, self.reference)
            if ok and cfg.enzyme in DAF_ENZYMES:
                ok, why = _has_md_tags(bam)
            if ok and not all(sq.get("M5") for sq in bam.header.to_dict().get("SQ", [])):
                # Names and lengths match, but without @SQ M5 the header does not
                # say which sequence the reads were aligned to: check the reads.
                ok, why = _reads_match_reference(path, self.reference.fasta)
            has_index = bam.has_index()
        if not ok:
            self.log(f"align: {os.path.basename(path)} is realigned: {why}")
            return
        if not has_index:
            self.log(f"align: {os.path.basename(path)} has no index; it is realigned "
                     "so the output is sorted and indexed")
            return
        self.use_aligned_input = path
        self.log(f"align: {os.path.basename(path)} is already aligned to this reference; "
                 f"it is used in place (no {self.sample}.aligned.bam is written)")

    # -- index -----------------------------------------------------------------
    def step_index(self) -> None:
        cfg = self.config
        if self.use_aligned_input:
            self.progress.step("index", "skipped", message="input already aligned")
            return
        self.progress.step("index", "running")
        self.aligner = mm2.find_aligner(cfg.aligner)
        t0 = time.time()
        try:
            self.index_path, built = mm2.ensure_index(
                self.reference.fasta, self.reference.contigs, cfg.preset(),
                self.aligner, threads=cfg.cores)
        except RuntimeError as exc:
            raise PipelineError(str(exc), hint="See the minimap2 message above.")
        message = (f"{'built' if built else 'cached'} {self.index_path} "
                   f"({self.aligner.describe()}, {time.time() - t0:.1f} s)")
        self.log(f"index: {message}")
        self.progress.step("index", "done", message=message)

    # -- align -----------------------------------------------------------------
    def _align_fingerprint(self) -> dict:
        cfg = self.config
        return {
            "reads": [file_fingerprint(rf.path, self.memo) for rf in self.read_files],
            "reference": self._reference_identity(),
            "preset": cfg.preset(),
            "min_mapq": cfg.min_mapq,
            "hard_clip": cfg.resolved_hard_clip(),
            "alignments": cfg.alignments,
            "origin_merge": cfg.origin_merge,
            "origin_tolerance": cfg.origin_tolerance,
            "regions": [list(r) for r in self.regions],
            "sample": self.sample,
        }

    def step_align(self) -> None:
        if self.use_aligned_input and not self.regions:
            self.aligned_bam = self.use_aligned_input
            self.progress.step("align", "skipped", message="input already aligned")
            return
        fingerprint = self._align_fingerprint()
        if self._is_complete("align", fingerprint):
            return
        self._start_step("align")
        clear_marker(self.outdir, "align")
        self.progress.step("align", "running", done=0, unit="reads")
        if self.use_aligned_input:
            stats = self._subset_aligned_input()
        else:
            stats = self._align()
        self.stats["align"] = stats
        self._write_marker("align", fingerprint,
                           {"bam": self.aligned_bam, "bai": self.aligned_bam + ".bai"}, stats)
        self.progress.step("align", "done", done=stats.get("kept"), unit="reads",
                           message=stats.get("message"))

    def _read_pieces(self, read) -> list[tuple[int, int]]:
        """The reference intervals a record covers. A record running past the
        end of a circular contig (the SAM circular representation) also covers
        the start -- the whole contig once it spans a full turn; on a linear
        contig the part past the end is not on the reference."""
        start, end = read.reference_start, read.reference_end
        if end is None or end <= start:
            end = start + 1
        if getattr(self, "_lengths_of", None) is not self.reference:
            self._lengths = {c.name: (c.length, c.circular) for c in self.reference.contigs}
            self._lengths_of = self.reference
        length, circular = self._lengths.get(read.reference_name, (None, False))
        if length and end > length:
            if not circular:
                return [(start, length)] if start < length else []
            if end - start >= length:
                return [(0, length)]
            return [(start, length), (0, end - length)]
        return [(start, end)]

    def _overlaps_regions(self, read, regions=None) -> bool:
        regions = self.regions if regions is None else regions
        if not regions:
            return True
        chrom = read.reference_name
        return any(chrom == c and ps < e and pe > s
                   for c, s, e in regions for ps, pe in self._read_pieces(read))

    def _subset_aligned_input(self) -> dict:
        tmp = os.path.join(self.outdir, STATE_DIR, "tmp", f"{self.sample}.unsorted.bam")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        kept = 0
        lengths = {c.name: c.length for c in self.reference.contigs}
        # Fetch windows per contig: the regions, plus the last base, where
        # records that run past the contig end (through a circular origin)
        # are indexed. Merged into disjoint, non-touching windows so each
        # stored record is fetched once per window it overlaps; a record is
        # written from the first window it overlaps only. Records are thus
        # kept by occurrence: identical records (two molecules with the same
        # name and alignment) both stay.
        windows: dict[str, list[tuple[int, int]]] = {}
        for chrom, start, end in self.regions:
            windows.setdefault(chrom, []).append((start, end))
            length = lengths.get(chrom)
            if length:
                windows[chrom].append((length - 1, length))
        with pysam.AlignmentFile(self.use_aligned_input) as src:
            header = decorate_header(src.header.to_dict(), self.reference)
            with pysam.AlignmentFile(tmp, "wb", header=header) as out:
                for chrom, spans in windows.items():
                    merged: list[list[int]] = []
                    for start, end in sorted(spans):
                        if merged and start <= merged[-1][1]:
                            merged[-1][1] = max(merged[-1][1], end)
                        else:
                            merged.append([start, end])
                    previous_end = None
                    for start, end in merged:
                        for read in src.fetch(chrom, start, end):
                            if previous_end is not None and read.reference_start < previous_end:
                                continue  # overlaps an earlier window: already seen there
                            if not self._overlaps_regions(read):
                                continue
                            out.write(read)
                            kept += 1
                        previous_end = end
        _sort_index_publish(tmp, self.aligned_bam, self.config.cores)
        message = f"{kept} records in {len(self.regions)} region(s) of the aligned input"
        self.log(f"align: {message}")
        return {"kept": kept, "message": message}

    def _align(self) -> dict:
        cfg = self.config
        ref = self.reference
        preset = cfg.preset()
        carry_tags = any(
            (rf.kind == "fastq" and mm2.fastq_has_sam_tags(rf.path))
            or (rf.kind != "fastq" and mm2.bam_has_mod_tags(rf.path))
            for rf in self.read_files)
        rg = {"ID": self.sample, "SM": self.sample,
              "PL": "ONT" if (cfg.resolved_seq() or "nanopore") == "nanopore" else "PACBIO"}
        tmpdir = os.path.join(self.outdir, STATE_DIR, "tmp")
        os.makedirs(tmpdir, exist_ok=True)
        log_path = os.path.join(self.outdir, "logs", "minimap2.log")
        loaded = mm2.index_identity(self.index_path)
        stream, feeder, source_program = self._open_alignment(preset, rg, carry_tags,
                                                              log_path)
        stale = mm2.index_mismatch(stream, ref.contigs)
        if stale:
            # A cache entry that does not hold this reference (it would place
            # reads on other contigs): drop it (unless another run has already
            # replaced it), rebuild, and start again.
            self._abort_stream(stream)
            mm2.discard_index(self.index_path, loaded)
            self.log(f"index: {stale}; the cached index was removed and is rebuilt",
                     "warning")
            try:
                self.index_path, _ = mm2.ensure_index(ref.fasta, ref.contigs, preset,
                                                      self.aligner, threads=cfg.cores)
            except RuntimeError as exc:
                raise PipelineError(str(exc), hint="See the minimap2 message above.")
            stream, feeder, source_program = self._open_alignment(preset, rg, carry_tags,
                                                                  log_path)
            stale = mm2.index_mismatch(stream, ref.contigs)
            if stale:
                self._abort_stream(stream)
                raise PipelineError(f"alignment index: {stale}",
                                    hint=f"Remove {self.index_path} and run again.")

        header = stream.header.to_dict()
        if not any(pg.get("ID") == source_program["ID"] for pg in header.get("PG", [])):
            header.setdefault("PG", []).append(source_program)
        decorate_header(header, ref)
        pgs = header.setdefault("PG", [])
        pipeline_pg = {
            "ID": "fiberhmm-pipeline", "PN": "fiberhmm-pipeline", "VN": __version__,
            "CL": "fiberhmm-pipeline " + " ".join(shlex.quote(a) for a in sys.argv[1:]),
            "DS": (f"primary MAPQ>={cfg.min_mapq}; alignments={cfg.alignments}; "
                   f"origin_merge={'on' if cfg.origin_merge else 'off'}; "
                   f"hard_clip={cfg.resolved_hard_clip()}"),
        }
        if pgs:
            pipeline_pg["PP"] = pgs[-1]["ID"]
        pgs.append(pipeline_pg)
        header.setdefault("HD", {"VN": "1.6"})["SO"] = "unsorted"
        header["HD"].pop("GO", None)

        circular = {c.name for c in ref.contigs if c.circular}
        sequences: dict[str, str] = {}
        if circular and cfg.origin_merge:
            with pysam.FastaFile(ref.fasta) as fasta:
                for name in circular:
                    sequences[name] = fasta.fetch(name).upper()

        stats = {"reads": 0, "unmapped": 0, "low_mapq": 0, "outside_regions": 0,
                 "kept": 0, "supplementary_kept": 0, "origin_merged": 0,
                 "hard_clipped_reads": 0, "hard_clipped_bases": 0}
        unsorted = os.path.join(tmpdir, f"{self.sample}.unsorted.bam")
        started = time.time()

        def report():
            fraction = feeder.fraction()
            spent = time.time() - started
            eta = spent * (1 - fraction) / fraction if 0.02 < fraction < 1 else None
            rate = stats["reads"] / spent if spent > 0 else None
            self.progress.step("align", "running", done=stats["reads"], unit="reads",
                               rate=rate, eta_s=eta)

        ticker = mm2.progress_ticker(2.0, report)
        try:
            with pysam.AlignmentFile(unsorted, "wb", header=header,
                                     threads=min(4, max(1, cfg.cores))) as out:
                # Same @SQ lines in the same order: reference ids carry over.
                for group in _group_by_name(stream):
                    # One group per input record: restore (and release) its name.
                    name = feeder.restore_name(group[0].query_name)
                    for record in self._process_group(group, circular, sequences,
                                                      stats, name):
                        out.write(record)
        except BaseException:
            ticker.set()
            if isinstance(stream, mm2.Minimap2Stream):
                terminate_children()
            with contextlib.suppress(Exception):
                stream.close()
            raise
        ticker.set()
        try:
            stream.close()
        except RuntimeError as exc:
            raise PipelineError(f"alignment failed: {exc}", hint=f"See {log_path}")
        finally:
            if isinstance(stream, mm2.Minimap2Stream):
                _unregister(stream.proc)
        self.progress.step("align", "running", done=stats["reads"], unit="reads",
                           message="sorting and indexing")
        _sort_index_publish(unsorted, self.aligned_bam, cfg.cores)
        stats["message"] = (
            "{reads} reads: {kept} kept (primary, MAPQ>={q}), {unmapped} unmapped, "
            "{low_mapq} below MAPQ {q}".format(q=cfg.min_mapq, **stats)
            + (f", {stats['outside_regions']} outside --region" if self.regions else "")
            + (f"; {stats['origin_merged']} joined across a circular origin"
               if circular else "")
            + (f"; {stats['supplementary_kept']} supplementary records kept"
               if stats["supplementary_kept"] else "")
            + (f"; {stats['hard_clipped_reads']} hard-clipped"
               if stats["hard_clipped_reads"] else ""))
        self.log(f"align: {stats['message']}")
        if stats["kept"] == 0 and stats["supplementary_kept"] == 0:
            raise PipelineError(
                "no reads aligned to the reference",
                hint="Check that the reference matches the sample (for a plasmid, "
                     "the plasmid map; for amplicons, the genome).")
        return stats

    def _open_alignment(self, preset: str, rg: dict, carry_tags: bool, log_path: str):
        """Start the aligner on a fresh read feeder: ``(stream, feeder, @PG record)``."""
        cfg = self.config
        feeder = mm2.ReadFeeder(self.read_files, carry_tags)
        if self.aligner.kind == "minimap2":
            rg_line = "@RG\\t" + "\\t".join(f"{k}:{v}" for k, v in rg.items())
            cmd = mm2.minimap2_command(self.aligner, self.index_path, preset, cfg.cores,
                                       rg_line, carry_tags)
            self.log("align: " + " ".join(shlex.quote(c) for c in cmd))
            stream = mm2.Minimap2Stream(cmd, feeder, log_path)
            _register(stream.proc)
            source_program = {"ID": "minimap2", "PN": "minimap2",
                              "VN": self.aligner.version, "CL": " ".join(cmd)}
        else:
            source_program = {"ID": "mappy", "PN": "mappy", "VN": self.aligner.version,
                              "CL": f"mappy preset={preset} MD=True (as minimap2 -ax "
                                    f"{preset} --MD -Y)"}
            header = mm2.mappy_header(self.reference.contigs, rg, source_program)
            self.log(f"align: mappy {self.aligner.version}, preset {preset}")
            stream = mm2.MappyStream(self.index_path, preset, cfg.cores, feeder, header,
                                     carry_tags, self.sample)
        return stream, feeder, source_program

    @staticmethod
    def _abort_stream(stream) -> None:
        stream.abort()
        if isinstance(stream, mm2.Minimap2Stream):
            _unregister(stream.proc)

    def _process_group(self, group, circular, sequences, stats, name=None):
        """The records kept for one read: its primary (joined with an origin
        piece on a circular contig) and, unless ``alignments`` is
        ``primary``, its other supplementary records on linear contigs."""
        cfg = self.config
        stats["reads"] += 1
        primary = None
        supplementary = []
        for read in group:
            if read.is_secondary:
                continue
            if read.is_supplementary:
                supplementary.append(read)
            elif primary is None:
                primary = read
        if primary is None or primary.is_unmapped:
            stats["unmapped"] += 1
            return []
        # Each piece is filtered on its own MAPQ: a uniquely aligned
        # supplementary arm (an SV partner) is kept even when the primary
        # piece maps ambiguously.
        record = primary if primary.mapping_quality >= cfg.min_mapq else None
        if record is None:
            stats["low_mapq"] += 1
        contig = primary.reference_name
        merged_piece = None
        if record is not None and contig in circular and cfg.origin_merge:
            for sup in supplementary:
                merged = merge_origin_pieces(primary, sup, sequences[contig],
                                             cfg.origin_tolerance)
                if merged is not None:
                    record = merged
                    merged_piece = sup
                    stats["origin_merged"] += 1
                    break
        # The feeder numbered every input record (records sharing a name stay
        # separate molecules); the output keeps the name the reads came with.
        out_name = name if name is not None else mm2.original_name(record.query_name)
        # Supplementary records on linear contigs are other parts of the read
        # (the far side of an SV, an insertion's TE copy); on a circular
        # contig they are origin pieces or concatemer copies of the same
        # plasmid sequence, which would annotate the molecule twice.
        extra = []
        if cfg.alignments != "primary":
            extra = [sup for sup in supplementary
                     if sup is not merged_piece
                     and sup.reference_name not in circular
                     and sup.mapping_quality >= cfg.min_mapq]
        if record is None and cfg.alignments == "primary":
            return []
        kept = []
        if record is not None:
            if self._overlaps_regions(record):
                kept.append(record)
            else:
                stats["outside_regions"] += 1
        kept += [sup for sup in extra if self._overlaps_regions(sup)]
        if not kept:
            return []
        all_pieces_kept = (record is not None and merged_piece is None
                           and len(kept) == 1 + len(supplementary))
        for rec in kept:
            rec.query_name = out_name
            # SA lists the read's other pieces; it stays valid only when all
            # of them are kept, and is dropped where pieces were joined or dropped.
            if rec.has_tag("SA") and not all_pieces_kept:
                rec.set_tag("SA", None)
            if cfg.hard_clips(rec.reference_name in circular):
                removed = hard_clip(rec)
                if removed:
                    stats["hard_clipped_reads"] += 1
                    stats["hard_clipped_bases"] += removed
        stats["kept"] += 1 if record is not None and kept[0] is record else 0
        stats["supplementary_kept"] += sum(1 for rec in kept if rec is not record)
        return kept

    # -- call + qc -------------------------------------------------------------
    def _call_arg_files(self) -> list[dict]:
        """Content identity of every file named in --call-args (a model given with
        -m, a recall model, an NRL profile, a mask...): a changed file is a
        changed calling setup. The options are resolved by fiberhmm-call's own
        parser (:func:`call_arg_file_paths`); FiberHMM's bundled defaults are
        covered by the version."""
        found: dict[str, dict] = {}
        for candidate in call_arg_file_paths(self.config.call_args):
            identity = file_fingerprint(candidate, self.memo)
            found[identity["path"]] = identity
        return [found[path] for path in sorted(found)]

    def _call_fingerprint(self) -> dict:
        cfg = self.config
        return {
            "aligned": file_fingerprint(self.aligned_bam, self.memo),
            "reference": self._reference_identity(),
            "enzyme": cfg.enzyme,
            "seq": cfg.resolved_seq(),
            "min_mapq": cfg.min_mapq,
            "min_read_length": cfg.min_read_length,
            "dedup": cfg.dedup,
            "dedup_mode": cfg.dedup_mode,
            "snp_screen": cfg.snp_screen,
            "snp_mask": file_fingerprint(cfg.snp_mask, self.memo) if cfg.snp_mask else None,
            "chimera_filter": cfg.chimera_filter,
            "daf_mask_unaligned": cfg.daf_mask_unaligned,
            "alignments": cfg.alignments,
            "prob_threshold": cfg.prob_threshold,
            "use_m5c": cfg.use_m5c,
            "cpg_mask_policy": cfg.cpg_mask_policy,
            "call_args": list(cfg.call_args),
            "call_arg_files": self._call_arg_files(),
            "fiberhmm": __version__,
            **({"replace_chemistry": True} if self._replace_chemistry else {}),
        }

    def resumable_call(self, supported: set[str]) -> bool:
        """Use fiberhmm-call's resumable region-parallel mode?

        It pays off for genome-scale data spread over many regions. A targeted
        run (one amplicon or a plasmid) sits in one region, where the
        region-parallel pipeline runs on one core; its streaming mode is
        faster there, and rerunning a few minutes of calling costs little.
        """
        if self.config.call_mode == "streaming":
            return False
        if not {"--resume", "--work-dir", "--progress-json"} <= supported:
            return False
        if self.config.call_mode == "resumable":
            return True
        records, regions = _mapped_regions(self.aligned_bam, CALL_REGION_SIZE)
        return records >= RESUMABLE_MIN_RECORDS and regions >= self.config.cores

    def call_command(self, supported: set[str], resumable: bool = False
                     ) -> tuple[list[str], list[str]]:
        """``(fiberhmm-call argv, pass-through extras)``."""
        cfg = self.config
        cmd = [sys.executable, "-m", "fiberhmm.cli.call",
               "-i", self.aligned_bam, "-o", self.called_bam,
               "--enzyme", cfg.enzyme, "-c", str(cfg.cores),
               "--io-threads", str(max(1, min(4, cfg.cores))),
               "--min-mapq", str(cfg.min_mapq)]
        if cfg.resolved_seq():
            cmd += ["--seq", cfg.resolved_seq()]
        if cfg.min_read_length is not None:
            cmd += ["--min-read-length", str(cfg.min_read_length)]
        if cfg.alignments != "primary-supplementary":
            cmd += ["--alignments", cfg.alignments]
        if cfg.prob_threshold is not None:
            cmd += ["--prob-threshold", str(cfg.prob_threshold)]
        if cfg.enzyme in DAF_ENZYMES:
            cmd += {"on": ["--dedup"], "off": ["--no-dedup"]}.get(cfg.dedup, [])
            if cfg.dedup != "off" and cfg.dedup_mode == "collapse":
                cmd.append("--dedup-collapse")
            cmd += {"on": ["--daf-call-snps"],
                    "off": ["--no-daf-call-snps"]}.get(cfg.snp_screen, [])
            if cfg.snp_mask:
                cmd += ["--daf-snp-mask", os.path.abspath(cfg.snp_mask)]
            if not cfg.chimera_filter:
                cmd.append("--keep-chimeras")
            if not cfg.daf_mask_unaligned:
                cmd.append("--no-daf-mask-unaligned")
        if cfg.enzyme == "ddda":
            if cfg.use_m5c is not None:
                cmd.append("--use-m5c" if cfg.use_m5c else "--no-use-m5c")
            if cfg.cpg_mask_policy:
                cmd += ["--cpg-mask-policy", cfg.cpg_mask_policy]
        if self._replace_chemistry:
            cmd.append("--replace-chemistry")
        if cfg.force_chemistry:
            # fiberhmm-call checks the reads against --seq (MM specs, m6A
            # presence) itself; --force-chemistry has already accepted them.
            cmd.append("--force-seq")
        # QC runs as its own step (fiberhmm-qc on the called BAM).
        cmd.append("--no-qc")
        # Capabilities of a newer fiberhmm-call (resume, progress). They do not
        # change the calls, so they are not part of the step fingerprint.
        extra: list[str] = []
        if resumable:
            extra += ["--resume", "--work-dir", self.call_work_dir,
                      "--progress-json", self.call_progress_path]
        return cmd + list(cfg.call_args), extra

    @property
    def call_work_dir(self) -> str:
        return os.path.join(self.outdir, STATE_DIR, "call_work")

    def _prepare_call_state(self, fingerprint: dict, resumable: bool) -> None:
        """Keep the resumable calling state only if it belongs to this call.

        ``--redo`` reaching the call, or an alignment rerun in this run,
        discards it. An interrupted call made with other settings or inputs is
        refused like a finished one (``--redo call`` discards it); calling in
        streaming mode does not use it.
        """
        work = self.call_work_dir
        started_path = os.path.join(self.outdir, STATE_DIR, "call.started")
        discard = self._forced("call") or not resumable
        # The work directory has its own owner lock (fiberhmm-call --work-dir
        # holds it while running, possibly outside this pipeline): hold it
        # while the state is inspected and removed, and refuse a live owner.
        lock = self._lock_call_work(work)
        try:
            if not discard and lock is not None:
                try:
                    with open(started_path, encoding="utf-8") as handle:
                        started = json.load(handle).get("fingerprint")
                except (OSError, ValueError, AttributeError):
                    started = None
                if started != fingerprint:
                    changed = fingerprint_changes(started or {}, fingerprint)
                    if self._upstream_reran(changed, {"aligned": "align"}) or started is None:
                        discard = True
                    else:
                        self._refuse("call", changed, what="interrupted calling state")
            if discard:
                if lock is not None:
                    _empty_locked_dir(work)
                    lock.remove = True
                with contextlib.suppress(FileNotFoundError):
                    os.remove(started_path)
        finally:
            if lock is not None:
                lock.release()
                if discard:
                    with contextlib.suppress(OSError):
                        os.rmdir(work)
        if resumable:
            write_json_atomic(started_path, {"fingerprint": fingerprint})

    @staticmethod
    def _lock_call_work(work: str):
        """The call work directory's own lock (as ``fiberhmm-call`` takes it),
        or None when there is no work directory."""
        from fiberhmm.inference.region_resume import LOCK
        from fiberhmm.io.run_state import DirectoryBusy, DirectoryLock

        if not os.path.isdir(work):
            return None
        try:
            return DirectoryLock(os.path.join(work, LOCK),
                                 "fiberhmm-call work directory").acquire()
        except DirectoryBusy as exc:
            raise PipelineError(
                f"the calling work directory is in use: {exc}",
                hint="Another fiberhmm-call (or pipeline) is running on this output "
                     "directory's calling state; wait for it to finish or stop it.") from None

    def step_call(self) -> None:
        cfg = self.config
        fingerprint = self._call_fingerprint()
        if self._is_complete("call", fingerprint, upstream={"aligned": "align"}):
            return
        supported = call_supported_options()
        resumable = self.resumable_call(supported)
        # Refuses an incompatible interrupted call before OUTDIR changes.
        self._prepare_call_state(fingerprint, resumable)
        self._start_step("call")
        clear_marker(self.outdir, "call")
        cmd, extra = self.call_command(supported, resumable)
        total = (self.stats.get("align") or {}).get("kept")
        if total is None and not resumable:
            total = _primary_count(self.aligned_bam)
        self.progress.step("call", "running", done=0, total=None if resumable else total,
                           unit="regions" if resumable else "reads",
                           message="duplicate marking, SNP screen and NRL estimate")
        log_path = os.path.join(self.outdir, "logs", "fiberhmm-call.log")
        self.log("call: " + " ".join(shlex.quote(c) for c in cmd + extra))
        if resumable:
            self.log("call: resumable region-parallel mode (an interrupted run "
                     "continues from its finished regions)")
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.call_progress_path)

        def on_line(line: str) -> None:
            parsed = parse_call_progress(line)
            if parsed and not resumable:
                done, rate = parsed
                self.progress.step("call", "running", done=done, total=total,
                                   unit="reads", rate=rate)

        relay = _ProgressRelay(self.call_progress_path, self.progress) if resumable else None
        on_tick = relay.poll if relay else None
        try:
            code = self._run_logged(cmd + extra, log_path, on_line, on_tick)
        finally:
            if relay is not None:
                relay.poll()
        if code != 0:
            raise PipelineError(f"fiberhmm-call failed (exit {code})",
                                hint=f"Last lines of {log_path}:\n" + _tail(log_path, 20))
        if _ensure_reference_header(self.called_bam, self.reference, cfg.cores):
            self.log("call: reference identity (@SQ M5, FIBERHMM-REFERENCE) added "
                     "to the called BAM header")
        with pysam.AlignmentFile(self.called_bam, check_sq=False) as bam:
            ok, why = header_matches_reference(bam.header, self.reference)
        if not ok:
            raise PipelineError(f"the called BAM does not match the reference: {why}",
                                hint="Check that --reference is the reference the reads "
                                     "were aligned to, or realign them.")
        summary = {"log": log_path, **_count_called(self.called_bam)}
        self.stats["call"] = summary
        self._write_marker("call", fingerprint,
                           {"bam": self.called_bam, "bai": self.called_bam + ".bai"}, summary)
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(self.outdir, STATE_DIR, "call.started"))
        message = (f"{summary['called']} of {summary['records']} reads called"
                   + (f", {summary['duplicate_flagged']} duplicate-flagged"
                      if summary["duplicate_flagged"] else ""))
        self.log(f"call: {message}")
        self.progress.step("call", "done", done=summary["records"],
                           total=summary["records"], unit="reads", message=message)

    def step_qc(self) -> None:
        """``fiberhmm-qc`` on the called BAM. A QC failure never fails the run:
        the called BAM is valid without it."""
        cfg = self.config
        # outputs.json publishes QC only from this: the QC of the current called
        # BAM, validated by its marker (None when QC is off or failed).
        self.qc_outputs = None
        if not cfg.qc:
            self.progress.step("qc", "skipped", message="--no-qc")
            return
        fingerprint = {"called": file_fingerprint(self.called_bam, self.memo),
                       "fiberhmm": __version__,
                       "prob_threshold": cfg.prob_threshold, "min_mapq": cfg.min_mapq,
                       "snp_mask": (file_fingerprint(cfg.snp_mask, self.memo)
                                    if cfg.snp_mask else None)}
        if self._is_complete("qc", fingerprint, upstream={"called": "call"}):
            self.qc_outputs = dict((read_marker(self.outdir, "qc") or {}).get("outputs") or {})
            self.stats["qc"] = {"verdicts": qc_verdicts(self.qc_outputs.get("json"))}
            return
        self._start_step("qc")
        clear_marker(self.outdir, "qc")
        # Files of an earlier QC run must not pass for this one's.
        for leftover in self.qc_files().values():
            with contextlib.suppress(FileNotFoundError):
                os.remove(leftover)
        self.progress.step("qc", "running")
        cmd = [sys.executable, "-m", "fiberhmm.cli.qc", "-i", self.called_bam,
               "-o", os.path.dirname(self.qc_prefix), "--min-mapq", str(cfg.min_mapq),
               *self.qc_assay_args()]
        if cfg.prob_threshold is not None:
            cmd += ["--prob-threshold", str(cfg.prob_threshold)]
        if cfg.snp_mask:
            cmd += ["--snp-mask", os.path.abspath(cfg.snp_mask)]
        log_path = os.path.join(self.outdir, "logs", "fiberhmm-qc.log")
        self.log("qc: " + " ".join(shlex.quote(c) for c in cmd))
        code = self._run_logged(cmd, log_path)
        files = self.qc_files()
        if code != 0 or "json" not in files:
            self.log(f"qc: fiberhmm-qc could not run (exit {code}); see {log_path}. "
                     "The called BAM is complete.", "warning")
            self.progress.step("qc", "skipped", message=f"fiberhmm-qc failed (exit {code})")
            return
        verdicts = qc_verdicts(files["json"])
        self.stats["qc"] = {"verdicts": verdicts}
        self._write_marker("qc", fingerprint, files, self.stats["qc"])
        self.qc_outputs = files
        message = f"QC {verdicts.get('overall', '?')}"
        if verdicts.get("overall_score") is not None:
            message += f" ({verdicts['overall_score']:.0f}/100)"
        self.log(f"qc: {message}")
        self.progress.step("qc", "done", message=message)

    def qc_assay_args(self) -> list[str]:
        """The assay this run called, stated to fiberhmm-qc (never guessed)."""
        cfg = self.config
        if cfg.enzyme in DAF_ENZYMES:
            return ["--mode", "daf", "--enzyme", cfg.enzyme]
        if cfg.enzyme == "hia5" and cfg.seq in ("pacbio", "nanopore"):
            return ["--mode", f"{cfg.seq}-fiber", "--enzyme", "hia5"]
        return []

    def _run_logged(self, cmd: list[str], log_path: str,
                    on_line: Optional[Callable[[str], None]] = None,
                    on_tick: Optional[Callable[[], None]] = None) -> int:
        """Run ``cmd`` in its own process group, teeing its output to ``log_path``."""
        env = _child_env()
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}\n")
            log.flush()
            proc = _register(subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                start_new_session=True))
            ticker = mm2.progress_ticker(1.0, on_tick) if on_tick else None
            try:
                buffer = b""
                while True:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", "replace")
                    log.write(text)
                    log.flush()
                    if self.config.verbose:
                        sys.stderr.write(text)
                    buffer += chunk
                    *lines, buffer = re.split(rb"[\r\n]", buffer)
                    for line in lines:
                        if line.strip() and on_line:
                            on_line(line.decode("utf-8", "replace"))
                if buffer.strip() and on_line:
                    on_line(buffer.decode("utf-8", "replace"))
                code = proc.wait()
            finally:
                if ticker is not None:
                    ticker.set()
                if proc.poll() is None:
                    terminate_children()
                _unregister(proc)
                proc.stdout.close()
        return code

    def qc_files(self) -> dict:
        files = {}
        for key, suffix in (("json", ".qc.json"), ("pdf", ".qc.pdf"),
                            ("png", ".qc.png"), ("tsv", ".qc.tsv"),
                            ("curves", ".qc.curves.json")):
            path = self.qc_prefix + suffix
            if os.path.exists(path):
                files[key] = path
        return files

    # -- tracks ----------------------------------------------------------------
    def step_tracks(self) -> None:
        cfg = self.config
        tracks_dir = os.path.join(self.outdir, "tracks")
        bigbed = shutil.which("bedToBigBed") is not None
        fingerprint = {"called": file_fingerprint(self.called_bam, self.memo),
                       "bigbed": bigbed, "enzyme": cfg.enzyme,
                       "prob_threshold": cfg.prob_threshold, "min_mapq": cfg.min_mapq}
        if self._is_complete("tracks", fingerprint, upstream={"called": "call"}):
            marker = read_marker(self.outdir, "tracks") or {}
            self.track_files = list((marker.get("outputs") or {}).get("files") or [])
            return
        inventory = os.path.join(self.outdir, STATE_DIR, "tracks.files.json")
        previous = list(((read_marker(self.outdir, "tracks") or {}).get("outputs") or {})
                        .get("files") or [])
        try:
            with open(inventory, encoding="utf-8") as handle:
                previous += [str(f) for f in json.load(handle)]
        except (OSError, ValueError, TypeError):
            pass
        # Kept before the marker is cleared: an interrupted or failed run must
        # not lose the list of tracks published so far.
        previous = sorted({os.path.abspath(f) for f in previous})
        write_json_atomic(inventory, previous)
        self._start_step("tracks")
        clear_marker(self.outdir, "tracks")
        self.progress.step("tracks", "running")
        layers = ["--nucleosome", "--msp", "--tf"]
        layers.append("--deam" if cfg.enzyme in DAF_ENZYMES else "--m6a")
        # fiberhmm-extract writes into a private directory: everything in it is
        # this run's output, whatever names extract derives from the BAM's.
        staging = os.path.join(self.outdir, STATE_DIR, "tracks_staging")
        shutil.rmtree(staging, ignore_errors=True)
        cmd = [sys.executable, "-m", "fiberhmm.cli.extract_tags", "-i", self.called_bam,
               "-o", staging, "-c", str(cfg.cores), "-q", str(cfg.min_mapq), *layers]
        if cfg.prob_threshold is not None:
            cmd += ["-p", str(cfg.prob_threshold)]
        if not bigbed:
            cmd.append("--bed-only")
            self.log("tracks: bedToBigBed not found; writing BED only", "warning")
        log_path = os.path.join(self.outdir, "logs", "fiberhmm-extract.log")
        self.log("tracks: " + " ".join(shlex.quote(c) for c in cmd))
        code = self._run_logged(cmd, log_path)
        if code != 0:
            shutil.rmtree(staging, ignore_errors=True)
            raise PipelineError(f"fiberhmm-extract failed (exit {code})",
                                hint=f"Last lines of {log_path}:\n" + _tail(log_path, 20))
        produced = sorted(f for f in os.listdir(staging) if f.endswith((".bb", ".bed")))
        os.makedirs(tracks_dir, exist_ok=True)
        self.track_files = [os.path.join(tracks_dir, name) for name in produced]
        # Every track this pipeline has published here (kept until cleanup is
        # done, so an interrupted or failed run does not lose it).
        write_json_atomic(inventory, sorted(set(previous) | set(self.track_files)))
        for name, target in zip(produced, self.track_files):
            os.replace(os.path.join(staging, name), target)
        shutil.rmtree(staging, ignore_errors=True)
        # Tracks of the earlier run that this one did not make again (a layer
        # with no features now) would otherwise pass for current ones.
        for stale in previous:
            stale = os.path.abspath(stale)
            if (stale not in self.track_files
                    and os.path.dirname(stale) == os.path.abspath(tracks_dir)):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(stale)
        write_json_atomic(inventory, self.track_files)
        self._write_marker("tracks", fingerprint, {"files": self.track_files},
                           {"n_files": len(self.track_files)})
        self.progress.step("tracks", "done", message=f"{len(self.track_files)} files")

    # -- outputs -----------------------------------------------------------------
    def open_region(self) -> Optional[str]:
        if self.regions:
            return format_region(*self.regions[0])
        if self.reference.is_plasmid:
            return None
        return densest_region(self.called_bam)

    def write_outputs(self) -> dict:
        ref = self.reference
        qc = dict(self.qc_outputs or {})
        command = ["fiberbrowser", "-f", ref.fasta, "--dataset",
                   f"{self.called_bam}:{self.sample}"]
        outputs = {
            "schema": OUTPUTS_SCHEMA,
            "version": __version__,
            "sample": self.sample,
            "enzyme": self.config.enzyme,
            "aligned_bam": self.aligned_bam,
            "called_bam": self.called_bam,
            "qc_report": qc.get("pdf") or qc.get("json"),
            "qc": {
                "json": qc.get("json"),
                "report": qc.get("pdf") or qc.get("png"),
                "curves": qc.get("curves"),
                "tsv": qc.get("tsv"),
                "plots": {name: qc[key] for name, key in (("overview_png", "png"),
                                                          ("overview_pdf", "pdf"))
                          if key in qc},
                "verdicts": qc_verdicts(qc.get("json")),
            } if qc.get("json") else None,
            "settings": self.config.calling_settings(),
            "tracks": list(self.track_files),
            "reference_fasta": ref.fasta,
            "plasmid_map": ref.plasmid_map,
            "contigs": [{"name": c.name, "length": c.length, "circular": c.circular,
                         "md5": c.md5} for c in ref.contigs],
            "open": {
                "fasta": ref.fasta,
                "datasets": [self.called_bam],
                "plasmid_maps": [ref.plasmid_map] if ref.plasmid_map else [],
                "region": self.open_region(),
            },
            "fiberbrowser_command": " ".join(shlex.quote(c) for c in command),
            "stats": self.stats,
        }
        outputs = json.loads(json.dumps(outputs, default=str))
        write_json_atomic(os.path.join(self.outdir, "outputs.json"), outputs)
        self._save_memo()
        return outputs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _child_env() -> dict:
    """Environment for fiberhmm child commands: the same FiberHMM as this process."""
    import fiberhmm
    env = dict(os.environ)
    env.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")
    root = os.path.dirname(os.path.dirname(os.path.abspath(fiberhmm.__file__)))
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if root not in parts:
        env["PYTHONPATH"] = os.pathsep.join([root] + parts)
    return env


def _empty_locked_dir(path: str) -> None:
    """Remove everything in ``path`` except its lock file (still held)."""
    from fiberhmm.inference.region_resume import LOCK

    for entry in os.scandir(path):
        if entry.name == LOCK:
            continue
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path, ignore_errors=True)
        else:
            with contextlib.suppress(FileNotFoundError):
                os.remove(entry.path)


def _call_arg_value(call_args, flag) -> str:
    """The value given to ``flag`` in a --call-args list ('' when absent)."""
    for index, argument in enumerate(call_args or ()):
        if argument == flag and index + 1 < len(call_args):
            return call_args[index + 1]
        if argument.startswith(flag + "="):
            return argument.split("=", 1)[1]
    return ""


def _group_by_name(records):
    """Consecutive records sharing a query name (minimap2 output order).

    The names are the feeder's internal ones (one serial per input record), so
    a group is exactly one input record's alignments."""
    group = []
    name = None
    for read in records:
        if read.query_name != name and group:
            yield group
            group = []
        name = read.query_name
        group.append(read)
    if group:
        yield group


def _sort_index_publish(unsorted: str, final: str, cores: int) -> None:
    """Coordinate-sort and index with pysam, then publish BAM + index together
    (:func:`fiberhmm.inference.bam_output.commit_output`: never a mismatched
    pair, and the previous pair is restored if publication fails)."""
    from fiberhmm.inference.bam_output import commit_output
    tmp_sorted = f"{unsorted}.{os.getpid()}.sorted.bam"
    try:
        pysam.sort("--no-PG", "-o", tmp_sorted, "-@", str(max(1, min(4, cores))),
                   "-T", os.path.join(os.path.dirname(unsorted), "sort"), unsorted)
        pysam.index(tmp_sorted)
        commit_output(tmp_sorted, final)
    finally:
        for leftover in (tmp_sorted, tmp_sorted + ".bai"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(leftover)
    with contextlib.suppress(FileNotFoundError):
        os.remove(unsorted)


def _has_md_tags(bam, sample: int = 200) -> tuple[bool, str]:
    seen = 0
    for read in bam.fetch(until_eof=True):
        if read.is_unmapped or read.is_secondary:
            continue
        if not read.has_tag("MD"):
            return False, "its reads have no MD tags (DAF-seq calling needs them)"
        seen += 1
        if seen >= sample:
            break
    return (seen > 0), ("" if seen else "it has no mapped reads")


def _reads_match_reference(bam_path: str, fasta: str, sample: int = 200,
                           max_bases: int = 2_000_000) -> tuple[bool, str]:
    """Whether the first mapped reads were aligned to the sequence in ``fasta``.

    For a BAM whose @SQ lines carry no M5 checksum. With MD tags, the reference
    bases the aligner saw (MD) must equal the FASTA's (at most 0.1% may differ,
    e.g. IUPAC codes); without MD, at least 70% of the aligned read bases must
    match the FASTA (a different sequence of the same length matches ~25%).
    Positions past a contig's end (circular records) wrap.
    """
    from fiberhmm.daf.aligned_arrays import md_disagrees_with_cigar

    compared = differ = reads = 0
    used_md = False
    with pysam.FastaFile(fasta) as reference, pysam.AlignmentFile(bam_path) as bam:
        cache: dict[str, str] = {}
        for read in bam.fetch(until_eof=True):
            if (read.is_unmapped or read.is_secondary or read.is_supplementary
                    or not read.query_sequence or read.reference_name not in reference.references):
                continue
            name = read.reference_name
            if name not in cache:
                cache[name] = reference.fetch(name).upper()
            contig = cache[name]
            length = len(contig)
            if read.has_tag("MD"):
                used_md = True
                # An MD that does not describe the CIGAR has no defined
                # reference bases (pysam reads undefined memory for a short
                # one): the input is realigned, as for an unreadable MD.
                if md_disagrees_with_cigar(read):
                    return False, "its MD tags do not match their CIGAR strings"
                try:
                    pairs = read.get_aligned_pairs(matches_only=True, with_seq=True)
                except (ValueError, KeyError, AssertionError):
                    return False, "its MD tags cannot be read"
                for _, rpos, base in pairs:
                    compared += 1
                    differ += (base or "N").upper() != contig[rpos % length]
            else:
                query = read.query_sequence.upper()
                for qpos, rpos in read.get_aligned_pairs(matches_only=True):
                    compared += 1
                    differ += query[qpos] != contig[rpos % length]
            reads += 1
            if reads >= sample or compared >= max_bases:
                break
    if not compared:
        return False, "it has no mapped reads to compare with the reference"
    fraction = differ / compared
    if used_md and fraction > 0.001:
        return False, (f"its @SQ lines have no M5 checksum and {fraction:.1%} of the reference "
                       "bases in its MD tags differ from --reference")
    if not used_md and fraction > 0.3:
        return False, (f"its @SQ lines have no M5 checksum and {fraction:.0%} of the aligned "
                       "bases differ from --reference")
    return True, ""


class _ProgressRelay:
    """Translate fiberhmm-call ``fiberhmm.progress.v1`` lines into ``step`` events."""

    def __init__(self, path: str, progress: ProgressReporter):
        self.path = path
        self.progress = progress
        self.position = 0
        self.lock = threading.Lock()

    def poll(self) -> None:
        with self.lock:
            try:
                with open(self.path, encoding="utf-8") as handle:
                    handle.seek(self.position)
                    while True:
                        line = handle.readline()
                        if not line or not line.endswith("\n"):
                            break
                        self.position = handle.tell()
                        try:
                            self.translate(json.loads(line))
                        except (ValueError, TypeError):
                            continue
            except FileNotFoundError:
                return

    def translate(self, event: dict) -> None:
        kind = event.get("event")
        done = event.get("regions_done")
        total = event.get("regions_total")
        if kind == "start":
            reused = event.get("regions_reused") or 0
            self.progress.step("call", "running", done=done, total=total, unit="regions",
                               message=(f"resuming: {reused} of {total} regions already "
                                        "done" if reused else f"{total} regions"))
        elif kind == "region":
            reads = event.get("reads") or 0
            rate = event.get("reads_per_s")
            self.progress.step(
                "call", "running", done=done, total=total, unit="regions",
                eta_s=event.get("eta_s"),
                message=f"{reads:,} reads" + (f", {rate:,.0f} reads/s" if rate else ""))
        elif kind == "merge":
            self.progress.step("call", "running", done=total, total=total, unit="regions",
                               message="merging regions")


def _mapped_regions(bam_path: str, region_size: int) -> tuple[int, int]:
    """(mapped records, regions of ``region_size`` on contigs that have reads)."""
    try:
        with pysam.AlignmentFile(bam_path) as bam:
            lengths = dict(zip(bam.references, bam.lengths))
            records = regions = 0
            for stat in bam.get_index_statistics():
                if stat.mapped:
                    records += stat.mapped
                    regions += -(-lengths.get(stat.contig, 1) // region_size)
            return records, regions
    except (OSError, ValueError):
        return 0, 0


def _primary_count(bam_path: str) -> Optional[int]:
    try:
        with pysam.AlignmentFile(bam_path) as bam:
            return sum(1 for r in bam.fetch(until_eof=True)
                       if not (r.is_secondary or r.is_supplementary or r.is_unmapped))
    except (OSError, ValueError):
        return None


def _ensure_reference_header(bam_path: str, ref: ReferenceInfo, cores: int) -> bool:
    """Add @SQ M5/TP and FIBERHMM-REFERENCE to a BAM that lacks them (re-index)."""
    with pysam.AlignmentFile(bam_path, check_sq=False) as bam:
        header = bam.header.to_dict()
    has_m5 = all(sq.get("M5") for sq in header.get("SQ", []))
    needs_comment = ref.is_plasmid or bool(ref.circular_contigs)
    if has_m5 and (not needs_comment or declared_references(header)):
        return False
    decorate_header(header, ref)
    from fiberhmm.inference.bam_output import commit_output
    directory, name = os.path.split(bam_path)
    tmp = os.path.join(directory, f".{name}.{os.getpid()}.reheader.tmp.bam")
    try:
        with pysam.AlignmentFile(bam_path, check_sq=False) as src, \
                pysam.AlignmentFile(tmp, "wb", header=header,
                                    threads=max(1, min(4, cores))) as out:
            for read in src.fetch(until_eof=True):
                out.write(read)
        pysam.index(tmp)
        commit_output(tmp, bam_path)
    finally:
        for leftover in (tmp, tmp + ".bai"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(leftover)
    return True


def densest_region(bam_path: str, bin_size: int = 1000, max_reads: int = 200_000,
                   min_fraction: float = 0.2) -> Optional[str]:
    """The covered span around the most covered bin (amplicon runs), or ``None``.

    Coverage is counted in ``bin_size`` bins from up to ``max_reads`` primary
    reads; the span extends from the peak while bins keep at least 20% of the
    peak depth. ``None`` when that span holds under ``min_fraction`` of the
    reads (whole-genome data has no single region to open).
    """
    bins: collections.Counter = collections.Counter()
    spans = []
    with pysam.AlignmentFile(bam_path) as bam:
        lengths = dict(zip(bam.references, bam.lengths))
        for i, read in enumerate(bam.fetch(until_eof=True)):
            if i >= max_reads:
                break
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            chrom = read.reference_name
            spans.append((chrom, read.reference_start, read.reference_end))
            for b in range(read.reference_start // bin_size,
                           (read.reference_end - 1) // bin_size + 1):
                bins[(chrom, b)] += 1
    if not bins:
        return None
    (chrom, peak), depth = bins.most_common(1)[0]
    left = right = peak
    while bins.get((chrom, left - 1), 0) >= 0.2 * depth:
        left -= 1
    while bins.get((chrom, right + 1), 0) >= 0.2 * depth:
        right += 1
    start = max(0, left * bin_size)
    end = min(lengths.get(chrom, (right + 1) * bin_size), (right + 1) * bin_size)
    inside = sum(1 for c, s, e in spans if c == chrom and s < end and e > start)
    if inside < min_fraction * len(spans):
        return None
    return format_region(chrom, start, end)


def _tail(path: str, lines: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read().replace("\r", "\n")
        return "\n".join([line for line in text.splitlines() if line.strip()][-lines:])
    except OSError:
        return ""


def qc_verdicts(path: Optional[str]) -> dict:
    """PASS/WARN/FAIL/INSUFFICIENT verdicts and scores from a ``.qc.json``."""
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    out = {}
    for key in ("overall", "signal", "periodicity", "efficiency", "background"):
        section = data.get(key) or {}
        if "status" in section:
            out[key] = section["status"]
        if section.get("score") is not None:
            out[f"{key}_score"] = round(float(section["score"]), 1)
    dedup = data.get("deduplication") or {}
    if dedup.get("duplicate_fraction") is not None:
        out["duplicate_fraction"] = dedup["duplicate_fraction"]
    if (data.get("overall") or {}).get("verdict_basis"):
        out["verdict_basis"] = data["overall"]["verdict_basis"]
    states = data.get("state_rates") or {}
    if states.get("available"):
        # Rates split by FiberHMM state (fiberhmm-qc schema 1.1, additive).
        out["state_rates"] = {
            "source": states.get("source"),
            "overall_rate": (states.get("all_states") or {}).get("aggregate_rate"),
            "msp_rate": (states.get("msp") or {}).get("median_per_read_rate"),
            "outside_msp_rate": (states.get("outside_msp") or {}).get("median_per_read_rate"),
            "aggregate_msp_rate": (states.get("msp") or {}).get("aggregate_rate"),
            "aggregate_outside_msp_rate": (states.get("outside_msp") or {}).get("aggregate_rate"),
            "msp_to_outside_ratio": states.get("msp_to_outside_ratio"),
            "msp_length_fraction": states.get("msp_length_fraction"),
            "min_msp_bp": (states.get("definition") or {}).get("min_msp_bp"),
        }
    return out


def _count_called(bam_path: str) -> dict:
    total = called = duplicates = 0
    with pysam.AlignmentFile(bam_path, check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            if read.is_secondary or read.is_supplementary:
                continue
            total += 1
            if read.is_duplicate:
                duplicates += 1
            if read.has_tag("MA") or read.has_tag("ns"):
                called += 1
    return {"records": total, "called": called, "duplicate_flagged": duplicates}
