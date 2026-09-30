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
from fiberhmm.pipeline import aligner as mm2
from fiberhmm.pipeline.circular import hard_clip, merge_origin_pieces
from fiberhmm.pipeline.progress import (
    STATE_DIR,
    ProgressReporter,
    file_fingerprint,
    fingerprint_changes,
    outputs_exist,
    read_marker,
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
    hard_clip: Optional[bool] = None  # None: on for DAF enzymes
    origin_merge: bool = True
    origin_tolerance: int = 100
    dedup: str = "auto"  # auto | on | off (fiberhmm-call: automatic for DAF files)
    dedup_mode: str = "flag"  # flag (mark 0x400, keep reads) | collapse
    snp_screen: str = "auto"  # auto | on | off
    snp_mask: Optional[str] = None  # user BED of SNP sites to mask
    chimera_filter: bool = True
    primary: bool = True
    prob_threshold: Optional[int] = None
    use_m5c: Optional[bool] = None  # None: fiberhmm-call default (on for ddda)
    cpg_mask_policy: Optional[str] = None
    qc: bool = True
    call_args: list[str] = field(default_factory=list)
    call_mode: str = "auto"  # auto | streaming | resumable
    tracks: bool = False
    aligner: str = "auto"
    redo: Optional[str] = None  # "all" or a step: redo it and the later ones
    verbose: bool = False
    quiet: bool = False

    def resolved_seq(self) -> Optional[str]:
        if self.seq:
            return self.seq
        return "nanopore" if self.enzyme in DAF_ENZYMES else None

    def resolved_hard_clip(self) -> bool:
        if self.hard_clip is not None:
            return self.hard_clip
        return self.enzyme in DAF_ENZYMES

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
            "primary_only": self.primary,
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
_CHILDREN_LOCK = threading.Lock()


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
    """SIGTERM (and SIGINT) stop the children and raise :class:`PipelineCancelled`."""

    def handler(signum, frame):
        terminate_children()
        raise PipelineCancelled(signum=signum)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# ---------------------------------------------------------------------------
# fiberhmm-call capabilities and progress
# ---------------------------------------------------------------------------

def call_supported_options() -> set[str]:
    """Option strings ``fiberhmm-call`` accepts (``--progress-json``, ``--resume``...).

    The parser is captured without running the command, the way
    ``tools/gen_cli_reference.py`` documents it.
    """
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
        return {flag for action in captured.parser._actions
                for flag in action.option_strings}
    finally:
        argparse.ArgumentParser.parse_known_args = original
        sys.argv = saved
    return set()


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
        self._redo_from = None
        if config.redo:
            self._redo_from = (0 if config.redo == "all" else
                               self.steps.index(config.redo) if config.redo in self.steps
                               else len(self.steps))

    def _setup(self) -> None:
        cfg = self.config
        cfg.reads = expand_read_inputs(cfg.reads)
        self.regions = [parse_region(r) for r in cfg.regions]
        self.sample = cfg.sample or default_sample_name(cfg.reads[0])
        os.makedirs(os.path.join(self.outdir, STATE_DIR), exist_ok=True)
        os.makedirs(os.path.join(self.outdir, "logs"), exist_ok=True)
        self._log_handle = open(os.path.join(self.outdir, "logs", "pipeline.log"), "a",
                                encoding="utf-8")
        self.aligned_bam = os.path.join(self.outdir, f"{self.sample}.aligned.bam")
        self.called_bam = os.path.join(self.outdir, f"{self.sample}.fiberhmm.bam")
        self.qc_prefix = os.path.join(self.outdir, "qc", f"{self.sample}.fiberhmm")
        self.call_progress_path = os.path.join(self.outdir, STATE_DIR,
                                               "call.progress.jsonl")

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

    def _is_complete(self, step: str, fingerprint: dict) -> bool:
        """True when the step can be skipped; refuse a changed setup."""
        if self._forced(step):
            return False
        marker = read_marker(self.outdir, step)
        if not marker:
            return False
        if marker.get("fingerprint") != fingerprint:
            changed = fingerprint_changes(marker.get("fingerprint") or {}, fingerprint)
            raise PipelineError(
                f"{self.outdir} already holds a '{step}' result made with different "
                f"settings or inputs ({', '.join(changed) or 'setup'})",
                hint=f"Use a new output directory (-o), or add --redo {step} to "
                     "replace the earlier result.")
        if not outputs_exist(marker.get("outputs")):
            return False
        self.stats[step] = marker.get("summary") or {}
        self.progress.step(step, "skipped", message="already complete")
        self.log(f"{step}: already complete, skipped")
        return True

    # -- run -----------------------------------------------------------------
    def run(self) -> dict:
        try:
            self._setup()
            self.progress.emit("start", version=__version__, sample=self.sample,
                               steps=list(self.steps), outdir=self.outdir,
                               settings=self.config.calling_settings())
            self.step_prepare_reference()
            self.step_index()
            self.step_align()
            self.step_call()
            self.step_qc()
            if self.config.tracks:
                self.step_tracks()
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
        if self._log_handle is not None:
            with contextlib.suppress(Exception):
                self._log_handle.close()

    # -- prepare_reference ----------------------------------------------------
    def step_prepare_reference(self) -> None:
        cfg = self.config
        self.progress.step("prepare_reference", "running")
        try:
            self.reference = prepare_reference(cfg.reference, self.outdir, cfg.topology,
                                               cache_dir=mm2.index_cache_dir())
        except (OSError, ValueError) as exc:
            raise PipelineError(f"reference: {exc}",
                                hint="Give a FASTA or a plasmid map (.dna, .gb/.gbk, .embl)")
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

    def _decide_alignment(self) -> None:
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
            has_index = bam.has_index()
        if not ok:
            self.log(f"align: {os.path.basename(path)} is realigned: {why}")
            return
        if not has_index:
            self.log(f"align: {os.path.basename(path)} has no index; it is realigned "
                     "so the output is sorted and indexed")
            return
        self.use_aligned_input = path
        self.log(f"align: {os.path.basename(path)} is already aligned to this reference")

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
                self.reference.fasta, self.reference.source_sha256, cfg.preset(),
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
            "reads": [file_fingerprint(rf.path) for rf in self.read_files],
            "reference": [asdict(c) for c in self.reference.contigs],
            "preset": cfg.preset(),
            "min_mapq": cfg.min_mapq,
            "hard_clip": cfg.resolved_hard_clip(),
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
        self.progress.step("align", "running", done=0, unit="reads")
        if self.use_aligned_input:
            stats = self._subset_aligned_input()
        else:
            stats = self._align()
        self.stats["align"] = stats
        write_marker(self.outdir, "align", fingerprint,
                     {"bam": self.aligned_bam, "bai": self.aligned_bam + ".bai"}, stats)
        self.progress.step("align", "done", done=stats.get("kept"), unit="reads",
                           message=stats.get("message"))

    def _overlaps_regions(self, read) -> bool:
        if not self.regions:
            return True
        chrom = read.reference_name
        return any(chrom == c and read.reference_start < e and read.reference_end > s
                   for c, s, e in self.regions)

    def _subset_aligned_input(self) -> dict:
        tmp = os.path.join(self.outdir, STATE_DIR, "tmp", f"{self.sample}.unsorted.bam")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        kept = 0
        seen: set = set()
        with pysam.AlignmentFile(self.use_aligned_input) as src:
            header = decorate_header(src.header.to_dict(), self.reference)
            with pysam.AlignmentFile(tmp, "wb", header=header) as out:
                for chrom, start, end in self.regions:
                    for read in src.fetch(chrom, start, end):
                        key = (read.query_name, read.flag, read.reference_start)
                        if key in seen:
                            continue
                        seen.add(key)
                        out.write(read)
                        kept += 1
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
        feeder = mm2.ReadFeeder(self.read_files, carry_tags)
        rg = {"ID": self.sample, "SM": self.sample,
              "PL": "ONT" if (cfg.resolved_seq() or "nanopore") == "nanopore" else "PACBIO"}
        rg_line = "@RG\\t" + "\\t".join(f"{k}:{v}" for k, v in rg.items())
        tmpdir = os.path.join(self.outdir, STATE_DIR, "tmp")
        os.makedirs(tmpdir, exist_ok=True)
        log_path = os.path.join(self.outdir, "logs", "minimap2.log")
        if self.aligner.kind == "minimap2":
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
            header = mm2.mappy_header(ref.contigs, rg, source_program)
            self.log(f"align: mappy {self.aligner.version}, preset {preset}")
            stream = mm2.MappyStream(self.index_path, preset, cfg.cores, feeder, header,
                                     carry_tags, self.sample)

        header = stream.header.to_dict()
        if not any(pg.get("ID") == source_program["ID"] for pg in header.get("PG", [])):
            header.setdefault("PG", []).append(source_program)
        decorate_header(header, ref)
        pgs = header.setdefault("PG", [])
        pipeline_pg = {
            "ID": "fiberhmm-pipeline", "PN": "fiberhmm-pipeline", "VN": __version__,
            "CL": "fiberhmm-pipeline " + " ".join(shlex.quote(a) for a in sys.argv[1:]),
            "DS": (f"primary MAPQ>={cfg.min_mapq}; "
                   f"origin_merge={'on' if cfg.origin_merge else 'off'}; "
                   f"hard_clip={'on' if cfg.resolved_hard_clip() else 'off'}"),
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
                 "kept": 0, "origin_merged": 0, "hard_clipped_reads": 0,
                 "hard_clipped_bases": 0}
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
                    record = self._process_group(group, circular, sequences, stats)
                    if record is not None:
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
            + (f"; {stats['hard_clipped_reads']} hard-clipped"
               if cfg.resolved_hard_clip() else ""))
        self.log(f"align: {stats['message']}")
        if stats["kept"] == 0:
            raise PipelineError(
                "no reads aligned to the reference",
                hint="Check that the reference matches the sample (for a plasmid, "
                     "the plasmid map; for amplicons, the genome).")
        return stats

    def _process_group(self, group, circular, sequences, stats):
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
            return None
        if primary.mapping_quality < cfg.min_mapq:
            stats["low_mapq"] += 1
            return None
        record = primary
        contig = primary.reference_name
        if contig in circular and cfg.origin_merge:
            for sup in supplementary:
                merged = merge_origin_pieces(primary, sup, sequences[contig],
                                             cfg.origin_tolerance)
                if merged is not None:
                    record = merged
                    stats["origin_merged"] += 1
                    break
        if not self._overlaps_regions(record):
            stats["outside_regions"] += 1
            return None
        if record.has_tag("SA"):
            record.set_tag("SA", None)
        if cfg.resolved_hard_clip():
            removed = hard_clip(record)
            if removed:
                stats["hard_clipped_reads"] += 1
                stats["hard_clipped_bases"] += removed
        stats["kept"] += 1
        return record

    # -- call + qc -------------------------------------------------------------
    def _call_fingerprint(self) -> dict:
        cfg = self.config
        return {
            "aligned": file_fingerprint(self.aligned_bam),
            "enzyme": cfg.enzyme,
            "seq": cfg.resolved_seq(),
            "min_mapq": cfg.min_mapq,
            "min_read_length": cfg.min_read_length,
            "dedup": cfg.dedup,
            "dedup_mode": cfg.dedup_mode,
            "snp_screen": cfg.snp_screen,
            "snp_mask": file_fingerprint(cfg.snp_mask) if cfg.snp_mask else None,
            "chimera_filter": cfg.chimera_filter,
            "primary": cfg.primary,
            "prob_threshold": cfg.prob_threshold,
            "use_m5c": cfg.use_m5c,
            "cpg_mask_policy": cfg.cpg_mask_policy,
            "call_args": list(cfg.call_args),
            "fiberhmm": __version__,
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
        if not cfg.primary:
            cmd.append("--no-primary")
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
        if cfg.enzyme == "ddda":
            if cfg.use_m5c is not None:
                cmd.append("--use-m5c" if cfg.use_m5c else "--no-use-m5c")
            if cfg.cpg_mask_policy:
                cmd += ["--cpg-mask-policy", cfg.cpg_mask_policy]
        # QC runs as its own step (fiberhmm-qc on the called BAM).
        cmd.append("--no-qc")
        # Capabilities of a newer fiberhmm-call (resume, progress). They do not
        # change the calls, so they are not part of the step fingerprint.
        extra: list[str] = []
        if resumable:
            extra += ["--resume", "--work-dir", os.path.join(self.outdir, STATE_DIR, "call_work"),
                      "--progress-json", self.call_progress_path]
        return cmd + list(cfg.call_args), extra

    def step_call(self) -> None:
        cfg = self.config
        fingerprint = self._call_fingerprint()
        if self._is_complete("call", fingerprint):
            return
        supported = call_supported_options()
        resumable = self.resumable_call(supported)
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
        summary = {"log": log_path, **_count_called(self.called_bam)}
        self.stats["call"] = summary
        write_marker(self.outdir, "call", fingerprint,
                     {"bam": self.called_bam, "bai": self.called_bam + ".bai"}, summary)
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
        if not cfg.qc:
            self.progress.step("qc", "skipped", message="--no-qc")
            return
        fingerprint = {"called": file_fingerprint(self.called_bam), "fiberhmm": __version__,
                       "prob_threshold": cfg.prob_threshold, "min_mapq": cfg.min_mapq}
        if self._is_complete("qc", fingerprint):
            return
        self.progress.step("qc", "running")
        cmd = [sys.executable, "-m", "fiberhmm.cli.qc", "-i", self.called_bam,
               "-o", os.path.dirname(self.qc_prefix), "--min-mapq", str(cfg.min_mapq)]
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
        write_marker(self.outdir, "qc", fingerprint, {"json": files["json"]},
                     self.stats["qc"])
        message = f"QC {verdicts.get('overall', '?')}"
        if verdicts.get("overall_score") is not None:
            message += f" ({verdicts['overall_score']:.0f}/100)"
        self.log(f"qc: {message}")
        self.progress.step("qc", "done", message=message)

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
        fingerprint = {"called": file_fingerprint(self.called_bam), "bigbed": bigbed,
                       "enzyme": cfg.enzyme}
        if self._is_complete("tracks", fingerprint):
            marker = read_marker(self.outdir, "tracks") or {}
            self.track_files = list((marker.get("outputs") or {}).get("files") or [])
            return
        self.progress.step("tracks", "running")
        layers = ["--nucleosome", "--msp", "--tf"]
        layers.append("--deam" if cfg.enzyme in DAF_ENZYMES else "--m6a")
        cmd = [sys.executable, "-m", "fiberhmm.cli.extract_tags", "-i", self.called_bam,
               "-o", tracks_dir, "-c", str(cfg.cores), *layers]
        if not bigbed:
            cmd.append("--bed-only")
            self.log("tracks: bedToBigBed not found; writing BED only", "warning")
        log_path = os.path.join(self.outdir, "logs", "fiberhmm-extract.log")
        self.log("tracks: " + " ".join(shlex.quote(c) for c in cmd))
        code = self._run_logged(cmd, log_path)
        if code != 0:
            raise PipelineError(f"fiberhmm-extract failed (exit {code})",
                                hint=f"Last lines of {log_path}:\n" + _tail(log_path, 20))
        stem = os.path.basename(self.called_bam)[:-len(".bam")]
        self.track_files = sorted(
            os.path.join(tracks_dir, f) for f in os.listdir(tracks_dir)
            if f.startswith(stem) and f.endswith((".bb", ".bed")))
        write_marker(self.outdir, "tracks", fingerprint, {"files": self.track_files},
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
        qc = self.qc_files()
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
        path = os.path.join(self.outdir, "outputs.json")
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(outputs, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
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


def _group_by_name(records):
    """Consecutive records sharing a query name (minimap2 output order)."""
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
    """Coordinate-sort and index with pysam, then move BAM + index into place."""
    tmp_sorted = unsorted + ".sorted.bam"
    pysam.sort("--no-PG", "-o", tmp_sorted, "-@", str(max(1, min(4, cores))),
               "-T", os.path.join(os.path.dirname(unsorted), "sort"), unsorted)
    pysam.index(tmp_sorted)
    os.replace(tmp_sorted + ".bai", final + ".bai")
    os.replace(tmp_sorted, final)
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
    tmp = bam_path + ".reheader.tmp.bam"
    with pysam.AlignmentFile(bam_path, check_sq=False) as src, \
            pysam.AlignmentFile(tmp, "wb", header=header,
                                threads=max(1, min(4, cores))) as out:
        for read in src.fetch(until_eof=True):
            out.write(read)
    pysam.index(tmp)
    os.replace(tmp + ".bai", bam_path + ".bai")
    os.replace(tmp, bam_path)
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
    for key in ("overall", "signal", "periodicity"):
        section = data.get(key) or {}
        if "status" in section:
            out[key] = section["status"]
        if section.get("score") is not None:
            out[f"{key}_score"] = round(float(section["score"]), 1)
    dedup = data.get("deduplication") or {}
    if dedup.get("duplicate_fraction") is not None:
        out["duplicate_fraction"] = dedup["duplicate_fraction"]
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
