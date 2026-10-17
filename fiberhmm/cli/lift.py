#!/usr/bin/env python3
"""fiberhmm-lift -- move FiberHMM BAMs to another assembly through a pangenome graph.

Runs the four stages of ``ft pg-lift --bam`` (``ft pg-pansn`` adding the
source panSN prefix, ``vg inject``, ``vg surject`` onto the target sample,
``ft pg-pansn --strip``) between a pre-lift stash and a post-lift repair
(:mod:`fiberhmm.io.graph_lift`), then sorts and indexes. The stages are run
here rather than through ``ft pg-lift --bam``, which fails on fibertools-rs
<= 0.13 (it passes ``--out`` to ``ft pg-pansn``, which takes the output path
positionally).

``--repair-only`` repairs a BAM lifted with ``ft pg-lift`` (or the same
stages) directly; it needs the source BAM to recover what the lift lost.
``--bed`` lifts a BED12 file (``fiberhmm-extract`` output) block by block
through ``ft pg-lift``'s BED mode and reassembles the rows.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

TOOL = "fiberhmm-lift"


class LiftError(RuntimeError):
    """A lift stage failed, or a tool it needs is missing."""


def find_tool(name: str, explicit: str | None, flag: str) -> str:
    """Path of ``vg``/``ft``: ``explicit`` (a binary or wrapper script) or PATH."""
    if explicit:
        path = explicit if os.path.sep in explicit else shutil.which(explicit)
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        raise LiftError(f"{flag} {explicit}: not an executable file")
    path = shutil.which(name)
    if path:
        return path
    hints = {
        "vg": "install vg (https://github.com/vgteam/vg/releases; Linux binaries, or a "
              "wrapper script that runs it in a container) and put it on PATH or pass --vg",
        "ft": "install fibertools-rs >= 0.13 (`ft`, with the pg-pansn subcommand) and put it "
              "on PATH or pass --ft",
    }
    raise LiftError(f"`{name}` not found: {hints[name]}")


def _sample_of(prefix: str | None, delimiter: str) -> str | None:
    if not prefix:
        return None
    return prefix.split(delimiter)[0] or None


def _tail(path: str, lines: int = 8) -> str:
    try:
        with open(path, errors="replace") as handle:
            return "".join(handle.readlines()[-lines:]).rstrip()
    except OSError:
        return ""


def run_lift_stages(stashed: str, lifted: str, *, graph: str, prefix: str, target: str,
                    delimiter: str, vg: str, ft: str, threads: int, work_dir: str) -> None:
    """``ft pg-pansn | vg inject | vg surject | ft pg-pansn --strip`` into ``lifted``.

    Every stage's exit status is checked: a failure mid-pipe would otherwise
    leave a short, valid-looking BAM."""
    header_out = os.path.join(work_dir, "source.header.sam")
    vg_threads = str(max(1, threads))
    stages = [
        ("ft pg-pansn", [ft, "pg-pansn", stashed, "--prefix", prefix, "--uncompressed",
                         "--header-out", header_out]),
        ("vg inject", [vg, "inject", "-t", vg_threads, "-x", graph, "-"]),
        ("vg surject", [vg, "surject", "-C", "0", "-t", vg_threads, "-x", graph, "-r", "-b",
                        "-n", target, "-"]),
        ("ft pg-pansn --strip", [ft, "pg-pansn", "-", lifted, "--strip", "--delimiter",
                                 delimiter, "--copy-header", header_out, "--uncompressed"]),
    ]
    procs, logs = [], []
    upstream = None
    try:
        for i, (label, cmd) in enumerate(stages):
            log_path = os.path.join(work_dir, f"stage{i + 1}.log")
            log = open(log_path, "wb")
            logs.append((label, log_path, log))
            last = i == len(stages) - 1
            proc = subprocess.Popen(cmd, stdin=upstream,
                                    stdout=subprocess.DEVNULL if last else subprocess.PIPE,
                                    stderr=log)
            if upstream is not None:
                upstream.close()   # the next stage owns it now
            upstream = proc.stdout
            procs.append(proc)
        codes = [proc.wait() for proc in procs]
    finally:
        for _label, _path, log in logs:
            log.close()
    for (label, log_path, _log), (_l, cmd), code in zip(logs, stages, codes):
        if code != 0:
            raise LiftError(f"{label} failed (exit {code}): {' '.join(cmd)}\n{_tail(log_path)}")


def _count_records(path: str, threads: int) -> int:
    import pysam

    with pysam.AlignmentFile(path, "rb", check_sq=False, threads=threads) as bam:
        return sum(1 for _ in bam.fetch(until_eof=True))


def _graph_lift(args, work_dir: str) -> int:
    from fiberhmm.io import graph_lift as gl

    import pysam

    with pysam.AlignmentFile(args.input, "rb", check_sq=False) as bam:
        pending = gl.header_lift_state(bam.header)["lift"]
    if pending:
        raise LiftError(f"{args.input} was lifted (@PG {', '.join(pending)}) without "
                        "fiberhmm-lift's repair; repair it first (--repair-only)")
    vg = find_tool("vg", args.vg, "--vg")
    ft = find_tool("ft", args.ft, "--ft")
    stashed = os.path.join(work_dir, "stashed.bam")
    stash = gl.stash_bam(args.input, stashed, threads=args.threads)
    print(f"{TOOL}: stashed {stash.stashed} records "
          f"({stash.filled_supplementary} hard-clipped supplementaries filled, "
          f"{stash.excluded_supplementary} excluded without their primary; "
          f"{stash.dropped_secondary} secondary and {stash.dropped_unmapped} unmapped "
          f"records not lifted)", file=sys.stderr)
    lifted = os.path.join(work_dir, "lifted.bam")
    run_lift_stages(stashed, lifted, graph=args.graph, prefix=args.prefix, target=args.target,
                    delimiter=args.delimiter, vg=vg, ft=ft, threads=args.threads,
                    work_dir=work_dir)
    stash_info = stash.as_dict()
    stash_info["unlifted"] = max(0, stash.stashed - _count_records(lifted, args.threads))
    stats = gl.repair_bam(
        lifted, args.output, work_dir=work_dir, target_fasta=args.target_fasta,
        threads=args.threads,
        pg_args={"mode": "lift", "source": _sample_of(args.prefix, args.delimiter),
                 "target": args.target, "graph": args.graph, "stash": stash_info})
    _report(stats, stash_info)
    return 0


def _repair_only(args, work_dir: str) -> int:
    import pysam

    from fiberhmm.io import graph_lift as gl

    with pysam.AlignmentFile(args.input, "rb", check_sq=False) as bam:
        state = gl.header_lift_state(bam.header)
        first = next(bam.fetch(until_eof=True), None)
    if state["repaired"]:
        raise LiftError(f"{args.input}: already repaired (its last lift has a fiberhmm-lift @PG)")
    stash_info: dict = {}
    current = args.input
    if first is not None and not first.has_tag(gl.TAG_FLAG):
        if not args.source:
            raise LiftError(
                f"{args.input} was lifted without fiberhmm-lift's stash; give the BAM it "
                "was lifted from with --source")
        current = os.path.join(work_dir, "restashed.bam")
        match = gl.stash_from_source(args.input, args.source, current, threads=args.threads)
        stash_info["unmatched"] = match.unmatched
        print(f"{TOOL}: matched {match.matched} lifted records to the source "
              f"({match.unmatched} unmatched, dropped)", file=sys.stderr)
    stats = gl.repair_bam(
        current, args.output, work_dir=work_dir, target_fasta=args.target_fasta,
        threads=args.threads,
        pg_args={"mode": "repair-only", "source": _sample_of(args.prefix, args.delimiter),
                 "target": args.target, "graph": args.graph, "stash": stash_info})
    _report(stats, stash_info)
    return 0


def _report(stats, stash_info: dict) -> None:
    lost = stash_info.get("unlifted")
    print(f"{TOOL}: wrote {stats.records_out} records"
          + (f" ({lost} did not lift)" if lost else "")
          + f"; restored 0x800 on {stats.supplementary_restored}, dropped "
          f"{stats.orphan_supplementary_dropped} supplementaries whose primary did not lift; "
          f"{stats.strand_flipped} changed strand (st swapped on {stats.st_swapped}); "
          + ("MD/NM recomputed, SA rebuilt" if stats.calmd
             else "MD/NM/SA dropped (pass --target-fasta to recompute them)"),
          file=sys.stderr)


def _bed_lift(args, work_dir: str) -> int:
    from fiberhmm.io import graph_lift as gl

    if not args.source_fasta:
        raise LiftError("--bed needs --source-fasta (the FASTA of the source assembly)")
    vg = find_tool("vg", args.vg, "--vg")
    ft = find_tool("ft", args.ft, "--ft")
    with open(args.input) as handle:
        bed6, rows, header = gl.explode_bed12(handle)
    blocks_in = os.path.join(work_dir, "blocks.bed")
    blocks_out = os.path.join(work_dir, "blocks.lifted.bed")
    with open(blocks_in, "w") as handle:
        handle.writelines(line + "\n" for line in bed6)
    cmd = [ft, "pg-lift", "-i", blocks_in, "-g", args.graph, "--prefix", args.prefix,
           "--target", args.target, "--vg-binary", vg, "--vg-threads", str(max(1, args.threads)),
           "-d", args.delimiter, "-o", blocks_out, args.source_fasta]
    log_path = os.path.join(work_dir, "ft-pg-lift.log")
    with open(log_path, "wb") as log:
        code = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log).returncode
    if code != 0:
        raise LiftError(f"ft pg-lift failed (exit {code}): {' '.join(cmd)}\n{_tail(log_path)}")
    with open(blocks_out) as handle:
        out, stats = gl.reassemble_bed12(handle, rows)
    with open(args.output, "w") as handle:
        handle.writelines(line + "\n" for line in header + out)
    print(f"{TOOL}: {stats.rows_out} of {stats.rows_in} BED12 rows lifted "
          f"({stats.rows_lost_blocks} lost a block, {stats.rows_split} split across contigs, "
          f"{stats.rows_rearranged} rearranged; {stats.rows_flipped} changed strand)",
          file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="Lift a FiberHMM BAM (or BED12) to another assembly through a pangenome "
                    "graph (vg + fibertools-rs pg-pansn), repairing what the lift leaves "
                    "stale: supplementary flags, hard clips, st on strand changes, MD/NM/SA, "
                    "@PG history. Footprints are carried in molecular coordinates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # HG002 maternal haplotype -> GRCh38, MD/NM/SA recomputed against GRCh38
    fiberhmm-lift -i calls.hg002_mat.bam -o calls.grch38.bam -g hprc.gbz \\
        --prefix 'HG002#1#' --target GRCh38 --target-fasta GRCh38.fa -t 16

    # Repair a BAM lifted with `ft pg-lift --bam` directly
    fiberhmm-lift --repair-only -i lifted.bam --source calls.hg002_mat.bam \\
        -o lifted.repaired.bam --target-fasta GRCh38.fa

    # Lift fiberhmm-extract BED12 rows block by block
    fiberhmm-lift --bed -i calls_nucleosome.bed -o calls_nucleosome.grch38.bed \\
        -g hprc.gbz --prefix 'HG002#1#' --target GRCh38 --source-fasta hg002.fa
""",
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Source BAM (BED12 with --bed; the lifted BAM with --repair-only)")
    parser.add_argument("-o", "--output", required=True, help="Output BAM (sorted, indexed) or BED")
    parser.add_argument("-g", "--graph", help="Pangenome graph (GBZ) with source and target paths")
    parser.add_argument("-p", "--prefix",
                        help="panSN prefix of the source assembly's paths in the graph, "
                             "e.g. 'HG002#1#' (as ft pg-lift --prefix)")
    parser.add_argument("--target",
                        help="Target sample to surject onto, e.g. GRCh38 (as ft pg-lift --target)")
    parser.add_argument("--target-fasta",
                        help="FASTA of the target assembly (contig names as in the lifted "
                             "@SQ). Recomputes MD/NM and rebuilds SA; without it they are "
                             "dropped. Recommended.")
    parser.add_argument("-d", "--delimiter", default="#", help="panSN delimiter (default: #)")
    parser.add_argument("--vg", help="vg binary or wrapper script (default: vg on PATH)")
    parser.add_argument("--ft", help="fibertools-rs ft binary (default: ft on PATH)")
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="Threads for vg, sorting and BAM I/O (default: 4)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--repair-only", action="store_true",
                      help="Only repair -i, a BAM already lifted (ft pg-lift, or the same "
                           "stages); needs --source unless fiberhmm-lift stashed it")
    mode.add_argument("--bed", action="store_true",
                      help="-i is BED12(+N) (fiberhmm-extract): lift each block with ft "
                           "pg-lift's BED mode and reassemble the rows")
    parser.add_argument("--source", help="--repair-only: the BAM that was lifted")
    parser.add_argument("--source-fasta", help="--bed: FASTA of the source assembly")
    parser.add_argument("--tmp-dir", help="Directory for intermediate files "
                                          "(default: next to the output)")
    parser.add_argument("--keep-intermediate", action="store_true",
                        help="Keep the intermediate files")
    from fiberhmm.cli.common import add_version_args
    add_version_args(parser)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    if not args.repair_only:
        missing = [flag for flag, value in (("--graph", args.graph), ("--prefix", args.prefix),
                                            ("--target", args.target)) if not value]
        if missing:
            parser.error(f"lifting needs {', '.join(missing)}")
    if args.source and not args.repair_only:
        parser.error("--source is for --repair-only")
    for flag, path in (("-i", args.input), ("--graph", args.graph), ("--source", args.source),
                       ("--target-fasta", args.target_fasta),
                       ("--source-fasta", args.source_fasta)):
        if path and not os.path.isfile(path):
            parser.error(f"{flag} {path}: no such file")
    out_dir = os.path.dirname(os.path.abspath(args.output))
    work_dir = tempfile.mkdtemp(prefix=".fiberhmm-lift-", dir=args.tmp_dir or out_dir)
    try:
        if args.bed:
            return _bed_lift(args, work_dir)
        if args.repair_only:
            return _repair_only(args, work_dir)
        return _graph_lift(args, work_dir)
    except LiftError as error:
        print(f"error: {TOOL}: {error}", file=sys.stderr)
        return 2
    except ValueError as error:
        from fiberhmm.io.graph_lift import FrameError, RepairError

        if isinstance(error, (FrameError, RepairError)):
            print(f"error: {TOOL}: {error}", file=sys.stderr)
            return 2
        raise
    finally:
        if args.keep_intermediate:
            print(f"{TOOL}: intermediate files kept in {work_dir}", file=sys.stderr)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
