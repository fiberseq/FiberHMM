#!/usr/bin/env python3
"""fiberhmm-lift -- move FiberHMM BAMs to another assembly through a pangenome graph.

Two engines run between a pre-lift stash and a post-lift repair
(:mod:`fiberhmm.io.graph_lift`), then sort and index:

``native`` (default with ``--map``)
    graphlift's ``liftbam`` lifts the records through a graphlift chain map
    (``gbz2chain`` output, ``X.chain.graphlift.json``) derived from the
    graph. No vg, no container.
``vg`` (default with ``--graph``)
    the four stages of ``ft pg-lift --bam`` (``ft pg-pansn`` adding the
    source panSN prefix, ``vg inject``, ``vg surject`` onto the target
    sample, ``ft pg-pansn --strip``). The stages are run here rather than
    through ``ft pg-lift --bam``, which fails on fibertools-rs <= 0.13 (it
    passes ``--out`` to ``ft pg-pansn``, which takes the output path
    positionally).

``--repair-only`` repairs a BAM lifted with ``ft pg-lift`` (or the same
stages) directly; it needs the source BAM to recover what the lift lost.
``--bed`` lifts BED intervals: graphlift's ``liftbed`` through ``--map``
(native, the default with ``--map``) or ``ft pg-lift``'s BED mode through
``--graph`` (vg). BED12 rows (``fiberhmm-extract`` output) are lifted block
by block and reassembled; other BED files row by row.
"""
from __future__ import annotations

import argparse
import json
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
        "liftbam": "build graphlift with its bam feature (`cargo build --release --features "
                   "bam`; the binary is target/release/liftbam) and put it on PATH or pass "
                   "--liftbam",
        "liftbed": "build graphlift (`cargo build --release`; the binary is "
                   "target/release/liftbed) and put it on PATH or pass --liftbed",
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


ENGINES = ("native", "vg")


def read_lift_map(path: str) -> dict:
    """The graphlift map manifest's samples: ``{'source': 'HG002', 'target': 'GRCh38'}``
    (sample names of the source and target paths, comma-joined when several)."""
    try:
        with open(path) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as error:
        raise LiftError(f"--map {path}: not a readable graphlift map ({error})") from None
    if not isinstance(manifest, dict) or manifest.get("format") != "graphlift.sidecar":
        raise LiftError(f"--map {path}: not a graphlift map manifest (X.chain.graphlift.json)")

    def samples(key):
        names = sorted({str(c.get("path", "")).split("#")[0] for c in manifest.get(key, [])} - {""})
        return ",".join(names) or None

    return {"source": samples("sources"), "target": samples("targets")}


def run_liftbam(stashed: str, lifted: str, *, liftbam: str, lift_map: str, threads: int,
                work_dir: str, split_gap: int, alt: str, unlifted: str | None,
                target_fasta: str | None = None) -> dict:
    """``liftbam`` from ``stashed`` into ``lifted``; returns its report.

    The stash has put every footprint tag in molecular frame, hence
    ``--footprint-frame molecular``. With ``target_fasta`` its contig lengths
    go to ``--target-sizes`` (the map may only have inferred them; the repair
    checks the lifted @SQ against the FASTA). MD/NM/SA are left to the repair."""
    report = os.path.join(work_dir, "liftbam.json")
    cmd = [liftbam, "-i", stashed, "-m", lift_map, "-o", lifted, "--footprint-frame", "molecular",
           "--report", report, "-t", str(max(1, threads)), "--split-gap", str(split_gap),
           "--alt", alt]
    if unlifted:
        cmd += ["--unlifted", unlifted]
    if target_fasta:
        import pysam

        sizes = os.path.join(work_dir, "target.sizes")
        with pysam.FastaFile(target_fasta) as fasta, open(sizes, "w") as handle:
            for name, length in zip(fasta.references, fasta.lengths):
                handle.write(f"{name}\t{length}\n")
        cmd += ["--target-sizes", sizes]
    log_path = os.path.join(work_dir, "liftbam.log")
    with open(log_path, "wb") as log:
        code = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log).returncode
    if code != 0:
        raise LiftError(f"liftbam failed (exit {code}): {' '.join(cmd)}\n{_tail(log_path)}")
    with open(report) as handle:
        return json.load(handle)


def _count_records(path: str, threads: int) -> int:
    import pysam

    with pysam.AlignmentFile(path, "rb", check_sq=False, threads=threads) as bam:
        return sum(1 for _ in bam.fetch(until_eof=True))


def _graph_lift(args, work_dir: str) -> int:
    from fiberhmm.io import graph_lift as gl

    import pysam

    with pysam.AlignmentFile(args.input, "rb", check_sq=False) as bam:
        pending = gl.header_lift_state(bam.header)["lift"]
        source_pgs = bam.header.to_dict().get("PG", [])
    if pending:
        raise LiftError(f"{args.input} was lifted (@PG {', '.join(pending)}) without "
                        "fiberhmm-lift's repair; repair it first (--repair-only)")
    if args.engine == "native":
        liftbam = find_tool("liftbam", args.liftbam, "--liftbam")
        samples = read_lift_map(args.map)
    else:
        vg = find_tool("vg", args.vg, "--vg")
        ft = find_tool("ft", args.ft, "--ft")
    stashed = os.path.join(work_dir, "stashed.bam")
    stash = gl.stash_bam(args.input, stashed, threads=args.threads)
    print(f"{TOOL}: stashed {stash.stashed} records "
          f"({stash.filled_supplementary} hard-clipped supplementaries filled, "
          f"{stash.excluded_supplementary} excluded, "
          f"{stash.orphan_supplementary_no_primary_in_input} supplementaries whose primary "
          f"is not in the input left out; "
          f"{stash.dropped_secondary} secondary and {stash.dropped_unmapped} unmapped "
          f"records not lifted)", file=sys.stderr)
    lifted = os.path.join(work_dir, "lifted.bam")
    stash_info = stash.as_dict()
    if args.engine == "native":
        report = run_liftbam(stashed, lifted, liftbam=liftbam, lift_map=args.map,
                             threads=args.threads, work_dir=work_dir, split_gap=args.split_gap,
                             alt=args.alt, unlifted=args.unlifted,
                             target_fasta=args.target_fasta)
        stash_info["unlifted"] = int(report.get("unlifted", 0))
        stash_info["split_records"] = int(report.get("split_pieces", 0))
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(report.get("unlifted_by_reason", {}).items()))
        print(f"{TOOL}: liftbam lifted {report.get('lifted', 0)} records "
              f"({report.get('split', 0)} split into {report.get('split_pieces', 0)} extra "
              f"records); {stash_info['unlifted']} could not be lifted"
              + (f" ({reasons})" if reasons else "")
              + (f", written to {args.unlifted}" if args.unlifted and stash_info["unlifted"] else "")
              + f"; {report.get('projected_bases', 0)} of {report.get('aligned_bases', 0)} "
              "aligned bases projected", file=sys.stderr)
        pg_args = {"mode": "lift", "engine": "native",
                   "source": _sample_of(args.prefix, args.delimiter) or samples["source"],
                   "target": args.target or samples["target"], "graph": args.map,
                   "stash": stash_info}
    else:
        run_lift_stages(stashed, lifted, graph=args.graph, prefix=args.prefix, target=args.target,
                        delimiter=args.delimiter, vg=vg, ft=ft, threads=args.threads,
                        work_dir=work_dir)
        stash_info["unlifted"] = max(0, stash.stashed - _count_records(lifted, args.threads))
        pg_args = {"mode": "lift", "engine": "vg",
                   "source": _sample_of(args.prefix, args.delimiter),
                   "target": args.target, "graph": args.graph, "stash": stash_info}
    stats = gl.repair_bam(
        lifted, args.output, work_dir=work_dir, target_fasta=args.target_fasta,
        threads=args.threads, source_pgs=source_pgs, mapq=args.mapq, pg_args=pg_args)
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
    with pysam.AlignmentFile(args.input, "rb", check_sq=False) as bam:
        pgs = bam.header.to_dict().get("PG", [])
    by_liftbam = any(gl._is_liftbam(pg) for pg in pgs if str(pg.get("ID")) in state["lift"])
    if by_liftbam and first is not None and not first.has_tag(gl.TAG_FLAG):
        raise LiftError(
            f"{args.input} was lifted by graphlift's liftbam directly: its flags, st, hard clips "
            "and MD/NM/SA are already consistent, and --repair-only cannot rebuild its stash "
            "(liftbam may split a record into several). Lift the source BAM with "
            "fiberhmm-lift --map instead to get fiberhmm-lift's MAPQ policy and provenance.")
    stash_info: dict = {}
    current = args.input
    source_pgs = None
    if args.source:
        with pysam.AlignmentFile(args.source, "rb", check_sq=False) as bam:
            source_pgs = bam.header.to_dict().get("PG", [])
    if first is not None and not first.has_tag(gl.TAG_FLAG):
        if not args.source:
            raise LiftError(
                f"{args.input} was lifted without fiberhmm-lift's stash; give the BAM it "
                "was lifted from with --source")
        current = os.path.join(work_dir, "restashed.bam")
        match = gl.stash_from_source(args.input, args.source, current, threads=args.threads)
        stash_info["unmatched"] = match.unmatched
        stash_info["ambiguous"] = match.ambiguous
        print(f"{TOOL}: matched {match.matched} lifted records to the source "
              f"({match.unmatched} unmatched and {match.ambiguous} ambiguous, dropped)",
              file=sys.stderr)
    stats = gl.repair_bam(
        current, args.output, work_dir=work_dir, target_fasta=args.target_fasta,
        threads=args.threads, source_pgs=source_pgs, mapq=args.mapq,
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


def _bed_is_bed12(path: str) -> bool:
    """Whether the data rows of a BED file are BED12 (blocks); all rows must agree."""
    from fiberhmm.io.graph_lift import _BED_HEADER_PREFIXES

    kinds = set()
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith(_BED_HEADER_PREFIXES):
                continue
            kinds.add(len(line.split("\t")) >= 12)
            if len(kinds) > 1:
                raise LiftError(f"{path}: some rows are BED12 and some are not")
    return kinds == {True}


def run_liftbed(bed_in: str, bed_out: str, *, liftbed: str, lift_map: str, work_dir: str,
                split_gap: int, alt: str, min_match: float, unlifted: str | None) -> dict:
    """``liftbed`` from ``bed_in`` into ``bed_out`` (best placement of split rows); returns
    its report."""
    report = os.path.join(work_dir, "liftbed.json")
    cmd = [liftbed, "-i", bed_in, "-m", lift_map, "-o", bed_out, "--report", report,
           "--split", "best", "--split-gap", str(split_gap), "--alt", alt,
           "--min-match", str(min_match)]
    if unlifted:
        cmd += ["--unlifted", unlifted]
    log_path = os.path.join(work_dir, "liftbed.log")
    with open(log_path, "wb") as log:
        code = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=log).returncode
    if code != 0:
        raise LiftError(f"liftbed failed (exit {code}): {' '.join(cmd)}\n{_tail(log_path)}")
    with open(report) as handle:
        return json.load(handle)


def _read_unlifted_bed(path: str) -> list:
    """``[(reason, row), ...]`` of a liftOver-style unlifted file (``#reason`` then the row)."""
    out, reason = [], None
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("#") and reason is None:
                reason = line[1:]
            elif reason is not None:
                out.append((reason, line))
                reason = None
    return out


def _lift_bed_file(args, work_dir: str, bed_in: str, bed_out: str, unlifted: str | None) -> dict:
    """Lifts a BED6+ file with the chosen engine; returns the native report (empty for vg)."""
    if args.engine == "native":
        liftbed = find_tool("liftbed", args.liftbed or _sibling(args.liftbam, "liftbed"),
                            "--liftbed")
        return run_liftbed(bed_in, bed_out, liftbed=liftbed, lift_map=args.map,
                           work_dir=work_dir, split_gap=args.split_gap, alt=args.alt,
                           min_match=args.min_match, unlifted=unlifted)
    if not args.source_fasta:
        raise LiftError("--bed with --engine vg needs --source-fasta (the FASTA of the source "
                        "assembly)")
    vg = find_tool("vg", args.vg, "--vg")
    ft = find_tool("ft", args.ft, "--ft")
    cmd = [ft, "pg-lift", "-i", bed_in, "-g", args.graph, "--prefix", args.prefix,
           "--target", args.target, "--vg-binary", vg, "--vg-threads", str(max(1, args.threads)),
           "-d", args.delimiter, "-o", bed_out, args.source_fasta]
    log_path = os.path.join(work_dir, "ft-pg-lift.log")
    with open(log_path, "wb") as log:
        code = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=log).returncode
    if code != 0:
        raise LiftError(f"ft pg-lift failed (exit {code}): {' '.join(cmd)}\n{_tail(log_path)}")
    return {}


def _sibling(path: str | None, name: str) -> str | None:
    """``name`` next to an explicit ``path`` (a graphlift target/release dir), if it is there."""
    if not path or os.path.sep not in path:
        return None
    candidate = os.path.join(os.path.dirname(path), name)
    return candidate if os.path.isfile(candidate) else None


def _bed_lift(args, work_dir: str) -> int:
    from fiberhmm.io import graph_lift as gl

    native = args.engine == "native"
    if not _bed_is_bed12(args.input):
        unlifted = args.unlifted
        report = _lift_bed_file(args, work_dir, args.input, args.output, unlifted)
        if native:
            reasons = ", ".join(f"{k}={v}" for k, v in sorted(report.get("unlifted_by_reason", {}).items()))
            print(f"{TOOL}: liftbed lifted {report.get('lifted', 0)} of {report.get('rows_in', 0)} "
                  f"BED rows ({report.get('split', 0)} split, best placement kept; "
                  f"{report.get('partial', 0)} partly unprojected; "
                  f"{report.get('strand_flipped', 0)} changed strand); "
                  f"{report.get('unlifted', 0)} not lifted" + (f" ({reasons})" if reasons else ""),
                  file=sys.stderr)
        else:
            print(f"{TOOL}: ft pg-lift lifted the BED rows (the vg engine does not count "
                  "unlifted rows)", file=sys.stderr)
        return 0
    with open(args.input) as handle:
        bed6, rows, header = gl.explode_bed12(handle, unstranded_as="+" if native else None)
    blocks_in = os.path.join(work_dir, "blocks.bed")
    blocks_out = os.path.join(work_dir, "blocks.lifted.bed")
    blocks_unlifted = os.path.join(work_dir, "blocks.unlifted.bed") if native else None
    with open(blocks_in, "w") as handle:
        handle.writelines(line + "\n" for line in bed6)
    report = _lift_bed_file(args, work_dir, blocks_in, blocks_out, blocks_unlifted)
    dropped: list = []
    with open(blocks_out) as handle:
        out, stats = gl.reassemble_bed12(handle, rows, dropped=dropped, explicit_strand=native)
    with open(args.output, "w") as handle:
        handle.writelines(line + "\n" for line in header + out)
    if native:
        block_reason = {}
        for reason, row in _read_unlifted_bed(blocks_unlifted):
            fields = row.split("\t")
            if len(fields) > 3 and "|" in fields[3]:
                block_reason.setdefault(int(fields[3].rpartition("|")[0]), reason)
        if args.unlifted:
            with open(args.unlifted, "w") as handle:
                for index, reason in dropped:
                    if reason == "lost_block" and index in block_reason:
                        reason = f"block_{block_reason[index]}"
                    handle.write(f"#{reason}\n" + "\t".join(rows[index]) + "\n")
    print(f"{TOOL}: {stats.rows_out} of {stats.rows_in} BED12 rows lifted "
          f"({stats.rows_lost_blocks} lost a block, {stats.rows_split} split across contigs, "
          f"{stats.rows_rearranged} rearranged; {stats.rows_flipped} changed strand)"
          + (f"; liftbed: {report.get('lifted', 0)} of {report.get('rows_in', 0)} blocks lifted, "
             f"{report.get('split', 0)} split (best placement kept), "
             f"{report.get('partial', 0)} partly unprojected" if native else "")
          + (f"; dropped rows written to {args.unlifted}" if native and args.unlifted and dropped else ""),
          file=sys.stderr)
    return 0


def _mapq_mode(value: str) -> str:
    from fiberhmm.io.graph_lift import parse_mapq_mode

    try:
        return parse_mapq_mode(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _has_files(path: str) -> bool:
    try:
        return any(name.endswith((".bam", ".bed")) for name in os.listdir(path))
    except OSError:
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL,
        description="Lift a FiberHMM BAM (or BED12) to another assembly through a pangenome "
                    "graph, repairing what the lift leaves stale: supplementary flags, hard "
                    "clips, st on strand changes, MD/NM/SA, @PG history. Footprints are "
                    "carried in molecular coordinates. The native engine lifts through a "
                    "graphlift chain map (--map) with graphlift's liftbam; the vg engine "
                    "through the GBZ (--graph) with vg + fibertools-rs pg-pansn.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # HG002 (both haplotypes) -> GRCh38 natively, through a graphlift map
    # (gbz2chain -g hprc.gbz -t 'GRCh38#0' -s HG002 --hap-suffix 1=_PATERNAL,2=_MATERNAL ...)
    fiberhmm-lift -i calls.hg002.bam -o calls.grch38.bam \\
        --map hprc.GRCh38.HG002.chain.graphlift.json --target-fasta GRCh38.fa -t 16

    # The same through vg: HG002 maternal haplotype -> GRCh38
    fiberhmm-lift -i calls.hg002_mat.bam -o calls.grch38.bam -g hprc.gbz \\
        --prefix 'HG002#1#' --target GRCh38 --target-fasta GRCh38.fa -t 16

    # Repair a BAM lifted with `ft pg-lift --bam` directly
    fiberhmm-lift --repair-only -i lifted.bam --source calls.hg002_mat.bam \\
        -o lifted.repaired.bam --target-fasta GRCh38.fa

    # Lift fiberhmm-extract BED12 rows block by block, natively through the map
    fiberhmm-lift --bed -i calls_nucleosome.bed -o calls_nucleosome.grch38.bed \\
        --map hprc.GRCh38.HG002.chain.graphlift.json --unlifted calls_nucleosome.unlifted.bed

    # The same through vg (ft pg-lift's BED mode)
    fiberhmm-lift --bed -i calls_nucleosome.bed -o calls_nucleosome.grch38.bed \\
        -g hprc.gbz --prefix 'HG002#1#' --target GRCh38 --source-fasta hg002.fa
""",
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Source BAM (BED12 with --bed; the lifted BAM with --repair-only)")
    parser.add_argument("-o", "--output", required=True, help="Output BAM (sorted, indexed) or BED")
    parser.add_argument("--engine", choices=ENGINES,
                        help="native: graphlift liftbam (liftbed with --bed) through --map (the "
                             "default when --map is given); vg: vg inject/surject (ft pg-lift's "
                             "BED mode with --bed) through --graph (the default otherwise)")
    parser.add_argument("-m", "--map",
                        help="graphlift chain map manifest (X.chain.graphlift.json, from "
                             "graphlift gbz2chain) whose sources are the input's contigs")
    parser.add_argument("--liftbam", help="graphlift liftbam binary (default: liftbam on PATH)")
    parser.add_argument("--liftbed", help="--bed: graphlift liftbed binary (default: next to "
                                          "--liftbam, else liftbed on PATH)")
    parser.add_argument("--split-gap", type=int, default=10_000, metavar="BP",
                        help="native: split a record into supplementaries at a read or target "
                             "gap of at least BP (default: 10000)")
    parser.add_argument("--alt", choices=("never", "fallback"), default="never",
                        help="native: 'fallback' lifts a record through an alternative chain "
                             "when no primary chain of the map projects it (default: never)")
    parser.add_argument("--unlifted",
                        help="native: write the records that could not be lifted here (source "
                             "coordinates, Yu:Z:<reason>; with --bed the rows, after a "
                             "#reason line); they are always counted")
    parser.add_argument("--min-match", type=float, default=0.0, metavar="F",
                        help="native --bed: lift a BED interval (or BED12 block) only if a "
                             "fraction F of its bases project (liftOver's -minMatch; default 0: "
                             "any projected base, the interval shrinks to them)")
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
    parser.add_argument("--mapq", default="source", type=_mapq_mode, metavar="{source,fixed:N}",
                        help="MAPQ of lifted records: 'source' (default) keeps the MAPQ of the "
                             "source alignment, as CrossMap and ft pg-lift do; 'fixed:N' sets "
                             "every lifted record to N. Against a diploid source assembly, "
                             "MAPQ is low wherever the haplotypes are identical. Yq keeps the "
                             "source value either way.")
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
                      help="-i is a BED file: BED12(+N) rows (fiberhmm-extract) are lifted "
                           "block by block and reassembled, other BED files row by row")
    parser.add_argument("--source", help="--repair-only: the BAM that was lifted")
    parser.add_argument("--source-fasta", help="--bed --engine vg: FASTA of the source assembly")
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
    if args.engine is None:
        args.engine = "native" if args.map else "vg"
    if not 0.0 <= args.min_match <= 1.0:
        parser.error("--min-match must be between 0 and 1")
    if args.min_match and not (args.bed and args.engine == "native"):
        parser.error("--min-match is for --bed with the native engine")
    if args.engine == "native" and not args.repair_only:
        if not args.map:
            parser.error("--engine native needs --map (a graphlift chain map)")
        if args.split_gap < 1:
            parser.error("--split-gap must be >= 1")
    elif not args.repair_only:
        missing = [flag for flag, value in (("--graph", args.graph), ("--prefix", args.prefix),
                                            ("--target", args.target)) if not value]
        if missing:
            parser.error(f"lifting needs {', '.join(missing)}"
                         + (" (or --map for the native engine)" if not args.map else ""))
    if args.unlifted and (args.engine != "native" or args.repair_only):
        parser.error("--unlifted is for the native engine")
    if args.source and not args.repair_only:
        parser.error("--source is for --repair-only")
    for flag, path in (("-i", args.input), ("--graph", args.graph), ("--map", args.map),
                       ("--source", args.source),
                       ("--target-fasta", args.target_fasta),
                       ("--source-fasta", args.source_fasta)):
        if path and not os.path.isfile(path):
            parser.error(f"{flag} {path}: no such file")
    out_dir = os.path.dirname(os.path.abspath(args.output))
    work_dir = tempfile.mkdtemp(prefix=".fiberhmm-lift-", dir=args.tmp_dir or out_dir)
    succeeded = False
    try:
        if args.bed:
            code = _bed_lift(args, work_dir)
        elif args.repair_only:
            code = _repair_only(args, work_dir)
        else:
            code = _graph_lift(args, work_dir)
        succeeded = code == 0
        return code
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
        # A failed run keeps whatever it produced (a finished BAM that could
        # not be moved into place, say); only empty work dirs go.
        kept = args.keep_intermediate or (not succeeded and _has_files(work_dir))
        if kept:
            print(f"{TOOL}: intermediate files kept in {work_dir}", file=sys.stderr)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
