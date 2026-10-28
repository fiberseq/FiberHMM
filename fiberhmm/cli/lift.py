#!/usr/bin/env python3
"""fiberhmm-lift -- move FiberHMM BAMs to another assembly through a pangenome graph.

Two engines run between a pre-lift stash and a post-lift repair
(:mod:`fiberhmm.io.graph_lift`), then sort and index:

``native`` (the primary route; default with ``--map``)
    graphlift's ``liftbam`` lifts the records through a graphlift chain map
    (``gbz2chain`` output, ``X.chain.graphlift.json``) derived from the
    graph. No vg, no container.
``vg`` (optional fallback; default with ``--graph`` alone)
    ``vg inject`` and ``vg surject`` onto the target sample, between
    fibertools-rs ``ft pg-pansn`` stages that add and strip the source panSN
    prefix.

``--repair-only`` repairs a BAM lifted by vg-based tools directly; it needs
the source BAM to recover what the lift lost. ``--bed`` lifts BED intervals:
graphlift's ``liftbed`` through ``--map`` (native), or, as a fallback,
``vg inject``/``vg surject`` of reads cut from the source FASTA through
``--graph`` (vg; no fibertools-rs). BED12 rows (``fiberhmm-extract`` output) are lifted
block by block and reassembled; other BED files row by row.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import shutil
import subprocess
import sys
import tempfile

TOOL = "fiberhmm-lift"

# How to build graphlift's binaries (the same commands as graphlift's README).
BUILD_HINTS = {
    "liftbam": "liftbam is built only with graphlift's bam feature: in a graphlift checkout "
               "run `cargo build --release --locked --features bam`",
    "liftbed": "build graphlift: in a graphlift checkout run `cargo build --release --locked`",
}


class LiftError(RuntimeError):
    """A lift stage failed, or a tool it needs is missing."""


def find_tool(name: str, explicit: str | None, flag: str) -> str:
    """Path of ``vg``/``ft``: ``explicit`` (a binary or wrapper script) or PATH."""
    if explicit:
        path = explicit if os.path.sep in explicit else shutil.which(explicit)
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        raise LiftError(f"{flag} {explicit}: not an executable file"
                        + (f" ({BUILD_HINTS[name]})" if name in BUILD_HINTS else ""))
    path = shutil.which(name)
    if path:
        return path
    hints = {
        "vg": "install vg (https://github.com/vgteam/vg/releases; Linux binaries, or a "
              "wrapper script that runs it in a container) and put it on PATH or pass --vg",
        "ft": "install fibertools-rs >= 0.13 (`ft`, with the pg-pansn subcommand) and put it "
              "on PATH or pass --ft",
        "liftbam": BUILD_HINTS["liftbam"] + ", then put target/release on PATH or pass "
                   "--liftbam target/release/liftbam",
        "liftbed": BUILD_HINTS["liftbed"] + ", then put target/release on PATH or pass "
                   "--liftbed target/release/liftbed",
    }
    message = f"`{name}` not found: {hints[name]}"
    if name == "liftbam":
        liftbed = shutil.which("liftbed")
        if liftbed:
            message += (f". ({liftbed} is on PATH: that graphlift build lacks the bam feature; "
                        "rebuild it with --features bam)")
    raise LiftError(message)


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
          f"{stash.orphan_supplementary_no_primary_in_input} hard-clipped supplementaries whose "
          f"primary is not in the input left out, {stash.orphan_supplementary_kept} soft-clipped "
          f"ones lifted; "
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
          + f"; restored 0x800 on {stats.supplementary_restored}, kept "
          f"{stats.orphan_supplementary_kept} orphan supplementaries (Yo:i:1); "
          f"{stats.strand_flipped} changed strand (st swapped on {stats.st_swapped}); "
          + ("MD/NM recomputed, SA rebuilt" if stats.calmd
             else "MD/NM/SA dropped (pass --target-fasta to recompute them)"),
          file=sys.stderr)


BED_FORMATS = ("auto", "bed", "bed12", "narrowPeak")


def _is_gzip(path: str) -> bool:
    with open(path, "rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def materialize_bed(path: str, work_dir: str) -> str:
    """A plain-text copy of BED input ``path`` in ``work_dir`` when it is ``-`` (stdin) or
    gzip/bgzip; ``path`` itself otherwise."""
    import gzip

    if path == "-":
        raw = os.path.join(work_dir, "stdin.bed")
        with open(raw, "wb") as handle:
            shutil.copyfileobj(sys.stdin.buffer, handle)
        path = raw
    if not _is_gzip(path):
        return path
    plain = os.path.join(work_dir, "input.bed")
    try:
        with gzip.open(path, "rb") as src, open(plain, "wb") as dst:
            shutil.copyfileobj(src, dst)
    except (OSError, EOFError) as error:
        raise LiftError(f"-i: cannot decompress the gzip input ({error})") from None
    return plain


def finalize_output(tmp: str, final: str) -> None:
    """Moves ``tmp`` to ``final``, BGZF-compressed (gzip-readable, tabix-indexable) when
    ``final`` ends in ``.gz``; through ``final.partial`` so a failure leaves no
    truncated ``final``."""
    partial = final + ".partial"
    try:
        if final.endswith(".gz"):
            import pysam

            pysam.tabix_compress(tmp, partial, force=True)
            os.remove(tmp)
        else:
            shutil.move(tmp, partial)
        os.replace(partial, final)
    except OSError as error:
        raise LiftError(f"cannot write {final}: {error}") from None


def detect_bed_format(path: str, declared: str, name: str) -> tuple:
    """``(format, lift_thick)`` of a plain-text BED file.

    ``auto``: BED12 when every data row has 12+ columns (some but not all is an error);
    narrowPeak when the input name ends in ``.narrowPeak[.gz]`` or a ``track`` line says
    ``type=narrowPeak``; BED otherwise. Ten columns alone do not make a narrowPeak. A
    declared format is checked against the rows. For BED, thickStart/thickEnd are lifted
    when every row has integer columns 7-8 within the row."""
    from fiberhmm.io.graph_lift import _BED_HEADER_PREFIXES

    wide = narrow = 0
    rows = 0
    thick = True
    summit = True
    track_np = False
    with open(path) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith(_BED_HEADER_PREFIXES):
                if line.startswith("track") and "type=narrowPeak" in line:
                    track_np = True
                continue
            fields = line.split("\t")
            rows += 1
            if len(fields) >= 12:
                wide += 1
            else:
                narrow += 1
            try:
                start, end = int(fields[1]), int(fields[2])
                ts, te = int(fields[6]), int(fields[7])
                thick = thick and start <= ts <= te <= end
            except (IndexError, ValueError):
                thick = False
            try:
                int(fields[9])
            except (IndexError, ValueError):
                summit = False
    if declared == "auto":
        if wide and narrow:
            raise LiftError(f"{name}: some rows are BED12 and some are not")
        if rows and wide == rows:
            fmt = "bed12"
        elif name.endswith((".narrowPeak", ".narrowPeak.gz")) or track_np:
            fmt = "narrowPeak"
        else:
            fmt = "bed"
    else:
        fmt = declared
        if fmt == "bed12" and narrow:
            raise LiftError(f"{name}: --bed-format bed12, but {narrow} rows have fewer than "
                            "12 columns")
    if fmt == "narrowPeak" and rows and not summit:
        raise LiftError(f"{name}: narrowPeak rows need an integer summit offset in column 10")
    return fmt, fmt == "bed" and rows > 0 and thick


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
    """Lifts a BED6+ file with the chosen engine (vg: named rows only, see ``run_vg_bed``);
    returns the engine's report."""
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
    return run_vg_bed(bed_in, bed_out, graph=args.graph, prefix=args.prefix, target=args.target,
                      delimiter=args.delimiter, vg=vg, threads=args.threads,
                      source_fasta=args.source_fasta, work_dir=work_dir, unlifted=unlifted,
                      chunk=VG_BED_CHUNK, flank=VG_BED_FLANK)


def strip_pansn(name: str, delimiter: str) -> str:
    """``GRCh38#0#chr1`` -> ``chr1``: everything after the second delimiter (as
    ``ft pg-pansn --strip``); a name with fewer delimiters is kept."""
    parts = name.split(delimiter, 2)
    return parts[2] if len(parts) == 3 else name


VG_BED_CHUNK = 50_000
VG_BED_FLANK = 2000


def run_vg_bed(bed_in: str, bed_out: str, *, graph: str, prefix: str, target: str,
               delimiter: str, vg: str, threads: int, source_fasta: str, work_dir: str,
               unlifted: str | None, chunk: int = VG_BED_CHUNK, flank: int = VG_BED_FLANK) -> dict:
    """Lifts named BED6 rows (``<row>|<unit>``, from ``explode_bed12``/``explode_bed_rows``)
    through the graph with vg alone. The source contigs are cut into reads of ``chunk``
    bp overlapping by half (only where rows lie), each aligned end to end on ``prefix +
    contig``; a row is carried by the read in which it lies most centrally (a row longer
    than half a chunk gets a read of its own with ``flank`` bp on either side). ``vg
    inject`` puts the reads on the graph and ``vg surject`` onto the ``target`` sample's
    paths. A row's lifted interval runs from the first to the last target base aligned
    to its own bases (bases with no target shrink it), in the surjected record that
    aligns most of them; a reverse-strand record swaps a ``+``/``-`` strand column. Rows
    with no aligned base are unlifted (``no_projection``; rows outside the source FASTA
    ``contig_not_in_fasta``, empty rows ``empty``). Returns a report shaped like
    liftbed's."""
    import numpy as np
    import pysam

    step = max(1, chunk // 2)
    rows, header_lines, reasons = {}, [], {}
    carried = defaultdict(list)          # read name -> [(row name, start, end) in read offsets]
    reads = {}                           # read name -> (contig, start, end)
    with pysam.FastaFile(source_fasta) as fasta, open(bed_in) as handle:
        lengths = dict(zip(fasta.references, fasta.lengths))
        contigs = list(fasta.references)
        tid = {c: i for i, c in enumerate(contigs)}
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(("#", "track", "browser")):
                header_lines.append(line)
                continue
            fields = line.split("\t")
            name = fields[3]
            rows[name] = fields
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
            if chrom not in tid or end > lengths[chrom]:
                reasons[name] = "contig_not_in_fasta"
                continue
            if end <= start:
                reasons[name] = "empty"
                continue
            if end - start > step:
                lo, hi = max(0, start - flank), min(lengths[chrom], end + flank)
                read = f"r{len(reads)}"
            else:
                k = max(0, round((start + end) / 2 / step) - 1)
                lo, hi = k * step, min(lengths[chrom], k * step + chunk)
                read = f"c{tid[chrom]}_{k}"
            reads.setdefault(read, (chrom, lo, hi))
            carried[read].append((name, start - lo, end - lo))
        bam_header = {"HD": {"VN": "1.6", "SO": "unsorted"},
                      "SQ": [{"SN": prefix + c, "LN": lengths[c]} for c in contigs]}
        reads_path = os.path.join(work_dir, "vg_bed.reads.bam")
        with pysam.AlignmentFile(reads_path, "wb", header=bam_header) as out:
            for read_name, (chrom, lo, hi) in reads.items():
                read = pysam.AlignedSegment(out.header)
                read.query_name = read_name
                read.flag = 0
                read.reference_id = tid[chrom]
                read.reference_start = lo
                read.mapping_quality = 60
                read.cigarstring = f"{hi - lo}M"
                read.query_sequence = fasta.fetch(chrom, lo, hi).upper()
                out.write(read)
    surjected = os.path.join(work_dir, "vg_bed.surjected.bam")
    vg_threads = str(max(1, threads))
    inject = [vg, "inject", "-t", vg_threads, "-x", graph, "-"]
    surject = [vg, "surject", "-C", "0", "-t", vg_threads, "-x", graph, "-r", "-b", "-n", target,
               "-"]
    logs = [os.path.join(work_dir, "vg_bed.inject.log"), os.path.join(work_dir, "vg_bed.surject.log")]
    with open(reads_path, "rb") as src, open(surjected, "wb") as dst, \
            open(logs[0], "wb") as log1, open(logs[1], "wb") as log2:
        p1 = subprocess.Popen(inject, stdin=src, stdout=subprocess.PIPE, stderr=log1)
        p2 = subprocess.Popen(surject, stdin=p1.stdout, stdout=dst, stderr=log2)
        p1.stdout.close()
        codes = [p1.wait(), p2.wait()]
    for label, cmd, code, log in zip(("vg inject", "vg surject"), (inject, surject), codes, logs):
        if code != 0:
            raise LiftError(f"{label} failed (exit {code}): {' '.join(cmd)}\n{_tail(log)}")
    best = {}                            # row name -> (aligned bases, contig, start, end, reverse)
    with pysam.AlignmentFile(surjected, "rb", check_sq=False) as bam:
        for record in bam.fetch(until_eof=True):
            if record.is_unmapped or record.is_secondary or record.query_name not in carried:
                continue
            chrom, lo, hi = reads[record.query_name]
            length = hi - lo
            q2r = np.full(length, -1, dtype=np.int64)
            pairs = np.array(record.get_aligned_pairs(matches_only=True), dtype=np.int64)
            if not len(pairs):
                continue
            if record.is_reverse:
                pairs[:, 0] = length - 1 - pairs[:, 0]
            q2r[pairs[:, 0]] = pairs[:, 1]
            contig = strip_pansn(record.reference_name, delimiter)
            for name, a, b in carried[record.query_name]:
                refs = q2r[a:b]
                refs = refs[refs >= 0]
                if len(refs) and (name not in best or len(refs) > best[name][0]):
                    best[name] = (len(refs), contig, int(refs.min()), int(refs.max()) + 1,
                                  record.is_reverse)
    flipped = 0
    with open(bed_out, "w") as out:
        out.writelines(line + "\n" for line in header_lines)
        for name, fields in rows.items():
            hit = best.get(name)
            if hit is None:
                reasons.setdefault(name, "no_projection")
                continue
            new = list(fields)
            new[0], new[1], new[2] = hit[1], str(hit[2]), str(hit[3])
            if hit[4]:
                flipped += 1
                if len(new) > 5 and new[5] in ("+", "-"):
                    new[5] = "-" if new[5] == "+" else "+"
            out.write("\t".join(new) + "\n")
    if unlifted:
        with open(unlifted, "w") as out:
            for name, reason in reasons.items():
                out.write(f"#{reason}\n" + "\t".join(rows[name]) + "\n")
    by_reason: dict = {}
    for reason in reasons.values():
        by_reason[reason] = by_reason.get(reason, 0) + 1
    return {"rows_in": len(rows), "lifted": len(rows) - len(reasons), "unlifted": len(reasons),
            "strand_flipped": flipped, "split": 0, "partial": 0, "unlifted_by_reason": by_reason}


def _sibling(path: str | None, name: str) -> str | None:
    """``name`` next to an explicit ``path`` (a graphlift target/release dir), if it is there."""
    if not path or os.path.sep not in path:
        return None
    candidate = os.path.join(os.path.dirname(path), name)
    return candidate if os.path.isfile(candidate) else None


def _block_reasons(path: str) -> dict:
    """``{(row, unit): reason}`` of a liftbed unlifted file of ``<row>|<unit>`` rows."""
    out = {}
    for reason, row in _read_unlifted_bed(path):
        fields = row.split("\t")
        if len(fields) > 3 and "|" in fields[3]:
            index, _, unit = fields[3].rpartition("|")
            if index.isdigit():
                out.setdefault((int(index), unit), reason)
    return out


def _bed_lift(args, work_dir: str) -> int:
    bed_in = materialize_bed(args.input, work_dir)
    fmt, thick = detect_bed_format(bed_in, args.bed_format, args.input)
    if args.bed12_partial_explicit == "keep" and fmt != "bed12":
        raise LiftError(f"--bed12-partial keep is for BED12 input; {args.input} is read as {fmt}")
    out_tmp = os.path.join(work_dir, "output.bed")
    unl_tmp = os.path.join(work_dir, "unlifted.out.bed") if args.unlifted else None
    try:
        if fmt == "bed12":
            code = _bed12_lift(args, work_dir, bed_in, out_tmp, unl_tmp)
        elif fmt == "narrowPeak" or thick or args.engine != "native":
            code = _bed_rows_lift(args, work_dir, bed_in, out_tmp, unl_tmp, fmt, thick)
        else:
            code = _bed_plain_lift(args, work_dir, bed_in, out_tmp, unl_tmp)
    except ValueError as error:
        raise LiftError(f"{args.input}: {error}") from None
    finalize_output(out_tmp, args.output)
    if unl_tmp:
        if not os.path.exists(unl_tmp):
            open(unl_tmp, "w").close()
        finalize_output(unl_tmp, args.unlifted)
    return code


def _bed_plain_lift(args, work_dir, bed_in, out_tmp, unl_tmp) -> int:
    """Native, BED with no coordinate columns of its own: straight through liftbed."""
    report = _lift_bed_file(args, work_dir, bed_in, out_tmp, unl_tmp)
    reasons = ", ".join(f"{k}={v}" for k, v in sorted(report.get("unlifted_by_reason", {}).items()))
    print(f"{TOOL}: liftbed lifted {report.get('lifted', 0)} of {report.get('rows_in', 0)} "
          f"BED rows ({report.get('split', 0)} split, best placement kept; "
          f"{report.get('partial', 0)} partly unprojected; "
          f"{report.get('strand_flipped', 0)} changed strand); "
          f"{report.get('unlifted', 0)} not lifted" + (f" ({reasons})" if reasons else ""),
          file=sys.stderr)
    return 0


def _lift_units(args, work_dir, bed6) -> tuple:
    """Lifts exploded ``<row>|<unit>`` BED6 rows; ``(lifted lines, report,
    {(row, unit): reason} of the unlifted ones)``."""
    units_in = os.path.join(work_dir, "units.bed")
    units_out = os.path.join(work_dir, "units.lifted.bed")
    units_unlifted = os.path.join(work_dir, "units.unlifted.bed")
    with open(units_in, "w") as handle:
        handle.writelines(line + "\n" for line in bed6)
    report = _lift_bed_file(args, work_dir, units_in, units_out, units_unlifted)
    with open(units_out) as handle:
        lifted = handle.readlines()
    return lifted, report, _block_reasons(units_unlifted)


def _write_header_and_rows(path, header, rows) -> None:
    with open(path, "w") as handle:
        handle.writelines(line + "\n" for line in header + rows)


def _bed12_lift(args, work_dir, bed_in, out_tmp, unl_tmp) -> int:
    from fiberhmm.io import graph_lift as gl

    native = args.engine == "native"
    with open(bed_in) as handle:
        bed6, rows, header = gl.explode_bed12(handle, unstranded_as="+")
    lifted, report, reasons = _lift_units(args, work_dir, bed6)
    dropped: list = []
    dropped_blocks: list = []
    out, stats = gl.reassemble_bed12(lifted, rows, dropped=dropped, explicit_strand=True,
                                     partial=args.bed12_partial, dropped_blocks=dropped_blocks)
    _write_header_and_rows(out_tmp, header, out)
    if unl_tmp:
        row_reason = {}
        for (index, unit), reason in reasons.items():
            if unit.isdigit():
                row_reason.setdefault(index, reason)
        with open(unl_tmp, "w") as handle:
            for index, reason in dropped:
                if reason == "lost_block" and index in row_reason:
                    reason = f"block_{row_reason[index]}"
                handle.write(f"#{reason}\n" + "\t".join(rows[index]) + "\n")
            for index, block in dropped_blocks:
                fields = rows[index]
                start = int(fields[1])
                size = int(fields[10].rstrip(",").split(",")[block])
                offset = int(fields[11].rstrip(",").split(",")[block])
                reason = reasons.get((index, str(block)), "lost")
                handle.write(f"#partial_{reason}\n" + "\t".join(
                    [fields[0], str(start + offset), str(start + offset + size),
                     f"{fields[3]}|{block}", fields[4], fields[5]]) + "\n")
    print(f"{TOOL}: {stats.rows_out} of {stats.rows_in} BED12 rows lifted "
          f"({stats.rows_lost_blocks} lost a block, {stats.rows_split} split across contigs, "
          f"{stats.rows_rearranged} rearranged; {stats.rows_flipped} changed strand)"
          + (f"; {stats.rows_partial} rows kept without {stats.blocks_dropped} unliftable "
             "blocks (--bed12-partial keep)" if args.bed12_partial == "keep" else "")
          + (f"; thickStart/thickEnd lifted on {stats.thick_lifted} rows, emptied on "
             f"{stats.thick_lost}" if stats.thick_lifted or stats.thick_lost else "")
          + (f"; liftbed: {report.get('lifted', 0)} of {report.get('rows_in', 0)} "
             + ("blocks" if report.get('rows_in', 0) == stats.blocks_in else "intervals")
             + f" lifted, {report.get('split', 0)} split (best placement kept), "
             f"{report.get('partial', 0)} partly unprojected" if native
             else f"; vg: {report.get('lifted', 0)} of {report.get('rows_in', 0)} intervals "
                  "lifted")
          + (f"; dropped rows and blocks written to {args.unlifted}"
             if unl_tmp and (dropped or dropped_blocks) else ""),
          file=sys.stderr)
    return 0


def _bed_rows_lift(args, work_dir, bed_in, out_tmp, unl_tmp, fmt, thick) -> int:
    """BED with thickStart/thickEnd, or narrowPeak: each row with its coordinate columns."""
    from fiberhmm.io import graph_lift as gl

    native = args.engine == "native"
    summit = fmt == "narrowPeak"
    with open(bed_in) as handle:
        bed6, rows, header = gl.explode_bed_rows(handle, thick=thick, summit=summit,
                                                 unstranded_as="+")
    lifted, report, reasons = _lift_units(args, work_dir, bed6)
    dropped: list = []
    out, stats = gl.reassemble_bed_rows(lifted, rows, thick=thick, summit=summit,
                                        dropped=dropped)
    _write_header_and_rows(out_tmp, header, out)
    if unl_tmp:
        with open(unl_tmp, "w") as handle:
            for index, _reason in dropped:
                reason = reasons.get((index, "m"), "unlifted")
                handle.write(f"#{reason}\n" + "\t".join(rows[index]) + "\n")
    by_reason: dict = {}
    for index, _reason in dropped:
        reason = reasons.get((index, "m"), "unlifted")
        by_reason[reason] = by_reason.get(reason, 0) + 1
    detail = ", ".join(f"{k}={v}" for k, v in sorted(by_reason.items()))
    print(f"{TOOL}: {stats.rows_out} of {stats.rows_in} {fmt} rows lifted "
          f"({stats.rows_flipped} changed strand); {stats.rows_unlifted} not lifted"
          + (f" ({detail})" if detail else "")
          + (f"; thickStart/thickEnd lifted on {stats.thick_lifted} rows inside the row, "
             f"emptied on {stats.thick_lost}" if thick else "")
          + (f"; summits lifted on {stats.summits_lifted} rows, set to -1 on "
             f"{stats.summits_lost}" if summit else "")
          + (f"; unlifted rows written to {args.unlifted}" if unl_tmp and dropped else ""),
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
                    "carried in molecular coordinates. Supplementaries without a lifted primary "
                    "are kept with 0x800 and Yo:i:1. The native engine lifts through a "
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

    # Repair a BAM lifted by a vg-based tool directly
    fiberhmm-lift --repair-only -i lifted.bam --source calls.hg002_mat.bam \\
        -o lifted.repaired.bam --target-fasta GRCh38.fa

    # Lift fiberhmm-extract BED12 rows block by block, natively through the map
    fiberhmm-lift --bed -i calls_nucleosome.bed -o calls_nucleosome.grch38.bed \\
        --map hprc.GRCh38.HG002.chain.graphlift.json --unlifted calls_nucleosome.unlifted.bed

    # The same through vg (fallback; needs the source FASTA)
    fiberhmm-lift --bed -i calls_nucleosome.bed -o calls_nucleosome.grch38.bed \\
        -g hprc.gbz --prefix 'HG002#1#' --target GRCh38 --source-fasta hg002.fa
""",
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Source BAM (a BED file with --bed: plain, gzip or bgzip, - for "
                             "stdin; the lifted BAM with --repair-only)")
    parser.add_argument("-o", "--output", required=True,
                        help="Output BAM (sorted, indexed) or BED (bgzip-compressed when the "
                             "name ends in .gz)")
    parser.add_argument("--engine", choices=ENGINES,
                        help="native: graphlift liftbam (liftbed with --bed) through --map (the "
                             "default when --map is given); vg (fallback): vg inject/surject "
                             "through --graph (the default otherwise)")
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
                             "#reason line, bgzip-compressed for a .gz name); they are always "
                             "counted")
    parser.add_argument("--min-match", type=float, default=0.0, metavar="F",
                        help="native --bed: lift a BED interval (or BED12 block) only if a "
                             "fraction F of its bases project (liftOver's -minMatch; default 0: "
                             "any projected base, the interval shrinks to them, so a mostly "
                             "unprojected block can lift as a remnant of a few bp, as with vg; "
                             "0.5 drops those)")
    parser.add_argument("--bed-format", choices=BED_FORMATS, default="auto",
                        help="--bed: how to read the input. bed12: blocks lifted one by one; "
                             "narrowPeak: the summit (column 10) lifted with the peak (-1 when "
                             "it does not lift); bed: row by row. thickStart/thickEnd (columns "
                             "7-8) are lifted for bed and bed12 when they hold coordinates. "
                             "auto (default): bed12 when every row has 12+ columns, narrowPeak "
                             "for a .narrowPeak[.gz] input or a 'track type=narrowPeak' line, "
                             "else bed")
    parser.add_argument("--bed12-partial", choices=("drop", "keep"), default=None,
                        help="--bed: a BED12 row some of whose blocks do not lift. keep "
                             "(default): keep the blocks that lift (blockCount, sizes, starts "
                             "and bounds recomputed) when at least one does, and list the "
                             "dropped blocks in --unlifted with their reasons; drop: drop the "
                             "whole row")
    parser.add_argument("-g", "--graph", help="Pangenome graph (GBZ) with source and target paths")
    parser.add_argument("-p", "--prefix",
                        help="panSN prefix of the source assembly's paths in the graph, "
                             "e.g. 'HG002#1#' (vg engine)")
    parser.add_argument("--target",
                        help="vg engine: target sample to surject onto, e.g. GRCh38")
    parser.add_argument("--target-fasta",
                        help="FASTA of the target assembly (contig names as in the lifted "
                             "@SQ). Recomputes MD/NM and rebuilds SA; without it they are "
                             "dropped. Recommended.")
    parser.add_argument("--mapq", default="source", type=_mapq_mode, metavar="{source,fixed:N}",
                        help="MAPQ of lifted records: 'source' (default) keeps the MAPQ of the "
                             "source alignment, as CrossMap does; 'fixed:N' sets "
                             "every lifted record to N. Against a diploid source assembly, "
                             "MAPQ is low wherever the haplotypes are identical. Yq keeps the "
                             "source value either way.")
    parser.add_argument("-d", "--delimiter", default="#", help="panSN delimiter (default: #)")
    parser.add_argument("--vg", help="vg binary or wrapper script (default: vg on PATH)")
    parser.add_argument("--ft", help="vg engine, BAM: fibertools-rs ft binary for its "
                                     "pg-pansn stages (default: ft on PATH)")
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="Threads for vg, sorting and BAM I/O (default: 4)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--repair-only", action="store_true",
                      help="Only repair -i, a BAM already lifted by a vg-based tool (vg "
                           "inject/surject); needs --source unless fiberhmm-lift stashed it. "
                           "Orphan supplementaries are kept with Yo:i:1 and 0x800")
    mode.add_argument("--bed", action="store_true",
                      help="-i is a BED file: BED12(+N) rows (fiberhmm-extract) are lifted "
                           "block by block and reassembled, other BED files row by row")
    parser.add_argument("--source", help="--repair-only: the BAM that was lifted")
    parser.add_argument("--source-fasta", help="--bed --engine vg: FASTA of the source assembly "
                                               "(each interval's sequence is surjected)")
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
    if args.unlifted and (args.repair_only or not (args.bed or args.engine == "native")):
        parser.error("--unlifted is for the native engine, or --bed")
    if args.source and not args.repair_only:
        parser.error("--source is for --repair-only")
    if not args.bed and (args.bed_format != "auto" or args.bed12_partial is not None):
        parser.error("--bed-format and --bed12-partial are for --bed")
    # keep is the default (owner, 27 Oct 2026); only an explicit keep is refused for non-BED12 input
    args.bed12_partial_explicit = args.bed12_partial
    if args.bed12_partial is None:
        args.bed12_partial = "keep"
    if args.input == "-" and not args.bed:
        parser.error("-i - (stdin) is for --bed")
    for flag, path in (("-i", None if args.input == "-" else args.input), ("--graph", args.graph), ("--map", args.map),
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
