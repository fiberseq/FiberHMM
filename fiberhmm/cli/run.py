#!/usr/bin/env python3
"""fiberhmm-run — full single-molecule footprinting pipeline in one command.

Chains the individual FiberHMM tools into a single streaming pipeline with
no intermediate files.  For DAF-seq data:

    fiberhmm-daf-encode | fiberhmm-apply | fiberhmm-recall-tfs [| ft fire]

For Hia5 (m6A fiber-seq, no encode step):

    fiberhmm-apply | fiberhmm-recall-tfs [| ft fire]

This is a convenience wrapper around the composable tools.  For custom recall
parameters, posteriors export, or non-standard workflows, run the individual
commands directly — fiberhmm-run does nothing they cannot do.

Examples:
  # DddB DAF-seq, full pipeline with FIRE scoring
  fiberhmm-run -i aligned.bam -o recalled.bam --enzyme dddb --fire -c 8

  # DddA amplicons (recall is required and always included)
  fiberhmm-run -i aligned.bam -o recalled.bam --enzyme ddda --fire -c 8

  # Hia5 PacBio, skip recall
  fiberhmm-run -i input.bam -o output.bam --enzyme hia5 --seq pacbio --no-recall

  # Stream to stdout for custom downstream (e.g. samtools view)
  fiberhmm-run -i input.bam -o - --enzyme dddb | samtools sort -o sorted.bam

  # Override recall threshold
  fiberhmm-run -i input.bam -o out.bam --enzyme dddb --min-llr 6.0
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from fiberhmm.models import SUPPORTED_ENZYMES
from fiberhmm.inference.tf_recaller import ENZYME_PRESETS

_DAF_ENZYMES = {'dddb', 'ddda'}
_ONT_ENZYMES  = {'dddb', 'ddda'}   # nanopore-based; hia5-nanopore handled via --seq


def _fiberhmm_tool(name: str) -> str:
    """Locate a fiberhmm CLI tool: same bin-dir as current interpreter first,
    then PATH."""
    candidate = os.path.join(os.path.dirname(sys.executable), name)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which(name)
    if found:
        return found
    print(f"error: {name!r} not found. Is fiberhmm installed?", file=sys.stderr)
    sys.exit(1)


def _find_ft(explicit: str | None) -> str | None:
    """Locate the ft (fibertools-rs) binary."""
    if explicit:
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit
        print(f"error: ft binary not found at {explicit!r}", file=sys.stderr)
        sys.exit(1)
    for candidate in [
        shutil.which('ft'),
        os.path.expanduser('~/.cargo/bin/ft'),
    ]:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _run_pipeline(stages: list[list[str]]) -> None:
    """Launch *stages* as a chain of subprocesses connected by pipes.

    Each stage's stdout is piped to the next stage's stdin.  The last stage
    inherits stdout from the parent process (so it reaches the file/terminal
    the caller set up).  All stages share the parent's stderr so their banners
    and progress lines are visible.

    Raises SystemExit(1) if any stage exits non-zero.
    """
    procs: list[subprocess.Popen] = []
    prev_stdout = None

    for i, cmd in enumerate(stages):
        is_last = (i == len(stages) - 1)
        p = subprocess.Popen(
            cmd,
            stdin=prev_stdout,
            stdout=None if is_last else subprocess.PIPE,
            stderr=sys.stderr,
        )
        if prev_stdout is not None:
            prev_stdout.close()   # hand ownership to the child
        prev_stdout = p.stdout
        procs.append(p)

    for p in procs:
        p.wait()

    failed = [i for i, p in enumerate(procs) if p.returncode != 0]
    if failed:
        print(
            f"error: pipeline stage(s) {failed} exited with non-zero status.",
            file=sys.stderr,
        )
        sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('-i', '--in-bam', required=True,
                   help='Input BAM (aligned). Use "-" for stdin.')
    p.add_argument('-o', '--out-bam', required=True,
                   help='Output BAM. Use "-" for stdout (unsorted; pipe to '
                        'samtools sort for a sorted file).')
    p.add_argument('--enzyme', choices=SUPPORTED_ENZYMES, required=True,
                   help='Enzyme: hia5, dddb, or ddda.')
    p.add_argument('--seq', choices=['pacbio', 'nanopore'], default=None,
                   help='Sequencing platform. Required for hia5 '
                        '(pacbio or nanopore); ignored for dddb/ddda.')
    p.add_argument('-c', '--cores', type=int, default=4,
                   help='CPU cores per pipeline stage (default: 4).')
    p.add_argument('--no-recall', action='store_true',
                   help='Skip fiberhmm-recall-tfs. Use when you only need '
                        'nucleosome / MSP calls without TF footprints.')
    p.add_argument('--fire', action='store_true',
                   help='Add ft fire scoring step (requires fibertools-rs '
                        '"ft" binary in PATH or --ft-path).')
    p.add_argument('--ft-path', default=None, metavar='PATH',
                   help='Explicit path to the ft binary (default: search PATH '
                        'and ~/.cargo/bin/ft).')
    p.add_argument('--min-llr', type=float, default=None,
                   help='Override recall-tfs min LLR threshold '
                        '(default: enzyme preset).')
    p.add_argument('--downstream-compat', action='store_true',
                   help='Pass --downstream-compat to recall-tfs: write TF '
                        'calls into legacy ns/nl instead of MA/AQ spec tags.')
    p.add_argument('--io-threads', type=int, default=4,
                   help='htslib I/O compression threads per stage (default: 4).')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stdout_mode = (args.out_bam == '-')

    # ── Validate / locate tools ───────────────────────────────────────────────
    ft_path: str | None = None
    if args.fire:
        ft_path = _find_ft(args.ft_path)
        if ft_path is None:
            print(
                "error: --fire requires the fibertools-rs 'ft' binary.\n"
                "  Install: cargo install ft  (or conda install -c bioconda fibertools-rs)\n"
                "  Or pass --ft-path /path/to/ft",
                file=sys.stderr,
            )
            sys.exit(1)

    samtools = shutil.which('samtools') or 'samtools'

    # ── Enzyme flags shared by apply + recall ─────────────────────────────────
    enzyme_flags = ['--enzyme', args.enzyme]
    use_seq = args.seq
    if args.enzyme == 'hia5':
        if use_seq is None:
            print(
                "warning: --seq not specified for hia5; defaulting to nanopore. "
                "Use --seq pacbio for PacBio data.",
                file=sys.stderr,
            )
            use_seq = 'nanopore'
        enzyme_flags += ['--seq', use_seq]

    # ft fire: use --ont for nanopore-based data
    use_ont = args.enzyme in _ONT_ENZYMES or use_seq == 'nanopore'

    # ── Print banner ──────────────────────────────────────────────────────────
    steps = ['encode' if args.enzyme in _DAF_ENZYMES else None,
             'apply', None if args.no_recall else 'recall-tfs',
             'ft-fire' if args.fire else None]
    steps = [s for s in steps if s]
    print(
        f"[fiberhmm-run] enzyme={args.enzyme} "
        f"seq={use_seq or 'n/a'} cores={args.cores}\n"
        f"[fiberhmm-run] pipeline: {' | '.join(steps)} | sort+index",
        file=sys.stderr,
    )

    # ── Build pipeline stages ─────────────────────────────────────────────────
    stages: list[list[str]] = []

    # Stage 1: DAF encode (DAF enzymes only)
    if args.enzyme in _DAF_ENZYMES:
        stages.append([
            _fiberhmm_tool('fiberhmm-daf-encode'),
            '-i', args.in_bam, '-o', '-',
            '--io-threads', str(args.io_threads),
        ])
        apply_input = '-'
    else:
        apply_input = args.in_bam

    # Stage 2: apply
    apply_mode = ['--mode', 'daf'] if args.enzyme in _DAF_ENZYMES else []
    stages.append([
        _fiberhmm_tool('fiberhmm-apply'),
        '-i', apply_input, '-o', '-',
        '-c', str(args.cores),
        '--io-threads', str(args.io_threads),
    ] + enzyme_flags + apply_mode)

    # Stage 3: recall-tfs (optional)
    if not args.no_recall:
        recall_cmd = [
            _fiberhmm_tool('fiberhmm-recall-tfs'),
            '-i', '-', '-o', '-',
            '-c', str(args.cores),
            '--io-threads', str(args.io_threads),
        ] + enzyme_flags
        if args.min_llr is not None:
            recall_cmd += ['--min-llr', str(args.min_llr)]
        if args.downstream_compat:
            recall_cmd.append('--downstream-compat')
        stages.append(recall_cmd)

    # Stage 4: ft fire (optional)
    if args.fire:
        fire_cmd = [ft_path, 'fire']
        if use_ont:
            fire_cmd.append('--ont')
        fire_cmd += ['-', '-']
        stages.append(fire_cmd)

    # Stage 5: sort (when writing to a file; stdout mode skips sort)
    if not stdout_mode:
        stages.append([samtools, 'sort', '-@', '4', '-o', args.out_bam, '-'])

    # ── Run ───────────────────────────────────────────────────────────────────
    if stdout_mode:
        _run_pipeline(stages)
    else:
        _run_pipeline(stages)
        subprocess.run([samtools, 'index', args.out_bam], check=True)
        print(f"[fiberhmm-run] output: {args.out_bam}", file=sys.stderr)
        print(f"[fiberhmm-run] index:  {args.out_bam}.bai", file=sys.stderr)


if __name__ == '__main__':
    main()
