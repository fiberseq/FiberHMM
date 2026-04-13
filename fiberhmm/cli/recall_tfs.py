#!/usr/bin/env python3
"""fiberhmm-recall-tfs CLI  --  [BETA]  LLR-based TF footprint recaller.

*** BETA FEATURE ***
This tool ships as beta: the algorithm, tag schema, and per-enzyme
defaults are stable enough to use, but downstream integrations (fibertools,
FiberBrowser) may still be catching up and calibration outside the
validated enzymes (Hia5 PacBio, DddB DAF, DddA amplicons) has not been
exhaustively tested. Please report surprises on the FiberHMM issue tracker.

Runs as a 2nd pass on a BAM already tagged by ``fiberhmm-apply``.
Writes spec-compliant ``MA``/``AQ`` Molecular-annotation tags
(https://github.com/fiberseq/Molecular-annotation-spec) plus refreshed
legacy ``ns``/``nl``/``as``/``al`` tags reflecting the unified call set.

Supports stdin/stdout piping (``-i -`` / ``-o -``) for composition with
``fiberhmm-apply`` and downstream fibertools stages:

    fiberhmm-apply -o - ... | fiberhmm-recall-tfs -i - -o - --enzyme hia5 | ft fire

By default, v2 short nuc calls (``nl < --unify-threshold``) that overlap
a recaller TF call are demoted out of the ``nuc+`` annotation -- the
recaller version (with proper LLR + edge-ambiguity scoring) replaces
them in the ``tf+`` annotation.

Examples:
  # DddA amplicon BAM (two-pass workflow) -- bundled models, no -m needed
  fiberhmm-apply -i input.bam --enzyme ddda -o tmp/
  fiberhmm-recall-tfs -i tmp/input_footprints.bam -o recalled.bam \\
                       --enzyme ddda -c 8

  # Hia5 streaming composition
  fiberhmm-apply -i input.bam --enzyme hia5 -o - | \\
      fiberhmm-recall-tfs -i - -o recalled.bam --enzyme hia5 -c 8

  # Override with a custom model
  fiberhmm-recall-tfs -i input.bam -o recalled.bam \\
                       -m /path/to/custom.json --min-llr 4.0
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pysam

from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.models import SUPPORTED_ENZYMES, get_model_path as _get_bundled_model
from fiberhmm.inference.tf_recaller import (
    ENZYME_PRESETS,
    HAS_NUMBA,
    apply_emission_uplift,
    build_llr_tables,
    recall_read,
    write_ma_tags,
)


# ---------------------------------------------------------------------------
# Per-worker global state (set by the initializer; avoids repickling arrays)
# ---------------------------------------------------------------------------

_WORKER = {}


def _worker_init(llr_hit, llr_miss, mode, k,
                 min_llr, min_opps, unify_threshold,
                 also_write_legacy, downstream_compat, header_text):
    """Set per-process globals once per worker."""
    _WORKER['llr_hit'] = llr_hit
    _WORKER['llr_miss'] = llr_miss
    _WORKER['mode'] = mode
    _WORKER['k'] = k
    _WORKER['min_llr'] = min_llr
    _WORKER['min_opps'] = min_opps
    _WORKER['unify_threshold'] = unify_threshold
    _WORKER['also_write_legacy'] = also_write_legacy
    _WORKER['downstream_compat'] = downstream_compat
    _WORKER['header'] = pysam.AlignmentHeader.from_text(header_text)


def _process_record(record_str):
    """Process one BAM record (as SAM text) and return (new_sam_line, stats)."""
    read = pysam.AlignedSegment.fromstring(record_str, _WORKER['header'])
    stats = {'v2': 0, 'tf': 0, 'demoted': 0}

    if read.has_tag('ns') and read.has_tag('nl'):
        stats['v2'] = 1
        nl_list = list(read.get_tag('nl'))
        v2_short_count = sum(1 for l in nl_list
                              if 0 < int(l) < _WORKER['unify_threshold'])
        v2_nq = list(read.get_tag('nq')) if read.has_tag('nq') else None
    else:
        v2_short_count = 0
        v2_nq = None

    tf_calls, kept_nucs, msps = recall_read(
        read,
        _WORKER['llr_hit'], _WORKER['llr_miss'],
        _WORKER['mode'], _WORKER['k'],
        min_llr=_WORKER['min_llr'],
        min_opps=_WORKER['min_opps'],
        unify_threshold=_WORKER['unify_threshold'],
    )
    stats['tf'] = len(tf_calls)
    survived_short = sum(1 for _, l in kept_nucs
                          if l < _WORKER['unify_threshold'])
    stats['demoted'] = max(0, v2_short_count - survived_short)

    nq_for_kept = None
    if v2_nq is not None and read.has_tag('ns'):
        try:
            ns_old = list(read.get_tag('ns'))
            nl_old = list(read.get_tag('nl'))
            old_to_nq = {(int(s), int(l)): int(v2_nq[i])
                         for i, (s, l) in enumerate(zip(ns_old, nl_old))
                         if i < len(v2_nq)}
            nq_for_kept = [old_to_nq.get((s, l), 0) for s, l in kept_nucs]
        except Exception:
            nq_for_kept = None

    write_ma_tags(
        read,
        read_length=len(read.query_sequence) if read.query_sequence else 0,
        tf_calls=tf_calls,
        kept_nucs=kept_nucs,
        msps=msps,
        nq_for_kept_nucs=nq_for_kept,
        also_write_legacy=_WORKER['also_write_legacy'],
        downstream_compat=_WORKER['downstream_compat'],
    )
    return read.to_string(), stats


def _process_chunk(records):
    """Worker: process a list of SAM text records."""
    out = []
    total = {'v2': 0, 'tf': 0, 'demoted': 0}
    for rec in records:
        line, stats = _process_record(rec)
        out.append(line)
        for k in total:
            total[k] += stats[k]
    return out, total


def _single_thread_loop(bam_in, bam_out, header_text,
                       llr_hit, llr_miss, mode, k,
                       min_llr, min_opps, unify_threshold,
                       also_write_legacy, downstream_compat, max_reads):
    """Single-threaded fallback path (also used when cores=1)."""
    _worker_init(llr_hit, llr_miss, mode, k, min_llr, min_opps,
                 unify_threshold, also_write_legacy, downstream_compat,
                 header_text)
    n_reads = n_v2 = n_tf = n_demoted = 0
    for read in bam_in:
        if max_reads and n_reads >= max_reads:
            break
        line, stats = _process_record(read.to_string())
        out = pysam.AlignedSegment.fromstring(line, bam_out.header)
        bam_out.write(out)
        n_reads += 1
        n_v2 += stats['v2']; n_tf += stats['tf']; n_demoted += stats['demoted']
    return n_reads, n_v2, n_tf, n_demoted


def _parallel_loop(bam_in, bam_out, header_text,
                   llr_hit, llr_miss, mode, k,
                   min_llr, min_opps, unify_threshold,
                   also_write_legacy, downstream_compat,
                   max_reads, n_cores, chunk_size):
    """Multi-core path using multiprocessing.Pool.imap (preserves order)."""
    def _chunk_iter():
        buf = []
        n = 0
        for read in bam_in:
            if max_reads and n >= max_reads:
                break
            buf.append(read.to_string())
            n += 1
            if len(buf) >= chunk_size:
                yield buf
                buf = []
        if buf:
            yield buf

    n_reads = n_v2 = n_tf = n_demoted = 0
    with mp.Pool(
        processes=n_cores,
        initializer=_worker_init,
        initargs=(llr_hit, llr_miss, mode, k, min_llr, min_opps,
                  unify_threshold, also_write_legacy, downstream_compat,
                  header_text),
    ) as pool:
        for out_lines, stats in pool.imap(_process_chunk, _chunk_iter()):
            for line in out_lines:
                bam_out.write(pysam.AlignedSegment.fromstring(line, bam_out.header))
            n_reads += len(out_lines)
            n_v2 += stats['v2']; n_tf += stats['tf']; n_demoted += stats['demoted']
    return n_reads, n_v2, n_tf, n_demoted


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('-i', '--in-bam', required=True,
                   help='Input BAM tagged by fiberhmm-apply (has ns/nl/as/al). '
                        'Use "-" for stdin.')
    p.add_argument('-o', '--out-bam', required=True,
                   help='Output BAM with MA/AQ + refreshed legacy tags. '
                        'Use "-" for stdout (for piping to ft fire, samtools, etc).')
    p.add_argument('-m', '--model', default=None,
                   help='FiberHMM model JSON. If omitted, the bundled model '
                        'for --enzyme/--seq is used automatically.')
    p.add_argument('--enzyme', choices=sorted(ENZYME_PRESETS.keys()),
                   default=None,
                   help='Enzyme preset: auto-selects the bundled model and '
                        f'min-llr/emission-uplift defaults '
                        f'({", ".join(sorted(ENZYME_PRESETS))}).')
    p.add_argument('--seq', choices=['pacbio', 'nanopore'], default=None,
                   help='Sequencing platform. Required for hia5 '
                        '(pacbio or nanopore); ignored for dddb/ddda.')
    p.add_argument('--min-llr', type=float, default=None,
                   help='Override min LLR (nats). Default: enzyme preset.')
    p.add_argument('--min-opps', type=int, default=3,
                   help='Min informative target positions per call (default 3)')
    p.add_argument('--emission-uplift', type=float, default=None,
                   help='Power transform on emission probabilities. Default 1.0 '
                        '(identity). Use a pre-uplifted model file (e.g. '
                        'ddda_TF.json) for DddA rather than setting this.')
    p.add_argument('--unify-threshold', type=int, default=90,
                   help='v2 nucs with nl < this are scanned + may be demoted '
                        'to tf+ if overlapped by a recaller call (default 90)')
    p.add_argument('--no-legacy-tags', action='store_true',
                   help='Skip refreshed ns/nl/as/al -- emit only MA/AQ.')
    p.add_argument('--downstream-compat', action='store_true',
                   help='Downstream-compatibility mode: skip MA/AQ entirely '
                        'and write TF calls INTO the legacy ns/nl tag '
                        'alongside nucleosomes (sorted by start). Use for '
                        'older tools that do not understand the '
                        'Molecular-annotation spec. Loses per-TF quality '
                        'scoring (tq/el/er) -- positions and lengths only.')
    p.add_argument('-c', '--cores', type=int, default=1,
                   help='Worker processes. 0 = auto-detect (default 1).')
    p.add_argument('--chunk-size', type=int, default=256,
                   help='Reads per worker chunk (default 256).')
    p.add_argument('--io-threads', type=int, default=4,
                   help='htslib BAM compression threads (default 4).')
    p.add_argument('--mode', default=None,
                   help='Override observation mode. Default: read from model.')
    p.add_argument('--context-size', type=int, default=None,
                   help='Override context size. Default: read from model.')
    p.add_argument('--max-reads', type=int, default=0,
                   help='0 = no limit (default)')
    return p.parse_args()


def _resolve_model_metadata(model_path):
    mode = 'pacbio-fiber'
    k = 3
    if model_path.endswith('.json'):
        try:
            with open(model_path) as f:
                d = json.load(f)
            mode = d.get('mode', mode)
            k = int(d.get('context_size', k))
        except (OSError, ValueError):
            pass
    return mode, k


def main():
    args = parse_args()

    stdout_mode = (args.out_bam == '-')
    if stdout_mode:
        # Redirect informational prints to stderr so BAM stream on stdout stays clean
        sys.stdout = sys.stderr

    # Resolve model path: explicit -m wins; else use bundled model for --enzyme
    model_path = args.model
    if model_path is None:
        if args.enzyme is None:
            print(
                "error: one of --model or --enzyme must be provided.\n"
                "  Use --enzyme hia5/dddb/ddda to pick a bundled model, or\n"
                "  use --model /path/to/model.json for a custom model.",
                file=sys.stderr,
            )
            sys.exit(1)
        from fiberhmm.models import get_model_path as _get_bundled
        try:
            model_path = _get_bundled(args.enzyme, tool='recall', seq=args.seq)
        except (KeyError, FileNotFoundError) as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"[recall_tfs] using bundled model: {model_path}", file=sys.stderr)

    # Prominent beta banner: this feature is new, expect rough edges.
    # Mode line varies based on --downstream-compat.
    if args.downstream_compat:
        mode_banner = (
            "  MODE: DOWNSTREAM-COMPAT -- TF calls written into legacy ns/nl.\n"
            "  MA/AQ spec tags are NOT emitted. Per-TF quality scoring is lost.\n"
            "  Use this only for older tools that cannot read the MA/AQ spec.\n"
        )
    else:
        mode_banner = (
            "  MODE: SPEC -- MA/AQ tags emitted per the fiberseq Molecular-\n"
            "  annotation spec (tf+QQQ carries LLR + edge-ambiguity scores).\n"
            "  -> Update FiberBrowser to the MA/AQ-aware release to visualize.\n"
            "  -> Read the spec: https://github.com/fiberseq/Molecular-annotation-spec\n"
            "  -> For tools that do not yet speak MA/AQ, re-run with\n"
            "     --downstream-compat to put TF calls into legacy ns/nl instead.\n"
        )
    print(
        "\n"
        "========================================================================\n"
        "  fiberhmm-recall-tfs  [BETA]\n"
        "  LLR TF footprint recaller -- beta feature shipped in fiberhmm 2.6.0.\n"
        "  Defaults validated on Hia5 PacBio, DddB DAF, and DddA amplicons.\n"
        "\n"
        + mode_banner +
        "\n"
        "  File issues at https://github.com/fiberseq/FiberHMM/issues\n"
        "========================================================================",
        file=sys.stderr,
    )

    # Resolve cores
    if args.cores == 0:
        n_cores = mp.cpu_count()
    else:
        n_cores = max(1, args.cores)

    # Presets + overrides
    preset = ENZYME_PRESETS.get(args.enzyme, {}) if args.enzyme else {}
    min_llr = args.min_llr if args.min_llr is not None else preset.get('min_llr', 5.0)
    uplift = args.emission_uplift if args.emission_uplift is not None \
        else preset.get('emission_uplift', 1.0)

    model, model_k, model_mode = load_model_with_metadata(model_path)
    if not model_mode or not model_k:
        fb_mode, fb_k = _resolve_model_metadata(model_path)
        model_mode = model_mode or fb_mode
        model_k = model_k or fb_k
    mode = args.mode or model_mode
    k = args.context_size or int(model_k)

    llr_hit, llr_miss = build_llr_tables(model)
    if abs(uplift - 1.0) > 1e-9:
        llr_hit, llr_miss = apply_emission_uplift(llr_hit, llr_miss, model, uplift)

    print(
        f"[recall_tfs] enzyme={args.enzyme or 'custom'} mode={mode} k={k} "
        f"min_llr={min_llr:.2f} uplift={uplift:.2f} "
        f"unify_threshold={args.unify_threshold} cores={n_cores} "
        f"numba={'on' if HAS_NUMBA else 'off'}",
        file=sys.stderr,
    )

    # Open BAMs with io-threads. pysam accepts "-" as stdin/stdout natively.
    bam_in = pysam.AlignmentFile(args.in_bam, 'rb',
                                  check_sq=False,
                                  threads=args.io_threads)
    bam_out = pysam.AlignmentFile(args.out_bam, 'wb',
                                   template=bam_in,
                                   threads=args.io_threads)
    header_text = str(bam_in.header)

    # Compat mode always writes legacy tags (TFs live in ns/nl there).
    also_write_legacy = True if args.downstream_compat else (not args.no_legacy_tags)

    if n_cores == 1:
        n_reads, n_v2, n_tf, n_demoted = _single_thread_loop(
            bam_in, bam_out, header_text,
            llr_hit, llr_miss, mode, k,
            min_llr, args.min_opps, args.unify_threshold,
            also_write_legacy, args.downstream_compat, args.max_reads,
        )
    else:
        n_reads, n_v2, n_tf, n_demoted = _parallel_loop(
            bam_in, bam_out, header_text,
            llr_hit, llr_miss, mode, k,
            min_llr, args.min_opps, args.unify_threshold,
            also_write_legacy, args.downstream_compat, args.max_reads,
            n_cores, args.chunk_size,
        )

    bam_in.close()
    bam_out.close()

    print(
        f"[recall_tfs] processed {n_reads} reads; {n_v2} carried v2 tags; "
        f"{n_tf} TF calls emitted; {n_demoted} v2 short nucs demoted to tf+",
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
