#!/usr/bin/env python3
"""fiberhmm-merge -- build both-strand consensus reads from cross-strand pairs.

Consumes a BAM tagged by ``fiberhmm-pair`` (reads carrying ``mt:A:P`` + ``mp:Z``)
and, for each resolved CT/GA pair, emits one both-strand consensus read spanning
the union of the two spans (see :mod:`fiberhmm.crossstrand.consensus`). The
consensus applies both strands' deaminations to a reference-frame sequence and
records the strand-coverage regime in the MA ``deam+``/``deam-`` track, so the
model -- once taught to read it -- knows where to expect both / C-only / G-only.

By default the two source reads of each merged pair are replaced by their
consensus and all other reads pass through unchanged (a complete callset, denser
where both strands were recovered). ``--pairs-only`` emits just the consensus
reads. Output is coordinate-sorted and indexed.

Consensus reads carry:
    MA:Z   ...;deam+:<CT coverage>;deam-:<GA coverage>
    cs:Z   source read names, "<ct_name>;<ga_name>"
    mc/mg/pm/pa/sb/sd/sr/sg  pairing evidence carried from fiberhmm-pair
    dc:i   reference-frame deamination count (C->T and G->A)
    bc:i   canonical source-base conflicts replaced by N
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pysam

from fiberhmm.crossstrand.consensus import build_consensus
from fiberhmm.crossstrand.pairing import FLAVOR_CT, read_flavor
from fiberhmm.crossstrand.recall import RecallContext, recall_consensus_full
from fiberhmm.io.bam_header import append_ma_types

_TAG_SOURCES = 'cs'
_PAIR_TAGS = ('mc', 'mg', 'pm', 'pa', 'sb', 'sd', 'sr', 'sg')


def _make_consensus_segment(cons, header, tid, pair_tags=None):
    seg = pysam.AlignedSegment(header)
    seg.query_name = f"{cons.ct_name}.cs"
    seg.flag = 0
    seg.reference_id = tid
    seg.reference_start = cons.ref_start
    seg.mapping_quality = cons.mapq
    seg.cigartuples = [(0, cons.length)]  # all-M, reference frame
    seg.query_sequence = cons.seq
    # This is a synthetic reference-frame molecule. Do not invent Q40 base
    # qualities or misuse NM as a deamination counter; annotation confidence
    # is carried by AQ, while dc/bc describe consensus construction.
    seg.query_qualities = None
    seg.set_tag('dc', cons.deam_count, value_type='i')
    seg.set_tag('bc', cons.base_conflicts, value_type='i')
    seg.set_tag('MA', cons.ma, value_type='Z')
    seg.set_tag(_TAG_SOURCES, f"{cons.ct_name};{cons.ga_name}", value_type='Z')
    for tag, value, value_type in pair_tags or ():
        seg.set_tag(tag, value, value_type=value_type)
    return seg


def _n50(lengths):
    if not lengths:
        return 0
    s = sorted(lengths, reverse=True)
    half = sum(s) / 2.0
    run = 0
    for x in s:
        run += x
        if run >= half:
            return x
    return 0


def _merged_bp(intervals):
    """Total unique bp covered by a list of (start, end) intervals."""
    if not intervals:
        return 0
    ivs = sorted(intervals)
    total = 0
    cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total


def run_merge(in_bam, out_bam, prob_threshold=0, pairs_only=False, io_threads=4,
              recall=False, enzyme='ddda', phase_nrl=196,
              nuc_recall_policy='conservative',
              derived_tf_max_edge_ambiguity=12):
    t0 = time.time()
    ctx = None
    if recall:
        ctx = RecallContext(enzyme)
        print(f"merge-recall: loaded {enzyme} apply+recall models (k={ctx.k}); "
              f"re-calling footprints (HMM + nuc + TF recallers) on both-strand "
              f"consensus reads", file=sys.stderr)
    bam = pysam.AlignmentFile(in_bam, 'rb')
    header = append_ma_types(bam.header, ['deam'])

    # Pass 1: per chromosome, collect the paired reads and build consensus.
    merged_names = set()          # source read names replaced by a consensus
    consensus_lengths = []
    both_intervals = defaultdict(list)   # chrom -> [(start,end)] both-strand
    n_consensus = n_pairs_seen = n_build_fail = 0
    seen_paired_names = set()

    unsorted = out_bam + '.unsorted.bam'
    out = pysam.AlignmentFile(unsorted, 'wb', header=header, threads=io_threads)

    def flush(tid, paired):
        """Build consensus reads for one chromosome's paired-read dict."""
        nonlocal n_consensus, n_pairs_seen, n_build_fail
        if tid < 0 or not paired:
            return
        chrom = bam.get_reference_name(tid)
        done = set()
        for name, read in paired.items():
            if name in done:
                continue
            mate_name = read.get_tag('mp')
            mate = paired.get(mate_name)
            if mate is None:
                raise ValueError(
                    f"paired read {name!r} names missing partner {mate_name!r}"
                )
            if (not mate.has_tag('mp') or mate.get_tag('mp') != name or
                    not mate.has_tag('mt') or mate.get_tag('mt') != 'P'):
                raise ValueError(
                    f"pair tags are not reciprocal for {name!r}/{mate_name!r}"
                )
            done.add(name)
            done.add(mate_name)
            n_pairs_seen += 1
            fl = read_flavor(read, prob_threshold)
            mate_fl = read_flavor(mate, prob_threshold)
            if fl is None or mate_fl is None or fl == mate_fl:
                raise ValueError(
                    f"pair {name!r}/{mate_name!r} does not contain one CT and one GA read"
                )
            ct_read, ga_read = (read, mate) if fl == FLAVOR_CT else (mate, read)
            cons = build_consensus(ct_read, ga_read)
            if cons is None:
                n_build_fail += 1
                continue
            pair_method = read.get_tag('pm') if read.has_tag('pm') else None
            pair_tags = [
                item for item in read.get_tags(with_value_type=True)
                if item[0] in _PAIR_TAGS
                and not (item[0] == 'mg' and pair_method != 'F')
            ]
            seg = _make_consensus_segment(cons, header, tid, pair_tags)
            if ctx is not None:
                recalled = recall_consensus_full(
                    seg, ctx, phase_nrl=phase_nrl,
                    nuc_recall_policy=nuc_recall_policy,
                    derived_tf_max_edge_ambiguity=(
                        derived_tf_max_edge_ambiguity),
                )
                if not recalled:
                    n_build_fail += 1
                    continue
            out.write(seg)
            n_consensus += 1
            merged_names.add(ct_read.query_name)
            merged_names.add(ga_read.query_name)
            consensus_lengths.append(cons.length)
            if cons.both_end > cons.both_start:
                both_intervals[chrom].append((cons.both_start, cons.both_end))

    # Pass 1: one streaming pass; the input is coordinate-sorted, so all reads
    # of a chromosome are contiguous -- collect paired reads per chromosome and
    # flush at each reference-id transition (no index required).
    cur_tid = None
    paired = {}
    try:
        for read in bam.fetch(until_eof=True):
            if read.is_secondary or read.is_supplementary:
                continue
            if read.reference_id != cur_tid:
                if cur_tid is not None:
                    flush(cur_tid, paired)
                    print(f"  {bam.get_reference_name(cur_tid)}: {n_consensus:,} "
                          f"consensus so far [{time.time()-t0:.0f}s]", file=sys.stderr)
                cur_tid = read.reference_id
                paired = {}
            if read.has_tag('mt') and read.get_tag('mt') == 'P' and read.has_tag('mp'):
                if read.query_name in seen_paired_names:
                    raise ValueError(
                        "paired-duplex merge requires unique primary query names; "
                        f"duplicate paired name {read.query_name!r}"
                    )
                seen_paired_names.add(read.query_name)
                paired[read.query_name] = read.__copy__()
        flush(cur_tid, paired)
    except Exception:
        out.close()
        bam.close()
        try:
            os.remove(unsorted)
        except OSError:
            pass
        raise

    # Pass 2: pass through non-merged reads (unless --pairs-only). Reopen the
    # input -- the pass-1 handle is spent at EOF.
    n_passthrough = 0
    if not pairs_only:
        with pysam.AlignmentFile(in_bam, 'rb') as bam2:
            for read in bam2.fetch(until_eof=True):
                # A merged molecule replaces the complete source-name record
                # set. Retaining secondary/supplementary records under the old
                # name would orphan them from a primary alignment.
                if read.query_name in merged_names:
                    continue
                out.write(read)
                n_passthrough += 1
    out.close()
    genome = sum(l for r, l in zip(bam.references, bam.lengths)
                 if r.startswith('chr') and '_' not in r and r != 'chrM')
    bam.close()

    pysam.sort('-@', str(io_threads), '-o', out_bam, unsorted)
    pysam.index(out_bam)
    try:
        os.remove(unsorted)
    except OSError:
        pass

    # ---- stats ----
    both_bp = sum(_merged_bp(v) for v in both_intervals.values())
    cl = np.array(consensus_lengths) if consensus_lengths else np.array([0])
    print(f"\n=== fiberhmm-merge ===", file=sys.stderr)
    print(f"resolved pairs seen: {n_pairs_seen:,}  | consensus reads built: "
          f"{n_consensus:,}  | build failures (MD/coverage): {n_build_fail:,}",
          file=sys.stderr)
    print(f"reads passed through (unmerged): {n_passthrough:,}", file=sys.stderr)
    print(f"consensus length: median {int(np.median(cl)):,}  N50 {_n50(consensus_lengths):,}  "
          f"max {int(cl.max()):,}", file=sys.stderr)
    print(f"2-strand (both-strand) genome coverage: {both_bp/1e6:.1f} Mb "
          f"= {100.0*both_bp/max(genome,1):.2f}% of the {genome/1e9:.2f} Gb main genome",
          file=sys.stderr)
    print(f"-> {out_bam} [{time.time()-t0:.0f}s]", file=sys.stderr)
    return {'n_consensus': n_consensus, 'both_bp': both_bp, 'genome': genome}


def main():
    p = argparse.ArgumentParser(
        prog='fiberhmm-merge',
        description='Build both-strand consensus reads from fiberhmm-pair output.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    fiberhmm-pair  -i calls.bam        -o calls.paired.bam
    fiberhmm-merge -i calls.paired.bam -o calls.consensus.bam
    fiberhmm-merge -i calls.paired.bam -o consensus_only.bam --pairs-only
        """,
    )
    p.add_argument('-i', '--input', required=True, help='BAM from fiberhmm-pair (mt/mp tags)')
    p.add_argument('-o', '--output', required=True, help='Output consensus BAM (sorted + indexed)')
    p.add_argument('--pairs-only', action='store_true',
                   help='Emit only consensus reads (default: also pass through unmerged reads)')
    p.add_argument('--recall', action='store_true',
                   help='Re-call footprints on each both-strand consensus read (HMM '
                        'layer over both strands; writes ns/nl/as/al + MA nuc./msp.). '
                        'Reads the deam+/deam- regime and uses C and G targets jointly.')
    p.add_argument('--enzyme', default='ddda', help='Model preset for --recall (default ddda)')
    p.add_argument('--phase-nrl', type=int, default=196,
                   help='Nucleosome repeat length for consensus recall (default 196)')
    p.add_argument('--nuc-recall-policy', choices=['conservative', 'topology'],
                   default='conservative',
                   help='Nucleosome geometry policy for consensus recall')
    p.add_argument(
        '--ddda-derived-tf-max-edge-gap', type=int, default=12, metavar='BP',
        help='With --recall, require TF calls exposed solely by DddA radial '
             'nucleosome refinement to have a deamination hit within BP on '
             'both sides (default 12; -1 disables).',
    )
    p.add_argument('-p', '--prob-threshold', type=int, default=0,
                   help='Min ML prob for MM/ML dU calls (default 0)')
    p.add_argument('--io-threads', type=int, default=4, help='htslib compression threads (default 4)')
    args = p.parse_args()

    if args.ddda_derived_tf_max_edge_gap < -1:
        p.error("--ddda-derived-tf-max-edge-gap must be -1 or >= 0")

    if not os.path.exists(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    run_merge(args.input, args.output, prob_threshold=args.prob_threshold,
              pairs_only=args.pairs_only, io_threads=args.io_threads,
              recall=args.recall, enzyme=args.enzyme,
              phase_nrl=args.phase_nrl,
              nuc_recall_policy=args.nuc_recall_policy,
              derived_tf_max_edge_ambiguity=(
                  None if args.ddda_derived_tf_max_edge_gap < 0
                  else args.ddda_derived_tf_max_edge_gap))


if __name__ == '__main__':
    main()
