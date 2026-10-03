#!/usr/bin/env python3
"""fiberhmm-merge -- build both-strand consensus reads from cross-strand pairs.

DEPRECATED as a command: use ``fiberhmm-pair``, which pairs, merges and re-calls
in one step, or ``fiberhmm-pair --from-paired`` for an already paired BAM. This
module keeps ``run_merge`` (the merge stage) and a working compatibility CLI.

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
    mc/mg/pm/pa/sb/sd/sr/sg/dm/mv  pairing evidence carried from fiberhmm-pair
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

from fiberhmm.models import DEFAULT_PROB_THRESHOLD
from fiberhmm.crossstrand.consensus import build_consensus
from fiberhmm.crossstrand.pairing import FLAVOR_CT, read_flavor
from fiberhmm.crossstrand.recall import (
    RecallContext,
    consensus_cpg_intervals,
    recall_consensus_full,
)
from fiberhmm.inference.bam_output import atomic_output, temporary_output_path
from fiberhmm.io.bam_header import append_ma_types

_TAG_SOURCES = 'cs'
_PAIR_TAGS = ('mc', 'mg', 'pm', 'pa', 'sb', 'sd', 'sr', 'sg', 'dm', 'mv')


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


def _merge_output_header(header, *, recall, enzyme, prob_threshold, pairs_only,
                         nuc_recall_policy, phase_nrl, cpg_mask_policy):
    """Input header plus this merge's @PG and, with recall, its chemistry.

    The joint recall calls footprints with the bundled ``enzyme`` tables, so
    the output declares that chemistry with the tables' digests (as
    ``fiberhmm-call`` does); a merge without recall makes no calls and only
    records its @PG.
    """
    from types import SimpleNamespace

    import fiberhmm
    from fiberhmm.cli.provenance import (
        chemistry_declaration,
        output_header_with_provenance,
    )
    from fiberhmm.models import get_model_path

    from fiberhmm.io.annotation_frame import append_coord_to_ds, pass_through_frame
    # Consensus records are forward (frame-free); every other record keeps
    # the input's footprint tags, so the output frame is the input's.
    record = {
        'PN': 'fiberhmm-merge',
        'VN': getattr(fiberhmm, '__version__', 'unknown'),
        'CL': ' '.join(sys.argv),
        'DS': append_coord_to_ds(
            f"both-strand consensus of CT/GA pairs; recall={'on' if recall else 'off'} "
            f"enzyme={enzyme if recall else 'n/a'} prob_threshold={prob_threshold} "
            f"pairs_only={'on' if pairs_only else 'off'}"
            + (f" nuc_recall_policy={nuc_recall_policy} phase_nrl={phase_nrl} "
               f"cpg_mask={cpg_mask_policy or 'off'}" if recall else ''),
            pass_through_frame(header)),
    }
    if recall:
        # The DddA tables do not depend on the platform; declare the one the
        # input declares (e.g. DddA on Nanopore) rather than the DddA default,
        # which would conflict with it.
        from fiberhmm.io.bam_header import declared_chemistries

        platforms = {
            str(d.get('platform', '')).lower() for d in declared_chemistries(header)
            if str(d.get('mode', '')).lower() == 'daf'
        } - {'', 'unknown'}
        seq = next(iter(platforms)) if len(platforms) == 1 else None
        record['chemistry'] = chemistry_declaration(
            SimpleNamespace(enzyme=enzyme, seq=seq), 'daf',
            get_model_path(enzyme, tool='apply'),
            get_model_path(enzyme, tool='recall'),
            nuc_model_path=(get_model_path(enzyme, tool='nuc_refine')
                            if enzyme == 'ddda' else None),
        )
    return output_header_with_provenance(header, record)


def run_merge(in_bam, out_bam, prob_threshold=0, pairs_only=False, io_threads=4,
              recall=False, enzyme='ddda', phase_nrl=0,
              nuc_recall_policy='conservative',
              derived_tf_max_edge_ambiguity=12, use_m5c=None):
    t0 = time.time()
    ctx = None
    if recall:
        ctx = RecallContext(enzyme, use_m5c=use_m5c)
        print(f"merge-recall: loaded {enzyme} apply+recall models (k={ctx.k}); "
              f"re-calling footprints (HMM + nuc + TF recallers) on both-strand "
              f"consensus reads; CpG-aware recall "
              f"{ctx.cpg_mask_policy or 'off'}", file=sys.stderr)
    bam = pysam.AlignmentFile(in_bam, 'rb')
    header = append_ma_types(
        bam.header, ['deam', 'nuc', 'msp', 'tf'] if recall else ['deam'])
    header = _merge_output_header(
        header, recall=recall, enzyme=enzyme, prob_threshold=prob_threshold,
        pairs_only=pairs_only, nuc_recall_policy=nuc_recall_policy,
        phase_nrl=phase_nrl, cpg_mask_policy=ctx.cpg_mask_policy if ctx else None)

    # Records are written unsorted to a hidden sibling, sorted into another
    # hidden sibling and only then renamed to out_bam; the unsorted file is
    # removed however the run ends (it used to survive as out.bam.unsorted.bam).
    unsorted = temporary_output_path(out_bam)
    try:
        return _run_merge_passes(
            bam, in_bam, out_bam, unsorted, header, prob_threshold, pairs_only,
            io_threads, ctx, phase_nrl, nuc_recall_policy,
            derived_tf_max_edge_ambiguity, t0,
        )
    finally:
        try:
            os.remove(unsorted)
        except OSError:
            pass


def _run_merge_passes(bam, in_bam, out_bam, unsorted, header, prob_threshold,
                      pairs_only, io_threads, ctx, phase_nrl, nuc_recall_policy,
                      derived_tf_max_edge_ambiguity, t0):
    """Both passes of :func:`run_merge` plus the atomic sort/index."""
    # Pass 1: per chromosome, collect the paired reads and build consensus.
    merged_names = set()          # source read names replaced by a consensus
    consensus_lengths = []
    both_intervals = defaultdict(list)   # chrom -> [(start,end)] both-strand
    n_consensus = n_pairs_seen = n_build_fail = 0
    seen_paired_names = set()

    out = pysam.AlignmentFile(unsorted, 'wb', header=header, threads=io_threads)

    def flush(tid, paired):
        """Build consensus reads for one chromosome's paired-read dict."""
        nonlocal n_consensus, n_pairs_seen, n_build_fail
        # tid is None when the input has no primary records at all.
        if tid is None or tid < 0 or not paired:
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
            cons = build_consensus(ct_read, ga_read, prob_threshold=prob_threshold)
            if cons is None:
                n_build_fail += 1
                continue
            pair_method = read.get_tag('pm') if read.has_tag('pm') else None
            if pair_method=='D':
                for tag in ('pm','dm','mg','mv'):
                    if not read.has_tag(tag) or not mate.has_tag(tag) or read.get_tag(tag)!=mate.get_tag(tag):
                        raise ValueError(f'Conflicting or missing duplex {tag} provenance for {name!r}/{mate_name!r}')
            pair_tags = [
                item for item in read.get_tags(with_value_type=True)
                if item[0] in _PAIR_TAGS
                and not (item[0] == 'mg' and pair_method not in ('F','D'))
            ]
            seg = _make_consensus_segment(cons, header, tid, pair_tags)
            if ctx is not None:
                cpg_intervals = (
                    consensus_cpg_intervals(ct_read, ga_read,
                                            cons.ref_start, cons.length)
                    if ctx.cpg_mask_policy else None
                )
                recalled = recall_consensus_full(
                    seg, ctx, phase_nrl=phase_nrl,
                    nuc_recall_policy=nuc_recall_policy,
                    derived_tf_max_edge_ambiguity=(
                        derived_tf_max_edge_ambiguity),
                    cpg_intervals=cpg_intervals,
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

    # Sort into the temporary and index it there; the BAM and its index are
    # published together, so a failed index keeps any previous output intact.
    with atomic_output(out_bam, finalize=pysam.index) as sorted_path:
        pysam.sort('-@', str(io_threads), '-O', 'bam', '-o', sorted_path,
                   unsorted)

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
    print('fiberhmm-merge is deprecated: use `fiberhmm-pair` (pair -> merge -> recall), '
          'or `fiberhmm-pair --from-paired` for an already paired BAM.', file=sys.stderr)
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
                   help='Re-call footprints on each both-strand consensus read (HMM, '
                        'nucleosome and TF recall over both strands; writes '
                        'ns/nl/as/al + MA nuc/msp/tf). '
                        'Reads the deam+/deam- regime and uses C and G targets jointly.')
    p.add_argument('--enzyme', default='ddda', help='Model preset for --recall (default ddda)')
    p.add_argument('--phase-nrl', type=int, default=0,
                   help='Periodicity prior for consensus recall: nucleosome repeat length '
                        'in bp, or 0 for off (default 0). DddA radial recall ignores it.')
    p.add_argument('--nuc-recall-policy', choices=['conservative', 'topology'],
                   default='conservative',
                   help='Nucleosome geometry policy for consensus recall')
    p.add_argument(
        '--ddda-derived-tf-max-edge-gap', type=int, default=12, metavar='BP',
        help='With --recall, require TF calls exposed solely by DddA radial '
             'nucleosome refinement to have a deamination hit within BP on '
             'both sides (default 12; -1 disables).',
    )
    p.add_argument('--use-m5c', action=argparse.BooleanOptionalAction,
                   default=None,
                   help='With --recall: DddA CpG-aware recall, as in '
                        'fiberhmm-call and fiberhmm-recall-tfs (CpGs excluded '
                        'except inside the source reads\' ddda_ucg islands). '
                        'Default: on for --enzyme ddda.')
    p.add_argument('-p', '--prob-threshold', type=int,
                   default=DEFAULT_PROB_THRESHOLD,
                   help='Min ML probability for MM/ML-native dU calls (0-255; '
                        f'default {DEFAULT_PROB_THRESHOLD}, the same as '
                        'fiberhmm-call). R/Y- and MD-encoded input is binary '
                        'and ignores it.')
    p.add_argument('--io-threads', type=int, default=4, help='htslib compression threads (default 4)')
    from fiberhmm.core.bam_reader import add_daf_run_mask_arguments, apply_daf_run_mask_arguments
    add_daf_run_mask_arguments(p)
    from fiberhmm.cli.common import add_version_args
    add_version_args(p)
    args = p.parse_args()
    try:
        apply_daf_run_mask_arguments(args, args.enzyme)
    except ValueError as exc:
        p.error(str(exc))

    if args.ddda_derived_tf_max_edge_gap < -1:
        p.error("--ddda-derived-tf-max-edge-gap must be -1 or >= 0")

    if not os.path.exists(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    # -i X -o X is refused, not done in place.
    from fiberhmm.cli.common import PathAliasError, check_path_aliases
    try:
        check_path_aliases(inputs={'--input': args.input},
                           outputs={'--output': args.output})
    except PathAliasError as exc:
        p.error(str(exc))
    run_merge(args.input, args.output, prob_threshold=args.prob_threshold,
              pairs_only=args.pairs_only, io_threads=args.io_threads,
              recall=args.recall, enzyme=args.enzyme,
              phase_nrl=args.phase_nrl,
              nuc_recall_policy=args.nuc_recall_policy,
              derived_tf_max_edge_ambiguity=(
                  None if args.ddda_derived_tf_max_edge_gap < 0
                  else args.ddda_derived_tf_max_edge_gap),
              use_m5c=args.use_m5c)


if __name__ == '__main__':
    main()
