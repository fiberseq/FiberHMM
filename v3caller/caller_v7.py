#!/usr/bin/env python3
"""Nucleosome caller v7 — permissive atoms + Poisson-merge.

Pipeline per read:
  Pass 1 (permissive atom detection):
    - W bp rolling rate
    - Find contiguous runs where rate < read_baseline, no bridging
    - Structurally merge atoms whose ref-frame windows overlap

  Pass 2 (single-sweep gap merging):
    - For each inter-atom gap, compute one-sided Poisson p-value
      for "gap is significantly elevated above baseline" (= real linker)
    - Merge if p >= merge_alpha (gap is not significantly elevated)
    - Also merge if gap is structurally small (<= gap_override_length
      bp AND < gap_override_opp opportunities) — the 2D biological
      prior: short low-info gaps are bursty breathing, not linkers
    - A single left-to-right sweep is equivalent to iterated merging
      because each gap's statistics depend only on its own ref
      interval, which doesn't change when its neighbors merge.

  Pass 3 (filter + emit):
    - Drop runs shorter than min_footprint
    - nq = tightness of best (lowest-rate) window inside the run,
      255 * (1 - min_rate / baseline), clamped
    - Map ref intervals -> query coords, emit ns/nl/as/al/nq

Parameters (two meaningful knobs, everything else biologically fixed):
  --merge-alpha       0.05    Poisson significance threshold
  --min-read-rate     0.05    reject reads below this overall rate
  --W                 40
  --min-footprint     80
  --gap-override-length 30    short-gap biological override
  --gap-override-opp    10    low-info biological override

No min_sep, no max_flank, no core_ratio, no bridge_length, no grow_ratio.
"""

import argparse
import array
import os
import sys

import numpy as np
import pysam
from scipy.stats import poisson

from enzyme_extractors import get_extractor


# -------------------------------------------------------------------
# Core primitives (shared with plot_core_profile / core_sweep / v6)
# -------------------------------------------------------------------

def windowed_rate(opp, hit, W):
    """Rolling W-bp hit rate along the read."""
    opp_cum = np.concatenate([[0], np.cumsum(opp, dtype=np.int32)])
    hit_cum = np.concatenate([[0], np.cumsum(hit, dtype=np.int32)])
    opp_win = opp_cum[W:] - opp_cum[:-W]
    hit_win = hit_cum[W:] - hit_cum[:-W]
    valid = opp_win >= 5
    rate = np.zeros_like(opp_win, dtype=np.float32)
    rate[valid] = hit_win[valid] / opp_win[valid]
    return rate, valid


def build_ref_to_query_map(read, L):
    """Length-L int32 array mapping read-local ref pos -> query pos
    (-1 for unaligned positions)."""
    m = np.full(L, -1, dtype=np.int32)
    for qp, rp in read.get_aligned_pairs(matches_only=True):
        if rp is None:
            continue
        rr = rp - read.reference_start
        if 0 <= rr < L:
            m[rr] = qp
    return m


def ref_interval_to_query(ref_to_q, s, e):
    """[s, e) ref-local -> (q_start, q_end_exclusive) query coords.
    Returns None if the interval contains no aligned bases."""
    if e <= s:
        return None
    sub = ref_to_q[s:e]
    mask = sub >= 0
    if not mask.any():
        return None
    idx = np.where(mask)[0]
    q_start = int(sub[idx[0]])
    q_end = int(sub[idx[-1]]) + 1
    if q_end <= q_start:
        return None
    return q_start, q_end


# -------------------------------------------------------------------
# Per-read caller
# -------------------------------------------------------------------

def find_pass1_atoms(rate, valid, W, baseline):
    """Pass 1: contiguous runs where rate < baseline (no bridging).

    Returns list of (ref_start, ref_end_exclusive) in read-local ref
    coords. Overlapping atoms (due to W-bp window overlap) are
    structurally merged.
    """
    below = valid & (rate < baseline)
    if not below.any():
        return []

    # Find run boundaries in the rate frame via diff on a padded array
    padded = np.concatenate([[False], below, [False]])
    diff = np.diff(padded.astype(np.int8))
    rate_starts = np.where(diff == 1)[0]
    rate_ends = np.where(diff == -1)[0]  # exclusive

    # Convert rate-frame runs to ref-frame intervals. A protected run
    # [s, e) in the rate frame covers ref [s, e + W - 1) — the first
    # protected window starts at ref s, the last (at rate index e-1)
    # ends at ref e-1+W = e+W-1.
    ref_starts = rate_starts
    ref_ends = rate_ends + (W - 1)  # exclusive

    # Structural merge: if atoms overlap in ref frame (possible because
    # adjacent W-bp windows can overlap even when the intervening rate
    # indices aren't themselves protected), collapse them.
    atoms = []
    cur_s = int(ref_starts[0])
    cur_e = int(ref_ends[0])
    for s, e in zip(ref_starts[1:], ref_ends[1:]):
        s = int(s)
        e = int(e)
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            atoms.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    atoms.append((cur_s, cur_e))
    return atoms


def poisson_merge_atoms(atoms, opp, hit, baseline,
                        merge_alpha, gap_override_length, gap_override_opp):
    """Pass 2: single-sweep Poisson merge of adjacent atoms.

    For each inter-atom gap, test the one-sided null "gap rate is at
    baseline" against the alternative "gap rate is elevated." Merge if
    we can't reject the null (p >= merge_alpha) — the absence of
    statistical evidence of elevation is interpreted as "protected-
    looking."

    Additionally apply a 2D biological prior: short + low-info gaps
    are bursty breathing, not linkers — merge unconditionally.

    Single left-to-right sweep is mathematically equivalent to
    iterated merging because each gap's statistics depend only on its
    own fixed ref interval.
    """
    if len(atoms) < 2:
        return list(atoms)

    result = [atoms[0]]
    for s, e in atoms[1:]:
        prev_s, prev_e = result[-1]
        gap_s = prev_e
        gap_e = s

        if gap_e <= gap_s:
            # Should not happen after structural merge, but be safe:
            # intervals touch or overlap — absorb.
            result[-1] = (prev_s, max(prev_e, e))
            continue

        gap_len = gap_e - gap_s
        gap_opp = int(opp[gap_s:gap_e].sum())
        gap_hit = int(hit[gap_s:gap_e].sum())

        # 2D biological prior: short low-info gap -> always merge
        biological_override = (gap_len <= gap_override_length and
                               gap_opp < gap_override_opp)

        if biological_override:
            result[-1] = (prev_s, max(prev_e, e))
            continue

        # Poisson test for elevation:
        #   H0: gap rate = baseline (expected hits = gap_opp * baseline)
        #   HA: gap rate > baseline (real linker)
        #   p = P(X >= gap_hit | Poisson(lambda))
        # If p >= merge_alpha we fail to reject H0, meaning we don't
        # have evidence this is a linker -> merge.
        if gap_opp == 0:
            # No opportunities, no information -> merge conservatively
            result[-1] = (prev_s, max(prev_e, e))
            continue
        if gap_hit == 0:
            # Zero observed hits, cannot be elevated -> merge
            result[-1] = (prev_s, max(prev_e, e))
            continue

        lam = gap_opp * baseline
        p_val = float(poisson.sf(gap_hit - 1, lam))

        if p_val >= merge_alpha:
            # Not significantly elevated -> merge
            result[-1] = (prev_s, max(prev_e, e))
        else:
            # Significantly elevated -> real linker, keep split
            result.append((s, e))

    return result


def call_read(read, ref_seq, extractor, W,
              min_read_rate, merge_alpha,
              gap_override_length, gap_override_opp, min_footprint):
    """Run v7 pipeline on one read; return dict of ns/nl/as/al/nq
    or None if no calls."""
    L = read.reference_end - read.reference_start
    if L < W + 100:
        return None

    opp, hit = extractor.read_to_arrays(read, ref_seq)
    n_opp = int(opp.sum())
    if n_opp < 50:
        return None
    baseline = int(hit.sum()) / n_opp
    if baseline < min_read_rate:
        return None

    rate, valid = windowed_rate(opp, hit, W)
    if valid.sum() < 100:
        return None

    # Pass 1: permissive atoms
    atoms = find_pass1_atoms(rate, valid, W, baseline)
    if not atoms:
        return None

    # Pass 2: Poisson merge
    merged = poisson_merge_atoms(atoms, opp, hit, baseline,
                                  merge_alpha,
                                  gap_override_length,
                                  gap_override_opp)
    if not merged:
        return None

    # Clip to read range
    merged_clipped = []
    for s, e in merged:
        s2 = max(0, s)
        e2 = min(L, e)
        if e2 - s2 >= min_footprint:
            merged_clipped.append((s2, e2))
    if not merged_clipped:
        return None

    ref_to_q = build_ref_to_query_map(read, L)

    nuc_intervals = []
    nuc_nq = []
    for s, e in merged_clipped:
        # nq: tightness of the BEST (lowest-rate) windowed position
        # whose window lies within the footprint [s, e). The rate
        # index p corresponds to ref window [p, p+W); for the window
        # to be inside the footprint we need p >= s and p + W <= e.
        rate_s = max(0, s)
        rate_e = max(rate_s, min(len(rate), e - W + 1))
        nq_val = 0
        if rate_e > rate_s:
            sub = rate[rate_s:rate_e]
            sub_valid = valid[rate_s:rate_e]
            if sub_valid.any():
                min_rate = float(np.min(sub[sub_valid]))
                if baseline > 0:
                    protection = 1.0 - min_rate / baseline
                    if protection < 0.0:
                        protection = 0.0
                    elif protection > 1.0:
                        protection = 1.0
                    nq_val = int(round(protection * 255))

        qi = ref_interval_to_query(ref_to_q, s, e)
        if qi is None:
            continue
        nuc_intervals.append(qi)
        nuc_nq.append(nq_val)

    if not nuc_intervals:
        return None

    # Sort and absorb any residual overlaps in query coords
    pairs = sorted(zip(nuc_intervals, nuc_nq), key=lambda x: x[0][0])
    merged_intervals = [pairs[0][0]]
    merged_nq = [pairs[0][1]]
    for (qs, qe), nq_val in pairs[1:]:
        last_qs, last_qe = merged_intervals[-1]
        if qs <= last_qe:
            merged_intervals[-1] = (last_qs, max(last_qe, qe))
            if nq_val > merged_nq[-1]:
                merged_nq[-1] = nq_val
        else:
            merged_intervals.append((qs, qe))
            merged_nq.append(nq_val)

    ns = [qs for qs, _ in merged_intervals]
    nl = [qe - qs for qs, qe in merged_intervals]
    nq = merged_nq

    # Accessible = complement in [0, query_length)
    qlen = read.query_length or 0
    as_list = []
    al_list = []
    cursor = 0
    for qs, qe in merged_intervals:
        if qs > cursor:
            as_list.append(cursor)
            al_list.append(qs - cursor)
        cursor = max(cursor, qe)
    if cursor < qlen:
        as_list.append(cursor)
        al_list.append(qlen - cursor)

    return {
        'ns': ns,
        'nl': nl,
        'nq': nq,
        'as': as_list,
        'al': al_list,
    }


# -------------------------------------------------------------------
# BAM I/O helpers (same as v6)
# -------------------------------------------------------------------

STALE_CALL_TAGS = ('ns', 'nl', 'nq', 'as', 'al')
MOD_TAGS = ('MM', 'ML', 'Mm', 'Ml')


def set_array_tag(read, tag, values):
    if values:
        read.set_tag(tag, array.array('I', [max(0, int(v)) for v in values]))


def clear_stale_tags(read):
    for tag in STALE_CALL_TAGS:
        if read.has_tag(tag):
            read.set_tag(tag, None)


def strip_mod_tags(read):
    for tag in MOD_TAGS:
        if read.has_tag(tag):
            read.set_tag(tag, None)


# -------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', required=True)
    ap.add_argument('--out-bam', required=True)
    ap.add_argument('--fa', required=True)
    ap.add_argument('--enzyme', required=True, choices=['daf', 'hia5'])
    ap.add_argument('--W', type=int, default=40,
                    help='scan window width (default 40)')
    ap.add_argument('--min-read-rate', type=float, default=0.05,
                    help='reject reads with overall rate below this')
    ap.add_argument('--merge-alpha', type=float, default=0.05,
                    help='Poisson significance threshold for keeping '
                         'a split. p >= alpha -> merge (default 0.05)')
    ap.add_argument('--gap-override-length', type=int, default=30,
                    help='2D biological prior: gaps <= this bp are '
                         'force-merged regardless of p-value (bursty '
                         'breathing, not linkers)')
    ap.add_argument('--gap-override-opp', type=int, default=10,
                    help='2D biological prior: gaps with fewer than '
                         'this many opportunities are force-merged '
                         '(low-info noise)')
    ap.add_argument('--min-footprint', type=int, default=80,
                    help='reject merged runs shorter than this')
    ap.add_argument('--max-reads', type=int, default=0,
                    help='0 = no limit')
    ap.add_argument('--strip-mods', action='store_true',
                    help='drop MM/ML tags to shrink output')
    ap.add_argument('--ml-threshold', type=int, default=128,
                    help='Hia5 only: m6A ML threshold')
    ap.add_argument('--progress-every', type=int, default=1000)
    args = ap.parse_args()

    extractor_kwargs = {}
    if args.enzyme == 'hia5':
        extractor_kwargs['ml_threshold'] = args.ml_threshold
    extractor = get_extractor(args.enzyme, **extractor_kwargs)

    out_dir = os.path.dirname(args.out_bam) or '.'
    os.makedirs(out_dir, exist_ok=True)

    bam_in = pysam.AlignmentFile(args.in_bam, 'rb', check_sq=False)
    fa = pysam.FastaFile(args.fa)
    bam_out = pysam.AlignmentFile(args.out_bam, 'wb', template=bam_in)

    n_reads = 0
    n_called = 0
    n_skipped = 0
    n_passthrough = 0
    nuc_counts = []

    print(f'caller_v7  enzyme={args.enzyme}  W={args.W}  '
          f'merge_alpha={args.merge_alpha}  '
          f'gap_override=({args.gap_override_length}bp,'
          f'{args.gap_override_opp}opp)  '
          f'min_footprint={args.min_footprint}  '
          f'min_read_rate={args.min_read_rate}',
          flush=True)

    for read in bam_in.fetch(until_eof=True):
        n_reads += 1
        if args.max_reads and n_reads > args.max_reads:
            break

        if read.is_secondary or read.is_supplementary or read.is_unmapped:
            bam_out.write(read)
            n_passthrough += 1
            continue

        try:
            ref_seq = fa.fetch(read.reference_name,
                               read.reference_start,
                               read.reference_end).upper()
        except Exception:
            clear_stale_tags(read)
            if args.strip_mods:
                strip_mod_tags(read)
            bam_out.write(read)
            n_skipped += 1
            continue
        if len(ref_seq) != read.reference_end - read.reference_start:
            clear_stale_tags(read)
            if args.strip_mods:
                strip_mod_tags(read)
            bam_out.write(read)
            n_skipped += 1
            continue

        result = call_read(read, ref_seq, extractor,
                           args.W, args.min_read_rate, args.merge_alpha,
                           args.gap_override_length, args.gap_override_opp,
                           args.min_footprint)

        clear_stale_tags(read)
        if args.strip_mods:
            strip_mod_tags(read)

        if result is None:
            bam_out.write(read)
            n_skipped += 1
        else:
            set_array_tag(read, 'ns', result['ns'])
            set_array_tag(read, 'nl', result['nl'])
            set_array_tag(read, 'as', result['as'])
            set_array_tag(read, 'al', result['al'])
            if result['nq']:
                read.set_tag('nq', array.array('B', result['nq']))
            bam_out.write(read)
            n_called += 1
            nuc_counts.append(len(result['ns']))

        if n_called and n_called % args.progress_every == 0:
            print(f'  {n_called} called, {n_skipped} skipped, '
                  f'{n_passthrough} passthrough', flush=True)

    bam_in.close()
    bam_out.close()
    fa.close()

    # Sort + index
    try:
        tmp = args.out_bam + '.tmp'
        pysam.sort('-o', tmp, args.out_bam)
        os.replace(tmp, args.out_bam)
        pysam.index(args.out_bam)
    except Exception as e:
        print(f'Warning: sort/index failed: {e}', flush=True)

    nc = np.array(nuc_counts) if nuc_counts else np.array([])
    print()
    print('=' * 60)
    print(f'Total reads:   {n_reads}')
    print(f'Called:        {n_called}')
    print(f'Skipped:       {n_skipped}')
    print(f'Passthrough:   {n_passthrough}')
    if len(nc):
        print(f'Nucs per read: mean={nc.mean():.1f}  '
              f'median={int(np.median(nc))}  '
              f'min={nc.min()}  max={nc.max()}')
    print('=' * 60)


if __name__ == '__main__':
    main()
