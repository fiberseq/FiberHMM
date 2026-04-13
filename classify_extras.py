#!/usr/bin/env python3
"""Classify every recaller tn/tl call by its relationship to v2's features.

For each recaller call, find which v2 feature (if any) contains it:
  - INSIDE v2 short nuc (nl < long_nuc_min): normal match
  - INSIDE v2 big nuc (nl >= long_nuc_min): would require boundary-sweep;
    should be 0 with default --boundary-sweep 0
  - INSIDE v2 MSP (as/al): true "rescue" -- v2 missed this
  - UNASSIGNED: outside any v2 feature

For matched short-nuc calls, also report:
  - 1:1 match (one recaller call = one v2 short-nuc)
  - N:1 match (multiple recaller calls overlap the same v2 short-nuc -- v2 "merged"
    multiple real features into one)

Also quantify length disparity for matched pairs.
"""
import argparse
from collections import Counter, defaultdict

import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_arr(r, tag):
    try: return list(r.get_tag(tag))
    except KeyError: return []


def interval_overlap(a_s, a_e, b_s, b_e):
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def contains(outer_s, outer_e, inner_s, inner_e, slop=5):
    return (outer_s - slop) <= inner_s and inner_e <= (outer_e + slop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--long-nuc-min', type=int, default=90)
    ap.add_argument('--min-ts', type=int, default=0)
    args = ap.parse_args()

    cats = Counter()          # classification of recaller calls
    overlap_count = Counter() # N recaller calls per v2 short nuc
    length_diffs = []         # (v2_len, sum_rc_len_overlap) for each v2 short nuc
    matched_pair_lengths = [] # (v2_len, rc_len) per matched pair

    bam = pysam.AlignmentFile(args.in_bam, 'rb', check_sq=False)
    n_reads = 0
    for r in bam:
        ns = get_arr(r, 'ns'); nl = get_arr(r, 'nl')
        as_ = get_arr(r, 'as'); al = get_arr(r, 'al')
        if not ns: continue
        tn = get_arr(r, 'tn'); tl = get_arr(r, 'tl')
        ts = get_arr(r, 'ts') if r.has_tag('ts') else [255] * len(tn)
        rc = [(int(s), int(l), int(t)) for s, l, t in zip(tn, tl, ts) if t >= args.min_ts]
        v2_short = [(int(s), int(s) + int(l), int(l)) for s, l in zip(ns, nl)
                    if 0 < int(l) < args.long_nuc_min]
        v2_big = [(int(s), int(s) + int(l), int(l)) for s, l in zip(ns, nl)
                  if int(l) >= args.long_nuc_min]
        v2_msp = [(int(s), int(s) + int(l)) for s, l in zip(as_, al) if int(l) > 0]
        n_reads += 1

        # Per v2 short nuc: count overlapping recaller calls + sum their length
        for vs, ve, vl in v2_short:
            n_ov = 0
            sum_rc_len = 0
            for rs, rl, _ in rc:
                if interval_overlap(vs, ve, rs, rs + rl) > 0:
                    n_ov += 1
                    sum_rc_len += rl
                    matched_pair_lengths.append((vl, rl))
            overlap_count[n_ov] += 1
            length_diffs.append((vl, sum_rc_len, n_ov))

        # Per recaller call: classify against v2 features
        for rs, rl, rt in rc:
            re = rs + rl
            # Short nuc match: any overlap
            short_hit = any(interval_overlap(vs, ve, rs, re) > 0 for vs, ve, _ in v2_short)
            if short_hit:
                cats['matches_v2_short_nuc'] += 1
                continue
            # Big-nuc containment
            big_hit = any(contains(vs, ve, rs, re) for vs, ve, _ in v2_big)
            if big_hit:
                cats['inside_v2_big_nuc'] += 1
                continue
            # MSP containment
            msp_hit = any(contains(ms, me, rs, re) for ms, me in v2_msp)
            if msp_hit:
                cats['inside_v2_msp_no_short_match'] += 1
                continue
            cats['unassigned'] += 1

    bam.close()

    total = sum(cats.values())
    print(f'\n{n_reads} reads processed; {total} recaller calls classified (ts>={args.min_ts})\n')
    print('=== Recaller call classification ===')
    for k in ['matches_v2_short_nuc', 'inside_v2_big_nuc',
              'inside_v2_msp_no_short_match', 'unassigned']:
        v = cats.get(k, 0)
        print(f'  {k:40s}: {v:>7}  ({100*v/max(1,total):5.1f}%)')

    print('\n=== Recaller calls per v2 short nuc (N:1 analysis) ===')
    total_v2 = sum(overlap_count.values())
    for n in sorted(overlap_count):
        v = overlap_count[n]
        label = f'{n} recaller calls' if n != 1 else '1 recaller call'
        print(f'  {label:25s}: {v:>7}  ({100*v/max(1,total_v2):5.1f}%)')

    # How many v2 short nucs are "merged" (overlapped by >=2 recaller calls)?
    merged = sum(v for n, v in overlap_count.items() if n >= 2)
    merged_frac = merged / max(1, total_v2 - overlap_count.get(0, 0))
    print(f'\nAmong MATCHED v2 short nucs (at least 1 recaller call overlapping),')
    print(f'  fraction overlapped by >=2 recaller calls: {merged_frac:.1%}')
    print(f'  ("v2 merged N real features" hypothesis)')

    # Length disparity: v2_len - recaller_match_len (per pair)
    if matched_pair_lengths:
        v2l = np.array([p[0] for p in matched_pair_lengths])
        rcl = np.array([p[1] for p in matched_pair_lengths])
        diff = v2l - rcl
        print(f'\n=== Length disparity (matched pair v2_len - rc_len) ===')
        print(f'  n matched pairs: {len(diff)}')
        print(f'  median v2_len - rc_len = {np.median(diff):.0f} bp')
        print(f'  mean   v2_len - rc_len = {np.mean(diff):.1f} bp')
        print(f'  v2_len median: {np.median(v2l):.0f}; rc_len median: {np.median(rcl):.0f}')

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Classification pie
    ax = axes[0, 0]
    labels = []; sizes = []; colors = []
    order = [('matches_v2_short_nuc', '#1e3a8a'),
             ('inside_v2_msp_no_short_match', '#f59e0b'),
             ('inside_v2_big_nuc', '#dc2626'),
             ('unassigned', '#6b7280')]
    for k, c in order:
        v = cats.get(k, 0)
        if v > 0:
            labels.append(f'{k}\n({v:,}; {100*v/total:.1f}%)')
            sizes.append(v); colors.append(c)
    ax.pie(sizes, labels=labels, colors=colors, startangle=90,
           textprops={'fontsize': 9})
    ax.set_title(f'Recaller call classification  (n={total})')

    # 2. Histogram: recaller calls per v2 short nuc
    ax = axes[0, 1]
    ns_ = sorted(overlap_count)
    vs_ = [overlap_count[k] for k in ns_]
    ax.bar(ns_, vs_, color='#1e3a8a')
    ax.set_xlabel('recaller calls overlapping one v2 short nuc')
    ax.set_ylabel('count of v2 short nucs')
    ax.set_title('N:1 recaller-to-v2 match')
    ax.grid(alpha=0.3)

    # 3. Length disparity histogram
    ax = axes[1, 0]
    if matched_pair_lengths:
        ax.hist(diff, bins=np.arange(-60, 100, 3), color='#dc2626', alpha=0.75)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.axvline(np.median(diff), color='blue', linestyle='-', linewidth=1,
                   label=f'median={np.median(diff):.0f} bp')
        ax.set_xlabel('v2_len - recaller_match_len (bp)')
        ax.set_ylabel('count')
        ax.set_title('Per-matched-pair length disparity')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # 4. v2 len vs TOTAL rc len coverage (when >=1 overlap)
    ax = axes[1, 1]
    pairs = [(v2_len, sum_rc, n) for v2_len, sum_rc, n in length_diffs if n >= 1]
    if pairs:
        v2ls = np.array([p[0] for p in pairs])
        srl = np.array([p[1] for p in pairs])
        n_ov = np.array([p[2] for p in pairs])
        idx = np.random.choice(len(v2ls), min(15000, len(v2ls)), replace=False)
        sc = ax.scatter(v2ls[idx], srl[idx], c=n_ov[idx], s=4, alpha=0.35, cmap='viridis')
        mx = max(v2ls.max(), srl.max(), 90)
        ax.plot([0, mx], [0, mx], 'r--', linewidth=0.7, alpha=0.7, label='y=x')
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.set_xlabel('v2 short nuc length (bp)')
        ax.set_ylabel('sum of matching recaller call lengths (bp)')
        ax.set_title('v2 length vs summed recaller coverage  (color = #rc calls)')
        ax.legend(fontsize=9)
        plt.colorbar(sc, ax=ax, label='# recaller calls overlapping')
        ax.grid(alpha=0.3)

    fig.suptitle(f'Classifying recaller extras: where do they come from?  '
                 f'(ts>={args.min_ts}, boundary-sweep OFF)', fontsize=12)
    fig.tight_layout()
    out_png = args.out_prefix + '_classify.png'
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    print(f'\nWrote {out_png}')


if __name__ == '__main__':
    main()
