#!/usr/bin/env python3
"""v2 vs v3 per-call overlap analysis.

For each read in the input BAM (must have both `fp_v2+` and
v3 nuc/tf annotations in MA tag):

1. Match each v2 footprint to its best-overlapping v3 call (nuc or tf).
2. Classify v2 footprints:
   - **Preserved-as-nuc**: v2 size ≥90 + matched v3 nuc, IoU ≥ 0.5
   - **Preserved-as-tf**: v2 size <90 + matched v3 tf, IoU ≥ 0.5
   - **Reclassified**: matched but v3 category differs from v2's size bin
   - **Split**: v2 overlaps ≥2 v3 calls (overmerge that v3 broke up)
   - **Lost**: no v3 call overlaps v2 above IoU 0.25
3. Classify v3 calls:
   - **Shared**: matched to a v2 footprint at IoU ≥ 0.5
   - **New**: no v2 footprint within IoU 0.25

Outputs:
- `overlap_summary.tsv`: per-read counts in each category
- `overlap_calls.tsv`: per-call row (chrom/start/length/category/ious)
- `iou_distribution.png`: IoU hist of matched pairs
- `size_comparison.png`: v2 vs v3 size scatter for matched pairs
- `classification_breakdown.png`: stacked bars of v2 fates by size bucket
- `v3_vs_v2_hit_rates.png`: v3 quality vs match rate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_ma_calls import iter_reads


def iou(a_start, a_len, b_start, b_len):
    a_end = a_start + a_len
    b_end = b_start + b_len
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def best_match(query, pool, min_iou=0.25):
    """Return (idx, iou) of the best-overlapping item in `pool`
    (list of (s, l, ...)), or (None, 0.0) if nothing clears min_iou."""
    best_i, best_iou = None, 0.0
    qs, ql = query[0], query[1]
    for i, item in enumerate(pool):
        bs, bl = item[0], item[1]
        u = iou(qs, ql, bs, bl)
        if u > best_iou:
            best_i, best_iou = i, u
    if best_iou < min_iou:
        return None, best_iou
    return best_i, best_iou


def all_overlapping(query, pool, min_iou=0.1):
    """Return list of (idx, iou) for items with iou ≥ min_iou."""
    qs, ql = query[0], query[1]
    return [(i, iou(qs, ql, item[0], item[1]))
            for i, item in enumerate(pool)
            if iou(qs, ql, item[0], item[1]) >= min_iou]


def classify_v2(v2, v3_nucs, v3_tfs, match_iou=0.5, split_iou=0.3,
                  loss_iou=0.25):
    """Return dict with fate + details for a single v2 footprint."""
    # Gather overlaps with all v3 calls (nuc and tf separately)
    v3_all = [(s, l, 'nuc', i) for i, (s, l, _) in enumerate(v3_nucs)] + \
              [(s, l, 'tf', i) for i, (s, l, _) in enumerate(v3_tfs)]
    overlaps = []
    for s, l, kind, idx in v3_all:
        u = iou(v2[0], v2[1], s, l)
        if u >= loss_iou:
            overlaps.append((kind, idx, s, l, u))

    if not overlaps:
        return {'fate': 'lost', 'best_iou': 0.0, 'n_overlap': 0}

    overlaps.sort(key=lambda x: -x[4])
    best = overlaps[0]
    best_iou_v = best[4]

    # Split: 2+ v3 calls overlap this single v2 at decent IoU (≥ split_iou
    # on each, implying v2 was the merged big thing)
    strong = [o for o in overlaps if o[4] >= split_iou]
    if len(strong) >= 2:
        return {'fate': 'split', 'best_iou': best_iou_v,
                'n_overlap': len(strong),
                'best_kind': best[0]}

    if best_iou_v >= match_iou:
        # Preserved category?
        v2_len = v2[1]
        v2_is_nuc = v2_len >= 90
        if (v2_is_nuc and best[0] == 'nuc'):
            return {'fate': 'preserved_nuc', 'best_iou': best_iou_v,
                    'best_kind': 'nuc', 'n_overlap': 1,
                    'v3_len': best[3]}
        elif (not v2_is_nuc and best[0] == 'tf'):
            return {'fate': 'preserved_tf', 'best_iou': best_iou_v,
                    'best_kind': 'tf', 'n_overlap': 1,
                    'v3_len': best[3]}
        else:
            return {'fate': 'reclassified', 'best_iou': best_iou_v,
                    'best_kind': best[0], 'n_overlap': 1,
                    'v3_len': best[3]}

    # Weak overlap only
    return {'fate': 'weak_overlap', 'best_iou': best_iou_v,
            'n_overlap': len(overlaps),
            'best_kind': best[0]}


def classify_v3(v3, v2s, match_iou=0.5, match_iou_low=0.25):
    """Return dict for a v3 call: 'shared' if IoU ≥ 0.5 to any v2,
    'weak_shared' if 0.25-0.5, 'new' otherwise."""
    best_i, best_u = best_match(v3, v2s, min_iou=0.0)
    if best_u >= match_iou:
        return {'fate': 'shared', 'best_iou': best_u,
                'v2_len': v2s[best_i][1] if best_i is not None else None}
    if best_u >= match_iou_low:
        return {'fate': 'weak_shared', 'best_iou': best_u,
                'v2_len': v2s[best_i][1] if best_i is not None else None}
    return {'fate': 'new', 'best_iou': best_u, 'v2_len': None}


def size_bucket(length):
    if length < 30: return '<30'
    if length < 60: return '30-60'
    if length < 90: return '60-90'
    if length < 150: return '90-150'
    if length < 250: return '150-250'
    if length < 500: return '250-500'
    return '>500'


SIZE_BUCKETS = ['<30', '30-60', '60-90', '90-150', '150-250',
                  '250-500', '>500']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', action='append', required=True,
                    help='Annotated v3 BAM (repeatable)')
    ap.add_argument('--label', required=True,
                    help='Dataset label for output files + plot titles')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-reads', type=int, default=0,
                    help='cap reads per BAM (0 = all)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Per-call rows
    call_rows = []
    # Summary counters
    fate_counts = Counter()
    v3_fate_counts = Counter()
    fate_by_v2_size = defaultdict(Counter)  # size_bucket -> fate -> count
    matched_pairs = []  # (v2_len, v3_len, iou, v3_kind)
    v3_iou_by_kind_size = {'nuc': defaultdict(list), 'tf': defaultdict(list)}
    # v3 quality vs match
    nuc_qual_match = []  # (nq, matched_bool)
    tf_qual_match = []

    per_read_rows = []

    n_reads = 0
    for bam in args.in_bam:
        print(f'[{bam}]', flush=True)
        for read in iter_reads(bam):
            n_reads += 1
            if args.max_reads and n_reads > args.max_reads:
                break
            v3_nucs = read['nuc']  # [(s, l, q)]
            v3_tfs = read['tf']
            v2s = read['fp_v2']    # [(s, l)]

            # Per-v2 classification
            for v2 in v2s:
                cls = classify_v2(v2, v3_nucs, v3_tfs)
                fate_counts[cls['fate']] += 1
                sb = size_bucket(v2[1])
                fate_by_v2_size[sb][cls['fate']] += 1
                if cls['fate'] in ('preserved_nuc', 'preserved_tf',
                                      'reclassified'):
                    matched_pairs.append(
                        (v2[1], cls.get('v3_len', 0), cls['best_iou'],
                         cls.get('best_kind', '?')))
                call_rows.append({
                    'kind': 'v2',
                    'read': read['name'],
                    'chrom': read['chrom'],
                    'start': v2[0],
                    'length': v2[1],
                    'fate': cls['fate'],
                    'best_iou': cls['best_iou'],
                    'best_v3_kind': cls.get('best_kind', ''),
                })

            # Per-v3-nuc classification
            for (s, l, q) in v3_nucs:
                cls = classify_v3((s, l), v2s)
                v3_fate_counts[('nuc', cls['fate'])] += 1
                v3_iou_by_kind_size['nuc'][size_bucket(l)].append(
                    cls['best_iou'])
                if q is not None:
                    nq = q[0] if len(q) > 0 else 0
                    nuc_qual_match.append(
                        (nq, cls['fate'] in ('shared', 'weak_shared')))
                call_rows.append({
                    'kind': 'v3_nuc',
                    'read': read['name'],
                    'chrom': read['chrom'],
                    'start': s, 'length': l,
                    'fate': cls['fate'],
                    'best_iou': cls['best_iou'],
                    'nq': q[0] if q else None,
                })
            for (s, l, q) in v3_tfs:
                cls = classify_v3((s, l), v2s)
                v3_fate_counts[('tf', cls['fate'])] += 1
                v3_iou_by_kind_size['tf'][size_bucket(l)].append(
                    cls['best_iou'])
                if q is not None:
                    tq = q[0] if len(q) > 0 else 0
                    tf_qual_match.append(
                        (tq, cls['fate'] in ('shared', 'weak_shared')))
                call_rows.append({
                    'kind': 'v3_tf',
                    'read': read['name'],
                    'chrom': read['chrom'],
                    'start': s, 'length': l,
                    'fate': cls['fate'],
                    'best_iou': cls['best_iou'],
                    'tq': q[0] if q else None,
                })

            per_read_rows.append({
                'read': read['name'],
                'n_v2': len(v2s),
                'n_v3_nuc': len(v3_nucs),
                'n_v3_tf': len(v3_tfs),
                'n_v3_msp': len(read['msp']),
            })

    print(f'\nTotal reads processed: {len(per_read_rows)}')
    print(f'\n=== v2 footprint fates ===')
    total_v2 = sum(fate_counts.values())
    for fate in ['preserved_nuc', 'preserved_tf', 'split',
                  'reclassified', 'weak_overlap', 'lost']:
        n = fate_counts.get(fate, 0)
        pct = 100 * n / max(1, total_v2)
        print(f'  {fate:<20s} {n:>10d}  ({pct:5.1f}%)')
    print(f'  ---\n  total v2 footprints: {total_v2}')

    print(f'\n=== v3 call fates ===')
    for kind in ['nuc', 'tf']:
        total = sum(v for (k, _), v in v3_fate_counts.items() if k == kind)
        print(f'  {kind}:')
        for fate in ['shared', 'weak_shared', 'new']:
            n = v3_fate_counts.get((kind, fate), 0)
            pct = 100 * n / max(1, total)
            print(f'    {fate:<15s} {n:>10d}  ({pct:5.1f}%)')

    # -------- Write TSVs --------
    summary_tsv = os.path.join(args.out_dir, f'{args.label}_summary.tsv')
    with open(summary_tsv, 'w') as f:
        f.write('category\tfate\tcount\n')
        for fate, n in fate_counts.items():
            f.write(f'v2\t{fate}\t{n}\n')
        for (kind, fate), n in v3_fate_counts.items():
            f.write(f'v3_{kind}\t{fate}\t{n}\n')
    print(f'\nWrote {summary_tsv}')

    calls_tsv = os.path.join(args.out_dir, f'{args.label}_calls.tsv')
    with open(calls_tsv, 'w') as f:
        f.write('kind\tread\tchrom\tstart\tlength\tfate\tbest_iou\tbest_v3_kind_or_nq_or_tq\n')
        for row in call_rows:
            extra = row.get('best_v3_kind') or (
                row.get('nq') if row.get('nq') is not None
                else (row.get('tq') if row.get('tq') is not None else ''))
            f.write(f"{row['kind']}\t{row['read']}\t{row['chrom']}\t"
                    f"{row['start']}\t{row['length']}\t{row['fate']}\t"
                    f"{row['best_iou']:.3f}\t{extra}\n")
    print(f'Wrote {calls_tsv}')

    # -------- Figures --------
    # 1. IoU distribution of matched pairs
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ious = [p[2] for p in matched_pairs]
    ax.hist(ious, bins=40, color='#1e3a8a', alpha=0.8)
    ax.axvline(0.5, color='#dc2626', linestyle='--', label='match threshold (0.5)')
    ax.axvline(0.25, color='#f59e0b', linestyle='--', label='loss threshold (0.25)')
    ax.set_xlabel('IoU (v2 footprint vs best v3 call)')
    ax.set_ylabel('count')
    ax.set_title(f'{args.label}: IoU distribution, matched v2–v3 pairs (n={len(ious)})')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(args.out_dir, f'{args.label}_iou_distribution.png')
    fig.savefig(png, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {png}')

    # 2. Size scatter
    fig, ax = plt.subplots(figsize=(8, 7))
    if matched_pairs:
        v2_lens = [p[0] for p in matched_pairs]
        v3_lens = [p[1] for p in matched_pairs]
        kinds = [p[3] for p in matched_pairs]
        colors = ['#1e3a8a' if k == 'nuc' else '#dc2626' for k in kinds]
        ax.scatter(v2_lens, v3_lens, c=colors, alpha=0.25, s=8)
        lims = [1, max(max(v2_lens), max(v3_lens)) * 1.1]
        ax.plot(lims, lims, 'k--', alpha=0.3, label='y=x')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('v2 footprint length (bp)')
    ax.set_ylabel('v3 call length (bp)')
    ax.set_title(f'{args.label}: v2 vs v3 size, matched pairs\n'
                  'blue = v3 nuc, red = v3 tf')
    ax.legend(); ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    png = os.path.join(args.out_dir, f'{args.label}_size_comparison.png')
    fig.savefig(png, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {png}')

    # 3. Classification breakdown by v2 size bucket
    fig, ax = plt.subplots(figsize=(11, 6))
    fate_order = ['preserved_nuc', 'preserved_tf', 'split',
                   'reclassified', 'weak_overlap', 'lost']
    fate_colors = {
        'preserved_nuc': '#1e3a8a',
        'preserved_tf':  '#2563eb',
        'split':         '#16a34a',
        'reclassified':  '#f59e0b',
        'weak_overlap':  '#94a3b8',
        'lost':          '#dc2626',
    }
    x = np.arange(len(SIZE_BUCKETS))
    bottoms = np.zeros(len(SIZE_BUCKETS))
    totals = np.array([sum(fate_by_v2_size[b].values()) for b in SIZE_BUCKETS],
                       dtype=float)
    totals_safe = np.where(totals > 0, totals, 1)
    for fate in fate_order:
        vals = np.array([100 * fate_by_v2_size[b].get(fate, 0) / t
                          for b, t in zip(SIZE_BUCKETS, totals_safe)])
        ax.bar(x, vals, bottom=bottoms,
                color=fate_colors[fate], label=fate, edgecolor='white',
                linewidth=0.5)
        bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels(SIZE_BUCKETS)
    ax.set_xlabel('v2 footprint length bucket (bp)')
    ax.set_ylabel('% of v2 footprints in bucket')
    ax.set_title(f'{args.label}: v2 footprint fate by size (n per bucket shown)')
    ax.legend(loc='upper right', fontsize=9)
    for i, t in enumerate(totals):
        ax.text(i, 102, f'{int(t):,}', ha='center', fontsize=8)
    ax.set_ylim(0, 115)
    fig.tight_layout()
    png = os.path.join(args.out_dir, f'{args.label}_classification_breakdown.png')
    fig.savefig(png, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {png}')

    # 4. v3 quality vs match probability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    def qual_match_plot(ax, pairs, kind, color):
        if not pairs:
            ax.text(0.5, 0.5, f'no {kind} calls with quality',
                     ha='center', va='center', transform=ax.transAxes)
            return
        q = np.array([p[0] for p in pairs])
        m = np.array([p[1] for p in pairs], dtype=float)
        # Bin by quality deciles
        edges = np.linspace(0, 256, 11)
        rates = []
        counts = []
        centers = []
        for i in range(len(edges) - 1):
            mask = (q >= edges[i]) & (q < edges[i+1])
            if mask.sum() > 0:
                rates.append(m[mask].mean())
                counts.append(mask.sum())
                centers.append((edges[i] + edges[i+1]) / 2)
        ax.bar(centers, rates, width=20, color=color, alpha=0.8)
        ax.set_xlabel(f'v3 {kind} quality (0-255)')
        ax.set_ylabel(f'fraction shared with v2 (IoU ≥ 0.25)')
        ax.set_title(f'{kind} match rate vs quality (n={len(pairs)})')
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
        # annotate count per bin
        for c, r, n in zip(centers, rates, counts):
            ax.text(c, r + 0.02, f'{n}', ha='center', fontsize=7)
    qual_match_plot(ax1, nuc_qual_match, 'nuc', '#1e3a8a')
    qual_match_plot(ax2, tf_qual_match, 'tf', '#dc2626')
    fig.suptitle(f'{args.label}: v3 quality vs v2 presence', fontsize=11)
    fig.tight_layout()
    png = os.path.join(args.out_dir, f'{args.label}_quality_vs_match.png')
    fig.savefig(png, dpi=130, bbox_inches='tight'); plt.close(fig)
    print(f'Wrote {png}')

    # 5. Summary JSON
    summary = {
        'label': args.label,
        'n_reads': len(per_read_rows),
        'n_v2_footprints': total_v2,
        'v2_fates': dict(fate_counts),
        'v3_fates': {f'{k}_{f}': v for (k, f), v in v3_fate_counts.items()},
        'per_read_means': {
            'v2': float(np.mean([r['n_v2'] for r in per_read_rows])) if per_read_rows else 0,
            'v3_nuc': float(np.mean([r['n_v3_nuc'] for r in per_read_rows])) if per_read_rows else 0,
            'v3_tf': float(np.mean([r['n_v3_tf'] for r in per_read_rows])) if per_read_rows else 0,
            'v3_msp': float(np.mean([r['n_v3_msp'] for r in per_read_rows])) if per_read_rows else 0,
        },
    }
    jsn = os.path.join(args.out_dir, f'{args.label}_summary.json')
    with open(jsn, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {jsn}')


if __name__ == '__main__':
    main()
