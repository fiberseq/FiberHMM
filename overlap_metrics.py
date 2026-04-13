#!/usr/bin/env python3
"""Quantify overlap between v2 short-nuc calls (ns/nl with nl < cutoff)
and tf_recaller calls (tn/tl/ts).

Outputs:
  - PNG with 6 panels
  - TSV with numeric summary
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-bam', required=True,
                    help='Recaller output BAM (has both v2 ns/nl + new tn/tl/ts)')
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--long-nuc-min', type=int, default=90,
                    help='v2 ns/nl entries with nl < this are treated as v2 TFs')
    ap.add_argument('--min-ts', type=int, default=0,
                    help='Pre-filter recaller TFs below this ts (LLR * 5)')
    ap.add_argument('--max-reads', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    # Collections
    v2_lengths = []
    rc_lengths = []
    # Per-matched-pair (v2 -> best recaller overlap)
    jaccard_matched = []
    length_pairs = []       # (v2_len, rc_len) for matched pairs
    ts_matched = []         # ts scores of matched recaller calls
    ts_unmatched = []       # ts scores of recaller calls with no overlapping v2 short nuc
    v2_unmatched_lens = []  # v2 short nucs with no overlapping recaller call
    # Per-read coverage vectors (pooled by hashing absolute ref positions into bins)
    v2_cov_samples = defaultdict(int)
    rc_cov_samples = defaultdict(int)
    BIN = 10  # 10 bp bins for genome-wide correlation

    n_reads = 0
    n_v2_short = 0
    n_rc_total = 0
    n_rc_pre_filter = 0

    bam = pysam.AlignmentFile(args.in_bam, 'rb', check_sq=False)
    for r in bam:
        if args.max_reads and n_reads >= args.max_reads:
            break
        ns = get_arr(r, 'ns')
        nl = get_arr(r, 'nl')
        if not ns:
            continue
        tn = get_arr(r, 'tn')
        tl = get_arr(r, 'tl')
        ts = get_arr(r, 'ts') if r.has_tag('ts') else [255] * len(tn)

        # Pre-filter recaller calls on ts
        rc_calls_all = list(zip(tn, tl, ts))
        n_rc_pre_filter += len(rc_calls_all)
        rc_calls = [(s, l, t) for s, l, t in rc_calls_all if t >= args.min_ts]

        v2_short = [(int(s), int(l)) for s, l in zip(ns, nl)
                    if 0 < int(l) < args.long_nuc_min]
        n_v2_short += len(v2_short)
        n_rc_total += len(rc_calls)
        n_reads += 1

        v2_lengths.extend(l for _, l in v2_short)
        rc_lengths.extend(l for _, l, _ in rc_calls)

        # Map query positions to ref via aligned_pairs (for genome-wide coverage)
        try:
            qr_pairs = r.get_aligned_pairs(matches_only=True)
            qr_map = {qp: rp for qp, rp in qr_pairs}
        except ValueError:
            qr_map = {}

        def project(q_s, q_l):
            for d in range(6):
                rp_s = qr_map.get(q_s + d) or qr_map.get(q_s - d)
                if rp_s is not None: break
            else: return None
            for d in range(6):
                rp_e = qr_map.get(q_s + q_l + d) or qr_map.get(q_s + q_l - d)
                if rp_e is not None: break
            else: return None
            if rp_e < rp_s: rp_s, rp_e = rp_e, rp_s
            return rp_s, rp_e

        # Fill coverage bins
        for s, l in v2_short:
            pr = project(s, l)
            if pr is None: continue
            rs, re = pr
            for b in range(rs // BIN, (re + BIN) // BIN):
                v2_cov_samples[b] += 1
        for s, l, _ in rc_calls:
            pr = project(s, l)
            if pr is None: continue
            rs, re = pr
            for b in range(rs // BIN, (re + BIN) // BIN):
                rc_cov_samples[b] += 1

        # Per-v2-short match to best overlapping recaller call
        for vs, vl in v2_short:
            ve = vs + vl
            best_iou = 0.0
            best_rc = None
            for rs, rl, rt in rc_calls:
                re = rs + rl
                ov = interval_overlap(vs, ve, rs, re)
                if ov <= 0: continue
                union = (ve - vs) + (re - rs) - ov
                iou = ov / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_rc = (rs, rl, rt)
            if best_rc is not None:
                jaccard_matched.append(best_iou)
                length_pairs.append((vl, best_rc[1]))
                ts_matched.append(best_rc[2])
            else:
                v2_unmatched_lens.append(vl)

        # Per-recaller match to any v2 short nuc (precision)
        for rs, rl, rt in rc_calls:
            re = rs + rl
            hit = False
            for vs, vl in v2_short:
                if interval_overlap(rs, re, vs, vs + vl) > 0:
                    hit = True
                    break
            if not hit:
                ts_unmatched.append(rt)
    bam.close()

    # Recall = fraction of v2 short nucs with any overlapping recaller call
    recall = len(jaccard_matched) / max(1, n_v2_short)
    # Precision = fraction of recaller calls overlapping a v2 short nuc
    n_rc_matched_v2 = n_rc_total - len(ts_unmatched)
    precision = n_rc_matched_v2 / max(1, n_rc_total)

    # Per-bp coverage correlation
    all_bins = set(v2_cov_samples) | set(rc_cov_samples)
    v2_vec = np.array([v2_cov_samples.get(b, 0) for b in all_bins])
    rc_vec = np.array([rc_cov_samples.get(b, 0) for b in all_bins])
    if len(v2_vec) > 1 and v2_vec.std() > 0 and rc_vec.std() > 0:
        pearson = float(np.corrcoef(v2_vec, rc_vec)[0, 1])
    else:
        pearson = float('nan')
    # Spearman: rank correlate
    def _rank(x):
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x))
        return ranks
    if len(v2_vec) > 1:
        spearman = float(np.corrcoef(_rank(v2_vec), _rank(rc_vec))[0, 1])
    else:
        spearman = float('nan')

    # ---- Plot ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Length distributions overlaid
    ax = axes[0, 0]
    bins = np.arange(0, 121, 5)
    ax.hist(v2_lengths, bins=bins, alpha=0.55, label=f'v2 short (n={len(v2_lengths)})',
            color='#dc2626', density=True)
    ax.hist(rc_lengths, bins=bins, alpha=0.55, label=f'recaller (n={len(rc_lengths)})',
            color='#1e3a8a', density=True)
    ax.axvline(args.long_nuc_min, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('footprint length (bp)'); ax.set_ylabel('density')
    ax.set_title('Length distribution'); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 2. Jaccard / IoU distribution
    ax = axes[0, 1]
    ax.hist(jaccard_matched, bins=30, color='#1e3a8a', alpha=0.75)
    ax.axvline(np.median(jaccard_matched) if jaccard_matched else 0,
               color='red', linestyle='--', linewidth=1,
               label=f'median={np.median(jaccard_matched):.2f}' if jaccard_matched else 'no data')
    ax.set_xlabel('IoU (matched pair)'); ax.set_ylabel('count')
    ax.set_title(f'Per-pair IoU  (n_matched={len(jaccard_matched)})')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 3. Length vs length scatter
    ax = axes[0, 2]
    if length_pairs:
        v2l, rcl = zip(*length_pairs)
        # subsample if huge
        idx = np.random.choice(len(v2l), min(20000, len(v2l)), replace=False)
        v2l = np.array(v2l)[idx]; rcl = np.array(rcl)[idx]
        ax.scatter(v2l, rcl, s=2, alpha=0.15, color='#1e3a8a')
        maxl = max(v2l.max(), rcl.max(), 90)
        ax.plot([0, maxl], [0, maxl], 'r--', linewidth=0.7, alpha=0.7, label='y=x')
        r = float(np.corrcoef(v2l, rcl)[0, 1])
        ax.set_xlim(0, maxl); ax.set_ylim(0, maxl)
        ax.set_title(f'v2 length vs recaller length  (Pearson r={r:.3f})')
    else:
        ax.text(0.5, 0.5, 'no matched pairs', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('v2 nl (bp)'); ax.set_ylabel('matched recaller tl (bp)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 4. ts distributions matched vs unmatched
    ax = axes[1, 0]
    bins2 = np.arange(0, 261, 10)
    ax.hist(ts_matched, bins=bins2, alpha=0.55,
            label=f'matched v2 (n={len(ts_matched)})', color='#1e3a8a', density=True)
    ax.hist(ts_unmatched, bins=bins2, alpha=0.55,
            label=f'no v2 overlap (n={len(ts_unmatched)})', color='#9ca3af', density=True)
    ax.set_xlabel('recaller ts (LLR*5, 0-255)'); ax.set_ylabel('density')
    ax.set_title('Recaller ts: calls with v2 support vs without')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # 5. Per-bp coverage correlation (10 bp bins)
    ax = axes[1, 1]
    if len(v2_vec) > 0:
        ax.scatter(v2_vec, rc_vec, s=2, alpha=0.15, color='#1e3a8a')
        mx = max(v2_vec.max(), rc_vec.max(), 1)
        ax.plot([0, mx], [0, mx], 'r--', linewidth=0.7, alpha=0.7, label='y=x')
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.set_xlabel(f'v2 coverage (per 10 bp bin)')
        ax.set_ylabel(f'recaller coverage')
        ax.set_title(f'Per-bin coverage (Pearson={pearson:.3f}, Spearman={spearman:.3f})')
        ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 6. Recall / Precision headline panel
    ax = axes[1, 2]
    ax.axis('off')
    lines = [
        f'n_reads: {n_reads}',
        f'v2 short-nucs (nl < {args.long_nuc_min}): {n_v2_short}',
        f'recaller TFs emitted: {n_rc_pre_filter}',
        f'  after ts >= {args.min_ts} filter: {n_rc_total}',
        '',
        f'RECALL  (v2 -> recaller): {recall:.1%}',
        f'  v2 short nucs matched: {len(jaccard_matched)}',
        f'  v2 short nucs missed: {len(v2_unmatched_lens)}',
        '',
        f'PRECISION (recaller -> v2): {precision:.1%}',
        f'  recaller calls matched to a v2 short nuc: {n_rc_matched_v2}',
        f'  recaller calls with no v2 overlap: {len(ts_unmatched)}',
        '',
        f'IoU median:  {np.median(jaccard_matched):.2f}' if jaccard_matched else 'IoU: n/a',
        f'Length r:    {r:.3f}' if length_pairs else 'Length r: n/a',
        f'Coverage r:  {pearson:.3f}',
    ]
    ax.text(0.03, 0.97, '\n'.join(lines), fontsize=11, ha='left', va='top',
            family='monospace', transform=ax.transAxes)

    fig.suptitle(f'Recaller vs v2 short-nuc overlap (ts filter >= {args.min_ts})',
                 fontsize=12, y=0.995)
    fig.tight_layout()
    png = args.out_prefix + '_overlap.png'
    fig.savefig(png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {png}')

    # TSV summary
    tsv = args.out_prefix + '_overlap_summary.tsv'
    with open(tsv, 'w') as f:
        f.write('metric\tvalue\n')
        f.write(f'n_reads\t{n_reads}\n')
        f.write(f'n_v2_short\t{n_v2_short}\n')
        f.write(f'n_recaller_total\t{n_rc_total}\n')
        f.write(f'recall\t{recall:.4f}\n')
        f.write(f'precision\t{precision:.4f}\n')
        f.write(f'iou_median\t{np.median(jaccard_matched) if jaccard_matched else "nan"}\n')
        f.write(f'length_pearson\t{r if length_pairs else "nan"}\n')
        f.write(f'coverage_pearson\t{pearson}\n')
        f.write(f'coverage_spearman\t{spearman}\n')
    print(f'Wrote {tsv}')


if __name__ == '__main__':
    main()
