"""Diagnostic for the iter-16 overcall TF caller.

Reads the emitted BAM (after caller_v8 runs) and plots the
distribution of the three TF quality scores so we can sanity-check
the scaling and confirm a filter knob like `tq >= 128 & el,er >= 128`
is the right "best guess" for downstream users.

Usage:
    python bench/diagnose_tf_overcall.py \
        --bam /tmp/napa_v8_iter16_smoke.bam \
        --label NAPA

Outputs:
    bench/output/tf_overcall_<LABEL>.png  (6-panel figure)
    stdout: summary stats + filter-knob table
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pysam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))


def collect_tf_calls(bam_path):
    """Return per-read TF counts + flat arrays of (length, tq, el, er)."""
    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    tn_counts = []
    lengths = []
    tqs = []
    els = []
    ers = []
    n_reads = 0
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        n_reads += 1
        try:
            tn = list(r.get_tag('tn'))
        except KeyError:
            tn_counts.append(0)
            continue
        tn_counts.append(len(tn))
        lengths.extend(r.get_tag('tl'))
        tqs.extend(r.get_tag('tq'))
        els.extend(r.get_tag('el'))
        ers.extend(r.get_tag('er'))
    bam.close()
    return (n_reads,
            np.asarray(tn_counts, dtype=np.int32),
            np.asarray(lengths, dtype=np.int32),
            np.asarray(tqs, dtype=np.int32),
            np.asarray(els, dtype=np.int32),
            np.asarray(ers, dtype=np.int32))


def print_summary(label, n_reads, counts, lengths, tqs, els, ers):
    print(f'\n== {label} ==')
    print(f'Reads: {n_reads}')
    print(f'Total TF calls: {len(lengths)}')
    print(f'TFs per read:    mean={counts.mean():.1f} '
          f'median={int(np.median(counts))} '
          f'p95={int(np.percentile(counts, 95))} '
          f'max={counts.max()}')
    print(f'Length (bp):     median={int(np.median(lengths))} '
          f'p75={int(np.percentile(lengths, 75))} '
          f'p95={int(np.percentile(lengths, 95))} '
          f'max={int(lengths.max())}')
    print(f'tq (significance): median={int(np.median(tqs))} '
          f'p75={int(np.percentile(tqs, 75))} '
          f'p95={int(np.percentile(tqs, 95))}')
    print(f'el (left edge):    median={int(np.median(els))} '
          f'p5={int(np.percentile(els, 5))}')
    print(f'er (right edge):   median={int(np.median(ers))} '
          f'p5={int(np.percentile(ers, 5))}')

    print('\nFilter knobs (strict AND of all three thresholds):')
    print(f'{"tq>=":>6} {"&el,er>=":>10} {"kept":>8} {"%":>6} {"/read":>7} '
          f'{"med_len":>9}')
    for tq_cut in (0, 64, 128, 192):
        for edge_cut in (0, 64, 128):
            mask = (tqs >= tq_cut) & (els >= edge_cut) & (ers >= edge_cut)
            kept = int(mask.sum())
            if kept == 0:
                med = 0
            else:
                med = int(np.median(lengths[mask]))
            pct = 100 * kept / max(1, len(tqs))
            print(f'{tq_cut:>6} {edge_cut:>10} {kept:>8} {pct:>5.1f}% '
                  f'{kept/max(1,n_reads):>7.2f} {med:>9}')


def plot_one(bam_path, label, out_dir):
    n_reads, counts, lengths, tqs, els, ers = collect_tf_calls(bam_path)
    if len(lengths) == 0:
        print(f'{label}: no TFs found')
        return
    print_summary(label, n_reads, counts, lengths, tqs, els, ers)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # A. tq histogram
    ax = axes[0, 0]
    ax.hist(tqs, bins=np.arange(0, 260, 8), color='#be185d', alpha=0.85)
    for cut, c, lab in [(64, '#facc15', 'tq=64'),
                         (128, '#22c55e', 'tq=128'),
                         (192, '#3b82f6', 'tq=192')]:
        ax.axvline(cut, color=c, ls='--', lw=1, label=lab)
    ax.set_xlabel('tq (significance)')
    ax.set_ylabel('Count')
    ax.set_title('A. tq distribution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # B. length histogram
    ax = axes[0, 1]
    ax.hist(lengths[lengths <= 120], bins=np.arange(0, 121, 3),
            color='#7c3aed', alpha=0.85)
    ax.set_xlabel('Length (bp)')
    ax.set_ylabel('Count')
    ax.set_title(f'B. TF length  (n={len(lengths)})')
    ax.grid(alpha=0.2)

    # C. edge quality scatter
    ax = axes[0, 2]
    sample = np.random.choice(len(els), size=min(4000, len(els)),
                                replace=False)
    sc = ax.scatter(els[sample], ers[sample], c=tqs[sample],
                    s=6, alpha=0.5, cmap='viridis', vmin=0, vmax=255)
    ax.axvline(128, color='#22c55e', ls='--', lw=0.6)
    ax.axhline(128, color='#22c55e', ls='--', lw=0.6)
    ax.set_xlabel('el (left edge q)')
    ax.set_ylabel('er (right edge q)')
    ax.set_title('C. el vs er  (color = tq)')
    plt.colorbar(sc, ax=ax, fraction=0.04)
    ax.grid(alpha=0.2)

    # D. tq vs length
    ax = axes[1, 0]
    ax.scatter(lengths[sample], tqs[sample], s=6, alpha=0.35,
                color='#be185d')
    ax.axhline(128, color='#22c55e', ls='--', lw=0.6)
    ax.set_xlabel('Length (bp)')
    ax.set_ylabel('tq')
    ax.set_xlim(0, min(100, lengths.max()))
    ax.set_title('D. length vs tq  (bp vs significance)')
    ax.grid(alpha=0.2)

    # E. Filter cascade: how many TFs survive each cut
    ax = axes[1, 1]
    cuts = np.arange(0, 256, 16)
    kept_tq = [(tqs >= c).sum() for c in cuts]
    kept_both = [((tqs >= c) & (els >= c) & (ers >= c)).sum() for c in cuts]
    kept_edges_only = [((els >= c) & (ers >= c)).sum() for c in cuts]
    ax.plot(cuts, np.array(kept_tq)/n_reads,
            color='#be185d', lw=2, label='tq>=k only')
    ax.plot(cuts, np.array(kept_edges_only)/n_reads,
            color='#3b82f6', lw=2, label='el,er>=k only')
    ax.plot(cuts, np.array(kept_both)/n_reads,
            color='#15803d', lw=2.5, label='ALL three >=k')
    ax.set_xlabel('Threshold k (applied to tq & edges)')
    ax.set_ylabel('TFs per read after filter')
    ax.set_title('E. Filter cascade')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # F. "best guess" length histogram after recommended filter
    ax = axes[1, 2]
    rec_mask = (tqs >= 128) & (els >= 128) & (ers >= 128)
    rec_lengths = lengths[rec_mask]
    if len(rec_lengths):
        ax.hist(rec_lengths[rec_lengths <= 120],
                bins=np.arange(0, 121, 3),
                color='#15803d', alpha=0.85)
    ax.set_xlabel('Length (bp)')
    ax.set_ylabel('Count')
    ax.set_title(f'F. Recommended (tq,el,er>=128) '
                  f'n={rec_mask.sum()} ({rec_mask.sum()/n_reads:.1f}/read)')
    ax.grid(alpha=0.2)

    fig.suptitle(f'{label} — iter-16 overcall TF caller diagnostic  '
                  f'({n_reads} reads, {len(lengths)} TF candidates)',
                  fontsize=12)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f'tf_overcall_{label}.png')
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_png}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bam', required=True, help='caller_v8 output BAM')
    ap.add_argument('--label', required=True, help='dataset label for figure')
    args = ap.parse_args()
    out_dir = os.path.join(HERE, 'output')
    plot_one(args.bam, args.label, out_dir)


if __name__ == '__main__':
    main()
