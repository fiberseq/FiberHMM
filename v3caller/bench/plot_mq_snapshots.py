"""Per-read snapshots colored by merge quality (mq).

Shows v8 default and v8_gapcdf side-by-side, 12 reads per dataset,
for all 10 DddA datasets. Each nucleosome rectangle is colored on
a gradient:

  - mq = 255 (pure Pass-1 atom, no merge)      → solid dark green
  - mq ~ 100 (moderate-confidence merge)        → medium magenta
  - mq ~ 0 (tail-of-Poisson merge, borderline)  → light pink

This visually answers the question "which of my v8 calls came from
the merge step, and how confident was each merge?". Users can post-
filter BAM calls by `mq` threshold without re-running the caller.

Reads MA tags via pure-Python parser (no dependency on the Rust
molecular_annotation package).
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
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ma_tags import parse_ma_tag, parse_aq_array  # noqa: E402


# mq colormap: dark green for pure Pass-1 (255), medium magenta for
# borderline merge (~100), light pink for tail merge (~0)
MQ_CMAP = LinearSegmentedColormap.from_list(
    'mq', [
        (0.0,  '#fecaca'),  # very light pink (mq=0, weak merge)
        (0.3,  '#f9a8d4'),  # light pink
        (0.6,  '#c026d3'),  # magenta
        (0.95, '#166534'),  # dark green (mq near 255)
        (1.0,  '#14532d'),  # deepest green (pure pass-1)
    ]
)


def mq_to_color(mq_val: int) -> tuple:
    """Normalize a 0-255 mq value to an RGBA tuple from MQ_CMAP."""
    return MQ_CMAP(max(0, min(255, int(mq_val))) / 255.0)


def read_calls_with_mq(bam_path, read_ids):
    """For each requested read_id, return (query_seq, nuc_intervals,
    mq_values, msp_intervals). Parses MA/AQ if present, falls back
    to ns/nl/nq/as/al + mq-via-pure-nq (255) otherwise."""
    out = {}
    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        if r.query_name not in read_ids:
            continue
        q = r.query_sequence or ''
        if r.has_tag('MA'):
            ma_str = r.get_tag('MA')
            parsed = parse_ma_tag(ma_str)
            nucs = parsed['nuc']
            msps = parsed['msp']
            if r.has_tag('AQ'):
                aq = list(r.get_tag('AQ'))
                # Build qual spec list from raw_types in order
                qual_specs = [rt[2] for rt in parsed['raw_types']]
                n_per_type = [len(rt[3]) for rt in parsed['raw_types']]
                per_ann = parse_aq_array(aq, qual_specs, n_per_type)
                # Split per_ann back by type
                mqs = []
                idx = 0
                for rt in parsed['raw_types']:
                    name = rt[0]
                    count = len(rt[3])
                    spec = rt[2]
                    if name == 'nuc' and len(spec) >= 2 and spec[:2] == 'QQ':
                        # Layout is (nq, mq, ...). Second quality is mq.
                        for k in range(count):
                            mqs.append(per_ann[idx + k][1])
                    idx += count
            else:
                mqs = [255] * len(nucs)
        else:
            # Legacy path
            try:
                ns = list(r.get_tag('ns'))
                nl = list(r.get_tag('nl'))
            except KeyError:
                ns, nl = [], []
            try:
                as_ = list(r.get_tag('as'))
                al = list(r.get_tag('al'))
            except KeyError:
                as_, al = [], []
            nucs = list(zip(ns, nl))
            msps = list(zip(as_, al))
            mqs = [255] * len(nucs)  # legacy = unknown = assume pure
        out[r.query_name] = {
            'q': q,
            'nucs': nucs,  # [(start, length), ...]
            'mqs': mqs,
            'msps': msps,
        }
    bam.close()
    return out


def pick_reads(bam_dir, dataset, n_reads=12):
    """Unbiased picker: evenly sample n_reads >= 2 kb from whichever
    v8 BAM is present. The read set is identical across callers
    because all output BAMs stream from the same input."""
    path = os.path.join(bam_dir, f'{dataset}__v8.bam')
    if not os.path.exists(path):
        path = os.path.join(bam_dir, f'{dataset}__protected_runs+merge.bam')
    bam = pysam.AlignmentFile(path, 'rb', check_sq=False)
    long_reads = []
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        if (r.query_length or 0) < 2000:
            continue
        long_reads.append(r.query_name)
    bam.close()
    if not long_reads:
        return []
    if len(long_reads) <= n_reads:
        return long_reads
    idx = [int(i) for i in np.linspace(0, len(long_reads) - 1, n_reads)]
    return [long_reads[i] for i in idx]


def draw_read(ax, tags, y, hit_color='#dc2626'):
    q = tags['q']
    qlen = len(q)
    ax.plot([0, qlen], [y, y], color='#94a3b8', lw=0.5)
    # Hit ticks
    hits = [i for i, c in enumerate(q) if c in 'YyRr']
    if hits:
        ax.vlines(hits, y - 0.14, y + 0.14, color=hit_color, lw=0.4)
    # Nucleosomes, colored by mq
    for (s, l), mq in zip(tags['nucs'], tags['mqs']):
        color = mq_to_color(mq)
        ax.add_patch(plt.Rectangle(
            (s, y - 0.33), l, 0.66,
            facecolor=color, edgecolor='none', alpha=0.92,
        ))


def plot_dataset(dataset, bam_dir, out_dir, n_reads=12):
    picks = pick_reads(bam_dir, dataset, n_reads=n_reads)
    if not picks:
        return False
    pick_set = set(picks)

    callers = [
        ('v7 stock',   f'{dataset}__protected_runs+merge.bam'),
        ('v8 default', f'{dataset}__v8.bam'),
        ('v8 gap_cdf', f'{dataset}__v8_gapcdf.bam'),
    ]
    data = {}
    for label, fname in callers:
        p = os.path.join(bam_dir, fname)
        if not os.path.exists(p):
            data[label] = {}
            continue
        data[label] = read_calls_with_mq(p, pick_set)

    # Figure: 1 row, 3 columns. One per caller.
    max_qlen = max(
        (len(t['q'])
         for caller_data in data.values()
         for t in caller_data.values()),
        default=0,
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.35 * n_reads + 1.2),
                              sharex=True, sharey=True)
    for col, (label, _) in enumerate(callers):
        ax = axes[col]
        tags = data[label]
        for i, rid in enumerate(picks):
            t = tags.get(rid)
            if t:
                draw_read(ax, t, i)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(-0.8, len(picks) - 0.2)
        ax.set_yticks([])
        ax.set_xlim(0, max_qlen)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Read position (bp)', fontsize=8)

    # Shared colorbar for mq
    sm = plt.cm.ScalarMappable(cmap=MQ_CMAP,
                                 norm=plt.Normalize(vmin=0, vmax=255))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.7, pad=0.02,
                         orientation='vertical')
    cbar.set_label('Merge quality (mq)  —  255 = pure Pass-1, 0 = tail merge',
                    fontsize=9)

    fig.suptitle(f'{dataset}  —  {n_reads} reads  (green = confident, pink = borderline merge)',
                  fontsize=12)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'mq_snapshot_{dataset}.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'output'))
    ap.add_argument('--datasets', default=None)
    ap.add_argument('--n-reads', type=int, default=12)
    args = ap.parse_args()
    bam_dir = os.path.join(args.out, 'bam')
    out_dir = os.path.join(args.out, 'mq_snapshots')

    if args.datasets:
        datasets = args.datasets.split(',')
    else:
        datasets = [
            'scDAF_PS00758', 'NAPA_PS00626', 'UBA1_PS00685', 'ENH30',
            'GLI2', 'PS01498', 'PS01530',
            'ftz_22', '4RZV9P_6_eve_GA', '4RZV9P_2_sna',
        ]
    for ds in datasets:
        try:
            plot_dataset(ds, bam_dir, out_dir, n_reads=args.n_reads)
        except Exception as e:
            print(f'  error on {ds}: {e}')


if __name__ == '__main__':
    main()
