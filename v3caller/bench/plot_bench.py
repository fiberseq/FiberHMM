"""Generate the benchmark comparison figure from bench outputs.

Reads:
    output/summary.tsv
    output/nl_dump.json
    output/bam/*.bam    (for per-read size details on the winners)

Writes:
    output/figure_main.png          (multi-panel caller comparison)
    output/figure_size_grid.png     (caller × dataset size grid)
    output/figure_merge_effect.png  (first pass alone vs + merge)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


CALLER_ORDER = [
    'gap_cdf',
    'xdrop',
    'hmm2',
    'core_peaks',
    'protected_runs',
    'profile_guided',
]

# v8 callers: protected_runs+v8_merge (default) and gap_cdf+v8_merge
# (the iteration-4 winner). Both get their own rows alongside the
# +merge variants where relevant.
V8_ORDER = ['v8', 'v8_gapcdf']

DATASET_ORDER = [
    'scDAF_PS00758',
    'NAPA_PS00626',
    'UBA1_PS00685',
    'ENH30',
    'GLI2',
    'PS01498',
    'PS01530',
    'ftz_22',
    '4RZV9P_6_eve_GA',
    '4RZV9P_2_sna',
]

# Rough categorical color map
CALLER_COLORS = {
    'gap_cdf':        '#8b5cf6',
    'xdrop':          '#f59e0b',
    'hmm2':           '#10b981',
    'core_peaks':     '#3b82f6',
    'protected_runs': '#dc2626',
    'profile_guided': '#0891b2',
    'v8':             '#be185d',  # magenta for v8 (protected_runs+v8_merge)
    'v8_gapcdf':      '#7c2d12',  # dark brown for v8_gapcdf
}


def load_summary(out_dir):
    path = os.path.join(out_dir, 'summary.tsv')
    rows = []
    with open(path) as fh:
        r = csv.DictReader(fh, delimiter='\t')
        for row in r:
            rows.append(row)
    return rows


def load_nl_dump(out_dir):
    with open(os.path.join(out_dir, 'nl_dump.json')) as fh:
        return json.load(fh)


def _get_rows(rows, caller, dataset):
    for r in rows:
        if r['caller'] == caller and r['dataset'] == dataset:
            return r
    return None


def plot_size_grid(out_dir, rows, nl_dump, variant='+merge'):
    """Grid: rows = datasets, cols = callers. Each cell = size histogram."""
    datasets = [d for d in DATASET_ORDER
                if any(r['dataset'] == d for r in rows)]
    callers = [f'{c}{variant}' if variant else c for c in CALLER_ORDER]
    if variant == '+merge':
        # Include both v8 variants as extra columns at the right
        # when showing merged variants. v8_gapcdf is the new top
        # recommendation; v8 is the protected_runs-based default.
        callers = callers + V8_ORDER
    nrows = len(datasets)
    ncols = len(callers)
    if nrows == 0 or ncols == 0:
        return
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(2.2 * ncols, 1.7 * nrows),
                              sharex=True, sharey=False)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes[:, None]
    for i, ds in enumerate(datasets):
        for j, caller in enumerate(callers):
            ax = axes[i, j]
            nl = nl_dump.get(caller, {}).get(ds, [])
            if nl:
                nl_a = np.asarray(nl, dtype=np.int32)
                nl_c = nl_a[(nl_a > 0) & (nl_a <= 800)]
                ax.hist(nl_c, bins=np.arange(0, 801, 15),
                        color=CALLER_COLORS.get(caller.replace('+merge', ''),
                                                 '#777'),
                        alpha=0.8)
                for pos in (147, 300, 450):
                    ax.axvline(pos, color='red', ls='--', lw=0.4, alpha=0.5)
                # Show median of the full (uncapped) distribution — the
                # capped histogram is purely a display choice.
                full_med = int(np.median(nl_a[nl_a > 0])) if (nl_a > 0).any() else 0
                ax.text(0.97, 0.95, f'med={full_med}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=7, bbox=dict(facecolor='white',
                                               alpha=0.8, edgecolor='none'))
            ax.set_xlim(0, 800)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(caller, fontsize=8)
            if j == 0:
                ax.set_ylabel(ds, fontsize=8)
    fig.suptitle(f'Footprint size distributions  (variant: {variant or "standalone"})',
                 fontsize=12, y=1.00)
    fig.tight_layout()
    out = os.path.join(out_dir,
                        f'figure_size_grid_{"standalone" if not variant else "merged"}.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def plot_merge_effect(out_dir, rows):
    """Per-caller bar chart: before vs after +merge. Each caller has two
    bars (alone, +merge) showing median footprint size, averaged across
    datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # Average over datasets
    per_caller_alone = {}
    per_caller_merge = {}
    per_caller_alone_nucs = {}
    per_caller_merge_nucs = {}
    per_caller_alone_pct = {}
    per_caller_merge_pct = {}
    for c in CALLER_ORDER:
        a_meds = [float(r['median_footprint']) for r in rows
                  if r['caller'] == c and r['dataset'] in DATASET_ORDER]
        m_meds = [float(r['median_footprint']) for r in rows
                  if r['caller'] == f'{c}+merge' and r['dataset'] in DATASET_ORDER]
        per_caller_alone[c] = np.mean(a_meds) if a_meds else 0
        per_caller_merge[c] = np.mean(m_meds) if m_meds else 0
        per_caller_alone_nucs[c] = np.mean([float(r['mean_nucs_per_read']) for r in rows
                                              if r['caller'] == c and r['dataset'] in DATASET_ORDER] or [0])
        per_caller_merge_nucs[c] = np.mean([float(r['mean_nucs_per_read']) for r in rows
                                              if r['caller'] == f'{c}+merge' and r['dataset'] in DATASET_ORDER] or [0])
        per_caller_alone_pct[c] = np.mean([float(r['pct_mono_100_220']) for r in rows
                                              if r['caller'] == c and r['dataset'] in DATASET_ORDER] or [0])
        per_caller_merge_pct[c] = np.mean([float(r['pct_mono_100_220']) for r in rows
                                              if r['caller'] == f'{c}+merge' and r['dataset'] in DATASET_ORDER] or [0])

    x = np.arange(len(CALLER_ORDER))
    w = 0.35
    ax = axes[0]
    ax.bar(x - w/2, [per_caller_alone[c] for c in CALLER_ORDER], w,
           label='alone', color='#94a3b8')
    ax.bar(x + w/2, [per_caller_merge[c] for c in CALLER_ORDER], w,
           label='+merge', color='#1e293b')
    ax.axhspan(100, 220, color='#22c55e', alpha=0.1, label='mono range')
    ax.set_xticks(x)
    ax.set_xticklabels(CALLER_ORDER, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Median footprint (bp)')
    ax.set_title('Median nucleosome size')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.bar(x - w/2, [per_caller_alone_nucs[c] for c in CALLER_ORDER], w,
           label='alone', color='#94a3b8')
    ax.bar(x + w/2, [per_caller_merge_nucs[c] for c in CALLER_ORDER], w,
           label='+merge', color='#1e293b')
    ax.set_xticks(x)
    ax.set_xticklabels(CALLER_ORDER, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Mean nucleosomes per read')
    ax.set_title('Nucleosome density')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.bar(x - w/2, [per_caller_alone_pct[c] for c in CALLER_ORDER], w,
           label='alone', color='#94a3b8')
    ax.bar(x + w/2, [per_caller_merge_pct[c] for c in CALLER_ORDER], w,
           label='+merge', color='#1e293b')
    ax.set_xticks(x)
    ax.set_xticklabels(CALLER_ORDER, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('% footprints in 100-220 bp')
    ax.set_title('Mononucleosome fraction')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    fig.suptitle('Effect of v7 Poisson merge on each first-pass', fontsize=13)
    fig.tight_layout()
    out = os.path.join(out_dir, 'figure_merge_effect.png')
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def plot_main(out_dir, rows, nl_dump):
    """Multi-panel summary figure."""
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # Include v8 in the merged ranking set. v8 is listed last so it
    # appears below the other +merge rows in every heatmap.
    merged_callers = [f'{c}+merge' for c in CALLER_ORDER] + V8_ORDER

    # Panel A: pooled size distribution, one line per caller (all datasets)
    ax = fig.add_subplot(gs[0, 0])
    for c in merged_callers:
        all_nl = []
        for ds in DATASET_ORDER:
            all_nl.extend(nl_dump.get(c, {}).get(ds, []))
        if all_nl:
            nl_a = np.asarray(all_nl)
            nl_c = nl_a[(nl_a > 0) & (nl_a <= 800)]
            if len(nl_c):
                hist, edges = np.histogram(nl_c, bins=np.arange(0, 801, 15))
                hist = hist / max(hist.sum(), 1)
                color_key = c.replace('+merge', '')
                ax.plot(edges[:-1] + 7.5, hist,
                        color=CALLER_COLORS.get(color_key, '#be185d'),
                        label=color_key,
                        lw=2.2 if c == 'v8' else 1.3,
                        zorder=10 if c == 'v8' else 1)
    for pos in (147, 300, 450):
        ax.axvline(pos, color='red', ls='--', lw=0.5, alpha=0.6)
    ax.axvspan(100, 220, color='#22c55e', alpha=0.08)
    ax.set_xlabel('Footprint size (bp)')
    ax.set_ylabel('Density')
    ax.set_title('A. Footprint size (all datasets, +merge variant)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.2)

    # Panel B: median footprint heatmap
    ax = fig.add_subplot(gs[0, 1])
    ds_list = [d for d in DATASET_ORDER
                if any(r['dataset'] == d for r in rows)]
    H = np.zeros((len(merged_callers), len(ds_list)))
    for i, c in enumerate(merged_callers):
        for j, d in enumerate(ds_list):
            r = _get_rows(rows, c, d)
            H[i, j] = float(r['median_footprint']) if r else np.nan
    im = ax.imshow(H, aspect='auto', cmap='viridis', vmin=80, vmax=450)
    ax.set_yticks(range(len(merged_callers)))
    ax.set_yticklabels([c.replace('+merge', '') for c in merged_callers],
                        fontsize=7)
    ax.set_xticks(range(len(ds_list)))
    ax.set_xticklabels(ds_list, rotation=45, ha='right', fontsize=7)
    ax.set_title('B. Median footprint (bp)')
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            ax.text(j, i, f'{int(H[i,j])}' if not np.isnan(H[i,j]) else '',
                    ha='center', va='center', fontsize=7,
                    color='white' if H[i,j] < 280 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.035)

    # Panel C: nucs/read heatmap
    ax = fig.add_subplot(gs[0, 2])
    H2 = np.zeros((len(merged_callers), len(ds_list)))
    for i, c in enumerate(merged_callers):
        for j, d in enumerate(ds_list):
            r = _get_rows(rows, c, d)
            H2[i, j] = float(r['mean_nucs_per_read']) if r else np.nan
    im2 = ax.imshow(H2, aspect='auto', cmap='magma')
    ax.set_yticks(range(len(merged_callers)))
    ax.set_yticklabels([c.replace('+merge', '') for c in merged_callers],
                        fontsize=7)
    ax.set_xticks(range(len(ds_list)))
    ax.set_xticklabels(ds_list, rotation=45, ha='right', fontsize=7)
    ax.set_title('C. Mean nucs / read')
    for i in range(H2.shape[0]):
        for j in range(H2.shape[1]):
            ax.text(j, i, f'{H2[i,j]:.1f}' if not np.isnan(H2[i,j]) else '',
                    ha='center', va='center', fontsize=7, color='white')
    plt.colorbar(im2, ax=ax, fraction=0.035)

    # Panel D: oversplit rate (% <100 bp) heatmap
    ax = fig.add_subplot(gs[1, 0])
    H3 = np.zeros((len(merged_callers), len(ds_list)))
    for i, c in enumerate(merged_callers):
        for j, d in enumerate(ds_list):
            r = _get_rows(rows, c, d)
            H3[i, j] = float(r['pct_sub_100']) if r else np.nan
    im3 = ax.imshow(H3, aspect='auto', cmap='Reds', vmin=0, vmax=50)
    ax.set_yticks(range(len(merged_callers)))
    ax.set_yticklabels([c.replace('+merge', '') for c in merged_callers],
                        fontsize=7)
    ax.set_xticks(range(len(ds_list)))
    ax.set_xticklabels(ds_list, rotation=45, ha='right', fontsize=7)
    ax.set_title('D. % sub-100 bp (oversplit)')
    for i in range(H3.shape[0]):
        for j in range(H3.shape[1]):
            v = H3[i, j]
            ax.text(j, i, f'{v:.0f}' if not np.isnan(v) else '',
                    ha='center', va='center', fontsize=7,
                    color='white' if v > 25 else 'black')
    plt.colorbar(im3, ax=ax, fraction=0.035)

    # Panel E: overmerge rate (% >360 bp) heatmap
    ax = fig.add_subplot(gs[1, 1])
    H4 = np.zeros((len(merged_callers), len(ds_list)))
    for i, c in enumerate(merged_callers):
        for j, d in enumerate(ds_list):
            r = _get_rows(rows, c, d)
            H4[i, j] = float(r['pct_over_360']) if r else np.nan
    im4 = ax.imshow(H4, aspect='auto', cmap='Oranges', vmin=0, vmax=50)
    ax.set_yticks(range(len(merged_callers)))
    ax.set_yticklabels([c.replace('+merge', '') for c in merged_callers],
                        fontsize=7)
    ax.set_xticks(range(len(ds_list)))
    ax.set_xticklabels(ds_list, rotation=45, ha='right', fontsize=7)
    ax.set_title('E. % > 360 bp (overmerge)')
    for i in range(H4.shape[0]):
        for j in range(H4.shape[1]):
            v = H4[i, j]
            ax.text(j, i, f'{v:.0f}' if not np.isnan(v) else '',
                    ha='center', va='center', fontsize=7,
                    color='white' if v > 25 else 'black')
    plt.colorbar(im4, ax=ax, fraction=0.035)

    # Panel F: median spacing heatmap
    ax = fig.add_subplot(gs[1, 2])
    H5 = np.zeros((len(merged_callers), len(ds_list)))
    for i, c in enumerate(merged_callers):
        for j, d in enumerate(ds_list):
            r = _get_rows(rows, c, d)
            H5[i, j] = float(r['median_spacing']) if r else np.nan
    im5 = ax.imshow(H5, aspect='auto', cmap='cividis',
                     vmin=100, vmax=400)
    ax.set_yticks(range(len(merged_callers)))
    ax.set_yticklabels([c.replace('+merge', '') for c in merged_callers],
                        fontsize=7)
    ax.set_xticks(range(len(ds_list)))
    ax.set_xticklabels(ds_list, rotation=45, ha='right', fontsize=7)
    ax.set_title('F. Median spacing (center-to-center, bp)')
    for i in range(H5.shape[0]):
        for j in range(H5.shape[1]):
            v = H5[i, j]
            ax.text(j, i, f'{int(v)}' if not np.isnan(v) else '',
                    ha='center', va='center', fontsize=7,
                    color='white' if v < 250 else 'black')
    plt.colorbar(im5, ax=ax, fraction=0.035)

    # Panel G: alone-vs-merge comparison (medians)
    ax = fig.add_subplot(gs[2, 0])
    alone_med = [np.mean([float(r['median_footprint']) for r in rows
                            if r['caller'] == c and r['dataset'] in DATASET_ORDER])
                  for c in CALLER_ORDER]
    merge_med = [np.mean([float(r['median_footprint']) for r in rows
                            if r['caller'] == f'{c}+merge' and r['dataset'] in DATASET_ORDER])
                  for c in CALLER_ORDER]
    x = np.arange(len(CALLER_ORDER))
    ax.bar(x - 0.2, alone_med, 0.4, color='#94a3b8', label='alone')
    ax.bar(x + 0.2, merge_med, 0.4, color='#1e293b', label='+merge')
    ax.axhspan(100, 220, color='#22c55e', alpha=0.12)
    ax.set_xticks(x)
    ax.set_xticklabels(CALLER_ORDER, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Median footprint (bp)')
    ax.set_title('G. Alone vs. +merge median')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # Panel H: % mono (100-220) bar
    ax = fig.add_subplot(gs[2, 1])
    alone_pct = [np.mean([float(r['pct_mono_100_220']) for r in rows
                            if r['caller'] == c and r['dataset'] in DATASET_ORDER])
                  for c in CALLER_ORDER]
    merge_pct = [np.mean([float(r['pct_mono_100_220']) for r in rows
                            if r['caller'] == f'{c}+merge' and r['dataset'] in DATASET_ORDER])
                  for c in CALLER_ORDER]
    ax.bar(x - 0.2, alone_pct, 0.4, color='#94a3b8', label='alone')
    ax.bar(x + 0.2, merge_pct, 0.4, color='#1e293b', label='+merge')
    ax.set_xticks(x)
    ax.set_xticklabels(CALLER_ORDER, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('% footprints 100-220 bp')
    ax.set_title('H. Mononucleosome fraction')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    # Panel I: runtime
    ax = fig.add_subplot(gs[2, 2])
    alone_rt = [np.mean([float(r['runtime_s']) for r in rows
                          if r['caller'] == c and r['dataset'] in DATASET_ORDER])
                 for c in CALLER_ORDER]
    merge_rt = [np.mean([float(r['runtime_s']) for r in rows
                          if r['caller'] == f'{c}+merge' and r['dataset'] in DATASET_ORDER])
                 for c in CALLER_ORDER]
    ax.bar(x - 0.2, alone_rt, 0.4, color='#94a3b8', label='alone')
    ax.bar(x + 0.2, merge_rt, 0.4, color='#1e293b', label='+merge')
    ax.set_xticks(x)
    ax.set_xticklabels(CALLER_ORDER, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Runtime (s, 2000 reads)')
    ax.set_title('I. Runtime')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)

    fig.suptitle('Nucleosome-caller benchmark on DddA (DAF-seq)',
                  fontsize=15, y=0.995)
    out = os.path.join(out_dir, 'figure_main.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'output'))
    args = ap.parse_args()

    rows = load_summary(args.out)
    nl_dump = load_nl_dump(args.out)
    print(f'Loaded {len(rows)} summary rows')

    plot_main(args.out, rows, nl_dump)
    plot_size_grid(args.out, rows, nl_dump, variant='+merge')
    plot_size_grid(args.out, rows, nl_dump, variant='')
    plot_merge_effect(args.out, rows)


if __name__ == '__main__':
    main()
