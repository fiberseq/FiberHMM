"""Side-by-side snapshots: full v8_gapcdf call set vs best_guess
filtered set vs full set colored by nq (protection quality).

Purpose: eyeball whether the best_guess filter (nuc mq >= 64,
TF tq >= 128/64 + edges >= 128) is too harsh. For each of 12 reads
per dataset, we draw three tracks on the SAME x-axis:

  Col 1: all v8_gapcdf nucs colored by nq (0 = baseline rate, 255 =
          deeply protected). Shows raw biological confidence.
  Col 2: all v8_gapcdf nucs colored by mq (0 = tail-of-Poisson
          merge, 255 = pure Pass-1). Shows algorithmic confidence.
  Col 3: only the calls surviving best_guess — same color scheme
          as Col 2 but limited to the passing set. TFs are drawn
          as yellow squares below the read line.

One PNG per dataset, written to bench/output/best_guess_snapshots/.
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
from enzyme_extractors import DAFExtractor  # noqa: E402
from best_guess import best_guess_calls, recommended_tf_tq_threshold  # noqa: E402


# Colormap reuse: mq = dark green → magenta → pink
MQ_CMAP = LinearSegmentedColormap.from_list(
    'mq',
    [
        (0.0,  '#fecaca'),
        (0.3,  '#f9a8d4'),
        (0.6,  '#c026d3'),
        (0.95, '#166534'),
        (1.0,  '#14532d'),
    ],
)
# nq: white → blue → dark blue (protection depth)
NQ_CMAP = LinearSegmentedColormap.from_list(
    'nq',
    [
        (0.0, '#fee2e2'),  # light pink = baseline rate
        (0.3, '#fed7aa'),  # peach
        (0.6, '#fdba74'),  # orange
        (0.8, '#1e3a8a'),  # navy
        (1.0, '#0c1b52'),  # deepest
    ],
)


def q_to_color(q, cmap):
    return cmap(max(0, min(255, int(q))) / 255.0)


def read_calls(bam_path, read_ids):
    """Parse every iter-16 tag for the requested reads."""
    out = {}
    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        if r.query_name not in read_ids:
            continue
        q = r.query_sequence or ''
        nucs = []  # [(s, l, nq, mq, lq, rq), ...]
        tfs = []   # [(s, l, tq, el, er), ...]
        msps = []

        if r.has_tag('MA') and r.has_tag('AQ'):
            parsed = parse_ma_tag(r.get_tag('MA'))
            aq = list(r.get_tag('AQ'))
            qspecs = [rt[2] for rt in parsed['raw_types']]
            npers = [len(rt[3]) for rt in parsed['raw_types']]
            per_ann = parse_aq_array(aq, qspecs, npers)
            idx = 0
            for rt in parsed['raw_types']:
                name = rt[0]
                count = len(rt[3])
                if name == 'nuc':
                    for k, (s, l) in enumerate(rt[3]):
                        row = per_ann[idx + k]
                        nq = row[0] if len(row) >= 1 else 0
                        mq = row[1] if len(row) >= 2 else 255
                        lq = row[2] if len(row) >= 3 else 0
                        rq = row[3] if len(row) >= 4 else 0
                        nucs.append((s, l, nq, mq, lq, rq))
                elif name == 'msp':
                    msps = list(rt[3])
                elif name == 'tf':
                    for k, (s, l) in enumerate(rt[3]):
                        row = per_ann[idx + k]
                        tq = row[0] if len(row) >= 1 else 0
                        el = row[1] if len(row) >= 2 else 0
                        er = row[2] if len(row) >= 3 else 0
                        tfs.append((s, l, tq, el, er))
                idx += count
        else:
            # Legacy-only fallback
            try:
                ns = list(r.get_tag('ns'))
                nl = list(r.get_tag('nl'))
                nq_t = list(r.get_tag('nq')) if r.has_tag('nq') else [0] * len(ns)
                mq_t = [255] * len(ns)
                lq_t = list(r.get_tag('lq')) if r.has_tag('lq') else [0] * len(ns)
                rq_t = list(r.get_tag('rq')) if r.has_tag('rq') else [0] * len(ns)
                for s, l, nq, mq, lq, rq in zip(ns, nl, nq_t, mq_t, lq_t, rq_t):
                    nucs.append((s, l, nq, mq, lq, rq))
            except KeyError:
                pass

        out[r.query_name] = {
            'read': r, 'q': q, 'nucs': nucs, 'tfs': tfs, 'msps': msps,
        }
    bam.close()
    return out


def pick_reads(bam_path, n_reads=12, min_len=2000):
    bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
    long_reads = []
    for r in bam.fetch(until_eof=True):
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            continue
        if (r.query_length or 0) < min_len:
            continue
        long_reads.append(r.query_name)
    bam.close()
    if not long_reads:
        return []
    if len(long_reads) <= n_reads:
        return long_reads
    idx = [int(i) for i in np.linspace(0, len(long_reads) - 1, n_reads)]
    return [long_reads[i] for i in idx]


def draw_nucs(ax, tags, y, nucs, color_by, cmap):
    q = tags['q']
    qlen = len(q)
    ax.plot([0, qlen], [y, y], color='#94a3b8', lw=0.4)
    # Hit ticks
    hits = [i for i, c in enumerate(q) if c in 'YyRr']
    if hits:
        ax.vlines(hits, y - 0.13, y + 0.13, color='#dc2626', lw=0.35)
    # Nucleosome boxes
    for entry in nucs:
        s, l, nq, mq, lq, rq = entry
        qv = {'nq': nq, 'mq': mq}[color_by]
        ax.add_patch(plt.Rectangle(
            (s, y - 0.32), l, 0.64,
            facecolor=q_to_color(qv, cmap), edgecolor='none', alpha=0.92,
        ))


def draw_tfs(ax, tags, y, tfs):
    for s, l, tq, el, er in tfs:
        # TF = small yellow square just above the nuc row
        ax.add_patch(plt.Rectangle(
            (s, y + 0.28), l, 0.18,
            facecolor='#facc15', edgecolor='#a16207', lw=0.3, alpha=0.9,
        ))


def per_read_baseline(read, extractor):
    try:
        ref = read.get_reference_sequence().upper()
    except Exception:
        return None
    L = read.reference_end - read.reference_start
    if not ref or len(ref) != L:
        return None
    opp, hit = extractor.read_to_arrays(read, ref)
    n_opp = int(opp.sum())
    if n_opp < 10:
        return None
    return int(hit.sum()) / n_opp


def plot_dataset(dataset, bam_dir, out_dir, n_reads=12):
    bam_path = os.path.join(bam_dir, f'{dataset}__v8_gapcdf.bam')
    if not os.path.exists(bam_path):
        print(f'  skip {dataset}: no v8_gapcdf BAM')
        return False
    picks = pick_reads(bam_path, n_reads=n_reads)
    if not picks:
        return False
    pick_set = set(picks)
    data = read_calls(bam_path, pick_set)

    # For col 3 (best_guess) apply the filter to each read using its
    # own per-read baseline.
    extractor = DAFExtractor()
    bg_data = {}
    for rid, t in data.items():
        r = t['read']
        baseline = per_read_baseline(r, extractor)
        bg = best_guess_calls(r, baseline=baseline)
        kept_set = set(tuple(x) for x in (bg['nucs'] if bg else []))
        kept_tfs = set(tuple(x) for x in (bg['tfs'] if bg else []))
        bg_nucs = [(s, l, nq, mq, lq, rq) for (s, l, nq, mq, lq, rq) in t['nucs']
                    if (s, l) in kept_set]
        bg_tfs = [(s, l, tq, el, er) for (s, l, tq, el, er) in t['tfs']
                    if (s, l) in kept_tfs]
        bg_data[rid] = {'q': t['q'], 'nucs': bg_nucs, 'tfs': bg_tfs,
                         'baseline': baseline}

    max_qlen = max((len(t['q']) for t in data.values()), default=0)
    fig, axes = plt.subplots(1, 3, figsize=(17, 0.35 * n_reads + 1.5),
                              sharex=True, sharey=True)
    titles = [
        'A. Full set — colored by nq (protection)',
        'B. Full set — colored by mq (merge quality)',
        'C. Best-guess filtered (same as B, kept only)',
    ]
    cols = [
        ('nq', NQ_CMAP, data),
        ('mq', MQ_CMAP, data),
        ('mq', MQ_CMAP, bg_data),
    ]
    for col, (color_by, cmap, col_data) in enumerate(cols):
        ax = axes[col]
        for i, rid in enumerate(picks):
            t = col_data.get(rid)
            if t is None:
                continue
            draw_nucs(ax, t, i, t.get('nucs', []), color_by, cmap)
            draw_tfs(ax, t, i, t.get('tfs', []))
        ax.set_title(titles[col], fontsize=10)
        ax.set_ylim(-0.8, len(picks) - 0.2)
        ax.set_yticks([])
        ax.set_xlim(0, max_qlen)
        ax.tick_params(labelsize=7)
        ax.set_xlabel('Read position (bp)', fontsize=8)

    # Per-read baseline labels next to col 3
    ax_last = axes[-1]
    for i, rid in enumerate(picks):
        b = bg_data[rid].get('baseline')
        if b is not None:
            n_raw = len(data[rid]['nucs'])
            n_bg = len(bg_data[rid]['nucs'])
            ax_last.text(
                max_qlen * 1.005, i,
                f'b={b:.2f}  nucs {n_raw}→{n_bg}',
                va='center', fontsize=7, color='#444',
            )

    # Two color bars side by side
    sm_nq = plt.cm.ScalarMappable(cmap=NQ_CMAP,
                                    norm=plt.Normalize(vmin=0, vmax=255))
    sm_nq.set_array([])
    cbar_nq = fig.colorbar(sm_nq, ax=axes[0], shrink=0.7, pad=0.01,
                            orientation='vertical')
    cbar_nq.set_label('nq (0=baseline, 255=deep protected)', fontsize=8)
    sm_mq = plt.cm.ScalarMappable(cmap=MQ_CMAP,
                                    norm=plt.Normalize(vmin=0, vmax=255))
    sm_mq.set_array([])
    cbar_mq = fig.colorbar(sm_mq, ax=axes[1:], shrink=0.7, pad=0.01,
                            orientation='vertical')
    cbar_mq.set_label('mq (0=tail merge, 255=pure Pass-1)', fontsize=8)

    fig.suptitle(
        f'{dataset}  —  {n_reads} reads  |  v8 gap_cdf full (A/B) vs best_guess (C)  '
        f'|  yellow squares = TF calls',
        fontsize=11,
    )
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'best_guess_snapshot_{dataset}.png')
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
    out_dir = os.path.join(args.out, 'best_guess_snapshots')

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
            import traceback
            print(f'  error on {ds}: {e}')
            traceback.print_exc()


if __name__ == '__main__':
    main()
