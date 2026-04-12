#!/usr/bin/env python3
"""Pull caller_v8's "best guess" nucleosome + TF calls out of a BAM.

Motivation: caller_v8 emits every reasonable candidate along with
three quality scores (tq/el/er for TFs, nq/mq/lq/rq for nucleosomes)
so power users can filter post-hoc. For 99 % of users that is too
much to reason about — they just want "the confident calls". This
module centralises the recommended filter so that downstream code
can call

    from best_guess import best_guess_calls
    calls = best_guess_calls(read, baseline=None)

and get back a simple dict of filtered nucleosomes + TFs.

Thresholds (iter-16b — loosened for dense DAF biology):

    nucleosomes:
        mq >= 0  (keep all — the mq is still in the BAM tag for
                   users who want a confidence gradient; the
                   "best guess" here is the caller's raw output).
        Note: pure Pass-1 atoms (no internal merges) have mq=255.

    TFs (baseline-aware):
        el >= 128 AND er >= 128    (both edges pinned to a HIT)
        tq >= 22 if baseline >= 0.20 else 12
            At baseline 0.24 (typical NAPA/amplicon), tq=22
            corresponds to ~2 consecutive missed deaminations — that
            is the smallest biologically meaningful TF footprint in
            dense DAF data. At baseline 0.10 (scDAF), tq=12 scales
            down for the reduced per-miss information.

            Rationale: in dense DAF-seq data every opportunity is
            hit unless something protects it, so even a couple of
            consecutive misses inside an MSP is a real footprint.
            The earlier tq>=128 default was a statistically strict
            frequentist cutoff that matched nothing in biology.

            Iter-16e: these are 2/3 of the iter-16b values because
            _TFP_MAX_NEG_LOG10 moved from 2.0 to 3.0 in caller_v8
            (less saturation at the top of the 0-255 range).

CLI usage:
    python best_guess.py --in-bam output.bam --out-bam filtered.bam

rewrites the BAM in place, stripping any TF calls that fail the
recommended filter and any nucleosomes with mq < 64. The MA string
is rebuilt so downstream MA-aware tools see only the confident set.
"""

from __future__ import annotations

import argparse
import os
import sys

import pysam

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ma_tags import format_ma_tag, format_aq_array, parse_ma_tag, parse_aq_array
from caller_v7 import set_array_tag
from caller_v8 import clear_all_call_tags, NEW_LEGACY_TAGS, MA_TAGS
from enzyme_extractors import DAFExtractor


DEFAULT_NUC_MQ_MIN = 0       # keep all nucleosomes — mq still in tag
DEFAULT_TF_EDGE_MIN = 128    # both edges pinned to a HIT
DEFAULT_TF_TQ_HI_BASELINE = 22  # baseline >= 0.20 — ~2 misses @ 0.24
DEFAULT_TF_TQ_LO_BASELINE = 12  # baseline <  0.20 — ~3 misses @ 0.10
BASELINE_SPLIT = 0.20
# Iter-16e: thresholds were scaled 2/3× when _TFP_MAX_NEG_LOG10 moved
# from 2.0 to 3.0 in caller_v8.py. Biology unchanged ("2 consecutive
# missed deaminations is a footprint"), just the integer mapping.


def recommended_tf_tq_threshold(baseline: float) -> int:
    """Baseline-adaptive tq threshold for 'best guess' filtering."""
    if baseline is None:
        return DEFAULT_TF_TQ_HI_BASELINE
    return (DEFAULT_TF_TQ_HI_BASELINE
            if baseline >= BASELINE_SPLIT
            else DEFAULT_TF_TQ_LO_BASELINE)


def best_guess_calls(read, baseline: float = None,
                       nuc_mq_min: int = DEFAULT_NUC_MQ_MIN,
                       tf_edge_min: int = DEFAULT_TF_EDGE_MIN,
                       tf_tq_min: int = None):
    """Read iter-16 tags from `read` and return a filtered dict.

    Args:
        read: pysam AlignedSegment
        baseline: optional per-read deamination baseline (used to pick
            a tq floor). If None, the high-baseline floor is used.
        nuc_mq_min: drop nucleosomes with mq below this (default 64).
        tf_edge_min: drop TFs with el or er below this (default 128).
        tf_tq_min: drop TFs with tq below this; if None, chosen from
            baseline.

    Returns:
        {'nucs': [(s, l), ...], 'msps': [(s, l), ...],
         'tfs': [(s, l), ...]} — all in query coords, legacy convention.
        Nothing is emitted for reads that don't have call tags.
    """
    if tf_tq_min is None:
        tf_tq_min = recommended_tf_tq_threshold(baseline)

    # Nucleosomes
    try:
        ns = list(read.get_tag('ns'))
        nl = list(read.get_tag('nl'))
    except KeyError:
        return None
    try:
        mq = list(read.get_tag('MA')) if False else None  # placeholder
    except KeyError:
        mq = None

    # Prefer MA parsing (canonical). Fall back to legacy if absent.
    if read.has_tag('MA'):
        ma = parse_ma_tag(read.get_tag('MA'))
        nucs_all = list(ma['nuc'])
        msps = list(ma['msp'])
        # Extract mq from AQ
        mqs = [255] * len(nucs_all)
        tfs_all = []
        tqs = []
        els = []
        ers = []
        if read.has_tag('AQ'):
            aq = list(read.get_tag('AQ'))
            qual_specs = [rt[2] for rt in ma['raw_types']]
            n_per = [len(rt[3]) for rt in ma['raw_types']]
            per_ann = parse_aq_array(aq, qual_specs, n_per)
            idx = 0
            for rt in ma['raw_types']:
                name = rt[0]
                count = len(rt[3])
                spec = rt[2]
                if name == 'nuc' and len(spec) >= 2:
                    # Layout is (nq, mq, [lq, rq]). Second is mq.
                    for k in range(count):
                        mqs[k] = per_ann[idx + k][1]
                elif name == 'tf' and len(spec) >= 1:
                    # Layout is (tq, [el, er]). Collect TFs from MA +
                    # matching per_ann rows, parallel.
                    for k, (s, l) in enumerate(rt[3]):
                        row = per_ann[idx + k]
                        tq = row[0] if len(row) >= 1 else 0
                        el = row[1] if len(row) >= 2 else 0
                        er = row[2] if len(row) >= 3 else 0
                        tfs_all.append((s, l))
                        tqs.append(tq)
                        els.append(el)
                        ers.append(er)
                idx += count
    else:
        nucs_all = list(zip(ns, nl))
        msps = []
        try:
            as_ = list(read.get_tag('as'))
            al = list(read.get_tag('al'))
            msps = list(zip(as_, al))
        except KeyError:
            pass
        try:
            mqs = list(read.get_tag('nq'))  # no dedicated mq tag in legacy-only
        except KeyError:
            mqs = [255] * len(nucs_all)
        tfs_all = []
        tqs = []
        els = []
        ers = []
        try:
            tn = list(read.get_tag('tn'))
            tl = list(read.get_tag('tl'))
            tq = list(read.get_tag('tq'))
            el = list(read.get_tag('el'))
            er = list(read.get_tag('er'))
            tfs_all = list(zip(tn, tl))
            tqs, els, ers = tq, el, er
        except KeyError:
            pass

    # Apply filters
    nucs_keep = [(s, l) for (s, l), m in zip(nucs_all, mqs) if m >= nuc_mq_min]
    tfs_keep = [
        (s, l) for (s, l), tq, el, er in zip(tfs_all, tqs, els, ers)
        if tq >= tf_tq_min and el >= tf_edge_min and er >= tf_edge_min
    ]
    return {'nucs': nucs_keep, 'msps': msps, 'tfs': tfs_keep}


def _per_read_baseline(read, extractor):
    """Compute the DAF baseline (hits / opportunities) for one read.

    Returns None if the read can't be processed (unmapped, missing
    MD tag, etc.).
    """
    try:
        ref_seq = read.get_reference_sequence().upper()
    except Exception:
        return None
    L = read.reference_end - read.reference_start
    if not ref_seq or len(ref_seq) != L:
        return None
    opp, hit = extractor.read_to_arrays(read, ref_seq)
    n_opp = int(opp.sum())
    if n_opp < 10:
        return None
    return int(hit.sum()) / n_opp


def _rewrite_bam(in_path, out_path, nuc_mq_min, tf_edge_min, tf_tq_hi, tf_tq_lo):
    src = pysam.AlignmentFile(in_path, 'rb', check_sq=False)
    dst = pysam.AlignmentFile(out_path, 'wb', template=src)
    extractor = DAFExtractor()
    n = n_rewritten = 0
    for r in src.fetch(until_eof=True):
        n += 1
        if r.is_unmapped or r.is_secondary or r.is_supplementary:
            dst.write(r)
            continue
        if not r.has_tag('ns') and not r.has_tag('MA'):
            dst.write(r)
            continue
        # Per-read baseline for the tq threshold. Falls back to the hi
        # threshold if the read can't be bundled (rare — usually means
        # no MD tag).
        baseline = _per_read_baseline(r, extractor)
        tf_tq_min = (
            tf_tq_hi if baseline is None or baseline >= BASELINE_SPLIT
            else tf_tq_lo
        )
        best = best_guess_calls(
            r, baseline=baseline,
            nuc_mq_min=nuc_mq_min,
            tf_edge_min=tf_edge_min,
            tf_tq_min=tf_tq_min,
        )
        if best is None:
            dst.write(r)
            continue

        # Re-emit filtered tags. We do NOT touch MQ/LQ/RQ/TQ arrays
        # here because filtering drops entries uniformly — reading
        # back the tags requires careful parallel indexing. The
        # simpler approach: keep the same AQ array but rewrite MA
        # to reflect the filtered set, and zero out mq-below arrays.
        #
        # For this CLI helper we instead emit a "flattened" minimal
        # tag set — legacy ns/nl + as/al + tn/tl only (no qualities),
        # so downstream tools that don't want to think about qualities
        # get the canonical "best guess" interval list directly.
        clear_all_call_tags(r)
        if best['nucs']:
            set_array_tag(r, 'ns', [s for s, _ in best['nucs']])
            set_array_tag(r, 'nl', [l for _, l in best['nucs']])
        if best['msps']:
            set_array_tag(r, 'as', [s for s, _ in best['msps']])
            set_array_tag(r, 'al', [l for _, l in best['msps']])
        if best['tfs']:
            set_array_tag(r, 'tn', [s for s, _ in best['tfs']])
            set_array_tag(r, 'tl', [l for _, l in best['tfs']])
        # Also re-emit a minimal MA for downstream MA-aware viewers
        qlen = r.query_length or 0
        ma_str = format_ma_tag(
            qlen, best['nucs'], best['msps'],
            tf_intervals=best['tfs'],
            nuc_qual_spec='',  # no qualities in the best-guess file
            tf_qual_spec='',
        )
        r.set_tag('MA', ma_str, value_type='Z')
        dst.write(r)
        n_rewritten += 1
    src.close()
    dst.close()
    return n, n_rewritten


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--in-bam', required=True)
    ap.add_argument('--out-bam', required=True)
    ap.add_argument('--nuc-mq-min', type=int, default=DEFAULT_NUC_MQ_MIN,
                    help='drop nucleosomes with mq below this (default 64)')
    ap.add_argument('--tf-edge-min', type=int, default=DEFAULT_TF_EDGE_MIN,
                    help='drop TFs with el or er below this (default 128)')
    ap.add_argument('--tf-tq-hi', type=int, default=DEFAULT_TF_TQ_HI_BASELINE,
                    help='drop TFs with tq below this when baseline>=0.20 '
                         '(default 128)')
    ap.add_argument('--tf-tq-lo', type=int, default=DEFAULT_TF_TQ_LO_BASELINE,
                    help='drop TFs with tq below this when baseline<0.20 '
                         '(default 64)')
    args = ap.parse_args()
    n, n_rewritten = _rewrite_bam(
        args.in_bam, args.out_bam,
        nuc_mq_min=args.nuc_mq_min,
        tf_edge_min=args.tf_edge_min,
        tf_tq_hi=args.tf_tq_hi,
        tf_tq_lo=args.tf_tq_lo,
    )
    print(f'{n} reads scanned, {n_rewritten} rewritten with filtered tags')
    print(f'Wrote {args.out_bam}')


if __name__ == '__main__':
    main()
