#!/usr/bin/env python3
"""tf_recaller.py -- LLR TF footprint recaller on v2 FiberHMM output.

Runs as a 2nd pass on BAMs already tagged by fiberhmm-apply.
Does NOT modify any v2 tags (ns/nl/nq/as/al/aq); only adds tn/tl/ts.

Scan space per read (sources merged, clipped to read bounds):
  - All MSP spans (as/al)
  - Full span of every v2 nuc with nl < --long-nuc-min
    (these ARE v2's already-called sub-nucleosomal footprints; recalling them
     with a TF-calibrated LLR is the primary sanity check)
  - Outer --boundary-sweep bp on each edge of any nuc with nl >= --long-nuc-min.
    DISABLED BY DEFAULT (--boundary-sweep 0). When enabled, the Kadane scan
    runs for only ~sweep bp inside each nuc edge; because a dense nuc has
    near-zero hits, the LLR climbs monotonically and is force-flushed at the
    interval boundary, producing a spurious ~sweep-length TF call at every
    nucleosome edge. Only re-enable once call_tfs_in_interval is rewritten
    to suppress the boundary flush in sweep-derived intervals.

Scoring (per target position inside a scan interval):
  miss step: +[ log P(miss|ctx, protected) - log P(miss|ctx, accessible) ]
  hit  step: +[ log P(hit |ctx, protected) - log P(hit |ctx, accessible) ]
both pulled directly from v2's per-context emission table.

Kadane-style accumulation: running sum floored at 0; on drop to <=0,
flush the peak if LLR >= min_llr and n_target_positions >= min_opps.
This lets a real footprint absorb one rogue hit without being split.

QC: for every v2 ns/nl entry with nl < --long-nuc-min (v2's already-called
sub-nucleosomal footprints), check that our recaller emits an overlapping
tn/tl call. Report recovery rates by size bin (20-35 / 35-65 / 65-90 bp).
The 35-65 bp bin is the Pol II red-flag bin; if recovery there drops
below --qc-warn-threshold, stderr prints a warning.
"""
import argparse
import array as pyarray
import json
import sys
from collections import defaultdict

import numpy as np
import pysam

from fiberhmm.core.model_io import load_model
from fiberhmm.core.bam_reader import (
    encode_from_query_sequence,
    parse_mm_tag_query_positions,
    extract_daf_iupac_positions,
    has_iupac_encoding,
    detect_daf_strand,
)

N_CTX = 4096           # 4^6 hexamer contexts (k=3)
NON_TARGET = N_CTX     # code 4096
UNMETH_OFFSET = 4097   # miss codes live at [4097, 4097+4096)

TS_SCALE = 5.0         # ts tag = min(255, round(llr * TS_SCALE)); saturates at LLR=51


def build_llr_tables(model):
    """Return (llr_hit, llr_miss) lookup arrays of length N_CTX.

    llr_hit[c]  = log P(hit  | c, protected) - log P(hit  | c, accessible)
    llr_miss[c] = log P(miss | c, protected) - log P(miss | c, accessible)

    load_model() calls normalize_states(), which guarantees state 0=protected,
    state 1=accessible. Verified: startprob for state 0 is ~1.0 (reads begin
    protected) in the shipped hia5/ddda/dddb models.
    """
    EP = np.asarray(model.emissionprob_, dtype=np.float64)
    if EP.shape[0] != 2:
        raise ValueError(f"Expected 2-state model, got {EP.shape[0]}")
    if EP.shape[1] < UNMETH_OFFSET + N_CTX:
        raise ValueError(f"Emission table too small: {EP.shape[1]} columns, need {UNMETH_OFFSET + N_CTX}")
    eps = 1e-30
    hit_prot = np.clip(EP[0, :N_CTX], eps, 1.0)
    hit_acc = np.clip(EP[1, :N_CTX], eps, 1.0)
    miss_prot = np.clip(EP[0, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], eps, 1.0)
    miss_acc = np.clip(EP[1, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], eps, 1.0)
    return np.log(hit_prot) - np.log(hit_acc), np.log(miss_prot) - np.log(miss_acc)


def model_p_hit_accessible(model):
    """Mean P(hit | context, accessible) across contexts -- used for per-read
    baseline correction. Averaged over contexts with equal weight (actual
    context frequency in genomic DNA is close to uniform for hexamer).
    """
    EP = np.asarray(model.emissionprob_, dtype=np.float64)
    hit_acc = EP[1, :N_CTX]
    miss_acc = EP[1, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX]
    per_ctx_p = hit_acc / np.clip(hit_acc + miss_acc, 1e-30, None)
    return float(per_ctx_p.mean())


def uplift_llr_tables(llr_hit, llr_miss, model, uplift):
    """Sharpen the emission table and rebuild LLR tables.

    Per context c, compute conditional P(hit | target, state) from the raw
    emission probabilities, then apply a power transform that moves the
    accessible state's P(hit) toward 1 and the protected state's P(hit)
    toward 0:

        p_hit_acc_new  = 1 - (1 - p_hit_acc)  ** uplift
        p_hit_prot_new =      p_hit_prot      ** uplift

    Use cases: DddA on a dddb-trained model (DddA is ~3-5x higher efficiency,
    so the true accessible-state P(hit) is closer to 1 than dddb emissions
    say). Uplift=1.0 is identity.
    """
    if abs(uplift - 1.0) < 1e-6:
        return llr_hit, llr_miss
    EP = np.asarray(model.emissionprob_, dtype=np.float64)
    eps = 1e-30
    hit_prot = np.clip(EP[0, :N_CTX], eps, 1.0)
    hit_acc = np.clip(EP[1, :N_CTX], eps, 1.0)
    miss_prot = np.clip(EP[0, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], eps, 1.0)
    miss_acc = np.clip(EP[1, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX], eps, 1.0)

    p_hit_acc = hit_acc / (hit_acc + miss_acc)
    p_hit_prot = hit_prot / (hit_prot + miss_prot)
    p_hit_acc_new = 1.0 - np.power(np.clip(1.0 - p_hit_acc, eps, 1.0), uplift)
    p_hit_prot_new = np.power(np.clip(p_hit_prot, eps, 1.0), uplift)
    p_hit_acc_new = np.clip(p_hit_acc_new, eps, 1.0 - eps)
    p_hit_prot_new = np.clip(p_hit_prot_new, eps, 1.0 - eps)

    new_llr_miss = np.log(1.0 - p_hit_prot_new) - np.log(1.0 - p_hit_acc_new)
    new_llr_hit = np.log(p_hit_prot_new) - np.log(p_hit_acc_new)
    return new_llr_hit, new_llr_miss


def auto_min_llr(llr_miss, enzyme_mode, multiplier=6.0):
    """Auto-calibrated min_llr = multiplier * median(|llr_miss|).

    Rationale: the per-miss LLR is enzyme-dependent (DAF ~0.3, Hia5 ~1.7)
    because the accessible-state hit probability differs. A fixed min_llr
    over-fires on Hia5 (where a 3-miss random streak clears 5.0) and
    under-fires on DAF (where a 5-miss streak barely clears 2). Scaling
    by median llr_miss produces enzyme-matched thresholds:
        DAF (median 1.0)  -> ~6
        Hia5 (median 1.7) -> ~10
    """
    return float(multiplier * np.median(llr_miss))


def load_model_with_meta(path):
    """Return (FiberHMM, mode_str, context_size_int). Reads JSON metadata."""
    model = load_model(path)
    mode = 'pacbio-fiber'
    k = 3
    if path.endswith('.json'):
        with open(path) as f:
            d = json.load(f)
        mode = d.get('mode', mode)
        k = int(d.get('context_size', k))
    return model, mode, k


def merged_scan_intervals(ns, nl, as_, al, read_len, boundary_sweep, long_nuc_min):
    """Build merged list of [start, end) query intervals to scan.

    Sources:
      1. All MSP spans (as_, al).
      2. Full span of every v2 nuc with nl < long_nuc_min (v2's sub-nuc TF calls).
      3. Outer `boundary_sweep` bp on each edge of every v2 nuc with nl >= long_nuc_min.
         A 200 bp 'nuc' may be 147 bp real nuc + 50 bp over-merged TF at one end;
         scanning both outer strips surfaces either side.
    """
    iv = []
    for s, l in zip(as_, al):
        s = int(s); l = int(l)
        if l > 0:
            iv.append((s, s + l))
    for s, l in zip(ns, nl):
        s = int(s); l = int(l)
        if l <= 0:
            continue
        if l < long_nuc_min:
            # v2 short nuc -- scan full span
            iv.append((s, s + l))
        else:
            left_end = min(s + boundary_sweep, s + l)
            iv.append((s, left_end))
            right_start = max(s, s + l - boundary_sweep)
            iv.append((right_start, s + l))
    # Clip and drop empties
    iv = [(max(0, a), min(read_len, b)) for a, b in iv]
    iv = [(a, b) for a, b in iv if b > a]
    if not iv:
        return []
    iv.sort()
    merged = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def call_tfs_in_interval(obs, lo, hi, llr_hit, llr_miss, min_llr, min_opps):
    """Kadane-local scan inside obs[lo:hi]. Returns [(start, length, llr, n_opps), ...]."""
    calls = []
    # "Inside a run" state -- None means we haven't started climbing
    cur_start = None
    running = 0.0
    opps = 0
    peak_llr = 0.0
    peak_end = lo
    peak_opps = 0

    for i in range(lo, hi):
        code = int(obs[i])
        if 0 <= code < N_CTX:
            step = llr_hit[code]
            is_opp = True
        elif UNMETH_OFFSET <= code < UNMETH_OFFSET + N_CTX:
            step = llr_miss[code - UNMETH_OFFSET]
            is_opp = True
        else:
            step = 0.0
            is_opp = False

        if cur_start is None:
            # Only start a run when a positive step pulls us off 0
            if step > 0:
                cur_start = i
                running = step
                opps = 1 if is_opp else 0
                peak_llr = running
                peak_end = i + 1
                peak_opps = opps
            continue

        running += step
        if is_opp:
            opps += 1

        if running > peak_llr:
            peak_llr = running
            peak_end = i + 1
            peak_opps = opps

        if running <= 0:
            if peak_llr >= min_llr and peak_opps >= min_opps:
                calls.append((cur_start, peak_end - cur_start, peak_llr, peak_opps))
            cur_start = None
            running = 0.0
            opps = 0
            peak_llr = 0.0
            peak_end = i + 1
            peak_opps = 0

    # End-of-interval flush
    if cur_start is not None and peak_llr >= min_llr and peak_opps >= min_opps:
        calls.append((cur_start, peak_end - cur_start, peak_llr, peak_opps))
    return calls


def size_bin(l):
    if l < 35:
        return '20-35'
    if l < 65:
        return '35-65'
    if l < 90:
        return '65-90'
    return '>=90'


def _get_tag_list(read, name):
    try:
        return list(read.get_tag(name))
    except KeyError:
        return []


def process_read(read, llr_hit, llr_miss, mode, context_size,
                 min_llr, min_opps, boundary_sweep, long_nuc_min,
                 per_read_baseline=False, model_p_hit_acc=None):
    """Return (tn, tl, ts, qc_dict). qc_dict: {bin -> (total, recovered)}.

    If per_read_baseline=True, the per-miss and per-hit LLR tables are
    shifted to match the read's empirical P(hit) inside v2 MSPs:
        p_obs = (target hits in MSPs) / (target positions in MSPs)
        miss shift: log(1 - p_model_acc) - log(1 - p_obs)
        hit  shift: log(p_model_acc)     - log(p_obs)
    This corrects for labeling-efficiency variation across reads.
    """
    ns = _get_tag_list(read, 'ns')
    nl = _get_tag_list(read, 'nl')
    as_ = _get_tag_list(read, 'as')
    al = _get_tag_list(read, 'al')
    if not (ns or as_):
        return [], [], [], {}

    seq = read.query_sequence
    if seq is None or len(seq) < 2 * context_size + 1:
        return [], [], [], {}

    # Extract modification positions + strand.
    # NOTE: we use the pure-Python MM/ML parser instead of pysam's
    # read.modified_bases property because the latter segfaults on some
    # long/dense Hia5 reads (SIGSEGV cannot be caught in Python).
    strand = '.'
    if mode == 'daf' and has_iupac_encoding(seq):
        try:
            st_tag = read.get_tag('st')
        except KeyError:
            st_tag = None
        mod_pos, strand, seq = extract_daf_iupac_positions(seq, st_tag)
    else:
        try:
            mm_tag = read.get_tag('MM') if read.has_tag('MM') else read.get_tag('Mm')
        except KeyError:
            mm_tag = ''
        try:
            ml_tag = list(read.get_tag('ML')) if read.has_tag('ML') else list(read.get_tag('Ml'))
        except KeyError:
            ml_tag = []
        if not mm_tag or not ml_tag:
            return [], [], [], {}
        mod_pos = parse_mm_tag_query_positions(
            mm_tag, ml_tag, seq, read.is_reverse,
            prob_threshold=125, mode=mode
        )
        if mode == 'daf':
            strand = detect_daf_strand(seq, mod_pos)

    obs = encode_from_query_sequence(
        seq, mod_pos, edge_trim=10, mode=mode, strand=strand, context_size=context_size
    )
    read_len = len(seq)

    intervals = merged_scan_intervals(ns, nl, as_, al, read_len,
                                      boundary_sweep=boundary_sweep,
                                      long_nuc_min=long_nuc_min)

    # Per-read baseline correction: shift LLR tables to match observed hit rate
    if per_read_baseline and model_p_hit_acc is not None:
        n_target = 0
        n_hit = 0
        for s_, l_ in zip(as_, al):
            s_ = int(s_); l_ = int(l_)
            if l_ <= 0:
                continue
            sub = obs[max(0, s_):min(read_len, s_ + l_)]
            hit_mask = (sub >= 0) & (sub < N_CTX)
            miss_mask = (sub >= UNMETH_OFFSET) & (sub < UNMETH_OFFSET + N_CTX)
            n_target += int(hit_mask.sum() + miss_mask.sum())
            n_hit += int(hit_mask.sum())
        if n_target >= 50:  # need enough evidence to estimate
            p_obs = max(0.01, min(0.99, n_hit / n_target))
            p_mdl = max(0.01, min(0.99, model_p_hit_acc))
            miss_shift = np.log(1.0 - p_mdl) - np.log(1.0 - p_obs)
            hit_shift = np.log(p_mdl) - np.log(p_obs)
            llr_miss = llr_miss + miss_shift
            llr_hit = llr_hit + hit_shift

    tn, tl, ts = [], [], []
    for lo, hi in intervals:
        for start, length, llr, _ in call_tfs_in_interval(
            obs, lo, hi, llr_hit, llr_miss, min_llr, min_opps
        ):
            tn.append(int(start))
            tl.append(int(length))
            ts.append(int(max(0, min(255, round(llr * TS_SCALE)))))

    # QC: per v2 short nuc, did we produce an overlapping TF call?
    qc = defaultdict(lambda: [0, 0])
    for s, l in zip(ns, nl):
        s = int(s); l = int(l)
        if l >= long_nuc_min or l <= 0:
            continue
        b = size_bin(l)
        qc[b][0] += 1
        end = s + l
        for tn_s, tn_l in zip(tn, tl):
            if tn_s < end and tn_s + tn_l > s:
                qc[b][1] += 1
                break
    return tn, tl, ts, {k: tuple(v) for k, v in qc.items()}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--in-bam', required=True)
    ap.add_argument('--out-bam', required=True)
    ap.add_argument('--model', required=True,
                    help='v2 FiberHMM model (JSON recommended; carries mode+context_size)')
    ap.add_argument('--mode', default=None,
                    help='Override mode (pacbio-fiber | nanopore-fiber | daf)')
    ap.add_argument('--context-size', type=int, default=None,
                    help='Override context size (default: read from model JSON)')
    ap.add_argument('--min-llr', type=float, default=5.0,
                    help='Minimum Kadane-peak LLR (nats) to emit a call (default 5.0)')
    ap.add_argument('--emission-uplift', type=float, default=1.0,
                    help='Sharpen the emission table before LLR: moves p_hit_acc '
                         'toward 1 and p_hit_prot toward 0 by the given power. '
                         'Default 1.0 (identity). Use for e.g. DddA on a dddb-'
                         'trained model: uplift 2-3 accounts for DddA being '
                         'much higher efficiency than DddB.')
    ap.add_argument('--auto-threshold', action='store_true',
                    help='Override --min-llr with enzyme-calibrated threshold: '
                         '6.0 * median(llr_miss). DAF->~6, Hia5->~10')
    ap.add_argument('--auto-threshold-mult', type=float, default=6.0,
                    help='Multiplier for --auto-threshold (default 6.0)')
    ap.add_argument('--per-read-baseline', action='store_true',
                    help='Shift LLR tables per-read to match empirical hit '
                         'rate inside v2 MSPs (corrects for labeling efficiency '
                         'variation). Recommended for Hia5 datasets.')
    ap.add_argument('--min-opps', type=int, default=3,
                    help='Minimum target positions inside a call (default 3)')
    ap.add_argument('--boundary-sweep', type=int, default=0,
                    help='Outer bp scanned on each edge of nucs with nl >= long-nuc-min. '
                         'DEFAULT 0 (disabled). Non-zero values are unsafe with the '
                         'current end-of-interval flush: Kadane hits the artificial cut '
                         'with peak LLR still climbing inside a real nuc, emitting a '
                         'uniform carpet of ~30 bp pseudo-TFs across the genome. '
                         'TFs fused into big nucs by v2 over-merging are left on the '
                         'table until the flush is rewritten to require a natural '
                         'hit-bounded drop inside the sweep window.')
    ap.add_argument('--long-nuc-min', type=int, default=90,
                    help='v2 nucs >= this are treated as real nucleosomes (scan outer edges); '
                         'v2 nucs < this are sub-nuc TFs used for QC recovery check')
    ap.add_argument('--max-reads', type=int, default=0, help='0 = no limit')
    ap.add_argument('--qc-warn-threshold', type=float, default=0.85,
                    help='Stderr warning if 35-65 bp Pol II bin recovery drops below this')
    ap.add_argument('--preserve-check', type=int, default=100,
                    help='Assert v2 tag bytes unchanged on the first N processed reads '
                         '(0 disables; defends against accidental v2-tag clobbering)')
    args = ap.parse_args()

    model, model_mode, model_k = load_model_with_meta(args.model)
    mode = args.mode or model_mode
    k = args.context_size or model_k
    llr_hit, llr_miss = build_llr_tables(model)
    if args.emission_uplift != 1.0:
        llr_hit, llr_miss = uplift_llr_tables(llr_hit, llr_miss, model, args.emission_uplift)
        print(f"[tf_recaller] emission uplift = {args.emission_uplift} applied", file=sys.stderr)

    min_llr_effective = args.min_llr
    if args.auto_threshold:
        min_llr_effective = auto_min_llr(llr_miss, mode, args.auto_threshold_mult)
        print(f"[tf_recaller] auto-threshold: min_llr={min_llr_effective:.2f} "
              f"(= {args.auto_threshold_mult} * median miss-LLR)",
              file=sys.stderr)

    p_hit_acc_model = model_p_hit_accessible(model) if args.per_read_baseline else None
    if args.per_read_baseline:
        print(f"[tf_recaller] per-read baseline ON; model p_hit_acc = {p_hit_acc_model:.3f}",
              file=sys.stderr)

    print(f"[tf_recaller] model={args.model} mode={mode} k={k} min_llr={min_llr_effective:.2f} "
          f"min_opps={args.min_opps} boundary_sweep={args.boundary_sweep} "
          f"long_nuc_min={args.long_nuc_min}", file=sys.stderr)
    print(f"[tf_recaller] llr_miss range: [{llr_miss.min():.3f}, {llr_miss.max():.3f}] "
          f"median {np.median(llr_miss):.3f}", file=sys.stderr)
    print(f"[tf_recaller] llr_hit  range: [{llr_hit.min():.3f}, {llr_hit.max():.3f}] "
          f"median {np.median(llr_hit):.3f}", file=sys.stderr)

    bam_in = pysam.AlignmentFile(args.in_bam, 'rb', check_sq=False)
    bam_out = pysam.AlignmentFile(args.out_bam, 'wb', template=bam_in)

    # Tags we must pass through untouched. t* are the new tags we add.
    V2_TAGS = ('ns', 'nl', 'nq', 'as', 'al', 'aq')
    preserve_checks_left = args.preserve_check

    n_reads = 0
    n_tagged = 0
    total_tfs = 0
    qc_totals = defaultdict(lambda: [0, 0])

    for read in bam_in:
        if args.max_reads and n_reads >= args.max_reads:
            break
        n_reads += 1

        if preserve_checks_left > 0:
            pre = {t: read.get_tag(t) if read.has_tag(t) else None for t in V2_TAGS}

        tn, tl, ts, qc = process_read(
            read, llr_hit, llr_miss, mode, k,
            min_llr_effective, args.min_opps, args.boundary_sweep, args.long_nuc_min,
            per_read_baseline=args.per_read_baseline,
            model_p_hit_acc=p_hit_acc_model,
        )

        if tn:
            n_tagged += 1
            total_tfs += len(tn)
            read.set_tag('tn', pyarray.array('I', tn))
            read.set_tag('tl', pyarray.array('I', tl))
            read.set_tag('ts', pyarray.array('B', ts))

        if preserve_checks_left > 0:
            for t in V2_TAGS:
                now = read.get_tag(t) if read.has_tag(t) else None
                if pre[t] != now:
                    raise RuntimeError(f"v2 tag {t!r} was modified on read {read.query_name!r}")
            preserve_checks_left -= 1

        for b, (tot, rec) in qc.items():
            qc_totals[b][0] += tot
            qc_totals[b][1] += rec

        bam_out.write(read)

    bam_in.close()
    bam_out.close()

    print(f"\n[tf_recaller] Processed {n_reads} reads; {n_tagged} carry TF tags; "
          f"{total_tfs} TF calls emitted", file=sys.stderr)
    print(f"[tf_recaller] v2 short-nuc recovery (sanity check -- these are "
          f"the sub-nucleosomal footprints v2 already calls):", file=sys.stderr)
    for b in ('20-35', '35-65', '65-90'):
        tot, rec = qc_totals.get(b, [0, 0])
        pct = 100.0 * rec / tot if tot else float('nan')
        tag = ''
        if b == '35-65':
            tag = '  <- Pol II primary bin'
            if tot and pct < 100.0 * args.qc_warn_threshold:
                tag += f'  !! BELOW {args.qc_warn_threshold:.0%} -- recaller likely broken'
        if tot == 0:
            print(f"  {b} bp: (no v2 short nucs in this bin)", file=sys.stderr)
        else:
            print(f"  {b} bp: {rec:>7}/{tot:<7} ({pct:5.1f}%){tag}", file=sys.stderr)


if __name__ == '__main__':
    main()
