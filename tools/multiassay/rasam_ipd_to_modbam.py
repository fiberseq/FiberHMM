#!/usr/bin/env python3
"""
rasam_ipd_to_modbam.py — call m6A from RASAM raw-IPD pickles into a standard modBAM.

RASAM (Ramani, GSE245558) publishes ONLY raw IPD (no called BAMs): each ZMW ->
{'read': seq, 'forwardM'/'reverseM': per-position IPD ratio, 'forwardNSub'/'reverseNsub':
subread counts}. This does the binarization RASAM never shipped — a per-read 2-component
GMM on the IPD at target positions (A on forward, T on reverse = A on the opposite strand),
giving posterior P(m6A) -> MM/ML. Output = UNALIGNED modBAM with the real read sequence and
standard tags (A+a, T+a), ready to align (minimap2/pbmm2 to mm39) and feed pacbio-fiber mode.

CAVEAT: single-molecule PacBio IPD is weakly bimodal (meth/unmeth overlap heavily), so these
calls are noisier than instrument MM/ML. Proof-of-concept / IDLI comparison, not gold truth.

Usage:
  rasam_ipd_to_modbam.py -i block.pickle[.gz] -o out.unaligned.bam [--min-callable 50]
"""
import argparse
import array
import gzip
import pickle
import sys

import numpy as np
import pysam
from sklearn.mixture import GaussianMixture


def call_read(seq, fM, rM, min_callable):
    """Return dict pos->prob(0..1) for m6A, keyed by 0-based read position.
    A positions scored from forward IPD, T positions from reverse IPD."""
    idx, vals, kinds = [], [], []
    for i, b in enumerate(seq):
        if b == "A" and np.isfinite(fM[i]):
            idx.append(i); vals.append(fM[i]); kinds.append("A")
        elif b == "T" and np.isfinite(rM[i]):
            idx.append(i); vals.append(rM[i]); kinds.append("T")
    if len(vals) < min_callable:
        return None
    v = np.asarray(vals, float).reshape(-1, 1)
    # per-read 2-component GMM; the higher-mean component is "methylated"
    try:
        gm = GaussianMixture(n_components=2, covariance_type="full", n_init=1,
                             means_init=[[np.percentile(v, 40)], [np.percentile(v, 98)]])
        gm.fit(v)
    except Exception:
        return None
    hi = int(np.argmax(gm.means_.ravel()))
    post = gm.predict_proba(v)[:, hi]
    return list(zip(idx, kinds, post))


def build_tags(seq, calls):
    """Build MM/ML for m6A (A+a and T+a) from (pos,kind,prob) calls."""
    by = {"A": {}, "T": {}}
    for pos, kind, p in calls:
        by[kind][pos] = p
    mm_parts, ml = [], []
    for base in ("A", "T"):
        deltas, skip = [], 0
        for i, b in enumerate(seq):
            if b != base:
                continue
            if i in by[base]:
                deltas.append(skip)
                q = int(round(by[base][i] * 255))
                ml.append(0 if q < 0 else 255 if q > 255 else q)
                skip = 0
            else:
                skip += 1
        if deltas:
            mm_parts.append(f"{base}+a," + ",".join(map(str, deltas)))
    return (";".join(mm_parts) + ";") if mm_parts else "", array.array("B", ml)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", required=True, help="RASAM block pickle (.pickle or .pickle.gz)")
    ap.add_argument("-o", "--output", required=True, help="output UNALIGNED modBAM")
    ap.add_argument("--min-callable", type=int, default=50)
    ap.add_argument("--max-reads", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    opener = gzip.open if args.input.endswith(".gz") else open
    with opener(args.input, "rb") as fh:
        d = pickle.load(fh)

    header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "unaligned", "LN": 1}]}
    hdr = pysam.AlignmentHeader.from_dict(header)
    n_out = n_skip = 0
    meth_frac = []
    with pysam.AlignmentFile(args.output, "wb", header=header) as out:
        for j, (zmw, inner) in enumerate(d.items()):
            if args.max_reads and n_out >= args.max_reads:
                break
            seq = inner["read"]
            fM = np.asarray(inner["forwardM"], float)
            rM = np.asarray(inner["reverseM"], float)
            if len(fM) != len(seq) or len(rM) != len(seq):
                n_skip += 1; continue
            calls = call_read(seq, fM, rM, args.min_callable)
            if not calls:
                n_skip += 1; continue
            mm, ml = build_tags(seq, calls)
            if not mm:
                n_skip += 1; continue
            meth_frac.append(np.mean([p for _, _, p in calls]))
            a = pysam.AlignedSegment(header=hdr)
            a.query_name = f"rasam_zmw{zmw}"
            a.flag = 4  # unmapped
            a.query_sequence = seq
            a.query_qualities = None
            a.set_tag("MM", mm, "Z")
            a.set_tag("ML", ml)
            out.write(a)
            n_out += 1
            if n_out % 2000 == 0:
                print(f"  {n_out} reads...", file=sys.stderr)

    mf = float(np.mean(meth_frac)) if meth_frac else 0.0
    print(f"wrote {n_out:,} reads (skipped {n_skip:,}) -> {args.output}", file=sys.stderr)
    print(f"mean per-read m6A fraction: {mf:.3f} (expect ~0.05-0.15 for EcoGII footprinting)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
