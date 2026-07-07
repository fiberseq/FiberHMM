#!/usr/bin/env python3
"""
samosa_precalled_to_modbam.py — reconstruct an aligned m6A modBAM from SAMOSA
pre-called data, WITHOUT the raw PacBio signal (no ccs, no pod5, no basecalling).

SAMOSA (Ramani, GSE162410 / Zenodo 4289173) ships two matched files per sample:
  *_bingmm.pickle  : dict{ZMW -> float32[molecule_len]} = per-position P(m6A),
                     NaN at non-callable (non-A/T) bases.
  *_zmwinfo.pickle : DataFrame with genomic alignment per ZMW
                     (zmw, chr, refStart, refEnd, cclen, ...).

The literal read sequence is the only missing piece, and it is recoverable: the
molecule maps to chr:refStart-refEnd, so for GAPLESS molecules (|cclen-refspan|<=tol)
the reference sequence over that span IS the read sequence, 1:1. We then place
standard MM/ML m6A calls (A+a at A, T+a at T = both strands, as pacbio-fiber expects)
using the bingmm probabilities, and emit an aligned modBAM.

Output feeds FiberHMM's existing `pacbio-fiber` mode directly. We keep SAMOSA's own
validated calls, so treated/naked controls are preserved.

Strand note: zmwinfo has no strand column; we assume forward. This is harmless for
FiberHMM training — footprint run-lengths are orientation-invariant and pacbio-fiber
RC-normalizes context. (A minority of reads flipped only adds symmetric noise.)

Usage:
  samosa_precalled_to_modbam.py --bingmm S.bingmm.pickle --zmwinfo S.zmwinfo.pickle \
      --ref hg38.fa -o S.m6a.bam [--tol 2] [--min-prob-bit 0]
"""
import argparse
import array
import pickle
import sys

import numpy as np
import pysam


def build_mm_ml(seq, prob):
    """Return (mm_string, ml_bytes) for m6A at A and T positions.

    seq: str reference/read sequence (uppercase). prob: float array, len==len(seq),
    NaN where not callable. Lists every target-base occurrence that has a finite prob.
    """
    mm_parts = []
    ml = []
    for base in ("A", "T"):
        deltas = []
        skip = 0
        for i, b in enumerate(seq):
            if b != base:
                continue
            p = prob[i]
            if np.isfinite(p):
                deltas.append(skip)
                v = int(round(float(p) * 255))
                ml.append(0 if v < 0 else 255 if v > 255 else v)
                skip = 0
            else:
                skip += 1
        if deltas:
            # base + strand '+' + code 'a' (6mA), explicit calls
            mm_parts.append(f"{base}+a," + ",".join(map(str, deltas)))
    mm = ";".join(mm_parts) + (";" if mm_parts else "")
    return mm, array.array("B", ml)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bingmm", required=True)
    ap.add_argument("--zmwinfo", required=True)
    ap.add_argument("--ref", required=True, help="reference FASTA (indexed .fai)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--tol", type=int, default=2, help="max |cclen-refspan| to accept as gapless")
    args = ap.parse_args()

    def load_pickle(path):
        # handles plain or gzipped (RASAM ships *.pickle.gz); some preps need latin1
        import gzip
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as fh:
            try:
                return pickle.load(fh)
            except UnicodeDecodeError:
                fh.seek(0)
                return pickle.load(fh, encoding="latin1")

    bg = load_pickle(args.bingmm)
    zi = load_pickle(args.zmwinfo).set_index("zmw")
    fa = pysam.FastaFile(args.ref)
    ref_names = set(fa.references)

    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": n, "LN": fa.get_reference_length(n)} for n in fa.references]}
    hdr = pysam.AlignmentHeader.from_dict(header)

    n_written = n_skip_nogap = n_skip_meta = n_skip_len = n_skip_chrom = 0
    recs = []
    for zmw, prob in bg.items():
        if zmw not in zi.index:
            n_skip_meta += 1; continue
        row = zi.loc[zmw]
        chrom = str(row["chr"]); rs = int(row["refStart"]); re = int(row["refEnd"])
        if chrom not in ref_names:
            n_skip_chrom += 1; continue
        if abs(int(row["cclen"]) - (re - rs)) > args.tol:
            n_skip_nogap += 1; continue
        seq = fa.fetch(chrom, rs, re).upper()
        if len(seq) != len(prob):
            n_skip_len += 1; continue
        mm, ml = build_mm_ml(seq, prob)
        a = pysam.AlignedSegment(header=hdr)
        a.query_name = f"samosa_zmw{zmw}"
        a.flag = 0
        a.reference_name = chrom
        a.reference_start = rs
        a.mapping_quality = 60
        a.cigartuples = [(0, len(seq))]  # gapless: all match
        a.query_sequence = seq
        a.query_qualities = None
        if mm:
            a.set_tag("MM", mm, "Z")
            a.set_tag("ML", ml)
        recs.append((chrom, rs, a))
        n_written += 1

    # coordinate sort by (ref order, pos)
    order = {n: i for i, n in enumerate(fa.references)}
    recs.sort(key=lambda t: (order[t[0]], t[1]))
    with pysam.AlignmentFile(args.output, "wb", header=header) as out:
        for _, _, a in recs:
            out.write(a)
    pysam.index(args.output)

    print(f"wrote {n_written:,} reads -> {args.output}", file=sys.stderr)
    print(f"skipped: non-gapless={n_skip_nogap:,} len-mismatch={n_skip_len:,} "
          f"no-meta={n_skip_meta:,} chrom-absent={n_skip_chrom:,}", file=sys.stderr)


if __name__ == "__main__":
    main()
