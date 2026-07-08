#!/usr/bin/env python3
"""Convert Dennis & Clark (NAR 2024, GSE243114) M.EcoGII PacBio m6A BAMs to modBAM.

Their AdenineFootprinter/MethyladenosineFinder output stores per-molecule m6A in an
unusual way: the read SEQ is empty, the read is aligned to the yeast reference, and
each methylated adenine is encoded as a 1-bp insertion (``1I``) in the CIGAR. The A
that is methylated is the reference base immediately BEFORE the insertion (verified:
100% of insertions fall on an A or T at ref_pos-1; T = the bottom-strand m6A, since
EcoGII methylates A on both strands).

This reconstructs a standard modBAM: SEQ = reference over the aligned span, with
MM/ML m6A calls at every A/T (methylated where a 1I was present, unmethylated
otherwise). We report BOTH methylated and unmethylated target calls so downstream
per-context methylation rates are correct. Output uses the fibertools two-strand
convention (``A+a`` at A, ``T+a`` at T) that FiberHMM's pacbio-fiber mode expects.

Usage:
    ecogii_cigar_to_modbam.py --ref R64.fa --in GSM..._m6A.bam --out out.modbam.bam
"""
import argparse
import array

import numpy as np
import pysam


def build_mm_ml(seq, prob):
    """(mm_string, ml_bytes) for m6A at A and T positions; NaN prob = not a target."""
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
            mm_parts.append(f"{base}+a," + ",".join(map(str, deltas)))
    mm = ";".join(mm_parts) + (";" if mm_parts else "")
    return mm, array.array("B", ml)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="Yeast R64 FASTA (chroms I,II,...)")
    ap.add_argument("--in", dest="inbam", required=True, help="input *_m6A.bam")
    ap.add_argument("--out", required=True, help="output modBAM")
    ap.add_argument("--min-length", type=int, default=200,
                    help="skip reconstructed reads shorter than this")
    args = ap.parse_args()

    ref = pysam.FastaFile(args.ref)
    src = pysam.AlignmentFile(args.inbam, check_sq=False)
    out = pysam.AlignmentFile(args.out, "wb", template=src)

    n_in = n_out = n_meth = 0
    for r in src:
        n_in += 1
        if r.reference_name is None or r.cigartuples is None:
            continue
        chrom = r.reference_name
        ref_start = r.reference_start
        refpos = ref_start
        m6a_ref = []
        for op, ln in r.cigartuples:
            if op == 0:        # M — consumes reference
                refpos += ln
            elif op in (2, 3):  # D / N — consumes reference
                refpos += ln
            elif op == 1:      # I — one m6A marker at the preceding base
                m6a_ref.append(refpos - 1)
        ref_end = refpos
        if ref_end - ref_start < args.min_length:
            continue
        seq = ref.fetch(chrom, ref_start, ref_end).upper()
        L = len(seq)
        meth = np.zeros(L, dtype=bool)
        for p in m6a_ref:
            i = p - ref_start
            if 0 <= i < L:
                meth[i] = True

        # prob: 1.0 methylated / 0.0 unmethylated at every A/T target, NaN elsewhere
        prob = np.full(L, np.nan)
        sb = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
        is_at = (sb == ord("A")) | (sb == ord("T"))
        prob[is_at] = 0.0
        prob[meth & is_at] = 1.0
        n_meth += int((meth & is_at).sum())

        mm, ml = build_mm_ml(seq, prob)

        a = pysam.AlignedSegment(out.header)
        a.query_name = r.query_name
        a.query_sequence = seq
        a.flag = 0
        a.reference_id = out.header.get_tid(chrom)
        a.reference_start = ref_start
        a.mapping_quality = 60
        a.cigartuples = [(0, L)]
        if mm:
            a.set_tag("MM", mm, "Z")
            a.set_tag("ML", ml)
        out.write(a)
        n_out += 1

    src.close()
    out.close()
    print(f"{args.inbam}: {n_in} reads -> {n_out} modBAM reads, {n_meth} m6A calls")


if __name__ == "__main__":
    main()
