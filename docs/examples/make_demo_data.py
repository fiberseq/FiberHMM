#!/usr/bin/env python3
"""Write a small synthetic data set for trying the FiberHMM commands.

    python make_demo_data.py demo/

writes, into ``demo/``:

    ref.fa (+ .fai)            one 30 kb contig, ``chrDemo``, with a CpG-rich block
    hia5_pacbio.bam            Hia5 Fiber-seq, PacBio-style MM/ML (A+a and T-a)
    hia5_nanopore.bam          Hia5 Fiber-seq, Nanopore-style MM/ML (A+a only)
    hia5_pacbio.unaligned.bam  the PacBio reads as an unaligned BAM (uBAM)
    hia5_pacbio.naked.bam      Hia5 PacBio reads of naked (all-accessible) DNA,
                               an accessible control for fiberhmm-probs
    dddb.bam                   DddB DAF-seq, aligned with MD tags
    ddda.bam                   DddA DAF-seq, aligned with MD tags; includes PCR
                               duplicates and both strands of some molecules

Every aligned BAM is coordinate-sorted and indexed. The molecules share a
phased nucleosome array and two transcription-factor sites (24 bp at 10,040
and 16 bp at 10,110, 0-based), so the calls have structure to find. The data
are synthetic: they exercise the commands, they are not a benchmark.
Needs only numpy and pysam (both FiberHMM dependencies).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pysam

CONTIG = "chrDemo"
LENGTH = 30_000
TF_SITES = ((10_040, 10_064, 0.7), (10_110, 10_126, 0.5))  # start, end, occupancy
COMPLEMENT = str.maketrans("ACGTRY", "TGCAYR")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def make_reference(rng) -> str:
    seq = rng.choice(list("ACGT"), size=LENGTH, p=[0.3, 0.2, 0.2, 0.3])
    # A CpG-island-like block for fiberhmm-tag-m5c.
    block = rng.choice(list("ACGT"), size=1_600, p=[0.15, 0.35, 0.35, 0.15])
    for i in range(0, len(block) - 1, 9):
        block[i], block[i + 1] = "C", "G"
    seq[20_000:21_600] = block
    return "".join(seq)


def protected_mask(rng, start: int, end: int) -> np.ndarray:
    """One molecule's protection over [start, end): nucleosomes and bound TFs."""
    mask = np.zeros(end - start, dtype=bool)
    pos = 60 + int(rng.integers(-15, 16))
    while pos < LENGTH:
        if not 9_990 <= pos <= 10_150 and rng.random() < 0.9:  # an NFR at the TF sites
            lo, hi = max(pos, start), min(pos + 147, end)
            if lo < hi:
                mask[lo - start:hi - start] = True
        pos += 147 + 45 + int(rng.integers(-10, 11))
    for tf_start, tf_end, occupancy in TF_SITES:
        if rng.random() < occupancy:
            lo, hi = max(tf_start, start), min(tf_end, end)
            if lo < hi:
                mask[lo - start:hi - start] = True
    return mask


def mm_tag(sequence: str, positions, base: str, strand: str, code: str) -> str:
    """One MM entry: skip counts over ``base`` in ``sequence`` (read orientation)."""
    index = {pos: i for i, pos in enumerate(p for p, b in enumerate(sequence) if b == base)}
    skips, previous = [], -1
    for pos in sorted(positions):
        skips.append(index[pos] - previous - 1)
        previous = index[pos]
    return f"{base}{strand}{code}." + "".join(f",{s}" for s in skips) + ";"


def m6a_read(rng, header, name, reference, start, end, platform, reverse, naked=False):
    """A Hia5 read: m6A (ML 250) on accessible A (and T on PacBio)."""
    ref_seq = reference[start:end]
    protected = (np.zeros(end - start, dtype=bool) if naked
                 else protected_mask(rng, start, end))
    rate = np.where(protected, 0.01, 0.7)
    original = revcomp(ref_seq) if reverse else ref_seq
    orig_rate = rate[::-1] if reverse else rate
    targets = ("A", "T") if platform == "pacbio" else ("A",)
    entries, ml = [], []
    for base in targets:
        hits = [i for i, b in enumerate(original) if b == base and rng.random() < orig_rate[i]]
        if platform == "nanopore":  # sub-threshold ONT calls, dropped at ML >= 248
            hits_low = [i for i, b in enumerate(original)
                        if b == base and i not in set(hits) and rng.random() < 0.05]
            calls = sorted([(i, 250) for i in hits] + [(i, 200) for i in hits_low])
            entries.append(mm_tag(original, [i for i, _ in calls], base, "+", "a"))
            ml.extend(q for _, q in calls)
        else:
            strand = "+" if base == "A" else "-"
            entries.append(mm_tag(original, hits, base, strand, "a"))
            ml.extend([250] * len(hits))
    read = pysam.AlignedSegment(header)
    read.query_name = name
    read.query_sequence = revcomp(original) if reverse else original
    read.flag = 16 if reverse else 0
    read.reference_id = 0
    read.reference_start = start
    read.mapping_quality = 60
    read.cigartuples = [(0, end - start)]
    read.query_qualities = pysam.qualitystring_to_array("I" * (end - start))
    read.set_tag("MM", "".join(entries))
    read.set_tag("ML", ml)
    return read


def daf_read(rng, header, name, reference, start, end, flavour, reverse, snps,
             protected=None, inside_rate=0.03):
    """A DAF read with C->T (CT) or G->A (GA) conversions recorded in MD."""
    ref_seq = reference[start:end]
    if protected is None:
        protected = protected_mask(rng, start, end)
    target, converted = ("C", "T") if flavour == "CT" else ("G", "A")
    query = list(ref_seq)
    for i, base in enumerate(ref_seq):
        if base == target and rng.random() < (inside_rate if protected[i] else 0.45):
            query[i] = converted
    for pos, alt in snps.items():  # A/T alleles of this molecule's haplotype
        if start <= pos < end:
            query[pos - start] = alt
    md, run = [], 0
    for ref_base, query_base in zip(ref_seq, query):
        if ref_base != query_base:
            md.append(f"{run}{ref_base}")
            run = 0
        else:
            run += 1
    md.append(str(run))
    read = pysam.AlignedSegment(header)
    read.query_name = name
    read.query_sequence = "".join(query)
    read.flag = 16 if reverse else 0
    read.reference_id = 0
    read.reference_start = start
    read.mapping_quality = 60
    read.cigartuples = [(0, end - start)]
    read.query_qualities = pysam.qualitystring_to_array("I" * (end - start))
    read.set_tag("MD", "".join(md))
    return read


def write_sorted(path, header, reads):
    unsorted = path + ".unsorted.bam"
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for read in reads:
            out.write(read)
    pysam.sort("-o", path, unsorted)
    os.remove(unsorted)
    pysam.index(path)


def spans(rng, n, lo=3_000, hi=7_000):
    """Read spans: half anywhere on the contig, half across the TF sites."""
    for i in range(n):
        length = int(rng.integers(lo, hi))
        if i % 2:
            start = int(rng.integers(0, LENGTH - length))
        else:
            start = 10_070 - length // 2 + int(rng.integers(-1_200, 1_200))
        yield start, start + length


def main(outdir: str) -> int:
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(20260929)
    reference = make_reference(rng)
    fasta = os.path.join(outdir, "ref.fa")
    with open(fasta, "w") as handle:
        handle.write(f">{CONTIG}\n")
        for i in range(0, LENGTH, 60):
            handle.write(reference[i:i + 60] + "\n")
    pysam.faidx(fasta)
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": CONTIG, "LN": LENGTH}],
    })

    for platform in ("pacbio", "nanopore"):
        reads = [m6a_read(rng, header, f"{platform}_{i:04d}", reference, s, e,
                          platform, reverse=bool(i % 2))
                 for i, (s, e) in enumerate(spans(rng, 300))]
        write_sorted(os.path.join(outdir, f"hia5_{platform}.bam"), header, reads)
        if platform == "pacbio":
            unaligned = pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6", "SO": "unknown"}})
            with pysam.AlignmentFile(os.path.join(outdir, "hia5_pacbio.unaligned.bam"),
                                     "wb", header=unaligned) as out:
                for read in reads:
                    record = pysam.AlignedSegment(unaligned)
                    record.query_name = read.query_name
                    reverse = read.is_reverse
                    record.query_sequence = (revcomp(read.query_sequence) if reverse
                                             else read.query_sequence)
                    record.query_qualities = read.query_qualities
                    record.flag = 4
                    record.set_tag("MM", read.get_tag("MM"))
                    record.set_tag("ML", list(read.get_tag("ML")))
                    out.write(record)

    naked = [m6a_read(rng, header, f"naked_{i:04d}", reference, s, e, "pacbio",
                      reverse=bool(i % 2), naked=True)
             for i, (s, e) in enumerate(spans(rng, 100))]
    write_sorted(os.path.join(outdir, "hia5_pacbio.naked.bam"), header, naked)

    # Two haplotypes that differ at reference A/T positions (for duplex pairing).
    at_sites = [i for i in range(100, LENGTH - 100, 150) if reference[i] in "AT"]
    haplotypes = [{}, {pos: ("T" if reference[pos] == "A" else "A") for pos in at_sites}]

    dddb = [daf_read(rng, header, f"dddb_{i:04d}", reference, s, e,
                     "CT" if i % 2 else "GA", reverse=bool(i % 3 == 0), snps={})
            for i, (s, e) in enumerate(spans(rng, 300))]
    write_sorted(os.path.join(outdir, "dddb.bam"), header, dddb)

    ddda = []
    for i, (s, e) in enumerate(spans(rng, 260)):
        snps = haplotypes[i % 2]
        protected = protected_mask(rng, s, e)
        if i % 4 == 0:  # both strands of one molecule sequenced (a duplex)
            for flavour in ("CT", "GA"):
                ddda.append(daf_read(rng, header, f"ddda_{i:04d}_{flavour}", reference, s, e,
                                     flavour, reverse=flavour == "GA", snps=snps,
                                     protected=protected, inside_rate=0.08))
            continue
        flavour = "CT" if i % 2 else "GA"
        read = daf_read(rng, header, f"ddda_{i:04d}", reference, s, e, flavour,
                        reverse=bool(i % 3 == 0), snps=snps, protected=protected,
                        inside_rate=0.08)
        ddda.append(read)
        if i % 10 == 1:  # PCR copies: same molecule, a few extra errors
            for copy in range(2):
                duplicate = pysam.AlignedSegment.from_dict(read.to_dict(), header)
                duplicate.query_name = f"ddda_{i:04d}_pcr{copy}"
                ddda.append(duplicate)
    write_sorted(os.path.join(outdir, "ddda.bam"), header, ddda)

    print(f"wrote demo data to {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "demo"))
