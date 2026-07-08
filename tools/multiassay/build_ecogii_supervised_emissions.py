#!/usr/bin/env python3
"""Supervised EcoGII emissions from the Clark yeast chromatin sample.

Uses their own per-read nucleosome BED to label each A/T target as accessible
(outside nucleosomes) or protected (inside), then tallies per-context m6A rate for
each state. This gives an honest nonzero background (~0.035) instead of the bogus
no-enzyme mock (0.0002). Codes come straight from encode_from_query_sequence so the
emission table lines up with the apply-time encoder. Writes accessible/inaccessible
TSVs ready for `fiberhmm-train --base-model hia5_pacbio.json`.
"""
import gzip
import sys

import numpy as np
import pysam

sys.path.insert(0, "/mnt/g/Dropbox/Fiber-NET-seq/FiberHMM v1.0/Release v2.0.0")
from fiberhmm.core.bam_reader import (encode_from_query_sequence,
                                      get_modified_positions_pysam, ContextEncoder)

CHROM_BAM = "/mnt/z/fiberhmm_corpus/ecogii_yeast/chromatin_rep1.bam"
NUC_BED = "/mnt/z/fiberhmm_corpus/ecogii_yeast/GSM7779503_Nuclei_MEcoGII_Rep1_Nucleosome.bed.gz"
OUT = "/mnt/z/fiberhmm_corpus/ecogii_yeast/yeast_emis"
K = 3
NC = 4 ** (2 * K)
MAXREADS = 60000

# per-read nucleosome intervals (ref coords)
nuc = {}
with gzip.open(NUC_BED, "rt") as fh:
    next(fh)
    for line in fh:
        p = line.rstrip("\n").split("\t")
        if len(p) < 12:
            continue
        rid = p[3]; start = int(p[1])
        sizes = [int(x) for x in p[10].rstrip(",").split(",")]
        offs = [int(x) for x in p[11].rstrip(",").split(",")]
        nuc.setdefault(rid, []).extend((start + o, start + o + s) for o, s in zip(offs, sizes))

# per-context [hit, nohit] for accessible and protected
acc = np.zeros((NC, 2), dtype=np.int64)
pro = np.zeros((NC, 2), dtype=np.int64)

b = pysam.AlignmentFile(CHROM_BAM)
done = 0
for r in b:
    if r.query_name not in nuc or r.reference_start is None:
        continue
    ivs = nuc[r.query_name]
    rs = r.reference_start
    # protected mask in query coords (reconstructed reads are 1:1 M, so query==ref-rs)
    L = r.query_length
    prot = np.zeros(L, dtype=bool)
    for a, z in ivs:
        lo = max(0, a - rs); hi = min(L, z - rs)
        if hi > lo:
            prot[lo:hi] = True
    modpos = get_modified_positions_pysam(r, 128, "pacbio-fiber")
    obs = encode_from_query_sequence(r.query_sequence, modpos, edge_trim=5,
                                     mode="pacbio-fiber", context_size=K)
    obs = np.asarray(obs)
    # methylated target: obs < NC ; unmeth target: NC+1 <= obs < 2*NC+1
    meth = obs < NC
    unmeth = (obs >= NC + 1) & (obs < 2 * NC + 1)
    ctx_meth = obs[meth]
    ctx_un = obs[unmeth] - (NC + 1)
    idx = np.arange(L)
    for i in idx[meth]:
        c = obs[i]
        (pro if prot[i] else acc)[c, 0] += 1
    for i in idx[unmeth]:
        c = obs[i] - (NC + 1)
        (pro if prot[i] else acc)[c, 1] += 1
    done += 1
    if done >= MAXREADS:
        break

def write_tsv(counts, path, minobs=20):
    rows = []
    for code in range(NC):
        hit, nohit = counts[code]
        tot = hit + nohit
        if tot < minobs:
            continue
        rows.append((code, hit, nohit, hit / tot))
    with open(path, "w") as fh:
        fh.write("encode\tcontext\thit\tnohit\tratio\n")
        for code, hit, nohit, ratio in rows:
            fh.write(f"{code}\t.\t{hit}\t{nohit}\t{ratio:.6f}\n")
    return len(rows), np.mean([r[3] for r in rows]) if rows else 0

na, ma = write_tsv(acc, f"{OUT}_accessible_k3.tsv")
ni, mi = write_tsv(pro, f"{OUT}_inaccessible_k3.tsv")
print(f"reads used: {done}")
print(f"accessible: {na} contexts, mean rate={ma:.3f}")
print(f"inaccessible(protected): {ni} contexts, mean rate={mi:.3f}")
print(f"contrast ~{ma/mi:.1f}x" if mi else "no bg")
