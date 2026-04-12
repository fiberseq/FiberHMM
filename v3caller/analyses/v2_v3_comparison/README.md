# v2 HMM vs v3 caller comparison

## Question

How do the v3 Poisson-merge + rotational-correction + FP-aware calls
differ from the v2 HMM caller on the same datasets? Is v3 catching
distinct biology or just re-drawing the same regions?

## Reference datasets

### DddB spacetime (Drosophila, 7 time windows)

v2 output: `Drosophila_phase2/Datasets/DAF-seq/spacetime/fp/fiberhmm_*.m6a_footprints.bam`
v3 output: `Drosophila_phase2/Datasets/DAF-seq/spacetime/combined_bam/iter17_calls/*.called.bam`

Across all 7 windows (2026-04-12, **v3 with FP model + SNP mask**):

| metric | v2 (HMM) | v3 (iter-17 + FP) |
|---|---|---|
| Total reads (passthrough) | 336k | 336k |
| Reads with calls | 242k | 169k |
| Nucs (≥90 bp) / read | 9.7 mean / 5 med | 23.4 mean / 21 med |
| Median nuc size | 249 bp | 194 bp |
| Mean nuc size | 752 bp | 237 bp |
| ≥300 bp (overmerge) | 44.1% | 15.9% |
| ≥500 bp | 26.8% | 5.2% |
| TFs / read (explicit) | n/a (1.8 as <90bp nucs) | 8.1 |

**Key finding**: v2 HMM massively overmerged DddB — 44% of its
"nucleosomes" were ≥300 bp and mean size was 752 bp (implying many
dinucleosome/chromatosome-scale mega-calls). v3 with FP correction
cuts overmerge to 16%, drops mean nuc size to 237 bp (right at
Drosophila NRL), doubles the nuc count per read, and adds explicit
TF footprint calls.

The **FP + SNP correction** (new in this run) tightened the v3
numbers vs a prior FP-free v3 run: mean nuc size went from 282 → 237
bp, overmerge from 23.0% → 15.9%. The context-aware Poisson model
correctly withholds merges at CpG-adjacent sites where Nanopore FP
is elevated.

Fewer v3 reads have calls (169k vs 242k v2) because v3 rejects reads
with `min_read_rate < 0.05` (v2 was permissive) — the missing 73k
reads are low-signal and would have added noise.

### Hia5 (fly embryo, sna/eve/ftz loci)

Input: `data/test_hia5_2-4hr_sna_eve_ftz.bam`
v3 output: `/tmp/hia5_sna_eve_ftz_iter17.bam` (re-generate as needed)

| metric | v2 (HMM) | v3 (iter-17 + FP) |
|---|---|---|
| Reads tagged | 1225 | 1033 |
| Nucs (≥90 bp) / read | 64.5 mean / 63 med | 88.9 mean / 84 med |
| Median nuc size | 165 bp | 162 bp |
| Mean nuc size | 260 bp | 175 bp |
| ≥300 bp (overmerge) | 24.2% | 3.8% |
| ≥500 bp | 8.3% | 0.5% |
| TFs (explicit) / read | n/a (29.8 as <90bp nucs) | 146.2 |

**Key finding**: v2 HMM's overmerge on Hia5 drops **6.4× in v3**
(24.2% → 3.8%). Mean nuc size drops 260 → 175 bp, right at
Drosophila mono-nuc protected length. v3 also promotes what v2
called "short nucs" (29.8 per read) into explicit TFs (146.2 per
read), reflecting the rich m6A-detected regulatory landscape that
v2 was conflating with nucleosomes.

## Comparison methodology

v2 stores everything (nucs + TFs) in a single `ns/nl` tag. To
separate: entries with `nl < 90 bp` are TF-scale footprints,
entries with `nl ≥ 90 bp` are nucleosome-scale. v3 uses separate
tag tracks (`ns/nl` for nucs, `tn/tl` for TFs) so the separation
is explicit.

Shortcut for v2 comparison:
```python
ns = read.get_tag('ns')
nl = read.get_tag('nl')
v2_nucs = [(s, l) for s, l in zip(ns, nl) if l >= 90]
v2_tfs  = [(s, l) for s, l in zip(ns, nl) if l <  90]
```

For v3, use the separate tags directly.

## How to reproduce

```bash
# Re-call Hia5 (PacBio, fp model required)
python phase0/caller_v8.py \
  --in-bam phase0/data/test_hia5_2-4hr_sna_eve_ftz.bam \
  --out-bam /tmp/hia5_sna_eve_ftz_iter17.bam \
  --fa '' --enzyme hia5 \
  --fp-model phase0/data/fp_models/m6a_pacbio_fp_3mer.json \
  --penetration-fraction 0.0

# Re-call DddB (Nanopore, fp model + SNP mask required)
for W in 1-1.5 1.5-2 2-2.5 2.5-3 3-3.5 3.5-4 4-4.5; do
  python phase0/caller_v8.py \
    --in-bam .../spacetime/combined_bam/${W}.sorted.bam \
    --out-bam .../iter17_calls/${W}.called.bam \
    --fa '' --enzyme daf \
    --fp-model phase0/data/fp_models/ct_nanopore_fp_3mer.json \
    --snp-mask ../snp_detection/data/dddb_4-4.5_snps.bed \
    --penetration-fraction 0.0
done

# Regenerate comparison stats
python scripts/compare_v2_v3.py --version v3 --label dddb_v3 \
  --in-bam .../iter17_calls/*.called.bam
python scripts/compare_v2_v3.py --version v2 --label dddb_v2 \
  --in-bam .../spacetime/fp/fiberhmm_*.m6a_footprints.bam
```

## Caveats

- **Read-count differences**: v3 calls 23% more DddB reads
  (297k vs 242k) because the updated DAFExtractor handles raw
  C→T directly rather than requiring IUPAC encoding. Hia5 v3
  calls slightly fewer (1033 vs 1227) due to stricter
  `min_read_rate=0.05` vs v2's permissive threshold.
- **Overmerge threshold (≥300 bp)** is arbitrary but meaningful:
  Drosophila nucleosome repeat length is ~180 bp, so a "nuc"
  call >300 bp implies a di-nucleosome or chromatosome, not a
  simple mono-nuc. 23-28% on DddB still seems high but is often
  biologically real (heterochromatin has longer protected
  stretches). Contrast with Hia5 at 3.9% where overmerge is
  genuinely rare.
