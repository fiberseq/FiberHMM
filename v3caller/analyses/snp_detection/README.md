# SNP detection from amplicon data

## Question

On high-coverage amplicon data, any reference position where ~100%
of reads show a "deamination" is actually a SNP (sample genotype
differs from reference), not real enzyme activity. These
positions break nucleosome calls and inflate TF scores. How do
we detect and exclude them before calling?

## Methodology

1. Pileup per-reference-position hit count and opp count across
   all reads in the BAM.
2. For each position with coverage ≥ `min_coverage` (default 10),
   compute hit fraction = hits / opps.
3. Flag as SNP if fraction ≥ `min_fraction` (default 0.95).
4. Output as a BED file.

The caller uses the mask via `--snp-mask snps.bed`, which zeros
out opp and hit at those positions before calling. The caller
treats them as if the positions had no opportunity.

Works for both DAF (C→T, G→A, or IUPAC R/Y) and Hia5 (m6A via
MM/ML). Same 95% threshold works for both because the highest
real hit rates (accessible, fully-deaminated) top out around
50-60% — no real biological signal produces 95%+ rates.

## Reference results (2026-04-12)

On DddB spacetime 4-4.5 hr window (5000 reads, raw C→T):

| chrom | position | fraction | coverage |
|---|---|---|---|
| chr2L | 15475574 | 0.998 | 3165 |
| chr2L | 15484928 | 0.987 | 2857 |
| chr2L | 15485283 | 0.997 | 2905 |
| chr2L | 15485830 | 0.983 | 2936 |
| chr2L | 15486210 | 0.994 | 2940 |

5 homozygous SNPs at ≥98% fraction / ~3000× coverage. Full list
in `data/dddb_4-4.5_snps.bed`.

On haplotype-corrected NAPA: 0 SNPs (already pre-processed).
On ENH30 amplicon: max hit fraction 47.7%, no SNPs flagged
(expected — amplicon designs avoid common polymorphisms).

## Caveats

- **Coverage threshold matters**: low-coverage positions (< 10
  reads) are unreliable. With only 10 reads, hit fraction 1.0
  could easily be noise.
- **Heterozygous SNPs** (~50% fraction) are NOT caught by the
  default 95% threshold. For het SNPs in a diploid sample, you'd
  want a secondary filter at ~40-60% — but this overlaps with
  real biological accessibility signals and is harder to separate.
- **Systematic errors** at very high rates (rare) could be
  misflagged as SNPs. Unlikely in practice because sequencing
  errors rarely reach 95% at a specific position.

## How to reproduce

```bash
python scripts/snp_mask.py \
  --in-bam my_amplicon.bam \
  --out-bed my_snps.bed \
  --enzyme daf \
  --min-fraction 0.95 \
  --min-coverage 10

python caller_v8.py \
  --in-bam my_amplicon.bam \
  --out-bam my_called.bam \
  --enzyme daf \
  --snp-mask my_snps.bed
```
