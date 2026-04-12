# v2 HMM vs v3 caller comparison

## Question

How do the v3 Poisson-merge + rotational-correction + FP-aware calls
differ from the v2 HMM caller on the same datasets? Is v3 catching
distinct biology or just re-drawing the same regions?

## Reference datasets

### DddB spacetime (Drosophila, 7 time windows)

v2 output: `Drosophila_phase2/Datasets/DAF-seq/spacetime/fp/fiberhmm_*.m6a_footprints.bam`
v3 output: `Drosophila_phase2/Datasets/DAF-seq/spacetime/combined_bam/iter17_calls/*.called.bam`

Across all 7 windows (2026-04-12):

| metric | v2 (HMM) | v3 (iter-17) |
|---|---|---|
| Reads called | ~242k | ~297k (+23%) |
| Nucs (≥90 bp) / read | 10-13 | 18-22 (~2×) |
| Median nuc size | 169-238 bp | 202-215 bp |
| Mean nuc size | 641 bp | 282 bp |
| ≥300 bp (overmerge) | 37.1% | 23.0% |
| TFs / read (explicit) | n/a (2.1 as <90bp nucs) | 4-5 |

**Key finding**: v2 HMM was badly overmerging DddB — 37% of its
"nucleosomes" exceeded 300 bp (dinucleosome+ mega-calls), mean 641
bp. v3 cuts overmerge to 23%, doubles the nuc count per read,
and adds explicit TF footprint calls.

### Hia5 (fly embryo, sna/eve/ftz loci)

Input: `data/test_hia5_2-4hr_sna_eve_ftz.bam`
v3 output: `/tmp/hia5_sna_eve_ftz_iter17.bam` (re-generate as needed)

| metric | v2 (HMM) | v3 (iter-17) |
|---|---|---|
| Nucs (≥90 bp) / read | 64.6 | 87.0 |
| Median nuc size | 165 bp | 169 bp |
| ≥300 bp (overmerge) | 24.1% | 3.9% |
| ≥500 bp | 5.7% | 0.5% |
| TFs (explicit) / read | n/a (29.9 as <90bp nucs) | 133 |

**Key finding**: v2 HMM's overmerge on Hia5 drops 6× in v3 (24%
→ 3.9%). The v2 was emitting many <90 bp short "nucs" that v3
correctly identifies as TF footprints (separate tag track).

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
# Re-call Hia5
python ../../caller_v8.py \
  --in-bam data/test_hia5_2-4hr_sna_eve_ftz.bam \
  --out-bam /tmp/hia5_v3.bam \
  --fa '' --enzyme hia5 --tags both

# Re-call DddB (all 7 time windows) — see
# spacetime/combined_bam/iter17_calls/ for latest outputs.
# Uses --max-merge-len 0 (DddB: no merge) OR
# --fp-model fp_models/ct_nanopore_fp_3mer.json (unified path)
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
