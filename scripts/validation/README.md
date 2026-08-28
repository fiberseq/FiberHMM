# DAF SNP threshold validation

`validate_daf_snp_downsampling.py` calibrates the optional FiberHMM DAF SNP
mask by thinning bidirectionally classified fiber evidence over a prespecified
coverage ladder. It writes compact, paper-ready TSVs, a run manifest with input
and caller hashes, and PDF/SVG/PNG versions of the validation figure.

The script does not contain dataset paths. Supply each input explicitly as
`LABEL=COHORT=/path/to/input.bam`. Large evidence caches stay outside the paper
tree.

The full-depth "truth" label is deliberately limited to sites that independently
have at least 15 fibers, at least 15 mismatching fibers, and at least 80%
mismatch in **both** DAF direction classes. It is an unambiguous evidence class,
not a claim of orthogonal genotype truth. Likewise, background-site calls and
discordant calls are not labeled false positives without an independent
genotype assay.

Example:

```bash
PYTHONPATH=. python scripts/validation/validate_daf_snp_downsampling.py \
  --dataset sample_a=cohort_a=/data/sample_a.bam \
  --dataset sample_b=cohort_b=/data/sample_b.bam \
  --output-dir /paper/analysis/daf_snp_downsampling \
  --cache-dir /scratch/fiberhmm_daf_snp_validation_cache \
  --replicates 100 --seed 20260824
```

The production rule represented by a policy is fully conjunctive in each
direction: minimum depth **and** minimum mismatching-fiber count **and** minimum
mismatch fraction must all pass in both direction classes.
