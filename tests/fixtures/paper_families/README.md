# Archived paper regression fixtures

These small fixtures preserve real native opportunities, observations and model
emissions from the saved paper analyses. They are not synthetic biological data.
No BAM/FASTA or full datasets are required to run the regression tests.

- GATA1::TAL1: archived compact ~12.02 bp and composite ~33.05 bp models.
- CTCF: ~20.82 bp and ~35.28 bp frozen models; eight libraries across K562,
  GM12878, H1, H9 and HepG2.
- Pause: the consolidated 37.84 bp model at downstream PRO-seq landmarks.
- Upstream pause: the same model at opposite-strand upstream landmarks.
- TATA: archived ~15–18 bp models.
- CAGE: archived TSS-overlapping ~61–76 bp models.

There are 22 selected windows, including both genomic orientations. Each
expected_*.json retains the original site metadata, original source/result file
hashes, and selected archived per-call scores. models.json.gz includes the source
model archive hash and training-molecule exclusions. Evidence is reduced to the
selected molecules but retains each molecule's original calls and native arrays;
the frozen source model is not fitted again on this reduction.

The public fiberhmm-transfer command must reproduce archived decisions,
simulation counts, predictive-tail intervals (absolute tolerance 1e-12) and
original genomic spans. Selection deliberately includes compatible, rejected
and unassessed examples. These are regression tests, not unbiased estimates of
accuracy or a repeat of full population discovery. The separate public CLI test
runs actual staged discovery -> model export -> frozen transfer on synthetic
observations, and actual BAM/BED transport on both orientations.

Archived data originally used the analysis coordinates [50,350] with anchor 200.
Fixtures declare that origin explicitly and preserve the corresponding genomic
windows. Original absolute archive paths in provenance are informational; tests
use only the files in this directory.
