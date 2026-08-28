# FiberHMM QC control provenance

FiberHMM packages aggregate control curves, compact scoring summaries, and a
small set of anonymized visual molecule exemplars. It does **not** package
control BAMs, sequences, read names, genomic positions, barcodes, or source
per-read tables. `control_curves.json` explicitly records
`contains_individual_read_data: false` and contains only:

- 99 ECDF quantile knots for the per-read signal-rate reference; and
- a 741-point aggregate phasogram spanning 60–800 bp; and
- normalized aggregate nucleosome/TF footprint-size histograms, medians, and
  quartiles where the source control contains those tags.

`control_examples.json` contains 8–12 rate-stratified visual exemplars per
source control. Each exemplar retains only molecule span, signal rate, and
signal hatch positions relative to the molecule start. Read identities,
sequences, chromosomes, and genomic coordinates are stripped. These records
exist solely to support the sample-versus-control molecule panel; they are not
a substitute for, or redistribution of, the source datasets.

## Control sources

| Profile | Control | Packaged sample | Signal threshold |
|---|---|---:|---:|
| `dddb` | Drosophila Spacetime WT 2–4 h DddB, `yw_2-4.sorted.bam` | 2,000 reads | encoded/raw DAF calls; ML 125 when applicable |
| `ddda` | Human NAPA and UBA1 DddA DAF-seq, `206_NAPA` + `206_UBA1` | 2,000 reads each | encoded/raw DAF calls; ML 125 when applicable |
| `hia5_pacbio` | Drosophila embryo PacBio Fiber-seq 2–4 h, `2-4hr_14` chr2R | 2,000 reads | ML ≥125 |
| `hia5_nanopore` | Drosophila embryo ONT Fiber-seq zld 1.5–3 h | 2,000 reads | ML ≥248 |

Footprint-size overlays use the same fixed-seed bounded sampling policy, but
necessarily use footprint-called controls. The DddB footprint distribution is
from the latest high-quality single-embryo DddB call
`SE_DAF_20260823_O95_I2.fiberhmm.bam`; its rate and phasogram references remain
the independent Spacetime WT control above. DddA sizes come from the
footprint-called `206_NAPA` and `206_UBA1` controls. PacBio sizes come from
`2-4hr_14.aligned_footprints.chr2R.bam`, and Nanopore sizes from
`zld_1.5-3hr.fiberhmm.thr248.bam`. The legacy DddA and PacBio files carry
nucleosome tags but no distinct TF tags, so those two profiles deliberately do
not claim a TF-size reference.

These datasets remain external controls and are not redistributed with
FiberHMM. The identifiers above document biological and technical provenance;
the source-accounting entries in `control_curves.json` document only aggregate
sample sizes, bounded records examined, and sampling strategy.

## Curve construction

All curves use seed `20260824`, primary alignments with MAPQ ≥20, and the same
bounded sampler used by `fiberhmm-qc`. Indexed BAMs are sampled through 168
seeded 20-kb genomic windows, with a capped reservoir fill only when a sparse
index does not yield 2,000 reads. No reference dataset is scanned in full by
this procedure.

Per-read rates require at least 200 assay opportunities. Raw/aligned DAF rates
use MD- or FASTA-derived aligned reference C/G positions and C→T/G→A events;
PacBio Hia5 uses A+T and ONT Hia5 uses A. The packaged ECDF is evaluated at
probabilities 0.01–0.99.

For periodicity, all within-read signal-pair distances through 1,000 bp are
counted. The pair histogram is divided by a 121-bp moving baseline. The
packaged plotting curve retains lags 60–800 bp, while NRL and autocorrelation
strength are evaluated over the 160–220-bp nucleosome-scale search interval.
Signal-rate grades use the control IQR as the PASS interval and the empirical
5th–95th percentiles as the outer WARN interval; both are drawn in individual
and combined reports. Periodicity grades require concordant peak strength,
NRL, reference-curve correlation, and matched reference-pattern amplitude.
The last quantity is a mean-centered projection coefficient: 1.0 means the
sample contains the control-shaped oscillation at control amplitude, while a
near-flat sample approaches zero even if its correlation is spuriously high.

`references.json` stores the compact PASS/WARN thresholds and reference
summaries. `control_curves.json` stores the aggregate curves and size
distributions used as plot overlays. Automatic `fiberhmm-call` QC chooses
exactly one profile from the resolved enzyme/platform combination; it never
chooses a reference by sample filename. Endpoint-constrained
deamination-fingerprint deduplication is run only when explicitly requested by
`fiberhmm-call --dedup`; integrated dedup marks and retains all reads unless
destructive collapse is requested. Marked PCR copies are excluded from pooled
SNP, phase, and QC signal calculations, and the exact full-run duplication
summary is forwarded into automatic QC.

Automatic file-based DddA/DddB SNP calling adds two diagnostic panels when a
bounded local-depth preflight reaches the configured threshold. Low-coverage
samples skip the full caller; `--daf-call-snps` and `--no-daf-call-snps` force
or disable it. The mismatch landscape
plots every retained background C/G position as expected-direction mismatch
percentage versus opposite-direction mismatch percentage, colors points by
combined dominant-fiber depth, and outlines called recurrent SNPs. A companion
amplicon-discovery map draws every alignment-supported amplicon consensus above
the configured read-coverage threshold, labels its genomic interval and total
coverage, and places C→T/G→A SNPs at their relative positions along that
consensus. The matching `.amplicons.tsv` makes the discovered interval,
coverage, length, and relative SNP positions available outside the plot.
The caller retains at most 5,000 deterministic background positions plus all
candidate sites, keeping output and memory bounded while showing the typical
site distribution.

## Figure formats

Every per-sample and combined QC figure is written as both a 200-dpi PNG for
quick viewing and a vector PDF for publication assembly. The PDF backend embeds
TrueType fonts (`pdf.fonttype = 42`), so Illustrator retains labels, legends,
axis text, and table text as selectable text instead of glyph outlines. The
PDF and PNG use the same output prefix and contain the same panels.
