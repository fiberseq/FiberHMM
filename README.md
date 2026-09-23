# FiberHMM

Hidden Markov Model toolkit for calling chromatin footprints from Fiber-seq,
DAF-seq, and other single-molecule footprinting data.

FiberHMM identifies protected regions (nucleosomes, TF/Pol II footprints) and
accessible regions (methylase-sensitive patches, MSPs) from single-molecule DNA
modification data — m6A methylation (fiber-seq) and deamination marks (DAF-seq).

> **Development snapshot for the preprint-scale FiberHMM/FiberBrowser update.**
> No release number will be assigned before the caller, models, browser, paper
> and preprint package pass final integration. File-based DddA/DddB calls
> mark PCR duplicates automatically and nondestructively, screen adequately
> covered samples for recurrent C→T/G→A SNPs after duplicate marking, and run
> bounded assay-matched QC after footprint calling. `fiberhmm-qc` also accepts
> multiple BAMs for per-sample panels plus a combined comparison report. Use
> `--no-dedup`, `--no-daf-call-snps`, or `--no-qc` to disable an automatic
> stage; only `--dedup-collapse` removes duplicate reads.

- [Installation](#installation)
- [Quick start](#quick-start)
- [Choosing a command](#choosing-a-command)
- [Workflows](#workflows)
- [Command reference](#command-reference)
- [Output tags](#output-tags)
- [Pre-trained models](#pre-trained-models)
- [Performance tips](#performance-tips)
- [Deep reference](docs/reference.md) — MA/AQ schema, LLR scoring model, tag glossary
- [DddA CpG-island methylation](#ddda-cpg-island-methylation)

## Key features

- **`fiberhmm-call`** — recommended one-command pipeline: nucleosome/MSP HMM +
  nucleosome recall + TF recall fused in one process, with region-parallel
  scaling. Coordinate-sorted input → sorted + indexed output, no separate sort.
- **Nucleosome recaller (on by default)** — splits over-merged nucleosomes on
  accessible evidence, refines edges, runs an evidence-gated periodicity prior.
- **DAF duplicate marking + SNP masking** — file-based DddA/DddB calls
  automatically mark endpoint-concordant deamination-fingerprint duplicates,
  retain every read, then mask well-supported recurrent SNPs from HMM
  observations. The original sequence and `MD` tag are preserved.
- **Bounded QC (on by default after `fiberhmm-call`)** — prints an assay-aware
  PASS/WARN/FAIL scorecard and writes deamination/m6A-rate, nucleosome-scale
  periodicity, and conditional nuc/TF footprint-size plots. Indexed BAMs are
  sampled through deterministic random genomic windows, so QC never rescans a
  whole multi-terabyte callset. The same workflow is available as
  `fiberhmm-qc`; it never performs deduplication.
- **`fiberhmm-dedup`** — endpoint-constrained deamination-fingerprint PCR
  duplicate detection for DAF-seq, including amplicons where ordinary
  coordinate-only dedup cannot distinguish molecules.
- **`fiberhmm-footprint-model`** — learns footprint population models and
  overlapping TF-binding hypotheses directly from ordinary per-read TF calls.
  It reports overall and MSP-conditioned occupancy, writes portable analysis
  tables, and creates an identity-preserving per-read FiberBrowser overlay.
  Raw calls remain unchanged. See the
  [model and BAM contract](docs/footprint-models.md) and the
  [implementation/OCT handoff](docs/FOOTPRINT_POPULATION_MODEL_HANDOFF.md).
- **`fiberhmm-strand-rescue`** — focal, report-only normalization for stranded
  DddA, DddB, and Nanopore Hia5. It learns strand-balanced TF and nucleosome
  geometry populations, uses opposite-strand occupancy to promote
  weak-but-positive TFs only from MSPs, and proposes shared edges for accepted
  TF and nucleosome calls. Nucleosomes are never occupancy candidates and
  cannot be promoted, demoted, split, merged, or dropped; edge normalization is
  one-for-one and has no nucleosome-length ceiling. Repeated BAMs are one
  explicitly pooled same-assay cohort; independent assays are validation-only.
  Standard sequence, hard MM/ML (or DAF mismatches), and existing MA calls are
  sufficient.
- **`fiberhmm-strand-rescue-annotate`** — writes new indexed regional BAMs with
  complete `nuc_sr.QQQ` and `tf_sr.QQQ` shadow layers containing baseline,
  rescued, and edge-normalized calls. For every named rescue (`R`) or same-class
  edge alternative (`H`), `q0` is the probability of choosing SR over that
  call's ordinary baseline; `q1` and `q2` describe its molecular-left and
  molecular-right edges. A single FiberBrowser threshold can therefore switch
  each named alternative against its baseline. Ordinary nuc/TF annotations and
  input BAMs stay unchanged.
  `fiberhmm-strand-rescue-audit` validates MA/AQ/AN alignment, atomic multi-TF
  groups, geometry roles, scores, header contract, and indexing. See the
  [model/CLI contract](docs/strand_rescue.md).
- **`fiberhmm-tag-consensus`** — materializes a frozen
  site-consensus footprint-state assignment
  table entirely inside the BAM by extending `tf_sr.QQQ` to `tf_sr.QQQQQ`.
  The added `fi` byte is a locally reusable consensus-state slot (`0` = unassigned), and
  `fq` is the producer-declared call-to-state assignment confidence on a
  0–255 scale (not a biological occupancy posterior). IDs are interpreted
  together with genomic position; raw `tf` calls
  and normalized `tf_sr` intervals remain unchanged.
- **`fiberhmm-site-consensus discover/quantify`** — coverage-aware site-consensus state
  discovery for amplicons or selected genome-wide coordinates. It gates 1-kb
  cores on recurrent >=150-bp MSPs, learns chemistry-aware geometry from a
  deterministic source/strand-balanced cap, freezes that catalog, then scores
  the complete unbiased independent-molecule cohort with the sequence-context
  likelihood. The enriched discovery cohort never supplies occupancy. DAF
  quantification reuses the exact deamination-fingerprint PCR-collapse
  allowlist, and ordinary `tf`/`nuc` annotations are never overwritten.
  Quantification defaults to the unchanged CPU likelihood reference. Install a
  CUDA-enabled PyTorch build and pass `--likelihood-backend cuda` to evaluate
  anchored-state, spatial-null, and diffuse-null likelihoods in deterministic
  float64 resident batches;
  `--likelihood-backend auto` falls back to CPU when no CUDA device is
  accessible. CUDA uses VRAM-aware, sparse genomic-locality batches, exact
  vectorized eligibility, and CPU replay only for assignments within
  `--cuda-replay-guard-nats` of either decision threshold. Multiple
  same-chemistry BAMs are pooled by repeating
  `-i/--input` in the same order for discovery and quantification. Use
  `quantify --reuse-models-from PRIOR_QUANTIFICATION` to rescore a strictly
  provenance-matched frozen model set with different thresholds or a different
  backend without repeating the geometry fit.
  For tens to thousands of predefined loci, use the BED batch path rather than
  scanning intervening chromosomes:

  ```bash
  fiberhmm-site-consensus batch \
    --bed oct4_peaks.bed \
    -i replicate1.bam -i replicate2.bam \
    --site-padding 500 --unit-workers 8 -c 1 \
    --likelihood-backend cpu --skip-input-hash \
    -o oct4_site_consensus_scan
  ```

  Batch targets are snapped to the global 1-kb discovery grid. Overlapping
  padded targets are never split; nearby targets share bounded indexed BAM
  fetches; and gap-only families are removed before fitting. Discovery finishes
  first, then the batch freezes one calibration mask containing every padded
  target and every retained consensus state's complete candidate envelope. Every
  work unit calibrates against that same mask and full-alignment evidence, so
  BED partitioning changes results only at floating-point roundoff. Discovery
  and CPU quantification run across work units in parallel. CUDA quantification
  uses one fresh, sequential device process per unit—preventing CUDA/fork state
  inheritance—and is most useful for dense merged units; sparse ChIP-peak lists
  generally run faster with the CPU default.
  `--resume` restarts only incomplete units. Parent progress is concise, while
  detailed discovery and quantification logs remain beside each unit.
  Aggregate TSVs include both padded-target and direct BED-overlap membership,
  globally allocated reusable consensus-state slots, and explicit `unscorable` rows.
  Per-unit model/score provenance is stored as compact JSONL rather than one
  small file per consensus state.
- **`fiberhmm-pair`** — the scDAF paired-duplex workflow. By default it combines
  independently sequence-supported CT/GA assignments with high-confidence
  assignments from the frozen sequence-free model. The latter uses the
  nucleosome lattice, aligned geometry, and raw and component-residual DddA
  protection. At the default two-sided abstention margin, the archived-call
  calibration achieved 51/55 correct sequence-checkable external assignments
  (92.7%); the rotational-recall calibration achieved 48/53 (90.6%). Sequence
  evidence takes precedence on conflicts and every accepted read records its
  route as `pm:S` or `pm:D`. Use `--sequence-only` when every accepted pair must
  have direct A/T support. `--merge --recall` unions the two observed channels
  and runs the ordinary callers on the joint molecule. See the
  [paired-duplex contract](docs/paired-duplex.md) and
  [sequence-free validation](docs/duplex.md).
- **DddA-inferred complete-island mCG states** — `fiberhmm-tag-m5c`
  infers CpG islands from the indexed reference by default and assigns one
  molecule-specific state to each complete island using only observations in
  the initial MSP. Confident methylated islands are written as `ddda_mcg.` MA
  intervals so DddA TF recall can exclude confounded CpG opportunities without claiming an
  internal methylation boundary.
- **No genome context files** — hexamer context computed from read sequences.
- **Spec-compliant tags** — `ns`/`nl`/`as`/`al` legacy tags plus `MA`/`AQ`
  [Molecular-annotation spec](https://github.com/fiberseq/Molecular-annotation-spec)
  tags with `nuc.QQQ` / `tf.QQQ` scoring, plus advisory `MA-TYPES:v1` header
  declarations so FiberBrowser can initialize rare layers without sampling.
- **Multi-platform** — supported workflows cover Hia5 PacBio/Nanopore fiber-seq
  and DAF-seq (DddB/DddA). Development-only chemistry paths are identified
  explicitly below rather than presented as production support.
- **Native, fast** — no hmmlearn dependency; Numba JIT for ~10× speedup.

Population consensus reconstruction of a nuc as alternative TF combinations
is intentionally a regional FiberBrowser analysis, not a FiberHMM chemistry
caller. Its detailed implementation contract is in the
[FiberBrowser CR handoff](docs/FIBERBROWSER_CONSENSUS_RECONSTRUCTION.md).

## Installation

```bash
pip install fiberhmm
```

From source:

```bash
git clone https://github.com/fiberseq/FiberHMM.git
cd FiberHMM && pip install -e .
```

Optional extras:

```bash
pip install numba        # ~10x faster HMM computation (recommended)
pip install matplotlib   # --stats visualization
pip install h5py         # HDF5 posteriors export
```

For bigBed output, install [UCSC tools](https://hgdownload.soe.ucsc.edu/admin/exe/)
(`bedToBigBed`, and `bigBedInfo`/`bigBedToBed` for `fiberhmm-utils fix-bigbed`).

## Quick start

`fiberhmm-call` is the entry point for almost everything. Pre-trained models are
bundled — `--enzyme` selects the chemistry and `--seq` selects the platform when
applicable; `-m` is only for custom models.

```bash
# Fiber-seq (Hia5), sorted+indexed BAM — region-parallel is fastest
fiberhmm-call -i sorted.bam -o calls.bam --enzyme hia5 --seq pacbio \
              -c 8 --region-parallel --skip-scaffolds

# DAF-seq (DddB), aligned BAM with MD tags; the enzyme selects DAF mode
fiberhmm-call -i aligned.bam -o calls.bam --enzyme dddb \
              -c 8 --region-parallel

# DAF-seq amplicons (DddA): duplicate marking, SNP screening, and QC are automatic
fiberhmm-call -i aligned.bam -o calls.bam --enzyme ddda \
              -c 8 --region-parallel

# Compare several completed datasets in one QC report
fiberhmm-qc -i embryo.bam spatial_1.bam spatial_2.bam -o qc_comparison/

# Unaligned / stdin → streaming mode, pipe straight into FIRE
fiberhmm-call -i unaligned.bam -o - --enzyme hia5 --seq pacbio -c 8 \
    | ft fire - final.bam

# Extract calls to BED12 / bigBed for browsing
fiberhmm-extract -i calls.bam --nucleosome --msp --tf

# Learn a footprint population model and per-read FiberBrowser overlay
fiberhmm-footprint-model -i calls.bam -o results/sample --genome dm6 --bigbed
```

## Choosing a command

| Situation | Command |
|-----------|---------|
| **Full pipeline, sorted+indexed BAM** (default) | `fiberhmm-call --region-parallel` |
| Bounded QC for an existing BAM | `fiberhmm-qc -i calls.bam` |
| Unaligned/unsorted BAM, or reading from stdin | `fiberhmm-call` (streaming, no `--region-parallel`) |
| File-based DddA/DddB call | `fiberhmm-call` (automatic nondestructive dedup → SNP mask → footprint call → QC) |
| Compare one or more completed BAMs | `fiberhmm-qc -i sample1.bam [sample2.bam …]` |
| Only nucleosome/MSP calls, no TF recall | `fiberhmm-apply` |
| Already have an apply-tagged BAM, add TF calls | `fiberhmm-recall-tfs` |
| Apply-tagged BAM, full recall without re-running the HMM | `fiberhmm-recall-nucs` |
| Recalled BAM → footprint population model and binding hypotheses | `fiberhmm-footprint-model` |
| Add one molecule-specific DddA mCG state per complete CpG island | `fiberhmm-tag-m5c -i calls.bam -o mcg.bam -r ref.fa --enzyme ddda` |
| Build an aggregate DddA mCG validation BED | `fiberhmm-call-m5c` |
| Focal two-strand TF rescue + TF/nuc edge normalization | `fiberhmm-strand-rescue` |
| Strand-rescue report → normalized regional shadow layers | `fiberhmm-strand-rescue-annotate` |
| Audit normalized strand-rescue BAMs | `fiberhmm-strand-rescue-audit` |
| Add compact consensus-state slots/confidence to normalized TF calls | `fiberhmm-tag-consensus` |
| Infer scDAF CT/GA physical pairs | `fiberhmm-pair` |
| Require direct sequence support for every pair | `fiberhmm-pair --sequence-only` |
| Pair, merge, and jointly re-call inferred scDAF duplexes | `fiberhmm-pair --merge --recall` |
| Calls → BED12 / bigBed | `fiberhmm-extract` |
| Add or repair MA layer discovery metadata | `fiberhmm-utils ma-types calls.bam --types ...` (known names) or `--scan` (every alignment) |

`fiberhmm-call` has two execution strategies: **region-parallel**
(`--region-parallel`, requires a coordinate-sorted + indexed BAM; near-linear
scaling up to chromosome count, writes sorted+indexed output) and **streaming**
(default; accepts unaligned/unsorted BAM or stdin `-i -`, and pipes to stdout
`-o -` for `ft fire`). These are separate from the HMM observation setting
exposed by the advanced `--mode` override, which is selected automatically from
`--enzyme` and, for Hia5, `--seq` in normal use.

> `fiberhmm-run` was removed in 2.8.0 — it chained apply + recall + fire as
> separate piped subprocesses. `fiberhmm-call` fuses those stages in-process and
> is 2–9× faster. Replace `fiberhmm-run` with `fiberhmm-call [| ft fire]`.

## Workflows

### Fiber-seq (Hia5)

```bash
fiberhmm-call -i sorted.bam -o calls.bam --enzyme hia5 --seq pacbio \
              -c 8 --region-parallel --skip-scaffolds
```

Set `--seq` explicitly for Hia5: PacBio detects m6A on both strands, while
Nanopore detects it on one. If omitted, the current resolver warns and defaults
to `pacbio`. Add FIRE scoring as a second step: `ft fire calls.bam final.bam`, or
stream it (see Quick start).

### DAF-seq (DddB)

`--enzyme dddb` selects the bundled DddB model, whose metadata sets the
observation mode to `daf`. You normally do not need to add `--mode daf`;
`--mode` is retained as an advanced override for custom models.

FiberHMM must distinguish DAF conversions from bases already present in the
reference. For normal file-input workflows, the supported sources are used in
this precedence order:

1. **R/Y IUPAC codes** in the stored sequence (from `fiberhmm-daf-encode`) — fast path
2. **A usable `MD` tag** — alignment metadata describing matches, mismatches,
   and deletions relative to the reference. FiberHMM combines `MD`, CIGAR, and
   the query sequence to locate reference-C/query-T and reference-G/query-A
   substitutions; `MD` is not itself a modification tag.
3. **`--reference ref.fa`** — fallback when `MD` is absent, cannot be decoded,
   or its encoded reference span disagrees with the CIGAR. FiberHMM fetches the
   aligned reference bases and compares them with the query sequence in memory.

The FASTA must match the BAM's assembly and contig names and must be indexed
(`samtools faidx ref.fa`). Supplying it does **not** realign reads, rewrite the
BAM sequence, add R/Y codes, regenerate `MD`, or select a different model. R/Y
and a structurally usable `MD` tag take precedence, so `--reference` does not
force FASTA comparison. If an `MD` tag is stale but still has the same span as
the CIGAR, refresh it explicitly:
`samtools calmd -b aligned.bam ref.fa > aligned.calmd.bam`. The separate
`fiberhmm-tag-m5c` step is the exception: it always requires and reads the
FASTA for CpG-island inference and DddA sequence context, even when R/Y or a
usable `MD` tag is present.

```bash
# Raw DAF BAM with MD tags (from `minimap2 --MD` or `samtools calmd`)
fiberhmm-call -i aligned.bam -o calls.bam --enzyme dddb --region-parallel

# No usable MD tags: supply the indexed alignment reference
samtools faidx ref.fa
fiberhmm-call -i aligned.bam -o calls.bam --enzyme dddb \
              --reference ref.fa --region-parallel
```

`fiberhmm-call` preflights the first mapped reads of file inputs and stops if it
sees none of R/Y, MD, or `--reference`, instead of silently skipping every read.
Running `fiberhmm-daf-encode` first is optional; it uses the same conversion
detector but also stamps R/Y into the stored sequence for downstream R/Y-aware
tools.

Important options shared by DddB and DddA workflows:

| Option | When to use it | Effect |
|--------|----------------|--------|
| `--reference ref.fa` | R/Y is absent and `MD` is missing/unusable | Uses an indexed FASTA as the per-read mismatch fallback described above. The separate `fiberhmm-tag-m5c` command always requires the indexed reference for CpG-island inference and DddA context. |
| `--dedup` / `--no-dedup` | Force or disable the automatic file-based DddA/DddB pre-pass | By default, nondestructively marks PCR duplicates by similar alignment ends plus deamination-pattern similarity **before** SNP/footprint calling. Every read is retained with `0x400` and `di`/`ds`; pooled SNP, phase, and QC calculations ignore marked copies. Requires fingerprintable MM/ML dU, R/Y, or usable `MD` calls; this pre-pass does not use `--reference`. |
| `--dedup-collapse` | You explicitly want a smaller unique-molecule BAM | Destructively removes non-representative cluster members instead of the automatic mark-and-retain behavior. |
| `--no-daf-call-snps` | You do not want automatic recurrent-SNP screening | Disables the post-dedup, coverage-gated SNP mask. Low-depth samples skip it automatically. |
| `--no-qc` | You do not want bounded post-call QC | Disables the default assay-matched QC report. |
| `--keep-chimeras` | QC or intentional retention of strand-swap reads | Disables the default DAF strand-swap chimera filter. |
| `--region-parallel` | Coordinate-sorted, indexed BAMs | Processes genomic regions in parallel and writes sorted, indexed output. |

See [PCR deduplication](#pcr-deduplication) and
[`fiberhmm-dedup`](#fiberhmm-dedup) for behavior and tuning options.
For any BAM that depends on the FASTA fallback, first add a usable `MD` tag with
`samtools calmd` or run `fiberhmm-daf-encode --reference ref.fa` before enabling
the automatic dedup pre-pass (or use `--no-dedup`).

### DAF-seq amplicons (DddA)

`--enzyme ddda` handles DddA's specifics automatically:

- **Independent state models** — `ddda_nuc.json` drives the first-pass
  nucleosome HMM; `ddda_TF.json` contains physical-duplex-calibrated TF
  emissions; and the internal `ddda_nuc_refine.json` freezes the likelihoods
  used by radial nucleosome refinement. Updating TF calibration therefore does
  not silently retune the HMM or radial nucleosome caller. (For QC you can run
  the stages separately: `fiberhmm-apply --enzyme ddda` followed by
  `fiberhmm-recall-tfs --enzyme ddda`.)
- **Phase-aware radial nucleosome recall** is **on by default**. DddA deaminates
  *inside* nucleosomes, so the standard accessible-cut split would shatter them.
  FiberHMM instead uses the radial deamination profile to nominate dyads and a
  molecule-local, sequence-context-aware likelihood marginalized over uncertain
  helical register and 9–12-bp local pitch to infer each edge. Every
  dyad-nominated raw edge uses the same continuous posterior estimator; broad
  support lowers `el`/`er` rather than selecting a different raw-edge
  estimator. Final emitted boundaries can still be constrained by the
  molecule-local HMM-versus-posterior configuration test and non-overlapping
  tiling. Calls without a radial dyad and promoted/fallback calls are outside
  this posterior-edge contract.
  Use `--no-recall-nucs` for raw HMM nucleosomes.
- **Nuc-derived TF edge safeguard** — TF candidates exposed only because radial
  recall opened new scan space require observed deaminations within 12 bp on
  both sides. Calls already supported by the original HMM-accessible scan space
  are unchanged. Use `--ddda-derived-tf-max-edge-gap 8` for a stricter
  exploratory view or `-1` to reproduce the ungated behavior.
- **Strand-swap chimera filter** is on by default (reads deaminated C→T in one
  segment and G→A in another are dropped and counted). `--keep-chimeras` to
  disable; `--chimera-min-seg` / `--chimera-purity` to tune.

```bash
# Automatic dedup needs fingerprintable MM/ML dU, R/Y, or usable MD evidence
fiberhmm-call -i aligned.bam -o calls.bam --enzyme ddda \
              -c 8 --region-parallel
```

### DddA CpG-island methylation

Genome-wide DddA amplification removes native 5mC tags, but methylated CpGs
retain a strong DddA rate signature. FiberHMM reports this signal at the scale
of complete CpG islands. Run the ordinary footprint caller first, then the
island caller and mCG-aware recall:

```bash
fiberhmm-call -i aligned.bam -o calls.initial.bam --enzyme ddda \
              -c 8 --region-parallel
fiberhmm-tag-m5c -i calls.initial.bam -o calls.m5c.bam \
                 -r reference.fa --enzyme ddda \
                 --write-cpg-islands islands.used.bed \
                 --calls-tsv island_calls.tsv
fiberhmm-recall-tfs -i calls.m5c.bam -o calls.bam \
                    --enzyme ddda --use-m5c -c 8
```

By default the tagger derives islands from 200-bp reference windows stepped by
10 bp, retaining windows with GC fraction at least 0.50 and CpG
observed/expected at least 0.60 and merging overlaps. `--cpg-islands` accepts a
preferred merged BED catalog. This changes which islands are tested, while the
resolution remains one state per complete island. Only CpG and non-CpG
observations inside the molecule's initial MSP contribute. Calls require at
least 15 CpGs, 10 non-CpGs and posterior at least 0.99 (methylated) or at most
0.01 (unmethylated); other overlaps are recorded as uninformative in the audit
table. Only confident methylated islands become `ddda_mcg.` MA intervals.

`fiberhmm-recall-tfs` removes CpGs inside those intervals from its opportunity
lattice by default for DddA; use
`--no-use-m5c` for an ablation. The retired `fiberhmm-call --ddda-mcg` spelling
now exits with a migration message rather than running the older per-CpG mode.

This caller and its emission correction are calibrated specifically for
genome-wide DddA DAF-seq. They must not be applied to DddB and are not part of
the ordinary targeted/amplicon DddA workflow.

> **DddA amplicons can be heavily PCR-duplicated**, and coordinate dedup
> (Picard/markdup) does **not** apply — every read piles up on the same locus with
> primer-fixed ends. Automatic dedup requires similar alignment ends and a matching
> deamination fingerprint, then marks duplicates *before* SNP/footprint calling
> (see [`fiberhmm-dedup`](#fiberhmm-dedup)). Integrated dedup retains every read
> unless `--dedup-collapse` is explicitly requested.

> The phase-aware DddA nucleosome caller is the production default. Its locked
> profile was validated on twelve independent HG002 scDAF libraries and the
> GM12878 NAPA and UBA1 targeted cohorts. The baseline caller remains
> molecule-local: population/strand consensus is a separate optional pass and
> is not required to obtain these nucleosome calls.

### Second-pass recall on an apply-tagged BAM

If you already have a BAM tagged by `fiberhmm-apply` and want to add calls without
re-running the HMM, use the recallers. Both reconstruct the per-base observations
from each read's available modification/deamination evidence and sequence, then
reuse the existing `ns`/`nl`/`as`/`al` tags — the HMM is **not** re-run.

```bash
# TF recall only (over the original apply MSPs + short nucs)
fiberhmm-recall-tfs  -i apply.bam -o recalled.bam --enzyme hia5 --seq pacbio -c 8

# Full recall: nucleosome refine → MSP re-derive → TF recall → promotion
fiberhmm-recall-nucs -i apply.bam -o recalled.bam --enzyme hia5 --seq pacbio -c 8

# Nanopore m6A: auto selects topology-constrained recall
fiberhmm-recall-nucs -i ont.apply.bam -o ont.recalled.bam \
                     --enzyme hia5 --seq nanopore -c 8
```

`fiberhmm-recall-nucs` produces equivalent footprint tags to
`fiberhmm-call --recall-nucs` for matched `--phase-nrl` and
`--nuc-recall-policy`; their BAM headers retain the distinct command histories.
For Nanopore, the default
`auto` policy resolves to `topology`: an accessible cut must leave a
nucleosome-sized candidate on every side, and unresolved single-strand edge
ambiguity remains protected rather than being labeled accessible. Use
`--nuc-recall-policy conservative` only to request the explicit
conservative-edge behavior. **Linear reads only** — circular reads must use
`fiberhmm-call -r --recall-nucs`.

### PCR deduplication

`fiberhmm-call` automatically uses the second, nondestructive behavior below
for file-based DddA/DddB input. The standalone command retains its explicit
collapse default so existing scripts do not change semantics.

```bash
# Collapse to one representative read per molecule (default)
fiberhmm-dedup -i sample.bam -o sample.dedup.bam

# Mark duplicates instead of removing them (0x400 + di/ds tags)
fiberhmm-dedup -i sample.bam -o sample.markdup.bam --flag-only
```

See [`fiberhmm-dedup`](#fiberhmm-dedup) for how it works and when to use it.

## Command reference

### fiberhmm-call

Fused apply + nucleosome recall + TF recall in one process. See
[Choosing a command](#choosing-a-command) for the region-parallel vs streaming
execution strategies.

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | Input BAM, or `-` for stdin. |
| `-o/--output` | required | Output BAM, or `-` for stdout (unsorted). |
| `--enzyme` | — | Supported presets are `hia5`, `dddb`, and `ddda`. |
| `--seq` | chemistry-dependent | Hia5 supports `pacbio`/`nanopore` (omission warns and defaults to `pacbio`); ignored for DddA/DddB after DAF encoding. |
| `--mode` | from model | Advanced observation-mode override; normally inferred from the selected model. Supported models use `pacbio-fiber`, `nanopore-fiber`, or `daf`; `gpc` and `cpg` are development-only. |
| `--reference` | — | Indexed FASTA fallback for ordinary DAF reads with no R/Y and missing/unusable `MD`; does not override R/Y or usable `MD`. |
| `--ddda-mcg` | retired | Emits a migration error directing users to `fiberhmm-tag-m5c`, which reports complete CpG-island states. |
| `-c/--cores` | 4 | Worker processes. |
| `--io-threads` | 8 | htslib I/O threads. |
| `--region-parallel` | off | Per-region worker pool (requires sorted+indexed input). |
| `--skip-scaffolds` | off | Drop small scaffolds (region-parallel). |
| `--chroms chr1 …` | all | Restrict to specific chromosomes (region-parallel). |
| `--no-recall-nucs` | recall on | Disable nucleosome recall (baseline HMM `nuc.Q`). |
| `--nuc-recall-policy` | `auto` | `auto` uses topology-constrained, ambiguity-preserving recall for Nanopore and `conservative` edges otherwise; either policy can be forced explicitly. |
| `--ddda-derived-tf-max-edge-gap` | 12 | DddA phase-aware radial recall only: maximum full-molecule gap to an observed deamination on each side of a TF call exposed solely by nuc refinement; `-1` disables. |
| `--phase-nrl` | `auto` | Periodicity prior: `auto` (estimate, ~150–215 bp), `off`, or a fixed bp. |
| `--min-llr` | enzyme preset | Override TF LLR threshold. |
| `-r/--circular` | off | Circular molecule mode (see [reference](docs/reference.md#circular-molecules)). |
| `--keep-chimeras` | off | DAF: keep strand-swap chimeric reads (default: filter + count). |
| `--no-legacy-tags` | off | Emit only `MA`/`AQ`, skip `ns/nl/as/al`. |
| `--downstream-compat` | off | Write TF calls into legacy `ns/nl` (skip `MA/AQ`). |
| `--dedup` / `--no-dedup` | auto for file-based DddA/DddB | Force or disable endpoint-constrained fingerprint dedup before SNP/footprint calling. Automatic/default behavior marks and retains every read. Ignored for non-DAF modes. |
| `--dedup-collapse` | off | Explicitly remove duplicate copies instead of nondestructive marking. This is the only destructive integrated dedup mode. |
| `--dedup-max-end-diff` | 50 | Maximum difference at both aligned reference ends for duplicate matching. |
| `--daf-call-snps` / `--no-daf-call-snps` | auto for file-based DddA/DddB | Force or disable recurrent opposite-conversion SNP discovery. Auto mode first performs a bounded local-depth screen and skips full SNP passes when depth is below threshold. |
| `--daf-snp-mask BED` | — | **DAF only.** Exclude supplied 0-based BED sites from DAF observations without changing the `MD` tag. |
| `--daf-snp-min-amplicon-reads` | 20 | Minimum primary MAPQ-filtered reads required to report an amplicon consensus in SNP QC. |
| `--qc` / `--no-qc` | on | Run/disable bounded post-call QC for file outputs. Stdout BAM streams are skipped. |
| `--qc-sample-reads` | 2000 | Target deterministic random-window QC sample size; this is not a whole-BAM pass. |
| `--qc-min-mapq` | 20 | Minimum mapping quality in the QC sample, independent of the calling filter. |

Automatic QC always selects its reference from the resolved `--enzyme` and
`--seq` combination used by `fiberhmm-call`. Explicit profile overrides are
available only in standalone `fiberhmm-qc`.

`--dedup` tunables (forwarded to the dedup pass): `--dedup-min-jaccard` (0.95),
`--dedup-collapse`, `--dedup-max-end-diff`, `--dedup-min-deam`, `--dedup-prob-threshold`,
`--dedup-ignore-strand`, `--dedup-stats-tsv`. MinHash internals stay at defaults —
use standalone `fiberhmm-dedup` to tune those. `--dedup-flag-only` is retained
as a compatibility spelling for the integrated nondestructive default.

### fiberhmm-qc

Run the same bounded QC independently on an existing FiberHMM-compatible BAM:

```bash
fiberhmm-qc -i calls.bam --enzyme dddb
fiberhmm-qc -i sample1.bam sample2.bam sample3.bam -o results/qc --enzyme dddb
```

With one input, outputs are `<BAM directory>/qc/<BAM stem>.qc.json`, `.qc.tsv`,
and, when matplotlib is installed, `.qc.png` plus an Illustrator-compatible
vector `.qc.pdf`. PDF fonts are embedded as editable TrueType text rather than
converted to outlines. `-o/--output-dir` overrides the
QC directory. Multiple inputs produce every per-sample artifact plus
`combined.qc.json`, `combined.qc.tsv`, `combined.qc.png`, `combined.qc.pdf`, and
a self-contained index page, `combined.qc.html`. Every individual figure includes the matched
control ECDF/median, matched control phasogram, empirical control
nucleosome/TF size distributions where available, PCR duplication, a
mismatch-percentage landscape, an amplicon SNP-location map, and representative
sample/control molecule hatchmarks. The combined figure compares signal median, periodicity score,
inferred repeat length, autocorrelation strength, nucleosome size, and TF size
across samples; it does not obscure differences by overlaying raw phasograms. Inputs
from different directories require an explicit `-o` to avoid placing a report
in a surprising directory. Automatic QC after `fiberhmm-call` uses
`<output BAM directory>/qc/<BAM stem>.qc.*` unless `--qc-output-prefix` is set.

The terminal and JSON contain separate signal-rate
and nucleosome-periodicity scores plus an overall PASS/WARN/FAIL status.
Low-evidence samples are reported as `INSUFFICIENT` rather than failed.
Rate grading uses the same bands drawn in the figure: the control IQR
(25th–75th percentiles) is PASS, the remainder of the 5th–95th percentile
interval is WARN, and values outside those empirical tails are FAIL. The panel
prints the sample median, control median, and percentage-point delta.
Periodicity is conjunctive: peak strength, NRL, reference-curve correlation,
and the amplitude of the reference-shaped component must all support the
grade. Thus a flat or noisy curve cannot pass merely because its largest
autocorrelation value happens to lie in the nucleosome-scale search interval.
Nucleosome and TF size panels are populated only when `MA` (or legacy `nl` for
nucleosomes) is present.

For coordinate-indexed BAMs, reads are selected from seeded random genomic
windows by alignment start. Unindexed BAMs use a reservoir over at most a
bounded prefix (default cap: 10× the requested sample, never the entire file).
The JSON records the strategy, seed, records examined, and
`whole_bam_scanned: false`. QC never invokes `fiberhmm-dedup`. When integrated
calling used `--dedup`, it writes an aggregate `.dedup.json` sidecar and QC
plots the exact full-run duplicate fraction plus the distribution of copies per
original molecule. Otherwise standard duplicate flags or FiberHMM `di`/`ds`
tags support a bounded-sample summary; if neither exists, the panel explicitly
says deduplication was not run.

The bundled empirical profiles identify their source datasets in the JSON.
Automatic post-call selection is locked to the resolved enzyme/platform pair:
Drosophila embryo DddB DAF-seq, human DddA DAF-seq, Drosophila embryo PacBio
Fiber-seq, or Drosophila embryo Nanopore Fiber-seq. Incompatible
mode/enzyme/reference combinations are rejected.
Only aggregate control curves, summary metadata, and a few anonymous visual
molecule exemplars are packaged—never control BAMs, sequences, read identifiers,
barcodes, or genomic coordinates. Each exemplar contains only molecule span,
relative signal hatch positions, and signal rate. See
[`fiberhmm/qc/README.md`](fiberhmm/qc/README.md) for source provenance and the
curve-construction method.
Reference intervals are screening diagnostics, not biological exclusion
criteria. If an m6A ML threshold differs materially from the calibrated
profile, the rate component is capped at WARN and the mismatch is reported.

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | One or more input BAM/CRAM files; `-i` may be repeated. |
| `-o/--output-dir` | `qc/` beside inputs | Output directory for individual and combined reports. |
| `--mode` / `--enzyme` | `auto` | Override assay inference. |
| `--reference-profile` | `auto` | Empirical profile or `none`. |
| `--sample-reads` | 2000 | Target bounded sample size. |
| `--seed` | 20260824 | Reproducible sampler seed. |
| `--prob-threshold` | 125 | MM/ML probability threshold. |
| `--snp-mask` / `--snp-report` | — | Single-input DAF QC: apply a BED mask to recomputed rate/periodicity and plot an optional caller JSON. |
| `--fail-on-qc` | off | Exit 2 on final FAIL; WARN/INSUFFICIENT still exit 0. |

### fiberhmm-daf-snps

DAF conversions and C/T or G/A variants can be confounded. The optional caller
classifies each informative fiber by its dominant conversion direction, then
calls a recurrent C→T mismatch only across otherwise G→A-dominant fibers (and
G→A only across C→T-dominant fibers). This keeps genuine molecule-specific
deamination out of the SNP denominator.

```bash
# Write BED, VCF, JSON, and an amplicon-coverage TSV; do not rewrite BAM/MD
fiberhmm-daf-snps -i aligned.bam -o qc/sample.daf_snps

# File-based DddA/DddB calls do this automatically when local depth is adequate
fiberhmm-call -i aligned.bam -o calls.bam --enzyme dddb \
              --region-parallel

# Or apply a previously reviewed mask
fiberhmm-call -i aligned.bam -o calls.bam --enzyme dddb \
              --region-parallel --daf-snp-mask qc/sample.daf_snps.bed
```

The named production policy `bidirectional_five_fiber_v1` requires at least
20% mismatch fraction, depth five, and five mismatch
fibers independently in **each** conversion-direction class; all three tests
are conjunctive in both classes. The five-fiber default was selected by the
reproducible downsampling validation in
`scripts/validation/validate_daf_snp_downsampling.py`: it retained 98.4% of
prespecified unambiguous sites at an expected minimum bidirectional depth of
10.5, versus 61.2% with a separate depth-10 cliff, while neither rule called a
site in the prespecified high-depth/low-mismatch background class. This
symmetric support rule rejects direction-specific artifacts without discarding
a clear 18/18 site merely because it falls below an arbitrary hard-depth cliff.
Amplicons require 20 primary MAPQ-filtered reads by default; adjust this
with standalone `--min-amplicon-reads` or integrated
`--daf-snp-min-amplicon-reads`. Use `--daf-snp-min-fraction`,
`--daf-snp-min-depth`, and related flags to tune the integrated caller; the
standalone equivalents are `--min-fraction`, `--min-depth`, and
`--min-alt-fibers`. Any override is recorded as a `custom` threshold policy in
the caller JSON alongside the validated defaults.
Masking is observational: the original sequence
and `MD` tag are preserved, while listed reference positions are excluded from
DAF emissions and QC rate/periodicity calculations. The caller JSON also keeps
a deterministic bounded background sample of ordinary C/G positions. QC plots
mismatch percentage on expected-conversion fibers against mismatch percentage
on opposite-conversion fibers for this whole site population, colors the points
by dominant-fiber depth, outlines calls, and maps their locations along the
consensus of every discovered amplicon. Each map row reports the amplicon's
genomic interval and total aligned-read coverage; the caller also writes these
values and relative SNP coordinates to `.amplicons.tsv`.

### fiberhmm-apply

Apply a trained HMM to call nucleosomes/MSPs (no TF recall). Streaming pipeline
with stdin/stdout support.

```bash
fiberhmm-apply -i experiment.bam --enzyme hia5 --seq pacbio -o output/ -c 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | Input BAM, or `-` for stdin. |
| `-m/--model` | optional | Custom model (`.json`/`.npz`/`.pickle`); overrides `--enzyme`. |
| `--enzyme` | optional | Supported presets are `hia5`, `dddb`, and `ddda`. Required unless `-m` is given. |
| `--seq` | chemistry-dependent | Hia5 supports `pacbio`/`nanopore` (omission warns and defaults to `pacbio`); ignored for DddA/DddB after DAF encoding. |
| `-o/--outdir` | required | Output directory, or `-` for stdout BAM. |
| `--mode` | from model | Advanced observation-mode override; normally inferred from the selected model. |
| `-c/--cores` | 1 | CPU cores (0 = auto). |
| `--io-threads` | 4 | htslib I/O threads. |
| `-q/--min-mapq` | 0 | Min mapping quality (`0` = no filtering). |
| `--min-read-length` | 1000 | Min aligned read length (`0` to disable). |
| `-e/--edge-trim` | 10 | Edge masking (bp). |
| `--msp-min-size` | 0 | Minimum MSP region size (bp). |
| `--scores` | off | Compute per-footprint confidence scores (`nq`/`aq`). |
| `-r/--circular` | off | Circular molecule mode. |
| `--skip-scaffolds` / `--chroms` / `--primary` | — | As in `fiberhmm-call`. |

Reads are passed through unchanged (no footprint tags) when MAPQ or length is
below threshold, when no usable modification or deamination observations are
found, when unmapped, or when the HMM finds no footprints.

### fiberhmm-recall-tfs / fiberhmm-recall-nucs

LLR-based second-pass recallers over an apply-tagged BAM. `recall-tfs` adds TF/Pol
II footprints; `recall-nucs` additionally refines nucleosomes (= `recall-tfs
--recall-nucs`). For DddA the TF recall pass is **required** (the nucleosome model
doesn't emit sub-nucleosomal calls) and `ddda_TF.json` is selected automatically.

```bash
fiberhmm-recall-tfs -i apply.bam -o recalled.bam --enzyme hia5 --seq pacbio -c 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--in-bam` | required | Input BAM tagged by `fiberhmm-apply`. `-` for stdin. |
| `-o/--out-bam` | required | Output BAM (`MA`/`AQ` + refreshed legacy tags). `-` for stdout. |
| `-m/--model` | optional | Custom model JSON; overrides `--enzyme`. |
| `--enzyme` | optional | Supported presets are `hia5`/`dddb`/`ddda`; sets the model and `--min-llr` preset. |
| `--seq` | chemistry-dependent | Hia5 platform selector as above; ignored for DddA/DddB after DAF encoding. |
| `--min-llr` | preset | Min cumulative LLR (nats) per call (`dddb` 4.0; `ddda` 7.0; other enzyme presets 5.0). DddA's held-out physical-mate operating point corresponds to `TQ >= 70`. |
| `--min-opps` | 3 | Min informative target positions per call. |
| `--unify-threshold` | 90 | Footprints with `nl <` this may be demoted to `tf.`. |
| `--nuc-recall-policy` | `auto` | With nucleosome recall, use Nanopore-aware `topology` automatically or force `topology`/`conservative`. |
| `--ddda-derived-tf-max-edge-gap` | 12 | Same DddA phase-aware radial-recall safeguard as `fiberhmm-call`; `-1` disables. |
| `--no-legacy-tags` | off | Emit only `MA`/`AQ`. |
| `--downstream-compat` | off | TF calls into legacy `ns/nl`, no `MA/AQ` (per-TF quality lost). |
| `-c/--cores` | 1 | Worker processes (0 = auto). |
| `--io-threads` | 4 | htslib threads. |

See the [deep reference](docs/reference.md) for the MA/AQ schema, the `tq`/`el`/`er`
quality bytes, output modes, circular molecules, and how to parse the output.

### fiberhmm-footprint-model

Infer a data-derived footprint population model from ordinary `tf` and `msp`
MA groups. The command preserves every projectable TF call, learns overlapping
TF-binding hypotheses, reports molecule-level overall and MSP-conditioned
occupancy, and writes both portable tables and a per-read FiberBrowser layer.

```bash
fiberhmm-footprint-model -i recalled.bam -o results/sample --genome dm6 --bigbed
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | One recalled BAM containing ordinary MA annotations. |
| `-o/--output-prefix` | required | Filename prefix for the model bundle. |
| `--region` | complete BAM | Repeatable zero-based, half-open region; requires an index. |
| `-q/--min-mapq` | 0 | Minimum alignment MAPQ. |
| `--include-duplicates` | off | Include BAM duplicate-flagged records. |
| `--bigbed` | off | Create indexed population and FiberBrowser BigBeds. |
| `--genomewide` | off | Assert genome-wide scope in the manifest. |

Records without `MA` are excluded because they are not known to have passed
footprint calling; valid MA records with no TF are retained as denominators.
There is no TQ filter. Full semantics, model parameters, outputs, and Python API
are in [the footprint-model documentation](docs/footprint-models.md).

### fiberhmm-extract

Extract nucleosome/MSP/TF/m6A/m5C/deamination features from tagged BAMs to
BED12 / bigBed (bigBed by default; one file per feature type).

```bash
fiberhmm-extract -i calls.bam -o output/ -c 8        # all types
fiberhmm-extract -i calls.bam --nucleosome --msp --tf
fiberhmm-extract -i calls.bam --keep-bed             # keep BED alongside bigBed
fiberhmm-extract -i calls.bam --tf --msp --circular-groups   # FiberBrowser grouping
fiberhmm-extract -i calls.bam --nucleosome --msp --tf --haplotype-fields
```

| Flag | Default | Description |
|------|---------|-------------|
| `--nucleosome` / `--msp` / `--tf` / `--m6a` / `--m5c` / `--deam` / `--both-strand` | all | Feature types to extract (default: all). `--both-strand` is the paired-duplex DAF coverage intersection. |
| `--bed-only` / `--keep-bed` | off | BED only / keep BED beside bigBed. |
| `--block-scores` | off | Append per-block quality columns (BED12+N). |
| `--circular-groups` | off | Emit circular grouping fields for nucleosome/MSP/TF features in FiberBrowser. |
| `--haplotype-fields` | off | Append scalar `hp` and `ps` columns copied from BAM `HP`/`PS`; `-1` means missing. |
| `--sample-name` | BAM stem | Sample tag embedded in each bigBed's autoSQL. |
| `-S/--sort-mem` | `1G` | Buffer for the BED sort (`sort -S`; e.g. `8G`). |
| `--sort-parallel` | `--cores` | Sort threads (GNU sort; feature-detected). |
| `-c/--cores` | 1 | Worker processes. |

Every FiberHMM BED/bigBed schema also appends `isDuplicate` (copied from BAM
flag `0x400`). FiberBrowser hides those rows by default and can reveal them with
its **Show duplicate reads** switch; older bigBeds without this field are treated
as having no duplicate annotation.

The post-extract sort runs under `LC_ALL=C` (a large speedup on its own); `-S` and
`--sort-parallel` help further on deep/whole-genome BAMs. Each bigBed embeds a
`Sample:` autoSQL tag (sanitized to a dot/space-free token) that FiberBrowser uses
to group a sample's layers; repair older bigBeds with
[`fiberhmm-utils fix-bigbed`](#fiberhmm-utils).

`--haplotype-fields` is deliberately opt-in so default BED rows and bigBed
schemas remain unchanged. When enabled, the signed integer fields are appended
after every other optional field in the order `hp`, `ps`. Each is copied from
the source read independently; a missing or non-integer tag is `-1`. FiberHMM
does not infer or revise phasing during extraction.

### fiberhmm-dedup

PCR-duplicate detection for DAF-seq via the per-read **deamination pattern**.
DAF-seq amplicons pile up on one locus with primer-fixed ends, so coordinate dedup
(Picard/markdup) has no positional signal and there are no UMIs. The molecular
fingerprint is the set of reference positions deaminated by the enzyme (R/Y, MM/ML
dU, or MD mismatch — same sources as `fiberhmm-extract --deam`). PCR copies share
that pattern but rarely *exactly* (sequencing error and missed/over-called
deaminations perturb a handful of the hundreds of calls), so exact-match dedup
misses most duplicates.

`fiberhmm-dedup` first requires both aligned reference ends to agree within 50
bp, then clusters reads whose deamination sets match within a Jaccard threshold
(MinHash + LSH, near-linear). It **collapses each cluster to one
representative by default** (highest MAPQ / most-complete). Representatives of
duplicate clusters (size >1) carry `ds` = number of copies represented;
singletons do not receive `di`/`ds`.

```bash
fiberhmm-dedup -i sample.bam -o sample.dedup.bam              # collapse (default)
fiberhmm-dedup -i sample.bam -o sample.markdup.bam --flag-only # mark only
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i/--input` | required | Input DAF-seq BAM (R/Y-, MM/ML-dU-, or MD-encoded). |
| `-o/--output` | required | Output BAM (stays coordinate-sorted if the input was). |
| `--flag-only` | off | Mark duplicates (`0x400` + `di`/`ds`) instead of collapsing. |
| `--min-jaccard` | 0.95 | Min deamination-set Jaccard to call two reads the same molecule (bimodal gap ~0.90–0.95). |
| `--max-end-diff` | 50 | Maximum difference at both aligned reference ends; fingerprint similarity alone can never collapse differently ended molecules. |
| `--min-deam` | 10 | Reads with fewer calls aren't fingerprinted and are copied unchanged when an output is produced. If no reads are fingerprintable, the command reports that and writes no output. |
| `--ignore-strand` | off | Allow opposite-strand reads to be duplicates. |
| `-p/--prob-threshold` | 0 | Min ML probability for MM/ML-native dU calls. |
| `--stats-tsv` | — | Write a `cluster_id<TAB>n_reads` table. |

### fiberhmm-daf-snps

Call recurrent C→T/G→A genomic mismatches in DAF-seq while requiring support
from both conversion-direction fiber classes. The validated defaults require
the minimum fraction, depth, and alternate-fiber count to pass independently
in both classes. Duplicate-flagged reads are excluded. Outputs include a BED
mask, VCF, JSON report, mismatch landscape, amplicon table, and an amplicon map
with coverage, coordinates, and SNP positions.

```bash
fiberhmm-daf-snps -i aligned.bam -o qc/sample.daf_snps
```

The mask removes those positions only from DAF observations supplied to the
HMM; it does not rewrite read bases or `MD` tags. `fiberhmm-call` runs the same
caller automatically after duplicate marking when a bounded preflight finds
adequate local depth. Thresholds remain available as
`--daf-snp-min-fraction`, `--daf-snp-min-depth`, and
`--daf-snp-min-alt-fibers`.

### fiberhmm-qc

Generate bounded per-sample QC for a completed FiberHMM-compatible BAM, or pass
several BAMs for individual reports plus a combined comparison dashboard.
Indexed inputs are sampled through deterministic random genomic windows; the
default target is 2,000 reads, so QC does not rescan a multi-terabyte BAM.

```bash
# One BAM: writes qc/sample.qc.{png,pdf,json,tsv}
fiberhmm-qc -i sample.bam

# Multiple BAMs: individual reports plus combined.qc.{png,pdf,json,tsv,html}
fiberhmm-qc -i sample_1.bam sample_2.bam sample_3.bam -o comparison_qc/
```

Reference selection follows the enzyme/sequencing combination recorded by
`fiberhmm-call`. Reports include signal-rate median and empirical 5th–95th
range, nucleosome-scale phasing metrics, nucleosome/TF size distributions,
example molecules, duplicate statistics, and SNP diagnostics when available.
The terminal and report show PASS/WARN/FAIL scores. PDFs embed TrueType fonts
as editable text for Illustrator. The package contains only aggregate control
curves and anonymized visual exemplars—not source datasets; see
[QC control provenance](fiberhmm/qc/README.md) for sources and construction.

### fiberhmm-daf-encode

Preprocess plain aligned DAF-seq BAMs: identify C→T / G→A mismatches via a
usable `MD` tag or reference-FASTA fallback, encode them as IUPAC Y/R in the
query sequence, and add an `st:Z` strand tag.

**Optional** — `fiberhmm-call --enzyme dddb` (or `--enzyme ddda`) reads usable
`MD` tags directly and can use `--reference` as the fallback described above.
Use the encoder when downstream tools need R/Y/`st:Z`, or to make a BAM that
depends on FASTA fallback fingerprintable by `--dedup`.

```bash
fiberhmm-daf-encode -i aligned.bam -o encoded.bam
```

Key flags: `--reference` (indexed FASTA fallback if MD is missing/unusable),
`-q/--min-mapq` (20), `--min-read-length` (1000), `--strand`
(`CT`/`GA`/`auto`), `--io-threads` (4).

### fiberhmm-posteriors

Export per-position HMM posterior P(footprint) for downstream analysis (CNN
training, custom scoring). Input is the same BAM you'd pass to `fiberhmm-apply`.

```bash
fiberhmm-posteriors -i experiment.bam --enzyme hia5 --seq pacbio -o post.tsv.gz -c 4
fiberhmm-posteriors -i experiment.bam --enzyme hia5 --seq pacbio -o post.h5 -c 4  # needs h5py
```

### fiberhmm-probs / fiberhmm-train

Train custom models. `fiberhmm-probs` builds emission tables from accessible /
inaccessible control BAMs; `fiberhmm-train` fits the HMM using them.

```bash
fiberhmm-probs -a accessible.bam -u inaccessible.bam -o probs/ --mode pacbio-fiber -k 3 4 5 6 --stats
fiberhmm-train -i sample.bam -p probs/tables/accessible_A_k3.tsv probs/tables/inaccessible_A_k3.tsv -o models/ -k 3 --stats
```

`fiberhmm-train` writes `best-model.json` (recommended), `.npz`, all iterations,
training read IDs, config, and `--stats` plots.

### fiberhmm-utils

Model, BAM-header, and bigBed utilities:

```bash
fiberhmm-utils convert old_model.pickle new_model.json   # legacy → JSON
fiberhmm-utils inspect model.json [--full]               # metadata + emissions
fiberhmm-utils transfer --target daf.bam --reference-bam fiber.bam -o probs/ --mode daf
fiberhmm-utils adjust model.json --state accessible --scale 1.1 -o adjusted.json
fiberhmm-utils ma-types calls.bam --types nuc,msp,tf,ddda_mcg
fiberhmm-utils ma-types calls.bam --scan
fiberhmm-utils fix-bigbed sample.filtered_T_*.bb sample.filtered_GA_*.bb --in-place
```

`ma-types` repairs the optional FiberBrowser discovery declaration in place.
`--types` accepts logical names without strand/quality suffixes; `--scan`
exhaustively visits every alignment, so rare annotations are not missed. The
command safely rewrites through a temporary BAM, preserves all records and
per-read tags, and rebuilds an existing BAI/CSI index before replacement.

`fix-bigbed` repairs the embedded `Sample:` autoSQL tag in existing bigBeds (use
when split/genotype-filtered pools loaded side-by-side in FiberBrowser had layers
go missing because their tags weren't distinct). Rebuilds via `bigBedToBed →
bedToBigBed`; needs UCSC `bigBedInfo`/`bigBedToBed`/`bedToBigBed`.

## Output tags

`fiberhmm-apply` writes fibertools-style legacy tags (`ns`/`nl` nucleosomes,
`as`/`al` MSPs, `nq`/`aq` quality). The TF recaller adds spec-compliant `MA`/`AQ`
tags carrying `nuc.Q` / `msp.` / `tf.QQQ` with full LLR scoring. **TF calls live
only in `MA`/`AQ`** (legacy tags carry nucleosomes only, by design); use
`--downstream-compat` to fold TFs into `ns`/`nl` for tools that read only legacy
tags. MA-producing commands also append an advisory `@CO` `MA-TYPES:v1:`
declaration of logical names for fast FiberBrowser layer discovery; it never
creates empty per-read annotations and remains non-authoritative.
`fiberhmm-call` additionally emits an authoritative versioned
`@CO FIBERHMM-CHEMISTRY:v1:` declaration containing assay, enzyme, sequencing
platform, observation mode, and model identity. This lets FiberBrowser choose
the correct likelihood model without relying on filenames. Full schema, header
semantics, byte layouts, and parsing examples are in the
[deep reference](docs/reference.md).

This makes FiberHMM output directly usable across the
[fibertools](https://github.com/fiberseq/fibertools-rs) ecosystem (`ft extract`,
`ft fire`, FiberBrowser).

## Pre-trained models

Supported models are bundled with the package. `--enzyme` plus the
platform-specific `--seq` selects one automatically; `-m` is only for custom
models.

For bundled models, the enzyme/platform registry is authoritative even if stale
model metadata disagrees. Custom models use their embedded `mode`; a custom
model without valid mode metadata now stops with an actionable error instead of
silently being treated as PacBio. The old high-level `--mode` option remains
accepted but hidden for backward compatibility: it emits a warning and, when
supplied, explicitly overrides inference or model metadata. New workflows
should not use it. Low-level `fiberhmm-probs`, training, and transfer commands
still expose mode where it is an actual input to model construction.

| Model | `--enzyme` | `--seq` | Mode | Used by |
|-------|-----------|---------|------|---------|
| `hia5_pacbio.json` | `hia5` | `pacbio` | `pacbio-fiber` | apply / recall-tfs |
| `hia5_nanopore.json` | `hia5` | `nanopore` | `nanopore-fiber` | apply / recall-tfs |
| `ddda_nuc.json` | `ddda` | — | `daf` | apply — **nucleosomes only** |
| `ddda_TF.json` | `ddda` | — | `daf` | recall-tfs — **required 2nd pass** |
| `ddda_nuc_refine.json` | `ddda` | — | `daf` | internal radial-nucleosome likelihood snapshot |
| `dddb_nanopore.json` | `dddb` | — | `daf` | apply / recall-tfs |

For DddA, `fiberhmm-call --enzyme ddda` runs both models in one pass. Hia5 and
DddB each use one model for both nucleosomes and small footprints, so the recall
pass is optional refinement. Older models live in `models/legacy/`
(reproducibility only); custom models load with `-m`. Formats: `.json` (primary),
`.npz`, `.pickle` (legacy, load-only) — convert with `fiberhmm-utils convert`.

For TF calls, `TQ = min(255, round(10 × LLR))` is continuous evidence under the selected
emission table. It is not a posterior probability or calibrated FDR, and its
numeric scale is model-version specific; values at 255 are saturated. The DddA default was selected by
leave-one-library-out scoring of untouched physical mates from 23,388 scDAF
duplexes across twelve libraries; users can retain the continuous TQ values and
sweep stricter thresholds downstream.

### Model-development artifacts

The public preset surface is defined by
`fiberhmm/models/SUPPORTED_MODES.json` and contains only Hia5, DddB and DddA.
Additional chemistry files and low-level encoders may be present for model
development, but they are not accepted as `--enzyme` presets, are not claimed
as supported workflows and require an explicit custom model path.

## Performance tips

1. **Multiple cores** — `-c 8` (or more).
2. **`--io-threads`** — for BAM (de)compression.
3. **`--skip-scaffolds`** — avoid thousands of small contigs in region-parallel mode.
4. **`pip install numba`** — ~10× faster HMM computation.
5. **Pipe directly** — `-o -` into `ft fire`/`samtools` with no intermediate files.

## License

MIT License. See [LICENSE](LICENSE).
