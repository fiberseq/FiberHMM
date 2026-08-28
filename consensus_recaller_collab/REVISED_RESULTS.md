# Revised two-pass consensus prototype — 2026-07-12

> Historical v3/v4 model-development record. The self-contained hierarchical
> v6 N-versus-TF-complex model and complete shadow-callset semantics supersede
> the composite counts below. The former leave-one-library-out inference was a
> modeling error; independent libraries are now evaluation-only. Boundary
> decoys, molecule-collapse sensitivity, and the current report-only operating
> point are documented in
> [`PRODUCTION_VALIDATION.md`](./PRODUCTION_VALIDATION.md).

This is a report-only research implementation. It reads standard sequence,
MM/ML, and existing MA annotations and never rewrites a BAM. Production
FiberHMM and the separately developed merge tool are untouched.

## Two distinct passes

1. **Strand rescue** asks whether strong evidence on one physical strand can
   promote weak-but-supported TF evidence on the other strand. It runs for
   DddA/DddB DAF and stranded Nanopore Hia5. Nuc rewriting is disabled in this
   pass.
2. **Composite deconvolution** asks whether one nuc-like protected block is
   better explained by a population-recurrent TF configuration. It runs
   independently and can therefore follow strand rescue, but rescued calls do
   not train its templates or priors.

PacBio Fiber-seq uses only the second pass: both `A+a` and `T-a` channels are
already present on every HiFi molecule, and forward/reverse alignment flags are
never treated as biochemical strands.

## Composite model

- Focal templates and configuration geometries are frozen from explicit,
  high-TQ TF calls only. Ambiguous nucs supply neither TF nor N training labels.
- Each target block is tested against every sufficiently supported projected
  source configuration, including arbitrary non-overlapping TF subsets.
- TF likelihood marginalizes over complete source-observed component
  intervals, not median site rectangles.
- Multi-TF configurations may have an accessible separator or a continuously
  protected bridge/complex. The bridge prior is explicit and sensitivity
  tested. A single TF is never expanded to fill a nuc; single-TF-plus-gap must
  explain the gap from target hard calls.
- N likelihood uses 90--220 bp background nuc lengths collected away from
  focal templates and integrates over every position capable of covering the
  focal sites. N is not given the exact target HMM interval for free.
- Latent N and TF geometries use the same Gaussian call-edge kernel relative to
  the nominated MA block. This is a symmetric approximate call-generation
  likelihood; shifted-boundary decoys calibrate its locus specificity.
- Reports expose expected TF posterior mass as well as MAP and 0.95-threshold
  counts. Global N:TF prior odds and protected-gap priors remain separate from
  all likelihood terms.

## Exact-boundary PacBio panel

Five 2--4 hr PacBio BAMs in `/mnt/g/v3seg_mp` were pooled virtually. Sites
required at least 50 explicit high-TQ calls and configurations required 20
source examples. Composite deconvolution has a hard 220 bp ceiling; only
current nuc blocks of 90--220 bp were tested, and focal blocks above the ceiling
are reported as skipped. The table uses neutral global N:TF prior odds,
protected-gap prior 0.25, review posterior 0.5, and strict posterior 0.95.

| Boundary | Candidate blocks | Skipped >220 bp | MAP TF configuration | Expected TF mass | Strict composite |
|---|---:|---:|---:|---:|---:|
| Nhomie | 198 | 223 | 72 | 69.2 | 13 |
| Homie | 574 | 155 | 64 | 63.1 | 37 |
| SF1 | 1,590 | 614 | 88 | 88.5 | 55 |
| SF2 | 1,618 | 896 | 153 | 146.3 | 43 |

The calls are concentrated in multi-site classes. Isolated-site classes at
Nhomie, SF1, and SF2 contribute essentially zero posterior mass despite
hundreds of candidates, which is an important internal specificity control.

### Homie paired-site class

The clearest prespecified class contains 103 current 90--120 bp nuc calls
spanning the Homie TF pair:

- 55 are MAP paired-TF configurations;
- expected paired/composite posterior mass is 53.1 molecules; and
- 34 exceed posterior 0.95.

Leave-one-BAM-out learning gives 55 MAP, 52.4 expected, and 34 strict across
the same 103 short candidates. Every fold has at least one strict paired call.
Thus candidate molecules do not need to nominate their own sites or influence
their configuration library.

Boundary-shift controls at neutral N odds:

| Configuration library | MAP composite | Expected TF mass | Strict |
|---|---:|---:|---:|
| True Homie geometry | 55 | 53.1 | 34 |
| Shifted -25 bp | 20 | 19.6 | 6 |
| Shifted +25 bp | 17 | 17.7 | 0 |
| Shifted -50 bp | 7 | 7.3 | 4 |
| Shifted +50 bp | 0 | 0.24 | 0 |

Shifts of 75 bp or more in either direction have essentially zero posterior TF
mass. The asymmetric near decoys reflect local sequence/protection structure.

The same test is specific at the other three requested boundaries. Values
below are expected TF mass / strict calls in each prespecified 90--120 bp
multi-site class; “best” is the larger result from the positive and negative
shift.

| Boundary class | Short candidates | True | Best +/-25 bp | Best +/-50 bp |
|---|---:|---:|---:|---:|
| Nhomie sites 1--4 | 25 | 17.74 / 3 | 8.73 / 0 | 0.003 / 0 |
| Homie sites 1--2 | 103 | 53.07 / 34 | 19.56 / 6 | 7.27 / 4 |
| SF1 sites 3--4 | 66 | 50.63 / 40 | 12.24 / 1 | 0.008 / 0 |
| SF2 sites 2--3 | 57 | 43.69 / 36 | 14.61 / 3 | 0.009 / 0 |

These are coordinate-shift controls, not independent biological negative
loci. Production calibration still needs held-out negative loci.

Global N-prior sensitivity for the 103 short pair blocks:

| N:TF prior odds | MAP composite | Expected TF mass | Strict |
|---:|---:|---:|---:|
| 0.1 | 63 | 63.1 | 49 |
| 1 | 55 | 53.1 | 34 |
| 10 | 40 | 38.9 | 18 |
| 100 | 21 | 22.7 | 4 |

Protected-gap sensitivity at neutral N odds:

| Protected-gap prior | MAP composite | Expected TF mass | Strict |
|---:|---:|---:|---:|
| 0 | 43 | 42.4 | 23 |
| 0.25 | 55 | 53.1 | 34 |
| 0.5 | 57 | 54.9 | 36 |

The signal therefore does not require a protected bridge, although allowing a
touching/continuous complex increases support as expected.

## Independent strand-rescue checks

- The source state fit now always retains `N` as a competitor even when nuc
  rewriting is disabled. This prevents a protected nucleosome population from
  being forced into the TF prior while keeping strand rescue and composite nuc
  deconvolution independent.
- At the first two DddB-only nominated sites in the full `ind` panel, corrected
  strand rescue proposes 156 CT calls from a GA-dominant site and 12 GA calls
  from a CT-dominant site. Clean opportunity-matched shifts produce 0 and 1.
- Pooled fly Nanopore Hia5 uses the requested `ML >= 248` hard-call threshold.
  With externally anchored geometry and `N` retained, it proposes no calls at
  the first site and four at the second; a clean shift produces one.
- No nuc candidate is examined when composite deconvolution is disabled.

These remain implementation checks, not final DAF/Nanopore calibration panels.
See `validation/INITIAL_RESULTS.md` for molecule counts and cross-assay controls.

## Items resolved by the production-validation pass

- MA edge bandwidth is now selected by held-out explicit source configurations;
  with five PacBio inputs, the target BAM is excluded from its configuration
  prior. True geometry must also beat ±25/±50 bp boundary decoys by 3 log units.
- A 10:1 N:TF prior is the actionable scenario; 1:1 and 100:1 remain sensitivity
  outputs. Four prespecified boundary loci and 15,920 decoy scores calibrate the
  operating point.
- Keep blocks above 220 bp out of this pass. They should be split by the
  upstream nuc recaller; if they survive that pass, the focal evidence is too
  weak for trustworthy composite deconvolution.
- Full-read hard-deamination family collapse now runs before amplified-DAF
  strand/configuration fitting and has a dedicated threshold-sensitivity audit.
- DAF/Nanopore preserve physical/read-strand geometry, require focal source
  enrichment, and emit strong versus review proposals separately.
- The nested A/TF/N existence model now uses five opportunity-matched controls,
  deduplicated genomic pseudo-nulls, other-locus calibration, and per-library
  PacBio replication.

The remaining intentional boundary is BAM mutation: inference writes atomic
JSON/TSV proposals and never changes `MA`, `MM`, or `ML` tags.
