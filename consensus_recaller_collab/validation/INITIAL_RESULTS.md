# Initial hierarchical validation results

> Historical one-control stage. The five-control, held-out-locus calibration
> and final report-only operating point supersede these thresholds; see
> [`../PRODUCTION_VALIDATION.md`](../PRODUCTION_VALIDATION.md).

These are July 2026 proof-of-principle results from the report-only prototype.
They are not production calls. The hierarchy emission settings, BIC
approximation, and one-control local-enrichment rule still require held-out
calibration.

## Corrected depth audit

The authoritative files are now used throughout.

| Panel | Region-loaded hard-evidence reads | Molecules after DAF collapse |
|---|---:|---:|
| GM NAPA full PacBio | 132 | 132 |
| GM NAPA targeted DddA | 25,671 | 16,324 |
| GM UBA1 full PacBio | 143 | 143 |
| GM UBA1 targeted DddA | 6,340 | 5,097 |
| fly `ind` five-BAM PacBio pool | 1,941 | 1,941 |
| fly `ind` targeted DddB | 14,600 | 14,304 |
| fly `ind` early siGAF Nanopore | 2,100 | 2,100 |
| fly `ind` late siGAF Nanopore | 373 | 373 |

These are model-usable reads after mapping/tag filters, not raw `samtools view`
overlap counts. For example, the five fly PacBio BAMs have 2,160 raw overlaps
at `ind`, and the DddB BAM has 24,793; the table reports the subset carrying
usable standard hard-call evidence.

## GM12878 cross-assay fine-TF validation

| Panel | TF candidates | independently present in both | DddA only | PacBio only |
|---|---:|---:|---:|---:|
| NAPA | 51 | 42 | 9 | 0 |
| UBA1 | 83 | 55 | 20 | 8 |

PacBio “only” and DddA “only” mean leave-the-other-family-out presence under
the current preliminary hierarchy. Positive-only DddA evidence can confirm a
site but cannot create negative truth.

One opportunity-matched non-overlapping control was available for 49 NAPA and
72 UBA1 candidates:

| Controlled subset | Both-present sites | enriched in both assays | enriched in at least one |
|---|---:|---:|---:|
| NAPA | 40 | 21 | 35 |
| UBA1 | 46 | 26 | 42 |

Among controlled assay-specific sites, 7/9 NAPA and 13/18 UBA1 DddA-only sites
were locally enriched in DddA. Seven of eight controlled UBA1 PacBio-only sites
were locally enriched in PacBio. These reciprocal classes are useful
calibration material rather than failures to be forcibly reconciled.

NAPA also contains a clear sequence-limit example: one strongly DddA-supported
interval has zero PacBio A/T opportunities in the tested footprint. The correct
PacBio label is `unresolved_no_anchor`, not absent.

## Nine-island fly DddB/PacBio batch

All nine high-depth DddB islands in the manifest were evaluated against the
five-BAM PacBio pool with one matched local control per TF candidate where
possible.

| Fly panel | TF candidates | DddB supported | PacBio locally enriched | DddB locally enriched | enriched in both | DddB-only seeds |
|---|---:|---:|---:|---:|---:|---:|
| chr2L:15.48 Mb | 101 | 93 | 96 | 50 | 47 | 1 |
| chr2R:11.75 Mb | 53 | 52 | 47/52 controlled | 19/52 | 18/52 | 1 |
| chr2R:25.23 Mb | 71 | 62 | 62 | 34 | 29 | 0 |
| chr3L:1.46 Mb | 44 | 40 | 44 | 22 | 22 | 1 |
| `ind` | 102 | 93 | 97 | 43 | 40 | 3 |
| `zen` | 79 | 75 | 70 | 43 | 40 | 4 |
| `hb` | 86 | 80 | 77 | 38 | 33 | 0 |
| chr3R:28.59 Mb | 57 | 53 | 47 | 31 | 28 | 0 |
| `tll` | 100 | 80 | 91 | 42 | 38 | 0 |
| **Total** | **693** | **628** | **631/692 controlled** | **322/692** | **295/692** | **10** |

PacBio has positive raw site evidence for all 693 candidates at the preliminary
logBF threshold, but the control comparison retains 631. DddB has positive raw
evidence for 628 and local enrichment for 322. This difference is appropriate:
most candidates were discovered by PacBio, so DddB is an independent support
assay rather than an equal boundary voter. A total of 658/692 controlled sites
are locally enriched in at least one assay.

The ten DddB-only discovery seeds stratify naturally:

- Six are locally enriched in both PacBio and DddB: all three `ind` sites, the
  chr3L:1.46 Mb seed, the chr2L:15.48 Mb seed, and `zen.tf077`.
- Two overlapping `zen` seeds are DddB-enriched and have positive PacBio raw
  evidence, but their selected PacBio controls are stronger. They are
  one-assay-focal review candidates, not automatic rescues.
- `zen.tf014` is weaker than its selected control in both assays and belongs in
  the ambiguous/reject tier despite a large absolute BF.
- The chr2R:11.75 Mb seed is strongly supported in both raw models but sits at
  the target edge, where no non-overlapping matched shift is available.

Representative DddB-only seeds show the intended borderline-PacBio pattern:

| Candidate | Interval | PacBio explicit reads | PacBio logBF / local delta | DddB logBF / local delta |
|---|---|---:|---:|---:|
| `ind.tf016` | chr3L:15,039,948–15,040,022 | 12 | 3,573 / +3,411 | 4,640 / +4,618 |
| `ind.tf017` | chr3L:15,040,154–15,040,235 | 11 | 221 / +170 | 10,415 / +7,072 |
| `ind.tf077` | chr3L:15,048,129–15,048,210 | 21 | 3,143 / +3,132 | 9,467 / +9,382 |
| chr3L:1.46 Mb `.tf001` | chr3L:1,459,953–1,459,986 | 11 | 171 / +134 | 2,622 / +1,477 |
| chr2L:15.48 Mb `.tf001` | chr2L:15,474,793–15,474,847 | 25 | 429 / +187 | 2,918 / +1,525 |
| chr2R:11.75 Mb `.tf053` | chr2R:11,758,790–11,758,848 | 3 | 86 / no control | 6,899 / no control |
| `zen.tf077` | chr3R:6,759,694–6,759,764 | 125 | 3,893 / +2,850 | 21,622 / +16,173 |

These are not unsupported calls. Even the three-PacBio-read edge example has
positive raw PacBio likelihood at the DddB geometry; the consensus machinery
is deciding where a borderline existing signal merits revisiting.

## Fly `ind` three-assay proof of principle

The 12 kb target island contains 102 focal TF candidates and 94 broad PacBio
nucleosome candidates.

- PacBio raw likelihood supports all 102 TF candidates under the current
  presence threshold; 97/102 beat their opportunity-matched PacBio control.
- DddB has positive site evidence at 93/102, but only 43/102 beat the local
  DddB control. This is expected because 97 candidates were nominated by
  PacBio, not by DddB.
- Forty sites beat their matched control in both PacBio and DddB.
- Early siGAF Nanopore supports 97/102 and late siGAF supports 70/102 by raw
  likelihood, but neither sample votes in truth.

DddB nominates three sites that did not meet the PacBio discovery-support
cutoff. All three are independently strong and locally enriched in PacBio:

| Interval (dm6, chr3L) | DddB discovery support | PacBio logBF / local delta | DddB logBF / local delta |
|---|---:|---:|---:|
| 15,039,948–15,040,022 | 148 | 3,573 / +3,411 | 4,640 / +4,618 |
| 15,040,154–15,040,235 | 105 | 221 / +170 | 10,415 / +7,072 |
| 15,048,129–15,048,210 | 140 | 3,143 / +3,132 | 9,467 / +9,382 |

PacBio already has 11–21 explicit high-TQ reads near these sites, just below
the default pooled discovery threshold, while the raw A/T mixture finds a
larger focal component. This is the intended consensus-recall case: depth in a
complementary assay identifies where to revisit borderline, not unsupported,
PacBio molecules.

## Strand rescue at the first two fly sites

After retaining nucleosome as a competing source state:

- Site 1 has 154 high-confidence DddB GA calls versus 10 CT calls. The GA prior
  proposes 156 additional CT TF calls at posterior at least 0.95.
- Site 2 has 112 CT calls versus 6 GA calls. The CT prior proposes 12 additional
  GA calls.
- Clean opportunity-matched shifts produce 0 and 1 proposals, respectively.
  A different fixed +100 bp shift landed on another real protected site and
  produced 107 proposals, demonstrating why arbitrary single shifts are unsafe.

Pooled threshold-248 Nanopore has almost no explicit calls at these externally
anchored intervals. With `N` retained in the source mixture, it proposes no
rescues at site 1 and four at site 2 (two per orientation). A clean site-2 shift
produces one. The fitted Nanopore source populations are 71–81% nucleosome and
12–18% TF, rather than the spuriously 85–91% TF obtained when `N` was omitted.

## Composite nucleosome deconvolution

Population overlap did not justify indiscriminate nuc breaking:

- At the first two exact DddB intervals, 863 individual PacBio 90–220 bp nuc
  calls were tested. None became TF-favored at neutral nuc odds.
- At the third interval, 643 nuc calls were tested. None became TF-favored;
  one reached review-level TF posterior 0.418.

The focal sites therefore appear mostly as mutually exclusive TF and true
nucleosome populations, not wholesale fused-nucleosome errors. The composite
pass correctly remains stricter than site-level TF rescue.

## What is ready versus unresolved

Ready as a research prototype:

- authoritative manifest and coverage audit;
- assay-axis hierarchy and leave-one-family-out adjudication;
- full-read DAF molecule-family collapse;
- nested A/TF/N hard-call mixture evidence;
- per-strand diagnostics and guarded cross-strand priors;
- explicit external site nomination without external boundary authority;
- opportunity-matched local controls;
- conservative 90–220 bp composite nuc review.

Still required before a production caller:

- multiple-control empirical nulls, independent biological negative loci, and
  replicate-held-out calibration;
- calibrated occupancy-frequency comparison rather than binary presence only;
- final arbitration between TF strand rescue and composite-nuc review;
- optional DddA radial nucleosome evidence after the core cross-assay model is
  stable;
- a no-write proposal format followed by explicit user review before any BAM
  tag mutation is implemented.
