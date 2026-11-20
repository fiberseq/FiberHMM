# Nucleosome sizes, spacing and arrays

`fiberhmm.nuc` measures nucleosome footprints, spacing and array structure on
each molecule of a called BAM, and finds patches where arrays recur across
molecules. It reads FiberHMM's calls (MA `nuc`/`msp`/`tf`, or `ns/nl/as/al`)
and the marks; it does not call footprints. It runs as the `nuc` module of
`fiberhmm-scan` or from Python.

Everything is measured in **molecular coordinates** (bp along the molecule),
so indels and insertions never stretch a length or a gap. Reference positions
are used only to place marks and to pool molecules.

## What it reports

**Per footprint** (nucleosome and TF calls):

- `z`: the footprint's deviation from the density-matched mono-nucleosome
  length. Footprint length depends on mark density (sparse marking lets a
  footprint run to the nearest mark), so each dataset gets a reference curve
  μ_mono(λ) of the mono mode against the local accessible marking rate λ.
- a **coarse call**: short (z < −2), mono, long, multi(2), multi(3+), or
  can't tell. Multi(m) sits near (m − 1) × NRL + μ_mono. A footprint is
  called when the class's posterior under the edge noise is at least 0.8 and
  both edges have at least 3 informative sites; otherwise it is can't tell.
  Footprints that end at a read end are edge-truncated.
- No named (biological) classes are assigned: size output is the measured
  length, z and the coarse call above.

**Dyad eligibility**: a nucleosome call contributes its midpoint when it is
not edge-truncated and its length is within `NucParams.mono_frac`
(0.45–1.25 × pooled NRL; bootstrap 90–230 bp). TF calls do not contribute
dyads. Coarse size class remains a separate annotation, so a can’t-tell
footprint can still supply a dyad. Multimers retain virtual dyads for
continuity, without spacing evidence.

**Per molecule**: repeat length (NRL, median gap between adjacent eligible
dyads), regularity S (dyad autocorrelation at the repeat minus at half of
it), spectral SNR, gap CV, ρ_read (marks / informative sites), λ_acc (marks /
informative sites inside MSPs), class counts and segment fractions. Metrics
below their minimum window (4 gaps for NRL, 6 for S) are marked too short.

**Segments along each molecule** partition its measurable stretches into
four states:

- **regular** (with the repeat q̂): the HMM's posterior P(regular) is at least
  0.7, **or** the model-free rule holds: at least 6 consecutive spacings of
  eligible dyads, each within 25 bp of one or two repeats of the molecule's NRL;
- **disordered**: the spacing evidence beats every regular repeat by at least
  3 nats (under the same edge noise);
- **undetermined**: neither, for example where dyads are too sparse or hidden;
- **low-information**: marking below the calibrated floor.

A factorial HMM over dyad gaps decodes the states; regular runs from the HMM
also need 5 dyads, 800 bp and a log-likelihood ratio of 3 nats against the
other orders. Rule runs stop at MSPs longer than 300 bp. Gaps are read on
the dyad grid: a call outside the dyad length window can hide a dyad, and
the run stays regular when
the gap fits two or three repeats. The regular jitter, the irregular spread
and the share of hidden dyads come from the dataset's calibration. Long MSPs
(> 300 bp) split molecules: they belong to the NFR module. `seg_decode`
switches the decoder (`threshold`, the default; `viterbi`; `posterior`) and
`seg_rule` the rule.

**Disrupted nucleosomes**, overlaid on the segments:

- **undersized runs**: stretches where nucleosome-scale footprints well below
  the density-matched mono size (z ≤ −2; nucleosome calls and TF calls of at
  least 60 bp) are more common than the dataset's background at that density.
  Runs are scored by log-likelihood ratio and need 7 nats, so false runs stay
  at about one per 8,000 footprints whatever the density. No names are given.
- **missing nucleosomes**: inside a regular array, an open stretch of
  120–300 bp between two confident nucleosome calls whose dyads are two
  repeats apart, with marks that favour accessible over protected DNA.
  Longer openings belong to the NFR module.

**Fuzzy or dephased loci** (at anchors): molecules that stay periodic near an
anchor but no longer share a phase. `fiberhmm.nuc.dephasing.anchor_phase`
reports the phase coherence across molecules, the within-molecule periodicity
and the positional dispersion; `dephasing_test` compares anchors with matched
background anchors (for example enhancers against control peaks) within
density and depth strata.

**Window metrics** (a peak, hit, BED row or selection, with an optional
anchor): class mix, NRL, S, state and disruption fractions, phasing in the
anchor frame and in the molecule's own frame (the edge of its open region),
per observation (dataset, haplotype, strand or read set). With background
anchors, the anchor's phase coherence among its matched backgrounds.

**Patches** (scan): stretches where regular arrays, disordered spacing,
undersized runs or missing nucleosomes recur across molecules more than
chance. Chance is each read's own calls placed anywhere along the read: a
test per 500 bp window with an exact null, and Benjamini–Hochberg at FDR 0.1
per kind over every window the scan tests, on all contigs, so the FDR holds
for the scan's whole patch list (`enrich_fdr_scope: "contig"` runs it per
contig instead). The manifest records each kind's threshold and number of
tested windows (`modules.nuc.scan_statistics`). A smaller scan tests fewer
windows, so it can select other patches from the same window statistics. A
read that carries a kind end to end gives no positional evidence. Phased patches mark windows whose molecules share nucleosome
positions; fuzzy patches mark recurrent arrays whose molecules do not.

## Calibration

Dataset-level quantities live in one calibration record
(`fiberhmm.nuc.calibration.v1`): the repeat length, the reference curve, the
gap density, segmentation composition and the thinning results.

The default calibration **thins** the densest quarter of a hash-ranked
population of molecules: marks are dropped at random and the molecules are
re-called by the dataset's own caller with the emission table rescaled to
match. This gives the edge noise and the structural outcomes (kept, merged,
split, dropped) as marks thin, the λ range that both natural and thinned data
cover (footprints outside it are can't tell), and the low-information floor:
the λ_acc below which the coarse can't-tell rate exceeds 50%. Thinning is a
degradation model; it never assigns true lengths.

In a scan, each dataset's calibration population comes from up to 40 fixed
50 kb windows that hold its records, taken in a stable hash order
(`calibration_units`) over every contig of the reference that has reads
(`calibration_scope: "data"`, the default). The sample depends on the data
only, not on what is scanned: a scan of one window, a genome scan and every
extension of a browser view scan calibrate alike, so their molecule results
agree and an extended scan reuses its finished units. A genome scan of
targeted data calibrates on the target. `calibration_scope: "territory"`
draws the windows from the scanned territory instead.

## Tags

With annotation the scan writes four MA groups in molecular coordinates:
`nucsz` (one interval per footprint: coarse code, a reserved byte (always 0),
posterior, z), `nucseg` (one per segment: state, posterior, q̂; plus one row per
undersized run or missing nucleosome), `nucmol` (one per record: the
molecule scalars) and `nucreg` (one per regularity tile: score and local
repeat length). See [BAM tags](../reference/bam-tags.md#nucleosome-groups).
They survive `fiberhmm-lift`. Any tool that re-calls footprints
(`fiberhmm-call`, `-apply`, `-recall-tfs`, `-recall-nucs`) drops them,
because they describe the old calls.

## Regularity along each molecule (`nucreg`)

Single-fibre regularity, not aligned phasing: per molecule, 2 kb windows
every 250 bp of the raw m6A calls (or, with `--nucreg-signal occupancy`, of
nucleosome-call occupancy), mean-removed, Hann tapered and Fourier
transformed. The score is log2 of the strongest power at 140–260 bp periods
over the median power at 50–1000 bp periods; the period of that peak is the
local repeat length. m6A windows need 15 calls. Each window centre owns one
250 bp tile, and each scored tile is one `nucreg` mark.

```bash
# change the window, mask open stretches of 300 bp or more, thin to 25 m6A/kb
fiberhmm-scan --bam calls.bam --genome --modules nuc --nucreg-window 1500 \
    --nucreg-mask-open 300 --nucreg-thin 25 -o nuc_scan
# no regularity layer
fiberhmm-scan --bam calls.bam --genome --modules nuc --no-nucreg -o nuc_scan
```

Masking sets open stretches (gaps between nucleosome calls of at least the
given length, and read ends beyond the outer calls) to the window mean, so
NFRs and enhancers do not drive the score; a window more than half open has
no score. Thinning keeps each call with probability target / density, seeded
from the read name, so a molecule always thins the same way. In Python,
`fiberhmm.nuc.regularity` has `sliding_regularity`, `reference_scores` (the
random-packing comparison: each window's own footprints placed at random,
m6A simulated at the window's marking rates) and `masked_mark_acf`.

## Regularity patches and shared geometries (`nucpatch`, `nuc.geometry`)

`--nucpatch` turns the regularity track into discrete patches, read as the
excess over the dataset's random-packing reference median (computed on the
calibration population, or given with `--nucpatch-ref`). A patch starts at a
tile at least 2.0 log2 above the reference and extends over tiles at least
1.25 above it (at least 2 tiles). A single tile without a score is bridged;
two or more end the patch (no call, not irregular). A patch touching the
molecule's first or last tile is open-ended. Each patch carries its mean
score and median period. With `--nucreg-ends truncated` the cut windows at
read ends (and on reads shorter than the window) are shorter FFTs whose
random-packing null lies lower, so the reference is also computed per window
width (in `step_bp` classes, from the molecules' own cut windows) and each
tile is read against the reference of its own width; a `--nucpatch-ref`
override shifts those per-width references with it.

Edges start on the tile grid (first tile centre − 125, last tile centre +
125). Closed edges then snap to a nucleosome boundary: the nucleosome-sized
footprint (nucleosome call of at least 90 bp) next to the nearest open linker
(at least 300 bp between footprints, or beyond the outermost footprint)
within 1 kb of the grid edge, on the patch side. With no open linker in reach
the grid edge stays. `--nucpatch-snap period` uses linkers whose dyad
spacing is off the patch period by more than 25% instead; `off` keeps the grid
edges. The grid edges are always kept in the Python API.

For BED units, the patches of the unit's molecules (default: records that
overlap the unit, as the browser's panel; `--nucgeo-cohort spanning` keeps
only records spanning it; one record per physical molecule and dataset, the
first primary record) are grouped by start and end: the (start, end) whose ±500 bp box
holds the most molecules becomes a geometry (consensus = median ends), its
members are removed, and the search repeats until a geometry would hold fewer
than 3% of the molecules. Each geometry is a `nuc.geometry` hit with a share
per dataset (`share` of all molecules, and `share_dm`: the share of callable
molecules, standardised to the pooled m6A/kb quintiles when the unit has
several datasets, since scores rise with marking density; one dataset and
`all`: the plain callable share), and each record gets a state: carries, absent, no call (most of the span unscored) or not spanned.

A patch that runs past the unit's edge is clipped there; when less than
`--nucgeo-edge-min-bp` (default 500) of it lies inside the unit it is left out
of the grouping, so short slivers at the edge never form geometries of their
own (the patch itself stays in the `nucpatch` tag).

A geometry is nested in a longer one when it lies inside it (with the
tolerance as slack), shares one end within the tolerance and its other end
moved by more than the tolerance. `--nucgeo-fold` decides what happens to
nested geometries: `none` keeps them separate; `near` (default) folds a nested
geometry into its longest parent only when the other end moved at most
`--nucgeo-near-max-bp` (default 1250); `all` folds every nested geometry.
Folding is transitive (a child joins the top parent) and leaves the grouping
itself unchanged. Molecules carrying only a folded child carry the parent
partly: the hit's `parent` column names the parent, the values add
`share_partial`, and such records get state `0` with reason `partial`.

Snapping parameters: `--nucpatch-search-bp` (window, default 1000),
`--nucpatch-nuc-min-bp` (nucleosome-sized footprint, default 90),
`--nucpatch-open-bp` (open linker, default 300) and `--nucpatch-tol-frac`
(period rule, default 0.25).

```bash
fiberhmm-scan --bam a.bam --bam b.bam --bed loci.bed --modules nuc --nucpatch --annotate -o nuc_scan
```

In Python: `fiberhmm.nuc.patches.call_patches(track, mol.nucs, ref_median=...)`,
`fiberhmm.nuc.geometries.molecule_layout`, `find_geometries` and
`assign_states`.

## Running the scan

```bash
# genome-wide; per-molecule results go into annotated BAMs as MA groups
fiberhmm-scan --bam calls.bam --genome --modules nuc --cores 8 -o nuc_scan

# one region, self calibration instead of thinning, results as tables
echo '{"nuc": {"calibration": "self"}}' > nuc_params.json
fiberhmm-scan --bam calls.bam --region chr1:20000000-21000000 --modules nuc \
    --parameters nuc_params.json --no-annotate -o nuc_region
```

The scan directory holds `nuc/hits` (patches: `nuc.patch.regular`,
`.disordered`, `.undersized`, `.missing`, `.phased` and `.fuzzy`),
`nuc/values` (per-group patch values and the test's numbers), `nuc/molecules` (the per-molecule scalars), `nuc/marks` (one row per
footprint and per segment, when results are written as tables) and
`nuc/extra/calibration.json`. `fiberhmm-scan --schema` lists the module's
parameters and defaults. See [scan format](../reference/scan-format.md).

## Python

```python
from fiberhmm.nuc import (iter_molecules, calibrate_thin, molecule_outputs,
                          window_metrics, Observation)

cal = calibrate_thin('calls.bam', label='sample', cores=8)
cal.save('nuc_calibration.json')
for mol in iter_molecules('calls.bam', region='chr1:20000000-20100000'):
    res = molecule_outputs(mol, cal=cal)
    print(mol.name, res.scalars['nrl'], [s.as_dict() for s in res.segments])

mols = list(iter_molecules('calls.bam', region='chr1:20000000-20100000'))
w = window_metrics(mols, ('chr1', 20_040_000, 20_042_000), anchor=20_041_000, cal=cal,
                   observations=[Observation('HP1', haplotype=1), Observation('HP2', haplotype=2)])
```

`fiberhmm.nuc.mark_acf` measures the repeat length from the marks alone, with
no calls, as a check on the caller.

## Limits

- Hia5 PacBio footprints show one broad mode per density stratum; there are
  no resolved sub-nucleosome classes without a control.
- On recalled BAMs, footprints under 85 bp are TF calls, so the short end
  of the size readout is read from the TF calls.
- The module is experimental in this release: its outputs and defaults can
  change between versions (the scan records the module version).
