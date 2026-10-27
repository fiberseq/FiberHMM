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
- **Named classes** (hexasome, octasome, chromatosome) only with an absolute
  calibration from a control of known core length (601 arrays or
  spike-ins). Without one the size mixture is not identifiable, and no names
  are given.

**Per molecule**: repeat length (NRL, median gap between adjacent mono
dyads), regularity S (dyad autocorrelation at the repeat minus at half of
it), spectral SNR, gap CV, ρ_read (marks / informative sites), λ_acc (marks /
informative sites inside MSPs), class counts and segment fractions. Metrics
below their minimum window (4 gaps for NRL, 6 for S) are marked too short.

**Segments along each molecule** partition its measurable stretches into
four states:

- **regular** (with the repeat q̂): the HMM's posterior P(regular) is at least
  0.7, **or** the model-free rule holds: at least 6 consecutive spacings of
  mono dyads, each within 25 bp of one or two repeats of the molecule's NRL;
- **disordered**: the spacing evidence beats every regular repeat by at least
  3 nats (under the same edge noise);
- **undetermined**: neither, for example where dyads are too sparse or hidden;
- **low-information**: marking below the calibrated floor.

A factorial HMM over dyad gaps decodes the states; regular runs from the HMM
also need 5 dyads, 800 bp and a log-likelihood ratio of 3 nats against the
other orders. Rule runs stop at MSPs longer than 300 bp. Gaps are read on the dyad grid: a can't-tell, short or
long call between two mono calls hides a dyad, and the run stays regular when
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
per contig. A read that carries a kind end to end gives no positional
evidence. Phased patches mark windows whose molecules share nucleosome
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
50 kb windows of the scanned territory that hold its records, taken in a
stable hash order (`calibration_units`), so the sample does not depend on the
tiling and a genome scan of targeted data calibrates on the target.

## Tags

With annotation the scan writes three MA groups in molecular coordinates:
`nucsz` (one interval per footprint: coarse code, named code, posterior,
z), `nucseg` (one per segment: state, posterior, q̂; plus one row per
undersized run or missing nucleosome) and `nucmol` (one per record: the
molecule scalars). See [BAM tags](../reference/bam-tags.md#nucleosome-groups).
They survive `fiberhmm-lift`. Any tool that re-calls footprints
(`fiberhmm-call`, `-apply`, `-recall-tfs`, `-recall-nucs`) drops them,
because they describe the old calls.

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
