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

**Segments along each molecule**: regular arrays (with the repeat q̂),
irregular stretches and low-information stretches, with a sub-rich flag for
runs dense in short and TF footprints. A factorial HMM over dyad gaps and
footprint composition decodes them; regular segments need 5 dyads and
800 bp. Gaps are read on the dyad grid: a can't-tell, short or long call
between two mono calls hides a dyad, and the run stays regular when the gap
fits two or three repeats. The regular jitter, the irregular spread and the
share of hidden dyads come from the dataset's calibration (real arrays jitter
by 14–17 bp, so the design's 8 bp is kept as a second, tight regular state).
Low-information segments are where marking is below the calibrated floor or
nothing can be measured. Long MSPs (> 300 bp) split molecules: they belong to
the NFR module.

**Window metrics** (a peak, hit, BED row or selection, with an optional
anchor): class mix, NRL, S, phasing in the anchor frame and in the
molecule's own frame (the edge of its open region), per observation (dataset,
haplotype, strand or read set).

**Patches** (scan): regular, irregular, sub-rich and phased stretches where
the fraction of measured molecules passes a threshold.

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

## Tags

With annotation the scan writes three MA groups in molecular coordinates:
`nucsz` (one interval per footprint: coarse code, named code, posterior,
z), `nucseg` (one per segment: state, posterior, q̂) and `nucmol` (one per
record: the molecule scalars). See [BAM tags](../reference/bam-tags.md#nucleosome-groups).
They survive `fiberhmm-lift`. Any tool that re-calls footprints
(`fiberhmm-call`, `-apply`, `-recall-tfs`, `-recall-nucs`, `-merge`) drops
them, because they describe the old calls.

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
