# Per-context false-positive rate calibration

## Question

What's the false-positive rate of the modification caller (PacBio
kinetics model, Nanopore basecaller) on DNA that had no enzyme
treatment, and how does it vary with sequence context?

## Motivation

The caller's merge step needs to distinguish "real hit inside a
nucleosome" (evidence of accessibility, kill the merge) from "FP
hit from basecaller noise" (allow the merge). Using a uniform
baseline misclassifies real DddB hits as noise (because DddB's
low baseline is close to the FP floor). A per-context FP model
gives the right null for each position.

## Methodology

For each untreated control BAM:

1. **Parse m6A calls** (MM/ML tags, threshold 128 by default) or
   **C→T mismatches** (ref=C, query=T; ref=G, query=A), depending
   on modification type.
2. **Extract trinucleotide context** from the REFERENCE at each
   target position (not the query — the query has the mismatched
   base at the center, which biases the context distribution).
3. **Count hits vs opps** per context.
4. **FP rate = hits / opps** per context.
5. **Global rate** used for contexts with <100 observations or
   unseen contexts.

Context size 3-mer chosen from a sweep of 1/3/5/7-mer on the
PacBio Hia5 untreated sample:

| context size | unique contexts | CV (variation in rates) |
|---|---|---|
| 1-mer | 2 (A, T) | 0.06 (no useful bias) |
| **3-mer** | **32** | **0.59** ← sweet spot |
| 5-mer | 513 | 0.69 (marginal gain, sparser) |
| 7-mer | 8193 | 0.83 (overfit, many empty) |

3-mer captures most variance with only 32 parameters.

## Calibrated models (iter-17)

Three models from 2026-04-12 untreated control samples:

### PacBio m6A (for Hia5 PacBio)
- Source: `hia5_pacbio_untreated.bam` (5000 reads)
- Global FP rate: **0.94%**
- Range across contexts: 0.14% to 2.9%
- Highest FP: CTC (2.9%), GTC (2.6%), GAG (2.4%) — CpG-adjacent
- Lowest FP: TTA (0.14%), TAA (0.14%), AAA (0.21%) — poly-A/T
- **Interpretation**: PacBio kinetics model is confounded by CpG
  dinucleotide signatures, producing FP m6A calls at CpG-adjacent
  A/T positions.

### Nanopore m6A (for Hia5 Nanopore)
- Source: `hia5_nanopore_untreated.bam` (5000 reads)
- Global FP rate: **0.70%**
- Range: 0.0% to 1.7%
- Different bias pattern than PacBio (AAC, GAC top FP)

### Nanopore C→T (for DddA/DddB Nanopore)
- Source: `dddb_nanopore_untreated.bam` (5000 reads)
- Global FP rate: **0.82%**
- Range: 0.35% (TGC) to 1.9% (CGA)
- Highest FP at CG dinucleotides (CGA, TCG, ACG) — known Nanopore
  systematic error at CpG positions

## Data files

Each JSON is a dict:
```json
{
  "context_size": 3,
  "target_bases": "AT",
  "global_rate": 0.00938,
  "n_reads": 5000,
  "n_opps": 35909430,
  "n_hits": 336667,
  "rates": {"AAA": 0.00211, "AAT": 0.00153, ...}
}
```

Loaded in the caller via:
```python
from context_fp_model import ContextFPModel
model = ContextFPModel.load('m6a_pacbio_fp_3mer.json')
fp = model.fp_rate_at(pos, query_seq)    # per-position rate
expected = model.expected_fp_hits(positions, query_seq)  # sum
```

## How to regenerate

```bash
python scripts/context_fp_model.py  # (via Python API)

# Or inline:
python -c "
from context_fp_model import ContextFPModel
m = ContextFPModel.from_bam('untreated.m6a.bam',
                             context_size=3,
                             ml_threshold=128,
                             target_bases='AT',
                             max_reads=5000)
m.save('my_model.json')
print(m.summary())
"
```

For C→T models, a custom loop is needed (parse MD tags, not
MM/ML) — see the method used in
`bench/periodicity_compare.py` for reference code.

## Caveats

- **Context size**: 3-mer chosen from PacBio m6A data. Nanopore
  and C→T models may benefit from different context sizes.
  Sweep per-model before finalizing.
- **Target bases matter**: m6A uses `target_bases='AT'` (A on
  fwd, T on rev strand = A on fwd if single-strand BAM), C→T uses
  `target_bases='CG'`. Using the wrong target conflates different
  error distributions.
- **Threshold drift**: Hia5 model uses ML threshold 128. Lower
  thresholds (e.g., 192 for high-confidence calls) would produce
  a different FP distribution and require a new calibration.
- **Drosophila-specific**: these controls are all from Drosophila.
  Human data might have different sequence composition biases
  (though the basecaller error profile should be similar).
