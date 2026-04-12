# v3 caller analyses

Permanent home for analysis scripts, figures, and summary data that
support the v3 caller's design decisions. Each subfolder contains
one self-contained analysis with:

- `scripts/` — the scripts that generated the outputs
- `figures/` — final PNG/PDF figures
- `data/` — summary TSVs, JSONs, BEDs
- `README.md` — question, methodology, results, caveats, how to
  reproduce

Everything committed here should be defensible for a paper —
methodology clearly described, caveats enumerated, reproducibility
documented.

## Current analyses

| folder | question | status |
|---|---|---|
| `rotational_periodicity/` | Is the 10.4 bp DAF pair-correlation enzyme-intrinsic or chromatin-phased? | **Resolved** — chromatin-phased. Calibration figure locked for iter-17 merge correction. |
| `context_fp_calibration/` | What's the per-context FP rate of modification calls on untreated controls? | **Resolved** — 3-mer context, 3 platform models. 15× range across contexts. |
| `nucleosome_penetration/` | What fraction of accessible enzyme activity penetrates nucleosome bodies? | **In progress** — preliminary 0.36 from amplicon bulk troughs. Needs tightening (prominence filter, FP subtraction). |
| `snp_detection/` | How do we exclude SNPs that look like enzyme hits? | **Resolved** — 95% hit-fraction threshold in bulk pileup. |
| `v2_v3_comparison/` | How does v3 differ from v2 HMM on the same data? | **Resolved** — v3 halves overmerge, adds explicit TFs. |

## Convention

When adding a new analysis:

1. Create `analyses/<name>/` with the three subfolders + README
2. Put scripts under `scripts/` (NOT in `bench/` anymore)
3. Save figures with descriptive names (e.g.
   `daf_penetration_9amplicons.png`, not `fig1.png`)
4. Save data alongside figures (TSV for tables, JSON for summary stats)
5. README must include: question, methodology, results, caveats,
   how-to-reproduce command
6. Commit source + outputs together so figures are always
   regeneratable from the state of the repo at that commit

## Housekeeping

- Figures are kept in git because they're small and we want them
  versioned alongside the code that produced them. Large data files
  (BAMs) stay outside git — scripts use absolute paths or symlinks.
- Each analysis should be runnable from a clean checkout given the
  standard data locations. If a BAM path is hardcoded, document it
  in the README.
