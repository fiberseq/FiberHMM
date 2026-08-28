# Consensus Recaller (`cr`) — Session Handoff

> **Superseded ownership decision (2026-07-14): do not execute the “Next
> steps” below.** CR has been extracted from FiberHMM and is now specified as an
> annotation-only regional FiberBrowser analysis in
> [`../docs/FIBERBROWSER_CONSENSUS_RECONSTRUCTION.md`](../docs/FIBERBROWSER_CONSENSUS_RECONSTRUCTION.md).
> In particular, do not wire `reconstruct_nuc_call()` into a FiberHMM CLI.
> Chemistry-aware physical-strand rescue remains in FiberHMM as
> `fiberhmm-strand-rescue`.

_2026-07-14. Read this before touching `cr`. Several earlier conclusions in this
directory's other docs were overturned during this session — see "What I got
wrong" below._

## The core reframe (this took several wrong turns to land)

`cr` is an **alternative-proposal generator, not a decider.** It does not try to
prove a `nuc` call wrong — the per-base LLR already handles evidence-based
splitting, and that is the baseline caller's job. `cr` asks: **given the alt
states observed at this site, can this nuc call be reconstructed as a union of
common alt states?** Seeing `-[]---` and `---[]-` separately, then `-[  ]-`,
suggests the last is the union of the first two.

Critical consequence: **the merged case has no gap.** So accessible-separator
evidence is the *wrong* thing to hunt for. The molecule's own bases can only ever
**veto** (m6A inside a proposed footprint), never confirm. Support comes from
**geometry + population frequency**, not from beating N. Do not add
0.95-posterior or decoy gates to this pass — a merged union claims no
per-molecule geometric specificity, so a decoy margin correctly collapses to ~0.
That is not a failure; it means decoy-gating is the wrong control here.

## What was actually broken (and fixed)

1. **The edge kernel** penalized the TF hull for not matching the nuc call's
   boundaries (Gaussian, up to −68 nats, **correlation 0.982** with rejection of
   TF at Homie). But geometry-vs-boundaries is the *right* idea backwards — the
   union *predicts* the block's extent while a nucleosome could sit anywhere.
   That asymmetry is the actual evidence. `reconstruct_nuc_call()` now scores
   exactly this.

2. **m6A-deposition failures** produced 20–33 kb "nucleosomes" (molecules with
   <2% m6A rate → median nuc 20,145 bp). Fix (user's call): **drop reads with
   <5 footprints.** Same disease as the DAF hyper-deamination filter.

3. **Eligibility rule** dropped 33–57% of blocks by requiring a template
   *center* inside the block. Changed to *overlap*. **This is wired into the CLI.**

## What I got wrong and corrected mid-session (flag these to the checker)

- **The nuc recaller is fine; the LLR is fine.** I nearly blamed the recaller for
  big nucs. Checked: >220 bp blocks have **zero internal accessibility** (median
  best accessible-run LLR = 0.00 at ≥5 opportunities, identical to
  mononucleosomes). Nothing to split. The 220 bp cap is defensible.
- **The `/mnt/g/v3seg_mp` BAMs were already properly recalled.** A full
  from-scratch `fiberhmm-call v2.16.3` re-run changed almost nothing (nuc median
  157→156, >220 bp 17.2%→16.9%).
- **N+TF state: added, then removed** at user's push-back. Support scores were
  0.01–0.07 — indistinguishable from plain N. A 250 bp protected block with no
  internal accessibility is a nucleosome, not nuc+edge-TF. Inventing that state
  manufactured a composite the bases do not support. Cap reverted 400→220.
- Earlier in the session I also reverted a **resolvable-mass gate** and a
  **circular empirical-N-geometry prior** (it fit N's geometry from the same
  `read.nucs` under test) — both were my own over-corrections. If you see
  references to those in git history, they are dead ends.

## Current state of the code

**Wired into the CLI (`analyze_composite_deconvolution`):**
- Overlap-based eligibility (not center-based)
- Cap at 90–220 bp
- Per-molecule deamination calibration (from an earlier part of this session)

**NOT wired — the main gap:** `reconstruct_nuc_call()` exists, is validated
against real data, but is **not** the CLI's scorer. `analyze_composite_
deconvolution` still calls the old `score_composite_candidate` (a segment model
built for the wrong problem). So `fiberhmm-consensus-recall` does not yet emit
reconstruction proposals — the reconstructor has only been driven directly from
scripts.

**Tests:** 87 consensus pass, 646 full regression pass, 3 skipped.

## Validated results (reconstructor driven directly, TF-union only, new basis)

| locus  | templates | blocks scanned | reconstructible | support median | >0.5 |
|--------|----------:|---------------:|----------------:|---------------:|-----:|
| Homie  |         8 |            457 |      102 (22%)  |          0.25  |   0% |
| Nhomie |         4 |            354 |      138 (39%)  |          0.58  |  55% |

Supporting evidence the reconstruction is real, not chance: Homie's focal "nucs"
are **123 bp vs 150 bp background (Mann-Whitney p = 8e-44)** — too short to be
nucleosomes. Nhomie's are 146 vs 147 bp (n.s.) — mostly genuine nucleosomes. The
two elements genuinely differ; do not expect symmetric behavior.

## Data locations

- **New basis (use this):** `/mnt/g/v3seg_basis/2-4hr_{4,6,7,9,11}.bam` — full
  recall + <5-footprint filter, indexed.
- Old basis: `/mnt/g/v3seg_mp/` (raw, unfiltered, kinetics-heavy — do not use).
- Elements:
  - Homie  `chr2R:9988750-9989118`
  - Nhomie `chr2R:9972790-9973386`
  - SF1    `chr3R:6853644-6855684`
  - SF2    `chr3R:6869630-6871751`
- Stale panels/BAMs in `consensus_validation_outputs/regenerated_20260714/` and
  `consensus_visualization_outputs/regenerated_20260714/` were built by the
  **old** edge-kernel model — do not cite.

## Next steps (in order)

1. Wire `reconstruct_nuc_call` into `analyze_composite_deconvolution`, replacing
   `score_composite_candidate`.
2. **Delete** the decoy/scenario/rescore plumbing (~200 lines) — meaningless for
   this pass, do not adapt it.
3. Add tests for the reconstruction path.
4. Regenerate the four panels + shadow BAMs on `/mnt/g/v3seg_basis/`.
5. Loosen default site discovery — templates you never find are proposals you can
   never make (Nhomie went 8%→16%→39% reconstructible as discovery loosened).

## Open questions for the checker to weigh in on

- **Reconstruction tolerance** is fixed at 20 bp — should be a swept parameter,
  and 20 vs 10 materially changes yield.
- **The `nuc_support` prior term:** N is scored as
  `log(nuc_prior_odds) + log(nuc_support + prior_alpha)` against each config's
  `log(support) + geometry + veto`. This puts N and reconstructions on the same
  frequency footing, but the exact prior form is a judgment call worth a second
  opinion.
- Nothing is committed; `consensus_recaller_collab/` is still untracked.

## Related docs (context, but predate the reconstruction reframe)

`CHANGES_2026_07_14.md` and `CALIBRATION_AUDIT.md` describe the *earlier* fixes
(per-molecule deamination calibration, validation q-value studentization,
annotator overwrite guard). Those fixes are still valid, but both docs predate
`reconstruct_nuc_call` and describe the superseded scoring model. Treat their
proposal-count tables as historical.
