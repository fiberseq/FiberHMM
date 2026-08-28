# Archived consensus-recaller research workspace

This directory preserves the CR/SR prototype code, validation reports, and
historical outputs developed through 2026-07-14. It is **not installed as part
of FiberHMM**, is not included in the default test suite, and its provisional
`fiberhmm-consensus-*` commands have been removed.

The two ideas now have separate owners:

- Consensus reconstruction (CR) is an annotation-only, regional FiberBrowser
  feature. The current implementation handoff is
  [`../docs/FIBERBROWSER_CONSENSUS_RECONSTRUCTION.md`](../docs/FIBERBROWSER_CONSENSUS_RECONSTRUCTION.md).
- Physical/read-strand rescue (SR) is chemistry-aware FiberHMM functionality.
  Its production modules are `fiberhmm/inference/strand_rescue.py` and the
  `fiberhmm-strand-rescue*` commands.

Do not wire `reconstruct_nuc_call()` or `score_composite_candidate()` back into
FiberHMM. The old CR reports and BAMs remain useful as development history and
test data, but their algorithms and command names are superseded.

