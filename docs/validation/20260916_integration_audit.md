# Claude / AGY audit review — 2026-09-16

Both requested external audits completed. Raw reports are retained in the local `release_validation_20260916` handoff. These are reviewer claims, not automatically accepted findings.
The repositories were dirty before this work; no reset, commit, or publication
was performed. A supplementary Claude inline follow-up was canceled after the
full repository audit returned; it supplied no additional findings.

## Accepted and fixed

- Duplex merge dropped `dm`/`mv` and dropped `mg` on `pm:D` pairs. All are retained;
  reciprocal partners must agree on the model/score/margin provenance.
- Joint molecules entered single-strand evidence encoding. Shared BAM preparation
  now uses the existing both-strand encoder with MA coverage masks; one joint
  read contributes one unit. Both source names follow the evidence, presentation,
  CL-CR deduplication and frozen-transfer training-exclusion paths.
- Browser's deamination opportunity rug was absent on joint reads. The rug now
  follows each channel's MA coverage, including missing coverage/deletions.
- The merge probability threshold did not reach consensus construction. It now
  reaches the same MM/ML extraction used to determine pairing flavor.
- Empty recall annotations could remove MA before restoring channel coverage.
  Restoration now works when no TF/nucleosome/MSP annotation remains.
- Reverse MM/ML positions were converted to stored SEQ/reference orientation,
  but the deamination flavor remained molecular-oriented. Confirmed with real
  pysam records and corrected in FiberHMM extraction/pairing and Browser parsing.
- Browser `all_fp` analysis omitted AN/layer identities. Identity now follows
  the same deduplicated family-bearing interval as its quality metadata.
- Named family outlines could match an unrelated unnamed local slot; named
  metadata was also omitted for ordinary assignments. Explicit AN identity is
  preserved for color/display matching, with FI fallback for unnamed lookups.
- Backend cross-family confidence ignored `xq`. It now reads cross support;
  producer-declared unavailable confidence remains unavailable.
- Family target-coverage calculations used the outer hull of observed calls.
  They now use the family model interval when present.
- Repeated catalog definitions for the same frozen family at distinct windows
  were rejected solely for different bounds. Compatible definitions now union
  their bounds; conflicting semantic definitions still fail.
- Real dataset addition attempted to create BAI despite an existing CSI index.
  CSI is now recognized in the addition path as well as the compact loader.

## Findings not reproduced / intentionally retained behavior

AGY 1.1 and 3.1 (universal unit-key crashes): `prepare_input` prefixes unit and
ledger IDs; the lookup keys agree. Actual BAM transfer/export and all archived
replays pass. Changing these lookups as suggested would introduce a defect.

AGY 1.2 (missing fold ID): native preparation constructs the evidence/fold
identity before parent scoring. Parent replay tests pass.

AGY 1.3 (zero-origin provenance loss): `pool_payloads` already supplies genomic
provenance; the conditional block only shifts nonzero historical model origins.
Both orientation paths are covered by transport tests.

AGY 1.4 (alias kind / double-colon parent ID): current producers reuse bounded
parents and name them `P:<id>`. Child IDs use a different explicit convention.
The proposed hypothetical mixed-kind alias is not produced by this workflow.

AGY 2.1 (missing duplex identity): `cs` is decoded by the BAM evidence loader
into `physical_source_names`; molecule keys include both names. New tests cover
merged/source exclusion and deduplication.

AGY 2.2 and Claude 2 (unresolved pairs): silently counting both strands is not
acceptable. Merge preserves failed source records by default, and population
CR rejects live unmerged pair annotations. This remains deliberate. The error
and workflow documentation explain `fiberhmm-merge --recall --pairs-only` when
only successful joint molecules should proceed; that option also excludes
ordinary unpaired reads. No input reads are silently discarded by default.

Claude 6 (ordinary callers on merged reads): the supported joint recall path is
`fiberhmm-merge --recall`, not a later single-strand caller. This limitation is
now explicit in the workflow documentation. Broad support for re-running every
single-strand CLI on joint BAMs is not claimed.

## Validation and practical limits

Final full runs: FiberHMM 2,072 passed / 7 skipped / 26 deselected; Browser backend
1,450 passed / 73 skipped; frontend 772 passed. The reverse-MM fix followed those
full runs and passed 73 focused FiberHMM and 66 focused Browser tests. Skips and
deselections follow the existing suite configuration; they are not counted as
passes. Browser emitted one pre-existing synthetic clustering convergence warning.

The final wheel passed 61 tests outside the checkout. Two real Chromium tests pass against that installed wheel, covering the automatic
full caller, consolidation controls, checkpoint continuation, family cart,
curation/history, MA-tagged subset export, and reload through the actual API.
Three real SRR33130336 rotational-model duplex pairs pass pairing → full merge
recall → joint CR preparation → actual Browser loader, with both opportunity
channels visible in overlapping coverage. This small integration test does not
re-estimate duplex pairing accuracy.

The fresh virtual environment uses system-site-packages for existing numerical
and web dependencies. FiberHMM itself is installed from a newly built wheel;
Browser is copied outside the adjacent development tree. This tests package
contents, console entry points and installed-engine integration, not a fresh
solver/install of every dependency or every supported OS.

The 22 archived GATA/TAL1, CTCF and Pol II/PIC windows are numerical replay
regressions, not a new biological accuracy estimate or a full discovery rerun.
