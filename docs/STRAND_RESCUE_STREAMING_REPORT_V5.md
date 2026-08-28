# Strand-rescue v5 streaming report and action contract

Status: implementation specification, July 2026.

This document defines the production report and annotation contract for
targeted, high-depth strand rescue. It addresses the fact that the scientific
result is small enough to express as per-read actions, while the current v4
diagnostic representation stores several kilobytes for every matched ordinary
call. At full targeted DAF depth, that distinction is the difference between a
sub-gigabyte compressed action artifact and a report that cannot fit in memory.

The v5 design does not alter the strand-rescue model, site discovery, hard-call
evidence, nucleosome cardinality rule, or `nuc_sr.QQQ` / `tf_sr.QQQ` layer
semantics. It changes how finalized actions and diagnostics are retained and
how the annotator consumes them.

## Requirements

The implementation MUST satisfy all of the following:

1. The generator and annotator must have memory bounded independently of the
   total number of per-read actions. Site models and source sufficient
   statistics are outside this report-format guarantee, but the generator must
   not retain a cohort-wide list of decision dictionaries merely to serialize
   it.
2. Every actionable MSP-to-TF rescue and accepted TF/nucleosome edge update
   must remain materializable. Low `q0` actions are retained so FiberBrowser can
   apply its connected-layer slider after the fact.
3. The action artifact must retain the exact three output quality bytes:
   alternative-hypothesis `q0`, molecular-left edge `q1`, and molecular-right
   edge `q2`.
4. Ordinary `nuc`, `msp`, and `tf` annotations and their qualities remain
   untouched. Baseline `nuc_sr` and `tf_sr` rows are synthesized from ordinary
   MA/AQ at annotation time; they do not need one report record per ordinary
   call.
5. Nucleosome identity and cardinality remain fixed. A v5 nucleosome action is
   one-for-one edge normalization only.
6. The annotator must validate that each action targets the exact BAM record
   and exact ordinary molecular annotation seen by the generator.
7. Existing v2, v3, and v4 inline reports remain readable without behavior
   changes. A new report must identify itself as v5 so an old annotator cannot
   silently ignore external actions.
8. Scientific outputs must be staged beside their destination on the mounted
   data volume. The implementation must not depend on WSL `/tmp` for persistent
   or recoverable state.

## Measured scale and why v4 cannot be used

The full UBA1 DddA baseline has 527,515 BAM records, including 524,909 primary
records. It contains 15,334,182 ordinary TF calls and 5,138,982 ordinary
nucleosome calls.

On the representative 1.2% UBA1 subset, using the intended full physical
amplicon (`chrX:47190560-47194939`), `--min-support 100`,
`--max-auto-sites 0`, and `--max-auto-nuc-sites 0` produced:

| quantity | measured subset value |
|---|---:|
| raw reads | 6,340 |
| collapsed population molecules | 5,097 |
| TF site populations | 102 |
| nucleosome geometry populations | 144 |
| candidate MSP groups | 25,491 |
| emitted rescue decisions | 1,919 |
| strong / review / retain-current | 730 / 1,044 / 145 |
| 1-TF / 2-TF / 3-TF alternatives | 1,884 / 34 / 1 |
| TF calls matched to a site | 55,385 |
| nucleosome calls matched to a site | 102,543 |

Simple depth projection gives approximately 2.12 million candidate MSP groups,
160,000 rescue decisions, and 163,000 rescued TF components at full depth.
Matched-call projection gives 13.1 million harmonization status records.
Adjusting by the call composition in the newly recalled full BAM gives an
approximately 13--15 million status-record range. Prior UBA1 action fractions
predict about 9.7 million accepted `edge_update` actions; 8--11 million is a
reasonable planning range.

Existing v4 UBA1 records average approximately 2.95 KB compact JSON for TF
harmonization, 3.10 KB for nucleosome harmonization, and 2.59 KB for rescue
decisions. A full inline report is therefore projected at approximately 40--46
GB compact or 67--77 GB with the current indented serialization. Parsed Python
objects measured roughly 4.3 times the compact JSON size. The v4 generator also
constructs a second giant string in `json.dumps` before writing. Its projected
peak is approximately 240--275 GB, before allowing for all temporary inference
objects. `json.loads`, typed decision copies, and per-read indexes make the
current annotator infeasible for the same reason.

A compact grouped-row prototype measured 233 bytes per action for UBA1, 246
bytes for NAPA, and 343 bytes for pooled DddB before the further ordinal and
pre-projection reductions specified here. At 9.9 million UBA1 actions, 2.3 GB
uncompressed is a conservative bound; grouping many actions on each full-depth
read further amortizes identity fields. BGZF is expected to reduce this to a
sub-gigabyte artifact. This is an estimate, not a release acceptance limit.

## Actionable and diagnostic-only records

### Rescue decisions

Every rescue decision with a nonempty proposed TF configuration and `q0` at or
above the annotator's requested minimum is actionable. The default minimum is
zero. `proposal_tier` is intentionally not an emission filter: even a
`retain_current` row remains available to the connected-layer slider.
Because v5 stores the finalized byte rather than a float, the comparison is
exactly `q0 / 255.0 >= minimum_posterior`; H actions are never thresholded.

The action path needs only:

- the exact input record;
- the source MSP annotation ordinal and molecular interval;
- one or more alternative molecular TF intervals;
- the atomic alternative `q0` and per-component molecular edge `q1`/`q2`;
- a stable naming token.

Proposal tier, baseline/current posterior, configuration posterior,
molecule/population/support/geometry probabilities, likelihoods, priors, raw
posterior masses, configuration counts, template diagnostics, site evidence,
and source-prior evidence are diagnostics. They are not read by the annotator.

### Edge harmonizations

For v3 and v4 the annotator accepts only a harmonization whose status is
`edge_update`. The full set of generator statuses is:

- `edge_update` -- actionable;
- `already_canonical` -- diagnostic-only;
- `unassigned_retained` -- diagnostic-only;
- `canonical_interval_not_spanned` -- diagnostic-only;
- `topology_conflict_retained` -- diagnostic-only;
- `insufficient_opposite_geometry_samples` -- diagnostic-only;
- `joint_topology_conflict_retained` -- diagnostic-only;
- `rescue_topology_conflict_retained` -- diagnostic-only.

`insufficient_shared_geometry` is an aggregate count; generation exits before
creating a per-call harmonization record.

An accepted edge action needs only:

- the exact input record;
- source type (`tf` or `nuc`), ordinary annotation ordinal, and exact current
  molecular interval;
- the alternative molecular interval;
- final molecular-orientation `q0`, `q1`, and `q2`;
- a stable naming token.

Site identifier, target/opposite strand, assignment probabilities and scores,
null score, assignment boolean, topology-conflict details, edge-shift flags,
molecule/population/geometry probabilities, raw target-edge evidence, and raw
source evidence are diagnostic-only.

## v5 main report

The main JSON schema is `fiberhmm.strand_rescue.v5` with `schema_version: 5`.
It retains the existing producer, normalized-layer contract, input metadata,
parameters, molecule and efficiency diagnostics, site templates, site models,
aggregate counts, guardrails, and performance sections. Per-call actions move
under a manifest at `strand_rescue.action_storage`.

The required shape is:

```json
{
  "schema": "fiberhmm.strand_rescue.v5",
  "schema_version": 5,
  "input": {
    "bams": ["/absolute/input0.bam"],
    "files": [{"path": "/absolute/input0.bam", "size_bytes": 1, "mtime_ns": 1}],
    "loaded_region": ["chrX", 47188560, 47196939]
  },
  "strand_rescue": {
    "applicable": true,
    "strands": ["CT", "GA"],
    "sites": [],
    "target_sites": [],
    "site_models": {},
    "counts": {},
    "edge_refinement": {
      "tf": {"call_type": "tf", "sites": [], "counts": {}},
      "nuc": {"call_type": "nuc", "sites": [], "counts": {}}
    },
    "action_storage": {
      "layout": "per_input_bgzf_jsonl_v1",
      "stream_schema": "fiberhmm.strand_rescue.actions.v1",
      "quality_encoding": "uint8_round_255_times_unit_probability",
      "coordinate_frame": "molecular_zero_based_start_length",
      "streams": [],
      "totals": {
        "fetch_records": 0,
        "action_records": 0,
        "rescue_decisions": 0,
        "rescue_components": 0,
        "tf_edge_updates": 0,
        "nuc_edge_updates": 0
      }
    },
    "diagnostic_storage": {
      "mode": "aggregate"
    }
  }
}
```

`edge_refinement.tf.harmonizations`,
`edge_refinement.nuc.harmonizations`, and `strand_rescue.decisions` MUST NOT be
present in a streamed v5 report. Their presence together with external actions
is an error, not a merge rule.

Each `action_storage.streams` item is required to contain:

```json
{
  "input_index": 0,
  "input_id": "input0000",
  "path": "report-name.input0000.sr-actions.<sha12>.jsonl.bgz",
  "gzi_path": "report-name.input0000.sr-actions.<sha12>.jsonl.bgz.gzi",
  "loaded_region": ["chrX", 47188560, 47196939],
  "fetch_record_count": 527515,
  "action_record_count": 500000,
  "first_action_ordinal": 0,
  "last_action_ordinal": 527514,
  "rescue_decision_count": 160000,
  "rescue_component_count": 163000,
  "tf_edge_update_count": 2500000,
  "nuc_edge_update_count": 7200000,
  "compressed_size_bytes": 1,
  "uncompressed_size_bytes": 1,
  "bgzf_sha256": "64-lowercase-hex",
  "jsonl_sha256": "64-lowercase-hex",
  "gzi_size_bytes": 1,
  "gzi_sha256": "64-lowercase-hex"
}
```

Rules:

- `input_index` indexes `input.bams` and `input.files`. `input_id` is exactly
  `input` plus a four-digit zero-padded index.
- Stream paths are UTF-8 relative paths resolved against the main report's
  directory. Absolute paths and `..` traversal are invalid. The `<sha12>` name
  token is exactly the first 12 hexadecimal characters of `jsonl_sha256` (the
  decompressed canonical stream), making already-published sidecars immutable
  across compression settings.
- An input with no actions still has a header/trailer stream and manifest row.
  `first_action_ordinal` and `last_action_ordinal` are then `null`.
- All counts are non-negative integers and must agree with the stream trailer
  and the actions actually consumed.
- SHA-256 strings are lowercase hexadecimal. `bgzf_sha256` covers the exact
  compressed bytes, `jsonl_sha256` covers every decompressed byte including
  newlines, and `gzi_sha256` covers the exact standard GZI file.
- The main report's SHA-256 plus the sidecar hashes form output provenance. The
  annotator must record both the report hash and selected action-stream hash in
  its BAM header.

## Regional input-record ordinal

Ordinals are independent for each input BAM. They are defined by exactly:

```python
for input_record_ordinal, read in enumerate(
    source.fetch(chrom, loaded_start, loaded_end)
):
    ...
```

The ordinal is assigned before all eligibility filtering. Every record yielded
by the indexed regional fetch advances it, including primary, secondary,
supplementary, duplicate, QC-fail, low-MAPQ, and records with no usable hard
calls or MA annotations. Regional fetch does not yield the unmapped tail, but
if HTSlib returns any unusual record it still advances the ordinal.

The generator and annotator must use the exact `loaded_region` in the stream
manifest and the same indexed `fetch` semantics. Ordinals reset to zero for
each input. `fetch_record_count` is the total number of yielded records, not
the number used by inference. `--max-reads` is a scientific inference limit,
not permission to redefine ordinal space; a limited run still records ordinals
from the original regional fetch for every action and records the full regional
fetch count before publication.

One action line is permitted per ordinal. Lines are strictly increasing in
ordinal. An action may target only a primary mapped record accepted by the
inference filter. The row also carries query name and a SHA-256 of the exact
input `read.to_string()` value. Ordinal is the merge key; name and hash are
mandatory drift/corruption checks.

## BGZF JSONL action stream

The stream is UTF-8 JSON Lines compressed with BGZF. Serialization is
canonical for checksums: `allow_nan=False`, compact separators `(',', ':')`, no
trailing spaces, and exactly one `\n` after every object. Producers should use
stable key insertion order as shown below; consumers use parsed keys and do not
depend on object-key order.

The first line is:

```json
{"kind":"header","schema":"fiberhmm.strand_rescue.actions.v1","input_index":0,"input_id":"input0000","loaded_region":["chrX",47188560,47196939],"ordinal_base":0}
```

An action line is:

```json
{
  "kind": "actions",
  "ordinal": 123,
  "read": "movie/zmw/ccs",
  "record_sha256": "64-lowercase-hex",
  "rescues": [],
  "edge_updates": []
}
```

The final line is:

```json
{
  "kind": "trailer",
  "fetch_record_count": 527515,
  "action_record_count": 500000,
  "first_action_ordinal": 0,
  "last_action_ordinal": 527514,
  "rescue_decision_count": 160000,
  "rescue_component_count": 163000,
  "tf_edge_update_count": 2500000,
  "nuc_edge_update_count": 7200000
}
```

Empty `actions` rows are forbidden. A row must contain at least one rescue or
edge update. Unknown top-level keys are rejected for actions-v1 so accidental
schema drift fails loudly.

### Rescue action

```json
{
  "token": "16-lowercase-hex",
  "source_ordinal": 7,
  "source_interval": [120, 90],
  "q0": 173,
  "components": [
    {"component_index": 0, "interval": [168, 22], "q1": 201, "q2": 230},
    {"component_index": 1, "interval": [130, 18], "q1": 220, "q2": 214}
  ]
}
```

`source_ordinal` indexes the concatenated ordinary `msp` annotation list in MA
order. `source_interval` and component `interval` values are zero-based
molecular `(start, length)`. Components retain the original v4 proposed
configuration order and carry consecutive `component_index` values beginning
at zero. This is deliberately not molecular sort order: on a reverse
multi-component read, v4 assigns `R0`, `R1`, ... before molecular-frame sorting,
so preserving configuration order is required for byte-identical AN names.
Component intervals are non-overlapping, unique, fully contained by the source
MSP, and atomic: the annotator applies all components or none. `q0`, `q1`, and
`q2` are integers in `[0,255]`. `q0` applies to the complete alternative and is
repeated into each component's `tf_sr.QQQ` row.

The token is exactly the existing v4 annotation token:

```python
sha256(f"sr:{library_id}:{decision_id}".encode()).hexdigest()[:16]
```

The output names remain `fhsr_{token}_R{component_index}`.

### Edge-update action

```json
{
  "token": "16-lowercase-hex",
  "call_type": "nuc",
  "source_ordinal": 3,
  "source_interval": [410, 151],
  "alternative_interval": [405, 158],
  "q": [191, 225, 219]
}
```

`call_type` is exactly `tf` or `nuc`. `source_ordinal` indexes the corresponding
ordinary annotation list in MA order. Source and alternative intervals are
zero-based molecular `(start, length)`. `q` is exactly final `(q0,q1,q2)`.

The token is the existing v4 token:

```python
sha256(
    f"sr-edge:{call_type}:{library_id}:{decision_id}".encode()
).hexdigest()[:16]
```

The output name remains `fhsr_{token}_O{source_ordinal}_H`.

### Molecular projection and quality orientation

The generator, not the annotator, projects reference alternatives to molecular
coordinates for v5. Projection must be identical to v4 annotation behavior:

- both reference endpoints must map;
- at least 95% of bases in the reference interval must be mapped;
- the query span is from the minimum to maximum mapped query position;
- reverse reads are transformed with
  `L - (query_start + length), length`;
- a projection failure suppresses the action and increments a diagnostic
  rejection count.

The v5 stream stores final molecular-orientation edge bytes. For a reverse read,
reference-left and reference-right confidence are swapped before rounding.
Each probability is converted exactly with
`max(0, min(255, round(255 * p)))`. Storing the final bytes avoids float or
orientation differences during later annotation.

The generator must validate that no two tokens would create the same AN name
within one alignment. A collision is a hard generation error.

## Per-alignment finalization

An action must be emitted only after all rescue and edge candidates for its
alignment are known and finalized. Global population statistics are prepared
first, but collision resolution is local to one alignment identity.

For each eligible alignment, generation must:

1. score and finalize all atomic MSP-to-TF rescues;
2. score provisional TF and nucleosome edge updates;
3. apply existing within-type ordering constraints;
4. apply joint TF/nucleosome edge-topology rejection;
5. apply rescue-versus-edge topology rejection;
6. project every retained alternative to molecular coordinates and orient Q
   bytes;
7. emit one grouped action row, then discard its per-call details.

No `edge_update` may be streamed before joint and rescue-topology resolution.
The generator must not build global `harmonizations` lists and later prune
them. Diagnostic-only outcomes update aggregate counters as soon as final
status is known. If detailed diagnostics are explicitly requested, they go to
a separate streaming diagnostic artifact and never back into the main Python
report object.

## Diagnostic storage

The default is:

```json
{"mode":"aggregate"}
```

All existing status counts, site summaries, evidence-calibration summaries,
and performance measurements remain in the main report. This is sufficient for
production annotation and routine QC.

An explicit diagnostic mode may add one or more separate BGZF JSONL manifests
under `diagnostic_storage`. These may carry the full v4-style per-call evidence
for debugging. They use separate schema and checksums, are never opened by the
annotator, and must not be required to reproduce MA/AQ. Diagnostic streaming
must obey the same destination-local staging and manifest-last publication
rules.

## Layout selection and auto spill

The report CLI should expose:

```text
--report-layout auto|inline|stream
--diagnostics aggregate|stream
```

`auto` is the default. It changes irreversibly from an in-memory inline buffer
to the stream sink when either:

- 100,000 per-record actions/status details would be retained, or
- estimated compact inline JSON reaches 256 MiB.

The threshold is inclusive. Once streaming begins, buffered finalized actions
are flushed in ordinal order and the run never switches back. Implementations
may choose streaming from the outset when input depth and discovered site count
already make crossing either boundary certain. `stream` forces v5 sidecars.
`inline` is an explicit small/debug mode and must fail with a clear size-limit
error rather than consume unbounded memory if a configured hard ceiling is
crossed.

The v2--v4 writer remains available for regression fixtures and existing
workflows. A v5 streamed report does not masquerade as v4.

## BGZF, GZI, checksums, and publication

Each stream is written with `pysam.BGZFile(..., 'wb', index=gzi_path)` or an
equivalent htslib implementation, producing a standards-compliant BGZF EOF
block and GZI index. The initial implementation uses a sequential merge join;
the GZI is retained for integrity tooling and future seek/resume support.

Publication is a manifest-last transaction:

1. Create hidden, uniquely named stream and GZI staging files in the final
   report directory.
2. Write header, action rows, and trailer; close both BGZF and its index.
3. Flush and fsync files. Reopen the stream, validate BGZF decoding, schema,
   ordering, trailer counts, and decompressed checksum.
4. Compute compressed BGZF and GZI sizes and SHA-256 values.
5. Rename validated sidecars to content-derived immutable final names. Never
   modify a content-addressed artifact already referenced by a report; if it
   exists, validate it and reuse it or fail.
6. Build the main report with final relative paths and checksums.
7. Atomically replace the main report last using a staging file in the report
   directory.

If generation fails before step 7, the previously published report remains
valid. Hidden staging files are not scientific outputs and may be cleaned by
the generating process. Content-addressed orphan sidecars are harmless and may
be reused after full validation. No report may reference a staging filename.

Replacing an existing report is safe because the old manifest remains visible
until the new manifest is atomically committed. Implementations must not delete
old sidecars during publication; garbage collection is a separate explicit
operation.

## Streaming annotation merge join

For v2--v4, `fiberhmm-strand-rescue-annotate` keeps its current inline path.
For v5 it must not call `json.loads` on an action collection or build a
cohort-wide `decisions_by_read` index.

For each selected input:

1. Resolve the stream and GZI paths relative to the report.
2. Validate manifest shape, path safety, file sizes, GZI checksum, and BGZF
   compressed checksum.
3. Open the BGZF stream and validate its header against the main report.
4. Read the first action row and iterate
   `enumerate(source.fetch(*loaded_region))`.
5. If the next action ordinal is greater than the BAM ordinal, synthesize only
   baseline `nuc_sr` / `tf_sr` rows for this BAM record.
6. If ordinals are equal, validate query name and exact
   `sha256(read.to_string())`, validate source MA type/ordinal/interval, apply
   the grouped atomic actions, and advance the stream once.
7. If the next action ordinal is less than the BAM ordinal, or two rows share
   an ordinal, fail immediately.
8. At BAM-fetch EOF, require no action row remains, validate the stream trailer,
   fetch count, per-action counts, first/last ordinal, decompressed JSONL hash,
   and BGZF EOF.
9. Index and validate the staged output BAM.
10. Publish only after every selected input succeeds; multi-BAM cohorts retain
    the current all-or-nothing staged publication behavior.

The sequential path needs memory only for one decoded JSON line, one BAM
record, and that record's ordinary/SR annotations. It does not need expected or
applied decision-ID sets. Exact trailer/manifest counters replace those sets.

The annotator may avoid hashing BAM records that have no action row. Input
size/mtime metadata still provides early drift detection; the row hash provides
exact validation where a scientific change will be written.

The annotator records in the BAM header:

- main report SHA-256;
- selected BGZF SHA-256 and decompressed JSONL SHA-256;
- stream schema and input index;
- unchanged QQQ semantics and connected-layer threshold contract.

## Action application rules

V5 action application produces the same complete normalized layers as v4:

- every ordinary nucleosome gets one `nuc_sr.QQQ` baseline row `(255,0,0)`;
- every ordinary TF gets one `tf_sr.QQQ` baseline row `(255,0,0)`;
- an accepted H action replaces the corresponding baseline row's interval,
  QQQ, and AN name one-for-one;
- an R action adds its complete alternative TF configuration without deleting
  its source MSP or any ordinary annotation;
- no new overlap or order inversion may be introduced;
- a rescue configuration is atomic;
- stale `nuc_sr` and `tf_sr` groups are replaced, not appended.

Although the generator has already resolved topology, the annotator retains
the v4 defensive collision/order checks. Any discrepancy is a hard error under
the v5 exact-source contract rather than a silently unmatched action.

## Failures and recovery

The following are hard errors and prevent output publication:

- unsupported main or stream schema;
- missing sidecar or GZI;
- unsafe or absolute sidecar path;
- size or any checksum mismatch;
- malformed UTF-8/JSON, non-finite value, unknown key, or invalid Q byte;
- header/manifest/trailer disagreement;
- non-increasing, duplicate, negative, or out-of-range ordinal;
- row targeting a secondary/supplementary/unmapped record;
- query-name or record-hash mismatch;
- source type, ordinal, or molecular interval mismatch;
- alternative interval outside the read or rescue outside its MSP;
- duplicate/overlapping rescue components;
- newly introduced overlap/order inversion;
- unexpected action/trailer counts or regional fetch count;
- missing BGZF EOF or output BAM/index validation failure.

All BAMs and indexes are written to destination-local staging paths. On error,
staged outputs are not published and existing outputs remain unchanged.
Initial v5 annotation restarts from the beginning after failure; partial-output
resume is deliberately not part of v5. The GZI leaves room for a later
checkpointed implementation without weakening the simple exact merge contract.

If a checksum-valid sidecar exists but the main report was never committed, a
rerun may reuse it only after validating schema, input identity, region,
ordinal/count trailer, and all hashes. Otherwise it writes a new
content-addressed sidecar.

## Backward compatibility

- `collect_decisions`, `collect_harmonizations`, and the current per-read index
  remain the v2--v4 path.
- V5 uses a separate stream reader and ordinal merge join.
- The output `nuc_sr.QQQ` / `tf_sr.QQQ` schema, Q scale, AN naming, baseline
  sentinel, and header semantics remain unchanged.
- Old annotators reject v5 as unsupported. They must never emit a baseline-only
  BAM from a report whose actions they did not read.
- New annotators reject a v5 report that mixes streamed and legacy inline action
  surfaces.
- The report path remains the primary provenance object; sidecars are verified
  dependencies named by its manifest.

## Golden and failure-test plan

### Byte-semantic equivalence

For a small fixture, generate the same finalized decisions as v4 inline and v5
streamed, annotate both, and compare each corresponding record:

1. the first 11 SAM fields are identical;
2. all non-MA/AQ/AN tags are identical;
3. every ordinary MA/AQ/AN annotation and quality is identical;
4. `nuc_sr` and `tf_sr` molecular intervals, QQQ rows, order, and AN names are
   identical;
5. expected header differences are limited to v5 report/action provenance.

The fixture set must include:

- forward and reverse alignments, verifying molecular q1/q2 orientation;
- one-, two-, and three-component atomic rescues;
- low-q0 `retain_current` rescue retained at minimum zero;
- TF and nucleosome H actions;
- same-read TF/nuc joint topology conflict;
- rescue-versus-H topology conflict;
- duplicate ordinary intervals distinguished by annotation ordinal;
- insertions, deletions, soft clips, and a projection below the 95% threshold;
- an existing stale SR layer that is replaced;
- a read with MA but no actions, proving baseline synthesis needs no status row.

### Ordinal and multi-input behavior

Tests must cover:

- secondary, supplementary, QC-fail, duplicate, and low-MAPQ records advancing
  ordinal even though they receive no actions;
- repeated query names and identical alignment identities disambiguated by
  ordinal;
- two input BAMs with independent ordinal zero and separate manifests;
- an input with no actions but a valid header/trailer stream;
- a regional record count that differs from eligible inference-read count;
- explicit annotation of one selected input from a multi-input report.

### Stream integrity

Tests must reject:

- truncated BGZF and missing EOF;
- modified compressed bytes;
- modified decompressed JSON line;
- missing, modified, or wrong GZI;
- wrong header input index or region;
- missing/wrong trailer or manifest count;
- out-of-order, duplicate, negative, and beyond-EOF ordinals;
- extra action after BAM EOF;
- malformed action, unknown type/key, Q outside `[0,255]`, empty row, and token
  collision;
- unsafe `..` or absolute manifest path.

### Exact-source and topology safety

Tests must reject:

- changed input size/mtime without `--allow-input-drift`;
- changed query name or SAM-record hash even with drift allowed;
- source ordinal out of range;
- source interval not equal to the ordinary MA interval;
- alternative interval outside the read;
- rescue component outside its MSP or overlapping another component;
- H action that changes call type/cardinality;
- a new cross-layer overlap or same-layer order inversion.

### Atomicity and recovery

Fault-injection tests at stream close, GZI close, checksum validation, sidecar
rename, report commit, BAM write, BAM index, and second-BAM publication must
show:

- no report references a staging file;
- an existing report/output cohort remains readable and unchanged;
- a failed multi-input annotation publishes none of the new cohort;
- a validated content-addressed sidecar can be reused;
- a mismatched orphan is never reused.

### Auto spill and bounded memory

Tests use injectable thresholds rather than enormous fixtures:

- 99,999 details remain eligible for inline mode under the row threshold;
- the 100,000th detail triggers one-way spill;
- reaching 256 MiB triggers spill independently;
- buffered rows retain ordinal order after spill;
- explicit `stream` starts streamed;
- explicit `inline` fails at its hard ceiling without partial publication.

A synthetic high-action test should stream at least one million edge actions
through annotation while asserting that RSS growth is bounded by a fixed small
multiple of the largest action line, not total actions. Release validation must
record wall time, maximum RSS, compressed/uncompressed sizes, and action counts
on full targeted DddA and DddB amplicons.

## Implementation ownership boundary

Until discovery optimization is frozen, generator and inference changes belong
to the generator workstream. Annotator work may add v5 manifest/stream parsing,
ordinal merge join, action application, provenance, and overlay tests without
editing `fiberhmm/inference/strand_rescue.py` or
`fiberhmm/cli/strand_rescue.py`. Shared projection or schema helpers require
coordination so generator and annotator cannot diverge on molecular intervals,
Q orientation, or checksums.
