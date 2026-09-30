# Header declarations

FiberHMM records what it did, and what a BAM contains, in `@PG` and `@CO`
header lines. `@CO` lines are written as `@CO<TAB><text>`; pysam's
`header.to_dict()["CO"]` returns the text.

| Line | Written by | Purpose |
|---|---|---|
| `@PG` | every producer (`call`, `apply`, `recall-tfs`/`-nucs`, `pair`, `merge`, `tag-m5c`, `consensus`, `tag-consensus`, …) | provenance: command line, version, resolved settings |
| `@CO FIBERHMM-CHEMISTRY:v1:` | `call`, `apply`, `recall-tfs`/`-nucs` | authoritative chemistry of the calls |
| `@CO MA-TYPES:v1:` | every command that writes `MA` | advisory list of `MA` group names |
| `@CO fiberhmm:coord=molecular` | `apply`, `recall-tfs`/`-nucs` | coordinate-frame marker (`call` puts `coord=molecular` in its `@PG DS`) |
| `@CO FIBERHMM-CONSENSUS-MA:v1:` | `consensus`, `transfer` | meaning of the consensus layers' bytes |
| `@CO FIBERHMM-CONSENSUS-FAMILY:v1:` | `consensus`, `transfer` | one entry per class and layer |
| `@CO FIBERHMM-STRAND-RESCUE:v6:` | `strand-rescue-annotate` | shadow-layer contract |
| `@CO FIBERHMM-TF-FAMILY:v1:` | `tag-consensus` | family-slot contract |
| `@SQ M5`, `@SQ TP:circular`, `@CO FIBERHMM-REFERENCE:v1:` | `pipeline` | identity and topology of the reference (plasmid maps) |

Header lines are copied by `samtools` and most other tools, so they survive
downstream processing unless a tool rewrites the header.

## `@PG`

`fiberhmm-call`'s `DS` field states the frame and every resolved setting:

```text
FiberHMM fused apply+recall; coord=molecular (ns/nl/as/al/MA in molecular original-fiber coordinates); mode=daf enzyme=ddda prob_threshold=128 primary_only=on tf_decoder=multi_interval_v1 tf_interval_penalty=5.0 recall_nucs=True nuc_recall_policy=conservative nuc_profile=ddda_phase_posterior_v1 nuc_sha256=c86b05dc... phase_nrl=189 ddda_derived_tf_edge_gap=12 chimera_filter=on dedup=j0.95/mark/ends50 daf_snp_mask=on/0sites daf_run_mask=>=2/keep-one cpg_mask=unmethylated-only
```

The recallers record the recall subset. Recall reads the input's `@PG` to
warn when a SNP mask or reference used at calling time cannot be re-applied.

## FIBERHMM-CHEMISTRY

```text
@CO	FIBERHMM-CHEMISTRY:v1:assay=daf;enzyme=ddda;platform=pacbio;mode=daf;model=ddda_TF;nuc_model=ddda_phase_posterior_v1;nuc_sha256=c86b05dc07e45392880e3460cf7f8880593ecad174e0a338d36ac53b7d0172d6
@CO	FIBERHMM-CHEMISTRY:v1:assay=fiber-seq;enzyme=hia5;platform=nanopore;mode=nanopore-fiber;model=hia5_nanopore
```

`;`-separated `key=value` fields. Required: `assay`, `enzyme`, `platform`,
`mode`. Producers may add fields such as `model`, and `nuc_model` /
`nuc_sha256` (the nucleosome profile and its file digest) when a distinct
nucleosome profile is active. Keys match `[a-z][a-z0-9_]*`; values match
`[A-Za-z0-9_.+-]+`.

Vocabulary:

- `assay=daf;enzyme=ddda|dddb;mode=daf`, with the sequencing
  `platform=pacbio|nanopore` (`unknown` only when unavailable);
- `assay=fiber-seq;enzyme=hia5` with `platform=pacbio;mode=pacbio-fiber` or
  `platform=nanopore;mode=nanopore-fiber`;
- `custom` for a custom assay, enzyme or mode (for example a run with
  `-m` and no `--enzyme`: `enzyme=custom`).

A valid declaration is authoritative. Several lines with the same four
required fields are allowed (for example after successive models add
different `model` values); incompatible required fields are a conflict, and
FiberHMM producers refuse to relabel silently. Readers ignore malformed lines
and unknown versions. For BAMs made before the declaration existed, tools
infer the chemistry, with lower confidence, from a `fiberhmm-call` `@PG`
record; filenames are never used.

**Reconciliation.** `fiberhmm-call` and `fiberhmm-recall-tfs`/`-recall-nucs`
reconcile their declaration with the input's: a custom model without
`--enzyme` inherits the input's assay, enzyme and platform when the mode
matches; any other disagreement stops the run unless `--replace-chemistry`
is given, which drops the input's declaration and writes this run's (see
[Choosing the chemistry](../getting-started/choosing-chemistry.md#re-calling-a-bam-fiberhmm-already-called)).

**Readers.** Recall (missing `--seq`), `fiberhmm-extract` and `fiberhmm-qc`
(ML threshold, QC profile), and `fiberhmm-consensus`/`fiberhmm-transfer`
(emission model) take the chemistry from this line.

Python: `fiberhmm.io.bam_header.declared_chemistries(header)` returns the
valid declarations as dictionaries.

## MA-TYPES

```text
@CO	MA-TYPES:v1:nuc,msp,tf
@CO	MA-TYPES:v1:ddda_mcg,ddda_ucg
```

`MA` group names are extensible and otherwise appear only inside per-read
`MA:Z` values, so a viewer would have to scan reads to discover rare layers.
This line advertises the names a BAM may contain. It lists names only (no
strand or quality suffix); names are case-sensitive and match `[A-Za-z0-9_]+`.
Several lines are allowed; readers take the ordered union. Producers keep
existing lines and append one line with the names not yet declared.

It is a discovery hint, not part of `MA` correctness:

- records remain authoritative; a reader accepts names that were not
  declared;
- a missing name does not mean biological absence, and declarations never
  determine `AQ` arity;
- missing, stale or malformed declarations are ignored;
- declaring a name never creates an empty per-read group.

Repair an older BAM in place, from known names or by scanning every
alignment:

```bash
fiberhmm-utils ma-types out/matypes.bam --types nuc,msp,tf,ddda_mcg
fiberhmm-utils ma-types out/matypes.bam --scan
```

```text
Updated out/matypes.bam in place (300 alignments); added: ddda_mcg; declared union: nuc,msp,tf,ddda_mcg
Scanned all 300 alignments (300 with MA); observed: nuc,msp,tf
All requested MA types were already declared; BAM left unchanged.
```

The utility writes and validates a temporary BAM next to the original,
rebuilds an existing BAI/CSI index and then replaces both; per-read tags are
copied unchanged. Python: `declared_ma_types(header)` and
`append_ma_types(header, names)` in `fiberhmm.io.bam_header`.

## Consensus headers

`fiberhmm-consensus` and `fiberhmm-transfer` BAM exports carry:

- one `FIBERHMM-CONSENSUS-MA:v1:` line, a JSON object recording the engine,
  export scope and windows, the layers, the meaning of every byte
  (`quality_names`, `layer_quality_names`, `q0`, `fi`, `fq`, `op`, …), and the
  run's parameters and input digest;
- one `FIBERHMM-CONSENSUS-FAMILY:v1:` line per class and layer, a JSON object:

```text
@CO	FIBERHMM-CONSENSUS-FAMILY:v1:{"annotation_name": "fhcr_9ca5cf4d4ac4fc4ffb286f0b", "chrom": "chrDemo", "end": 10065, "extent": "class_consensus_span", "family_key": "class_001", "fi": 1, "input_digest": "...", "layer": "tf_consensus", "stage": "resolved", "start": 10041}
```

`annotation_name` is the `AN` token of the class's annotations; `fi` its
slot; `start`/`end` the class consensus span (lattice recaller) or the union
of the labelled calls (staged engine). DAF classes add a per-dataset
`strand_resolution` with `trusted_strand` (`CT`, `GA`, `both`, `none`),
`trusted_strands`, `supported_strands`, per-strand `resolution_nats` and the
threshold. `fiberhmm.inference.consensus.bam_export.read_family_catalog(header)`
returns `{"contracts": [...], "families": [...]}`.

## Strand rescue and family slots

`FIBERHMM-STRAND-RESCUE:v6:` (from `fiberhmm-strand-rescue-annotate`) is a
`;`-separated contract: the groups (`nuc_sr,tf_sr`), quality spec and scale,
the meaning of `q0`/`q1`/`q2` for each role, the display rule
(`display_sr_if=q0>=T;threshold_named_only=true`), the roles (`Rn`, `H`), the
baseline sentinel row (`255,0,0`) and the nucleosome invariants. v2–v5 BAMs
are still validated by `fiberhmm-strand-rescue-audit`.

`FIBERHMM-TF-FAMILY:v1:` (from `fiberhmm-tag-consensus`) declares
`layer=tf_sr;quality_spec=QQQQQ`, the five byte meanings, the slot reuse
distance (`fi_reuse_separation_bp=24`) and the SHA-256 of the assignment
table.

## FIBERHMM-REFERENCE

`fiberhmm-pipeline` records which reference a BAM was aligned to, so a viewer
can pair it with the right FASTA or plasmid map even after files are renamed
(see [Plasmids](../workflows/plasmids.md)):

- every `@SQ` line gets `M5`, the MD5 of the contig sequence (upper case, no
  whitespace; the SAM standard);
- a circular contig gets `TP:circular` (SAM standard). Its alignments may run
  past `LN`, the SAM-preferred form for reads through the origin of a
  circular reference: a position `p > LN` means `p - LN`;
- for a reference made from a plasmid map, and for every circular contig,
  one `@CO` line per contig:

```text
@CO	FIBERHMM-REFERENCE:v1:contig=N1;length=12179;md5=1276ace935ddb3051497a9b2b956a0c2;topology=circular;source=N1.dna;source_format=snapgene;source_sha256=02bb9705b987b8df586ff0723c7d24de49f5683a85654c669a5bfc8e80243d88
```

`;`-separated `key=value` fields; values are percent-encoded (RFC 3986:
`A–Z a–z 0–9 . _ - +` stay literal, so a file name with spaces or `;` is
safe). Fields:

| Key | Value |
|---|---|
| `contig` | the contig name (the `@SQ SN`) |
| `length` | contig length (the `@SQ LN`) |
| `md5` | as `@SQ M5` |
| `topology` | `circular` or `linear` |
| `source` | file name of the map or FASTA, without its directory |
| `source_format` | `snapgene`, `genbank`, `embl` or `fasta` |
| `source_sha256` | SHA-256 of that file as given |

Readers ignore unknown keys and unknown versions. `fiberhmm-call` and the other
FiberHMM producers copy the input header, so the lines survive calling.
Python: `fiberhmm.pipeline.reference.declared_references(header)` returns the
parsed lines; `parse_reference_comment(text)` parses one.

