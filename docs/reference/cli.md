# Command-line reference

Every FiberHMM command and every option it accepts, with its default. The
tables below are generated from the commands' own argument parsers, so they
match `--help` exactly. Behaviour that does not fit in a flag description is
explained on the [workflow pages](../workflows/calling.md); defaults that
depend on the chemistry are collected in
[Chemistries and platforms](../concepts/chemistries.md#default-settings-per-chemistry).
Every command also accepts `-h`/`--help` and `--version` (prints
`fiberhmm <version>`).

| Command | What it does | Guide |
|---|---|---|
| [`fiberhmm-pipeline`](#fiberhmm-pipeline) | Reads + reference (FASTA or plasmid map) to a called BAM: align, call, QC | [Quick DAF-seq](../getting-started/quick-daf-seq.md), [Plasmids](../workflows/plasmids.md) |
| [`fiberhmm-call`](#fiberhmm-call) | Call nucleosomes, MSPs and TF footprints (HMM + recall) in one pass | [Calling](../workflows/calling.md) |
| [`fiberhmm-apply`](#fiberhmm-apply) | HMM nucleosomes/MSPs only | [Calling](../workflows/calling.md#fiberhmm-apply-hmm-only) |
| [`fiberhmm-recall-tfs`](#fiberhmm-recall-tfs), [`fiberhmm-recall-nucs`](#fiberhmm-recall-nucs) | Re-call TFs (and nucleosomes) on a called BAM | [Re-calling](../workflows/recalling.md) |
| [`fiberhmm-qc`](#fiberhmm-qc) | Bounded QC report for one or more BAMs | [QC](../workflows/qc.md) |
| [`fiberhmm-extract`](#fiberhmm-extract) | Calls to BED12 / bigBed | [Extracting tracks](../workflows/extracting.md) |
| [`fiberhmm-dedup`](#fiberhmm-dedup) | DAF PCR-duplicate marking/collapse | [DAF-seq](../workflows/daf-seq.md#pcr-duplicates) |
| [`fiberhmm-daf-encode`](#fiberhmm-daf-encode) | Stamp R/Y deamination codes into a DAF BAM | [DAF-seq](../workflows/daf-seq.md#ry-encoding-fiberhmm-daf-encode) |
| [`fiberhmm-daf-snps`](#fiberhmm-daf-snps) | Recurrent C→T/G→A SNP mask for DAF-seq | [DAF-seq](../workflows/daf-seq.md#snp-screening) |
| [`fiberhmm-pair`](#fiberhmm-pair), [`fiberhmm-merge`](#fiberhmm-merge) | Pair, merge and jointly re-call scDAF duplexes | [Duplex](../workflows/duplex.md) |
| [`fiberhmm-tag-m5c`](#fiberhmm-tag-m5c), [`fiberhmm-call-m5c`](#fiberhmm-call-m5c) | DddA CpG-island methylation | [DAF-seq](../workflows/daf-seq.md#ddda-cpg-island-methylation) |
| [`fiberhmm-consensus`](#fiberhmm-consensus) | Footprint classes across molecules | [Consensus](../workflows/consensus.md) |
| [`fiberhmm-transfer`](#fiberhmm-transfer) | Apply frozen classes to new data | [Transfer](../workflows/transfer.md) |
| [`fiberhmm-footprint-model`](#fiberhmm-footprint-model) | Footprint population model | [Footprint model](../workflows/footprint-model.md) |
| [`fiberhmm-strand-rescue`](#fiberhmm-strand-rescue) (+ [`-annotate`](#fiberhmm-strand-rescue-annotate), [`-audit`](#fiberhmm-strand-rescue-audit)) | Two-strand TF rescue and edge normalization | [Strand rescue](../workflows/strand-rescue.md) |
| [`fiberhmm-tag-consensus`](#fiberhmm-tag-consensus) | Consensus-state slots on `tf_sr` calls | [Tag consensus](../workflows/tag-consensus.md) |
| [`fiberhmm-posteriors`](#fiberhmm-posteriors) | Per-position HMM posteriors | [Posteriors](../workflows/posteriors.md) |
| [`fiberhmm-probs`](#fiberhmm-probs), [`fiberhmm-train`](#fiberhmm-train) | Build a custom model | [Training](../workflows/training.md) |
| [`fiberhmm-utils`](#fiberhmm-utils) | Model, header and bigBed utilities | [Training](../workflows/training.md#model-utilities), [Extracting](../workflows/extracting.md#repairing-bigbed-sample-names) |

Every command also accepts `-h`/`--help`.

<!-- BEGIN GENERATED CLI REFERENCE (tools/gen_cli_reference.py) -->

Generated from each command's argparse definition by `python tools/gen_cli_reference.py`; do not edit by hand. Hidden compatibility options are omitted. `auto` means the value is resolved at run time as the description says.

## fiberhmm-pipeline

| Flag | Default | Description |
|------|---------|-------------|
| `reads` | required | Read files: FASTQ (.fastq/.fq, optionally .gz), unaligned BAM, a BAM aligned to --reference, or a directory of them. All reads given form one sample. |
| `--reference` | required | Reference FASTA, or a plasmid map (.dna, .gb/.gbk/.genbank, .embl) converted to a FASTA whose contig is named as FiberBrowser names the map. |
| `--enzyme` | required | Chemistry: ddda / dddb (DAF-seq) or hia5 (Fiber-seq). Choices: `ddda`, `dddb`, `hia5`. |
| `-o` / `--outdir` | required | Output directory. |
| `--sample` | — | Sample name for output files and the read group (default: the first input's name without extensions). One plain file name: no path separators, spaces or leading '.'/'-'. |
| `-c` / `--cores` | `4` | minimap2 threads and fiberhmm-call worker processes (default 4). |
| `--seq` | — | Sequencing platform. Default: detected from the reads. ddda/dddb: a BAM's FIBERHMM-CHEMISTRY declaration or @RG PL/@PG records; reads with no record are Nanopore when aligned here, and keep fiberhmm-call's default when called as given. hia5: also the MM tags of the first reads (T-a = PacBio, A+a only = Nanopore), an error when nothing settles it. Sets the minimap2 preset (map-ont / map-hifi), the read group's PL and fiberhmm-call's --seq. Choices: `nanopore`, `pacbio`. |
| `--force-chemistry` | off | Run although the reads contradict --enzyme/--seq (a FIBERHMM-CHEMISTRY declaration naming another enzyme or platform; m6A-tagged reads without deaminations for ddda/dddb; deaminated or untagged reads for hia5). Default: refuse. |
| `--topology` | `auto` | Reference topology. auto (default): a plasmid map's own topology, FASTA contigs linear. circular: every contig is circular (a plasmid FASTA). Choices: `auto`, `circular`, `linear`. |
| `--region` | — | Keep only reads overlapping this region (1-based, inclusive; repeatable). The first region is the one outputs.json asks FiberBrowser to open. |
| `--min-mapq` | `20` | Keep primary alignments with at least this MAPQ (default 20); also passed to fiberhmm-call and fiberhmm-qc. |
| `--keep-soft-clips` | off | Keep unaligned read arms as soft clips. Default: hard-clip them for ddda/dddb (concatemer and chimera arms), keep them for hia5. |
| `--no-origin-merge` | off | Do not join the two pieces of reads that run through the origin of a circular reference (keep the primary piece). |
| `--aligner` | `auto` | The minimap2 program on PATH or the mappy module (default: auto, program first). Choices: `auto`, `minimap2`, `mappy`. |
| `--min-read-length` | — | Minimum aligned read length to call (default 1000). |
| `--dedup` | `auto` | DAF PCR-duplicate detection (default auto: on for file input). Choices: `auto`, `on`, `off`. |
| `--dedup-mode` | `flag` | flag (default): mark duplicates 0x400 and keep them; collapse: keep one read per duplicate cluster. Choices: `flag`, `collapse`. |
| `--snp-screen` | `auto` | DAF recurrent-SNP screen and mask (default auto: after a depth preflight). Choices: `auto`, `on`, `off`. |
| `--snp-mask` | — | DAF: your own BED of SNP sites to exclude. |
| `--chimera-filter` / `--no-chimera-filter` | on | DAF: skip strand-swap chimeric reads (default on). |
| `--primary` / `--no-primary` | on | Call primary alignments only (default on). |
| `--prob-threshold` | — | ML threshold override, 0-255 (default: chemistry preset). |
| `--use-m5c` / `--no-use-m5c` | auto | DddA CpG-aware recall (default: on for ddda). |
| `--cpg-mask-policy` | — | DddA CpG mask policy (default unmethylated-only). Choices: `unmethylated-only`, `methylated-only`. |
| `--call-args` | — | Other fiberhmm-call options, quoted as one string (e.g. --call-args "--with-scores --min-llr 6"). |
| `--call-mode` | `auto` | auto (default): fiberhmm-call's resumable region-parallel mode for genome-scale data (&gt;=20,000 reads over at least --cores regions), streaming for targeted runs (one amplicon or plasmid), which it calls faster. Choices: `auto`, `streaming`, `resumable`. |
| `--no-qc` | off | Skip fiberhmm-qc. |
| `--tracks` | off | Also extract nucleosome/MSP/TF/deamination (or m6A) tracks into OUTDIR/tracks (bigBed; BED without bedToBigBed). |
| `--redo` | — | Redo this step and the later ones although complete (also needed to change the inputs or settings of an existing OUTDIR; discards an interrupted call's resumable state). Choices: `all`, `align`, `call`, `qc`, `tracks`. |
| `--progress-json` | — | Append JSON-lines progress events to FILE ('-' for stdout). |
| `-v` / `--verbose` | off | Echo the output of fiberhmm-call, -qc and -extract. |
| `-q` / `--quiet` | off | No progress messages on stderr. |

## fiberhmm-call

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input BAM. Use "-" for stdin (streaming mode). |
| `-o` / `--output` | required | Output BAM path or "-" for stdout (unsorted). |
| `-m` / `--model` | — | Apply HMM model JSON. If omitted, bundled model for --enzyme/--seq is used. |
| `--recall-model` | — | Separate model for TF LLR tables. Default: reuse apply model. |
| `--enzyme` | — | Bundled enzyme preset. Choices: `ddda`, `dddb`, `hia5`. |
| `--seq` | — | Sequencing platform. For Hia5 it selects the model; when omitted it is detected from the input (MM specs: PacBio T-a vs Nanopore A+a only; header records) and the run stops if the evidence conflicts. For dddb/ddda it only sets the declared platform. Choices: `pacbio`, `nanopore`. |
| `--replace-chemistry` | off | Replace, instead of reconcile with, the input BAM's FIBERHMM-CHEMISTRY declaration (re-calling a BAM with a deliberately different chemistry). |
| `--reference` | — | Reference FASTA for DAF-seq BAMs that lack both R/Y IUPAC encoding and MD tags. When present, acts as a fallback source for deamination-site detection (R/Y codes and a usable MD tag take precedence). Must match the BAM's assembly and be faidx-indexed. |
| `-k` / `--context-size` | — | Context size override. Default: from model. |
| `--edge-trim` | `10` | Bases to mask at edges (default 10) |
| `--min-mapq` | `0` | Min mapping quality (default 0) |
| `--prob-threshold` | — | Min MM/ML modification probability 0-255. Default: chemistry preset -- 248 for Hia5 Nanopore (--seq nanopore, given or detected), 128 otherwise. R/Y- and MD-encoded DAF input is binary and ignores it. |
| `--min-read-length` | `1000` | Min aligned read length (default 1000 — matches fiberhmm-apply) |
| `--msp-min-size` | `0` | Min MSP size (default 0) |
| `--nuc-min-size` | `85` | Min footprint size to count as nucleosome (default 85) |
| `--with-scores` / `--scores` | off | Write the HMM posterior-mean nq score of baseline nucleosomes (with --no-recall-nucs; nucleosome recall writes its own LLR-based nq). No aq is written. --scores is the fiberhmm-apply spelling. |
| `-r` / `--circular` | off | Enable circular molecule mode (3x tile internally, emit wrapped MA/AQ/AN annotations). |
| `--process-unmapped` / `--no-process-unmapped` | auto | Call unmapped reads that carry SEQ + MM/ML. Default: automatic -- on for stdin, unindexed and unaligned (uBAM) input, off (pass-through) for indexed aligned BAMs. A run that skips &gt;90% of records as unmapped fails unless --no-process-unmapped is given. |
| `--primary` / `--no-primary` | on | Call primary alignments only (default); secondary and supplementary records are passed through uncalled. --no-primary also calls them. Hard-clipped records whose MM/ML cannot match SEQ are always skipped (hard_clipped_mm). |
| `--min-llr` | — | Native LLR cost per TF interval in joint decoding (default: enzyme preset; not a calibrated FDR threshold). |
| `--min-opps` | `3` | Min informative target positions per TF call (default 3). |
| `--unify-threshold` | `90` | v2 nucs with nl &lt; this may be demoted to tf+ (default 90). |
| `--emission-uplift` | — | Emission power transform. Default: enzyme preset. |
| `--use-m5c` / `--no-use-m5c` | auto | DddA CpG-aware recall, as in fiberhmm-recall-tfs: CpG observations are excluded from nucleosome and TF recall except inside confident unmethylated island calls (MA ddda_ucg from fiberhmm-tag-m5c) the input already carries. Default: on for --enzyme ddda, off otherwise; --no-use-m5c for an ablation. |
| `--cpg-mask-policy` | `unmethylated-only` | With CpG-aware recall: keep CpGs only inside ddda_ucg islands (default), or mask only ddda_mcg spans (the former behaviour). Choices: `unmethylated-only`, `methylated-only`. |
| `--no-legacy-tags` | off | Skip ns/nl/as/al, emit only MA/AQ. |
| `--downstream-compat` | off | Skip MA/AQ; write TF calls into legacy ns/nl track. |
| `--recall-nucs` / `--no-recall-nucs` | auto | Split over-merged nucleosomes + resolve platform-aware edges (emits nuc.QQQ), promote nucleosome-sized TF leaks to nuc, and run the Pass-2 phase prior. ON by default for all enzymes (DddA uses phase-aware radial inference, others the accessible-cut Kadane split). Use --no-recall-nucs for baseline HMM nucleosomes (nuc.Q). |
| `--split-min-llr` | `4.0` | Min accessible-run LLR to split a nucleosome; for DddA, the molecule-local linker-residue configuration LLR (default 4.0). |
| `--split-min-opps` | `3` | Min informative positions in a nucleosome-splitting cut or DddA linker residue (default 3). |
| `--ddda-derived-tf-max-edge-gap` | `12` | DddA phase-aware radial recall only: TF calls exposed solely by nucleosome refinement must have a deamination hit within BP on both sides (default 12). Original HMM-accessible TF scan space is unchanged. Use -1 to disable the safeguard. |
| `--nuc-recall-policy` | `auto` | Nucleosome-recaller geometry policy. "auto" (default) uses topology-constrained, ambiguity-preserving recall for Nanopore and the conservative-edge policy otherwise. "topology" only accepts cuts that leave nucleosome-sized pieces and does not turn unresolved edge ambiguity into accessibility. Choices: `auto`, `conservative`, `topology`. |
| `--phase-nrl` | `auto` | Pass-2 periodicity prior (with --recall-nucs): "auto" (default; estimate the nucleosome repeat length from this sample after Pass 1, clamped to ~150-215 bp anchored at 185), "off", or a fixed bp value (e.g. 185). Long footprints are split at phase-predicted linkers using a lowered threshold gated on &gt;=1 local deamination event (never splits a signal-desert). |
| `--keep-chimeras` | off | DAF only: keep strand-swap chimeric reads (C-&gt;T in one segment + G-&gt;A in another). Default: filter them out and report the count. |
| `--chimera-min-seg` | `5` | DAF chimera: min same-strand deamination events per segment to call a swap (default 5). |
| `--chimera-purity` | `0.8` | DAF chimera: min same-strand purity per segment (default 0.8). |
| `--daf-mask-runs` | — | DAF only: thin targets lying in same-strand runs of &gt;= N original C (CT) or G (GA) bases (CC/GG and longer at N=2; see --daf-run-policy). Adjacent conversions are coupled and do not follow the per-site emission model. Default: 2 with keep-one for --enzyme ddda (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep the 5'-most target of each run (keep-one, default) or remove the whole run (drop). Choices: `keep-one`, `drop`. |
| `--daf-snp-mask` | — | DAF only: 0-based BED of recurrent C&gt;T/G&gt;A SNP sites to exclude from deamination observations. MD is preserved. |
| `--daf-call-snps` | auto | DAF only: force two-pass recurrent opposite-conversion SNP masking. By default file-based DddA/DddB runs screen automatically after a bounded depth preflight. |
| `--no-daf-call-snps` | auto | Disable automatic recurrent SNP screening. |
| `--daf-snp-output-prefix` | — | Output prefix for --daf-call-snps (default: qc/&lt;output BAM stem&gt;.daf_snps). |
| `--daf-snp-min-fraction` | `0.2` | Minimum opposite-direction fiber fraction for --daf-call-snps (validated default 0.2). |
| `--daf-snp-min-depth` | `5` | Minimum fiber depth in each conversion-direction class for --daf-call-snps (validated default 5; all thresholds must pass in both classes). |
| `--daf-snp-min-alt-fibers` | `5` | Minimum alternate fibers in each direction for --daf-call-snps (validated default 5). |
| `--daf-snp-min-dominant-events` | `5` | Minimum dominant conversions to classify a fiber for --daf-call-snps (default 5). |
| `--daf-snp-min-dominant-purity` | `0.8` | Minimum conversion-direction purity for --daf-call-snps (default 0.80). |
| `--daf-snp-min-amplicon-reads` | `20` | Minimum aligned reads required to discover and plot an amplicon consensus in SNP QC (default 20). |
| `--ddda-mcg` | off | Deprecated integrated per-CpG mode; retained only to emit a clear migration error. Run fiberhmm-call, then fiberhmm-tag-m5c (whole CpG islands), then fiberhmm-recall-tfs --use-m5c. |
| `--dedup` | auto | DAF (ddda/dddb) only: force PCR duplicate detection (already automatic for file-based DddA/DddB calls). Detect by deamination-pattern fingerprint (see fiberhmm-dedup) and similar alignment ends BEFORE footprinting. The integrated default is nondestructive: retain every read and set 0x400 plus di/ds cluster tags. Amplicon/UMI-less DAF libraries can be heavily PCR-duplicated and coordinate dedup does not apply. Requires a file input (not stdin). Ignored for fiber-seq (hia5). |
| `--no-dedup` | auto | Disable automatic DddA/DddB duplicate marking. |
| `--dedup-min-jaccard` | `0.95` | Deamination-set Jaccard threshold for --dedup (default 0.95). |
| `--dedup-flag-only` | off | Deprecated compatibility spelling for the nondestructive integrated default (0x400 + di/ds; retain every read). |
| `--dedup-collapse` | off | With --dedup: destructively collapse each duplicate cluster to one representative. Default: mark and retain all reads. |
| `--dedup-min-deam` | `10` | With --dedup: reads with fewer deamination calls are not fingerprinted and pass through (default 10). |
| `--dedup-prob-threshold` | — | With --dedup: min ML probability for MM/ML-native dU calls, 0-255 (default: the calling --prob-threshold, 128). R/Y and MD inputs are binary and ignore it. |
| `--dedup-ignore-strand` | off | With --dedup: cluster reads across deamination flavours (C-&gt;T with G-&gt;A reads). Default: only reads of the same flavour can be duplicates. |
| `--dedup-max-end-diff` | `50` | With --dedup: maximum difference at both aligned reference ends for duplicate matching (default 50 bp). |
| `--dedup-stats-tsv` | — | With --dedup: write a cluster_id&lt;TAB&gt;n_reads table. |
| `-c` / `--cores` | `4` | Worker processes (0 = all CPUs; default 4). |
| `--chunk-size` | `500` | Reads per worker chunk (default 500; streaming mode only). |
| `--io-threads` | `8` | htslib I/O threads per stage (default 8). |
| `--max-reads` | `0` | 0 = no limit (default; streaming mode only) |
| `--qc` / `--no-qc` | on | Run bounded signal/periodicity/footprint QC after a file output (default: on; use --no-qc to disable). |
| `--qc-sample-reads` | `2000` | Target random-window QC sample size (default 2000). |
| `--qc-seed` | `20260824` | Deterministic QC sampler seed (default 20260824). |
| `--qc-min-mapq` | `20` | Minimum mapping quality for the QC sample (default 20). |
| `--qc-output-prefix` | — | QC output prefix (default: qc/&lt;BAM stem&gt; beside output BAM). |
| `--region-parallel` | off | Process genomic regions in parallel (one worker per region). Scales linearly with --cores up to chromosome count. Requires coordinate-sorted + indexed input BAM. Recommended for full-genome runs; use streaming for stdin/unaligned. |
| `--region-size` | `10000000` | Region size in bp for --region-parallel (default 10 Mb). |
| `--skip-scaffolds` | off | Skip scaffold/contig chromosomes in region-parallel mode. |
| `--chroms` | — | Only process these chromosomes (region-parallel mode). |
| `--resume` | off | Continue an interrupted --region-parallel run: regions finished in the work directory are reused, missing or partial regions are rerun, then the output is merged and published as usual. Refused if the input BAM or any effective parameter changed. Implies --region-parallel for an indexed, aligned input; with no work directory present a new run starts. Streaming (stdin, unindexed) runs cannot resume. |
| `--work-dir` | — | Region-parallel work directory holding finished region BAMs and the resume manifest (default: .&lt;output name&gt;.fiberhmm-work beside the output). Kept when a run fails or is interrupted; removed after a successful publish. |
| `--keep-work-dir` | off | Keep the region-parallel work directory after a successful run. |
| `--progress-json` | — | Write machine-readable progress, one JSON object per line, to stderr (no value) or append to FILE: start, region (regions done/total, reads, reads/s, ETA), merge, done and stopped events (schema fiberhmm.progress.v1). |

## fiberhmm-apply

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input BAM file with modification calls, or "-" for stdin (unaligned and unindexed BAMs are streamed) |
| `-m` / `--model` | — | Path to trained HMM model (.json, .npz, or .pickle). If omitted, the bundled model for --enzyme/--seq is used. |
| `-o` / `--outdir` | required | Output directory, or "-" to write BAM to stdout (for piping) |
| `--enzyme` | — | Auto-select a supported bundled chemistry model. Use --seq pacbio\|nanopore for Hia5. Choices: `ddda`, `dddb`, `hia5`. |
| `--seq` | — | Hia5 sequencing platform. When omitted it is detected from the input (MM specs: PacBio T-a vs Nanopore A+a only; header records); conflicting evidence stops the run. Ignored for dddb/ddda. Choices: `pacbio`, `nanopore`. |
| `-k` / `--context-size` | — | Context size (auto-detected from model if not specified) |
| `--cores` / `-c` | `1` | Number of CPU cores (0=auto, default: 1) |
| `--io-threads` | `4` | Number of htslib decompression/compression threads for BAM I/O (default: 4) |
| `--streaming` | off | Use streaming pipeline mode (works with unaligned/unindexed BAMs and stdin). Recommended for unaligned data or when reading from pipes. |
| `--chunk-size` | `500` | Reads per compute chunk in streaming mode (default: 500) |
| `--min-mapq` / `-q` | `0` | Minimum mapping quality; reads below this are written to output unchanged without footprint/nucleosome tags. Default 0 (call on all mapped reads). Pass a positive value to filter. |
| `--prob-threshold` | — | Minimum MM/ML probability (0-255) to call a modification. Default: chemistry preset -- 248 for Hia5 Nanopore (--seq nanopore, given or detected), 128 otherwise. R/Y- and MD-encoded DAF input is binary and ignores it. |
| `--min-read-length` | `1000` | Minimum aligned read length in bp; shorter reads are written to output unchanged without footprint/nucleosome tags. Set to 0 to attempt calling on all reads regardless of length (default: 1000) |
| `-t` / `--train-reads` | — | TSV file of read IDs used in training (to exclude) |
| `--primary` / `--no-primary` | on | Call primary alignments only (default); secondary and supplementary records are passed through uncalled. --no-primary also calls them. Hard-clipped records whose MM/ML cannot match SEQ are always skipped (hard_clipped_mm). |
| `--process-unmapped` / `--no-process-unmapped` | auto | Process unmapped reads that have sequences and modification tags. Default: automatic -- on for stdin, unindexed and unaligned (uBAM) input. A run that skips &gt;90% of records as unmapped fails unless --no-process-unmapped is given. |
| `--edge-trim` / `-e` | `10` | Bases to trim from read edges (default: 10) |
| `-r` / `--circular` | off | Enable circular mode (tiles reads 3x) |
| `--scores` | off | Compute per-footprint confidence scores (slower but more informative) |
| `--msp-min-size` | `0` | Minimum size for MSP regions in bp. Default 0 (emit every accessible run; matches fibertools, which does not impose an MSP size filter at this stage). Pass a positive value to filter. |
| `--nuc-min-size` | `85` | Minimum footprint size (bp) to count as nucleosome-sized for MSP boundary detection. Only footprints &gt;= this size split MSPs; smaller footprints are absorbed (default: 85) |
| `--no-msps` | off | Do not write MSP tags (as/al/aq) to output BAM. Useful for Fiber-seq where MSPs are computed differently by fibertools |
| `--stats` | off | Generate summary statistics and plots |
| `--stats-sample` | `10000` | Number of reads to sample for statistics (default: 10000) |
| `--stats-seed` | `42` | Random seed for sampling (default: 42) |
| `--output-posteriors` | — | Export HMM posteriors to file (H5 or TSV) |
| `--debug-timing` | off | Show per-read timing breakdown |
| `--daf-mask-runs` | — | Thin DAF targets in same-strand runs of &gt;= N original C (CT) or G (GA) bases (N=2: CC/GG and longer). Default: 2 with keep-one for DddA (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. Choices: `keep-one`, `drop`. |

## fiberhmm-recall-tfs

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--in-bam` | required | Input BAM tagged by fiberhmm-apply (has ns/nl/as/al). Use "-" for stdin. |
| `-o` / `--out-bam` | required | Output BAM with MA/AQ + refreshed legacy tags. Use "-" for stdout (for piping to ft fire, samtools, etc). |
| `-m` / `--model` | — | FiberHMM model JSON. If omitted, the bundled model for --enzyme/--seq is used automatically. |
| `--enzyme` | — | Enzyme preset: auto-selects the bundled model and min-llr/emission-uplift defaults (ddda, dddb, hia5). Choices: `ddda`, `dddb`, `hia5`. |
| `--seq` | — | Hia5 sequencing platform. When omitted it is taken from the input's FIBERHMM-CHEMISTRY declaration or detected from its MM specs (PacBio T-a vs Nanopore A+a only); conflicting evidence stops the run. Ignored for dddb/ddda. Choices: `pacbio`, `nanopore`. |
| `--replace-chemistry` | off | Replace, instead of reconcile with, the input BAM's FIBERHMM-CHEMISTRY declaration. By default a custom --model inherits the input's enzyme/platform when its observation mode matches, and a conflicting --enzyme/--seq is refused. |
| `--daf-mask-runs` | — | DAF only: thin targets lying in same-strand runs of &gt;= N original C (CT) or G (GA) bases (CC/GG and longer at N=2; see --daf-run-policy). Adjacent conversions are coupled and do not follow the per-site emission model. Default: 2 with keep-one for --enzyme ddda (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. Choices: `keep-one`, `drop`. |
| `--min-llr` | — | Override native LLR cost per TF interval in joint decoding (nats; default: enzyme preset; not an FDR threshold). |
| `--prob-threshold` | — | Min MM/ML probability 0-255 for re-reading modification calls. Default: chemistry preset -- 248 for Hia5 Nanopore (--seq nanopore, or the input's declared chemistry), 125 otherwise. R/Y- and MD-encoded DAF input is binary and ignores it. |
| `--min-opps` | `3` | Min informative target positions per call (default 3) |
| `--emission-uplift` | — | Power transform on emission probabilities. Default 1.0 (identity). Use a pre-uplifted model file (e.g. ddda_TF.json) for DddA rather than setting this. |
| `--use-m5c` / `--no-use-m5c` | auto | Enable CpG-aware DddA recall. By default, retain CpGs only inside confident ddda_ucg MA spans. Enabled automatically for DddA, disabled for other enzymes; use --use-m5c explicitly with a custom DddA model. |
| `--cpg-mask-policy` | `unmethylated-only` | CpG-aware policy: retain CpGs only inside confident unmethylated islands (default), or reproduce the former behavior that masks only ddda_mcg spans. Choices: `unmethylated-only`, `methylated-only`. |
| `--unify-threshold` | `90` | v2 nucs with nl &lt; this are scanned + may be demoted to tf+ if overlapped by a recaller call (default 90) |
| `--input-frame` | `auto` | Coordinate frame of the input ns/nl/as/al tags. "auto" (default) follows the @PG PP chain: the last footprint writer decides (fibertools predict-m6a/add-nucleosomes/fire -&gt; molecular; FiberHMM call/apply/recall -&gt; molecular when declared coord=molecular, else query); no writer -&gt; molecular if coord=molecular is declared, else query (FiberHMM &lt;= 2.12). Merged histories whose writers disagree, with no declaration, stop the run at the first read with these tags. A wrong frame mirrors reverse-strand calls. fibertools Ma tags are always molecular. Choices: `auto`, `molecular`, `query`. |
| `--no-legacy-tags` | off | Skip refreshed ns/nl/as/al -- emit only MA/AQ. |
| `--downstream-compat` | off | Downstream-compatibility mode: skip MA/AQ entirely and write TF calls INTO the legacy ns/nl tag alongside nucleosomes (sorted by start). Use for older tools that do not understand the Molecular-annotation spec. Loses per-TF quality scoring (tq/el/er) -- positions and lengths only. |
| `-c` / `--cores` | `1` | Worker processes (0 = all CPUs; default 1). |
| `--chunk-size` | `1024` | Reads per worker chunk (default 1024). Larger values reduce IPC overhead; decrease if RAM is constrained (each chunk holds reads in memory). |
| `--io-threads` | `4` | htslib BAM compression threads (default 4). |
| `--context-size` | — | Override context size. Default: read from model. |
| `--max-reads` | `0` | 0 = no limit (default) |
| `--recall-nucs` / `--no-recall-nucs` | off | Enable nucleosome recall before TF recall. (Default on for fiberhmm-recall-nucs.) |
| `--split-min-llr` | `4.0` | Min accessible-cut LLR to split a footprint; for DddA, the molecule-local linker-residue configuration LLR (default 4.0) |
| `--split-min-opps` | `3` | Min informative positions for a split cut or DddA linker residue (default 3) |
| `--ddda-derived-tf-max-edge-gap` | `12` | DddA phase-aware radial recall only: require TF scan space opened solely by nucleosome refinement to have a deamination hit within BP on both sides (default 12; -1 disables). |
| `--nuc-recall-policy` | `auto` | "auto" uses topology-constrained, ambiguity-preserving recall for Nanopore and conservative edges otherwise. Choices: `auto`, `conservative`, `topology`. |
| `--nuc-min-size` | `85` | Min refined nucleosome size; smaller footprints are demoted to accessible/MSP (default 85) |
| `--msp-min-size` | `0` | Min re-derived MSP size to keep (default 0) |
| `--phase-nrl` | `auto` | Pass-2 periodicity prior: off / auto / fixed bp. "auto" (default) estimates the nucleosome repeat length from the input BAM's existing nuc tags (no HMM re-run). Lowers the split bar near phase-predicted linkers in long footprints. |

## fiberhmm-recall-nucs

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--in-bam` | required | Input BAM tagged by fiberhmm-apply (has ns/nl/as/al). Use "-" for stdin. |
| `-o` / `--out-bam` | required | Output BAM with MA/AQ + refreshed legacy tags. Use "-" for stdout (for piping to ft fire, samtools, etc). |
| `-m` / `--model` | — | FiberHMM model JSON. If omitted, the bundled model for --enzyme/--seq is used automatically. |
| `--enzyme` | — | Enzyme preset: auto-selects the bundled model and min-llr/emission-uplift defaults (ddda, dddb, hia5). Choices: `ddda`, `dddb`, `hia5`. |
| `--seq` | — | Hia5 sequencing platform. When omitted it is taken from the input's FIBERHMM-CHEMISTRY declaration or detected from its MM specs (PacBio T-a vs Nanopore A+a only); conflicting evidence stops the run. Ignored for dddb/ddda. Choices: `pacbio`, `nanopore`. |
| `--replace-chemistry` | off | Replace, instead of reconcile with, the input BAM's FIBERHMM-CHEMISTRY declaration. By default a custom --model inherits the input's enzyme/platform when its observation mode matches, and a conflicting --enzyme/--seq is refused. |
| `--daf-mask-runs` | — | DAF only: thin targets lying in same-strand runs of &gt;= N original C (CT) or G (GA) bases (CC/GG and longer at N=2; see --daf-run-policy). Adjacent conversions are coupled and do not follow the per-site emission model. Default: 2 with keep-one for --enzyme ddda (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. Choices: `keep-one`, `drop`. |
| `--min-llr` | — | Override native LLR cost per TF interval in joint decoding (nats; default: enzyme preset; not an FDR threshold). |
| `--prob-threshold` | — | Min MM/ML probability 0-255 for re-reading modification calls. Default: chemistry preset -- 248 for Hia5 Nanopore (--seq nanopore, or the input's declared chemistry), 125 otherwise. R/Y- and MD-encoded DAF input is binary and ignores it. |
| `--min-opps` | `3` | Min informative target positions per call (default 3) |
| `--emission-uplift` | — | Power transform on emission probabilities. Default 1.0 (identity). Use a pre-uplifted model file (e.g. ddda_TF.json) for DddA rather than setting this. |
| `--use-m5c` / `--no-use-m5c` | auto | Enable CpG-aware DddA recall. By default, retain CpGs only inside confident ddda_ucg MA spans. Enabled automatically for DddA, disabled for other enzymes; use --use-m5c explicitly with a custom DddA model. |
| `--cpg-mask-policy` | `unmethylated-only` | CpG-aware policy: retain CpGs only inside confident unmethylated islands (default), or reproduce the former behavior that masks only ddda_mcg spans. Choices: `unmethylated-only`, `methylated-only`. |
| `--unify-threshold` | `90` | v2 nucs with nl &lt; this are scanned + may be demoted to tf+ if overlapped by a recaller call (default 90) |
| `--input-frame` | `auto` | Coordinate frame of the input ns/nl/as/al tags. "auto" (default) follows the @PG PP chain: the last footprint writer decides (fibertools predict-m6a/add-nucleosomes/fire -&gt; molecular; FiberHMM call/apply/recall -&gt; molecular when declared coord=molecular, else query); no writer -&gt; molecular if coord=molecular is declared, else query (FiberHMM &lt;= 2.12). Merged histories whose writers disagree, with no declaration, stop the run at the first read with these tags. A wrong frame mirrors reverse-strand calls. fibertools Ma tags are always molecular. Choices: `auto`, `molecular`, `query`. |
| `--no-legacy-tags` | off | Skip refreshed ns/nl/as/al -- emit only MA/AQ. |
| `--downstream-compat` | off | Downstream-compatibility mode: skip MA/AQ entirely and write TF calls INTO the legacy ns/nl tag alongside nucleosomes (sorted by start). Use for older tools that do not understand the Molecular-annotation spec. Loses per-TF quality scoring (tq/el/er) -- positions and lengths only. |
| `-c` / `--cores` | `1` | Worker processes (0 = all CPUs; default 1). |
| `--chunk-size` | `1024` | Reads per worker chunk (default 1024). Larger values reduce IPC overhead; decrease if RAM is constrained (each chunk holds reads in memory). |
| `--io-threads` | `4` | htslib BAM compression threads (default 4). |
| `--context-size` | — | Override context size. Default: read from model. |
| `--max-reads` | `0` | 0 = no limit (default) |
| `--recall-nucs` / `--no-recall-nucs` | on | Enable nucleosome recall before TF recall. (Default on for fiberhmm-recall-nucs.) |
| `--split-min-llr` | `4.0` | Min accessible-cut LLR to split a footprint; for DddA, the molecule-local linker-residue configuration LLR (default 4.0) |
| `--split-min-opps` | `3` | Min informative positions for a split cut or DddA linker residue (default 3) |
| `--ddda-derived-tf-max-edge-gap` | `12` | DddA phase-aware radial recall only: require TF scan space opened solely by nucleosome refinement to have a deamination hit within BP on both sides (default 12; -1 disables). |
| `--nuc-recall-policy` | `auto` | "auto" uses topology-constrained, ambiguity-preserving recall for Nanopore and conservative edges otherwise. Choices: `auto`, `conservative`, `topology`. |
| `--nuc-min-size` | `85` | Min refined nucleosome size; smaller footprints are demoted to accessible/MSP (default 85) |
| `--msp-min-size` | `0` | Min re-derived MSP size to keep (default 0) |
| `--phase-nrl` | `auto` | Pass-2 periodicity prior: off / auto / fixed bp. "auto" (default) estimates the nucleosome repeat length from the input BAM's existing nuc tags (no HMM re-run). Lowers the split bar near phase-predicted linkers in long footprints. |

## fiberhmm-qc

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | One or more FiberHMM-compatible BAM/CRAM files; -i may be repeated |
| `-o` / `--output-dir` | — | QC output directory (default: qc/ beside the input BAMs) |
| `--mode` | `auto` | Observation mode (default: infer from @PG/tags) Choices: `auto`, `daf`, `pacbio-fiber`, `nanopore-fiber`. |
| `--enzyme` | `auto` | Enzyme (default: infer from @PG) Choices: `auto`, `hia5`, `ddda`, `dddb`. |
| `--reference-profile` | `auto` | Empirical QC reference (default: assay-aware auto selection) Choices: `auto`, `none`, `ddda`, `dddb`, `hia5_nanopore`, `hia5_pacbio`. |
| `--reference` | — | Indexed FASTA fallback for raw DAF BAMs lacking MD/R/Y |
| `--sample-reads` | `2000` | Target bounded sample size (default 2,000) |
| `--seed` | `20260824` | Deterministic sampler seed (default 20260824) |
| `--min-mapq` | `20` | Minimum mapping quality (default 20) |
| `--prob-threshold` | — | Minimum MM/ML probability, 0-255 (default: 248 for Hia5 Nanopore, the threshold its QC reference is calibrated at; 125 otherwise) |
| `--min-opportunities` | `200` | Minimum target sites per read for rate QC (default 200) |
| `--snp-mask` | — | Applied DAF SNP-mask BED to summarize (single input only) |
| `--snp-report` | — | fiberhmm-daf-snps JSON to plot (single input only) |
| `--state-source` | `auto` | Where the in-MSP/outside-MSP split comes from: the BAM's FiberHMM calls when the sample carries them, else a bounded light call with the bundled model of the declared chemistry (auto, default); calls only (tags); always re-call (light-call); or skip (none) Choices: `auto`, `tags`, `light-call`, `none`. |
| `--min-msp-bp` | `85` | Shortest MSP counted as accessible for the in-MSP rate; shorter gaps (linkers) count as outside-MSP (default 85; the packaged references are calibrated at this value) |
| `--light-call-reads` | `400` | Most sampled reads the light call runs on (default 400) |
| `--light-call-seconds` | `60.0` | Wall-time budget of the light call (default 60 s) |
| `--fail-on-qc` | off | Exit 2 when the final status is FAIL |

## fiberhmm-check

| Flag | Default | Description |
|------|---------|-------------|
| `PATH` | — | BAM/CRAM/SAM, &lt;prefix&gt;.qc.json, posteriors .tsv(.gz)/.h5, a fiberhmm-consensus output directory, or a fiberhmm-pipeline output directory |
| `--json` | off | Print one JSON document (schema fiberhmm.advisory_check.v1) with a fiberhmm.advisory_report.v1 report per path |
| `--scan-records` | — | BAM records to scan for read-level evidence (duplicate, pairing and call tags); 0 reads the header only (default: 5000) |
| `--no-sidecars` | off | Do not also check the QC report fiberhmm-call writes beside a BAM (qc/&lt;name&gt;.qc.json) |
| `--list` | off | List every known advisory and exit |

## fiberhmm-extract

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input tagged BAM file |
| `-o` / `--outdir` | — | Output directory (default: same as input) |
| `-c` / `--cores` | `1` | Number of CPU cores |
| `--nucleosome` / `--footprint` | off | Extract nucleosomes (ns/nl or MA nuc tags). (--footprint is a deprecated alias.) |
| `--msp` | off | Extract MSPs (as/al tags) |
| `--tf` | off | Extract TF/Pol II footprints from MA/AQ tag (tf.QQQ). Requires a BAM called by fiberhmm-call or fiberhmm-recall-tfs (not --downstream-compat). See --min-tq for the quality floor. |
| `--min-tq` | `50` | Minimum TF quality (tq) to extract. 0-255 scale where tq = min(255, round(LLR * 10)). Default 50 (LLR &gt;= 5 nats, ~148:1 likelihood ratio), the per-interval cost every bundled preset calls with, so it keeps essentially every call. Set 0 for every emitted call, 100+ for a stricter view. |
| `--m6a` | off | Extract m6A positions |
| `--m5c` | off | Extract DddA MA ddda_mcg spans, or native MM/ML 5mC positions |
| `--deam` | off | Extract DAF-seq deamination calls. Priority: (1) MM/ML-native dU calls (mod code "u" or ChEBI 55797); (2) IUPAC R/Y codes in the query sequence (fiberhmm-daf-encode output); (3) MD-tag ref mismatches as a fallback for raw DAF BAMs. First non-empty source wins per read. blockMod: 0 = R/GA-dea, 1 = Y/CT-dea, matching FiberBrowser flavor codes. |
| `--both-strand` / `--bothstrand` | off | Extract the deam+ / deam- intersection of paired DddA consensus reads as a BED12 coverage overlay. |
| `--all` | off | Extract all tag types (default if none specified) |
| `--bed-only` | off | Output BED only (no bigBed) |
| `--keep-bed` | off | Keep BED files when creating bigBed |
| `-q` / `--min-mapq` | `0` | Min mapping quality (default: 0, no filtering) |
| `-p` / `--prob-threshold` | — | Min probability for native MM/ML m6a/m5c/dU calls (0-255). Default: 248 when the BAM declares Hia5 Nanopore (FIBERHMM-CHEMISTRY header), 125 otherwise. Not applied to DddA MA ddda_mcg spans or to R/Y/MD deaminations (binary). |
| `--no-scores` | off | Omit scores from output |
| `--block-scores` | off | Append per-block quality as extra BED column(s) (BED12+N). nucleosome -&gt; blockNq/blockEl/blockEr, msp -&gt; blockAq, m6a/native-m5c -&gt; blockMl (DddA ddda_mcg spans -&gt; 0), tf -&gt; blockTq/blockEl/blockEr. bigBed uses -type=bed12+N and the matching autoSQL schema so FiberBrowser/UCSC can surface per-feature quality without a sidecar database. |
| `--circular-groups` | off | Append circId/circPart/circParts/molStart/molLength columns and preserve MA/AN groups for circular wrapped nucleosome, MSP, and TF features. |
| `--haplotype-fields` | off | Append scalar hp and ps columns copied from the source BAM HP/PS tags after every other optional BED field. Missing or non-integer tags are written as -1. Off by default to preserve the existing BED/bigBed schema byte-for-byte. |
| `--sample-name` | — | Sample/dataset identifier to embed in the autoSQL description of every output bigBed ("Sample: &lt;name&gt;. ..."). Default: BAM basename stem. Lets downstream tools match a bigBed to its source without filename parsing. |
| `--region-size` | `10000000` | Region size for parallel |
| `--skip-scaffolds` | off | Skip scaffold chromosomes |
| `--chroms` | — | Comma-separated chromosomes to process |
| `--sort-mem` / `-S` | `1G` | Memory buffer for the BED sort, passed to `sort -S` (e.g. 4G, 8G, or 50% on GNU sort). Bigger = fewer temp-file merge passes = faster. Default 1G; pass an empty string to disable. The sort also runs under LC_ALL=C regardless, which alone is a large speedup. |
| `--sort-parallel` | `0` | Threads for the BED sort (GNU coreutils only; ignored on BSD/macOS sort). 0 = use --cores. Feature-detected, so safe to leave on. |

## fiberhmm-dedup

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input DAF-seq BAM (indexed not required) |
| `-o` / `--output` | required | Output BAM |
| `--min-jaccard` | `0.95` | Min deamination-set Jaccard to call two reads the same molecule (default 0.95; the bimodal gap sits ~0.90-0.95). Lower = more aggressive collapsing. |
| `--flag-only` | off | Keep all reads and only mark duplicates: set the 0x400 duplicate flag on non-representatives and di/ds cluster tags on every member of a duplicate cluster. Default: collapse each cluster to one representative read. |
| `--min-deam` | `10` | Reads with fewer than this many deamination calls are not fingerprintable; passed through untouched (default 10). |
| `--max-end-diff` | `50` | Maximum difference at both aligned reference ends for two reads to be duplicates (default 50 bp). |
| `--ignore-strand` | off | Cluster across deamination flavours. Default: only reads with the same dominant flavour (C-&gt;T vs G-&gt;A, i.e. the same template strand) can be duplicates; alignment orientation is never used. |
| `-p` / `--prob-threshold` | `128` | Min ML probability for MM/ML-native dU calls (0-255; default 128, the same as fiberhmm-call). R/Y- and MD-encoded input is binary and ignores it. |
| `--num-hashes` | `32` | MinHash signature width (default 32). |
| `--bands` | `8` | LSH bands; rows = num-hashes / bands (default 8 -&gt; rows 4). More bands = higher recall, more candidate pairs. |
| `--seed` | `7` | MinHash RNG seed (default 7). |
| `--stats-tsv` | — | Write a cluster_id&lt;TAB&gt;n_reads table to this path. |
| `--io-threads` | `4` | htslib BAM compression threads for output (default 4). |

## fiberhmm-daf-encode

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input BAM file, or "-" for stdin |
| `-o` / `--output` | required | Output BAM path, or "-" for stdout |
| `--reference` | — | Reference FASTA (fallback if MD tag is missing) |
| `-q` / `--min-mapq` | `20` | Minimum mapping quality (default: 20) |
| `--min-read-length` | `1000` | Minimum aligned read length in bp (default: 1000) |
| `--io-threads` | `4` | htslib I/O threads for BAM compression/decompression (default: 4) |
| `--strand` | `auto` | Force conversion strand: CT (+ strand), GA (- strand), or auto (per-read consensus, default: auto) Choices: `CT`, `GA`, `auto`. |

## fiberhmm-daf-snps

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input aligned BAM with MD tags |
| `-o` / `--output-prefix` | — | Output prefix (default: &lt;BAM stem&gt;.daf_snps) |
| `--min-fraction` | `0.2` | Minimum mismatch fraction in each direction (validated default 0.2) |
| `--min-depth` | `5` | Minimum depth in each conversion-direction class (validated default 5; all thresholds must pass in both classes) |
| `--min-alt-fibers` | `5` | Minimum mismatch-supporting fibers in each direction (validated default 5) |
| `--min-dominant-events` | `5` | Minimum dominant conversions per classifiable fiber (default 5) |
| `--min-dominant-purity` | `0.8` | Minimum dominant-direction purity (default 0.80) |
| `--min-mapq` | `20` | Minimum mapping quality (default 20) |
| `--min-amplicon-reads` | `20` | Minimum aligned reads required for an amplicon consensus (default 20) |
| `--reference` | — | Indexed FASTA fallback for BAMs lacking MD |

## fiberhmm-pair

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Coordinate-sorted, indexed FiberHMM-called DddA BAM |
| `-o` / `--output` | required | Output BAM: joint duplex molecules by default; paired source reads with --stop-after pair |
| `-r` / `--reference` | — | Matching indexed FASTA. Required by the default sequence-free score; optional with --sequence-only when MD+CIGAR is available |
| `--sequence-only` | off | Accept only direct A/T sequence-supported pairs; disable the sequence-free model |
| `--pairs-tsv` | — | Write selected-pair evidence to this TSV |
| `--receipt-json` | — | Write a machine-readable pairing receipt |
| `--pairs-only` / `--paired-only` | off | Write only paired records: merged duplex molecules (default), or paired source reads with --stop-after pair. Otherwise unpaired reads pass through unchanged. |
| `--stop-after` | `recall` | Last stage to run: pair (tag pairs only), merge (both-strand molecules without re-calling) or recall (default) Choices: `pair`, `merge`, `recall`. |
| `--from-paired` | off | Input is already pair-tagged (mt/mp from an earlier --stop-after pair run): skip pairing and start at merge. Pairing-stage options (--reference, --sequence-only, --pairs-tsv, --model, --min-* ...) are rejected here. |
| `--model` | — | Override the bundled frozen sequence-free model JSON |
| `--call-layer` | `auto` | Nucleosome calibration for the sequence-free model (default auto) Choices: `auto`, `input-ma`, `rotational-recall`. |
| `--min-margin` | `1.0` | Minimum two-sided sequence-free model margin (default 1.0) |
| `--null-floor` | `0.0` | Virtual null model score for a lone candidate (default 0.0) |
| `--min-overlap` | `1500` | Min genomic overlap bp (default 1500) |
| `--min-nucs` | `4` | Min nucleosome dyads within the overlap, each read (default 4) |
| `--min-sequence-bases` | `500` | Min shared reference-A/T bases for a sequence edge (default 500) |
| `--min-component-discordance-rate` | `0.02` | Min rejected-edge difference rate to constrain a 2x2 (default 0.02) |
| `--max-sequence-pair-rate` | `0.01` | Max difference rate on a sequence-selected pair (default 0.01) |
| `--min-sequence-margin` | `0.002` | Min sequence preference/assignment margin (default 0.002) |
| `-p` / `--prob-threshold` | `128` | Min ML probability for MM/ML-native dU calls (0-255; default 128, the same as fiberhmm-call). R/Y- and MD-encoded input is binary and ignores it. |
| `--max-component` | `10000` | Safety ceiling for a complete overlap component (default 10000) |
| `--io-threads` | `4` | htslib compression threads for output (default 4) |
| `--no-index` | off | Do not index a paired-source output |
| `--phase-nrl` | `196` | Nucleosome repeat length for consensus recall (default 196) |
| `--nuc-recall-policy` | `conservative` | Nucleosome policy for consensus recall Choices: `conservative`, `topology`. |
| `--ddda-derived-tf-max-edge-gap` | `12` | Edge-evidence requirement for TF calls exposed only by DddA nucleosome refinement (default 12; -1 disables) |
| `--use-m5c` / `--no-use-m5c` | auto | Joint recall: DddA CpG-aware recall, as in fiberhmm-call and fiberhmm-recall-tfs -- CpG observations are excluded except inside the source reads' ddda_ucg islands (fiberhmm-tag-m5c). Default: on; --no-use-m5c for an ablation. |
| `--daf-mask-runs` | — | Thin DAF targets in same-strand runs of &gt;= N original C (CT) or G (GA) bases (N=2: CC/GG and longer). Default: 2 with keep-one for DddA (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. Choices: `keep-one`, `drop`. |

## fiberhmm-merge

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | BAM from fiberhmm-pair (mt/mp tags) |
| `-o` / `--output` | required | Output consensus BAM (sorted + indexed) |
| `--pairs-only` | off | Emit only consensus reads (default: also pass through unmerged reads) |
| `--recall` | off | Re-call footprints on each both-strand consensus read (HMM, nucleosome and TF recall over both strands; writes ns/nl/as/al + MA nuc/msp/tf). Reads the deam+/deam- regime and uses C and G targets jointly. |
| `--enzyme` | `ddda` | Model preset for --recall (default ddda) |
| `--phase-nrl` | `196` | Nucleosome repeat length for consensus recall (default 196) |
| `--nuc-recall-policy` | `conservative` | Nucleosome geometry policy for consensus recall Choices: `conservative`, `topology`. |
| `--ddda-derived-tf-max-edge-gap` | `12` | With --recall, require TF calls exposed solely by DddA radial nucleosome refinement to have a deamination hit within BP on both sides (default 12; -1 disables). |
| `--use-m5c` / `--no-use-m5c` | auto | With --recall: DddA CpG-aware recall, as in fiberhmm-call and fiberhmm-recall-tfs (CpGs excluded except inside the source reads' ddda_ucg islands). Default: on for --enzyme ddda. |
| `-p` / `--prob-threshold` | `128` | Min ML probability for MM/ML-native dU calls (0-255; default 128, the same as fiberhmm-call). R/Y- and MD-encoded input is binary and ignores it. |
| `--io-threads` | `4` | htslib compression threads (default 4) |
| `--daf-mask-runs` | — | Thin DAF targets in same-strand runs of &gt;= N original C (CT) or G (GA) bases (N=2: CC/GG and longer). Default: 2 with keep-one for DddA (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. Choices: `keep-one`, `drop`. |

## fiberhmm-tag-m5c

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | R/Y-encoded, FiberHMM-tagged coordinate BAM |
| `-o` / `--output` | required | Output BAM |
| `-r` / `--reference` | required | Reference FASTA |
| `--enzyme` | required | Required chemistry assertion. Only DddA DAF-seq is supported. Choices: `ddda`. |
| `--posterior` | `0.99` | Whole-island posterior needed for a call: &gt;= this is methylated (ddda_mcg), &lt;= 1 - this unmethylated (ddda_ucg) (default 0.99). |
| `--min-other` | `10` | Minimum non-CpG observations on the molecule's island overlap, the internal accessibility baseline (default 10). |
| `--cpg-islands` | — | Optional BED3 of merged, non-overlapping CpG islands. Default: infer islands from the reference sequence. |
| `--write-cpg-islands` | — | Optional BED or BED.GZ containing the exact island catalog used. |
| `--min-island-cpg` | `15` | Minimum CpG observations for a whole-island call (default 15). |
| `--calls-tsv` | — | Optional whole-island audit table containing methylated, unmethylated and uninformative molecule/island overlaps. |
| `--cpg-island-window` | `200` | Reference inference window in bp (default 200). |
| `--cpg-island-step` | `10` | Reference inference step in bp (default 10). |
| `--cpg-island-min-gc` | `0.5` | Minimum GC fraction for inferred islands (default 0.50). |
| `--cpg-island-min-oe` | `0.6` | Minimum CpG observed/expected for inferred islands (default 0.60). |
| `--five-prime-factors` | — | Comma-separated A,C,G,T factors; default calibrated DddA values |
| `--estimate-factors` | off | Estimate 5' factors from this BAM instead of using calibrated values |
| `--factor-sample-reads` | `5000` | Reads sampled by --estimate-factors (default 5000). |
| `--input-frame` | `auto` | Frame of legacy ns/nl tags when MA is absent; auto uses the FiberHMM header marker Choices: `auto`, `molecular`, `query`. |
| `--io-threads` | `4` | htslib BAM compression threads (default 4). |

## fiberhmm-call-m5c

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | One or more coordinate-sorted, indexed DAF FiberHMM BAMs |
| `-r` / `--reference` | required | Indexed reference FASTA |
| `-o` / `--output` | required | Output BED6, or - for stdout |
| `--region` | required | contig[:start-end], 1-based display syntax |
| `--enzyme` | required | Required chemistry assertion. Only DddA DAF-seq is supported. Choices: `ddda`. |
| `--window` | `1000` | Aggregate window size in bp (default 1000). |
| `--chunk-bp` | `5000000` | Bound observation collection in this many bp; the HMM still runs once across the complete region |
| `--min-other` | `10` | Minimum non-CpG observations for a molecule's per-window baseline (default 10). |
| `--min-cpg` | `10` | Minimum CpG observations for an informative window (default 10). |
| `--posterior` | `0.99` | Window posterior needed to join a called domain (default 0.99). |
| `--max-gap` | `1000` | Maximum gap in bp bridged between called windows of the same state (default 1000). |
| `--five-prime-factors` | — | Comma-separated A,C,G,T factors; default calibrated DddA values |
| `--estimate-factors` | off | Estimate 5' factors from the complete supplied region (explicit high-memory audit option) |
| `--tag-bam` | — | Optionally add whole-island annotations to this BAM |
| `--tag-output` | — | Output BAM for --tag-bam (required with --tag-bam) |
| `--tag-mode` | `island` | island: one molecule state per complete CpG island (default); locus: copy aggregate validation domains Choices: `island`, `locus`. |
| `--cpg-islands` | — | Optional merged BED3 island catalog; default: infer from reference |
| `--write-cpg-islands` | — | Optional BED/BED.GZ recording the exact catalog used |
| `--read-posterior` | `0.99` | Whole-island posterior threshold (default: 0.99) |
| `--read-min-island-cpg` | `15` | Minimum CpGs per molecule-island call (default: 15) |
| `--tag-input-frame` | `auto` | Frame of tag-BAM legacy ns/nl when MA is absent Choices: `auto`, `molecular`, `query`. |
| `--io-threads` | `4` | htslib BAM threads (default 4). |

## fiberhmm-consensus

| Flag | Default | Description |
|------|---------|-------------|
| `--schema` | off | Print every parameter group with defaults and help as JSON (cr.engine selects the engine; the default is lattice_recaller) |
| `--bam` | — | Repeat for separate datasets; chemistry comes from BAM @CO metadata |
| `--datasets` | — | JSON list of {dataset_id, paths: [BAMs], chemistry?: profile} |
| `--evidence` | — | Saved evidence.json.gz, including pooled evidence |
| `--resume` | — | Start a NEW run (in a new --output) from a finished run directory: reuses its evidence.json.gz, manifest.json and fit_cache to rerun stages. To finish an interrupted multi-window run in place, use --continue |
| `--bed` | — | BED3 windows for independent runs; BED6 with equal widths for CL-CR |
| `--region` | — | Alternative CHROM:START-END, explicitly 0-based half-open |
| `--pool-loci` | off | CL-CR: pool BED6 windows in their provided orientations |
| `--chemistry` | — | Explicit missing-metadata declaration for --bam; conflicts fail Choices: `ddda`, `dddb`, `hia5-pacbio`, `hia5-nanopore`. |
| `--parameters` | — | JSON parameter groups; see --schema |
| `--engine` | — | Consensus engine (default lattice_recaller; staged_native_families is the deprecated Monte Carlo engine). Overrides cr.engine from --parameters or --resume Choices: `lattice_recaller`, `staged_native_families`. |
| `--consolidation-bp` | — | staged_native_families only: shared-family edge allowance (default 10; 5 gives finer grouping) |
| `--stop-after` | — | staged_native_families only: last stage to compute (the lattice recaller runs in one pass) Choices: `native`, `parents`, `consolidated`, `resolved`. |
| `--start-at` | `native` | staged_native_families only: consolidation restarts from saved native fits Choices: `native`, `consolidation`. |
| `--cores` | — | Worker processes (sets compute.cores; default 4) |
| `--robust` | — | lattice_recaller only, optional and slower: also rerun discovery under N other read orders and mark which classes are robust to read order (supported in every one of the N+1 orders, or the share set by recaller.order_robust_fraction; classes.tsv columns order_robustness and robust). Takes about N+1 times as long; the classes themselves are unchanged. Sets recaller.order_replicates (default 0 = off) |
| `--cache` | — | staged_native_families only: persistent exact native-fit cache directory |
| `--json-progress` / `--progress-json` | off | Structured progress on stderr (JSON lines) |
| `--daf-mask-runs` | — | DAF only: thin targets in same-strand runs of &gt;= N original C (CT) or G (GA) bases in lattices and native replay (2 = CC/GG and longer; 0 = off). Default: per dataset chemistry, DddA keep-one on runs &gt;= 2 (duplex-validated), DddB off |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run Choices: `keep-one`, `drop`. |
| `--no-bam` | off | Save frozen results/reports without materializing family-tagged BAMs |
| `--bam-scope` | `regions` | Export whole alignments overlapping analyzed windows (default), or the full source BAM Choices: `regions`, `full`. |
| `--bam-grouping` | `datasets` | One BAM per logical dataset (default) or original source file Choices: `datasets`, `files`. |
| `--bam-recaller-layer` | off | lattice_recaller: also write the optional tf_recaller MA layer (the recaller's own per-molecule class calls at every prevalence tier; bytes tq,fi,tier,q0,lr,rr) to exported BAMs. Off by default; the calls are always in result.json.gz |
| `--output` | — | New or empty result directory (with --continue: the interrupted run's directory) |
| `--continue` | off | Finish an interrupted --bam + --bed/--region run in its existing --output directory: windows whose completion marker matches are kept, missing or partial windows are rerun, and the aggregate outputs (regions.json, report.html, BAMs) are rebuilt. Refused if the BAMs, windows or parameters differ from the run's consensus_run.json (--cores, --window-jobs and --json-progress may change). Not the same as --resume, which starts a new run from saved evidence |
| `--window-jobs` | `0` | Independent BED windows analysed at the same time (default 0 = automatic: up to --cores windows, each with an equal share of --cores; one at a time for staged_native_families, whose per-window compute.maximum_matrix_mb budget is a hard limit). 1 = one window at a time, each using every core |

## fiberhmm-transfer

| Flag | Default | Description |
|------|---------|-------------|
| `--freeze-run` | — | Completed consensus result directory: a lattice_recaller run (any frame) or an oriented staged_native_families CL-CR run; export its classes/models without refitting |
| `--models` | — | frozen_classes.json.gz (lattice_recaller), frozen_models.json.gz (staged), or the --freeze-run output directory |
| `--bam` | — | Target BAM; repeat for separate datasets (needs --bed) |
| `--datasets` | — | JSON list of dataset_id and BAM paths |
| `--evidence` | — | Saved oriented native evidence.json.gz (one window or pooled evidence) |
| `--bed` | — | Equal-width, explicitly oriented BED6 target windows |
| `--chemistry` | — | Explicit missing-metadata declaration for --bam, as in fiberhmm-consensus Choices: `ddda`, `dddb`, `hia5-pacbio`, `hia5-nanopore`. |
| `--parameters` | — | BAM preparation parameters; target calls are replayed, families are never fitted |
| `--chip-bed` | — | Optional independent ChIP peaks, joined only after scoring |
| `--no-bam` | off | Do not write family-tagged BAMs |
| `--bam-scope` | `regions` | Export alignments overlapping the target windows (default), or the full source BAM Choices: `regions`, `full`. |
| `--bam-grouping` | `datasets` | One BAM per logical dataset (default) or per source file Choices: `datasets`, `files`. |
| `--json-progress` | off | Structured progress on stderr |
| `--cores` | — | Worker processes for scoring (default: this machine's consensus default) |
| `--dataset-map` | — | Use the frozen per-channel boxes and spots of source dataset SOURCE for target dataset TARGET (needed when several source datasets share the target chemistry) |
| `--include-training-molecules` | off | Score molecules the catalog was trained on (default: exclude them, as for staged families); use for self-application checks |
| `--output` | required | New or empty output directory |

## fiberhmm-footprint-model

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Recalled BAM containing ordinary tf and msp MA groups |
| `-o` / `--output-prefix` | required | Output filename prefix (not a directory) |
| `--region` | — | Analyze a zero-based, half-open region; repeatable and requires a BAM index. A bare CONTIG selects that complete contig. |
| `--genome` | — | Genome/assembly label recorded in provenance (for example dm6) |
| `--genomewide` | off | Assert that scanning the complete BAM represents a genome-wide analysis |
| `--bigbed` | off | Also create indexed population and FiberBrowser BigBeds |
| `--bed-to-bigbed` | — | Path to UCSC bedToBigBed; implies --bigbed |
| `--force` | off | Replace existing artifacts for this output prefix |
| `-q` / `--min-mapq` | `0` | Minimum alignment MAPQ (default: 0) |
| `--include-duplicates` | off | Include records carrying the BAM duplicate flag (excluded by default) |
| `--smoothing-sigma` | `3.0` | Footprint-center Gaussian sigma in bp (default: 3) |
| `--peak-distance` | `15` | Minimum distance between center modes (default: 15) |
| `--assignment-radius` | `10` | Maximum center-to-mode assignment radius (default: 10) |
| `--edge-compatibility` | `12` | Maximum within-family diameter for each boundary (default: 12) |
| `--minimum-geometry-support-per-stratum` | `3` | Molecules per stratum needed for that stratum to vote on canonical geometry (default: 3) |
| `--minimum-geometry-support` | `3` | Descriptive geometry-ready threshold (default: 3) |
| `--minimum-population-support` | `3` | Descriptive population-ready threshold (default: 3) |
| `--minimum-mapped-fraction` | `0.95` | Required mapped fraction for geometry, MSP projection, and site denominators (default: 0.95) |

## fiberhmm-strand-rescue

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--bam` | required | Post-TF/post-nuc BAM; repeat only to pool shards or compatible timepoints from one inference cohort |
| `--preset` | required | Choices: `ddda`, `dddb`, `hia5-nanopore`. |
| `--region` | required |  |
| `--model` | — |  |
| `--nuc-model` | — | Protected/accessibility model used only for nucleosome edge evidence |
| `--prob-threshold` | — |  |
| `--control-flank` | `2000` |  |
| `--min-mapq` | `20` |  |
| `--min-support` | `10` |  |
| `--minimum-geometry-support` | `3` |  |
| `--source-boundary-margin` | `10` |  |
| `--center-radius` | `10` |  |
| `--peak-distance` | `15` |  |
| `--tf-edge-compatibility` | `12` | Maximum within-family diameter in bp for each TF boundary; co-centered calls outside this bound form distinct TF models |
| `--max-boundary-mad` | `12.0` |  |
| `--min-local-enrichment` | `2.0` |  |
| `--local-background-radius` | `250` |  |
| `--max-auto-sites` | `0` | Optional TF-family cap after scoring; 0 keeps the exhaustive locus map |
| `--site` | — | Externally seed one zero-based half-open START-END site; repeatable. Support and canonical edges are recomputed from ordinary cohort calls. |
| `--forced-sites-only` | off | Analyze only --site geometries rather than unioning automatic sites |
| `--nuc-min-support` | — | Required ordinary nuc calls on one strand (default: --min-support) |
| `--nuc-center-radius` | `25` |  |
| `--nuc-source-boundary-margin` | `20` |  |
| `--nuc-edge-assignment-radius` | `48` |  |
| `--nuc-max-boundary-mad` | `24.0` |  |
| `--max-auto-nuc-sites` | `0` | Optional nucleosome-family cap after scoring; 0 keeps all families |
| `--nuc-site` | — | Seed one zero-based half-open nucleosome population; repeatable. Edges are relearned from ordinary nuc calls and no length ceiling applies. |
| `--forced-nuc-sites-only` | off | Refine only --nuc-site populations rather than automatic nuc sites |
| `--skip-nuc-edge-refinement` | on | Do not run independent one-for-one nucleosome edge normalization (default; retained for command-line compatibility) |
| `--independent-nuc-edge-refinement` | off | Experimental population nucleosome-edge normalization. This is not the TF-conditioned consensus-nuc reconciliation stage. |
| `--strand-min-source-support` | — | Required source-strand ordinary TF calls (default: --min-support) |
| `--strand-min-source-enrichment` | `1.5` | Minimum source-strand focal enrichment over local background |
| `--strong-posterior` | `0.95` |  |
| `--review-posterior` | `0.5` |  |
| `--tf-class-pseudocount` | `0.5` | Symmetric per-configuration pseudocount for localized single/composite site-consensus priors |
| `--tf-class-locus-gap` | `30` | Maximum gap joining atomic footprint states into one consensus locus |
| `--tf-class-max-span` | `250` | Maximum span in bp of one localized site-consensus locus |
| `--tf-class-max-sites` | `10` | Maximum atomic states enumerated in a complete site-consensus action set; larger loci are reported as skipped |
| `--maximum-sites-per-decision` | `8` |  |
| `--accessible-site-gap` | `220` | Maximum gap joining TF sites into one MSP-origin decision |
| `--control-shifts` | `` | Optional comma-separated target-coordinate shifts. Source priors remain at the true sites; controls are diagnostic, not an FDR null. |
| `--max-reads` | `0` |  |
| `--molecule-collapse` | `auto` | Collapse amplified DAF PCR families; auto enables for DddA/DddB Choices: `auto`, `on`, `off`. |
| `--molecule-min-jaccard` | `0.95` |  |
| `--molecule-min-deam` | `10` |  |
| `--per-molecule-efficiency` | on | Calibrate hard-call efficiency from each molecule's MSPs (default) |
| `--global-efficiency` | off | Use the model-wide accessible hard-call rate for every molecule |
| `--efficiency-pseudo-count` | `20.0` |  |
| `--efficiency-min-opportunities` | `20` |  |
| `--report-layout` | `auto` | Report action storage: auto spills at the bounded v4 limits; stream forces v5 BGZF action sidecars Choices: `auto`, `inline`, `stream`. |
| `--diagnostics` | `aggregate` | Per-call diagnostic storage (aggregate is the production default) Choices: `aggregate`, `stream`. |
| `--proposal-tsv` | — |  |
| `-o` / `--output` | required |  |

## fiberhmm-strand-rescue-annotate

| Flag | Default | Description |
|------|---------|-------------|
| `--report` | required |  |
| `-i` / `--bam` | — | Report input BAM to materialize; default is every report BAM |
| `-o` / `--output` | — | Output BAM; requires exactly one selected input |
| `--output-dir` | — | Directory receiving one regional BAM per selected input |
| `--region` | — |  |
| `--minimum-posterior` | `0.0` | Optional output-size filter; default 0 preserves every decision |
| `--allow-input-drift` | off |  |
| `--io-threads` | `1` |  |

## fiberhmm-strand-rescue-audit

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--bam` | required |  |
| `-o` / `--output` | — |  |
| `--max-errors` | `100` |  |

## fiberhmm-tag-consensus

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input BAM with tf_sr.QQQ |
| `-o` / `--output` | required | New sorted/indexed BAM |
| `-a` / `--assignments` | required | v1/v2 family assignment TSV |
| `--force` | off | Replace an existing output |

## fiberhmm-posteriors

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | required | Input BAM file |
| `-m` / `--model` | — | FiberHMM model file. If omitted, uses the bundled model for --enzyme/--seq. |
| `-o` / `--output` | required | Output file (.tsv.gz for TSV, .h5/.hdf5 for HDF5) |
| `--format` | `auto` | Output format (default: auto-detect from extension) Choices: `auto`, `hdf5`, `tsv`. |
| `--enzyme` | — | Auto-select a bundled enzyme model. Choices: `ddda`, `dddb`, `hia5`. |
| `--seq` | — | Hia5 platform; detected from the input when omitted, as in fiberhmm-call. Ignored for dddb/ddda. Choices: `pacbio`, `nanopore`. |
| `--edge-trim` / `-e` | `10` | Bases to trim from read edges (default: 10) |
| `--prob-threshold` | — | Min ML probability (0-255) for an MM/ML modification call. Default: chemistry preset, as in fiberhmm-call -- 248 for Hia5 Nanopore (--seq nanopore, given or detected), 128 otherwise. |
| `--keep-chimeras` | off | Do not drop DAF strand-swap chimeric reads |
| `--chimera-min-seg` | `5` | DAF chimera: min same-strand deamination events per segment (default: 5) |
| `--chimera-purity` | `0.8` | DAF chimera: min same-strand purity per segment (default: 0.8) |
| `--daf-snp-mask` | — | 0-based BED of reference positions whose conversions are ignored (e.g. the mask fiberhmm-call used) |
| `--daf-mask-runs` | — | Thin DAF targets in same-strand runs of &gt;= N original C (CT) or G (GA) bases (N=2: CC/GG and longer). Default: 2 with keep-one for DddA (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. Choices: `keep-one`, `drop`. |
| `--cores` / `-c` | `4` | Worker processes (0 = all CPUs; default: 4) |
| `--region-size` | `5000000` | Region size in bp for parallel processing (default: 5,000,000) |
| `--skip-scaffolds` | off | Skip scaffold/contig chromosomes |
| `--chroms` | — | Only process these chromosomes |
| `--streaming` | off | Read the BAM once, in file order, in a single process (no index needed; --cores is not used). Default: region-parallel over an indexed BAM. |
| `--batch-size` | `1000` | Fibers per HDF5 write batch (default: 1000) |
| `-v` / `--verbose` | off | Also print model/region details and a progress bar (the summary is always printed, to stderr) |

## fiberhmm-probs

| Flag | Default | Description |
|------|---------|-------------|
| `--accessible` / `-a` | required | BAM file(s) from accessible/naked DNA (dechromatinized, MTase-treated) |
| `--inaccessible` / `-u` | required | BAM file(s) from inaccessible/untreated samples (native chromatin) |
| `-o` / `--output` | required | Output directory; its name is the table prefix. Writes tables/&lt;name&gt;\_accessible\_&lt;base&gt;\_k&lt;k&gt;.tsv, tables/&lt;name&gt;\_inaccessible\_&lt;base&gt;\_k&lt;k&gt;.tsv and a combined tables/&lt;name&gt;\_&lt;base&gt;\_k&lt;k&gt;\_probs.tsv |
| `-k` / `--context-sizes` | `3 4 5 6` | Context size(s) to compute (bases on each side: 3=7mer, 6=13mer). Single value or list. Default: 3 4 5 6 (default: [3, 4, 5, 6]) |
| `--mode` | `pacbio-fiber` | Analysis mode: pacbio-fiber (PacBio), nanopore-fiber (Nanopore), daf (DAF-seq) (default: pacbio-fiber) Choices: `pacbio-fiber`, `nanopore-fiber`, `daf`, `gpc`, `cpg`. |
| `-n` / `--max-reads` | `100000` | Maximum reads to process per sample type (0 = all) (default: 100000) |
| `-s` / `--seed` | `42` | Random seed for sampling (default: 42) |
| `-q` / `--min-mapq` | `20` | Minimum mapping quality (default: 20) |
| `-p` / `--prob-threshold` | `128` | Minimum ML probability for modification call (0-255) (default: 128) |
| `--min-read-length` | `1000` | Minimum aligned read length (default: 1000) |
| `-e` / `--edge-trim` | `10` | Bases to exclude at read edges (default: 10) |
| `--save-interval` | `10000` | Save intermediate results every N reads (default: 10000) |
| `--stats` | off | Generate summary statistics and QC plots |
| `--verbose` / `-v` | off | Show detailed filter statistics per BAM file |

## fiberhmm-train

| Flag | Default | Description |
|------|---------|-------------|
| `-i` / `--input` | — | Input BAM file(s) for training (not required with --base-model) |
| `-p` / `--probs` | required | Accessible and inaccessible probability files (.tsv or .probs.pkl) |
| `--base-model` | — | Use transitions from existing model with new emissions (skip training) |
| `-o` / `--outdir` | required | Output directory |
| `--mode` | `pacbio-fiber` | Analysis mode: pacbio-fiber (PacBio), nanopore-fiber (Nanopore), daf (DAF-seq) (default: pacbio-fiber) Choices: `pacbio-fiber`, `nanopore-fiber`, `daf`, `gpc`, `cpg`. |
| `-k` / `--context-size` | `3` | Context size (bases on each side): 3=7mer, 5=11mer, etc. (default: 3) |
| `-c` / `--iterations` | `10` | Training iterations (random initializations) (default: 10) |
| `-r` / `--read-count` | `500` | Total reads to sample for training (default: 500) |
| `-s` / `--seed` | `42` | Random seed (default: 42) |
| `-e` / `--edge-trim` | `10` | Edge masking (default: 10) |
| `-q` / `--min-mapq` | `20` | Min mapping quality (default: 20) |
| `--prob-threshold` | `125` | Min ML probability (default matches ft-extract) (default: 125) |
| `--min-read-length` | `1000` | Min aligned length (default: 1000) |
| `-a` / `--prob-adjust` | `1.0` | Accessible probability adjustment factor (default: 1.0) |
| `--use-hmmlearn` | off | Use hmmlearn instead of native implementation (for legacy compatibility) |
| `--stats` | off | Generate training statistics and example plots |
| `--n-examples` | `5` | Number of example reads to plot (with --stats) (default: 5) |
| `--daf-mask-runs` | — | Thin DAF targets in same-strand runs of &gt;= N original C (CT) or G (GA) bases (N=2: CC/GG and longer). Default: 2 with keep-one for DddA (duplex-validated), off otherwise; 0 disables. |
| `--daf-run-policy` | `keep-one` | With --daf-mask-runs: keep each run's 5'-most target (default) or drop the run. (default: keep-one) Choices: `keep-one`, `drop`. |

## fiberhmm-utils

### fiberhmm-utils convert

| Flag | Default | Description |
|------|---------|-------------|
| `input` | required | Input model file (.pickle, .pkl, or .npz) |
| `output` | required | Output JSON file |

### fiberhmm-utils inspect

| Flag | Default | Description |
|------|---------|-------------|
| `model` | required | Model file to inspect (.json, .npz, .pickle) |
| `--full` | off | Print full emission probability table |

### fiberhmm-utils transfer

| Flag | Default | Description |
|------|---------|-------------|
| `--target` / `-t` | required | Target BAM file (e.g., DAF-seq) |
| `-o` / `--output` | required | Output directory |
| `--mode` | `daf` | Analysis mode for target data (default: daf) Choices: `pacbio-fiber`, `nanopore-fiber`, `daf`, `gpc`, `cpg`. |
| `--reference-bam` / `-rb` | — | Reference BAM with footprint tags (ns/nl) |
| `--accessibility-priors` / `-ap` | — | Pre-computed P(accessible\|context) TSV |
| `-k` / `--context-sizes` | `3 4 5 6` | Context size(s) (default: [3, 4, 5, 6]) |
| `-n` / `--max-reads` | `100000` | Max reads to process (0 = all) (default: 100000) |
| `-q` / `--min-mapq` | `20` | Min mapping quality (default: 20) |
| `-p` / `--prob-threshold` | `128` | Min ML probability for modification call (default: 128) |
| `--min-read-length` | `1000` | Min aligned read length (default: 1000) |
| `-e` / `--edge-trim` | `10` | Bases to exclude at read edges (default: 10) |
| `--min-observations` | `100` | Min observations per context for regression (default: 100) |
| `--stats` | off | Generate diagnostic plots |

### fiberhmm-utils adjust

| Flag | Default | Description |
|------|---------|-------------|
| `model` | required | Input model file (.json) |
| `--state` | required | Which state(s) to adjust Choices: `accessible`, `inaccessible`, `both`. |
| `--scale` | required | Multiplier for emission probabilities |
| `-o` / `--output` | required | Output model file (.json) |

### fiberhmm-utils ma-types

| Flag | Default | Description |
|------|---------|-------------|
| `bam` | required | BAM to update in place |
| `--types` | — | Logical MA name(s), comma- or space-separated (no strand/quality suffixes) |
| `--scan` | off | Exhaustively scan every alignment and discover non-empty MA types |
| `--io-threads` | `4` | BAM compression and index threads (default: 4) |

### fiberhmm-utils fix-bigbed

| Flag | Default | Description |
|------|---------|-------------|
| `inputs` | required | Input bigBed file(s) |
| `--sample-name` | — | Explicit sample name to embed (sanitized to a dot/space-free token). Default: derived per file from the filename (stem minus the \_&lt;layer&gt; suffix). |
| `--in-place` | off | Overwrite the input bigBed(s) in place. |
| `-o` / `--output` | — | Output path (single input only). Default: write &lt;name&gt;.fixed.bb alongside each input. |

<!-- END GENERATED CLI REFERENCE -->
