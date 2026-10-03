# From a Plasmidsaurus run to footprints in minutes

A quick DAF-seq experiment is one cheap Nanopore run: send the DddB- (or
DddA-) treated DNA to a whole-plasmid or amplicon sequencing service such as
Plasmidsaurus, get a few thousand reads back, and look at the footprints the
same day. `fiberhmm-pipeline` does every step in one command:

```bash
fiberhmm-pipeline reads.fastq.gz --reference construct.dna --enzyme dddb -o my_run/
```

It aligns the reads, calls nucleosomes, accessible patches (MSPs) and TF
footprints, runs QC and tells you how to open the result in FiberBrowser.

## What you need

1. FiberHMM (see [Installation](installation.md)):

    ```bash
    pip install "fiberhmm[plots]"           # [plots] adds the QC report figures
    ```

2. minimap2, the read aligner, either as a program or as the `mappy` Python
   module:

    ```bash
    brew install minimap2                   # macOS (Homebrew)
    conda install -c bioconda minimap2      # conda / mamba, any platform
    pip install mappy                       # or: the Python module
    ```

3. Your reads: the FASTQ file(s) from the sequencing service (`.fastq`,
   `.fastq.gz`, or a folder of them; all files you give form one sample).

4. The reference:
    - for a **plasmid**, its map as you have it: a SnapGene `.dna`, a GenBank
      `.gb`/`.gbk` or an EMBL file (a FASTA works too);
    - for **amplicons of a genome**, the genome FASTA (for example
      `dm6.fa`).

## Run it

```bash
# A plasmid (the map's own topology says it is circular)
fiberhmm-pipeline QR4H6G_1.fastq --reference N1.dna --enzyme dddb -o n1_run/

# Amplicons on the fly genome
fiberhmm-pipeline fastq/ --reference dm6.fa --enzyme dddb -o sna_run/ -c 4
```

`--enzyme` is `dddb` or `ddda`; `-c` sets the CPU cores (default 4). The
output directory is created. The command prints each step as it goes:

```text
[10:31:02] reference: N1.dna -> N1.fa, 1 contig (circular: N1)
[10:31:02] index: built ~/.fiberhmm/minimap2_index/02a6527923193fe83549604d.mmi (minimap2 2.30-r1287, 0.0 s)
[10:31:02] align: minimap2 -a -x map-ont --MD -Y -t 4 -R '@RG\tID:QR4H6G_1\tSM:QR4H6G_1\tPL:ONT' ...
[10:31:05] align: 1892 reads: 1271 kept (primary, MAPQ>=20), 620 unmapped, 1 below MAPQ 20; 1159 joined across a circular origin; 1269 hard-clipped
[10:31:25] call: 1250 of 1271 reads called, 386 duplicate-flagged
[10:31:40] qc: QC FAIL (60/100)

Outputs:
  called bam       n1_run/QR4H6G_1.fiberhmm.bam
  aligned bam      n1_run/QR4H6G_1.aligned.bam
  reference fasta  n1_run/N1.fa
  plasmid map      n1_run/N1.dna
  qc report        n1_run/qc/QR4H6G_1.fiberhmm.qc.pdf

Open in FiberBrowser:
  fiberbrowser -f n1_run/N1.fa --dataset n1_run/QR4H6G_1.fiberhmm.bam:QR4H6G_1
```

How long it takes, on a laptop with `-c 4` (Apple M-series):

| Run | Reads | Wall time |
|---|---:|---:|
| Plasmid map (11.4 kb), 3,843 reads of which 1,072 plasmid | 3,843 | 17 s |
| Plasmid map (12.2 kb), 1,892 reads, 13 kb mean length, with `--tracks` | 1,892 | 44 s |
| 13.6 kb *sna* amplicon on dm6, first run (index built in 3 s) | 5,000 | 90 s |

Most of the time is calling (duplicate marking, the SNP screen and the
footprint models); alignment takes seconds. The minimap2 index of a
reference is built once and cached in `~/.fiberhmm/minimap2_index/`, so later
runs on the same genome start aligning at once.

## What you get

| File | What it is |
|---|---|
| `<sample>.fiberhmm.bam` (+ `.bai`) | the calls: nucleosomes, MSPs and TF footprints on every read ([BAM tags](../reference/bam-tags.md)) |
| `<sample>.aligned.bam` (+ `.bai`) | the alignment the calls were made on |
| `qc/<sample>.fiberhmm.qc.pdf`, `.png` | the QC report: deamination rate, nucleosome periodicity, footprint sizes, duplicates and SNPs ([Quality control](../workflows/qc.md)); written only with `fiberhmm[plots]` (matplotlib), otherwise the run lists the `.qc.json` as its QC report |
| `qc/<sample>.fiberhmm.qc.json`, `.qc.curves.json` | the same, machine-readable (always written) |
| `<contig>.fa` and a copy of the map | the reference the reads were aligned to (plasmid runs) |
| `tracks/` | bigBed/BED tracks, with `--tracks` ([Extracting tracks](../workflows/extracting.md)) |
| `outputs.json` | every path above and what to open in FiberBrowser |
| `logs/` | the log of each step |

The reference identity is recorded in the BAM header: each contig's MD5
(`@SQ M5`), `TP:circular` for circular contigs, and for a plasmid map a
`@CO FIBERHMM-REFERENCE:v1:` line with the map's file name and checksum
([Header declarations](../reference/headers.md#fiberhmm-reference)), so
FiberBrowser can pair the BAM with its map.

## Open it in FiberBrowser

FiberBrowser, the companion browser, is installed separately
(`pip install fiberbrowser`). Run the printed command, or open the called
BAM and the reference in the FiberBrowser window. For a plasmid, also load
the plasmid map (it adds the features); its contig name is the one the BAM uses, because
the pipeline names the contig the way FiberBrowser names the map: the file
name without its extension for SnapGene maps (`N1.dna` → `N1`), the `LOCUS`
name for GenBank. For amplicons on a genome, `outputs.json` also gives the
region with most reads (`"open": {"region": "chr2L:15474001-15489000"}`).

## What the pipeline does

1. **Reference.** A plasmid map is converted to a one-contig FASTA; a FASTA is
   used as it is. Contig MD5s are computed.
2. **Index.** minimap2 indexes the reference (cached).
3. **Align.** `minimap2 -a -x map-ont --MD -Y`, the lab's DAF-seq standard.
   Primary and supplementary alignments with MAPQ ≥ 20 are kept
   (`--min-mapq`; `--alignments primary` keeps the primary only). Unaligned
   read arms stay as soft clips on linear contigs (calling treats them as no
   evidence) and are hard-clipped on circular ones, where they are concatemer
   sequence (`--hard-clip` clips everywhere, `--keep-soft-clips` nowhere).
   On a circular plasmid, a read that runs
   through the position where the map starts is joined into one record, so
   the whole molecule is called ([Plasmids](../workflows/plasmids.md)).
   Sorting and indexing use pysam; samtools is not needed.
4. **Call.** `fiberhmm-call --enzyme dddb --seq nanopore` with its defaults:
   PCR duplicates are marked (flag `0x400`, reads kept), strand-swap chimeras
   are skipped, recurrent SNPs are screened and masked
   ([DAF-seq](../workflows/daf-seq.md)).
5. **QC.** `fiberhmm-qc` on the called BAM.
6. **Tracks** (`--tracks`): `fiberhmm-extract` for nucleosomes, MSPs, TFs and
   deaminations.

Re-running the same command skips the steps that are complete, so an
interrupted run (Ctrl-C, a closed laptop) continues where it stopped. A step
is skipped only when its outputs still have the size and SHA-256 it recorded;
a damaged or replaced output is made again. Running it again on the same
output directory with different reads, reference, settings or files named in
`--call-args` is refused before anything in the directory changes, naming
what changed; use a new `-o`, or `--redo STEP` to replace a step's result
(and everything after it, including an interrupted call's resumable state).
One run owns an output directory at a time; a second run started on it while
the first is running is refused. `outputs.json` is removed when a rerun
starts changing the directory and written again when it finishes.

## Common options

| Option | Effect |
|---|---|
| `--sample NAME` | name of the output files and read group (default: the first input's name); one plain file name: letters, digits, `_`, `-`, `.` (no path, no leading `.`) |
| `--region chr:start-end` | keep only reads overlapping the region (repeatable); the first is the one to open |
| `--min-read-length N` | shortest aligned read to call (default 1000; short amplicons may want 500) |
| `--dedup auto/on/off`, `--dedup-mode flag/collapse` | PCR-duplicate marking, and whether duplicates are kept (flagged) or collapsed |
| `--snp-screen auto/on/off`, `--snp-mask BED` | the recurrent-SNP screen, and your own sites to mask |
| `--no-chimera-filter` | call strand-swap chimeric reads |
| `--tracks` | also write bigBed/BED tracks |
| `--progress-json FILE` | machine-readable progress for a GUI |

Every option is in the [command-line reference](../reference/cli.md#fiberhmm-pipeline).

## Fiber-seq and already-aligned data

The pipeline also accepts a BAM that is already aligned to the reference
(with `MD` tags for DAF-seq): it skips alignment and calls it as it is. This
is the route for Hia5 Fiber-seq, which is usually aligned with `pbmm2` on a
cluster:

```bash
fiberhmm-pipeline sample.pbmm2.bam --reference hg38.fa --enzyme hia5 -o sample_run/ -c 8
```

For genome-scale inputs the call step uses `fiberhmm-call`'s resumable
region-parallel mode ([Long runs and resuming](../workflows/calling.md#long-runs-and-resuming)).
