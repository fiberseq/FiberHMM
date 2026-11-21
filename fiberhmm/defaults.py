"""Canonical call defaults shared by every FiberHMM tool.

Production calling (``fiberhmm-call``) defines these values. Tools that call
footprints, re-derive modification observations from a BAM, or read calls back
use them unless they document a difference. Tools take their defaults from this
module instead of repeating literals, and ``tests/test_defaults_parity.py``
checks every console script's argument parser, the run-time resolvers, the
engine function signatures and the parameter dataclasses against it. A tool that
differs on purpose is listed in :data:`ALLOWED_OVERRIDES` with its value and the
reason; any other difference fails the test.

The values are listed in the docs (Reference > Shared defaults).

This module imports nothing from the package, so any module can use it.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Read filters
# ---------------------------------------------------------------------------

#: Minimum alignment MAPQ (0: every mapped read is called).
MIN_MAPQ = 0

#: Minimum read length in bp, as the HMM models it (the full SEQ incl. soft
#: clips; DAF with --daf-mask-unaligned: the aligned length).
MIN_READ_LENGTH = 1000

#: Alignment records that are called: primary and supplementary records;
#: secondary records pass through uncalled.
ALIGNMENTS = "primary-supplementary"

# ---------------------------------------------------------------------------
# Modification observations
# ---------------------------------------------------------------------------

#: Minimum MM/ML probability (0-255) for a modification call from MM/ML.
#: R/Y- and MD-encoded DAF input is binary and ignores it.
PROB_THRESHOLD = 128

#: Chemistry-specific overrides of :data:`PROB_THRESHOLD`, keyed by
#: (enzyme, platform). Hia5 Nanopore m6A is a strict hard-call assay
#: calibrated at 248 (bundled QC reference, strand-rescue preset).
PROB_THRESHOLD_OVERRIDES = {
    ("hia5", "nanopore"): 248,
}

#: The historical fallback threshold of tools that re-read MM/ML from an
#: already-called BAM (recall-tfs/-nucs, extract, qc, the consensus loader).
#: Not a canonical value: every use is listed in ALLOWED_OVERRIDES. The
#: 125/128 split is a documented inconsistency kept for compatibility.
REREAD_PROB_THRESHOLD = 125

#: Bases masked at each read end.
EDGE_TRIM = 10

#: DddA CpG mask when CpG-aware calling is on (DddA in DAF mode; off for every
#: other chemistry): every CpG observation is neutral except inside MA
#: ``ddda_ucg`` islands.
CPG_MASK_POLICY = "unmethylated-only"

#: DAF same-strand run thinning: (min_run_length, policy) per enzyme; 0 = off.
DAF_RUN_MASK = {"ddda": (2, "keep-one")}
DAF_RUN_POLICY = "keep-one"

# ---------------------------------------------------------------------------
# Footprint sizes and recall
# ---------------------------------------------------------------------------

#: Smallest footprint (bp) counted as a nucleosome.
NUC_MIN_SIZE = 85

#: Nucleosome cutoff of the accessibility views (fiberhmm-scan states, NFR and
#: peaks, fiberhmm-nfr, nucleosome patches): footprints of at least this length
#: count as nucleosomes, shorter ones as accessible (FiberBrowser's rule).
#: A documented difference from NUC_MIN_SIZE; these tools must agree with
#: each other.
ACCESSIBILITY_NUC_MIN_BP = 90

#: Smallest MSP (bp) written (0: all).
MSP_MIN_SIZE = 0

#: TF recall interval cost (nats; not an FDR cutoff) and emission uplift, the
#: same for every bundled enzyme.
MIN_LLR = 5.0
EMISSION_UPLIFT = 1.0

#: Minimum informative target positions per TF call.
MIN_OPPS = 3

#: Baseline nucleosomes shorter than this may be demoted to TF+.
UNIFY_THRESHOLD = 90

#: Nucleosome splitter thresholds.
SPLIT_MIN_LLR = 4.0
SPLIT_MIN_OPPS = 3

#: DddA radial nucleosome recall: largest derived-TF edge gap (bp).
DDDA_DERIVED_TF_MAX_EDGE_GAP = 12

# ---------------------------------------------------------------------------
# DAF chimeras, duplicates and inserts
# ---------------------------------------------------------------------------

#: DAF strand-swap chimeras: flag (0x200 + xc:Z:strand_swap, left uncalled).
CHIMERA_MODE = "flag"
CHIMERA_MIN_SEG = 5
CHIMERA_PURITY = 0.8

#: Deamination-fingerprint PCR duplicate detection.
DEDUP_MIN_JACCARD = 0.95
DEDUP_MIN_DEAM = 10
DEDUP_MAX_END_DIFF = 50

#: DAF insert consensus: carriers needed before an insert is scored.
DAF_INSERT_MIN_CARRIERS = 20


#: Documented differences from the canonical values:
#: ``(tool, setting) -> (value, reason)`` or ``(value, reason, "pending")``.
#: ``tool`` is a console script (``fiberhmm-train``) or a dotted Python object
#: (``fiberhmm.inference.parallel.process_bam_for_footprints``); ``setting``
#: is the argparse dest, parameter or field name. The value is pinned: a
#: drifting override fails the parity test as well. ``pending`` marks a
#: difference another branch fixes; the test then accepts either value.
ALLOWED_OVERRIDES = {
    # -- ML threshold: the documented 125/128 compatibility split ----------
    ("fiberhmm-recall-tfs", "prob_threshold.fallback"): (REREAD_PROB_THRESHOLD,
        "re-reads MM/ML of a called BAM at the historical 125 (Hia5 Nanopore 248); "
        "documented inconsistency kept for compatibility (Chemistries page)"),
    ("fiberhmm-recall-nucs", "prob_threshold.fallback"): (REREAD_PROB_THRESHOLD,
        "same resolver as fiberhmm-recall-tfs"),
    ("fiberhmm-extract", "prob_threshold.fallback"): (REREAD_PROB_THRESHOLD,
        "documented 125/128 compatibility split (Chemistries page)"),
    ("fiberhmm-qc", "prob_threshold.fallback"): (REREAD_PROB_THRESHOLD,
        "QC rate references are calibrated at 125 (Hia5 Nanopore 248)"),
    ("fiberhmm-consensus", "prob_threshold.hia5-pacbio"): (REREAD_PROB_THRESHOLD,
        "consensus loader keeps the recall threshold; documented on the Chemistries page"),
    ("fiberhmm-train", "prob_threshold"): (REREAD_PROB_THRESHOLD,
        "matches fibertools extract for training data; documented"),
    ("fiberhmm-footprint", "ml"): (REREAD_PROB_THRESHOLD,
        "ft footprint analogue: require-unmodified m6A at fibertools' 125"),
    ("fiberhmm-probs", "prob_threshold"): (PROB_THRESHOLD,
        "fixed 128: no chemistry option (training tool); documented"),
    ("fiberhmm-utils", "transfer.prob_threshold"): (PROB_THRESHOLD,
        "fixed 128: no chemistry option (training tool); documented"),
    ("fiberhmm-dedup", "prob_threshold"): (PROB_THRESHOLD,
        "DAF-only tool: the DAF chemistry default is 128"),
    ("fiberhmm-pair", "prob_threshold"): (PROB_THRESHOLD,
        "DAF-only tool: the DAF chemistry default is 128"),
    ("fiberhmm-merge", "prob_threshold"): (PROB_THRESHOLD,
        "DAF-only tool: the DAF chemistry default is 128"),
    # -- nucleosome decoding: the call's own trim unless one is given ------
    ("fiberhmm.nuc.params.NucParams", "edge_trim"): (None,
        "unset = the recorded fiberhmm-call --edge-trim, else EDGE_TRIM "
        "(nuc.records.DecodeContext); any explicit value, 10 included, wins"),
    # -- fiberhmm-train: one context size by default ----------------------
    ("fiberhmm-train", "context_size"): (3,
        "trains one table at k=3, the bundled models' context size"),
    # -- fiberhmm-pipeline passes nothing, so fiberhmm-call's default applies
    ("fiberhmm-pipeline", "min_mapq"): (None,
        "unset: DEFAULT_MIN_MAPQ, or the value an earlier run recorded in OUTDIR"),
    ("fiberhmm-pipeline", "min_read_length"): (None,
        "unset: not passed, fiberhmm-call's default applies"),
    # -- fiberhmm-posteriors: the chimera thresholds follow the detector --
    ("fiberhmm-posteriors", "chimera_min_seg"): (None,
        "unset: CHIMERA_MIN_SEG under --chimera-detector standard (the default), "
        "10 under strict, as fiberhmm-call resolves it"),
    ("fiberhmm-posteriors", "chimera_purity"): (None,
        "unset: CHIMERA_PURITY under --chimera-detector standard (the default), "
        "0.9 under strict, as fiberhmm-call resolves it"),
    ("fiberhmm-pipeline", "chimeras"): (None,
        "unset: flag for DAF (passed to fiberhmm-call as --chimeras flag)"),
    # -- Python API defaults (every in-repo caller passes these) ----------
    ("fiberhmm.inference.parallel.process_bam_for_footprints", "prob_threshold"): (0,
        "API: callers pass the resolved threshold"),
    ("fiberhmm.inference.parallel.process_bam_for_footprints", "min_read_length"): (0,
        "API: callers pass the resolved length"),
    ("fiberhmm.inference.parallel.process_bam_for_footprints", "primary_only"): (False,
        "API: False is the older 'all' spelling (secondary records called); every in-repo caller passes the alignment set"),
    ("fiberhmm.inference.streaming_pipeline._process_bam_streaming_pipeline", "prob_threshold"): (0,
        "API: callers pass the resolved threshold"),
    ("fiberhmm.inference.streaming_pipeline._process_bam_streaming_pipeline", "min_read_length"): (0,
        "API: callers pass the resolved length"),
    ("fiberhmm.inference.streaming_pipeline._process_bam_streaming_pipeline", "primary_only"): (False,
        "API: False is the older 'all' spelling (secondary records called); every in-repo caller passes the alignment set"),
    ("fiberhmm.inference.region_pipeline._process_bam_region_parallel", "prob_threshold"): (0,
        "API: callers pass the resolved threshold"),
    ("fiberhmm.inference.region_pipeline._process_bam_region_parallel", "min_read_length"): (0,
        "API: callers pass the resolved length"),
    ("fiberhmm.inference.region_pipeline._process_bam_region_parallel", "primary_only"): (False,
        "API: False is the older 'all' spelling (secondary records called); every in-repo caller passes the alignment set"),
    ("fiberhmm.inference.streaming_pipeline._process_bam_streaming_pipeline_fused", "primary_only"): (False,
        "API: False is the older 'all' spelling (secondary records called); every in-repo caller passes the alignment set"),
    ("fiberhmm.inference.region_pipeline._process_bam_region_parallel_fused", "primary_only"): (False,
        "API: False is the older 'all' spelling (secondary records called); every in-repo caller passes the alignment set"),
    ("fiberhmm.inference.legacy_pipeline._process_bam_legacy_pipeline", "prob_threshold"): (0,
        "API: callers pass the resolved threshold"),
    ("fiberhmm.inference.legacy_pipeline._process_bam_legacy_pipeline", "min_read_length"): (0,
        "API: callers pass the resolved length"),
    ("fiberhmm.inference.legacy_pipeline._process_bam_legacy_pipeline", "primary_only"): (False,
        "API: False is the older 'all' spelling (secondary records called); every in-repo caller passes the alignment set"),
    ("fiberhmm.inference.tf_recaller.extract_modification_calls", "context_size"): (3,
        "API: the bundled models' context size; recall passes the model's k"),
    ("fiberhmm.nuc.recall.CallerSettings", "daf_mask_runs"): (0,
        "bare dataclass; CallerSettings.from_header resolves the chemistry default"),
    ("fiberhmm.pipeline.runner.PipelineConfig", "min_mapq"): (None,
        "unset: DEFAULT_MIN_MAPQ, or the value an earlier run recorded in OUTDIR"),
    ("fiberhmm.pipeline.runner.PipelineConfig", "min_read_length"): (None,
        "unset: not passed, fiberhmm-call's default applies"),
    ("fiberhmm.pipeline.runner.PipelineConfig", "chimeras"): (None,
        "unset: flag for DAF (passed to fiberhmm-call as --chimeras flag)"),
    ("fiberhmm.inference.engine.predict_footprints_and_msps", "msp_min_size"): (147,
        "API helper for nucleosome-scale MSPs; not on the call path"),
    ("fiberhmm.inference.read_filters.ReadFilterConfig", "min_read_length"): (0,
        "filter object: each pipeline passes the resolved length"),
    ("fiberhmm.inference.read_filters.ReadFilterConfig", "primary_only"): (False,
        "False is the older 'all' spelling; every pipeline passes the alignment set"),
    ("fiberhmm.inference.tf_recaller.extract_modification_calls", "prob_threshold"): (REREAD_PROB_THRESHOLD,
        "recall's historical fallback (RECALL_PROB_THRESHOLD)"),
    ("fiberhmm.cli.dedup.run_dedup", "prob_threshold"): (0,
        "API: every caller passes the resolved threshold"),
    # -- consensus duplicate collapse --------------------------------------
    ("fiberhmm.inference.consensus.parameters.InputOptions", "molecule_max_end_diff_bp"): (-1,
        "consensus duplicate collapse: the fiberhmm-dedup endpoint check is opt-in "
        "(-1 = off) so truncated copies of one template still collapse"),
}
