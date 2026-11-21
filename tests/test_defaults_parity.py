"""Every tool's effective defaults agree with fiberhmm.defaults.

``fiberhmm/defaults.py`` holds the canonical call defaults (production
``fiberhmm-call``). This test reads each tool's effective defaults at four
levels and fails on any difference that is not listed, with its value and a
reason, in ``fiberhmm.defaults.ALLOWED_OVERRIDES``:

1. the argparse parser of every console script in pyproject.toml (captured by
   running the tool's ``main()`` up to ``parse_args``);
2. the run-time resolvers that turn an unset option into a value per
   chemistry (ML threshold, DddA CpG mask, DAF run mask, recall presets);
3. the defaults of engine and API function signatures;
4. the defaults of parameter dataclasses (read filters, consensus input,
   nucleosome decoding and replay settings).

An override whose tool/setting is never checked, or whose value now equals the
canonical one, is stale and fails too (except entries marked ``pending``).
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

import pysam
import pytest

from fiberhmm import defaults as D

REPO = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Canonical value per setting name. ``None`` in the second slot means an
# unset (None) default is also accepted because a resolver chooses the value
# at run time; those resolvers are checked in part 2.
# ---------------------------------------------------------------------------
CANON = {
    "min_mapq": (D.MIN_MAPQ, False),
    "qc_min_mapq": (D.MIN_MAPQ, False),
    "minimum_mapq": (D.MIN_MAPQ, False),
    "min_read_length": (D.MIN_READ_LENGTH, False),
    "alignments": (D.ALIGNMENTS, False),
    "primary_only": (None, True),          # None = DEFAULT_ALIGNMENTS
    "prob_threshold": (None, True),        # chemistry-resolved (part 2)
    "edge_trim": (D.EDGE_TRIM, False),
    "context_size": (None, True),          # from the model
    "cpg_mask_policy": (D.CPG_MASK_POLICY, True),
    "use_m5c": (None, True),
    "daf_mask_runs": (None, True),         # chemistry default (part 2)
    "daf_run_policy": (D.DAF_RUN_POLICY, False),
    "daf_mask_unaligned": (True, False),
    "nuc_min_size": (D.NUC_MIN_SIZE, False),
    "msp_min_size": (D.MSP_MIN_SIZE, False),
    "min_llr": (D.MIN_LLR, True),
    "emission_uplift": (D.EMISSION_UPLIFT, True),
    "min_opps": (D.MIN_OPPS, False),
    "native_minimum_opportunities": (D.MIN_OPPS, False),
    "unify_threshold": (D.UNIFY_THRESHOLD, False),
    "split_min_llr": (D.SPLIT_MIN_LLR, False),
    "split_min_opps": (D.SPLIT_MIN_OPPS, False),
    "ddda_derived_tf_max_edge_gap": (D.DDDA_DERIVED_TF_MAX_EDGE_GAP, False),
    "chimeras": (D.CHIMERA_MODE, False),
    "chimera_min_seg": (D.CHIMERA_MIN_SEG, False),
    "chimera_purity": (D.CHIMERA_PURITY, False),
    "dedup_min_jaccard": (D.DEDUP_MIN_JACCARD, False),
    "min_jaccard": (D.DEDUP_MIN_JACCARD, False),
    "molecule_min_jaccard": (D.DEDUP_MIN_JACCARD, False),
    "dedup_min_deam": (D.DEDUP_MIN_DEAM, False),
    "min_deam": (D.DEDUP_MIN_DEAM, False),
    "molecule_min_deam": (D.DEDUP_MIN_DEAM, False),
    "dedup_max_end_diff": (D.DEDUP_MAX_END_DIFF, False),
    "max_end_diff": (D.DEDUP_MAX_END_DIFF, False),
    "daf_insert_min_carriers": (D.DAF_INSERT_MIN_CARRIERS, False),
    # consensus input options added by the consensus parity branches
    "ddda_cpg_mask_policy": (D.CPG_MASK_POLICY, False),
    "molecule_max_end_diff_bp": (D.DEDUP_MAX_END_DIFF, False),
    # DAF SNP screen thresholds (fiberhmm-call --daf-snp-*, fiberhmm-extract)
    "daf_snp_min_fraction": (D.DAF_SNP_MIN_FRACTION, False),
    "daf_snp_min_depth": (D.DAF_SNP_MIN_DEPTH, False),
    "daf_snp_min_alt_fibers": (D.DAF_SNP_MIN_ALT_FIBERS, False),
    # fiberhmm-extract --variants discover
    "variant_caller": (D.VARIANT_CALLER, False),
    "variants_chunk_size": (D.VARIANT_CHUNK_BP, False),
    "pileup_min_depth": (D.PILEUP_MIN_DEPTH, False),
    "pileup_min_alt_reads": (D.PILEUP_MIN_ALT_READS, False),
    "pileup_min_alt_fraction": (D.PILEUP_MIN_ALT_FRACTION, False),
    "pileup_min_baseq": (D.PILEUP_MIN_BASEQ, False),
    "pileup_min_mapq": (D.PILEUP_MIN_MAPQ, False),
    "pileup_strand_bias_p": (D.PILEUP_STRAND_BIAS_P, False),
    "pileup_homopolymer": (D.PILEUP_HOMOPOLYMER, False),
    "pileup_homopolymer_len": (D.PILEUP_HOMOPOLYMER_LEN, False),
    "pileup_max_indel_fraction": (D.PILEUP_MAX_INDEL_FRACTION, False),
    "pileup_mod_min_methylation": (D.PILEUP_MOD_MIN_METHYLATION, False),
    "pileup_mod_max_alt_fraction": (D.PILEUP_MOD_MAX_ALT_FRACTION, False),
    "pileup_hom_alt_fraction": (D.PILEUP_HOM_ALT_FRACTION, False),
}

#: Tool-specific option names that mean a canonical setting.
ALIASES = {
    ("fiberhmm-footprint", "ml"): ("prob_threshold", D.PROB_THRESHOLD),
    ("fiberhmm-extract", "min_tq"): ("min_llr", int(D.MIN_LLR * 10)),
}

#: An unset (None) default is resolved at run time. Such a tool must use one
#: of these shared resolvers (checked in its module source; their values are
#: checked per chemistry in part 2) or be listed in PASSTHROUGH.
RESOLVERS = {
    "prob_threshold": ("resolve_prob_threshold", "resolve_input_prob_threshold",
                       "default_prob_threshold", "default_qc_prob_threshold",
                       "_resolve_call_prob_threshold"),
    "cpg_mask_policy": ("resolve_cpg_mask_policy",),
    "use_m5c": ("resolve_cpg_mask_policy",),
    "daf_mask_runs": ("resolve_daf_run_mask", "apply_daf_run_mask_arguments",
                      "default_daf_run_mask"),
    "min_llr": ("ENZYME_PRESETS",),
    "emission_uplift": ("ENZYME_PRESETS",),
    "context_size": ("load_model_with_metadata", "model_context_size", "context_size_of"),
    "primary_only": ("resolve_alignments",),
}

#: Module(s) whose source holds a console script's resolution code.
TOOL_MODULES = {
    "fiberhmm-apply": ("fiberhmm.cli.apply", "fiberhmm.cli.common"),
    "fiberhmm-call": ("fiberhmm.cli.call",),
    "fiberhmm-pair": ("fiberhmm.cli.pair", "fiberhmm.crossstrand.recall"),
    "fiberhmm-merge": ("fiberhmm.cli.merge", "fiberhmm.crossstrand.recall"),
    "fiberhmm-posteriors": ("fiberhmm.cli.export_posteriors", "fiberhmm.cli.common"),
    "fiberhmm-recall-tfs": ("fiberhmm.cli.recall_tfs",),
    "fiberhmm-recall-nucs": ("fiberhmm.cli.recall_tfs",),
    "fiberhmm-extract": ("fiberhmm.cli.extract_tags",),
    "fiberhmm-qc": ("fiberhmm.cli.qc", "fiberhmm.qc.core"),
    "fiberhmm-consensus": ("fiberhmm.inference.consensus.bam",
                           "fiberhmm.inference.consensus.cli"),
    "fiberhmm-train": ("fiberhmm.cli.train", "fiberhmm.core.bam_reader"),
    "fiberhmm-strand-rescue": ("fiberhmm.cli.strand_rescue",
                               "fiberhmm.inference.strand_rescue"),
}

#: Unset defaults another component resolves; (tool, setting) -> reason.
PASSTHROUGH = {
    ("fiberhmm-pipeline", "prob_threshold"): "not passed: fiberhmm-call resolves it",
    ("fiberhmm-pipeline", "cpg_mask_policy"): "not passed: fiberhmm-call resolves it",
    ("fiberhmm-pipeline", "use_m5c"): "not passed: fiberhmm-call resolves it",
    ("fiberhmm.pipeline.runner.PipelineConfig", "prob_threshold"): "not passed: fiberhmm-call resolves it",
    ("fiberhmm.pipeline.runner.PipelineConfig", "cpg_mask_policy"): "not passed: fiberhmm-call resolves it",
    ("fiberhmm.pipeline.runner.PipelineConfig", "use_m5c"): "not passed: fiberhmm-call resolves it",
    ("fiberhmm.nuc.recall.CallerSettings", "cpg_mask_policy"): "from_header resolves it (part 2)",
    ("fiberhmm-strand-rescue", "prob_threshold"): (
        "the preset's threshold (strand_rescue.PRESETS; Hia5 Nanopore 248 checked in part 2); "
        "DddA/DddB presets fall back to extract_modification_calls' pinned 125"),
}
for _api in ("fiberhmm.inference.parallel.process_bam_for_footprints",
             "fiberhmm.inference.streaming_pipeline._process_bam_streaming_pipeline",
             "fiberhmm.inference.streaming_pipeline._process_bam_streaming_pipeline_fused",
             "fiberhmm.inference.region_pipeline._process_bam_region_parallel",
             "fiberhmm.inference.region_pipeline._process_bam_region_parallel_fused",
             "fiberhmm.inference.legacy_pipeline._process_bam_legacy_pipeline",
             "fiberhmm.cli.export_posteriors.extract_posteriors_from_read"):
    PASSTHROUGH[(_api, "cpg_mask_policy")] = "engine API: the calling command passes the resolved policy"

# The 3.2.1 tree's recall table: fixed per enzyme by the shared preset.
CHEMISTRIES = [("hia5", "pacbio"), ("hia5", "nanopore"), ("dddb", "nanopore"), ("ddda", "pacbio")]
MODES = {("hia5", "pacbio"): "pacbio-fiber", ("hia5", "nanopore"): "nanopore-fiber",
         ("dddb", "nanopore"): "daf", ("ddda", "pacbio"): "daf"}


def expected_threshold(enzyme, seq, fallback=D.PROB_THRESHOLD):
    return D.PROB_THRESHOLD_OVERRIDES.get((enzyme, seq), fallback)


class Report:
    """Collects differences; tracks which overrides were consulted."""

    def __init__(self, canon=None, overrides=None):
        self.canon = dict(CANON if canon is None else canon)
        self.overrides = dict(D.ALLOWED_OVERRIDES if overrides is None else overrides)
        self.problems = []
        self.visited = set()
        self.none_passes = set()

    def check(self, tool, setting, actual, expected, none_ok=False):
        key = (tool, setting)
        if key in self.overrides:
            self.visited.add(key)
            entry = self.overrides[key]
            pinned, pending = entry[0], len(entry) > 2 and entry[2] == "pending"
            if _same(actual, pinned) or (pending and _same(actual, expected)):
                if not pending and _same(actual, expected):
                    self.problems.append(f"stale override {key}: value equals the canonical {expected!r}")
                return
            self.problems.append(f"{tool} {setting} = {actual!r}; override pins {pinned!r}")
            return
        if actual is None and setting.split(".")[-1] in RESOLVERS:
            self.none_passes.add(key)  # resolved at run time: needs coverage
            return
        if _same(actual, expected):
            return
        self.problems.append(f"{tool} {setting} = {actual!r}; canonical {expected!r} "
                             "(fix it or add it to fiberhmm.defaults.ALLOWED_OVERRIDES)")

    def check_named(self, tool, name, actual):
        if (tool, name) in ALIASES:
            canon_name, expected = ALIASES[(tool, name)]
            self.check(tool, name, actual, expected)
            return
        if name in self.canon:
            expected, none_ok = self.canon[name]
            self.check(tool, name, actual, expected, none_ok)


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return a is not None and b is not None and abs(float(a) - float(b)) < 1e-12
        except (TypeError, ValueError):
            return False
    return a == b


# ---------------------------------------------------------------------------
# 1. argparse parsers of every console script
# ---------------------------------------------------------------------------

class _Captured(BaseException):
    def __init__(self, parser):
        super().__init__()
        self.parser = parser


def console_scripts():
    text = (REPO / "pyproject.toml").read_text()
    section = text.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    return dict(re.findall(r'^([\w-]+)\s*=\s*"([^"]+)"', section, re.M))


def _parser_defaults(parser, prefix=""):
    out = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                out.update(_parser_defaults(sub, f"{prefix}{name}."))
            continue
        if action.dest in ("help", "version"):
            continue
        out.setdefault(prefix + action.dest, action.default)  # first action sets it
    return out


def capture_parsers(monkeypatch):
    def capture(self, args=None, namespace=None):
        raise _Captured(self)

    monkeypatch.setenv("FIBERHMM_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_known_args", capture)
    parsers = {}
    for name, target in console_scripts().items():
        module, func = target.split(":")
        monkeypatch.setattr("sys.argv", [name])
        try:
            getattr(importlib.import_module(module), func)()
        except _Captured as captured:
            parsers[name] = _parser_defaults(captured.parser)
            continue
        raise AssertionError(f"{name}: main() returned without parsing arguments")
    return parsers


def check_parsers(report, parsers):
    for tool, options in sorted(parsers.items()):
        for dest, value in options.items():
            name = dest.split(".")[-1]
            if (tool, dest) in report.overrides and name in report.canon:
                expected, none_ok = report.canon[name]
                report.check(tool, dest, value, expected, none_ok)
            elif (tool, name) in ALIASES:
                report.check_named(tool, name, value)
            elif name in report.canon:
                expected, none_ok = report.canon[name]
                report.check(tool, dest, value, expected, none_ok)


# ---------------------------------------------------------------------------
# 2. run-time resolvers per chemistry
# ---------------------------------------------------------------------------

def _declared_header(enzyme, seq):
    from fiberhmm.io.bam_header import append_chemistry

    return append_chemistry(
        pysam.AlignmentHeader.from_dict({"HD": {"VN": "1.6"},
                                         "SQ": [{"SN": "chr1", "LN": 10_000}]}),
        {"assay": "fiber-seq", "enzyme": enzyme, "platform": seq,
         "mode": MODES[(enzyme, seq)]},
    )


def check_resolvers(report):
    from fiberhmm.cli import call, recall_tfs
    from fiberhmm.cli.common import resolve_input_prob_threshold
    from fiberhmm.core.bam_reader import resolve_daf_run_mask
    from fiberhmm.inference import tf_recaller
    from fiberhmm.inference.consensus import bam as consensus_bam
    from fiberhmm.inference.strand_rescue import PRESETS
    from fiberhmm.nuc.recall import CallerSettings
    from fiberhmm.nuc.records import DecodeContext
    from fiberhmm.qc.core import default_qc_prob_threshold

    def recall_fallback(tool):
        entry = report.overrides.get((tool, "prob_threshold.fallback"))
        return entry[0] if entry else D.PROB_THRESHOLD

    for enzyme, seq in CHEMISTRIES:
        chem = f"{enzyme}/{seq}"
        expected = expected_threshold(enzyme, seq)
        args = SimpleNamespace(prob_threshold=None, enzyme=enzyme, seq=seq)
        report.check("fiberhmm-call", f"prob_threshold[{chem}]",
                     call._resolve_call_prob_threshold(args), expected)
        report.check("fiberhmm-apply", f"prob_threshold[{chem}]",
                     resolve_input_prob_threshold(None, enzyme, seq, None), expected)
        # Tools with a documented 125 fallback: the fallback is pinned once
        # (prob_threshold.fallback) and Hia5 Nanopore must still be 248.
        for tool, value in (
            ("fiberhmm-recall-tfs", recall_tfs._resolve_recall_prob_threshold(args)),
            ("fiberhmm-extract", _extract_threshold(enzyme, seq)),
            ("fiberhmm-qc", default_qc_prob_threshold(MODES[(enzyme, seq)], enzyme)),
        ):
            if (enzyme, seq) in D.PROB_THRESHOLD_OVERRIDES:
                report.check(tool, f"prob_threshold[{chem}]", value, expected)
            else:
                report.check(tool, "prob_threshold.fallback", value, expected)
        header = _declared_header(enzyme, seq)
        report.check("fiberhmm.nuc DecodeContext", f"prob_threshold[{chem}]",
                     DecodeContext(header).mark_threshold, expected)
        report.check("fiberhmm.nuc CallerSettings", f"prob_threshold[{chem}]",
                     CallerSettings.from_header(header).prob_threshold, expected)
        # DddA CpG mask (call, apply, recall, posteriors, pair/merge share it)
        mode = MODES[(enzyme, seq)]
        report.check("resolve_cpg_mask_policy", f"default[{chem}]",
                     tf_recaller.resolve_cpg_mask_policy(None, None, enzyme, mode),
                     D.CPG_MASK_POLICY if enzyme == "ddda" else None)
        report.check("resolve_daf_run_mask", f"default[{enzyme}]",
                     tuple(resolve_daf_run_mask(None, D.DAF_RUN_POLICY, enzyme)),
                     D.DAF_RUN_MASK.get(enzyme, (0, D.DAF_RUN_POLICY)))
        preset = tf_recaller.ENZYME_PRESETS[enzyme]
        report.check("ENZYME_PRESETS", f"min_llr[{enzyme}]", preset["min_llr"], D.MIN_LLR)
        report.check("ENZYME_PRESETS", f"emission_uplift[{enzyme}]",
                     preset["emission_uplift"], D.EMISSION_UPLIFT)

    # recall-nucs runs the same resolver; consensus chemistry presets
    report.check("fiberhmm-recall-nucs", "prob_threshold.fallback",
                 tf_recaller.RECALL_PROB_THRESHOLD, D.PROB_THRESHOLD)
    stub = dict(PRESETS=PRESETS, resolve_resource=lambda p: p,
                load_model=lambda p: (None, 3, None), build_llr=lambda m: (None, None))
    for chemistry, (enzyme, seq) in (("hia5-pacbio", ("hia5", "pacbio")),
                                     ("hia5-nanopore", ("hia5", "nanopore"))):
        preset = consensus_bam._chemistry_runtime(stub, chemistry)[0]
        report.check("fiberhmm-consensus", f"prob_threshold.{chemistry}",
                     preset["prob_threshold"], expected_threshold(enzyme, seq))

    # shared constants that other modules re-export
    from fiberhmm import models
    from fiberhmm.inference import read_filters
    from fiberhmm.pipeline import runner

    report.check("fiberhmm.models", "DEFAULT_PROB_THRESHOLD",
                 models.DEFAULT_PROB_THRESHOLD, D.PROB_THRESHOLD)
    report.check("fiberhmm.models", "PROB_THRESHOLD_OVERRIDES",
                 dict(models.PROB_THRESHOLD_OVERRIDES), dict(D.PROB_THRESHOLD_OVERRIDES))
    report.check("read_filters", "DEFAULT_ALIGNMENTS", read_filters.DEFAULT_ALIGNMENTS, D.ALIGNMENTS)
    report.check("fiberhmm-pipeline", "DEFAULT_MIN_MAPQ", runner.DEFAULT_MIN_MAPQ, D.MIN_MAPQ)
    report.check("strand-rescue PRESETS", "prob_threshold[hia5-nanopore]",
                 PRESETS["hia5-nanopore"]["prob_threshold"], expected_threshold("hia5", "nanopore"))


def _extract_threshold(enzyme, seq):
    """fiberhmm-extract's own resolver on a BAM declaring the chemistry."""
    import tempfile

    from fiberhmm.cli import extract_tags

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "declared.bam")
        with pysam.AlignmentFile(path, "wb", header=_declared_header(enzyme, seq)):
            pass
        return extract_tags._default_extract_prob_threshold(path)


def check_scan_constants(report):
    """fiberhmm-scan's read policy and the accessibility nucleosome cutoff."""
    from fiberhmm.inference.accessibility import gaps
    from fiberhmm.nuc.patches import PatchParams
    from fiberhmm.scan import peaks_core, peaks_reads, states
    from fiberhmm.scan.modules import nfr
    from fiberhmm.scan.reads import DEFAULT_READ_POLICY

    filters = DEFAULT_READ_POLICY["filters"]
    report.check("fiberhmm-scan", "read_policy.min_mapq", filters["min_mapq"], D.MIN_MAPQ)
    report.check("fiberhmm-scan", "read_policy.alignments",
                 (filters["secondary"], filters["supplementary"]),
                 ("exclude", "include") if D.ALIGNMENTS == "primary-supplementary" else None)
    for tool, value in (
        ("fiberhmm.scan.states.NUC_MIN_BP", states.NUC_MIN_BP),
        ("fiberhmm.scan.modules.nfr.NUC_MIN_BP", nfr.NUC_MIN_BP),
        ("fiberhmm.scan.peaks_reads.SHORT_TERMINAL_FOOTPRINT_BP", peaks_reads.SHORT_TERMINAL_FOOTPRINT_BP),
        ("fiberhmm.scan.peaks_core.SUBNUCLEOSOMAL_BP", peaks_core.SUBNUCLEOSOMAL_BP),
        ("fiberhmm.inference.accessibility.gaps.NUC_MIN", gaps.NUC_MIN),
        ("fiberhmm.nuc.patches.PatchParams.nuc_min_bp", PatchParams().nuc_min_bp),
    ):
        report.check(tool, "accessibility_nuc_min_bp", value, D.ACCESSIBILITY_NUC_MIN_BP)


# ---------------------------------------------------------------------------
# 3. function signatures, 4. dataclasses
# ---------------------------------------------------------------------------

SIGNATURES = [
    "fiberhmm.inference.parallel.process_bam_for_footprints",
    "fiberhmm.inference.streaming_pipeline._process_bam_streaming_pipeline",
    "fiberhmm.inference.streaming_pipeline._process_bam_streaming_pipeline_fused",
    "fiberhmm.inference.region_pipeline._process_bam_region_parallel",
    "fiberhmm.inference.region_pipeline._process_bam_region_parallel_fused",
    "fiberhmm.inference.legacy_pipeline._process_bam_legacy_pipeline",
    "fiberhmm.inference.engine.predict_footprints_and_msps",
    "fiberhmm.inference.fused_stages.build_fused_recall_result",
    "fiberhmm.inference.tf_recaller.extract_modification_calls",
    "fiberhmm.cli.export_posteriors.export_posteriors_from_bam",
    "fiberhmm.cli.export_posteriors.export_posteriors",
    "fiberhmm.cli.export_posteriors.extract_posteriors_from_read",
    "fiberhmm.cli.dedup.run_dedup",
    "fiberhmm.cli.common.add_filter_args",
]

DATACLASSES = [
    "fiberhmm.inference.read_filters.ReadFilterConfig",
    "fiberhmm.inference.consensus.parameters.InputOptions",
    "fiberhmm.nuc.params.NucParams",
    "fiberhmm.nuc.recall.CallerSettings",
    "fiberhmm.pipeline.runner.PipelineConfig",
]


def _resolve(dotted):
    module, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(module), name)


def check_signatures(report):
    for dotted in SIGNATURES:
        for name, param in inspect.signature(_resolve(dotted)).parameters.items():
            if param.default is inspect.Parameter.empty or name not in report.canon:
                continue
            expected, none_ok = report.canon[name]
            if name == "prob_threshold":  # a fixed API default: the base value
                expected, none_ok = D.PROB_THRESHOLD, True
            report.check(dotted, name, param.default, expected, none_ok)


def check_dataclasses(report):
    for dotted in DATACLASSES:
        for f in dataclasses.fields(_resolve(dotted)):
            if f.name not in report.canon or f.default is dataclasses.MISSING:
                continue
            expected, none_ok = report.canon[f.name]
            if f.name == "prob_threshold":
                expected, none_ok = D.PROB_THRESHOLD, True
            report.check(dotted, f.name, f.default, expected, none_ok)


#: Variant discovery / DAF SNP screen settings checked by explicit name: the
#: generic names (min_depth, min_mapq) mean different things per caller.
VARIANT_DATACLASS_FIELDS = {
    "fiberhmm.io.variant_discovery.PileupParams": {
        "min_depth": D.PILEUP_MIN_DEPTH, "min_alt_reads": D.PILEUP_MIN_ALT_READS,
        "min_alt_fraction": D.PILEUP_MIN_ALT_FRACTION, "min_baseq": D.PILEUP_MIN_BASEQ,
        "min_mapq": D.PILEUP_MIN_MAPQ, "strand_bias_p": D.PILEUP_STRAND_BIAS_P,
        "homopolymer": D.PILEUP_HOMOPOLYMER, "homopolymer_len": D.PILEUP_HOMOPOLYMER_LEN,
        "max_indel_fraction": D.PILEUP_MAX_INDEL_FRACTION,
        "mod_min_methylation": D.PILEUP_MOD_MIN_METHYLATION,
        "mod_max_alt_fraction": D.PILEUP_MOD_MAX_ALT_FRACTION,
        "mod_prob_threshold": D.REREAD_PROB_THRESHOLD,
        "hom_alt_fraction": D.PILEUP_HOM_ALT_FRACTION,
    },
    "fiberhmm.io.variant_discovery.DafScreenParams": {
        "min_fraction": D.DAF_SNP_MIN_FRACTION, "min_depth": D.DAF_SNP_MIN_DEPTH,
        "min_alt_fibers": D.DAF_SNP_MIN_ALT_FIBERS, "min_mapq": D.MIN_MAPQ,
    },
    "fiberhmm.io.variant_discovery.DiscoveryOptions": {
        "caller": D.VARIANT_CALLER, "chunk_bp": D.VARIANT_CHUNK_BP,
    },
}


def check_variant_discovery(report, parsers):
    from fiberhmm.daf.snps import call_opposite_conversion_snps
    for dotted, expected in VARIANT_DATACLASS_FIELDS.items():
        fields = {f.name: f for f in dataclasses.fields(_resolve(dotted))}
        for name, value in expected.items():
            report.check(dotted, name, fields[name].default, value)
    params = inspect.signature(call_opposite_conversion_snps).parameters
    daf = parsers.get("fiberhmm-daf-snps", {})
    for name, value in (("min_fraction", D.DAF_SNP_MIN_FRACTION),
                        ("min_depth", D.DAF_SNP_MIN_DEPTH),
                        ("min_alt_fibers", D.DAF_SNP_MIN_ALT_FIBERS)):
        report.check("fiberhmm.daf.snps.call_opposite_conversion_snps", name,
                     params[name].default, value)
        report.check("fiberhmm-daf-snps", name, daf.get(name), value)
    extract = parsers.get("fiberhmm-extract", {})
    report.check("fiberhmm-extract", "pileup_mod_filter", extract.get("pileup_mod_filter"), True)


def run_all(monkeypatch, report):
    parsers = capture_parsers(monkeypatch)
    check_parsers(report, parsers)
    check_variant_discovery(report, parsers)
    check_resolvers(report)
    check_signatures(report)
    check_dataclasses(report)
    check_scan_constants(report)
    return report


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_every_tool_default_matches_the_shared_value(monkeypatch):
    report = run_all(monkeypatch, Report())
    assert not report.problems, "\n".join(report.problems)


def _module_sources(tool):
    if tool in TOOL_MODULES:
        names = TOOL_MODULES[tool]
    else:  # a dotted Python object: its own module
        names = (tool.rpartition(".")[0],)
        if names[0].rpartition(".")[2][:1].isupper():
            names = (names[0].rpartition(".")[0],)
    return "\n".join(inspect.getsource(importlib.import_module(n)) for n in names)


def test_unset_defaults_are_resolved_by_a_shared_resolver(monkeypatch):
    report = run_all(monkeypatch, Report())
    problems = []
    for tool, setting in sorted(report.none_passes):
        name = setting.split(".")[-1]
        if (tool, setting) in PASSTHROUGH:
            continue
        if tool not in TOOL_MODULES and not tool.startswith("fiberhmm."):
            problems.append(f"{tool} {setting}: unset default with no module mapping")
            continue
        source = _module_sources(tool)
        if not any(resolver in source for resolver in RESOLVERS[name]):
            problems.append(f"{tool} {setting}: unset default, but none of "
                            f"{RESOLVERS[name]} is used (add PASSTHROUGH with a reason)")
    assert not problems, "\n".join(problems)


def test_no_stale_or_unknown_overrides(monkeypatch):
    report = run_all(monkeypatch, Report())
    unknown = [key for key, entry in D.ALLOWED_OVERRIDES.items()
               if key not in report.visited and not (len(entry) > 2 and entry[2] == "pending")]
    assert not unknown, f"overrides never checked (typo or removed tool): {unknown}"
    for key, entry in D.ALLOWED_OVERRIDES.items():
        assert isinstance(entry[1], str) and len(entry[1]) > 10, f"{key}: give a reason"


def test_the_guard_reports_a_diverging_default(monkeypatch):
    canon = dict(CANON)
    canon["edge_trim"] = (D.EDGE_TRIM + 1, False)
    report = run_all(monkeypatch, Report(canon=canon))
    flagged = {p.split()[0] for p in report.problems if " edge_trim " in p}
    assert {"fiberhmm-call", "fiberhmm-apply", "fiberhmm-posteriors"} <= flagged


def test_the_guard_reports_a_drifting_override(monkeypatch):
    overrides = dict(D.ALLOWED_OVERRIDES)
    overrides[("fiberhmm-train", "prob_threshold")] = (120, "drift test")
    report = run_all(monkeypatch, Report(overrides=overrides))
    assert any("fiberhmm-train prob_threshold = 125; override pins 120" in p
               for p in report.problems)
