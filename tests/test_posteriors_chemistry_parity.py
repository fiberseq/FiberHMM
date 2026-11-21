"""fiberhmm-posteriors resolves chemistry defaults as fiberhmm-call does.

A custom ``--model`` on a BAM that declares its chemistry inherits that
chemistry's defaults (DddA: CpG mask and keep-one run thinning), and the
Python API resolves the ML threshold like the command line (Hia5 Nanopore
248).
"""
from __future__ import annotations

import sys

import pysam

from fiberhmm.cli import export_posteriors as ep


def _declared_bam(path, enzyme, platform, mode):
    from fiberhmm.io.bam_header import append_chemistry

    header = append_chemistry(
        pysam.AlignmentHeader.from_dict({
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 10_000}]}),
        {"assay": "fiber-seq", "enzyme": enzyme, "platform": platform, "mode": mode},
    )
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        r = pysam.AlignedSegment(header)
        r.query_name = "r1"
        r.query_sequence = "ACGT" * 50
        r.flag = 0
        r.reference_id = 0
        r.reference_start = 100
        r.mapping_quality = 60
        r.cigarstring = "200M"
        out.write(r)
    pysam.index(str(path))
    return path


def _run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["fiberhmm-posteriors", *map(str, argv)])
    try:
        ep.main()
    except SystemExit as exit:
        return exit.code or 0
    return 0


def test_custom_model_on_declared_ddda_bam_gets_the_ddda_masks(tmp_path, monkeypatch):
    from fiberhmm.models import get_model_path

    bam = _declared_bam(tmp_path / "ddda.bam", "ddda", "pacbio", "daf")
    captured = {}
    monkeypatch.setattr(ep, "export_posteriors", lambda **kw: captured.update(kw))
    custom = get_model_path("ddda", tool="apply", seq="pacbio")
    assert _run_cli(monkeypatch, "-i", bam, "-m", custom, "-o", tmp_path / "p.tsv.gz") == 0
    extraction = captured["extraction"]
    assert extraction["cpg_mask_policy"] == "unmethylated-only"
    assert tuple(extraction["daf_run_mask"]) == (2, "keep-one")


def test_python_api_threshold_follows_the_declared_chemistry(tmp_path, monkeypatch):
    from fiberhmm.models import get_model_path

    bam = _declared_bam(tmp_path / "ont.bam", "hia5", "nanopore", "nanopore-fiber")
    captured = {}
    monkeypatch.setattr(ep, "export_posteriors", lambda **kw: captured.update(kw))
    model = get_model_path("hia5", tool="apply", seq="nanopore")
    ep.export_posteriors_from_bam(str(bam), model, str(tmp_path / "p.tsv.gz"))
    assert captured["prob_threshold"] == 248
    ep.export_posteriors_from_bam(str(bam), model, str(tmp_path / "p.tsv.gz"),
                                  enzyme="hia5", seq="nanopore")
    assert captured["prob_threshold"] == 248
    ep.export_posteriors_from_bam(str(bam), model, str(tmp_path / "p.tsv.gz"),
                                  prob_threshold=90)
    assert captured["prob_threshold"] == 90
