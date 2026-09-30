"""Outputs record the FiberHMM version that made them (read by fiberhmm.advisories)."""

from __future__ import annotations

import gzip
import json

import pysam
import pytest

import fiberhmm
from fiberhmm.io.bam_header import declared_chemistries
from test_dedup import A_SITES, _make_bam


def test_dedup_output_records_its_program(tmp_path):
    from fiberhmm.cli.dedup import run_dedup

    source = tmp_path / "in.bam"
    _make_bam(source, [(f"A{i}", A_SITES, False, 60) for i in range(3)])
    output = tmp_path / "out.bam"
    run_dedup(str(source), str(output), collapse=False)
    with pysam.AlignmentFile(str(output), "rb", check_sq=False) as bam:
        program = bam.header.to_dict()["PG"][-1]
    assert program["PN"] == "fiberhmm-dedup"
    assert program["VN"] == fiberhmm.__version__
    assert "grouping=deamination_flavour" in program["DS"]
    assert "mode=flag" in program["DS"]


def test_merge_header_records_program_and_recall_chemistry():
    from fiberhmm.cli.merge import _merge_output_header
    from fiberhmm.models import get_model_path
    from fiberhmm.identity import file_sha256

    base = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10}]}
    plain = _merge_output_header(
        base, recall=False, enzyme="ddda", prob_threshold=128, pairs_only=True,
        nuc_recall_policy="conservative", phase_nrl=196, cpg_mask_policy=None)
    assert plain.to_dict()["PG"][-1]["PN"] == "fiberhmm-merge"
    assert declared_chemistries(plain) == []  # no calls made, no declaration

    recalled = _merge_output_header(
        base, recall=True, enzyme="ddda", prob_threshold=128, pairs_only=False,
        nuc_recall_policy="conservative", phase_nrl=196,
        cpg_mask_policy="unmethylated-only")
    program = recalled.to_dict()["PG"][-1]
    assert program["VN"] == fiberhmm.__version__
    assert "recall=on" in program["DS"] and "cpg_mask=unmethylated-only" in program["DS"]
    (declaration,) = declared_chemistries(recalled)
    assert declaration["enzyme"] == "ddda" and declaration["mode"] == "daf"
    assert declaration["pg"] == "fiberhmm-merge"
    assert declaration["apply_sha256"] == file_sha256(get_model_path("ddda", tool="apply"))
    assert declaration["recall_sha256"] == file_sha256(get_model_path("ddda", tool="recall"))


def test_posteriors_tsv_metadata_records_version(tmp_path):
    from fiberhmm.posteriors.tsv_backend import PosteriorsTSVWriter

    path = tmp_path / "p.tsv.gz"
    writer = PosteriorsTSVWriter(str(path), mode="daf", context_size=3,
                                 edge_trim=0, source_bam="x.bam")
    writer.close()
    with gzip.open(path, "rt") as handle:
        first = handle.readline()
    assert first.startswith("#metadata:")
    assert json.loads(first[len("#metadata:"):])["fiberhmm_version"] == fiberhmm.__version__


def test_qc_json_records_version(tmp_path):
    pytest.importorskip("matplotlib")
    from fiberhmm.qc.core import run_qc
    from test_qc import _write_unindexed_iupac_bam

    bam_path = tmp_path / "input.bam"
    _write_unindexed_iupac_bam(bam_path, n_reads=20)
    run_qc(str(bam_path), output_prefix=str(tmp_path / "s"), mode="daf",
           enzyme="dddb", reference_profile="none", sample_reads=5, stream=None)
    payload = json.loads((tmp_path / "s.qc.json").read_text())
    assert payload["fiberhmm_version"] == fiberhmm.__version__


def test_merge_recall_keeps_the_declared_platform_of_the_input():
    """A DddA-on-Nanopore input must not make the joint recall conflict (exit 2)."""
    from fiberhmm.cli.merge import _merge_output_header
    from fiberhmm.cli.provenance import ChemistryConflictError

    def header(declaration):
        return {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 10}],
                "CO": ["FIBERHMM-CHEMISTRY:v1:" + declaration]}

    kwargs = dict(recall=True, enzyme="ddda", prob_threshold=128, pairs_only=False,
                  nuc_recall_policy="conservative", phase_nrl=196,
                  cpg_mask_policy="unmethylated-only")
    ont = _merge_output_header(
        header("assay=daf;enzyme=ddda;platform=nanopore;mode=daf"), **kwargs)
    assert {d["platform"] for d in declared_chemistries(ont)} == {"nanopore"}
    pacbio = _merge_output_header(
        header("assay=daf;enzyme=ddda;platform=pacbio;mode=daf"), **kwargs)
    assert {d["platform"] for d in declared_chemistries(pacbio)} == {"pacbio"}
    # Re-calling a DddB-declared input with the DddA tables is refused, as in
    # fiberhmm-call, instead of writing DddA calls under a DddB declaration.
    with pytest.raises(ChemistryConflictError):
        _merge_output_header(
            header("assay=daf;enzyme=dddb;platform=nanopore;mode=daf"), **kwargs)
