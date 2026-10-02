"""m5C CLI guards (audit M19, M20).

- ``fiberhmm-call-m5c -o - --tag-output -`` wrote the BED and the BAM to the
  same stdout (an invalid BAM, exit 0).
- The DddA chemistry preflight matched "hia5" anywhere in the @PG history
  (file paths, directory names, other tools) and rejected valid DddA BAMs.
"""
import pysam
import pytest

from fiberhmm.cli.tag_m5c import _declared_enzymes, _preflight_input
from fiberhmm.io.bam_header import append_chemistry


def _ddda_bam(path, programs=(), declare=None):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 1000}],
              "PG": [dict(p) for p in programs]}
    if not header["PG"]:
        header.pop("PG")
    header = pysam.AlignmentHeader.from_dict(header)
    if declare:
        header = append_chemistry(header, declare)
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        read = pysam.AlignedSegment(out.header)
        read.query_name = "r"
        read.reference_id = 0
        read.reference_start = 10
        read.mapping_quality = 60
        read.cigarstring = "40M"
        read.query_sequence = "ACGTYACGTA" * 4
        read.set_tag("MA", "40;nuc.:1-20;msp.:21-20", value_type="Z")
        out.write(read)
    return str(path)


ALIGNER_WITH_HIA5_PATH = {
    "ID": "minimap2", "PN": "minimap2", "VN": "2.28",
    "CL": "minimap2 -ax map-ont ref.fa /data/ddda_vs_hia5_runs/sample.fastq",
}
DDDA_CALL = {
    "ID": "fiberhmm-call", "PN": "fiberhmm-call", "VN": "3.0.0",
    "CL": "fiberhmm-call -i /data/hia5_comparison/x.bam -o y.bam --enzyme ddda",
    "DS": "mode=daf enzyme=ddda",
}


def test_paths_naming_another_enzyme_do_not_reject_a_ddda_bam(tmp_path):
    path = _ddda_bam(tmp_path / "ddda.bam", [ALIGNER_WITH_HIA5_PATH, dict(DDDA_CALL, PP="minimap2")])
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        assert _declared_enzymes(bam.header) == {"ddda"}
    _preflight_input(path)  # accepted


def test_declared_other_chemistry_is_still_rejected(tmp_path):
    declared = _ddda_bam(tmp_path / "hia5.bam", declare=dict(
        assay="fiber-seq", enzyme="hia5", platform="pacbio", mode="pacbio-fiber"))
    with pytest.raises(SystemExit, match="incompatible or mixed enzyme provenance \\(hia5\\)"):
        _preflight_input(declared)
    mixed = _ddda_bam(tmp_path / "mixed.bam", [
        DDDA_CALL, dict(DDDA_CALL, ID="fiberhmm-call.1", PP="fiberhmm-call",
                        CL="fiberhmm-call --enzyme=dddb", DS="mode=daf enzyme=dddb")])
    with pytest.raises(SystemExit, match="ddda, dddb"):
        _preflight_input(mixed)


def test_call_m5c_refuses_bed_and_bam_on_one_stdout(tmp_path, capsys):
    from fiberhmm.cli.call_m5c import main

    # Refused before any input is opened (the inputs do not exist).
    with pytest.raises(SystemExit, match="same stdout"):
        main(["-i", str(tmp_path / "in.bam"), "-r", str(tmp_path / "ref.fa"), "-o", "-",
              "--region", "chr1", "--enzyme", "ddda", "--tag-bam", str(tmp_path / "in.bam"),
              "--tag-output", "-", "--tag-mode", "locus"])
    assert capsys.readouterr().out == ""
