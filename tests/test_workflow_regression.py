"""End-to-end workflow regression tests."""

from __future__ import annotations

from types import SimpleNamespace

import pysam
import pytest

from fiberhmm.cli import extract_tags, recall_tfs
from fiberhmm.inference.parallel import process_bam_for_footprints
from fiberhmm.models import get_model_path

WORKFLOW_READ_LENGTH = 600
PROTECTED_START = 250
PROTECTED_END = 290


WORKFLOW_CASES = [
    pytest.param(
        "hia5_pacbio",
        "m6a",
        "hia5",
        "pacbio",
        "pacbio-fiber",
        id="hia5-pacbio",
    ),
    pytest.param(
        "hia5_nanopore",
        "m6a",
        "hia5",
        "nanopore",
        "nanopore-fiber",
        id="hia5-nanopore",
    ),
    pytest.param(
        "dddb",
        "daf_iupac",
        "dddb",
        None,
        "daf",
        id="dddb-daf",
    ),
    pytest.param(
        "ddda",
        "daf_iupac",
        "ddda",
        None,
        "daf",
        id="ddda-daf",
    ),
]


def _make_mm_tag(sequence: str, mod_positions: list[int]) -> str:
    """Build a simple A+a MM tag for sorted query positions."""
    a_positions = [idx for idx, base in enumerate(sequence) if base == "A"]
    a_to_idx = {pos: idx for idx, pos in enumerate(a_positions)}
    skips = []
    previous_a_idx = -1
    for pos in mod_positions:
        current_a_idx = a_to_idx[pos]
        skips.append(current_a_idx - previous_a_idx - 1)
        previous_a_idx = current_a_idx
    return "A+a," + ",".join(str(skip) for skip in skips) + ";"


def _workflow_mod_positions() -> list[int]:
    protected_positions = set(range(PROTECTED_START, PROTECTED_END))
    return [
        pos
        for pos in range(20, WORKFLOW_READ_LENGTH - 20)
        if pos not in protected_positions
    ]


def _workflow_header() -> pysam.AlignmentHeader:
    return pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 10_000}],
    })


def _write_m6a_workflow_input(path, query_name: str) -> str:
    """Write a tiny Hia5-like BAM with one protected A-run."""
    sequence = "A" * WORKFLOW_READ_LENGTH
    mod_positions = _workflow_mod_positions()

    with pysam.AlignmentFile(str(path), "wb", header=_workflow_header()) as out:
        read = pysam.AlignedSegment()
        read.query_name = query_name
        read.query_sequence = sequence
        read.query_qualities = pysam.qualitystring_to_array(
            "I" * WORKFLOW_READ_LENGTH
        )
        read.flag = 0
        read.reference_id = 0
        read.reference_start = 1_000
        read.mapping_quality = 60
        read.cigar = [(0, WORKFLOW_READ_LENGTH)]
        read.set_tag("MM", _make_mm_tag(sequence, mod_positions))
        read.set_tag("ML", [240] * len(mod_positions))
        out.write(read)

    pysam.index(str(path))
    return str(path)


def _write_daf_iupac_workflow_input(path, query_name: str) -> str:
    """Write a tiny CT-strand DAF BAM with one protected C-run."""
    sequence = ["C"] * WORKFLOW_READ_LENGTH
    mod_positions = _workflow_mod_positions()
    for pos in mod_positions:
        sequence[pos] = "Y"

    with pysam.AlignmentFile(str(path), "wb", header=_workflow_header()) as out:
        read = pysam.AlignedSegment()
        read.query_name = query_name
        read.query_sequence = "".join(sequence)
        read.query_qualities = pysam.qualitystring_to_array(
            "I" * WORKFLOW_READ_LENGTH
        )
        read.flag = 0
        read.reference_id = 0
        read.reference_start = 1_000
        read.mapping_quality = 60
        read.cigar = [(0, WORKFLOW_READ_LENGTH)]
        read.set_tag("st", "CT", value_type="Z")
        out.write(read)

    pysam.index(str(path))
    return str(path)


def _write_workflow_input(path, input_kind: str, query_name: str) -> str:
    if input_kind == "m6a":
        return _write_m6a_workflow_input(path, query_name)
    if input_kind == "daf_iupac":
        return _write_daf_iupac_workflow_input(path, query_name)
    raise ValueError(f"unknown workflow input kind: {input_kind}")


def _read_only_record(path):
    with pysam.AlignmentFile(path, "rb", check_sq=False) as bam:
        reads = list(bam.fetch(until_eof=True))
    assert len(reads) == 1
    return reads[0]


def _extract_recalled_labels(recalled_bam: str, tmp_path):
    bed_paths = {
        "msp": str(tmp_path / "workflow_msp.bed"),
        "tf": str(tmp_path / "workflow_tf.bed"),
    }
    extract_tags._init_extract_worker({
        "extract_types": ["msp", "tf"],
        "min_mapq": 0,
        "prob_threshold": 0,
        "with_scores": True,
        "min_tq": 0,
        "block_scores": True,
    })
    return extract_tags._extract_region_worker(
        (("chr1", 0, 10_000), recalled_bam, bed_paths)
    )


@pytest.mark.parametrize(
    ("case_name", "input_kind", "enzyme", "seq", "mode"),
    WORKFLOW_CASES,
)
def test_hmm_apply_to_tf_recall_to_label_extraction_workflow_by_mode(
    case_name,
    input_kind,
    enzyme,
    seq,
    mode,
    tmp_path,
    monkeypatch,
):
    query_name = f"{case_name}_workflow_read"
    input_bam = _write_workflow_input(
        tmp_path / f"{case_name}_input.bam",
        input_kind,
        query_name,
    )
    applied_bam = str(tmp_path / f"{case_name}_applied.bam")
    recalled_bam = str(tmp_path / f"{case_name}_recalled.bam")
    apply_model_path = get_model_path(enzyme, tool="apply", seq=seq)
    recall_model_path = get_model_path(enzyme, tool="recall", seq=seq)

    assert process_bam_for_footprints(
        input_bam=input_bam,
        output_bam=applied_bam,
        model_or_path=apply_model_path,
        train_rids=set(),
        edge_trim=10,
        circular=False,
        mode=mode,
        context_size=3,
        msp_min_size=0,
        nuc_min_size=85,
        min_mapq=0,
        prob_threshold=0,
        min_read_length=0,
        with_scores=True,
        n_cores=1,
        streaming_pipeline=False,
        region_parallel=False,
        io_threads=1,
    ) == (1, 1)

    applied_read = _read_only_record(applied_bam)
    assert applied_read.has_tag("ns")
    assert applied_read.has_tag("nl")
    assert applied_read.has_tag("as")
    assert applied_read.has_tag("al")
    assert applied_read.has_tag("nq")
    assert applied_read.has_tag("aq")

    monkeypatch.setattr(
        recall_tfs,
        "parse_args",
        lambda: SimpleNamespace(
            in_bam=applied_bam,
            out_bam=recalled_bam,
            model=recall_model_path,
            enzyme=None,
            seq=None,
            downstream_compat=False,
            cores=1,
            min_llr=0.0,
            emission_uplift=None,
            unify_threshold=90,
            no_legacy_tags=False,
            min_opps=1,
            io_threads=1,
            mode=None,
            context_size=None,
            max_reads=0,
            chunk_size=10,
        ),
    )
    recall_tfs.main()
    pysam.index(recalled_bam)

    recalled_read = _read_only_record(recalled_bam)
    assert recalled_read.query_name == query_name
    assert recalled_read.has_tag("MA")
    assert recalled_read.has_tag("AQ")
    assert "tf+QQQ" in recalled_read.get_tag("MA")
    assert recalled_read.has_tag("as")
    assert recalled_read.has_tag("al")

    returned_paths, n_reads, n_features = _extract_recalled_labels(
        recalled_bam,
        tmp_path,
    )

    assert n_reads == 1
    assert n_features["msp"] >= 1
    assert n_features["tf"] >= 1

    msp_bed = tmp_path / "workflow_msp.bed"
    tf_bed = tmp_path / "workflow_tf.bed"
    assert returned_paths == {"msp": str(msp_bed), "tf": str(tf_bed)}
    assert msp_bed.read_text().strip()
    tf_lines = [line for line in tf_bed.read_text().splitlines() if line]
    assert tf_lines
    tf_cols = tf_lines[0].split("\t")
    assert len(tf_cols) == 15
    assert tf_cols[3] == query_name
    assert int(tf_cols[9]) == n_features["tf"]
    assert all(tf_cols[idx] for idx in (12, 13, 14))
