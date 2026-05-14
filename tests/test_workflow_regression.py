"""End-to-end workflow regression tests."""

from __future__ import annotations

from types import SimpleNamespace

import pysam

from fiberhmm.cli import extract_tags, recall_tfs
from fiberhmm.inference.parallel import process_bam_for_footprints


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


def _write_hia5_workflow_input(path) -> str:
    """Write a tiny Hia5-like BAM with one protected A-run."""
    read_length = 600
    sequence = "A" * read_length
    protected_positions = set(range(250, 290))
    mod_positions = [
        pos
        for pos in range(20, read_length - 20)
        if pos not in protected_positions
    ]

    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 10_000}],
    })

    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        read = pysam.AlignedSegment()
        read.query_name = "workflow_read"
        read.query_sequence = sequence
        read.query_qualities = pysam.qualitystring_to_array("I" * read_length)
        read.flag = 0
        read.reference_id = 0
        read.reference_start = 1_000
        read.mapping_quality = 60
        read.cigar = [(0, read_length)]
        read.set_tag("MM", _make_mm_tag(sequence, mod_positions))
        read.set_tag("ML", [240] * len(mod_positions))
        out.write(read)

    pysam.index(str(path))
    return str(path)


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


def test_hmm_apply_to_tf_recall_to_label_extraction_workflow(
    benchmark_model_path,
    tmp_path,
    monkeypatch,
):
    input_bam = _write_hia5_workflow_input(tmp_path / "workflow_input.bam")
    applied_bam = str(tmp_path / "workflow_applied.bam")
    recalled_bam = str(tmp_path / "workflow_recalled.bam")

    assert process_bam_for_footprints(
        input_bam=input_bam,
        output_bam=applied_bam,
        model_or_path=benchmark_model_path,
        train_rids=set(),
        edge_trim=10,
        circular=False,
        mode="pacbio-fiber",
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
            model=benchmark_model_path,
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
    assert int(tf_cols[9]) == n_features["tf"]
    assert all(tf_cols[idx] for idx in (12, 13, 14))
