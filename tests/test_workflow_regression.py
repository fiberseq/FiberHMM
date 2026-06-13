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
        "m6a_forward",
        None,
        "hia5",
        "pacbio",
        "pacbio-fiber",
        id="hia5-pacbio",
    ),
    pytest.param(
        "hia5_nanopore_forward",
        "m6a_forward",
        None,
        "hia5",
        "nanopore",
        "nanopore-fiber",
        id="hia5-nanopore-forward",
    ),
    pytest.param(
        "hia5_nanopore_reverse",
        "m6a_reverse",
        None,
        "hia5",
        "nanopore",
        "nanopore-fiber",
        id="hia5-nanopore-reverse",
    ),
]

for _enzyme in ("dddb", "ddda"):
    for _input_kind in ("daf_iupac", "daf_md", "daf_mmml"):
        for _strand in ("ct", "ga"):
            WORKFLOW_CASES.append(
                pytest.param(
                    f"{_enzyme}_{_input_kind}_{_strand}",
                    _input_kind,
                    _strand,
                    _enzyme,
                    None,
                    "daf",
                    id=f"{_enzyme}-{_input_kind}-{_strand}",
                )
            )


def _make_mm_tag(
    sequence: str,
    mod_positions: list[int],
    *,
    base: str = "A",
    mod_code: str = "a",
) -> str:
    """Build a simple MM tag for sorted query positions."""
    base_positions = [
        idx for idx, seq_base in enumerate(sequence) if seq_base == base
    ]
    base_to_idx = {pos: idx for idx, pos in enumerate(base_positions)}
    skips = []
    previous_base_idx = -1
    for pos in mod_positions:
        current_base_idx = base_to_idx[pos]
        skips.append(current_base_idx - previous_base_idx - 1)
        previous_base_idx = current_base_idx
    return f"{base}+{mod_code}," + ",".join(str(skip) for skip in skips) + ";"


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


def _md_for_mismatches(read_length: int, mismatch_positions: list[int], ref_base: str) -> str:
    parts = []
    previous = -1
    for pos in mismatch_positions:
        parts.append(str(pos - previous - 1))
        parts.append(ref_base)
        previous = pos
    parts.append(str(read_length - previous - 1))
    return "".join(parts)


def _write_m6a_workflow_input(path, query_name: str, *, is_reverse: bool) -> str:
    """Write a tiny Hia5-like BAM with one protected m6A target run."""
    if is_reverse:
        sequence = "T" * WORKFLOW_READ_LENGTH
        mm_walk_sequence = "A" * WORKFLOW_READ_LENGTH
        mod_positions = sorted(
            WORKFLOW_READ_LENGTH - 1 - pos for pos in _workflow_mod_positions()
        )
    else:
        sequence = "A" * WORKFLOW_READ_LENGTH
        mm_walk_sequence = sequence
        mod_positions = _workflow_mod_positions()

    with pysam.AlignmentFile(str(path), "wb", header=_workflow_header()) as out:
        read = pysam.AlignedSegment()
        read.query_name = query_name
        read.query_sequence = sequence
        read.query_qualities = pysam.qualitystring_to_array(
            "I" * WORKFLOW_READ_LENGTH
        )
        read.flag = 16 if is_reverse else 0
        read.reference_id = 0
        read.reference_start = 1_000
        read.mapping_quality = 60
        read.cigar = [(0, WORKFLOW_READ_LENGTH)]
        read.set_tag("MM", _make_mm_tag(mm_walk_sequence, mod_positions))
        read.set_tag("ML", [240] * len(mod_positions))
        out.write(read)

    pysam.index(str(path))
    return str(path)


def _daf_bases(strand: str, input_kind: str) -> tuple[str, str, str, str, str | None]:
    if strand == "ct":
        ref_base, deam_base, iupac_base, st_tag = "C", "T", "Y", "CT"
    elif strand == "ga":
        ref_base, deam_base, iupac_base, st_tag = "G", "A", "R", "GA"
    else:
        raise ValueError(f"unknown DAF strand: {strand}")

    observed_base = iupac_base if input_kind == "daf_iupac" else deam_base
    mm_base = deam_base
    return ref_base, deam_base, observed_base, mm_base, st_tag


def _write_daf_workflow_input(path, query_name: str, input_kind: str, strand: str) -> str:
    """Write a tiny DAF BAM using IUPAC, raw MD, or MM/ML encoding."""
    ref_base, _deam_base, observed_base, mm_base, st_tag = _daf_bases(
        strand, input_kind
    )
    sequence = [ref_base] * WORKFLOW_READ_LENGTH
    mod_positions = _workflow_mod_positions()
    for pos in mod_positions:
        sequence[pos] = observed_base
    sequence = "".join(sequence)

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
        if input_kind == "daf_iupac":
            read.set_tag("st", st_tag, value_type="Z")
        elif input_kind == "daf_md":
            read.set_tag(
                "MD",
                _md_for_mismatches(WORKFLOW_READ_LENGTH, mod_positions, ref_base),
                value_type="Z",
            )
        elif input_kind == "daf_mmml":
            read.set_tag(
                "MM",
                _make_mm_tag(
                    sequence,
                    mod_positions,
                    base=mm_base,
                    mod_code="u",
                ),
            )
            read.set_tag("ML", [240] * len(mod_positions))
        else:
            raise ValueError(f"unknown DAF input kind: {input_kind}")
        out.write(read)

    pysam.index(str(path))
    return str(path)


def _write_workflow_input(path, input_kind: str, strand: str | None, query_name: str) -> str:
    if input_kind == "m6a_forward":
        return _write_m6a_workflow_input(path, query_name, is_reverse=False)
    if input_kind == "m6a_reverse":
        return _write_m6a_workflow_input(path, query_name, is_reverse=True)
    if input_kind in {"daf_iupac", "daf_md", "daf_mmml"}:
        if strand is None:
            raise ValueError(f"{input_kind} needs a DAF strand")
        return _write_daf_workflow_input(path, query_name, input_kind, strand)
    raise ValueError(f"unknown workflow input kind: {input_kind}")


def _assert_workflow_input_shape(path, input_kind: str, strand: str | None) -> None:
    read = _read_only_record(path)
    mod_positions = _workflow_mod_positions()
    protected_positions = range(PROTECTED_START, PROTECTED_END)

    if input_kind.startswith("m6a"):
        assert read.has_tag("MM")
        assert read.has_tag("ML")
        assert read.is_reverse is (input_kind == "m6a_reverse")
        target_base = "T" if input_kind == "m6a_reverse" else "A"
        assert set(read.query_sequence) == {target_base}
        return

    assert strand is not None
    ref_base, _deam_base, observed_base, _mm_base, st_tag = _daf_bases(
        strand, input_kind
    )
    assert all(read.query_sequence[pos] == observed_base for pos in mod_positions)
    assert all(read.query_sequence[pos] == ref_base for pos in protected_positions)

    if input_kind == "daf_iupac":
        assert read.get_tag("st") == st_tag
        assert not read.has_tag("MD")
        assert not read.has_tag("MM")
    elif input_kind == "daf_md":
        assert read.has_tag("MD")
        assert not read.has_tag("st")
        assert not read.has_tag("MM")
    elif input_kind == "daf_mmml":
        assert read.has_tag("MM")
        assert read.has_tag("ML")
        assert not read.has_tag("MD")
        assert not read.has_tag("st")


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
    ("case_name", "input_kind", "strand", "enzyme", "seq", "mode"),
    WORKFLOW_CASES,
)
def test_hmm_apply_to_tf_recall_to_label_extraction_workflow_by_mode(
    case_name,
    input_kind,
    strand,
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
        strand,
        query_name,
    )
    _assert_workflow_input_shape(input_bam, input_kind, strand)
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

    # fibertools-compatibility invariants (must hold on reverse reads too): the
    # molecular-frame ns/nl and as/al tags must be sorted, non-overlapping, and
    # in-bounds -- ft's liftover panics otherwise.
    read_len = recalled_read.query_length
    for start_tag, len_tag in (("ns", "nl"), ("as", "al")):
        if not recalled_read.has_tag(start_tag):
            continue
        starts = list(recalled_read.get_tag(start_tag))
        lengths = list(recalled_read.get_tag(len_tag))
        assert starts == sorted(starts), f"{start_tag} not sorted"
        assert all(s + length <= read_len for s, length in zip(starts, lengths)), \
            f"{start_tag} out of bounds"
        assert all(starts[i] + lengths[i] <= starts[i + 1]
                   for i in range(len(starts) - 1)), f"{start_tag} overlaps"

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
