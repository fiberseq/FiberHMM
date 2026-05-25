"""Characterization tests for the fused `fiberhmm-call` pipelines."""

import os

import pysam
from conftest import make_synthetic_bam

from fiberhmm.inference.parallel import (
    _process_bam_region_parallel_fused,
    _process_bam_streaming_pipeline_fused,
)
from fiberhmm.models import get_model_path


def _run_fused_streaming(input_bam, output_bam, model_path, *,
                          with_scores=False, circular=False, min_llr=1000.0):
    return _process_bam_streaming_pipeline_fused(
        input_bam=input_bam,
        output_bam=output_bam,
        model_path=model_path,
        recall_model_path=None,
        train_rids=set(),
        edge_trim=10,
        circular=circular,
        mode="pacbio-fiber",
        context_size=3,
        msp_min_size=0,
        nuc_min_size=85,
        min_mapq=0,
        prob_threshold=0,
        min_read_length=0,
        with_scores=with_scores,
        min_llr=min_llr,
        min_opps=3,
        unify_threshold=90,
        emission_uplift=1.0,
        also_write_legacy=True,
        downstream_compat=False,
        max_reads=0,
        n_cores=1,
        chunk_size=10,
        io_threads=1,
    )


def _run_fused_region(input_bam, output_bam, model_path, *,
                       with_scores=False, circular=False, min_llr=1000.0):
    return _process_bam_region_parallel_fused(
        input_bam=input_bam,
        output_bam=output_bam,
        apply_model_path=model_path,
        recall_model_path=None,
        train_rids=set(),
        edge_trim=10,
        circular=circular,
        mode="pacbio-fiber",
        context_size=3,
        msp_min_size=0,
        nuc_min_size=85,
        min_mapq=0,
        prob_threshold=0,
        min_read_length=0,
        with_scores=with_scores,
        min_llr=min_llr,
        min_opps=3,
        unify_threshold=90,
        emission_uplift=1.0,
        also_write_legacy=True,
        downstream_compat=False,
        n_cores=1,
        region_size=10_000_000,
        skip_scaffolds=False,
        chroms=None,
        io_threads=1,
    )


def _run_daf_fused_streaming(input_bam, output_bam, *, ref_fasta_path=None):
    return _process_bam_streaming_pipeline_fused(
        input_bam=input_bam,
        output_bam=output_bam,
        model_path=get_model_path("ddda", tool="apply"),
        recall_model_path=get_model_path("ddda", tool="recall"),
        train_rids=set(),
        edge_trim=10,
        circular=False,
        mode="daf",
        context_size=3,
        msp_min_size=0,
        nuc_min_size=85,
        min_mapq=0,
        prob_threshold=0,
        min_read_length=0,
        with_scores=True,
        min_llr=1000.0,
        min_opps=3,
        unify_threshold=90,
        emission_uplift=1.0,
        also_write_legacy=True,
        downstream_compat=False,
        max_reads=0,
        n_cores=1,
        chunk_size=10,
        io_threads=1,
        ref_fasta_path=ref_fasta_path,
    )


def _read_tags_by_name(bam_path):
    tags_by_name = {}
    with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            tags = {}
            for tag in ("ns", "nl", "as", "al", "nq", "aq", "MA", "AQ"):
                if not read.has_tag(tag):
                    continue
                value = read.get_tag(tag)
                tags[tag] = value if isinstance(value, str) else list(value)
            tags_by_name[read.query_name] = tags
    return tags_by_name


def _md_for_ct_mismatches(read_length, mismatch_positions):
    parts = []
    previous = -1
    for pos in mismatch_positions:
        parts.append(str(pos - previous - 1))
        parts.append("C")
        previous = pos
    parts.append(str(read_length - previous - 1))
    return "".join(parts)


def _write_ct_daf_fixture_bams(tmp_path):
    """Write matched raw-MD, raw-reference, and IUPAC DAF BAMs."""
    read_length = 500
    chrom_length = 5_000
    n_reads = 6
    deam_positions = list(range(50, read_length - 50, 25))

    ref_fasta = tmp_path / "ct_ref.fa"
    ref_fasta.write_text(">chr1\n" + ("C" * chrom_length) + "\n")
    pysam.faidx(str(ref_fasta))

    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": chrom_length}],
    })
    paths = {
        "raw_md": tmp_path / "raw_md.bam",
        "raw_ref": tmp_path / "raw_ref.bam",
        "iupac": tmp_path / "iupac.bam",
    }

    for flavor, path in paths.items():
        with pysam.AlignmentFile(path, "wb", header=header) as out:
            for i in range(n_reads):
                seq = ["C"] * read_length
                for pos in deam_positions:
                    seq[pos] = "Y" if flavor == "iupac" else "T"

                read = pysam.AlignedSegment()
                read.query_name = f"daf_ct_{i:03d}"
                read.query_sequence = "".join(seq)
                read.query_qualities = pysam.qualitystring_to_array(
                    "I" * read_length
                )
                read.flag = 0
                read.reference_id = 0
                read.reference_start = i * (read_length + 10)
                read.mapping_quality = 60
                read.cigar = [(0, read_length)]
                if flavor == "raw_md":
                    read.set_tag("MD", _md_for_ct_mismatches(read_length, deam_positions))
                if flavor == "iupac":
                    read.set_tag("st", "CT", value_type="Z")
                out.write(read)
        pysam.index(str(path))

    return paths["raw_md"], paths["raw_ref"], paths["iupac"], ref_fasta


def test_fused_streaming_emits_spec_and_legacy_tags(
    synthetic_bam_small, benchmark_model_path, tmp_path
):
    output = str(tmp_path / "fused_stream.bam")

    total, with_fp = _run_fused_streaming(synthetic_bam_small, output, benchmark_model_path)

    assert total > 0
    assert with_fp > 0
    tags = _read_tags_by_name(output)
    assert any("MA" in t for t in tags.values())
    assert any("AQ" in t for t in tags.values())
    assert any("ns" in t and "nl" in t for t in tags.values())
    assert any("as" in t and "al" in t for t in tags.values())


def test_fused_with_scores_writes_nq_for_kept_nucs(
    synthetic_bam_small, benchmark_model_path, tmp_path
):
    output = str(tmp_path / "fused_scores.bam")

    _run_fused_streaming(
        synthetic_bam_small, output, benchmark_model_path, with_scores=True
    )

    tags = _read_tags_by_name(output)
    scored = [t for t in tags.values() if "ns" in t]
    assert scored
    assert any("nq" in t for t in scored)
    for tag_set in scored:
        if "nq" in tag_set:
            assert len(tag_set["nq"]) == len(tag_set["ns"])
            assert all(0 <= q <= 255 for q in tag_set["nq"])


def test_fused_region_matches_streaming_on_single_region(benchmark_model_path, tmp_path):
    input_bam = str(tmp_path / "one_chrom.bam")
    make_synthetic_bam(
        input_bam,
        n_reads=30,
        read_length=1500,
        n_chroms=1,
        chrom_length=100_000,
        seed=123,
    )
    stream_out = str(tmp_path / "stream.bam")
    region_out = str(tmp_path / "region.bam")

    stream_counts = _run_fused_streaming(input_bam, stream_out, benchmark_model_path)
    region_counts = _run_fused_region(input_bam, region_out, benchmark_model_path)

    assert stream_counts == region_counts
    assert _read_tags_by_name(stream_out) == _read_tags_by_name(region_out)
    assert not any(
        p.name.startswith(".fiberhmm_call_tmp_")
        for p in tmp_path.iterdir()
        if os.path.isdir(p)
    )


def test_daf_raw_md_and_reference_streaming_match_iupac_output(tmp_path):
    raw_md, raw_ref, iupac, ref_fasta = _write_ct_daf_fixture_bams(tmp_path)
    md_out = str(tmp_path / "raw_md_out.bam")
    ref_out = str(tmp_path / "raw_ref_out.bam")
    iupac_out = str(tmp_path / "iupac_out.bam")

    md_counts = _run_daf_fused_streaming(str(raw_md), md_out)
    ref_counts = _run_daf_fused_streaming(
        str(raw_ref), ref_out, ref_fasta_path=str(ref_fasta)
    )
    iupac_counts = _run_daf_fused_streaming(str(iupac), iupac_out)

    assert md_counts == iupac_counts
    assert ref_counts == iupac_counts

    iupac_tags = _read_tags_by_name(iupac_out)
    assert any("ns" in tag_set for tag_set in iupac_tags.values())
    assert _read_tags_by_name(md_out) == iupac_tags
    assert _read_tags_by_name(ref_out) == iupac_tags


def test_daf_reference_fasta_is_closed_after_streaming(tmp_path, monkeypatch):
    _, raw_ref, _, ref_fasta = _write_ct_daf_fixture_bams(tmp_path)
    output = str(tmp_path / "raw_ref_out.bam")
    real_fasta_file = pysam.FastaFile
    closed_paths = []

    class TrackingFastaFile:
        def __init__(self, path):
            self.path = path
            self._inner = real_fasta_file(path)

        def fetch(self, *args, **kwargs):
            return self._inner.fetch(*args, **kwargs)

        def close(self):
            closed_paths.append(self.path)
            self._inner.close()

    monkeypatch.setattr(pysam, "FastaFile", TrackingFastaFile)

    _run_daf_fused_streaming(str(raw_ref), output, ref_fasta_path=str(ref_fasta))

    assert closed_paths == [str(ref_fasta)]


def _assert_circular_output_invariants(input_bam, output_bam):
    """Shared invariants for --circular output: coords stay within read length,
    MA prefix == on-disk read length (not 3x), AN count matches MA annotation
    count, and at least one read carries tags."""
    read_lengths = {}
    with pysam.AlignmentFile(input_bam, "rb", check_sq=False) as src:
        for read in src.fetch(until_eof=True):
            read_lengths[read.query_name] = read.query_length

    tagged = 0
    with pysam.AlignmentFile(output_bam, "rb", check_sq=False) as out:
        for read in out.fetch(until_eof=True):
            rl = read_lengths.get(read.query_name)
            if rl is None or rl <= 0:
                continue

            if read.has_tag("ns"):
                ns = list(read.get_tag("ns"))
                nl = list(read.get_tag("nl"))
                assert ns and len(ns) == len(nl)
                for s, length in zip(ns, nl):
                    assert 0 <= s < rl, f"ns {s} outside [0, {rl}) — tiled coord leaked"
                    assert s + length <= rl, (
                        f"ns+nl {s + length} exceeds read length {rl}"
                    )
                tagged += 1

            if read.has_tag("as"):
                a_s = list(read.get_tag("as"))
                a_l = list(read.get_tag("al"))
                assert a_s and len(a_s) == len(a_l)
                for s, length in zip(a_s, a_l):
                    assert 0 <= s < rl, f"as {s} outside [0, {rl})"
                    assert s + length <= rl

            if read.has_tag("MA"):
                ma = read.get_tag("MA")
                prefix = int(ma.split(";", 1)[0])
                assert prefix == rl, (
                    f"MA prefix {prefix} must equal read length {rl}, not 3x"
                )
                if read.has_tag("AN"):
                    an_count = len(read.get_tag("AN").split(","))
                    ma_ann_count = sum(
                        len(seg.split(":", 1)[1].split(","))
                        for seg in ma.split(";")[1:]
                        if ":" in seg
                    )
                    assert an_count == ma_ann_count, (
                        f"AN tag has {an_count} names but MA has {ma_ann_count} annotations"
                    )

    assert tagged > 0, "circular run produced no tagged reads"


def test_circular_streaming_keeps_coordinates_within_read_length(
    synthetic_bam_small, benchmark_model_path, tmp_path
):
    """End-to-end check that --circular tiles internally but never leaks
    3x-tiled coordinates into output tags, and that MA prefixes use the
    on-disk read length."""
    output = str(tmp_path / "fused_circular.bam")

    total, with_fp = _run_fused_streaming(
        synthetic_bam_small, output, benchmark_model_path,
        circular=True, min_llr=4.0,
    )

    assert total > 0
    assert with_fp > 0
    _assert_circular_output_invariants(synthetic_bam_small, output)


def test_circular_region_parallel_keeps_coordinates_within_read_length(
    benchmark_model_path, tmp_path
):
    """Same invariants as the streaming circular test, but via the
    region-parallel fused worker path."""
    input_bam = str(tmp_path / "circular_region_input.bam")
    make_synthetic_bam(
        input_bam,
        n_reads=30,
        read_length=1500,
        n_chroms=1,
        chrom_length=100_000,
        seed=321,
    )
    output = str(tmp_path / "fused_circular_region.bam")

    total, with_fp = _run_fused_region(
        input_bam, output, benchmark_model_path,
        circular=True, min_llr=4.0,
    )

    assert total > 0
    assert with_fp > 0
    _assert_circular_output_invariants(input_bam, output)
