"""Tests for optional opposite-conversion DAF SNP calling and masking."""
from __future__ import annotations

from inspect import signature
from pathlib import Path

import pysam

from fiberhmm.daf.encoder import get_daf_positions
from fiberhmm.cli.daf_snps import parse_args as parse_snp_args
from fiberhmm.daf.snps import (
    DEFAULT_SNP_MIN_ALT_FIBERS,
    DEFAULT_SNP_MIN_DEPTH,
    DEFAULT_SNP_MIN_FRACTION,
    VALIDATED_SNP_POLICY_NAME,
    call_opposite_conversion_snps,
    describe_snp_threshold_policy,
    wrapped_reference_sites,
)


def test_validated_policy_is_canonical_but_cli_thresholds_are_overridable():
    policy = describe_snp_threshold_policy(
        DEFAULT_SNP_MIN_FRACTION,
        DEFAULT_SNP_MIN_DEPTH,
        DEFAULT_SNP_MIN_ALT_FIBERS,
    )
    assert policy["name"] == VALIDATED_SNP_POLICY_NAME
    assert policy["uses_validated_defaults"] is True
    assert describe_snp_threshold_policy(0.30, 10, 7)["name"] == "custom"

    api_defaults = signature(call_opposite_conversion_snps).parameters
    assert api_defaults["min_fraction"].default == DEFAULT_SNP_MIN_FRACTION
    assert api_defaults["min_depth"].default == DEFAULT_SNP_MIN_DEPTH
    assert api_defaults["min_alt_fibers"].default == DEFAULT_SNP_MIN_ALT_FIBERS

    defaults = parse_snp_args(["-i", "input.bam"])
    assert defaults.min_fraction == DEFAULT_SNP_MIN_FRACTION
    assert defaults.min_depth == DEFAULT_SNP_MIN_DEPTH
    assert defaults.min_alt_fibers == DEFAULT_SNP_MIN_ALT_FIBERS
    custom = parse_snp_args(
        [
            "-i",
            "input.bam",
            "--min-fraction",
            "0.3",
            "--min-depth",
            "10",
            "--min-alt-fibers",
            "7",
        ]
    )
    assert (custom.min_fraction, custom.min_depth, custom.min_alt_fibers) == (
        0.3,
        10,
        7,
    )


def _md_tag(reference: str, query: str) -> tuple[str, int]:
    fields = []
    matches = 0
    mismatches = 0
    for reference_base, query_base in zip(reference, query):
        if reference_base == query_base:
            matches += 1
        else:
            fields.extend((str(matches), reference_base))
            matches = 0
            mismatches += 1
    fields.append(str(matches))
    return "".join(fields), mismatches


def _write_snp_bam(path: Path) -> None:
    reference = "C" * 50 + "G" * 50
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    )
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        for index in range(20):
            query = list(reference)
            for position in range(50, 60):
                query[position] = "A"
            if index < 10:
                query[10] = "T"
            if index < 8:
                query[11] = "T"
            if index < 12:
                query[70] = "A"
            sequence = "".join(query)
            md, nm = _md_tag(reference, sequence)
            read = pysam.AlignedSegment(header)
            read.query_name = f"ga_{index}"
            read.query_sequence = sequence
            read.reference_id = 0
            read.reference_start = 100
            read.mapping_quality = 60
            read.cigar = [(0, len(sequence))]
            read.set_tag("MD", md)
            read.set_tag("NM", nm)
            bam.write(read)
        for index in range(20):
            query = list(reference)
            for position in range(20, 30):
                query[position] = "T"
            if index < 8:
                query[70] = "A"
            if index < 12:
                query[10] = "T"
            sequence = "".join(query)
            md, nm = _md_tag(reference, sequence)
            read = pysam.AlignedSegment(header)
            read.query_name = f"ct_{index}"
            read.query_sequence = sequence
            read.reference_id = 0
            read.reference_start = 100
            read.mapping_quality = 60
            read.cigar = [(0, len(sequence))]
            read.set_tag("MD", md)
            read.set_tag("NM", nm)
            bam.write(read)


def test_opposite_conversion_snp_caller_and_encoder_mask(tmp_path):
    bam_path = tmp_path / "snps.bam"
    _write_snp_bam(bam_path)
    payload = call_opposite_conversion_snps(
        str(bam_path),
        min_fraction=0.30,
        min_depth=10,
        min_alt_fibers=3,
        min_dominant_events=5,
        min_dominant_purity=0.80,
    )
    assert payload["threshold_policy"]["name"] == "custom"
    assert payload["threshold_policy"]["validated_defaults"] == {
        "min_fraction_each_direction": 0.20,
        "min_depth_each_direction": 5,
        "min_mismatch_fibers_each_direction": 5,
        "bidirectional_support_required": True,
    }
    calls = {
        (call["position_0based"], call["reference"], call["alternate"]): call
        for call in payload["calls"]
    }
    assert (110, "C", "T") in calls
    assert calls[(110, "C", "T")]["alternate_fraction"] == 0.5
    assert (170, "G", "A") in calls
    assert calls[(170, "G", "A")]["alternate_fraction"] == 0.4
    assert (111, "C", "T") not in calls
    landscape = {
        (site["position_0based"], site["reference"], site["alternate"]): site
        for site in payload["site_distribution"]
    }
    assert landscape[(110, "C", "T")]["expected_direction_mismatch_fraction"] == 0.6
    assert landscape[(110, "C", "T")]["opposite_direction_mismatch_fraction"] == 0.5
    assert landscape[(110, "C", "T")]["called_as_snp"] is True
    assert landscape[(170, "G", "A")]["expected_direction_mismatch_fraction"] == 0.6
    assert landscape[(170, "G", "A")]["opposite_direction_mismatch_fraction"] == 0.4
    assert payload["dominant_amplicon"] == {
        "chrom": "chr1",
        "start_0based": 100,
        "end_0based_exclusive": 200,
        "overlapping_dominant_fibers": 40,
        "selection": "highest-coverage discovered amplicon with >= 20 aligned reads",
    }
    assert payload["n_discovered_amplicons"] == 1
    assert payload["amplicons"][0]["total_aligned_reads"] == 40
    assert payload["amplicons"][0]["consensus_length_bp"] == 100
    assert payload["amplicons"][0]["n_called_snps"] == 2
    assert {
        (site["change"], site["relative_position_bp"])
        for site in payload["amplicons"][0]["snp_positions"]
    } == {("C>T", 10), ("G>A", 70)}

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        read = next(bam.fetch(until_eof=True))
        unmasked = get_daf_positions(read)
        masked = get_daf_positions(read, excluded_reference_positions={110})
    assert unmasked is not None and masked is not None
    assert 10 in unmasked[0]
    assert 10 not in masked[0]


# --- reference end: topology decides; every turn of a circular record --------

_EDGE_KWARGS = {"min_dominant_events": 1, "min_depth": 2, "min_alt_fibers": 2,
                "min_fraction": 0.2}


def _past_end_bam(path, sq, comments=()):
    """Ten records spanning [90, 110) on LN 100 that support C->T at unrolled 105."""
    header = pysam.AlignmentHeader.from_dict({"SQ": [sq], **({"CO": list(comments)}
                                                            if comments else {})})
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for i in range(10):
            r = pysam.AlignedSegment(header); r.query_name = f"x{i}"; r.reference_id = 0
            r.reference_start = 90; r.mapping_quality = 60; r.cigarstring = "20M"
            if i < 5:
                r.query_sequence = "T" * 20; r.set_tag("MD", "0C" * 20 + "0")
            else:
                r.query_sequence = "A" * 15 + "T" + "A" * 4
                r.set_tag("MD", "0G" * 15 + "0C" + "0G" * 4 + "0")
            out.write(r)
    return path


def test_linear_overhang_is_not_folded_onto_the_contig(tmp_path):
    from fiberhmm.daf.snps import _read_arrays
    for name, sq in (("tp_linear", {"SN": "p", "LN": 100, "TP": "linear"}),
                     ("no_tp", {"SN": "p", "LN": 100})):
        bam = _past_end_bam(tmp_path / f"{name}.bam", sq)
        result = call_opposite_conversion_snps(str(bam), **_EDGE_KWARGS)
        positions = [s["position_0based"] for s in result["site_distribution"]]
        assert 5 not in positions and not any(p >= 100 for p in positions), (name, positions)
        with pysam.AlignmentFile(str(bam)) as handle:
            read = next(iter(handle))
            rpos, ref_codes, query_codes = _read_arrays(read)
            assert rpos.tolist() == list(range(90, 100))
            assert len(ref_codes) == len(query_codes) == 10
            assert wrapped_reference_sites(read, {5}) == {5}


def test_circular_overhang_folds_by_tp_or_reference_comment(tmp_path):
    from fiberhmm.pipeline.reference import REFERENCE_COMMENT_PREFIX
    comment = REFERENCE_COMMENT_PREFIX + "contig=p;length=100;topology=circular"
    for name, sq, comments in (("tp", {"SN": "p", "LN": 100, "TP": "circular"}, ()),
                               ("comment", {"SN": "p", "LN": 100}, (comment,))):
        bam = _past_end_bam(tmp_path / f"{name}.bam", sq, comments)
        result = call_opposite_conversion_snps(str(bam), **_EDGE_KWARGS)
        assert [s["position_0based"] for s in result["site_distribution"]
                if s["called_as_snp"]] == [5], name


def _multi_turn_read(header, start=90, length=220, mismatch_turns=()):
    """A G->A-dominant record on LN 100 spanning [start, start+length): the
    reference is C at site 5 and G elsewhere; G->A at sites 20/30/40/50 on
    every turn, and C->T at site 5 on the listed turns (0-based turn number)."""
    r = pysam.AlignedSegment(header); r.query_name = "multi"; r.reference_id = 0
    r.reference_start = start; r.mapping_quality = 60; r.cigarstring = f"{length}M"
    query, md, run = [], [], 0
    for offset in range(length):
        position = start + offset
        site, turn = position % 100, position // 100
        if site == 5 and turn in mismatch_turns:
            query.append("T"); md.append(f"{run}C"); run = 0
        elif site == 5:
            query.append("C"); run += 1
        elif site in (20, 30, 40, 50):
            query.append("A"); md.append(f"{run}G"); run = 0
        else:
            query.append("G"); run += 1
    md.append(str(run))
    r.query_sequence = "".join(query); r.set_tag("MD", "".join(md))
    return r


def test_multi_turn_circular_records_are_masked_and_counted_on_every_turn(tmp_path):
    from fiberhmm.daf.snps import _read_arrays
    from fiberhmm.inference import engine
    header = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "p", "LN": 100, "TP": "circular"}]})
    read = _multi_turn_read(header)
    saved = engine._DAF_SNP_MASK
    try:
        engine._DAF_SNP_MASK = {"p": {5}}
        assert sorted(engine._daf_excluded_query_positions(read)) == [15, 115, 215]
    finally:
        engine._DAF_SNP_MASK = saved
    assert wrapped_reference_sites(read, {5}) == {5, 105, 205, 305}
    rpos, _, _ = _read_arrays(read)
    assert int(rpos.max()) < 100 and sorted(set(rpos.tolist())) == list(range(100))
    # The encoder mask drops the C->T event on every turn.
    marked = _multi_turn_read(header, mismatch_turns=(1, 2, 3))
    masked = get_daf_positions(marked, force_strand="CT",
                               excluded_reference_positions=wrapped_reference_sites(marked, {5}))
    assert masked == get_daf_positions(read, force_strand="CT")
    assert masked != get_daf_positions(marked, force_strand="CT")
    # The SNP screen counts a molecule once per site however many turns cover
    # it (and a mismatch once if any turn has it); pass 1 offers each site once.
    from fiberhmm.daf.snps import _SiteAccumulator, _covers_a_turn_twice
    accumulator = _SiteAccumulator({5: ("C", "T", "CT"), 20: ("G", "A", "GA")})
    for turns in ((1,), (), (0, 1, 2, 3)):
        rpos, ref_codes, query_codes = _read_arrays(_multi_turn_read(header, mismatch_turns=turns))
        accumulator.add_read("GA", rpos, ref_codes, query_codes)
    assert accumulator.opposite_depth.tolist() == [3, 0]
    assert accumulator.opposite_mismatches.tolist() == [2, 0]
    assert accumulator.expected_depth.tolist() == [0, 3]
    assert accumulator.expected_mismatches.tolist() == [0, 3]
    assert _covers_a_turn_twice(read)
    single = _multi_turn_read(header, start=90, length=20)
    assert not _covers_a_turn_twice(single)
    bam = tmp_path / "multi.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        for i in range(4):
            r = _multi_turn_read(header, mismatch_turns=(1,) if i < 2 else ())
            r.query_name = f"m{i}"
            out.write(r)
    result = call_opposite_conversion_snps(str(bam), max_profile_sites=5000, min_depth=1,
                                           min_alt_fibers=1, min_dominant_events=1)
    assert result["accounting"]["ga_dominant_records"] == 4


def test_wrapped_mask_follows_in_place_changes_of_a_mutable_mask(tmp_path):
    """A same-size, in-place change of a mutable mask set must not reuse the
    earlier expansion (the cache was keyed on the set's identity and size)."""
    from fiberhmm.daf.snps import load_snp_mask
    from fiberhmm.inference import engine
    header = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "p", "LN": 100, "TP": "circular"}]})
    read = _multi_turn_read(header)
    sites = {5}
    assert wrapped_reference_sites(read, sites) == {5, 105, 205, 305}
    sites.remove(5)
    sites.add(6)
    assert wrapped_reference_sites(read, sites) == {6, 106, 206, 306}
    saved = engine._DAF_SNP_MASK
    try:
        engine._DAF_SNP_MASK = {"p": sites}
        assert sorted(engine._daf_excluded_query_positions(read)) == [16, 116, 216]
        sites.remove(6)
        sites.add(7)
        assert sorted(engine._daf_excluded_query_positions(read)) == [17, 117, 217]
    finally:
        engine._DAF_SNP_MASK = saved
    # Masks loaded from a BED are immutable, so their expansion is memoised.
    bed = tmp_path / "mask.bed"
    bed.write_text("p\t5\t6\n")
    mask = load_snp_mask(str(bed))
    assert mask == {"p": frozenset({5})} and isinstance(mask["p"], frozenset)
    first = wrapped_reference_sites(read, mask["p"])
    assert first == {5, 105, 205, 305}
    assert wrapped_reference_sites(read, mask["p"]) is first
