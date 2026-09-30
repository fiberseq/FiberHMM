"""The vectorised DAF mismatch scans must reproduce the per-pair code exactly.

``fiberhmm.daf.aligned_arrays.matched_base_arrays`` replaced the Python loops
over ``get_aligned_pairs(with_seq=True)`` in the DAF SNP screen,
``get_daf_positions`` (NRL estimate, apply payloads, QC) and the dedup
fingerprint (``_deam_positions_list``).  These tests compare each against the
frozen pre-vectorisation implementations in ``tests/daf_mismatch_reference.py``
on synthetic alignments with planted SNPs, CT/GA/mixed deamination, both
orientations, soft/hard clips, insertions, deletions, N skips, =/X CIGARs,
IUPAC R/Y queries, reads without MD (FASTA fallback), malformed MD, filtered
records, and a position whose MD reference base disagrees between reads.

The equivalence tests pass on the old code by construction (it *is* the
reference); ``test_*_uses_vectorised_path`` fail before the change.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pysam
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import daf_mismatch_reference as reference  # noqa: E402

from fiberhmm.cli.extract_tags import _build_query_to_ref, _deam_positions_list  # noqa: E402
from fiberhmm.daf import encoder, snps  # noqa: E402
from fiberhmm.daf.aligned_arrays import matched_base_arrays  # noqa: E402

BASES = "ACGT"
CHROMS = {"amp": 3_000, "wide": 300_000}


def _md_and_query(ref, start, ops, rng, direction, rate, alleles, iupac,
                  claim_overrides):
    """Build (query, cigar, md) for ``ops`` placed at ``start`` on ``ref``."""
    query = []
    cigar = []
    md = []
    matches = 0
    r = start
    for op, length in ops:
        if op in (0, 7, 8):
            run = []
            for _ in range(length):
                base = alleles.get(r, ref[r])
                if direction in ("CT", "mixed") and base == "C" and rng.random() < rate:
                    base = "Y" if iupac else "T"
                elif direction in ("GA", "mixed") and base == "G" and rng.random() < rate:
                    base = "R" if iupac else "A"
                elif rng.random() < 0.01:
                    base = rng.choice(BASES)
                claimed = claim_overrides.get(r, ref[r])
                if claimed == "G" and r in claim_overrides:
                    base = "A"
                run.append((base, claimed))
                r += 1
            if op == 0:
                cigar.append((0, length))
            else:
                # Express the block as =/X runs.
                for base, claimed in run:
                    code = 7 if base == claimed else 8
                    if cigar and cigar[-1][0] == code:
                        cigar[-1] = (code, cigar[-1][1] + 1)
                    else:
                        cigar.append((code, 1))
            for base, claimed in run:
                query.append(base)
                if base == claimed:
                    matches += 1
                else:
                    md.append(f"{matches}{claimed}")
                    matches = 0
        elif op == 2:
            md.append(f"{matches}^{ref[r:r + length]}")
            matches = 0
            r += length
            cigar.append((2, length))
        elif op == 3:
            r += length
            cigar.append((3, length))
        elif op in (1, 4):
            query.extend(rng.choice(BASES) for _ in range(length))
            cigar.append((op, length))
        elif op == 5:
            cigar.append((5, length))
    md.append(str(matches))
    return "".join(query), cigar, "".join(md), r


def _random_ops(rng, span, allow_skip, eqx):
    ops = []
    if rng.random() < 0.3:
        ops.append((5, rng.randint(1, 20)))
    if rng.random() < 0.5:
        ops.append((4, rng.randint(1, 30)))
    consumed = 0
    while consumed < span:
        length = min(rng.randint(15, 120), span - consumed)
        ops.append((7 if eqx else 0, length))
        consumed += length
        if consumed >= span:
            break
        choice = rng.random()
        if choice < 0.35:
            ops.append((1, rng.randint(1, 4)))
        elif choice < 0.7:
            length = min(rng.randint(1, 4), span - consumed)
            ops.append((2, length))
            consumed += length
        elif allow_skip and choice < 0.75:
            length = min(rng.randint(5, 30), span - consumed)
            ops.append((3, length))
            consumed += length
    if ops[-1][0] in (1, 2, 3):
        ops.append((7 if eqx else 0, 5))
    if rng.random() < 0.5:
        ops.append((4, rng.randint(1, 30)))
    if rng.random() < 0.3:
        ops.append((5, rng.randint(1, 20)))
    return ops


def _write_dataset(tmp_path: Path, seed: int, short_md: bool):
    rng = random.Random(seed)
    refs = {name: "".join(rng.choice(BASES) for _ in range(length))
            for name, length in CHROMS.items()}
    fasta = tmp_path / "ref.fa"
    with fasta.open("w") as handle:
        for name, seq in refs.items():
            handle.write(f">{name}\n{seq}\n")
    pysam.faidx(str(fasta))

    amp = refs["amp"]
    c_sites = [i for i in range(200, 2800) if amp[i] == "C"]
    g_sites = [i for i in range(200, 2800) if amp[i] == "G"]
    snp_alt = {pos: "T" for pos in rng.sample(c_sites, 4)}
    snp_alt.update({pos: "A" for pos in rng.sample(g_sites, 4)})
    conflict = rng.choice(c_sites)

    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "SQ": [{"SN": name, "LN": length} for name, length in CHROMS.items()],
    })
    bam_path = tmp_path / "reads.bam"
    with pysam.AlignmentFile(str(bam_path), "wb", header=header) as bam:
        for index in range(420):
            chrom = "amp" if index < 320 else "wide"
            ref = refs[chrom]
            if chrom == "amp":
                start = rng.randint(0, 150)
                span = rng.randint(2_000, len(ref) - start - 10)
            else:
                start = rng.randint(0, len(ref) - 1_500)
                span = rng.randint(300, 1_400)
            direction = rng.choice(["CT", "GA", "CT", "GA", "mixed", "none"])
            rate = rng.choice([0.03, 0.1, 0.3])
            alleles = {pos: alt for pos, alt in snp_alt.items()
                       if chrom == "amp" and rng.random() < 0.45}
            claims = {}
            if chrom == "amp" and index % 7 == 0:
                claims[conflict] = "G"
            ops = _random_ops(rng, span, allow_skip=index % 11 == 0,
                              eqx=index % 13 == 0)
            query, cigar, md, _end = _md_and_query(
                ref, start, ops, rng, direction, rate, alleles,
                iupac=index % 9 == 0, claim_overrides=claims)
            read = pysam.AlignedSegment(header)
            read.query_name = f"r{index}"
            read.query_sequence = query
            read.reference_id = 0 if chrom == "amp" else 1
            read.reference_start = start
            read.cigartuples = cigar
            read.mapping_quality = 5 if index % 29 == 0 else 60
            read.is_reverse = rng.random() < 0.5
            if index % 31 == 0:
                read.is_duplicate = True
            if index % 37 == 0:
                read.is_secondary = True
            if index % 41 == 0:
                read.is_supplementary = True
            kind = index % 17
            if kind == 3:
                pass  # no MD: FASTA fallback / skip
            elif kind == 5:
                # MD longer than the CIGAR: pysam rejects it (AssertionError).
                read.set_tag("MD", re.sub(
                    r"(\d+)$", lambda m: str(int(m.group(1)) + 5000), md))
            elif kind == 8 and short_md:
                # MD shorter than the CIGAR: pysam's reference string then
                # reads undefined memory, so only paths that pre-validate
                # MD (get_daf_positions, dedup) are compared on these.
                read.set_tag("MD", md[:-1] + "0" if md[-1] != "0" else md)
            else:
                read.set_tag("MD", md)
            bam.write(read)
        unmapped = pysam.AlignedSegment(header)
        unmapped.query_name = "unmapped"
        unmapped.query_sequence = "ACGT" * 20
        unmapped.is_unmapped = True
        bam.write(unmapped)
    return bam_path, fasta


@pytest.fixture(scope="module", params=[11, 23])
def dataset(request, tmp_path_factory):
    return _write_dataset(tmp_path_factory.mktemp(f"daf{request.param}"),
                          request.param, short_md=False)


@pytest.fixture(scope="module", params=[False, True], ids=["md_ok", "md_short"])
def any_dataset(request, tmp_path_factory):
    return _write_dataset(tmp_path_factory.mktemp("dafany"), 31,
                          short_md=request.param)


def _reads(bam_path):
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        return list(bam.fetch(until_eof=True))


def _canonical(payload):
    return json.dumps(payload, sort_keys=True)


def test_matched_base_arrays_match_aligned_pairs(any_dataset):
    bam_path, _fasta = any_dataset
    covered = 0
    for read in _reads(bam_path):
        arrays = matched_base_arrays(read)
        if arrays is None:
            continue
        covered += 1
        qpos, rpos, ref_codes, query_codes = arrays
        pairs = [(q, r, b) for q, r, b in read.get_aligned_pairs(with_seq=True)
                 if q is not None and r is not None]
        assert qpos.tolist() == [q for q, _, _ in pairs]
        assert rpos.tolist() == [r for _, r, _ in pairs]
        assert bytes(ref_codes).decode() == "".join(b.upper() for _, _, b in pairs)
        sequence = read.query_sequence
        assert bytes(query_codes).decode() == "".join(sequence[q] for q, _, _ in pairs)
    assert covered > 300


@pytest.mark.parametrize("use_fasta", [False, True])
@pytest.mark.parametrize("max_profile_sites", [0, 20, 5000])
def test_snp_screen_matches_reference(dataset, use_fasta, max_profile_sites):
    bam_path, fasta = dataset
    kwargs = dict(
        reference_fasta=str(fasta) if use_fasta else None,
        max_profile_sites=max_profile_sites,
        min_depth=3,
        min_alt_fibers=3,
    )
    expected = reference.call_opposite_conversion_snps(str(bam_path), **kwargs)
    observed = snps.call_opposite_conversion_snps(str(bam_path), **kwargs)
    assert _canonical(observed) == _canonical(expected)


def test_snp_screen_calls_planted_snps(dataset):
    bam_path, fasta = dataset
    payload = snps.call_opposite_conversion_snps(
        str(bam_path), reference_fasta=str(fasta))
    # The synthetic data must exercise the call path, not just the empty case.
    assert payload["n_called_snps"] >= 4
    assert payload["n_profiled_sites"] > 100


def test_snp_screen_offer_cache_eviction_is_exact(dataset, monkeypatch):
    """Clearing the repeat-offer cache mid-run must not change the sample."""
    bam_path, _fasta = dataset
    monkeypatch.setattr(snps._OfferedSites, "_BLOCK_BITS", 6)
    monkeypatch.setattr(snps._OfferedSites, "_BLOCK_MASK", (1 << 6) - 1)
    monkeypatch.setattr(snps, "_OFFERED_SITES_MAX_BLOCKS", 3)
    kwargs = dict(max_profile_sites=40, min_depth=3, min_alt_fibers=3)
    expected = reference.call_opposite_conversion_snps(str(bam_path), **kwargs)
    observed = snps.call_opposite_conversion_snps(str(bam_path), **kwargs)
    assert _canonical(observed) == _canonical(expected)


def test_snp_screen_sparse_site_lookup_matches_reference(dataset, monkeypatch):
    """The searchsorted lookup (wide genomes) must agree with the dense one."""
    bam_path, _fasta = dataset
    monkeypatch.setattr(snps._SiteAccumulator, "_MAX_DENSE_SPAN", 0)
    kwargs = dict(min_depth=3, min_alt_fibers=3)
    expected = reference.call_opposite_conversion_snps(str(bam_path), **kwargs)
    observed = snps.call_opposite_conversion_snps(str(bam_path), **kwargs)
    assert _canonical(observed) == _canonical(expected)


def test_snp_screen_uses_vectorised_path(dataset, monkeypatch):
    """MD-bearing reads must not go through the per-pair ``_profile`` loop."""
    bam_path, _fasta = dataset
    calls = []
    original = snps._profile

    def spy(read, reference_handle=None):
        calls.append(read.query_name)
        return original(read, reference_handle)

    monkeypatch.setattr(snps, "_profile", spy)
    snps.call_opposite_conversion_snps(str(bam_path))
    vectorised = {
        read.query_name for read in _reads(bam_path)
        if matched_base_arrays(read) is not None
    }
    assert len(vectorised) > 250
    # Only reads the vectorised path defers (no MD, malformed MD) may use
    # the per-pair loop.
    assert calls
    assert not set(calls) & vectorised


@pytest.mark.parametrize("force_strand", [None, "CT", "GA"])
def test_get_daf_positions_matches_reference(any_dataset, force_strand):
    bam_path, fasta = any_dataset
    rng = random.Random(5)
    with pysam.FastaFile(str(fasta)) as ref_fasta:
        for read in _reads(bam_path):
            excluded = None
            if read.reference_end is not None and rng.random() < 0.5:
                span = range(read.reference_start, read.reference_end)
                excluded = set(rng.sample(span, min(40, len(span))))
            for handle in (None, ref_fasta):
                expected = reference.get_daf_positions(
                    read, force_strand=force_strand, ref_fasta=handle,
                    excluded_reference_positions=excluded)
                observed = encoder.get_daf_positions(
                    read, force_strand=force_strand, ref_fasta=handle,
                    excluded_reference_positions=excluded)
                assert observed == expected, read.query_name


def test_get_daf_positions_uses_vectorised_path(any_dataset, monkeypatch):
    bam_path, _fasta = any_dataset
    calls = []
    original = encoder._daf_positions_from_arrays

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(encoder, "_daf_positions_from_arrays", spy)
    for read in _reads(bam_path):
        encoder.get_daf_positions(read)
    assert len(calls) > 250


def test_dedup_deamination_fingerprint_matches_reference(any_dataset):
    bam_path, _fasta = any_dataset
    for read in _reads(bam_path):
        observed = _deam_positions_list(read, _build_query_to_ref(read))
        seq = read.query_sequence or ""
        if "R" in seq or "Y" in seq:
            continue  # IUPAC branch (priority 2) is not the MD code path
        assert observed == reference.deam_positions_md_branch(read), read.query_name


def test_md_reference_length_matches_reference():
    rng = random.Random(3)
    alphabet = "0123456789ACGTNacgt^^-*"
    samples = ["", "0", "12", "5^AC0T3", "^", "10A5^TT^G2", "3x4", "7é2", "٣A"]
    samples += ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
                for _ in range(3000)]
    for md in samples:
        try:
            expected = reference._md_tag_ref_length(md)
        except Exception as exc:  # noqa: BLE001 - compare failure modes too
            with pytest.raises(type(exc)):
                encoder._md_tag_ref_length(md)
            continue
        assert encoder._md_tag_ref_length(md) == expected, md


def test_matched_base_arrays_defers_on_non_pysam_reads():
    class Stub:
        query_sequence = "ACGT"
        cigartuples = [(0, 4)]

    assert matched_base_arrays(Stub()) is None
