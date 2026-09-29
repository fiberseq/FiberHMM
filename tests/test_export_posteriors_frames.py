"""fiberhmm-posteriors must decode the same observations as fiberhmm-call.

Regression tests for the 3.0 audit findings: reverse reads were encoded from
``modified_bases_forward`` (original-read frame) against SEQ, all mod types
were pooled, R/Y DAF input produced 0 fibers, reads spanning region
boundaries were written once per region, hard-clipped MM records were used,
and footprint sizes were 1 bp short.
"""
from __future__ import annotations

import array
import gzip
import random

import numpy as np
import pysam
import pytest

from fiberhmm.cli import export_posteriors as ep
from fiberhmm.core.bam_reader import encode_from_query_sequence
from fiberhmm.core.model_io import load_model_with_metadata
from fiberhmm.models import get_model_path

READ_LEN = 800
CHROM_LEN = 20_000


def _model(enzyme, seq=None):
    path = get_model_path(enzyme, tool="apply", seq=seq)
    model, k, _mode = load_model_with_metadata(path, normalize=True)
    return path, model, k


def _header():
    return {"HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [{"LN": CHROM_LEN, "SN": "chr1"}]}


def _rand_seq(rng, n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mm_entry(walk_seq, base, code, positions):
    """MM entry listing EVERY ``base`` of ``walk_seq`` (MM walk frame)."""
    targets = [i for i, b in enumerate(walk_seq) if b == base]
    ml = [250 if t in positions else 5 for t in targets]
    return f"{base}{'+' if base in 'AC' else '-'}{code}.," + ",".join(
        ["0"] * len(targets)) + ";", ml


def _hia5_read(header, name, start, is_reverse, rng, cigar=None):
    """PacBio-style Hia5 read: accessible flanks, protected 300-500 window."""
    seq = _rand_seq(rng, READ_LEN)
    comp = str.maketrans("ACGT", "TGCA")
    walk = seq.translate(comp)[::-1] if is_reverse else seq

    def accessible(i):
        return not (300 <= i < 500)

    # Mods chosen in the MM walk (original-read) frame, deliberately
    # asymmetric so a frame error changes the observations.
    a_mods = {i for i, b in enumerate(walk) if b == "A" and accessible(i)
              and rng.random() < 0.6}
    t_mods = {i for i, b in enumerate(walk) if b == "T" and accessible(i)
              and rng.random() < 0.6}
    c_mods = {i for i, b in enumerate(walk) if b == "C"}   # 5mC: must be ignored
    mm_a, ml_a = _mm_entry(walk, "A", "a", a_mods)
    mm_t, ml_t = _mm_entry(walk, "T", "a", t_mods)
    mm_c, ml_c = _mm_entry(walk, "C", "m", c_mods)

    a = pysam.AlignedSegment(header)
    a.query_name = name
    a.query_sequence = seq
    a.flag = 16 if is_reverse else 0
    a.reference_id = 0
    a.reference_start = start
    a.mapping_quality = 60
    a.cigartuples = cigar or [(0, READ_LEN)]
    a.query_qualities = pysam.qualitystring_to_array("I" * READ_LEN)
    a.set_tags([("MM", mm_c + mm_a + mm_t, "Z"),
                ("ML", array.array("B", ml_c + ml_a + ml_t))])
    return a


def _write_bam(path, reads):
    with pysam.AlignmentFile(str(path), "wb", header=_header()) as out:
        for r in sorted(reads, key=lambda r: r.reference_start):
            out.write(r)
    pysam.index(str(path))
    return str(path)


def _seq_frame_m6a(read, threshold=128):
    """Oracle: pysam's SEQ-frame m6A calls (the positions call decodes)."""
    out = set()
    for (base, _strand, code), pqs in read.modified_bases.items():
        if code != "a":
            continue
        out.update(p for p, q in pqs if q >= threshold)
    return out


@pytest.mark.parametrize("is_reverse", [False, True])
def test_posteriors_match_call_observations_on_both_strands(is_reverse):
    rng = random.Random(11)
    header = pysam.AlignmentHeader.from_dict(_header())
    read = _hia5_read(header, "r", 1_000, is_reverse, rng)
    _path, model, k = _model("hia5", "pacbio")

    got = ep.extract_posteriors_from_read(read, model, "pacbio-fiber", k, 10)
    assert got is not None

    obs = encode_from_query_sequence(
        read.query_sequence, _seq_frame_m6a(read), 10,
        mode="pacbio-fiber", context_size=k)
    states, post = model.predict_with_posteriors(obs)
    np.testing.assert_allclose(
        got["posteriors"].astype(np.float32),
        post[:, 0].astype(np.float16).astype(np.float32), atol=1e-3)

    # Footprint runs map to reference half-open intervals of full length
    # (full-match CIGAR: reference size == query run length).
    padded = np.concatenate([[1], states, [1]])
    d = np.diff(padded)
    q_starts, q_ends = np.where(d == -1)[0], np.where(d == 1)[0]
    assert len(q_starts) > 0
    np.testing.assert_array_equal(got["footprint_starts"], q_starts + 1_000)
    np.testing.assert_array_equal(got["footprint_sizes"], q_ends - q_starts)


def _tsv_rows(path):
    with gzip.open(path, "rt") as fh:
        return [ln.rstrip("\n").split("\t") for ln in fh
                if ln.strip() and not ln.startswith("#")]


def test_read_spanning_region_boundary_written_once(tmp_path):
    rng = random.Random(3)
    header = pysam.AlignmentHeader.from_dict(_header())
    reads = [_hia5_read(header, "inside", 200, False, rng),
             _hia5_read(header, "spanning", 4_700, True, rng)]
    bam = _write_bam(tmp_path / "in.bam", reads)
    model_path, _m, _k = _model("hia5", "pacbio")
    out = tmp_path / "post.tsv.gz"
    n = ep.export_posteriors_tsv(bam, model_path, str(out), n_cores=1,
                                 region_size=5_000, verbose=False,
                                 mode_override="pacbio-fiber")
    names = [row[0] for row in _tsv_rows(out)]
    assert sorted(names) == ["inside", "spanning"]
    assert n == 2


def test_hard_clipped_mm_record_is_not_exported(tmp_path):
    rng = random.Random(5)
    header = pysam.AlignmentHeader.from_dict(_header())
    clipped = _hia5_read(header, "clipped", 1_000, False, rng,
                         cigar=[(5, 400), (0, READ_LEN)])
    ok = _hia5_read(header, "ok", 3_000, False, rng)
    bam = _write_bam(tmp_path / "in.bam", [clipped, ok])
    model_path, model, k = _model("hia5", "pacbio")
    stats = ep._new_stats()
    assert ep.extract_posteriors_from_read(
        clipped, model, "pacbio-fiber", k, 10, stats=stats) is None
    assert stats["mm_not_applicable"] == 1
    out = tmp_path / "post.tsv.gz"
    ep.export_posteriors_tsv(bam, model_path, str(out), n_cores=1,
                             verbose=False, mode_override="pacbio-fiber")
    assert [row[0] for row in _tsv_rows(out)] == ["ok"]


def test_iupac_daf_input_exports_fibers(tmp_path):
    rng = random.Random(9)
    header = pysam.AlignmentHeader.from_dict(_header())
    seq = list(_rand_seq(rng, READ_LEN))
    for i, b in enumerate(seq):
        if b == "C" and not (300 <= i < 500) and rng.random() < 0.5:
            seq[i] = "Y"
    a = pysam.AlignedSegment(header)
    a.query_name = "daf"
    a.query_sequence = "".join(seq)
    a.flag = 0
    a.reference_id = 0
    a.reference_start = 1_000
    a.mapping_quality = 60
    a.cigartuples = [(0, READ_LEN)]
    a.set_tags([("st", "CT", "Z")])
    bam = _write_bam(tmp_path / "daf.bam", [a])
    model_path, _m, _k = _model("dddb", "nanopore")
    out = tmp_path / "post.tsv.gz"
    n = ep.export_posteriors_tsv(bam, model_path, str(out), n_cores=1,
                                 verbose=False, mode_override="daf")
    rows = _tsv_rows(out)
    assert n == 1 and [r[0] for r in rows] == ["daf"]
    assert rows[0][4] == "+"


def test_cli_defaults_match_call(monkeypatch, tmp_path):
    import sys

    captured = {}
    monkeypatch.setattr(ep, "load_model_with_metadata",
                        lambda *a, **k: (object(), 3, "pacbio-fiber"))
    monkeypatch.setattr(ep, "export_posteriors",
                        lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(sys, "argv", [
        "fiberhmm-posteriors", "-i", "in.bam", "-o", str(tmp_path / "p.tsv.gz"),
        "--enzyme", "hia5", "--seq", "pacbio"])
    ep.main()
    assert captured["edge_trim"] == 10
    assert captured["prob_threshold"] == 128
