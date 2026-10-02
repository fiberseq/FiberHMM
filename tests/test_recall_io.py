"""fiberhmm-recall-tfs / -recall-nucs input handling (release audit 2026-09-29).

- Output of ``--no-legacy-tags`` carries footprints only in FiberHMM's MA.
  The recallers read only ns/nl/as/al (or fibertools Ma), saw no footprints
  and stripped every MA annotation, exiting 0.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pysam
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _run_cli(module, *args, timeout=300):
    env = dict(os.environ)
    env.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout, env=env,
    )


def _revcomp(seq):
    return seq.translate(COMPLEMENT)[::-1]


def _mm_entry(sequence, positions, base, strand):
    index = {pos: i for i, pos in enumerate(
        p for p, b in enumerate(sequence) if b == base)}
    skips, previous = [], -1
    for pos in sorted(positions):
        skips.append(index[pos] - previous - 1)
        previous = index[pos]
    return f"{base}{strand}a." + "".join(f",{s}" for s in skips) + ";"


def make_hia5_bam(path, platform="pacbio", n_reads=12, length=3000, seed=5):
    """Hia5 reads over a phased nucleosome array, both strands, MM/ML in the
    platform's style (PacBio A+a and T-a; Nanopore A+a only)."""
    rng = np.random.default_rng(seed)
    contig = 20_000
    reference = "".join(rng.choice(list("ACGT"), size=contig))
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": contig}],
    })
    reads = []
    for i in range(n_reads):
        start = int(rng.integers(0, contig - length))
        protected = np.zeros(length, dtype=bool)
        pos = int(rng.integers(0, 60))
        while pos < length:
            if rng.random() < 0.9:
                protected[pos:pos + 147] = True
            pos += 192 + int(rng.integers(-10, 11))
        reverse = bool(i % 2)
        rate = np.where(protected, 0.01, 0.7)
        ref_seq = reference[start:start + length]
        original = _revcomp(ref_seq) if reverse else ref_seq
        orig_rate = rate[::-1] if reverse else rate
        entries, ml = [], []
        targets = (("A", "+"), ("T", "-")) if platform == "pacbio" else (("A", "+"),)
        for base, strand in targets:
            hits = [k for k, b in enumerate(original)
                    if b == base and rng.random() < orig_rate[k]]
            entries.append(_mm_entry(original, hits, base, strand))
            ml.extend([250] * len(hits))
        read = pysam.AlignedSegment(header)
        read.query_name = f"r{i:03d}"
        read.query_sequence = _revcomp(original) if reverse else original
        read.flag = 16 if reverse else 0
        read.reference_id = 0
        read.reference_start = start
        read.mapping_quality = 60
        read.cigartuples = [(0, length)]
        read.query_qualities = pysam.qualitystring_to_array("I" * length)
        read.set_tag("MM", "".join(entries))
        read.set_tag("ML", ml)
        reads.append(read)
    unsorted = str(path) + ".unsorted.bam"
    with pysam.AlignmentFile(unsorted, "wb", header=header) as out:
        for read in reads:
            out.write(read)
    pysam.sort("-o", str(path), unsorted)
    os.remove(unsorted)
    pysam.index(str(path))
    return str(path)


def _annotations(path):
    out = {}
    with pysam.AlignmentFile(path, check_sq=False) as bam:
        for read in bam.fetch(until_eof=True):
            out[read.query_name] = (
                read.get_tag("MA") if read.has_tag("MA") else None,
                list(read.get_tag("AQ")) if read.has_tag("AQ") else None,
            )
    return out


def _counts(annotations):
    from fiberhmm.io.ma_tags import parse_ma_tag

    counts = {"nuc": 0, "msp": 0, "tf": 0}
    for ma, _aq in annotations.values():
        if ma:
            parsed = parse_ma_tag(ma)
            for key in counts:
                counts[key] += len(parsed[key])
    return counts


@pytest.fixture(scope="module")
def called_bams(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("recall_io")
    bam = make_hia5_bam(tmp / "hia5.bam")
    common = ["-i", bam, "--enzyme", "hia5", "--seq", "pacbio", "-c", "1",
              "--no-qc", "--min-read-length", "0"]
    legacy = str(tmp / "legacy.call.bam")
    ma_only = str(tmp / "ma_only.call.bam")
    for output, extra in ((legacy, []), (ma_only, ["--no-legacy-tags"])):
        proc = _run_cli("fiberhmm.cli.call", *common, "-o", output, *extra)
        assert proc.returncode == 0, proc.stderr
    with pysam.AlignmentFile(ma_only, check_sq=False) as handle:
        assert not any(read.has_tag("ns") or read.has_tag("as")
                       for read in handle.fetch(until_eof=True))
    return tmp, legacy, ma_only


@pytest.mark.parametrize("recall_nucs", [False, True])
def test_recall_of_ma_only_input_matches_recall_of_legacy_input(
        called_bams, recall_nucs):
    tmp, legacy, ma_only = called_bams
    flag = ["--recall-nucs"] if recall_nucs else []
    results = []
    for name, source in (("legacy", legacy), ("ma_only", ma_only)):
        output = str(tmp / f"{name}.recall{int(recall_nucs)}.bam")
        proc = _run_cli("fiberhmm.cli.recall_tfs", "-i", source, "-o", output,
                        "--enzyme", "hia5", "--seq", "pacbio", "-c", "1", *flag)
        assert proc.returncode == 0, proc.stderr
        results.append(_annotations(output))
    from_legacy, from_ma_only = results
    counts = _counts(from_ma_only)
    # Before the fix every annotation was stripped (0/0/0).
    assert counts["nuc"] > 0 and counts["msp"] > 0
    assert from_ma_only == from_legacy
    # And the MA-only recall keeps the call's own annotations.
    assert counts == _counts(_annotations(ma_only)) or recall_nucs


def test_phase_nrl_estimate_reads_fiberhmm_ma(called_bams):
    from fiberhmm.cli.recall_tfs import _estimate_phase_nrl_from_tags

    _tmp, legacy, ma_only = called_bams
    from_ma = _estimate_phase_nrl_from_tags(ma_only, 85)
    assert from_ma["n_pairs"] > 0
    assert from_ma == _estimate_phase_nrl_from_tags(legacy, 85)
