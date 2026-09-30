"""The DAF SNP screen must give byte-identical outputs for identical inputs.

Two causes made FASTA-less runs differ from run to run:

* An MD tag shorter than the CIGAR's M/=/X/D span: pysam's
  ``get_aligned_pairs(with_seq=True)`` then copies whatever follows the parsed
  MD in memory into the "reference", so the screen saw different reference
  bases (and dominant directions) every run. Such reads now use the FASTA, or
  are unusable without one, as in ``get_daf_positions`` and dedup.
* A position profiled as both C and G (reads' MD tags disagree) kept
  whichever hypothesis set iteration visited last, which depends on
  ``PYTHONHASHSEED``. Both are now counted and the base reported by more
  dominant-direction fibers is kept (tie: C).

Hash-seed and memory effects are per process, so the byte-identity checks run
the screen in subprocesses.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pysam
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_daf_mismatch_fastpath as fastpath  # noqa: E402

from fiberhmm.daf import snps  # noqa: E402
from fiberhmm.daf.aligned_arrays import md_disagrees_with_cigar  # noqa: E402
from fiberhmm.qc.core import _aligned_reference_pairs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
_SUFFIXES = (".bed", ".vcf", ".amplicons.tsv", ".json")
_KWARGS = dict(max_profile_sites=5000, min_depth=3, min_alt_fibers=3)

# Seed 313 has 1/17 reads with a short MD, 1/17 with a long MD, and one C
# site whose MD reads G on every 7th amplicon read (both hypotheses profiled).
_SEED = 313


@pytest.fixture(scope="module")
def hard_dataset(tmp_path_factory):
    return fastpath._write_dataset(
        tmp_path_factory.mktemp("snpdet"), _SEED, short_md=True)


def _reads(bam_path):
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as bam:
        return list(bam.fetch(until_eof=True))


def _fasta_truth(read, fasta):
    """(rpos, ref_codes, query_codes) of the M/=/X bases, reference from FASTA."""
    reference = fasta.fetch(read.reference_name).upper()
    sequence = read.query_sequence.upper()
    pairs = read.get_aligned_pairs(matches_only=True)
    return ([r for _q, r in pairs], [ord(reference[r]) for _q, r in pairs],
            [ord(sequence[q]) for q, _r in pairs])


def _bad_md_reads(bam_path):
    """Mapped reads whose MD is shorter or longer than the CIGAR's span."""
    return [read for read in _reads(bam_path)
            if not read.is_unmapped and md_disagrees_with_cigar(read)]


class _SpyRead:
    """Wraps a pysam read and records ``get_aligned_pairs`` calls."""

    def __init__(self, read):
        self._read = read
        self.with_seq_calls = 0

    def get_aligned_pairs(self, *args, **kwargs):
        if kwargs.get("with_seq"):
            self.with_seq_calls += 1
        return self._read.get_aligned_pairs(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._read, name)


def test_md_reconstruction_is_skipped_when_md_does_not_describe_the_cigar(hard_dataset):
    bam_path, _fasta = hard_dataset
    bad = _bad_md_reads(bam_path)
    assert len(bad) >= 10
    for read in bad:
        spy = _SpyRead(read)
        assert snps._aligned_pairs(spy) is None
        assert _aligned_reference_pairs(spy) is None
        assert spy.with_seq_calls == 0
    # A valid MD still goes through pysam's reconstruction.
    good = next(read for read in _reads(bam_path)
                if not read.is_unmapped and read.has_tag("MD")
                and not md_disagrees_with_cigar(read))
    spy = _SpyRead(good)
    assert snps._aligned_pairs(spy) is not None
    assert spy.with_seq_calls == 1


def test_md_disagreement_matches_the_encoder_pre_validation(hard_dataset):
    from fiberhmm.daf.encoder import md_matches_cigar

    bam_path, _fasta = hard_dataset
    kinds = Counter()
    for read in _reads(bam_path):
        if read.is_unmapped:
            continue
        assert md_disagrees_with_cigar(read) == (not md_matches_cigar(read))
        kinds[md_disagrees_with_cigar(read)] += 1
    assert kinds[True] >= 10 and kinds[False] >= 300


def test_bad_md_reads_use_the_fasta_or_are_unusable(hard_dataset):
    bam_path, fasta_path = hard_dataset
    bad = _bad_md_reads(bam_path)
    assert bad
    with pysam.FastaFile(str(fasta_path)) as fasta:
        for read in bad:
            assert snps._read_arrays(read) is None
            rpos, ref_codes, query_codes = snps._read_arrays(read, fasta)
            assert (rpos.tolist(), ref_codes.tolist(), query_codes.tolist()) \
                == _fasta_truth(read, fasta)
            pairs = _aligned_reference_pairs(read, fasta)
            reference = fasta.fetch(read.reference_name).upper()
            assert pairs and all(base == reference[r] for _q, r, base in pairs)


_DRIVER = r"""
import json, sys
from fiberhmm.daf import snps
bam, fasta, prefix, kwargs = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
for use_fasta in (False, True):
    payload = snps.call_opposite_conversion_snps(
        bam, reference_fasta=fasta if use_fasta else None, **kwargs)
    snps.write_snp_outputs(payload, f"{prefix}.fasta{int(use_fasta)}")
"""


def _run_screen(bam_path, fasta_path, prefix, hash_seed):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hash_seed)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")
    result = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(bam_path), str(fasta_path),
         str(prefix), json.dumps(_KWARGS)],
        capture_output=True, env=env, timeout=300,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return {
        f"fasta{flag}{suffix}": Path(f"{prefix}.fasta{flag}{suffix}").read_bytes()
        for flag in (0, 1) for suffix in _SUFFIXES
    }


def test_snp_screen_outputs_are_identical_across_processes_and_hash_seeds(
        hard_dataset, tmp_path):
    bam_path, fasta_path = hard_dataset
    prefix = tmp_path / "screen"  # same prefix every run: JSON paths match too
    runs = [_run_screen(bam_path, fasta_path, prefix, seed) for seed in (0, 0, 1, 2, 3)]
    for run in runs[1:]:
        for name, data in runs[0].items():
            assert run[name] == data, name
    payload = json.loads(runs[0]["fasta0.json"])
    # The data exercise both causes: unusable short/long-MD reads and a
    # resolved reference-base conflict.
    assert payload["accounting"]["unusable_alignment_records"] > 0
    assert payload["accounting"]["reference_base_conflict_sites"] == 1
    assert payload["n_called_snps"] >= 4


def test_reference_conflict_keeps_the_majority_md_base(hard_dataset):
    bam_path, fasta_path = hard_dataset
    for fasta in (None, str(fasta_path)):
        payload = snps.call_opposite_conversion_snps(
            str(bam_path), reference_fasta=fasta, **_KWARGS)
        assert payload["accounting"]["reference_base_conflict_sites"] == 1
        with pysam.FastaFile(str(fasta_path)) as handle:
            reference = handle.fetch("amp").upper()
        # The minority (G) reads claim a reference G at a C site; the kept
        # hypothesis agrees with the true reference at every reported site.
        for row in payload["site_distribution"] + payload["calls"]:
            if row["chrom"] == "amp":
                assert reference[row["position_0based"]] == row["reference"]
        positions = Counter(
            row["position_0based"] for row in payload["site_distribution"])
        assert max(positions.values()) == 1


def test_reference_conflict_tie_keeps_c():
    def stats(expected, opposite):
        return Counter({"expected_depth": expected, "opposite_depth": opposite})

    site_stats = {
        ("x", 5, "C"): stats(3, 1), ("x", 5, "G"): stats(2, 2),   # tie -> C
        ("x", 9, "C"): stats(1, 1), ("x", 9, "G"): stats(2, 1),   # G majority
        ("y", 2, "G"): stats(4, 0),                               # C unseen
    }
    conflicted = {"x": {5: ("G", "A", "GA"), 9: ("G", "A", "GA")},
                  "y": {2: ("G", "A", "GA")}}
    assert snps._resolve_reference_conflicts(conflicted, site_stats) == {
        ("x", 5, "G"), ("x", 9, "C"), ("y", 2, "C")}


def test_snp_screen_is_independent_of_read_order(hard_dataset, tmp_path):
    """Passes before the fix too (apart from the short-MD noise): the screen's
    counts, bottom-k profile sample and amplicon endpoints are order-free."""
    bam_path, fasta_path = hard_dataset
    reads = _reads(bam_path)
    random.Random(5).shuffle(reads)
    shuffled = tmp_path / "shuffled.bam"
    with pysam.AlignmentFile(str(bam_path), "rb", check_sq=False) as source:
        header = source.header
    with pysam.AlignmentFile(str(shuffled), "wb", header=header) as out:
        for read in reads:
            out.write(read)
    for fasta in (None, str(fasta_path)):
        outputs = []
        for index, path in enumerate((bam_path, shuffled)):
            payload = snps.call_opposite_conversion_snps(
                str(path), reference_fasta=fasta, **_KWARGS)
            written = snps.write_snp_outputs(payload, str(tmp_path / f"o{index}"))
            files = {key: Path(value).read_bytes()
                     for key, value in written["outputs"].items() if key != "json"}
            report = json.loads(Path(written["outputs"]["json"]).read_text())
            report.pop("input"), report.pop("outputs")
            outputs.append((files, report))
        assert outputs[0] == outputs[1]


def test_call_snp_screen_is_identical_across_worker_counts(hard_dataset, tmp_path):
    """fiberhmm-call runs the screen once, single-process, after dedup; the
    worker count and hash seed must not change its outputs or the calls."""
    bam_path, _fasta = hard_dataset
    sorted_bam = tmp_path / "sorted.bam"
    pysam.sort("-o", str(sorted_bam), str(bam_path))
    pysam.index(str(sorted_bam))
    results = []
    for workers, hash_seed in ((1, 0), (2, 1), (4, 2)):
        out_dir = tmp_path / f"c{workers}"
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = str(hash_seed)
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")
        result = subprocess.run(
            [sys.executable, "-m", "fiberhmm.cli.call", "-i", str(sorted_bam),
             "-o", str(out_dir / "out.bam"), "--enzyme", "dddb", "--dedup",
             "--daf-call-snps", "--daf-snp-min-depth", "3",
             "--daf-snp-min-alt-fibers", "3", "--no-qc", "--no-recall-nucs",
             "--min-read-length", "0", "-c", str(workers), "--io-threads", "1"],
            capture_output=True, env=env, timeout=600,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        prefix = out_dir / "qc" / "out.daf_snps"
        files = {suffix: Path(f"{prefix}{suffix}").read_bytes()
                 for suffix in _SUFFIXES if suffix != ".json"}
        report = json.loads(Path(f"{prefix}.json").read_text())
        report.pop("outputs")
        with pysam.AlignmentFile(str(out_dir / "out.bam"), check_sq=False) as bam:
            records = [read.to_string() for read in bam.fetch(until_eof=True)]
        results.append((files, report, records))
    assert results[0][1]["n_called_snps"] >= 4
    for other in results[1:]:
        assert other == results[0]
