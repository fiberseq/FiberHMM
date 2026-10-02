"""Bounded, conditional platform sniffing (3.0 verify MEDIUM).

Regressions for the Codex verification pass on b006409.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pysam
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(module, *args, stdin=None, timeout=300):
    env = dict(os.environ)
    env.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=REPO_ROOT, capture_output=True, timeout=timeout, env=env,
        stdin=stdin,
    )


def _bundled(name):
    from fiberhmm.models import _bundled_model_path

    return _bundled_model_path(name)


def _pg_ds(path, program):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        records = [
            record for record in bam.header.to_dict().get("PG", [])
            if str(record.get("PN", "")).startswith(program)
        ]
    assert records, f"no {program} @PG record"
    return records[-1].get("DS", "")


def _ds_tokens(ds):
    return dict(
        token.split("=", 1) for token in ds.split() if "=" in token
    )


def _ma_tags(path):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        return [
            (read.query_name, read.get_tag("MA") if read.has_tag("MA") else None)
            for read in bam.fetch(until_eof=True)
        ]


def _declare(source, target, declaration):
    from fiberhmm.io.bam_header import append_chemistry

    with pysam.AlignmentFile(str(source), "rb", check_sq=False) as bam:
        header = append_chemistry(bam.header, declaration)
        with pysam.AlignmentFile(str(target), "wb", header=header) as out:
            for read in bam.fetch(until_eof=True):
                out.write(read)
    pysam.index(str(target))
    return target


# ---------------------------------------------------------------------------
# 4. bounded platform sniffing
# ---------------------------------------------------------------------------

class _FakeBam:
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000}],
        "RG": [{"ID": "r", "PL": "PACBIO"}]})

    def __init__(self, seen, tagged=False):
        self.seen = seen
        self.tagged = tagged

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def fetch(self, **kwargs):
        for index in range(1000):
            self.seen.append(index)
            read = pysam.AlignedSegment(self.header)
            read.query_sequence = "CCYACG"
            read.flag = 0x100 if (self.tagged and index % 2) else 0
            if self.tagged:
                read.set_tag("MM", "A+a.;T-a.;")  # PacBio-style, like the @RG
            yield read


@pytest.mark.parametrize("tagged", [False, True])
def test_platform_sniff_caps_records_consumed(tagged):
    from fiberhmm.cli.common import sniff_sequencing_platform

    seen = []
    with patch("pysam.AlignmentFile", return_value=_FakeBam(seen, tagged)):
        result = sniff_sequencing_platform("input.bam", n_reads=3)
    assert len(seen) <= 3
    assert result.platform == "pacbio"


def test_platform_sniff_skips_reads_when_settled(monkeypatch):
    from types import SimpleNamespace

    from fiberhmm.cli import common

    calls = []

    def spy(path, n_reads=common.PLATFORM_SNIFF_READS, **kwargs):
        calls.append(kwargs.get("inspect_reads", True))
        return common.PlatformEvidence(platform="pacbio", source="header")

    monkeypatch.setattr(common, "sniff_sequencing_platform", spy)
    # Explicit --seq for Hia5: the reads are still inspected, because an
    # explicit --seq is checked against their MM specs.
    common.resolve_platform_argument(
        SimpleNamespace(enzyme="hia5", seq="pacbio"), "x.bam", tool="t")
    # DAF enzymes: the platform never changes the observation mode.
    common.resolve_platform_argument(
        SimpleNamespace(enzyme="ddda", seq=None), "x.bam", tool="t")
    common.resolve_platform_argument(
        SimpleNamespace(enzyme="dddb", seq=None), "x.bam", tool="t")
    # Hia5 without --seq still inspects reads.
    common.resolve_platform_argument(
        SimpleNamespace(enzyme="hia5", seq=None), "x.bam", tool="t")
    assert calls == [True, False, False, True]


def test_platform_sniff_declaration_settles_without_reading():
    from fiberhmm.cli.common import sniff_sequencing_platform
    from fiberhmm.io.bam_header import append_chemistry

    seen = []
    fake = _FakeBam(seen, tagged=True)
    fake.header = append_chemistry(_FakeBam.header, {
        "assay": "fiber-seq", "enzyme": "hia5", "platform": "pacbio",
        "mode": "pacbio-fiber"})
    with patch("pysam.AlignmentFile", return_value=fake):
        evidence = sniff_sequencing_platform("input.bam")
    assert seen == []
    assert evidence.platform == "pacbio"
    assert "declaration" in evidence.source
