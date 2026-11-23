"""DddA m5C readers see FiberHMM calls after ``ft fire`` moved MA into Ma.

FiberHMM 3.3 writes calls to ``MA`` only; fibertools >= 0.13 ``ft fire``
moves every ``MA`` group into ``Ma`` (molecular frame). The m5C readers
(observation collection, initial-MSP/nucleosome extraction, the tag-m5c
structure preflight) must read the calls from ``Ma`` exactly as from ``MA``:
before this fix a read whose calls were in ``Ma`` was treated as uncalled
(no observations) and tag-m5c rejected the BAM.

The synthetic checks always run; the real ``fiberhmm-call | ft fire`` round
trip needs fibertools-rs >= 0.13 and skips cleanly without it.
"""
import array
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pysam
import pytest

from fiberhmm.cli.tag_m5c import _preflight_input
from fiberhmm.daf import m5c

REPO = Path(__file__).resolve().parents[1]
LENGTH = 400
REF = ("ACGTTCGAAC" * 40)[:LENGTH]

HEADER = {
    "HD": {"VN": "1.6", "SO": "coordinate"},
    "SQ": [{"SN": "chr1", "LN": 1000}],
    "PG": [{"ID": "fiberhmm-call", "PN": "fiberhmm-call", "VN": "3.3.0",
            "DS": "FiberHMM fused apply+recall; coord=molecular; mode=daf enzyme=ddda"}],
    "CO": ["fiberhmm:coord=molecular"],
}
# Molecular frame: nucleosome at 1-150, MSP at 151-250 (1-based start-length pairs).
MA_CALLS = f"{LENGTH};nuc.Q:1-150;msp.:151-100"


class _Fasta:
    def fetch(self, chrom, start, end):
        return ("N" * 100 + REF + "N" * 600)[start:end]


def _read(path, reverse, tag):
    header = pysam.AlignmentHeader.from_dict(HEADER)
    r = pysam.AlignedSegment(header)
    r.query_name = f"r_{tag}_{int(reverse)}"
    r.reference_id = 0
    r.reference_start = 100
    r.mapping_quality = 60
    r.cigarstring = f"{LENGTH}M"
    r.flag = 16 if reverse else 0
    r.query_sequence = REF.replace("C", "Y")
    r.set_tag(tag, MA_CALLS, value_type="Z")
    r.set_tag("AQ" if tag == "MA" else "Aq", array.array("B", [200]))
    if path is not None:
        with pysam.AlignmentFile(str(path), "wb", header=header) as out:
            out.write(r)
    return r


@pytest.mark.parametrize("reverse", [False, True])
def test_ma_and_fibertools_ma_give_the_same_m5c_inputs(reverse):
    ma, fibertools = _read(None, reverse, "MA"), _read(None, reverse, "Ma")
    want = m5c.collect_read_observations(ma, _Fasta(), input_molecular_frame=True)
    assert want, "the MA read must yield observations"
    got = m5c.collect_read_observations(fibertools, _Fasta(), input_molecular_frame=True)
    assert got == want
    assert (m5c._nucleosome_intervals_seq(fibertools, True)
            == m5c._nucleosome_intervals_seq(ma, True))
    assert (m5c._initial_msp_intervals_seq(fibertools, True)
            == m5c._initial_msp_intervals_seq(ma, True))
    # Ma is molecular: a reverse read's nucleosome is at the SEQ end.
    expected_nuc = [(LENGTH - 150, LENGTH)] if reverse else [(0, 150)]
    assert m5c._nucleosome_intervals_seq(fibertools, True) == expected_nuc


def test_stale_legacy_tags_do_not_replace_fibertools_ma_calls():
    read = _read(None, False, "Ma")
    read.set_tag("as", array.array("I", [0]))
    read.set_tag("al", array.array("I", [10]))
    read.set_tag("ns", array.array("I", [300]))
    read.set_tag("nl", array.array("I", [50]))
    assert m5c._initial_msp_intervals_seq(read, True) == [(150, 250)]
    assert m5c._nucleosome_intervals_seq(read, True) == [(0, 150)]


def test_tag_m5c_preflight_accepts_fibertools_ma(tmp_path):
    path = tmp_path / "fired.bam"
    _read(path, False, "Ma")
    _preflight_input(str(path))  # accepted


def test_paired_deam_coverage_is_read_from_fibertools_ma():
    read = _read(None, False, "Ma")
    read.set_tag("Ma", MA_CALLS + f";deam+:1-{LENGTH}", value_type="Z")
    assert m5c._deam_coverage(read) == {"+": [(0, LENGTH)]}
    c_obs, g_obs, cross = m5c.collect_read_m5c_channels(
        read, _Fasta(), min_deaminations=1, input_molecular_frame=True)
    assert cross and c_obs and not g_obs


# --- real fiberhmm-call (MA only) -> ft fire (Ma) -> tag-m5c / call-m5c ---------

def _ft_binary():
    path = os.environ.get("FIBERHMM_TEST_FT") or shutil.which("ft")
    if not path:
        return None
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
    if not match or (int(match.group(1)), int(match.group(2))) < (0, 13):
        return None
    return path


FT = _ft_binary()


@pytest.mark.fibertools
@pytest.mark.skipif(FT is None and os.environ.get("FIBERHMM_REQUIRE_FT") != "1",
                    reason="fibertools-rs >= 0.13 (ft) not installed")
def test_call_then_ft_fire_then_m5c_matches_the_ma_input(tmp_path):
    assert FT is not None
    env = dict(os.environ, HOME=str(tmp_path), PYTHONPATH=str(REPO), FIBERHMM_NO_UPDATE_CHECK="1")

    def run(*args):
        proc = subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env)
        assert proc.returncode == 0, proc.stderr[-2000:]
        return proc.stderr

    demo = tmp_path / "demo"
    run(str(REPO / "docs" / "examples" / "make_demo_data.py"), str(demo))
    run("-m", "fiberhmm.cli.daf_encode", "-i", str(demo / "ddda.bam"), "-o", str(tmp_path / "enc.bam"))
    called = tmp_path / "call.bam"
    run("-m", "fiberhmm.cli.call", "-i", str(tmp_path / "enc.bam"), "-o", str(called),
        "--enzyme", "ddda", "-c", "1", "--no-qc")
    fired = tmp_path / "fire.bam"
    proc = subprocess.run([FT, "fire", "-t", "1", str(called), str(fired)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    pysam.index(str(called))
    pysam.index(str(fired))
    with pysam.AlignmentFile(str(fired)) as bam:
        records = list(bam)
    assert records and not any(r.has_tag("MA") for r in records)
    assert all(r.has_tag("Ma") for r in records)

    ref = str(demo / "ref.fa")
    summaries, beds = {}, {}
    for label, path in (("ma", called), ("ma_fired", fired)):
        log = run("-m", "fiberhmm.cli.tag_m5c", "-i", str(path), "-o", str(tmp_path / f"{label}.m5c.bam"),
                  "-r", ref, "--enzyme", "ddda")
        summaries[label] = [line for line in log.splitlines() if "tagged_reads=" in line]
        bed = tmp_path / f"{label}.bed"
        run("-m", "fiberhmm.cli.call_m5c", "-i", str(path), "-r", ref, "-o", str(bed),
            "--region", "chrDemo", "--enzyme", "ddda")
        beds[label] = bed.read_text()
    assert summaries["ma"] and summaries["ma"] == summaries["ma_fired"]
    assert "tagged_reads=0" not in summaries["ma"][0]
    assert beds["ma"] == beds["ma_fired"]

    from fiberhmm.inference.tf_recaller import read_cpg_intervals

    def islands(path):
        with pysam.AlignmentFile(str(path)) as bam:
            return {r.query_name: read_cpg_intervals(r) for r in bam}

    assert islands(tmp_path / "ma.m5c.bam") == islands(tmp_path / "ma_fired.m5c.bam")
