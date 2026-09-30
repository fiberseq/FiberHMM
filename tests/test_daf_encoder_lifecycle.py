from __future__ import annotations

import pytest

from fiberhmm.daf import encoder


class _FakeHandle:
    def __init__(self):
        self.header = {}
        self.closed = False

    def close(self):
        self.closed = True


def test_daf_encode_closes_input_and_reference_when_md_check_fails(monkeypatch):
    handles = {
        "fasta": _FakeHandle(),
        "bam": _FakeHandle(),
    }

    monkeypatch.setattr(encoder.pysam, "FastaFile", lambda *args, **kwargs: handles["fasta"])
    monkeypatch.setattr(
        encoder.pysam,
        "AlignmentFile",
        lambda *args, **kwargs: handles["bam"],
    )

    def fail_md_check(*args, **kwargs):
        raise RuntimeError("md check failed")

    monkeypatch.setattr(encoder, "_check_md_tag", fail_md_check)

    with pytest.raises(RuntimeError, match="md check failed"):
        encoder.process_bam_daf_encode(
            "input.bam",
            "output.bam",
            reference="reference.fa",
        )

    assert handles["bam"].closed
    assert handles["fasta"].closed


def test_aligned_pairs_from_fasta_fetches_reference_span_once():
    class FakeRead:
        reference_name = "chr1"
        reference_start = 100
        reference_end = 105

        def get_aligned_pairs(self):
            return [(0, 100), (1, 101), (2, None), (3, 104)]

    class FakeFasta:
        def __init__(self):
            self.fetches = []

        def fetch(self, chrom, start, end):
            self.fetches.append((chrom, start, end))
            assert (chrom, start, end) == ("chr1", 100, 105)
            return "ACGTA"

    fasta = FakeFasta()

    assert encoder._aligned_pairs_from_fasta(FakeRead(), fasta) == [
        (0, 100, "A"),
        (1, 101, "C"),
        (2, None, None),
        (3, 104, "A"),
    ]
    assert fasta.fetches == [("chr1", 100, 105)]


def _write_md_bam(path, n_reads=15):
    """Small coordinate-sorted BAM of CT-deaminated reads carrying MD tags."""
    import pysam

    ref = ("ACGTCAGCTA" * 200)
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "chr1", "LN": len(ref)}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as sink:
        for i in range(n_reads):
            start = i * 10
            refseg = ref[start:start + 1200]
            # Deaminate every C -> T (CT flavour); MD records the ref C.
            query = refseg.replace("C", "T")
            md, run = [], 0
            for r, q in zip(refseg, query):
                if r == q:
                    run += 1
                else:
                    md.append(f"{run}{r}")
                    run = 0
            md.append(str(run))
            read = pysam.AlignedSegment(sink.header)
            read.query_name = f"r{i}"
            read.query_sequence = query
            read.flag = 0
            read.reference_id = 0
            read.reference_start = start
            read.mapping_quality = 60
            read.cigartuples = [(0, len(query))]
            read.query_qualities = pysam.qualitystring_to_array("I" * len(query))
            read.set_tag("MD", "".join(md))
            sink.write(read)


def test_daf_encode_reads_stdin_without_seeking(tmp_path):
    """Regression: ``fiberhmm-daf-encode -i -`` crashed with
    'seek not available in streams' because the MD peek rewound the input.
    Stdin output must match file-input output record for record."""
    import os
    import subprocess
    import sys

    import pysam

    src = tmp_path / "raw.bam"
    _write_md_bam(src)
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK="1")
    cmd = [sys.executable, "-m", "fiberhmm.cli.daf_encode", "--io-threads", "1"]

    from_file = tmp_path / "from_file.bam"
    subprocess.run(cmd + ["-i", str(src), "-o", str(from_file)],
                   check=True, env=env, capture_output=True)
    from_stdin = tmp_path / "from_stdin.bam"
    with open(src, "rb") as handle:
        proc = subprocess.run(cmd + ["-i", "-", "-o", str(from_stdin)],
                              stdin=handle, env=env, capture_output=True)
    assert proc.returncode == 0, proc.stderr.decode()[-2000:]

    def records(path):
        with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
            return [(r.query_name, r.query_sequence,
                     r.get_tag("st") if r.has_tag("st") else None)
                    for r in bam]

    file_records = records(from_file)
    assert len(file_records) == 15
    assert any("Y" in seq for _, seq, _ in file_records)
    assert records(from_stdin) == file_records


def _sha256(path):
    import hashlib
    from pathlib import Path

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hidden_leftovers(directory):
    return [p.name for p in directory.iterdir()
            if p.name.startswith(".") or ".sorting" in p.name]


def test_daf_encode_finalization_failure_keeps_previous_output(tmp_path):
    """A failed sort/index must not replace (or corrupt) an earlier output.

    Regression: daf-encode wrote its output in place and sorted/indexed the
    final file, so a failure after encoding left a half-published BAM next to
    a stale index describing the previous content."""
    from unittest.mock import patch

    import pysam

    src = tmp_path / "raw.bam"
    _write_md_bam(src)
    out = tmp_path / "encoded.bam"
    out.write_bytes(src.read_bytes())
    pysam.index(str(out))
    bam_digest, bai_digest = _sha256(out), _sha256(str(out) + ".bai")

    with patch.object(encoder, "_sort_and_index_bam",
                      side_effect=OSError("injected finalization failure")):
        with pytest.raises(OSError, match="injected"):
            encoder.process_bam_daf_encode(str(src), str(out), io_threads=1)
    assert _sha256(out) == bam_digest
    assert _sha256(str(out) + ".bai") == bai_digest
    assert _hidden_leftovers(tmp_path) == []


def test_daf_encode_publishes_sorted_bam_with_fresh_index(tmp_path):
    import pysam

    src = tmp_path / "raw.bam"
    _write_md_bam(src)
    out = tmp_path / "encoded.bam"
    out.write_bytes(b"stale")
    (tmp_path / "encoded.bam.bai").write_bytes(b"stale-index")

    summary = encoder.process_bam_daf_encode(str(src), str(out), io_threads=1)
    assert summary["encoded"] == 15
    with pysam.AlignmentFile(str(out), "rb") as bam:
        assert bam.has_index()
        reads = list(bam.fetch("chr1"))
    assert len(reads) == 15
    assert all(read.get_tag("st") == "CT" for read in reads)
    assert _hidden_leftovers(tmp_path) == []


def test_daf_encode_streams_to_stdout(tmp_path):
    """``-o -`` streams the BAM (no temporary, no sort/index) with the same
    records as a file output."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import pysam

    src = tmp_path / "raw.bam"
    _write_md_bam(src)
    env = dict(os.environ, FIBERHMM_NO_UPDATE_CHECK="1")
    cmd = [sys.executable, "-m", "fiberhmm.cli.daf_encode", "--io-threads", "1",
           "-i", str(src)]
    from_file = tmp_path / "from_file.bam"
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(cmd + ["-o", str(from_file)], check=True, env=env,
                   capture_output=True, cwd=repo)
    streamed = tmp_path / "streamed.bam"
    with open(streamed, "wb") as sink:
        proc = subprocess.run(cmd + ["-o", "-"], stdout=sink,
                              stderr=subprocess.PIPE, env=env, cwd=repo)
    assert proc.returncode == 0, proc.stderr.decode()[-2000:]

    def records(path):
        with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
            return [(r.query_name, r.query_sequence, r.get_tag("st"))
                    for r in bam.fetch(until_eof=True)]

    assert records(streamed) == records(from_file)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "from_file.bam", "from_file.bam.bai", "raw.bam", "streamed.bam"]
