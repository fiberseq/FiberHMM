"""fiberhmm-posteriors failure handling, determinism and provenance.

Regression tests for the 3.0 CLI audit, package P4: worker errors were
swallowed (rc 0, empty export, error on stdout; H7); a failed export left a
partial file at the final path (M8); row order followed worker timing (M9);
``--skip-scaffolds``/``--streaming``/``--chunk-size``/``--io-threads`` were
accepted and ignored (M10); the export recorded too little provenance (M11);
``-c 0`` ran serially and ``-v`` was always on (L3); ``-o x.h5`` without h5py
was a raw ``ModuleNotFoundError`` (L4).
"""
from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

import pysam
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fiberhmm.cli import export_posteriors as ep  # noqa: E402
from test_export_posteriors_frames import (  # noqa: E402
    _hia5_read,
    _model,
)

CHROM_LEN = 20_000


def _header(scaffold=False):
    sq = [{"LN": CHROM_LEN, "SN": "chr1"}]
    if scaffold:
        sq.append({"LN": CHROM_LEN, "SN": "chrUn_scaffold"})
    return {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": sq}


def _bam(path, layout, *, scaffold=False, index=True, seed=11):
    """``layout``: [(reference_id, start)] -> one Hia5 PacBio read each."""
    rng = random.Random(seed)
    header = pysam.AlignmentHeader.from_dict(_header(scaffold))
    reads = []
    for n, (ref_id, start) in enumerate(layout):
        read = _hia5_read(header, f"r{n:03d}", start, n % 2 == 1, rng)
        read.reference_id = ref_id
        reads.append(read)
    reads.sort(key=lambda r: (r.reference_id, r.reference_start))
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for read in reads:
            out.write(read)
    if index:
        pysam.index(str(path))
    return str(path)


def _lines(path):
    with gzip.open(path, "rt") as handle:
        return handle.read().splitlines()


def _metadata(path):
    first = _lines(path)[0]
    assert first.startswith("#metadata:")
    return json.loads(first[len("#metadata:"):])


def _run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["fiberhmm-posteriors", *map(str, argv)])
    try:
        ep.main()
    except SystemExit as exit:
        return exit.code or 0
    return 0


# ---------------------------------------------------------------------------
# H7: failures propagate
# ---------------------------------------------------------------------------

def test_unindexed_input_fails_without_writing(tmp_path, monkeypatch, capsys):
    bam = _bam(tmp_path / "noidx.bam", [(0, 1_000)], index=False)
    out = tmp_path / "p.tsv.gz"
    code = _run_cli(monkeypatch, "-i", bam, "--enzyme", "hia5", "--seq", "pacbio",
                    "-o", out, "-c", "2")
    captured = capsys.readouterr()
    assert code != 0
    assert "no index" in captured.err and "--streaming" in captured.err
    assert "error" not in captured.out.lower()
    assert not out.exists()


def test_worker_failure_is_raised_not_skipped(tmp_path):
    bam = _bam(tmp_path / "in.bam", [(0, 1_000)])
    params = ep._worker_params_for("pacbio-fiber", 3, 10, 128)
    # The worker initializer cannot load the model: every region fails.
    with pytest.raises(RuntimeError):
        ep._process_regions([("chr1", 0, 10_000), ("chr1", 10_000, 20_000)], bam,
                            str(tmp_path / "missing_model.json"), params,
                            n_cores=2, verbose=False,
                            result_callback=lambda chrom, results: None)


def test_serial_region_failure_names_the_region(tmp_path, monkeypatch):
    bam = _bam(tmp_path / "in.bam", [(0, 1_000)])
    model_path, _model_obj, k = _model("hia5", "pacbio")
    params = ep._worker_params_for("pacbio-fiber", k, 10, 128)

    def broken(args):
        raise OSError("fetch called on bamfile without index")

    monkeypatch.setattr(ep, "_process_region_worker", broken)
    with pytest.raises(ep.PosteriorExportError, match=r"chr1:0-10000 failed"):
        ep._process_regions([("chr1", 0, 10_000)], bam, model_path, params,
                            n_cores=1, verbose=False,
                            result_callback=lambda chrom, results: None)


# ---------------------------------------------------------------------------
# M8: nothing partial at the final path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["p.tsv.gz", "p.tsv", "p.h5"])
def test_failed_export_keeps_the_previous_file(tmp_path, monkeypatch, name):
    if name.endswith(".h5"):
        pytest.importorskip("h5py")
    bam = _bam(tmp_path / "in.bam", [(0, 1_000)])
    model_path, _m, _k = _model("hia5", "pacbio")
    out = tmp_path / name
    out.write_bytes(b"earlier export")

    def fail(*args, **kwargs):
        raise RuntimeError("region processing failed")

    monkeypatch.setattr(ep, "_process_regions", fail)
    with pytest.raises(RuntimeError, match="region processing failed"):
        ep.export_posteriors(bam, model_path, str(out), n_cores=1, verbose=False,
                             mode_override="pacbio-fiber")
    assert out.read_bytes() == b"earlier export"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        ["in.bam", "in.bam.bai", name])


def test_cli_failure_exits_nonzero_on_stderr(tmp_path, monkeypatch, capsys):
    bam = _bam(tmp_path / "in.bam", [(0, 1_000)])
    out = tmp_path / "p.tsv.gz"

    def fail(*args, **kwargs):
        raise ep.PosteriorExportError("region chr1:0-5000000 failed: boom")

    monkeypatch.setattr(ep, "_process_regions", fail)
    code = _run_cli(monkeypatch, "-i", bam, "--enzyme", "hia5", "--seq", "pacbio",
                    "-o", out, "-c", "1")
    captured = capsys.readouterr()
    assert code == 1
    assert "region chr1:0-5000000 failed: boom" in captured.err
    assert captured.out == ""
    assert not out.exists()


# ---------------------------------------------------------------------------
# M9: deterministic order
# ---------------------------------------------------------------------------

def test_row_order_is_region_order_for_any_core_count(tmp_path):
    # A heavy first region and light later ones: with completion-order
    # writes the later regions overtake it.
    layout = [(0, 100 + 10 * i) for i in range(40)]
    layout += [(0, 5_000 + 4_000 * j) for j in range(4)]
    bam = _bam(tmp_path / "in.bam", layout)
    model_path, _m, _k = _model("hia5", "pacbio")
    runs = []
    for cores in (1, 2):
        out = tmp_path / f"c{cores}.tsv.gz"
        ep.export_posteriors(bam, model_path, str(out), n_cores=cores,
                             region_size=4_000, verbose=False,
                             mode_override="pacbio-fiber")
        runs.append(_lines(out))
    assert runs[0] == runs[1]
    starts = [int(line.split("\t")[2]) for line in runs[0] if not line.startswith("#")]
    assert starts == sorted(starts) and len(starts) == len(layout)


# ---------------------------------------------------------------------------
# M10: options do what they say, or are refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("flag", ["--chunk-size", "--io-threads"])
def test_unimplemented_parallel_options_are_refused(tmp_path, monkeypatch, capsys, flag):
    monkeypatch.setattr(ep, "export_posteriors",
                        lambda **kwargs: pytest.fail("must not run"))
    code = _run_cli(monkeypatch, "-i", "in.bam", "--enzyme", "hia5", "--seq", "pacbio",
                    "-o", tmp_path / "p.tsv.gz", flag, "2")
    assert code == 2 and flag in capsys.readouterr().err


def test_skip_scaffolds_and_streaming(tmp_path):
    layout = [(0, 1_000), (0, 6_000), (1, 2_000)]
    indexed = _bam(tmp_path / "in.bam", layout, scaffold=True)
    plain = _bam(tmp_path / "plain.bam", layout, scaffold=True, index=False)
    model_path, _m, _k = _model("hia5", "pacbio")

    def names(out):
        return [line.split("\t")[0] for line in _lines(out) if not line.startswith("#")]

    out = tmp_path / "all.tsv.gz"
    ep.export_posteriors(indexed, model_path, str(out), n_cores=1, verbose=False,
                         mode_override="pacbio-fiber")
    everything = names(out)
    assert len(everything) == 3

    out = tmp_path / "main.tsv.gz"
    ep.export_posteriors(indexed, model_path, str(out), n_cores=1, verbose=False,
                         mode_override="pacbio-fiber", skip_scaffolds=True)
    assert names(out) == everything[:2]

    out = tmp_path / "stream.tsv.gz"
    ep.export_posteriors(plain, model_path, str(out), n_cores=1, verbose=False,
                         mode_override="pacbio-fiber", streaming=True)
    assert _lines(out)[1:] == _lines(tmp_path / "all.tsv.gz")[1:]

    out = tmp_path / "stream_main.tsv.gz"
    ep.export_posteriors(plain, model_path, str(out), n_cores=1, verbose=False,
                         mode_override="pacbio-fiber", streaming=True,
                         skip_scaffolds=True)
    assert names(out) == everything[:2]


def test_cli_passes_selection_options(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(ep, "load_model_with_metadata",
                        lambda *a, **k: (object(), 3, "pacbio-fiber"))
    monkeypatch.setattr(ep, "export_posteriors", lambda **kwargs: captured.update(kwargs))
    assert _run_cli(monkeypatch, "-i", "in.bam", "--enzyme", "hia5", "--seq", "pacbio",
                    "-o", tmp_path / "p.tsv.gz", "--skip-scaffolds", "--streaming",
                    "-c", "0") == 0
    import os

    assert captured["skip_scaffolds"] is True
    assert captured["streaming"] is True
    assert captured["n_cores"] == (os.cpu_count() or 1)  # L3: 0 = all CPUs
    assert captured["verbose"] is False                   # L3: -v is honoured
    assert captured["provenance"]["enzyme"] == "hia5"
    assert captured["provenance"]["platform"] == "pacbio"


# ---------------------------------------------------------------------------
# M11: provenance
# ---------------------------------------------------------------------------

def test_export_records_provenance(tmp_path):
    from fiberhmm.identity import file_sha256

    bam = _bam(tmp_path / "in.bam", [(0, 1_000)])
    model_path, _m, _k = _model("hia5", "pacbio")
    out = tmp_path / "p.tsv.gz"
    ep.export_posteriors(bam, model_path, str(out), n_cores=1, verbose=False,
                         mode_override="pacbio-fiber", prob_threshold=200,
                         extraction={"daf_run_mask": (0, "keep-one")},
                         provenance={"enzyme": "hia5", "platform": "pacbio"})
    meta = _metadata(out)
    assert meta["mode"] == "pacbio-fiber"          # existing fields unchanged
    assert meta["format_version"] == 1
    assert meta["model_sha256"] == file_sha256(model_path)
    assert meta["prob_threshold"] == 200
    assert meta["enzyme"] == "hia5" and meta["platform"] == "pacbio"
    assert meta["daf_mask_runs"] == 0 and meta["daf_run_policy"] == "keep-one"
    assert meta["filter_chimeras"] is True


def test_hdf5_export_records_provenance(tmp_path):
    h5py = pytest.importorskip("h5py")
    bam = _bam(tmp_path / "in.bam", [(0, 1_000)])
    model_path, _m, _k = _model("hia5", "pacbio")
    out = tmp_path / "p.h5"
    n = ep.export_posteriors(bam, model_path, str(out), n_cores=1, verbose=False,
                             mode_override="pacbio-fiber", prob_threshold=200)
    assert n == 1
    with h5py.File(out, "r") as handle:
        assert handle.attrs["prob_threshold"] == 200
        assert len(handle.attrs["model_sha256"]) == 64
        assert handle["chr1"].attrs["n_fibers"] == 1


# ---------------------------------------------------------------------------
# L4 and path aliasing
# ---------------------------------------------------------------------------

def test_hdf5_without_h5py_is_a_clear_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "h5py", None)
    code = _run_cli(monkeypatch, "-i", "in.bam", "--enzyme", "hia5", "--seq", "pacbio",
                    "-o", tmp_path / "p.h5")
    assert code == 2
    assert 'pip install "fiberhmm[posteriors]"' in capsys.readouterr().err


def test_output_cannot_be_the_input(tmp_path, monkeypatch, capsys):
    bam = _bam(tmp_path / "in.bam", [(0, 1_000)])
    before = Path(bam).read_bytes()
    code = _run_cli(monkeypatch, "-i", bam, "--enzyme", "hia5", "--seq", "pacbio",
                    "-o", bam, "--format", "tsv")
    assert code == 2 and "same file as --input" in capsys.readouterr().err
    assert Path(bam).read_bytes() == before


def test_output_cannot_be_the_bundled_model(tmp_path, monkeypatch, capsys):
    import fiberhmm.models as models

    bam = _bam(tmp_path / "in.bam", [(0, 1_000)])
    bundled = tmp_path / "bundled.json"
    bundled.write_text("{}")
    monkeypatch.setattr(models, "get_model_path", lambda *a, **k: str(bundled))
    code = _run_cli(monkeypatch, "-i", bam, "--enzyme", "hia5", "--seq", "pacbio",
                    "-o", bundled, "--format", "tsv")
    assert code == 2 and "same file as --model" in capsys.readouterr().err
    assert bundled.read_text() == "{}"
