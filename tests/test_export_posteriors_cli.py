"""Failure-path coverage for posterior export CLI helpers."""

import sys

import pytest

from fiberhmm.cli import export_posteriors
from fiberhmm.posteriors import tsv_backend


def test_cli_infers_dddb_mode_instead_of_defaulting_to_pacbio(
    monkeypatch, tmp_path, capsys
):
    captured = {}

    monkeypatch.setattr(
        export_posteriors,
        "load_model_with_metadata",
        lambda *args, **kwargs: (object(), 3, "pacbio-fiber"),
    )
    monkeypatch.setattr(
        export_posteriors,
        "export_posteriors",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fiberhmm-posteriors",
            "-i", "input.bam",
            "-o", str(tmp_path / "posteriors.tsv.gz"),
            "--enzyme", "dddb",
        ],
    )

    export_posteriors.main()

    assert captured["mode_override"] == "daf"
    assert "--enzyme/--seq selects 'daf'" in capsys.readouterr().err


def test_export_posteriors_tsv_closes_writer_when_region_processing_fails(
    monkeypatch, tmp_path
):
    instances = []

    class FakeTSVWriter:
        def __init__(self, output_path, **kwargs):
            self.output_path = output_path
            self.closed = False
            self.close_count = 0
            instances.append(self)

        def write_fiber(self, **kwargs):
            raise AssertionError("region processing should fail before writes")

        def close(self):
            self.closed = True
            self.close_count += 1
            return 0

    def fail_process_regions(*args, **kwargs):
        raise RuntimeError("region processing failed")

    monkeypatch.setattr(tsv_backend, "PosteriorsTSVWriter", FakeTSVWriter)
    monkeypatch.setattr(
        export_posteriors,
        "load_model_with_metadata",
        lambda *args, **kwargs: (object(), 3, "pacbio-fiber"),
    )
    monkeypatch.setattr(
        export_posteriors,
        "_get_genome_regions",
        lambda *args, **kwargs: [("chr1", 0, 100)],
    )
    monkeypatch.setattr(export_posteriors, "_process_regions", fail_process_regions)

    with pytest.raises(RuntimeError, match="region processing failed"):
        export_posteriors.export_posteriors_tsv(
            input_bam="input.bam",
            model_path="model.json",
            output_path=str(tmp_path / "posteriors.tsv.gz"),
            n_cores=1,
            verbose=False,
        )

    assert instances
    assert instances[0].closed
    assert instances[0].close_count == 1
