import pytest

from fiberhmm.posteriors import writer


def test_is_hdf5_output_path_matches_supported_extensions():
    assert writer._is_hdf5_output_path("posteriors.h5")
    assert writer._is_hdf5_output_path("posteriors.hdf5")
    assert not writer._is_hdf5_output_path("posteriors.tsv.gz")


def test_resolve_writer_format_auto_detects_hdf5_extensions():
    assert writer._resolve_writer_format("posteriors.h5", "auto") == "hdf5"
    assert writer._resolve_writer_format("posteriors.hdf5", "auto") == "hdf5"
    assert writer._resolve_writer_format("posteriors.tsv.gz", "auto") == "tsv"
    assert writer._resolve_writer_format("posteriors.any", "tsv") == "tsv"


def test_unknown_writer_format_message_names_valid_formats():
    message = writer._unknown_writer_format_message("json")

    assert "Unknown posteriors format: 'json'" in message
    assert "Use 'hdf5' or 'tsv'" in message


def test_create_writer_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unknown posteriors format"):
        writer.create_writer("posteriors.out", format="unknown")
