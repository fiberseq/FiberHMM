import pytest

from fiberhmm.posteriors import writer


def test_resolve_writer_format_auto_detects_hdf5_extensions():
    assert writer._resolve_writer_format("posteriors.h5", "auto") == "hdf5"
    assert writer._resolve_writer_format("posteriors.hdf5", "auto") == "hdf5"
    assert writer._resolve_writer_format("posteriors.tsv.gz", "auto") == "tsv"
    assert writer._resolve_writer_format("posteriors.any", "tsv") == "tsv"


def test_create_writer_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unknown posteriors format"):
        writer.create_writer("posteriors.out", format="unknown")
