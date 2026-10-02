"""Publishing a BAM with its index, and repairing bigBeds (audit M21-M23).

``fiberhmm-tag-consensus`` and ``fiberhmm-utils ma-types`` replaced the BAM
and then its index in two unguarded steps: a failure in between left the new
BAM beside the old index. ``fiberhmm-utils fix-bigbed`` reported missing files
and converter failures but exited 0, and published through ``shutil.move``
from the system TMPDIR (not atomic across filesystems).
"""
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pysam
import pytest

from test_tf_family_tags import _make_input, _write_assignments


def _fail_index_publication(monkeypatch, final_index):
    """Make the rename that installs ``final_index`` fail (once)."""
    real_replace = os.replace
    state = {"failed": False}

    def replace(source, destination, *args, **kwargs):
        if not state["failed"] and os.path.abspath(str(destination)) == os.path.abspath(final_index) \
                and os.path.basename(str(source)).startswith("."):
            state["failed"] = True
            raise OSError("injected failure while installing the index")
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    return state


def test_tag_consensus_failed_index_install_keeps_previous_bam_and_index(tmp_path, monkeypatch):
    from fiberhmm.cli.tag_families import tag_tf_families

    source = tmp_path / "source.bam"
    output = tmp_path / "family.bam"
    assignments = tmp_path / "assignments.tsv"
    _make_input(source)
    _write_assignments(assignments)
    tag_tf_families(source, output, assignments)
    before_bam = output.read_bytes()
    before_index = (tmp_path / "family.bam.bai").read_bytes()

    # A rerun whose BAM differs (other assignment file, so another header digest).
    other = tmp_path / "other.tsv"
    _write_assignments(other)
    other.write_text(other.read_text().replace("family-seven", "family-other"))
    state = _fail_index_publication(monkeypatch, str(output) + ".bai")
    with pytest.raises(OSError, match="injected"):
        tag_tf_families(source, output, other, force=True)
    assert state["failed"]
    assert output.read_bytes() == before_bam
    assert (tmp_path / "family.bam.bai").read_bytes() == before_index
    assert not [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_tag_consensus_replaces_a_stale_csi_with_the_new_index(tmp_path):
    from fiberhmm.cli.tag_families import tag_tf_families

    source = tmp_path / "source.bam"
    output = tmp_path / "family.bam"
    assignments = tmp_path / "assignments.tsv"
    _make_input(source)
    _write_assignments(assignments)
    tag_tf_families(source, output, assignments)
    (tmp_path / "family.bam.csi").write_bytes(b"stale")
    tag_tf_families(source, output, assignments, force=True)
    assert (tmp_path / "family.bam.bai").is_file()
    assert not (tmp_path / "family.bam.csi").exists()
    with pysam.AlignmentFile(output, "rb") as bam:
        assert len(list(bam.fetch("chr1"))) == 1


def _ma_bam(path):
    header = {"HD": {"VN": "1.6", "SO": "coordinate"}, "SQ": [{"SN": "chr1", "LN": 1000}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        read = pysam.AlignedSegment(bam.header)
        read.query_name = "r"
        read.query_sequence = "A" * 50
        read.reference_id = 0
        read.reference_start = 10
        read.cigarstring = "50M"
        read.set_tag("MA", "50;nuc.:1-10", value_type="Z")
        bam.write(read)
    pysam.index(str(path))


def test_ma_types_failed_index_install_restores_the_original_bam(tmp_path, monkeypatch):
    from fiberhmm.cli import utils

    bam = tmp_path / "calls.bam"
    _ma_bam(bam)
    before_bam = bam.read_bytes()
    before_index = (tmp_path / "calls.bam.bai").read_bytes()
    state = _fail_index_publication(monkeypatch, str(bam) + ".bai")
    with pytest.raises(OSError, match="injected"):
        utils._rewrite_bam_ma_types_in_place(str(bam), ["nuc", "tf"], io_threads=1)
    assert state["failed"]
    assert bam.read_bytes() == before_bam
    assert (tmp_path / "calls.bam.bai").read_bytes() == before_index
    assert not [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]


def _fix_args(inputs, **overrides):
    values = dict(inputs=inputs, output=None, in_place=False, sample_name=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fix_bigbed_missing_input_exits_nonzero(tmp_path, monkeypatch, capsys):
    from fiberhmm.cli import utils

    monkeypatch.setattr(shutil, "which", lambda tool: "/usr/bin/" + tool)
    with pytest.raises(SystemExit) as raised:
        utils.cmd_fix_bigbed(_fix_args([str(tmp_path / "missing.bb")]))
    assert raised.value.code == 1
    err = capsys.readouterr().err
    assert "not found" in err and "could not be fixed" in err


_UCSC = all(shutil.which(tool) for tool in ("bigBedInfo", "bigBedToBed", "bedToBigBed"))


def _bigbed(path, tmp_path):
    autosql = tmp_path / "x.as"
    autosql.write_text(
        'table fiberhmm_tf\n"Sample: old.name. FiberHMM TF calls."\n(\n'
        'string chrom; "c"\nuint chromStart; "s"\nuint chromEnd; "e"\n'
        'string name; "n"\nuint score; "sc"\nchar[1] strand; "st"\n'
        'uint thickStart; "ts"\nuint thickEnd; "te"\nuint reserved; "r"\n'
        'int blockCount; "bc"\nint[blockCount] blockSizes; "bs"\n'
        'int[blockCount] chromStarts; "cs"\n)\n')
    bed = tmp_path / "x.bed"
    bed.write_text("chr1\t10\t60\tr\t0\t+\t10\t60\t0\t1\t50,\t0,\n")
    sizes = tmp_path / "sizes"
    sizes.write_text("chr1\t1000\n")
    subprocess.run(["bedToBigBed", f"-as={autosql}", "-type=bed12", str(bed), str(sizes),
                    str(path)], check=True, capture_output=True)


@pytest.mark.skipif(not _UCSC, reason="UCSC bigBed tools not installed")
def test_fix_bigbed_in_place_publishes_beside_the_target(tmp_path, monkeypatch):
    from fiberhmm.cli import utils

    target = tmp_path / "out" / "s.filtered_T_tf.bb"
    target.parent.mkdir()
    _bigbed(target, tmp_path)
    os.chmod(target, 0o664)
    staged = []
    real_replace = os.replace

    def replace(source, destination, *args, **kwargs):
        staged.append(os.path.dirname(os.path.abspath(source)))
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    utils.cmd_fix_bigbed(_fix_args([str(target)], in_place=True, sample_name="new"))
    assert staged == [str(target.parent)]
    assert oct(os.stat(target).st_mode & 0o777) == oct(0o664)
    info = subprocess.run(["bigBedInfo", "-as", str(target)], capture_output=True, text=True)
    assert "Sample: new." in info.stdout
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]


@pytest.mark.skipif(not _UCSC, reason="UCSC bigBed tools not installed")
def test_fix_bigbed_not_a_bigbed_exits_nonzero_without_traceback(tmp_path, capsys):
    from fiberhmm.cli import utils

    bad = tmp_path / "bad.bb"
    bad.write_text("not a bigbed")
    with pytest.raises(SystemExit) as raised:
        utils.cmd_fix_bigbed(_fix_args([str(bad)]))
    assert raised.value.code == 1
    assert "[fail]" in capsys.readouterr().err
