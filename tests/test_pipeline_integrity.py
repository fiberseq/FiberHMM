"""fiberhmm-pipeline: reuse, publication and input safety.

Regression tests for the 3.0 release review (Codex, 29 Sep): each test is built
from the reviewer's reproduction and failed before the fix.
"""
from __future__ import annotations

import array
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pysam
import pytest

from fiberhmm.inference import bam_output
from fiberhmm.pipeline import runner
from fiberhmm.pipeline.circular import compute_md_nm, merge_origin_pieces
from fiberhmm.pipeline.progress import ProgressReporter, read_marker
from fiberhmm.pipeline.reference import sequence_md5
from fiberhmm.pipeline.runner import Pipeline, PipelineConfig, PipelineError

sys.path.insert(0, str(Path(__file__).parent))
from test_pipeline import (  # noqa: E402
    REPO, _events, _pieces, _record, _run_cli, needs_minimap2, plasmid_run, random_seq,
    write_genbank,
)

assert plasmid_run  # the fixture, re-exported for this module


def config(out, reads, ref, **kwargs):
    kwargs.setdefault("enzyme", "dddb")
    return PipelineConfig(reads=[str(r) for r in (reads if isinstance(reads, list) else [reads])],
                          reference=str(ref), outdir=str(out), cores=1, quiet=True,
                          qc=False, **kwargs)


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Names, samples, platform
# ---------------------------------------------------------------------------

@needs_minimap2
def test_reads_sharing_a_name_stay_separate_molecules(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBERHMM_MINIMAP2_INDEX_DIR", str(tmp_path / "mmi"))
    sequence = random_seq(6000, 2)
    ref = tmp_path / "qnames.fa"
    ref.write_text(">p\n" + sequence + "\n")
    fastqs = []
    for i in range(2):
        path = tmp_path / f"qname{i}.fq"
        mol = sequence[500 + i * 2500:2000 + i * 2500]
        path.write_text("@same\n" + mol + "\n+\n" + "I" * len(mol) + "\n")
        fastqs.append(path)
    p = Pipeline(config(tmp_path / "out", fastqs, ref))
    p._setup(); p.step_prepare_reference(); p.step_index(); p.step_align()
    with pysam.AlignmentFile(p.aligned_bam) as bam:
        records = list(bam.fetch(until_eof=True))
    p.close()
    assert [r.query_name for r in records] == ["same", "same"]
    assert sorted(r.reference_start for r in records) == [500, 3000]
    assert p.stats["align"]["reads"] == 2 and p.stats["align"]["kept"] == 2


@pytest.mark.parametrize("sample", ["../escape", "/tmp/abs", "a/b", "..", ".hidden", "a b", "x\ty"])
def test_sample_cannot_leave_outdir(tmp_path, sample):
    reads = tmp_path / "r.fastq"; reads.write_text("@r\nAAAA\n+\nIIII\n")
    ref = tmp_path / "ref.fa"; ref.write_text(">p\n" + "A" * 100 + "\n")
    p = Pipeline(config(tmp_path / "out", reads, ref, sample=sample))
    with pytest.raises(PipelineError, match="not a plain file name"):
        p._setup()
    p.close()
    assert not (tmp_path / "escape.fiberhmm.bam").exists()
    ok = Pipeline(config(tmp_path / "out", reads, ref, sample="run-1.b"))
    ok._setup()
    assert os.path.dirname(ok.called_bam) == str(tmp_path / "out")
    ok.close()


def _pacbio_fastq(path: Path, sequence: str, n: int = 3, mm: str = "A+a.,0;T-a.,0;") -> Path:
    with open(path, "w") as handle:
        for i in range(n):
            seq = sequence[500 + i * 100:2000 + i * 100]
            handle.write(f"@pb{i}\tMM:Z:{mm}\tML:B:C,255,255\n{seq}\n+\n{'I' * len(seq)}\n")
    return path


def test_hia5_platform_is_decided_from_the_reads(tmp_path):
    ref = tmp_path / "ref.fa"; sequence = random_seq(6000, 3)
    ref.write_text(">p\n" + sequence + "\n")
    pacbio = _pacbio_fastq(tmp_path / "pacbio.fastq", sequence)
    p = Pipeline(config(tmp_path / "o1", pacbio, ref, enzyme="hia5"))
    p._setup(); p.close()
    assert p.config.seq == "pacbio" and p.config.preset() == "map-hifi"
    assert p.config.calling_settings()["seq"] == "pacbio"
    nanopore = _pacbio_fastq(tmp_path / "ont.fastq", sequence, mm="A+a.,0;")
    p = Pipeline(config(tmp_path / "o2", nanopore, ref, enzyme="hia5"))
    p._setup(); p.close()
    assert p.config.seq == "nanopore" and p.config.preset() == "map-ont"
    plain = tmp_path / "plain.fastq"; plain.write_text("@r\nACGT\n+\nIIII\n")
    with pytest.raises(PipelineError, match="sequencing platform") as error:
        Pipeline(config(tmp_path / "o3", plain, ref, enzyme="hia5"))._setup()
    assert "--seq" in error.value.hint
    with pytest.raises(PipelineError, match="disagree"):
        Pipeline(config(tmp_path / "o4", [pacbio, nanopore], ref, enzyme="hia5"))._setup()


@needs_minimap2
def test_hia5_pacbio_fastq_runs_without_seq(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBERHMM_MINIMAP2_INDEX_DIR", str(tmp_path / "mmi"))
    sequence = random_seq(6000, 3)
    ref = tmp_path / "ref.fa"; ref.write_text(">p\n" + sequence + "\n")
    fastq = _pacbio_fastq(tmp_path / "pacbio.fastq", sequence, n=1)
    progress = tmp_path / "p.jsonl"
    result = _run_cli([str(fastq), "--reference", str(ref), "--enzyme", "hia5", "-c", "1",
                       "--no-qc", "--min-read-length", "0", "-o", str(tmp_path / "out"),
                       "--progress-json", str(progress)])
    assert result.returncode == 0, result.stderr[-3000:]
    events = _events(progress)
    assert events[0]["event"] == "start" and events[0]["settings"]["seq"] == "pacbio"
    outputs = json.loads((tmp_path / "out" / "outputs.json").read_text())
    with pysam.AlignmentFile(outputs["aligned_bam"]) as bam:
        assert bam.header.to_dict()["RG"][0]["PL"] == "PACBIO"


# ---------------------------------------------------------------------------
# Reference identity
# ---------------------------------------------------------------------------

def test_refused_rerun_keeps_the_previous_reference(tmp_path):
    ref = tmp_path / "reference.fa"; reads = tmp_path / "r.fastq"
    ref.write_text(">p\n" + "A" * 1000 + "\n"); reads.write_text("@r\nAAAA\n+\nIIII\n")
    out = tmp_path / "out"
    p = Pipeline(config(out, reads, ref)); p._setup(); p.step_prepare_reference()
    published = Path(p.reference.fasta)
    old = published.read_bytes()
    Path(p.aligned_bam).write_bytes(b"old-bam"); Path(p.aligned_bam + ".bai").write_bytes(b"old-index")
    p._write_marker("align", p._align_fingerprint(), {"bam": p.aligned_bam, "bai": p.aligned_bam + ".bai"})
    p.close()
    ref.write_text(">p\n" + "C" * 1000 + "\n")
    p = Pipeline(config(out, reads, ref)); p._setup()
    with pytest.raises(PipelineError, match=r"'align' result .*\(reference\)") as refused:
        p.step_prepare_reference()
    p.close()
    assert "--redo align" in refused.value.hint
    assert published.read_bytes() == old
    assert Path(published.as_posix() + ".fai").exists()
    # With --redo align the new reference is published.
    p = Pipeline(config(out, reads, ref, redo="align")); p._setup(); p.step_prepare_reference(); p.close()
    assert published.read_bytes() != old


def test_refused_rerun_keeps_the_previous_plasmid_map(tmp_path):
    reads = tmp_path / "r.fastq"; reads.write_text("@r\nAAAA\n+\nIIII\n")
    gb = tmp_path / "pX.gbk"
    write_genbank(gb, "pX", random_seq(1200, 5))
    out = tmp_path / "out"
    p = Pipeline(config(out, reads, gb)); p._setup(); p.step_prepare_reference()
    fasta, gb_copy = Path(p.reference.fasta), Path(p.reference.plasmid_map)
    before = (fasta.read_bytes(), gb_copy.read_bytes())
    Path(p.aligned_bam).write_bytes(b"bam")
    p._write_marker("align", p._align_fingerprint(), {"bam": p.aligned_bam})
    p.close()
    write_genbank(gb, "pX", random_seq(1200, 6))
    p = Pipeline(config(out, reads, gb)); p._setup()
    with pytest.raises(PipelineError, match="reference"):
        p.step_prepare_reference()
    p.close()
    assert (fasta.read_bytes(), gb_copy.read_bytes()) == before


def _aligned_circle_bam(tmp_path, seq, name="split.bam", m5=False):
    L = len(seq)
    sq = {"SN": "p", "LN": L}
    if m5:
        sq["M5"] = sequence_md5(seq)
    h = pysam.AlignmentHeader.from_dict({"HD": {"SO": "coordinate"}, "SQ": [sq]})
    mol = (seq + seq)[1500:3300]
    a = _record(h, "origin", mol, 1500, [(0, 500), (4, 1300)])
    b = _record(h, "origin", mol, 0, [(4, 500), (0, 1300)], supplementary=True)
    for rec in (a, b):
        md, nm = compute_md_nm(mol, rec.cigartuples, seq, rec.reference_start)
        rec.set_tag("MD", md); rec.set_tag("NM", nm)
    bam = tmp_path / name
    with pysam.AlignmentFile(str(bam), "wb", header=h) as out:
        out.write(b); out.write(a)
    pysam.index(str(bam))
    return bam, a, b


def test_m5_less_aligned_input_is_checked_against_the_reference(tmp_path):
    seq = random_seq(2000, 7)
    bam, _, _ = _aligned_circle_bam(tmp_path, seq)
    right = tmp_path / "right.fa"; right.write_text(">p\n" + seq + "\n"); pysam.faidx(str(right))
    wrong = tmp_path / "wrong.fa"; wrong.write_text(">p\n" + seq.translate(str.maketrans("ACGT", "TGCA")) + "\n")
    pysam.faidx(str(wrong))
    assert runner._reads_match_reference(str(bam), str(right)) == (True, "")
    ok, why = runner._reads_match_reference(str(bam), str(wrong))
    assert not ok and "M5" in why
    # The pipeline realigns it rather than calling it against the wrong sequence.
    p = Pipeline(config(tmp_path / "out", bam, wrong, topology="circular"))
    p._setup(); p.step_prepare_reference(); p.close()
    assert p.use_aligned_input is None
    p = Pipeline(config(tmp_path / "out2", bam, right, topology="circular"))
    p._setup(); p.step_prepare_reference(); p.close()
    assert p.use_aligned_input == str(bam)


def test_call_fingerprint_binds_the_reference_for_aligned_input(tmp_path):
    seq = random_seq(2000, 7)
    bam, _, _ = _aligned_circle_bam(tmp_path, seq)
    ref = tmp_path / "circle.fa"; ref.write_text(">p\n" + seq + "\n")
    out = tmp_path / "out"
    p = Pipeline(config(out, bam, ref, topology="circular")); p._setup(); p.step_prepare_reference()
    p.step_index(); p.step_align()
    assert p.aligned_bam == str(bam)
    fingerprint = p._call_fingerprint()
    assert fingerprint["reference"][0]["md5"] == sequence_md5(seq)
    Path(p.called_bam).write_bytes(b"called")
    p._write_marker("call", fingerprint, {"bam": p.called_bam})
    p.close()
    # Same length, different sequence; the aligned input's header has no M5 to disagree.
    other = seq[:1000] + seq[1000:][::-1]
    ref.write_text(">p\n" + other + "\n")
    p = Pipeline(config(out, bam, ref, topology="circular")); p._setup()
    with pytest.raises(PipelineError, match=r"'call' result .*\(reference\)"):
        p.step_prepare_reference()
    p.close()


# ---------------------------------------------------------------------------
# Circular records: region filter and SNP masks
# ---------------------------------------------------------------------------

def test_region_filter_keeps_reads_through_the_origin(tmp_path):
    seq = random_seq(2000, 7)
    bam, a, b = _aligned_circle_bam(tmp_path, seq, m5=True)
    merged = merge_origin_pieces(a, b, seq)
    assert (merged.reference_start, merged.reference_end) == (1500, 3300)
    ref = tmp_path / "circle.fa"; ref.write_text(">p\n" + seq + "\n")
    p = Pipeline(config(tmp_path / "o", bam, ref, topology="circular", regions=["p:1-100"]))
    p._setup(); p.step_prepare_reference()
    assert p._overlaps_regions(merged)
    assert not p._overlaps_regions(merged, [("p", 1400, 1450)])
    p.close()
    # The indexed subset of an aligned input finds it too.
    wrapped = tmp_path / "wrapped.bam"
    h = pysam.AlignmentHeader.from_dict({"HD": {"SO": "coordinate"},
                                         "SQ": [{"SN": "p", "LN": 2000, "M5": sequence_md5(seq),
                                                 "TP": "circular"}]})
    with pysam.AlignmentFile(str(wrapped), "wb", header=h) as out:
        record = pysam.AlignedSegment.from_dict(merged.to_dict(), h)
        out.write(record)
    pysam.index(str(wrapped))
    p = Pipeline(config(tmp_path / "o2", wrapped, ref, topology="circular", regions=["p:1-100"]))
    p._setup(); p.step_prepare_reference(); p.step_index(); p.step_align(); p.close()
    with pysam.AlignmentFile(p.aligned_bam) as handle:
        kept = list(handle.fetch(until_eof=True))
    assert [(r.query_name, r.reference_start, r.reference_end) for r in kept] == [("origin", 1500, 3300)]


def test_region_subset_keeps_identical_molecules_once_each(tmp_path):
    """Two identical stored records are two molecules: both stay, whether a
    record is fetched through several overlapping regions or through the
    circular-origin window as well as its region."""
    seq = "A" * 2000
    ref = tmp_path / "circle.fa"; ref.write_text(">p\n" + seq + "\n")
    h = pysam.AlignmentHeader.from_dict({"HD": {"SO": "coordinate"},
                                         "SQ": [{"SN": "p", "LN": 2000, "M5": sequence_md5(seq),
                                                 "TP": "circular"}]})
    bam = tmp_path / "dups.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=h) as out:
        for start, length, copies in ((100, 300, 2), (350, 100, 1), (1500, 1800, 2)):
            for _ in range(copies):
                r = pysam.AlignedSegment(h); r.query_name = "same"; r.reference_id = 0
                r.reference_start = start; r.query_sequence = "A" * length
                r.cigarstring = f"{length}M"; r.mapping_quality = 60; r.set_tag("MD", str(length))
                out.write(r)
    pysam.index(str(bam))

    def subset(name, regions):
        p = Pipeline(config(tmp_path / name, bam, ref, topology="circular", regions=regions))
        p._setup(); p.step_prepare_reference(); p.step_index(); p.step_align(); p.close()
        with pysam.AlignmentFile(p.aligned_bam) as handle:
            return sorted((r.reference_start, r.reference_end)
                          for r in handle.fetch(until_eof=True))
    # The origin-spanning pair via the wrapped part only.
    assert subset("origin", ["p:1-50"]) == [(1500, 3300)] * 2
    # Overlapping and touching regions, and the last-base window: each record once.
    assert subset("many", ["p:150-200", "p:180-400", "p:401-420", "p:1990-2000",
                           "p:1-10"]) == [(100, 400)] * 2 + [(350, 450)] + [(1500, 3300)] * 2


def _wrapped_read():
    h = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": "p", "LN": 2000, "TP": "circular"}]})
    r = pysam.AlignedSegment(h); r.query_name = "r"; r.reference_id = 0; r.reference_start = 1500
    r.query_sequence = "A" * 1800; r.cigarstring = "1800M"
    return r


def test_snp_mask_applies_after_the_origin():
    from fiberhmm.inference import engine
    read = _wrapped_read()
    saved = engine._DAF_SNP_MASK
    try:
        engine._DAF_SNP_MASK = {"p": {10, 1600}}
        assert engine._daf_excluded_query_positions(read) == {100, 510}
        assert 2010 in engine._daf_reference_mask(read) and 10 in engine._daf_reference_mask(read)
    finally:
        engine._DAF_SNP_MASK = saved
    # QC excludes the same site: a C->T at canonical site 10 (unrolled 2010) is not counted.
    from fiberhmm.qc.core import _signal_profile
    h = read.header
    ref = "C" * 2000
    r = pysam.AlignedSegment(h); r.query_name = "q"; r.reference_id = 0; r.reference_start = 1500
    seq = list("C" * 1800); seq[510] = "T"; seq[20] = "T"; r.query_sequence = "".join(seq)
    r.cigarstring = "1800M"
    md, _ = compute_md_nm(r.query_sequence, r.cigartuples, ref, 1500)
    r.set_tag("MD", md)
    plain = _signal_profile(r, "daf")
    masked = _signal_profile(r, "daf", snp_mask={"p": {10}})
    assert plain[0].tolist() == [1520, 2010] and masked[0].tolist() == [1520]
    assert masked[1] == plain[1] - 1
    # A masked site stays masked when it was the read's only event (the MM/ML fallback).
    only = _signal_profile(r, "daf", snp_mask={"p": {10, 1520}})
    assert only[0].tolist() == []
    # The SNP screen folds positions past the contig end back onto the contig.
    from fiberhmm.daf.snps import _read_arrays
    rpos, _, _ = _read_arrays(r)
    assert int(rpos.max()) < 2000 and 10 in set(rpos.tolist())


# ---------------------------------------------------------------------------
# Publication and cancellation
# ---------------------------------------------------------------------------

def test_failed_publication_keeps_the_previous_bam_and_index(tmp_path):
    dest = tmp_path / "atomic.bam"
    dest.write_bytes(b"old-bam"); Path(str(dest) + ".bai").write_bytes(b"old-index")
    replace = os.replace

    def fail_index(src, dst):
        if str(dst) == str(dest) + ".bai" and not str(src).endswith(".previous"):
            raise RuntimeError("injected index rename failure")
        return replace(src, dst)
    with pytest.raises(RuntimeError), patch.object(bam_output.os, "replace", fail_index):
        with bam_output.atomic_output(str(dest)) as tmp:
            Path(tmp).write_bytes(b"new-bam"); Path(tmp + ".bai").write_bytes(b"new-index")
    assert dest.read_bytes() == b"old-bam" and Path(str(dest) + ".bai").read_bytes() == b"old-index"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["atomic.bam", "atomic.bam.bai"]
    # A first publication that fails leaves nothing that looks like an output.
    fresh = tmp_path / "fresh.bam"
    with pytest.raises(RuntimeError), patch.object(bam_output.os, "replace", fail_index):
        with bam_output.atomic_output(str(fresh)) as tmp:
            Path(tmp).write_bytes(b"new")
            dest = fresh  # the failing index rename is fresh's now
            Path(tmp + ".bai").write_bytes(b"idx")
    assert not fresh.exists() and not Path(str(fresh) + ".bai").exists()
    # Success: the new pair, no leftovers.
    good = tmp_path / "good.bam"; good.write_bytes(b"old"); Path(str(good) + ".csi").write_bytes(b"old-csi")
    with bam_output.atomic_output(str(good)) as tmp:
        Path(tmp).write_bytes(b"new"); Path(tmp + ".bai").write_bytes(b"new-bai")
    assert good.read_bytes() == b"new" and Path(str(good) + ".bai").read_bytes() == b"new-bai"
    assert not Path(str(good) + ".csi").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def _publication_pair(root, name):
    final = root / f"{name}.bam"; temporary = root / f".{name}.tmp.bam"
    final.write_bytes(b"old-bam"); Path(str(final) + ".bai").write_bytes(b"old-index")
    temporary.write_bytes(b"new-bam"); Path(str(temporary) + ".bai").write_bytes(b"new-index")
    return final, temporary


def _pair_state(final):
    index = Path(str(final) + ".bai")
    return (final.read_bytes() if final.exists() else None,
            index.read_bytes() if index.exists() else None)


def test_publication_backup_falls_back_to_a_verified_copy(tmp_path):
    """Hard links failing for the BAM only (EMLINK), or everywhere (EOPNOTSUPP),
    must still give a complete backup generation: a failure or interrupt then
    restores the old BAM and index together, never new BAM + old index."""
    import errno
    link, replace = os.link, os.replace
    for case, error in (("emlink", errno.EMLINK), ("unsupported", errno.EOPNOTSUPP)):
        final, temporary = _publication_pair(tmp_path, case)

        def fail_link(src, dst):
            if case == "unsupported" or str(src) == str(final):
                raise OSError(error, os.strerror(error))
            return link(src, dst)

        def fail_index(src, dst):
            if str(src) == str(temporary) + ".bai":
                raise (KeyboardInterrupt if case == "unsupported" else RuntimeError)("injected")
            return replace(src, dst)
        with pytest.raises((RuntimeError, KeyboardInterrupt)), \
                patch.object(bam_output.os, "link", fail_link), \
                patch.object(bam_output.os, "replace", fail_index):
            bam_output.commit_output(str(temporary), str(final))
        assert _pair_state(final) == (b"old-bam", b"old-index"), case
        assert not [p for p in tmp_path.iterdir() if p.name.endswith(".previous")]
    # Normal success with copies instead of links leaves no backups behind.
    final, temporary = _publication_pair(tmp_path, "copied")
    with patch.object(bam_output.os, "link", side_effect=OSError(errno.EXDEV, "x")):
        bam_output.commit_output(str(temporary), str(final))
    assert _pair_state(final) == (b"new-bam", b"new-index")
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".previous")]


def test_publication_is_refused_when_no_backup_can_be_made(tmp_path):
    import errno
    final, temporary = _publication_pair(tmp_path, "refused")
    with pytest.raises(OSError), \
            patch.object(bam_output.os, "link", side_effect=OSError(errno.EOPNOTSUPP, "x")), \
            patch.object(bam_output.shutil, "copy2", side_effect=OSError(errno.ENOSPC, "full")):
        bam_output.commit_output(str(temporary), str(final))
    assert _pair_state(final) == (b"old-bam", b"old-index")
    assert temporary.read_bytes() == b"new-bam"   # nothing was touched
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".previous")]


def test_publication_never_restores_an_index_without_its_bam(tmp_path):
    """If the old BAM cannot be put back, no old index is put beside the new
    data, and the previous generation is kept rather than deleted."""
    final, temporary = _publication_pair(tmp_path, "stuck")
    replace = os.replace

    def fail(src, dst):
        if str(src) == str(temporary) + ".bai":
            raise RuntimeError("injected index rename failure")
        if str(src).endswith(".previous") and str(dst) == str(final):
            raise OSError("cannot restore")
        return replace(src, dst)
    with pytest.raises(RuntimeError), patch.object(bam_output.os, "replace", fail):
        bam_output.commit_output(str(temporary), str(final))
    assert _pair_state(final) == (b"new-bam", None)
    kept = sorted(p.read_bytes() for p in tmp_path.iterdir() if p.name.endswith(".previous"))
    assert kept == [b"old-bam", b"old-index"]
    # An orphan index (no previous BAM) is not resurrected by a rollback.
    orphan = tmp_path / "orphan.bam"; Path(str(orphan) + ".bai").write_bytes(b"orphan-index")
    tmp = tmp_path / ".orphan.tmp.bam"; tmp.write_bytes(b"new"); Path(str(tmp) + ".bai").write_bytes(b"i")

    def fail_orphan(src, dst):
        if str(src) == str(tmp) + ".bai":
            raise RuntimeError("injected")
        return replace(src, dst)
    with pytest.raises(RuntimeError), patch.object(bam_output.os, "replace", fail_orphan):
        bam_output.commit_output(str(tmp), str(orphan))
    assert _pair_state(orphan) == (None, None)


def test_publication_restores_a_symlinked_previous_output_as_a_symlink(tmp_path):
    target = tmp_path / "store.bam"; target.write_bytes(b"old-bam")
    final = tmp_path / "linked.bam"; final.symlink_to(target)
    temporary = tmp_path / ".linked.tmp.bam"; temporary.write_bytes(b"new-bam")
    Path(str(temporary) + ".bai").write_bytes(b"new-index")
    replace = os.replace

    def fail_index(src, dst):
        if str(src) == str(temporary) + ".bai":
            raise RuntimeError("injected")
        return replace(src, dst)
    with pytest.raises(RuntimeError), patch.object(bam_output.os, "replace", fail_index):
        bam_output.commit_output(str(temporary), str(final))
    assert final.is_symlink() and os.readlink(final) == str(target)
    assert target.read_bytes() == b"old-bam"


def test_pipeline_publication_never_pairs_new_index_with_old_bam(tmp_path):
    h = pysam.AlignmentHeader.from_dict({"HD": {"SO": "unsorted"}, "SQ": [{"SN": "p", "LN": 1000}]})
    unsorted = tmp_path / "x.unsorted.bam"
    with pysam.AlignmentFile(str(unsorted), "wb", header=h) as out:
        out.write(_record(h, "a", "ACGT" * 10, 10, [(0, 40)]))
    final = tmp_path / "x.bam"
    final.write_bytes(b"old-bam"); Path(str(final) + ".bai").write_bytes(b"old-index")
    replace = os.replace
    failed = []

    def fail_bam(src, dst):
        if str(dst) == str(final) and not failed:
            failed.append(src)
            raise RuntimeError("injected BAM rename failure")
        return replace(src, dst)
    with pytest.raises(RuntimeError), patch.object(bam_output.os, "replace", fail_bam):
        runner._sort_index_publish(str(unsorted), str(final), 1)
    assert final.read_bytes() == b"old-bam" and Path(str(final) + ".bai").read_bytes() == b"old-index"


def test_sigterm_while_the_child_registry_is_locked_does_not_hang():
    code = """import os,signal
from fiberhmm.pipeline.runner import install_cancel_handlers,_CHILDREN_LOCK,terminate_children,PipelineCancelled
install_cancel_handlers()
try:
    with _CHILDREN_LOCK:
        print("locked",flush=True)
        os.kill(os.getpid(),signal.SIGTERM)
        print("not reached",flush=True)
except PipelineCancelled as exc:
    terminate_children()
    print("cancelled",exc.exit_code,flush=True)
"""
    env = dict(os.environ, PYTHONPATH=str(REPO) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                            text=True, timeout=30)
    assert result.stdout.split() == ["locked", "cancelled", "143"], result.stderr


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------

def test_one_run_per_outdir(tmp_path):
    reads = tmp_path / "r.fastq"; reads.write_text("@r\nAAAA\n+\nIIII\n")
    ref = tmp_path / "ref.fa"; ref.write_text(">p\n" + "A" * 100 + "\n")
    first = Pipeline(config(tmp_path / "out", reads, ref)); first._setup()
    events = []
    second = Pipeline(config(tmp_path / "out", reads, ref), ProgressReporter(callback=events.append))
    with pytest.raises(PipelineError, match="in use by another running process"):
        second.run()
    assert events[-1]["event"] == "done" and events[-1]["status"] == "error"
    second.close(); first.close()
    third = Pipeline(config(tmp_path / "out", reads, ref)); third._setup(); third.close()


# ---------------------------------------------------------------------------
# Kept outputs, QC and --redo (end to end)
# ---------------------------------------------------------------------------

def _plasmid_args(run, *extra):
    return [str(run["reads"]), "--reference", str(run["map"]), "--enzyme", "dddb",
            "-o", str(run["out"]), "--sample", "s1", "-c", "1", "--min-read-length", "800",
            *extra]


@needs_minimap2
def test_damaged_outputs_are_rebuilt_and_stale_qc_is_never_published(plasmid_run):
    run = plasmid_run
    first = _run_cli(_plasmid_args(run))
    assert first.returncode == 0, first.stderr[-3000:]
    outputs = json.loads((run["out"] / "outputs.json").read_text())
    assert outputs["qc"] and Path(outputs["qc"]["json"]).exists()
    called = Path(outputs["called_bam"])
    with pysam.AlignmentFile(str(called)) as bam:
        reference_records = [r.to_string() for r in bam.fetch(until_eof=True)]

    # A changed calling setting with --redo call and --no-qc: QC is not published.
    progress = Path(str(run["progress"]) + ".noqc")
    redo = _run_cli(_plasmid_args(run, "--redo", "call", "--no-qc", "--min-read-length", "900",
                                  "--progress-json", str(progress)))
    assert redo.returncode == 0, redo.stderr[-3000:]
    outputs = json.loads((run["out"] / "outputs.json").read_text())
    assert outputs["settings"]["qc"] is False and outputs["qc"] is None
    assert outputs["qc_report"] is None and _events(progress)[-1]["outputs"]["qc"] is None

    # QC of the replaced BAM is forgotten, so re-enabling QC is not refused.
    assert read_marker(str(run["out"]), "qc") is None
    # Back to the first settings: the call is redone, then QC runs for the new BAM.
    back = _run_cli(_plasmid_args(run, "--redo", "call"))
    assert back.returncode == 0, back.stderr[-3000:]
    outputs = json.loads((run["out"] / "outputs.json").read_text())
    qc_marker = read_marker(str(run["out"]), "qc")
    assert qc_marker["fingerprint"]["called"]["sha256"] == sha(called)
    assert outputs["qc"]["json"] and Path(outputs["qc"]["json"]).exists()

    # The called BAM truncated to nothing: the rerun calls again instead of publishing it.
    called.write_bytes(b"")
    progress = Path(str(run["progress"]) + ".damaged")
    again = _run_cli(_plasmid_args(run, "--progress-json", str(progress)))
    assert again.returncode == 0, again.stderr[-3000:]
    status = {e["step"]: e["status"] for e in _events(progress)
              if e["event"] == "step" and e["status"] != "running"}
    assert status["align"] == "skipped" and status["call"] == "done"
    with pysam.AlignmentFile(str(called)) as bam:
        assert [r.to_string() for r in bam.fetch(until_eof=True)] == reference_records


def test_call_args_files_are_resolved_by_the_call_parser(tmp_path, monkeypatch):
    """Every spelling fiberhmm-call accepts for a file option is a dependency,
    including attached short options (-mPATH, -m/abs/path) and prefixes."""
    from fiberhmm.pipeline.runner import call_arg_file_paths
    model = tmp_path / "attached.json"; model.write_text("{}")
    recall = tmp_path / "recall.json"; recall.write_text("{}")
    mask = tmp_path / "mask.bed"; mask.write_text("")
    monkeypatch.chdir(tmp_path)
    for args in (["-m" + str(model)], ["-m", str(model)], ["--model=" + str(model)],
                 ["--model", str(model)], ["-mattached.json"], ["--mod", str(model)]):
        assert [os.path.abspath(p) for p in call_arg_file_paths(args)] == [str(model)], args
    found = call_arg_file_paths(["-m" + str(model), "--recall-model=" + str(recall),
                                 "--daf-snp-mask", str(mask), "--no-recall-nucs"])
    assert sorted(found) == sorted([str(model), str(recall), str(mask)])
    # Unknown or unparsable arguments still fall back to whole tokens.
    assert call_arg_file_paths(["--not-an-option", str(mask)]) == [str(mask)]
    assert call_arg_file_paths(["--min-mapq", "x", str(mask)]) == [str(mask)]
    assert call_arg_file_paths([]) == []


@needs_minimap2
@pytest.mark.parametrize("spelling", ["spaced", "attached"])
def test_changed_file_in_call_args_is_a_changed_setting(tmp_path, monkeypatch, spelling):
    monkeypatch.setenv("FIBERHMM_MINIMAP2_INDEX_DIR", str(tmp_path / "mmi"))
    sequence = random_seq(6000, 3)
    ref = tmp_path / "ref.fa"; ref.write_text(">p\n" + sequence + "\n")
    fastq = _pacbio_fastq(tmp_path / "pacbio.fastq", sequence, n=1)
    model = tmp_path / "custom.json"
    model.write_bytes((REPO / "fiberhmm" / "models" / "hia5_pacbio.json").read_bytes())
    args = [str(fastq), "--reference", str(ref), "--enzyme", "hia5", "--seq", "pacbio", "-c", "1",
            "--no-qc", "--min-read-length", "0", "-o", str(tmp_path / "out"),
            f"--call-args=-m{'' if spelling == 'attached' else ' '}{model} --no-recall-nucs"]
    first = _run_cli(args)
    assert first.returncode == 0, first.stderr[-3000:]
    data = json.loads(model.read_text()); data["transmat"] = [[.5, .5], [.5, .5]]
    model.write_text(json.dumps(data))
    progress = tmp_path / "p.jsonl"
    second = _run_cli(args + ["--progress-json", str(progress)])
    assert second.returncode == 1
    last = _events(progress)[-1]
    assert last["status"] == "error" and "call_arg_files" in last["error"]
    assert "--redo call" in last["hint"]


def test_pipeline_cleanup_refuses_a_live_work_dir_owner(tmp_path):
    """--redo cleanup must not delete a calling work directory that a live
    fiberhmm-call --work-dir owner holds (its own lock, not the OUTDIR lock)."""
    from fiberhmm.inference.region_resume import WorkDir
    reads = tmp_path / "r.fastq"; reads.write_text("@r\nAAAA\n+\nIIII\n")
    ref = tmp_path / "ref.fa"; ref.write_text(">p\n" + "A" * 1000 + "\n")
    p = Pipeline(config(tmp_path / "out", reads, ref, call_mode="resumable", redo="call"))
    p._setup()
    work = Path(p.call_work_dir)
    owner = WorkDir(work, {"parameter": 1}).open()
    try:
        inode = (work / ".lock").stat().st_ino
        manifest = (work / "manifest.json").read_text()
        for resumable in (True, False):
            with pytest.raises(PipelineError, match="in use"):
                p._prepare_call_state({"new_parameter": 2}, resumable)
            assert (work / ".lock").stat().st_ino == inode
            assert (work / "manifest.json").read_text() == manifest
        with pytest.raises(Exception, match="in use"):
            WorkDir(work, {"parameter": 2}).open()
    finally:
        owner.release()
    # Once the owner is gone the same cleanup proceeds.
    p._prepare_call_state({"new_parameter": 2}, True)
    assert not work.exists()
    p.close()


def test_redo_call_discards_incompatible_resumable_state(tmp_path, monkeypatch):
    from fiberhmm.inference.region_resume import WorkDir
    reads = tmp_path / "r.fastq"; reads.write_text("@r\nAAAA\n+\nIIII\n")
    ref = tmp_path / "ref.fa"; ref.write_text(">p\n" + "A" * 1000 + "\n")
    out = tmp_path / "out"
    p = Pipeline(config(out, reads, ref, call_mode="resumable", redo="call")); p._setup()
    p.step_prepare_reference()
    Path(p.aligned_bam).write_bytes(b"aligned")
    work = Path(p.call_work_dir)
    WorkDir(work, {"old_parameter": 1}).open().release()
    fingerprint = {"aligned": "x", "setting": 1}
    p._prepare_call_state(fingerprint, resumable=True)
    assert not work.exists()        # --redo call: the incompatible work is gone before calling
    p.close()
    # Without --redo, an interrupted call made with other settings is refused like a finished one.
    p = Pipeline(config(out, reads, ref, call_mode="resumable")); p._setup()
    WorkDir(work, {"old_parameter": 1}).open().release()
    with pytest.raises(PipelineError, match="interrupted calling state") as refused:
        p._prepare_call_state({"aligned": "x", "setting": 2}, resumable=True)
    assert "--redo call" in refused.value.hint and work.exists()
    p._prepare_call_state(fingerprint, resumable=True)   # the same call: kept for --resume
    assert work.exists()
    p.close()


def test_failed_qc_publishes_no_qc(tmp_path, monkeypatch):
    reads = tmp_path / "r.fastq"; reads.write_text("@r\nAAAA\n+\nIIII\n")
    ref = tmp_path / "ref.fa"; ref.write_text(">p\n" + "A" * 1000 + "\n")
    out = tmp_path / "out"
    cfg = config(out, reads, ref); cfg.qc = True
    p = Pipeline(cfg); p._setup(); p.step_prepare_reference()
    Path(p.called_bam).write_bytes(b"called")
    os.makedirs(os.path.dirname(p.qc_prefix), exist_ok=True)
    stale = Path(p.qc_prefix + ".qc.json"); stale.write_text('{"overall": {"status": "PASS"}}')
    monkeypatch.setattr(Pipeline, "_run_logged", lambda self, cmd, log, *a, **k: 3)
    p.step_qc()
    assert p.qc_outputs is None and not stale.exists()
    p.open_region = lambda: None
    outputs = p.write_outputs()
    assert outputs["qc"] is None and outputs["qc_report"] is None
    p.close()
