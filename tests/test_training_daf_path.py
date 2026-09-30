"""Training / table-building path (fiberhmm-probs, fiberhmm-train,
fiberhmm-utils transfer): DAF evidence, zero-read exits, 0/0 emissions,
per-read Baum-Welch lengths, and model saving (3.0 release fixes)."""
from __future__ import annotations

import random
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pysam
import pytest

from fiberhmm.cli import generate_probs, train
from fiberhmm.cli import utils as fh_utils
from fiberhmm.core.hmm import FiberHMM, train_model
from fiberhmm.probabilities.context_counter import ContextCounter
from fiberhmm.probabilities.utils import TrainingRead, extract_training_read

REF_LEN = 3000
READ_LEN = 2000


def _reference(seed=11):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(REF_LEN))


def _md(ref, query):
    out, run = [], 0
    for r, q in zip(ref, query):
        if r == q:
            run += 1
        else:          # includes R/Y bases, which still mismatch the reference
            out.append(f"{run}{r}")
            run = 0
    out.append(str(run))
    return "".join(out)


def _mm_tag(sequence, positions, base, code):
    targets = [i for i, b in enumerate(sequence) if b == base]
    skips, last = [], -1
    for pos in sorted(positions):
        idx = targets.index(pos)
        skips.append(idx - last - 1)
        last = idx
    return f"{base}+{code}.," + ",".join(map(str, skips)) + ";"


def _daf_read(header, ref, start, kind, name, n_deam=40):
    """A CT-strand DAF read in one of three encodings: iupac, md, mmml."""
    seg = ref[start:start + READ_LEN]
    c_sites = [i for i, b in enumerate(seg) if b == "C"][5:5 + n_deam * 3:3]
    q = list(seg)
    for i in c_sites:
        q[i] = "Y" if kind == "iupac" else "T"
    q = "".join(q)
    a = pysam.AlignedSegment(header)
    a.query_name = name
    a.query_sequence = q
    a.flag = 0
    a.reference_id = 0
    a.reference_start = start
    a.mapping_quality = 60
    a.cigartuples = [(0, READ_LEN)]
    a.query_qualities = pysam.qualitystring_to_array("I" * READ_LEN)
    if kind == "iupac":
        a.set_tag("st", "CT", "Z")
        a.set_tag("MD", _md(seg, q), "Z")
    elif kind == "md":
        a.set_tag("MD", _md(seg, q), "Z")
    elif kind == "mmml":
        a.set_tag("MM", _mm_tag(q, c_sites, "T", "u"))
        a.set_tag("ML", [240] * len(c_sites))
    return a, set(c_sites)


def _write_daf_bam(path, kind, n_reads=2):
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6", "SO": "coordinate"},
         "SQ": [{"LN": REF_LEN, "SN": "chr1"}]})
    ref = _reference()
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for i in range(n_reads):
            read, _ = _daf_read(header, ref, i * 500, kind, f"{kind}{i}")
            out.write(read)
    pysam.index(str(path))
    return str(path)


def _probs_args(**kw):
    base = dict(min_mapq=0, min_read_length=100, prob_threshold=128,
                edge_trim=10, max_reads=0)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("kind", ["iupac", "md", "mmml"])
def test_probs_counts_daf_reads_in_every_encoding(tmp_path, kind):
    """R/Y and MD DAF BAMs used to give 'Processed 0 reads' (no MM tag)."""
    bam = _write_daf_bam(tmp_path / f"{kind}.bam", kind)
    counters = {"C": ContextCounter(3, "C")}
    n, stats = generate_probs.process_bam(bam, counters, "daf", _probs_args())
    assert n == 2
    assert counters["C"].total_modified == 80
    assert counters["C"].total_positions > counters["C"].total_modified


def test_probs_encodings_agree(tmp_path):
    counts = {}
    for kind in ("iupac", "md", "mmml"):
        bam = _write_daf_bam(tmp_path / f"{kind}.bam", kind)
        counters = {"C": ContextCounter(3, "C")}
        generate_probs.process_bam(bam, counters, "daf", _probs_args())
        counts[kind] = dict(counters["C"].counts)
    assert counts["iupac"] == counts["md"] == counts["mmml"]


def test_probs_exits_nonzero_when_no_read_passes(tmp_path, monkeypatch):
    bam = _write_daf_bam(tmp_path / "md.bam", "md")
    out = tmp_path / "probs_out"
    monkeypatch.setattr(sys, "argv", [
        "fiberhmm-probs", "-a", bam, "-u", bam, "-o", str(out),
        "--mode", "pacbio-fiber", "-k", "3", "-q", "0",
        "--min-read-length", "100"])
    with pytest.raises(SystemExit) as exc:
        generate_probs.main()
    assert exc.value.code == 1


def test_transfer_daf_counts_deaminations(tmp_path):
    """transfer --mode daf used process_read on C/G counters, so deaminations
    (on T/A) never counted: P(m) estimates were 0 for every context."""
    bam = _write_daf_bam(tmp_path / "mm.bam", "mmml")
    counters = fh_utils._process_target_bam(bam, "daf", 3, _probs_args())
    assert set(counters) == {"C"}
    assert counters["C"].total_modified == 80
    bam = _write_daf_bam(tmp_path / "iupac.bam", "iupac")
    counters = fh_utils._process_target_bam(bam, "daf", 3, _probs_args())
    assert counters["C"].total_modified == 80


def test_transfer_exits_nonzero_on_zero_target_reads(tmp_path):
    bam = _write_daf_bam(tmp_path / "md.bam", "md")
    with pytest.raises(SystemExit) as exc:
        fh_utils._process_target_bam(bam, "pacbio-fiber", 3, _probs_args())
    assert exc.value.code == 1


def test_train_samples_daf_iupac_reads(tmp_path):
    bam = _write_daf_bam(tmp_path / "iupac.bam", "iupac", n_reads=3)
    reads = train.sample_reads_indexed(
        bam, n_samples=2, seed=1, mode="daf", min_mapq=0,
        prob_threshold=128, min_read_length=100)
    assert len(reads) == 2
    for r in reads:
        assert r.daf_strand == "+"
        assert "Y" not in r.query_sequence and len(r.m6a_query_positions) == 40


def test_train_exits_nonzero_when_no_read_passes(tmp_path, monkeypatch):
    bam = _write_daf_bam(tmp_path / "md.bam", "md")
    probs = tmp_path / "p.tsv"
    probs.write_text("encode\tratio\n0\t0.5\n")
    monkeypatch.setattr(sys, "argv", [
        "fiberhmm-train", "-i", bam, "-p", str(probs), str(probs),
        "-o", str(tmp_path / "out"), "--mode", "pacbio-fiber", "-r", "2",
        "-c", "1", "-q", "0", "--min-read-length", "100"])
    with pytest.raises(SystemExit) as exc:
        train.main()
    assert exc.value.code == 1


def test_train_saves_json_without_fake_npz(tmp_path, monkeypatch):
    k = 1
    n = 4 ** (2 * k)
    probs = tmp_path / "p.tsv"
    probs.write_text("encode\tratio\n" + "".join(f"{i}\t0.5\n" for i in range(n)))
    base = FiberHMM()
    base.startprob_ = np.array([0.5, 0.5])
    base.transmat_ = np.array([[0.9, 0.1], [0.1, 0.9]])
    base.emissionprob_ = np.full((2, 2 * n + 2), 1.0 / (2 * n + 2))
    from fiberhmm.core.model_io import save_model
    save_model(base, str(tmp_path / "base.json"), context_size=k)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "fiberhmm-train", "-p", str(probs), str(probs), "-o", str(out),
        "-k", str(k), "--base-model", str(tmp_path / "base.json")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        train.main()
    assert (out / "best-model.json").exists()
    assert not (out / "best-model.npz").exists()


def _dead_column_model():
    m = FiberHMM()
    m.startprob_ = np.array([0.5, 0.5])
    m.transmat_ = np.array([[0.95, 0.05], [0.05, 0.95]])
    m.emissionprob_ = np.array([[0.1, 0.9, 0.0], [0.8, 0.2, 0.0]])
    return m


def test_all_zero_emission_column_does_not_poison_posteriors():
    m = _dead_column_model()
    obs = np.array([0, 0, 1, 2, 1, 0, 1, 1] * 50)
    post = m.predict_proba(obs)
    assert np.all(np.isfinite(post))
    assert np.isfinite(m.score(obs))
    # the neutral symbol leaves the rest of the read as without it
    without = m.predict_proba(np.delete(obs, np.where(obs == 2)[0]))
    assert np.all(np.isfinite(without))


def test_zero_in_one_state_is_unchanged():
    m = _dead_column_model()
    m.emissionprob_ = np.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    m._compute_log_probs()
    assert np.isneginf(m._log_emissionprob[0, 2])


def test_train_model_uses_per_read_lengths(monkeypatch):
    seen = []
    orig = FiberHMM.fit

    def spy(self, X, lengths=None, **kw):
        seen.append(list(lengths))
        return orig(self, X, lengths=lengths, **kw)

    monkeypatch.setattr(FiberHMM, "fit", spy)
    emis = np.array([[0.1, 0.9], [0.8, 0.2]])
    data = {0: np.array([0, 1, 1, 0, 0, 1, 0])}
    train_model(emis, data, n_iterations=1, train_lengths={0: [3, 4]})
    assert seen == [[3, 4]]


def test_train_model_never_selects_non_finite_logprob():
    emis = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    # every observation impossible in the (hmmlearn-style) raw sense is made
    # neutral, so training still yields a finite best model
    best, _ = train_model(emis, {0: np.array([2, 2, 0, 1])}, n_iterations=1)
    assert best is not None
    assert np.all(np.isfinite(best.transmat_))


def test_generate_training_arrays_returns_lengths():
    reads = []
    for i, L in enumerate((50, 70)):
        fr = SimpleNamespace(query_sequence="ACGTTA" * (L // 6) + "A" * (L % 6),
                             m6a_query_positions={3, 9}, read_id=f"r{i}",
                             is_reverse=False)
        reads.append(fr)
    _, _, encoded, _, lengths = train.generate_training_arrays(
        reads, 0, 2, "pacbio-fiber", 3)
    assert sorted(lengths[0]) == sorted(len(e) for e in encoded)
    assert sorted(lengths[1]) == [50, 70]


def test_extract_training_read_guards_hard_clipped_mm():
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"LN": REF_LEN, "SN": "chr1"}]})
    a = pysam.AlignedSegment(header)
    a.query_name = "clip"
    a.query_sequence = "ACGTA" * 20
    a.reference_id = 0
    a.reference_start = 10
    a.cigartuples = [(5, 40), (0, 100)]
    a.set_tag("MM", "A+a.,0,0;")
    a.set_tag("ML", [250, 250])
    assert extract_training_read(a, "pacbio-fiber", 128) == "mm_not_applicable"
    a.cigartuples = [(0, 100)]
    ext = extract_training_read(a, "pacbio-fiber", 128)
    assert isinstance(ext, TrainingRead) and len(ext.mod_positions) == 2


def test_question_flag_bases_are_not_counted():
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"LN": REF_LEN, "SN": "chr1"}]})
    a = pysam.AlignedSegment(header)
    a.query_name = "q"
    a.query_sequence = _reference()[:600]
    a.reference_id = 0
    a.reference_start = 0
    a.cigartuples = [(0, 600)]
    a.set_tag("MM", "A+a?,10,10;")
    a.set_tag("ML", [250, 20])
    ext = extract_training_read(a, "nanopore-fiber", 128)
    assert len(ext.mod_positions) == 1 and ext.unknown_positions
    listed = ContextCounter(3, "A")
    listed.process_read(ext.sequence, ext.mod_positions, 10,
                        skip_positions=ext.unknown_positions)
    naive = ContextCounter(3, "A")
    naive.process_read(ext.sequence, ext.mod_positions, 10)
    assert listed.total_modified == naive.total_modified == 1
    assert listed.total_positions == 2
    assert naive.total_positions > 50


def _write_footprint_reference_bam(path, n_reads=6):
    """Fiber-seq-like reference: forward reads with ns/nl footprints."""
    header = pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6", "SO": "coordinate"},
         "SQ": [{"LN": REF_LEN, "SN": "chr1"}]})
    ref = _reference()
    with pysam.AlignmentFile(str(path), "wb", header=header) as out:
        for i in range(n_reads):
            start = i * 150
            a = pysam.AlignedSegment(header)
            a.query_name = f"ref{i}"
            a.query_sequence = ref[start:start + READ_LEN]
            a.flag = 0
            a.reference_id = 0
            a.reference_start = start
            a.mapping_quality = 60
            a.cigartuples = [(0, READ_LEN)]
            starts = list(range(20 + 7 * i, READ_LEN - 200, 200))
            a.set_tag("ns", starts)
            a.set_tag("nl", [147] * len(starts))
            out.write(a)
    pysam.index(str(path))
    return str(path)


def test_transfer_command_runs_end_to_end(tmp_path, monkeypatch):
    """`fiberhmm-utils transfer` always stopped with KeyError: 'total'
    (ContextCounter tables carry hit/nohit, not total)."""
    target = _write_daf_bam(tmp_path / "target.bam", "iupac", n_reads=3)
    reference = _write_footprint_reference_bam(tmp_path / "reference.bam")
    common = ["-t", target, "-k", "1", "2", "--min-observations", "1",
              "-q", "0", "--min-read-length", "100"]
    out = tmp_path / "from_bam"
    monkeypatch.setattr(sys, "argv", [
        "fiberhmm-utils", "transfer", "-rb", reference, "-o", str(out), *common])
    fh_utils.main()
    import pandas as pd
    tables = out / "tables"
    bam_k1 = pd.read_csv(tables / "from_bam_C_k1_probs.tsv", sep="\t")
    assert len(bam_k1) == 16
    assert (bam_k1["accessible_prob"] > 0).all()

    # Saved priors (written at the largest k) reproduce every smaller k.
    priors = tables / "from_bam_accessibility_priors_C_k2.tsv"
    out2 = tmp_path / "from_priors"
    monkeypatch.setattr(sys, "argv", [
        "fiberhmm-utils", "transfer", "-ap", str(priors), "-o", str(out2), *common])
    fh_utils.main()
    for k in (1, 2):
        a = pd.read_csv(tables / f"from_bam_C_k{k}_probs.tsv", sep="\t")
        b = pd.read_csv(out2 / "tables" / f"from_priors_C_k{k}_probs.tsv", sep="\t")
        pd.testing.assert_frame_equal(a, b)


def test_transfer_regression_uses_hit_plus_nohit_as_weight():
    import pandas as pd
    contexts = [f"A{b}C{c}" for b in "ACGT" for c in "ACGT"][:12]
    x = np.linspace(0.05, 0.95, len(contexts))
    rates = pd.DataFrame({"context": contexts, "hit": (100 * (0.1 + 0.5 * x)).astype(int),
                          "nohit": 100 - (100 * (0.1 + 0.5 * x)).astype(int)})
    rates["ratio"] = rates["hit"] / 100
    priors = pd.DataFrame({"context": contexts, "accessible_bp": (1000 * x).astype(int),
                           "total_bp": 1000, "p_accessible": x})
    p_acc, p_inacc, diag = fh_utils._estimate_emission_probs(rates, priors, 50)
    assert diag["n_contexts"] == 12 and diag["total_target_obs"] == 1200
    assert p_acc == pytest.approx(0.6, abs=0.02)
    assert p_inacc == pytest.approx(0.1, abs=0.02)
