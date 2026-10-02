"""Core audit package 6: model validation and numerical edge cases.

- 1.4: a custom k!=3 model passed validation, but TF/nucleosome recall read its
  table at the k=3 offsets and silently gave wrong calls; it is now refused.
- 1.5: 5mC (gpc/cpg) reverse-aligned reads encoded each target G by its
  forward (G-centred) context instead of the C-centred code the same cytosine
  gets on a forward read.
- 1.7: forward/backward gave NaN when both incoming paths were -inf (a zero
  start or transition probability).
"""
import json
import random

import numpy as np
import pytest

import fiberhmm.core.bam_reader as br
from fiberhmm.core.hmm import FiberHMM
from fiberhmm.core.model_io import load_model
from fiberhmm.inference.tf_recaller import (
    build_conditional_hit_tables,
    build_llr_tables,
    require_recall_table,
)


def _model_file(path, k):
    n = 4 ** (2 * k)
    row0 = np.concatenate([np.full(n, .1), [1.0], np.full(n, .9), [1.0]])
    row1 = np.concatenate([np.full(n, .8), [1.0], np.full(n, .2), [1.0]])
    emissions = np.vstack([row0 / row0.sum(), row1 / row1.sum()])
    path.write_text(json.dumps({
        "model_type": "FiberHMM", "version": "2.0", "n_states": 2,
        "startprob": [1.0, 0.0], "transmat": [[0.99, 0.01], [0.07, 0.93]],
        "emissionprob": emissions.tolist(), "context_size": k, "mode": "pacbio-fiber"}))
    return str(path)


def test_recall_refuses_a_k4_table(tmp_path):
    model = load_model(_model_file(tmp_path / "k4.json", 4))
    with pytest.raises(ValueError, match=r"k=3 emission table .*131074 columns .*k=4"):
        build_llr_tables(model)
    with pytest.raises(ValueError, match="k=3"):
        build_conditional_hit_tables(model)


def test_recall_accepts_k3_tables_with_or_without_the_trailing_column(tmp_path):
    model = load_model(_model_file(tmp_path / "k3.json", 3))
    hit, miss = build_llr_tables(model)
    assert hit.shape == miss.shape == (4096,)
    # P(hit|protected)=0.1 vs 0.8, P(miss)=0.9 vs 0.2 (the audit's k=4 case gave
    # a miss LLR of -2.079 instead of +1.504).
    assert hit[0] == pytest.approx(np.log(.1 / .8))
    assert miss[0] == pytest.approx(np.log(.9 / .2))
    assert require_recall_table(np.ones((2, 8193))).shape == (2, 8193)
    for bad in ((2, 8192), (2, 8195), (3, 8194)):
        with pytest.raises(ValueError):
            require_recall_table(np.ones(bad))


_RC = str.maketrans("ACGTN", "TGCAN")


def _rc(sequence):
    return sequence.translate(_RC)[::-1]


@pytest.mark.parametrize("motif", ["gpc", "cpg"])
@pytest.mark.parametrize("numba", [True, False])
def test_5mc_reverse_reads_are_the_mirror_of_forward_reads(motif, numba, monkeypatch):
    if not numba:
        monkeypatch.setattr(br, "_HAS_NUMBA", False)
    rng = random.Random(7)
    for _trial in range(150):
        n = rng.randint(0, 120)
        k = rng.choice([1, 2, 3])
        trim = rng.choice([0, 1, 3, 10])
        sequence = "".join(rng.choice("ACGT") for _ in range(n))
        mods = {i for i, base in enumerate(sequence) if base == "C" and rng.random() < .5}
        forward = br.encode_from_query_sequence(
            sequence, mods, edge_trim=trim, mode=motif, context_size=k, is_reverse=False)
        reverse = br.encode_from_query_sequence(
            _rc(sequence), {n - 1 - i for i in mods}, edge_trim=trim, mode=motif,
            context_size=k, is_reverse=True)
        np.testing.assert_array_equal(forward, reverse[::-1])


def test_5mc_reverse_gpc_gets_the_forward_code():
    sequence = "AAAAAAAAAAAAATGCAAAAAAAAAAAA"  # GpC: the C at 14
    c = sequence.index("GC") + 1
    n = len(sequence)
    forward = br.encode_from_query_sequence(sequence, {c}, edge_trim=0, mode="gpc",
                                            context_size=1, is_reverse=False)
    reverse = br.encode_from_query_sequence(_rc(sequence), {n - 1 - c}, edge_trim=0,
                                            mode="gpc", context_size=1, is_reverse=True)
    assert forward[c] == reverse[n - 1 - c]


def test_hmm_unreachable_state_gives_minus_inf_not_nan():
    model = FiberHMM(2)
    model.startprob_ = np.array([1.0, 0.0])
    model.transmat_ = np.eye(2)
    model.emissionprob_ = np.array([[.2, .8], [.8, .2]])
    obs = np.array([0, 0, 1])
    assert model.score(obs) == pytest.approx(np.log(.2 * .2 * .8))
    posteriors = model.predict_proba(obs)
    assert np.isfinite(posteriors).all()
    np.testing.assert_allclose(posteriors, [[1, 0], [1, 0], [1, 0]])
    alpha, _ = model._forward(obs)
    beta = model._backward(obs)
    assert not np.isnan(alpha).any() and not np.isnan(beta).any()
    assert np.isneginf(alpha[:, 1]).all()


@pytest.mark.parametrize("mode", [[], ["--region-parallel"]])
def test_call_refuses_a_k4_model_before_starting_workers(tmp_path, mode):
    """With --phase-nrl off the tables were first built in worker
    initializers, where the refusal hung the streaming pool."""
    from test_call_entrypoint_regressions import _run_cli, make_region_test_bam

    bam = make_region_test_bam(tmp_path / "in.bam", seed=3)
    output = tmp_path / "out.bam"
    result = _run_cli("fiberhmm.cli.call", "-i", bam, "-o", output,
                      "-m", _model_file(tmp_path / "k4.json", 4), "-k", "4",
                      "--enzyme", "hia5", "--seq", "pacbio", "--phase-nrl", "off",
                      "--no-qc", "--no-recall-nucs", "-c", "2", "--io-threads", "1",
                      *mode, timeout=120)
    assert result.returncode != 0
    assert b"needs a k=3" in result.stderr
    assert not output.exists()
