"""MM '?' semantics in standalone recall, strand rescue and training (3.0 verify HIGH).

Regressions for the Codex verification pass on b006409.
"""
from __future__ import annotations

import array
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pysam
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(module, *args, stdin=None, timeout=300):
    env = dict(os.environ)
    env.setdefault("FIBERHMM_NO_UPDATE_CHECK", "1")
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        cwd=REPO_ROOT, capture_output=True, timeout=timeout, env=env,
        stdin=stdin,
    )


def _bundled(name):
    from fiberhmm.models import _bundled_model_path

    return _bundled_model_path(name)


def _pg_ds(path, program):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        records = [
            record for record in bam.header.to_dict().get("PG", [])
            if str(record.get("PN", "")).startswith(program)
        ]
    assert records, f"no {program} @PG record"
    return records[-1].get("DS", "")


def _ds_tokens(ds):
    return dict(
        token.split("=", 1) for token in ds.split() if "=" in token
    )


def _ma_tags(path):
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        return [
            (read.query_name, read.get_tag("MA") if read.has_tag("MA") else None)
            for read in bam.fetch(until_eof=True)
        ]


def _declare(source, target, declaration):
    from fiberhmm.io.bam_header import append_chemistry

    with pysam.AlignmentFile(str(source), "rb", check_sq=False) as bam:
        header = append_chemistry(bam.header, declaration)
        with pysam.AlignmentFile(str(target), "wb", header=header) as out:
            for read in bam.fetch(until_eof=True):
                out.write(read)
    pysam.index(str(target))
    return target


# ---------------------------------------------------------------------------
# 2. MM '?' semantics
# ---------------------------------------------------------------------------

def _question_read(mm="A+a?,4,89;", ml=(255, 255)):
    read = pysam.AlignedSegment()
    read.query_name = "unknown"
    read.query_sequence = "CCCA" * 100
    read.flag = 0
    read.reference_id = 0
    read.reference_start = 0
    read.cigartuples = [(0, 400)]
    read.set_tag("as", array.array("I", [0]))
    read.set_tag("al", array.array("I", [400]))
    read.set_tag("ML", array.array("B", list(ml)))
    read.set_tag("MM", mm)
    return read


@pytest.fixture(scope="module")
def hia5_ont_tables():
    from fiberhmm.core.model_io import load_model_with_metadata
    from fiberhmm.inference.tf_recaller import build_llr_tables

    model, _, _ = load_model_with_metadata(_bundled("hia5_nanopore.json"))
    return model, build_llr_tables(model)


def _correct_obs(read, threshold=248, mode="nanopore-fiber"):
    from fiberhmm.core.bam_reader import (
        encode_from_query_sequence,
        parse_mm_tag_query_calls,
    )

    mods, unknown = parse_mm_tag_query_calls(
        read.get_tag("MM"), read.get_tag("ML"), read.query_sequence,
        bool(read.is_reverse), threshold, mode)
    return encode_from_query_sequence(
        read.query_sequence, mods, 10, mode=mode, context_size=3,
        is_reverse=bool(read.is_reverse), unknown_positions=unknown), unknown


def test_standalone_tf_recall_treats_unlisted_question_bases_as_unknown(
        hia5_ont_tables):
    from fiberhmm.inference.tf_recaller import call_tfs_in_interval, recall_read

    _model, (hit, miss) = hia5_ont_tables
    read = _question_read()
    correct, unknown = _correct_obs(read)
    assert len(unknown) == 98
    expected = call_tfs_in_interval(correct, 0, 400, hit, miss, 5.0, 3)
    observed = recall_read(read, hit, miss, "nanopore-fiber", 3, 5.0, 3, 90,
                           prob_threshold=248)[0]
    assert observed == expected == []
    # The same listed calls under '.' are complete: the unlisted As are misses.
    read.set_tag("MM", "A+a.,4,89;")
    dot = recall_read(read, hit, miss, "nanopore-fiber", 3, 5.0, 3, 90,
                      prob_threshold=248)[0]
    assert dot != observed


def test_standalone_nuc_recall_carries_question_unknowns(monkeypatch,
                                                          hia5_ont_tables):
    from fiberhmm.cli import recall_tfs

    _model, (hit, miss) = hia5_ont_tables
    captured = {}

    def fake_fused(fiber_read, apply_result, *args, **kwargs):
        captured["encoded"] = np.array(apply_result["encoded"])
        return {"tf_calls": [], "nl": []}

    monkeypatch.setattr(recall_tfs, "build_fused_recall_result", fake_fused)
    nuc_cfg = recall_tfs._NucCfg(
        recall_nucs=True, split_min_llr=None, split_min_opps=None,
        nuc_min_size=85, msp_min_size=0, phase_nrl=0,
        nuc_recall_policy="topology")
    recall_tfs._worker_init(hit, miss, "nanopore-fiber", 3, 5.0, 3, 90,
                            nuc_cfg=nuc_cfg, prob_threshold=248)
    read = _question_read()
    recall_tfs._process_payload_record(recall_tfs._make_payload(read))
    correct, _unknown = _correct_obs(read)
    np.testing.assert_array_equal(captured["encoded"], correct)


def test_empty_ml_question_spec_means_every_target_is_unknown():
    from fiberhmm.core.bam_reader import parse_mm_tag_query_calls

    sequence = "CCCA" * 4
    mods, unknown = parse_mm_tag_query_calls(
        "A+a?;", b"", sequence, False, 248, "nanopore-fiber")
    assert mods == set()
    assert unknown == {i for i, base in enumerate(sequence) if base == "A"}
    # '.' with empty ML: every target base is a known miss (unchanged).
    assert parse_mm_tag_query_calls(
        "A+a.;", b"", sequence, False, 248, "nanopore-fiber") == (set(), set())
    # ML absent entirely behaves like an empty ML.
    assert parse_mm_tag_query_calls(
        "A+a?;", None, sequence, False, 248, "nanopore-fiber")[1] == unknown


def test_empty_ml_question_training_counts_no_sites():
    from fiberhmm.probabilities.context_counter import ContextCounter
    from fiberhmm.probabilities.utils import extract_training_read

    read = _question_read(mm="A+a?;", ml=())
    extracted = extract_training_read(read, "nanopore-fiber", 248)
    assert len(extracted.unknown_positions) == 100
    counter = ContextCounter(3, "A")
    counter.process_read(extracted.sequence, extracted.mod_positions, 10,
                         skip_positions=extracted.unknown_positions)
    assert counter.total_positions == 0 and counter.total_modified == 0


def test_strand_rescue_encoding_carries_question_unknowns():
    from fiberhmm.inference import strand_rescue

    read = _question_read()
    observations, strand = strand_rescue.hard_observations(
        read, "fiber", "nanopore-fiber", 3, 248)
    correct, _unknown = _correct_obs(read)
    np.testing.assert_array_equal(observations, correct)
    assert strand == "FWD"


