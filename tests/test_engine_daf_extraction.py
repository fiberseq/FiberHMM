"""Engine read extraction: chimera filter on R/Y input, SNP mask on the MM/ML
DAF path, and MM ``?`` unknown bases (fh-core-daf, 3.0 release fixes)."""
from __future__ import annotations

import random

import pysam
import pytest

from fiberhmm.daf.encoder import encode_read_daf
from fiberhmm.inference import engine
from fiberhmm.inference.engine import (
    CHIMERA_SKIP,
    _extract_fiber_read_from_pysam,
    configure_daf_chimera_filter,
    configure_daf_snp_mask,
    extract_fiber_read_from_payload,
    make_apply_payload,
)

READ_LEN = 600
REF_START = 1_000


def _header():
    return pysam.AlignmentHeader.from_dict(
        {"HD": {"VN": "1.6"}, "SQ": [{"LN": 100_000, "SN": "chr1"}]})


def _reference(seed=7):
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(READ_LEN))


def _md(ref, query):
    out, run = [], 0
    for r, q in zip(ref, query):
        if r == q:
            run += 1
        else:
            out.append(f"{run}{r}")
            run = 0
    out.append(str(run))
    return "".join(out)


def _segment(seq, tags=(), flag=0):
    a = pysam.AlignedSegment(_header())
    a.query_name = "r"
    a.query_sequence = seq
    a.flag = flag
    a.reference_id = 0
    a.reference_start = REF_START
    a.mapping_quality = 60
    a.cigartuples = [(0, len(seq))]
    a.set_tags(list(tags))
    return a


def _chimera_read(ref, n_ct=8, n_ga=7):
    """Raw DAF read: C->T in the first half, G->A in the second half."""
    q = list(ref)
    c_sites = [i for i, b in enumerate(ref) if b == "C" and i < READ_LEN // 2][:n_ct]
    g_sites = [i for i, b in enumerate(ref) if b == "G" and i >= READ_LEN // 2][:n_ga]
    for i in c_sites:
        q[i] = "T"
    for i in g_sites:
        q[i] = "A"
    q = "".join(q)
    return _segment(q, [("MD", _md(ref, q), "Z")]), c_sites, g_sites


@pytest.fixture(autouse=True)
def _reset_engine_config():
    configure_daf_chimera_filter(True, 5, 0.8)
    configure_daf_snp_mask(None)
    yield
    configure_daf_chimera_filter(True, 5, 0.8)
    configure_daf_snp_mask(None)


def _rewrite_iupac(raw):
    new_seq, st, _n = encode_read_daf(raw)
    tags = [("MD", raw.get_tag("MD"), "Z"), ("st", st, "Z")]
    return _segment(new_seq, tags), st


def test_md_input_chimera_is_filtered():
    raw, _, _ = _chimera_read(_reference())
    assert _extract_fiber_read_from_pysam(raw, "daf", 128) is CHIMERA_SKIP


def test_iupac_input_chimera_is_filtered_like_md_input():
    """daf-encode keeps only the dominant flavour as Y/R; the other flavour is
    still visible through MD. Before 3.0 the R/Y branch returned before the
    chimera check, so the same molecule was filtered on raw input and called
    on encoded input."""
    raw, _, _ = _chimera_read(_reference())
    enc, st = _rewrite_iupac(raw)
    assert st == "CT" and "R" not in enc.query_sequence
    assert _extract_fiber_read_from_pysam(enc, "daf", 128) is CHIMERA_SKIP
    # slim-IPC path (streaming): the main process ships the raw mismatches
    payload = make_apply_payload(enc, mode="daf")
    assert extract_fiber_read_from_payload(payload, "daf", 128) is CHIMERA_SKIP


def test_iupac_chimera_without_md_uses_both_iupac_flavours():
    raw, c_sites, g_sites = _chimera_read(_reference())
    q = list(raw.query_sequence)
    for i in c_sites:
        q[i] = "Y"
    for i in g_sites:
        q[i] = "R"
    both = _segment("".join(q), [("st", "CT", "Z")])
    assert _extract_fiber_read_from_pysam(both, "daf", 128) is CHIMERA_SKIP


def test_keep_chimeras_and_thresholds_apply_to_iupac_input():
    raw, c_sites, _ = _chimera_read(_reference())
    enc, _ = _rewrite_iupac(raw)
    configure_daf_chimera_filter(False)
    fr = _extract_fiber_read_from_pysam(enc, "daf", 128)
    assert isinstance(fr, dict) and fr["m6a_query_positions"] == set(c_sites)
    # --chimera-min-seg above the minority count: not a chimera
    configure_daf_chimera_filter(True, min_seg=8)
    assert isinstance(_extract_fiber_read_from_pysam(enc, "daf", 128), dict)


def test_clean_iupac_read_unchanged():
    ref = _reference()
    raw, c_sites, _ = _chimera_read(ref, n_ct=8, n_ga=0)
    enc, _ = _rewrite_iupac(raw)
    fr = _extract_fiber_read_from_pysam(enc, "daf", 128)
    assert fr["m6a_query_positions"] == set(c_sites)
    assert fr["_daf_strand"] == "+"
    payload = make_apply_payload(enc, mode="daf")
    fr2 = extract_fiber_read_from_payload(payload, "daf", 128)
    assert fr2["m6a_query_positions"] == set(c_sites)


def _mm_tag(sequence, positions, base, code):
    targets = [i for i, b in enumerate(sequence) if b == base]
    skips, last = [], -1
    for pos in sorted(positions):
        idx = targets.index(pos)
        skips.append(idx - last - 1)
        last = idx
    return f"{base}+{code}.," + ",".join(map(str, skips)) + ";"


def test_snp_mask_applies_to_mm_ml_daf_calls(tmp_path):
    ref = _reference()
    c_sites = [i for i, b in enumerate(ref) if b == "C"][10:20]
    q = list(ref)
    for i in c_sites:
        q[i] = "T"
    q = "".join(q)
    read = _segment(q, [("MM", _mm_tag(q, c_sites, "T", "u")),
                        ("ML", [240] * len(c_sites))])
    fr = _extract_fiber_read_from_pysam(read, "daf", 128)
    assert fr["m6a_query_positions"] == set(c_sites)

    masked = c_sites[3]
    bed = tmp_path / "mask.bed"
    bed.write_text(f"chr1\t{REF_START + masked}\t{REF_START + masked + 1}\n")
    configure_daf_snp_mask(str(bed))
    fr = _extract_fiber_read_from_pysam(read, "daf", 128)
    assert fr["m6a_query_positions"] == set(c_sites) - {masked}
    payload = make_apply_payload(read, mode="daf")
    fr2 = extract_fiber_read_from_payload(payload, "daf", 128)
    assert fr2["m6a_query_positions"] == set(c_sites) - {masked}


@pytest.mark.parametrize("flag", [0, 16])
def test_question_flag_unknowns_reach_encoder(flag):
    ref = _reference()
    walk = ref if flag == 0 else ref.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    n_a = walk.count("A")
    sparse = _segment(ref, [("MM", "A+a?,3,5;"), ("ML", [250, 250])], flag=flag)
    dense_skips = ",".join(["0"] * n_a)
    ml = [0] * n_a
    a_idx = 3
    ml[a_idx] = 250
    ml[a_idx + 6] = 250
    dense = _segment(ref, [("MM", f"A+a.,{dense_skips};"), ("ML", ml)], flag=flag)

    fr_sparse = _extract_fiber_read_from_pysam(sparse, "nanopore-fiber", 128)
    fr_dense = _extract_fiber_read_from_pysam(dense, "nanopore-fiber", 128)
    assert fr_sparse["m6a_query_positions"] == fr_dense["m6a_query_positions"]
    assert "unknown_query_positions" not in fr_dense
    assert len(fr_sparse["unknown_query_positions"]) == n_a - 2
    # survives the slim-IPC payload round trip
    payload = make_apply_payload(sparse, mode="nanopore-fiber")
    fr_payload = extract_fiber_read_from_payload(payload, "nanopore-fiber", 128)
    assert fr_payload["unknown_query_positions"] == fr_sparse["unknown_query_positions"]

    class _Model:
        def __init__(self):
            self.seen = []

        def predict(self, obs):
            self.seen.append(obs.copy())
            import numpy as np
            return np.ones(len(obs), dtype=np.int8)

    m = _Model()
    for fr in (fr_sparse, fr_dense):
        engine._process_single_read(fr, m, 0, False, "nanopore-fiber", 3, 1, False,
                                    include_encoded=True)
    sparse_obs, dense_obs = m.seen
    unknown = sorted(fr_sparse["unknown_query_positions"])
    non_target_obs = 2 * 4 ** 6 + 1
    interior = [p for p in unknown if 3 <= p < READ_LEN - 3]
    assert interior
    assert all(sparse_obs[p] == non_target_obs for p in interior)
    assert all(dense_obs[p] != non_target_obs for p in interior)
