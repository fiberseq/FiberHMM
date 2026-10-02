"""State-aware (in-MSP / outside-MSP) modification rates in fiberhmm-qc."""
from __future__ import annotations

import array
import json
from pathlib import Path

import numpy as np
import pysam
import pytest

from fiberhmm.qc import states
from fiberhmm.qc.core import _overall_verdict, load_references, run_qc
from fiberhmm.qc.states import (
    compute_state_rates,
    grade_state_rates,
    msp_intervals_from_tags,
    terminal_exclusion,
)

COMPLEMENT = str.maketrans("ACGT", "TGCA")
L = 3000
# SEQ-frame MSPs of the planted molecule; (1000, 50) is a linker-sized gap
# that the QC counts as outside-MSP (< 85 bp).
PLANTED_MSPS = [(300, 400), (1000, 50), (1200, 300), (2200, 150)]
EDGE = 10  # fiberhmm-call --edge-trim (k=3 < 10)
CAP = 85   # terminal segments excluded up to min_msp_bp from each read end


def _revcomp(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def _sequence(seed: int = 5) -> str:
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), size=L))


def _in_planted_msp(position: int, min_msp: int = 85) -> bool:
    return any(start <= position < start + length
               for start, length in PLANTED_MSPS if length >= min_msp)


def _planted_mods(sequence: str, targets: str = "AT") -> set[int]:
    """SEQ positions: every 2nd in-MSP target, every 20th other target."""
    mods, inside, outside = set(), 0, 0
    for position, base in enumerate(sequence):
        if base not in targets:
            continue
        if _in_planted_msp(position):
            inside += 1
            if inside % 2 == 0:
                mods.add(position)
        else:
            outside += 1
            if outside % 20 == 0:
                mods.add(position)
    return mods


def _expected(sequence: str, mods: set[int], targets: str = "AT") -> dict:
    """Counts written out independently of the QC code."""
    counts = {"msp": [0, 0], "outside_msp": [0, 0]}
    first_msp, last_msp_end = PLANTED_MSPS[0][0], sum(PLANTED_MSPS[-1])
    for position, base in enumerate(sequence):
        if base not in targets or not (EDGE <= position < L - EDGE):
            continue
        if position < min(first_msp, CAP) or position >= max(last_msp_end, L - CAP):
            continue  # terminal (read-end-truncated) segment, capped
        key = "msp" if _in_planted_msp(position) else "outside_msp"
        counts[key][0] += position in mods
        counts[key][1] += 1
    return counts


def _header(coord_molecular: bool = True):
    ds = "FiberHMM fused apply+recall; mode=pacbio-fiber enzyme=hia5"
    if coord_molecular:
        ds += "; coord=molecular"
    return pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "coordinate"},
        "SQ": [{"SN": "chr1", "LN": 1_000_000}],
        "PG": [{"ID": "fiberhmm-call", "PN": "fiberhmm-call", "DS": ds,
                "CL": "fiberhmm-call -i a.bam -o b.bam --enzyme hia5 --seq pacbio"}],
        "CO": ["FIBERHMM-CHEMISTRY:v1:assay=fiber-seq;enzyme=hia5;"
               "platform=pacbio;mode=pacbio-fiber"],
    })


def _pacbio_read(header, sequence, mods, *, reverse, name, start=0,
                 msps=None, frame="molecular"):
    """PacBio m6A read; ``sequence``/``mods``/``msps`` in SEQ frame."""
    original = _revcomp(sequence) if reverse else sequence
    to_original = (lambda p: len(sequence) - 1 - p) if reverse else (lambda p: p)
    original_mods = {to_original(p) for p in mods}
    mm, ml = [], []
    for base, strand in (("A", "+"), ("T", "-")):
        listed = [i for i, b in enumerate(original) if b == base]
        mm.append(f"{base}{strand}a." + "".join(",0" for _ in listed))
        ml += [255 if i in original_mods else 0 for i in listed]
    read = pysam.AlignedSegment(header)
    read.query_name = name
    read.query_sequence = sequence
    read.flag = 16 if reverse else 0
    read.reference_id = 0
    read.reference_start = start
    read.mapping_quality = 60
    read.cigartuples = [(0, len(sequence))]
    read.set_tag("MM", ";".join(mm) + ";")
    read.set_tag("ML", ml)
    if msps is not None:
        intervals = msps
        if frame == "molecular" and reverse:
            intervals = sorted((len(sequence) - (s + n), n) for s, n in msps)
        body = ",".join(f"{s + 1}-{n}" for s, n in intervals)
        read.set_tag("MA", f"{len(sequence)};msp.:{body}")
    return read


@pytest.mark.parametrize("reverse", [False, True])
def test_called_path_counts_planted_per_state_rates_exactly(reverse):
    header = _header()
    molecule = _sequence()
    sequence = _revcomp(molecule) if reverse else molecule
    # Plant in molecule coordinates, then express everything in SEQ frame.
    mods_molecule = _planted_mods(molecule)
    seq_mods = ({L - 1 - p for p in mods_molecule} if reverse else mods_molecule)
    seq_msps = ([(L - (s + n), n) for s, n in PLANTED_MSPS] if reverse
                else PLANTED_MSPS)
    read = _pacbio_read(header, sequence, seq_mods, reverse=reverse,
                        name="r", msps=seq_msps)
    block, arrays = compute_state_rates([read], "pacbio-fiber", "hia5",
                                        header=header, prob_threshold=125)
    assert block["source"] == "tags" and block["available"]
    expected = _expected(molecule, mods_molecule)
    for key in ("msp", "outside_msp"):
        events, opportunities = expected[key]
        assert block[key]["n_events"] == events
        assert block[key]["n_opportunities"] == opportunities
        assert block[key]["aggregate_rate"] == pytest.approx(events / opportunities)
    assert block["msp"]["aggregate_rate"] == pytest.approx(0.5, abs=0.01)
    assert block["outside_msp"]["aggregate_rate"] == pytest.approx(0.05, abs=0.01)
    assert block["msp_length_fraction"] == pytest.approx(850 / L)
    # Every target counted once overall (edge trim only).
    all_targets = sum(1 for p, b in enumerate(molecule)
                      if b in "AT" and EDGE <= p < L - EDGE)
    assert block["all_states"]["n_opportunities"] == all_targets
    assert block["msp_to_outside_ratio"] == pytest.approx(
        block["msp"]["aggregate_rate"] / block["outside_msp"]["aggregate_rate"])


def test_called_path_is_orientation_invariant():
    header = _header()
    molecule = _sequence(seed=8)
    mods = _planted_mods(molecule)
    forward = _pacbio_read(header, molecule, mods, reverse=False, name="f",
                           msps=PLANTED_MSPS)
    reverse = _pacbio_read(header, _revcomp(molecule), {L - 1 - p for p in mods},
                           reverse=True, name="r",
                           msps=[(L - (s + n), n) for s, n in PLANTED_MSPS])
    a, _ = compute_state_rates([forward], "pacbio-fiber", "hia5", header=header)
    b, _ = compute_state_rates([reverse], "pacbio-fiber", "hia5", header=header)
    for key in ("msp", "outside_msp", "all_states"):
        assert a[key] == b[key]


def test_daf_iupac_read_counts_deaminated_strand_targets_only():
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 100_000}],
        "PG": [{"ID": "fiberhmm-call", "PN": "fiberhmm-call",
                "DS": "mode=daf enzyme=dddb coord=molecular"}]})
    molecule = _sequence(seed=9)
    mods = _planted_mods(molecule, targets="C")
    encoded = "".join("Y" if p in mods else b for p, b in enumerate(molecule))
    read = pysam.AlignedSegment(header)
    read.query_name = "daf"
    read.query_sequence = encoded
    read.flag = 0
    read.reference_id = 0
    read.reference_start = 0
    read.mapping_quality = 60
    read.cigartuples = [(0, L)]
    read.set_tag("st", "CT")
    read.set_tag("MA", f"{L};msp.:" + ",".join(f"{s + 1}-{n}" for s, n in PLANTED_MSPS))
    block, _ = compute_state_rates([read], "daf", "dddb", header=header)
    expected = _expected(molecule, mods, targets="C")
    for key in ("msp", "outside_msp"):
        assert (block[key]["n_events"], block[key]["n_opportunities"]) == tuple(expected[key])
    # G (the other strand's target) never counts on a CT read.
    assert block["all_states"]["n_opportunities"] == sum(
        1 for p, b in enumerate(molecule) if b == "C" and EDGE <= p < L - EDGE)


def test_terminal_policies():
    mask = np.zeros(1000, dtype=bool)
    mask[300:700] = True
    assert terminal_exclusion(mask, "keep").sum() == 0
    capped = terminal_exclusion(mask, "cap", 85)
    assert capped[:85].all() and not capped[85:915].any() and capped[915:].all()
    dropped = terminal_exclusion(mask, "drop")
    assert dropped[:300].all() and not dropped[300:700].any() and dropped[700:].all()
    short_end = np.zeros(1000, dtype=bool)
    short_end[40:900] = True  # first segment (40 bp) shorter than the cap
    capped = terminal_exclusion(short_end, "cap", 85)
    assert capped[:40].all() and not capped[40:915].any()
    # A single-segment molecule keeps its interior under 'cap'.
    single = terminal_exclusion(np.ones(500, dtype=bool), "cap", 85)
    assert single.sum() == 170
    assert terminal_exclusion(np.ones(500, dtype=bool), "drop").all()


def test_wrapped_msp_split_by_an_keeps_its_joined_length():
    """A circular MSP across the origin is written as two MA pieces sharing an
    AN name; the 85-bp cut-off applies to the joined 100-bp feature."""
    header = _header()
    read = _pacbio_read(header, _sequence(), set(), reverse=False, name="c")
    read.set_tag("MA", f"{L};nuc.Q:300-147;msp.:{L - 49}-50,1-50")
    read.set_tag("AN", "n1,m1,m1")
    intervals = msp_intervals_from_tags(read, {"MA": "molecular", "legacy": None})
    mask = states.msp_mask(intervals, L, 85)
    assert mask.sum() == 100 and mask[:50].all() and mask[-50:].all()
    read.set_tag("AN", None)  # without AN the pieces are two 50-bp MSPs
    assert states.msp_mask(msp_intervals_from_tags(
        read, {"MA": "molecular", "legacy": None}), L, 85).sum() == 0


def test_unsplit_wrapped_interval_flips_onto_both_read_ends():
    header = _header()
    read = _pacbio_read(header, _sequence()[:1000], set(), reverse=True, name="w")
    read.set_tag("MA", "1000;msp.:951-100")  # molecular [950, 1050) wraps
    (start, length, _feature), = msp_intervals_from_tags(read, {"MA": "molecular"})
    mask = states.msp_mask([(start, length)], 1000, 85)
    assert mask[:50].all() and mask[950:].all() and mask.sum() == 100


def test_daf_snp_mask_removes_opportunities_without_touching_calling_state(tmp_path):
    from fiberhmm.inference import engine

    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 100_000}],
        "PG": [{"ID": "fiberhmm-call", "PN": "fiberhmm-call",
                "DS": "mode=daf enzyme=dddb coord=molecular daf_run_mask=off"}]})
    molecule = _sequence(seed=9)
    mods = _planted_mods(molecule, targets="C")
    read = pysam.AlignedSegment(header)
    read.query_name, read.flag, read.reference_id = "daf", 0, 0
    read.query_sequence = "".join("Y" if p in mods else b for p, b in enumerate(molecule))
    read.reference_start, read.mapping_quality = 1000, 60
    read.cigartuples = [(0, L)]
    read.set_tag("st", "CT")
    read.set_tag("MA", f"{L};msp.:" + ",".join(f"{s + 1}-{n}" for s, n in PLANTED_MSPS))
    masked_event = sorted(p for p in mods if _in_planted_msp(p) and p > CAP)[0]
    masked_plain = next(p for p, b in enumerate(molecule)
                        if b == "C" and p not in mods and _in_planted_msp(p) and p > CAP)
    snp = {"chr1": {1000 + masked_event, 1000 + masked_plain}}
    sentinel = {"chrX": {1}}
    plain, _ = compute_state_rates([read], "daf", "dddb", header=header)
    engine._DAF_SNP_MASK, previous = sentinel, engine._DAF_SNP_MASK
    try:
        block, _ = compute_state_rates([read], "daf", "dddb", header=header, snp_mask=snp)
        assert engine._DAF_SNP_MASK is sentinel
        # A calling process's own mask never leaks into QC's counts.
        engine._DAF_SNP_MASK = {"chr1": {1000 + masked_event}}
        ambient, _ = compute_state_rates([read], "daf", "dddb", header=header)
        assert ambient["msp"] == plain["msp"]
    finally:
        engine._DAF_SNP_MASK = previous
    assert block["msp"]["n_opportunities"] == plain["msp"]["n_opportunities"] - 2
    assert block["msp"]["n_events"] == plain["msp"]["n_events"] - 1
    assert block["definition"]["daf_run_mask"] == "off"


def test_tagged_daf_reads_use_the_calls_declared_run_mask():
    from fiberhmm.qc.states import declared_run_mask

    def header(ds):
        return pysam.AlignmentHeader.from_dict({
            "HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 100}],
            "PG": [{"ID": "fiberhmm-call", "PN": "fiberhmm-call", "DS": ds}]})

    assert declared_run_mask(header("mode=daf daf_run_mask=off")) == (0, "keep-one")
    assert declared_run_mask(header("mode=daf daf_run_mask=>=2/keep-one")) == (2, "keep-one")
    assert declared_run_mask(header("mode=daf")) is None


def test_light_call_subset_is_a_seeded_shuffle_of_the_sample():
    header = _header()
    reads = _uncalled_reads(header, 8, seed=4)
    a, _ = compute_state_rates(reads, "pacbio-fiber", "hia5", header=header,
                               light_call_reads=3)
    b, _ = compute_state_rates(list(reversed(reads)), "pacbio-fiber", "hia5",
                               header=header, light_call_reads=3)
    assert a["msp"] == b["msp"] and a["outside_msp"] == b["outside_msp"]


def test_grading_requires_a_reference_for_the_samples_state_source():
    block = _block(0.45, 0.05)
    block["source"] = "light_call"
    efficiency, background = grade_state_rates(block, _state_reference(), 125)
    assert efficiency["status"] == background["status"] == "INSUFFICIENT"
    block["source"] = "fibertools_tags"
    assert grade_state_rates(block, _state_reference(), 125)[0]["score"] is None


def test_scalar_legacy_tags_are_not_footprint_calls():
    """Regression: a non-FiberHMM pipeline's integer ``ns`` tag (Spacetime
    DddB BAMs) was read as 'called, no MSPs', so every base was outside-MSP."""
    header = _header()
    read = _pacbio_read(header, _sequence(), set(), reverse=False, name="x")
    read.set_tag("ns", 3)
    frames = {"MA": "seq", "legacy": "molecular"}
    assert msp_intervals_from_tags(read, frames) is None
    read.set_tag("ns", array.array("I", [10, 400]))
    read.set_tag("nl", array.array("I", [150, 147]))
    assert msp_intervals_from_tags(read, frames) == []
    read.set_tag("as", array.array("I", [160]))
    read.set_tag("al", array.array("I", [240]))
    assert msp_intervals_from_tags(read, frames) == [(160, 240, 240)]


def _uncalled_reads(header, n, seed=0):
    """Molecules with 400 bp hyper-labelled patches between 600 bp of near
    silence: an unambiguous accessible/protected pattern."""
    reads = []
    rng = np.random.default_rng(seed)
    for index in range(n):
        sequence = "".join(rng.choice(list("ACGT"), size=4000))
        mods = set()
        for position, base in enumerate(sequence):
            if base not in "AT":
                continue
            accessible = (position % 1000) < 400
            if rng.random() < (0.6 if accessible else 0.01):
                mods.add(position)
        reads.append(_pacbio_read(header, sequence, mods, reverse=bool(index % 2),
                                  name=f"u{index}", start=index * 5000))
    return reads


def test_uncalled_reads_take_a_bounded_light_call():
    header = _header()
    reads = _uncalled_reads(header, 12)
    block, _ = compute_state_rates(reads, "pacbio-fiber", "hia5", header=header,
                                   light_call_reads=5)
    assert block["source"] == "light_call" and block["available"]
    light = block["light_call"]
    assert light["reads_called"] == 5 and light["stopped_by"] == "read budget"
    assert light["model"] == "hia5_pacbio.json" and len(light["model_sha256"]) == 64
    assert block["reads"]["reads_considered"] == 5
    assert block["msp"]["aggregate_rate"] > 0.45
    assert block["outside_msp"]["aggregate_rate"] < 0.1
    assert block["msp_length_fraction"] == pytest.approx(0.4, abs=0.08)


def test_light_call_time_budget_stops_early():
    header = _header()
    reads = _uncalled_reads(header, 6, seed=1)
    ticks = iter(range(0, 10_000, 100))
    block, _ = compute_state_rates(reads, "pacbio-fiber", "hia5", header=header,
                                   light_call_seconds=150, clock=lambda: next(ticks))
    assert block["light_call"]["stopped_by"] == "time budget"
    assert block["light_call"]["reads_called"] < 6


def test_light_call_reproduces_the_calling_encoding():
    """The light call's MSPs come from the same encode+Viterbi as fiberhmm-call."""
    from fiberhmm.inference.engine import (
        _extract_fiber_read_from_pysam,
        _process_single_read,
    )

    header = _header()
    read = _uncalled_reads(header, 1, seed=3)[0]
    caller = states.LightCaller("hia5", "pacbio-fiber")
    fiber_read = _extract_fiber_read_from_pysam(read, "pacbio-fiber", 125)
    intervals, encoded = caller.call(fiber_read, "pacbio-fiber", 10)
    reference = _process_single_read(fiber_read, caller.model, 10, False,
                                     "pacbio-fiber", 3, 0, False, nuc_min_size=85,
                                     include_encoded=True)
    assert intervals == [(s, n, n) for s, n in zip(reference["as"].tolist(),
                                                     reference["al"].tolist())]
    np.testing.assert_array_equal(encoded, states._encode_like_call(
        fiber_read, "pacbio-fiber", 3, 10))


def _state_reference(**definition):
    base = {"min_msp_bp": 85, "edge_trim_bp": 10, "terminal_segments": "cap",
            "min_state_opportunities_per_read": 50, "probability_threshold": 125}
    base.update(definition)
    q = {"probabilities": [0.05, 0.25, 0.5, 0.75, 0.95]}
    return {"state_rates": {"scoring_enabled": True, "definition": base,
                            "by_source": {"tags": {
                                "msp": {**q, "quantiles": [0.2, 0.3, 0.4, 0.5, 0.6]},
                                "outside_msp": {**q, "quantiles": [0.01, 0.03, 0.05, 0.07, 0.1]},
                            }}}}


def _block(msp, outside, n=100):
    return {"available": True, "source": "tags",
            "definition": {"min_msp_bp": 85, "edge_trim_bp": 10,
                           "terminal_segments": "cap",
                           "min_state_opportunities_per_read": 50},
            "msp": {"median_per_read_rate": msp, "n_rate_reads": n},
            "outside_msp": {"median_per_read_rate": outside, "n_rate_reads": n}}


# Bands relative to the reference median (0.4 here): PASS while less than 20%
# below, WARN 20-30% below, FAIL further below.
@pytest.mark.parametrize("msp, status", [(0.45, "PASS"), (0.33, "PASS"),
                                         (0.30, "WARN"), (0.25, "FAIL")])
def test_efficiency_is_graded_one_sided_low(msp, status):
    efficiency, _ = grade_state_rates(_block(msp, 0.05), _state_reference(), 125)
    assert efficiency["status"] == status
    assert efficiency["pass_min"] == pytest.approx(0.32) and efficiency["warn_min"] == pytest.approx(0.28)


# Background mirrors it above the reference median (0.05 here).
@pytest.mark.parametrize("outside, status", [(0.0, "PASS"), (0.059, "PASS"),
                                             (0.062, "WARN"), (0.08, "FAIL")])
def test_background_is_graded_one_sided_high(outside, status):
    _, background = grade_state_rates(_block(0.4, outside), _state_reference(), 125)
    assert background["status"] == status
    assert background["pass_max"] == pytest.approx(0.06) and background["warn_max"] == pytest.approx(0.065)


def test_a_reference_can_still_grade_by_its_quantiles():
    reference = _state_reference()
    reference["state_rates"]["grading"] = {
        "efficiency": {"pass_quantile": 0.25, "warn_quantile": 0.05},
        "background": {"pass_quantile": 0.75, "warn_quantile": 0.95}}
    efficiency, background = grade_state_rates(_block(0.25, 0.09), reference, 125)
    assert efficiency["pass_min"] == 0.3 and efficiency["warn_min"] == 0.2 and efficiency["status"] == "WARN"
    assert background["pass_max"] == 0.07 and background["warn_max"] == 0.1 and background["status"] == "WARN"


def test_the_shipped_references_grade_relative_to_their_median():
    import json
    from importlib import resources
    shipped = json.loads(resources.files("fiberhmm.qc").joinpath("references.json").read_text())
    graded = [p["state_rates"]["grading"] for p in shipped["profiles"].values()
              if isinstance(p.get("state_rates"), dict) and "grading" in p["state_rates"]]
    assert graded and all(g["efficiency"] == {"pass_relative": 0.2, "warn_relative": 0.3} for g in graded)


def test_grading_requires_the_reference_definition_and_threshold():
    efficiency, background = grade_state_rates(
        _block(0.4, 0.05), _state_reference(min_msp_bp=147), 125)
    assert efficiency["status"] == background["status"] == "INSUFFICIENT"
    assert "min_msp_bp" in efficiency["note"]
    efficiency, _ = grade_state_rates(_block(0.5, 0.05), _state_reference(), 200)
    assert efficiency["score"] == 69.0 and "ML threshold" in efficiency["note"]
    efficiency, _ = grade_state_rates(_block(0.5, 0.05, n=5), _state_reference(), 125)
    assert efficiency["status"] == "INSUFFICIENT"


def test_overall_verdict_uses_states_when_graded():
    analysis = {
        "signal": {"score": 20.0, "status": "FAIL"},
        "periodicity": {"score": 90.0, "status": "PASS"},
        "efficiency": {"score": 95.0, "status": "PASS"},
        "background": {"score": 80.0, "status": "PASS"},
    }
    overall = _overall_verdict(analysis)
    assert overall["status"] == "PASS" and overall["verdict_basis"] == "state-aware"
    assert overall["components"] == ["efficiency", "background", "periodicity"]
    # A graded failure is never hidden by the other component lacking evidence.
    analysis["efficiency"] = {"score": 10.0, "status": "FAIL"}
    analysis["background"] = {"score": None, "status": "INSUFFICIENT"}
    overall = _overall_verdict(analysis)
    assert overall["status"] == "FAIL" and overall["verdict_basis"] == "state-aware"
    assert overall["components"] == ["efficiency", "periodicity"]
    analysis["efficiency"] = {"score": None, "status": "INSUFFICIENT"}
    overall = _overall_verdict(analysis)
    assert overall["status"] == "FAIL" and overall["verdict_basis"] == "overall-rate"


def test_packaged_state_references_are_aggregate_and_definition_matched():
    profiles = load_references()["profiles"]
    for name in ("dddb", "ddda", "hia5_pacbio"):
        reference = profiles[name]["state_rates"]
        assert reference["scoring_enabled"]
        assert reference["definition"]["min_msp_bp"] == states.DEFAULT_MIN_MSP_BP
        assert reference["definition"]["terminal_segments"] == states.DEFAULT_TERMINAL_POLICY
        assert reference["definition"]["min_state_opportunities_per_read"] == \
            states.DEFAULT_MIN_STATE_OPPORTUNITIES
        for source in ("tags", "light_call"):
            entry = reference["by_source"][source]
            for key in ("msp", "outside_msp"):
                q = entry[key]["quantiles"]
                assert len(q) == 5 and q == sorted(q)
            assert entry["msp"]["quantiles"][2] > entry["outside_msp"]["quantiles"][2]
    assert profiles["hia5_nanopore"]["state_rates"]["scoring_enabled"] is False


# --- report schema: additive -------------------------------------------------

# Keys of a schema-1 report before state-aware rates (FiberHMM 3.0 pre-release).
LEGACY_TOP_LEVEL = {
    "schema_version", "fiberhmm_version", "input", "assay", "sampling", "signal",
    "periodicity", "footprints", "deduplication", "overall", "outputs",
    "variant_masking",
}
LEGACY_SIGNAL = {
    "label", "n_rate_reads", "n_signal_reads", "n_events", "n_opportunities",
    "aggregate_rate", "mean_per_read_rate", "p05_per_read_rate",
    "q25_per_read_rate", "median_per_read_rate", "q75_per_read_rate",
    "p95_per_read_rate", "score", "status", "note",
}


def _write_bam(path: Path, reads, header) -> str:
    unsorted = str(path) + ".unsorted.bam"
    with pysam.AlignmentFile(unsorted, "wb", header=header) as bam:
        for read in reads:
            bam.write(read)
    pysam.sort("-o", str(path), unsorted)
    pysam.index(str(path))
    return str(path)


def test_report_schema_is_additive_and_carries_state_fields(tmp_path):
    header = _header()
    molecule = _sequence()
    reads = []
    for index in range(30):
        mods = _planted_mods(molecule)
        reads.append(_pacbio_read(header, molecule, mods, reverse=False,
                                  name=f"r{index}", start=index * 4000,
                                  msps=PLANTED_MSPS))
    path = _write_bam(tmp_path / "calls.bam", reads, header)
    result = run_qc(path, output_prefix=str(tmp_path / "qc" / "calls"), stream=None)
    assert LEGACY_TOP_LEVEL <= set(result)
    assert LEGACY_SIGNAL <= set(result["signal"])
    assert {"score", "status"} <= set(result["overall"])
    assert result["schema_version"] == 1 and result["schema_minor_version"] == 1
    assert {"state_rates", "efficiency", "background"} <= set(result)
    block = result["state_rates"]
    assert block["source"] == "tags"
    assert block["msp"]["aggregate_rate"] == pytest.approx(0.5, abs=0.01)
    assert block["reference"]["msp"]["quantiles"]
    written = json.loads(Path(result["outputs"]["json"]).read_text())
    assert written["state_rates"]["msp"]["n_opportunities"] == block["msp"]["n_opportunities"]
    curves = json.loads(Path(result["outputs"]["curves"]).read_text())
    assert curves["schema"] == "fiberhmm.qc.curves.v1"
    assert curves["verdicts"]["efficiency"] == result["efficiency"]["status"]
    assert len(curves["state_rates"]["msp_per_read_rates"]) == block["msp"]["n_rate_reads"]
    tsv = Path(result["outputs"]["tsv"]).read_text().splitlines()
    assert "aggregate_msp_rate" in tsv[0].split("\t")
    # state_source none restores the overall-rate verdict.
    off = run_qc(path, output_prefix=str(tmp_path / "qc" / "off"), stream=None,
                 state_source="none")
    assert off["state_rates"]["available"] is False
    assert off["overall"]["verdict_basis"] == "overall-rate"


def test_pipeline_qc_verdicts_carry_state_fields(tmp_path):
    from fiberhmm.pipeline.runner import qc_verdicts

    report = {
        "overall": {"status": "PASS", "score": 90.0, "verdict_basis": "state-aware"},
        "signal": {"status": "WARN", "score": 60.0},
        "periodicity": {"status": "PASS", "score": 80.0},
        "efficiency": {"status": "PASS", "score": 95.0},
        "background": {"status": "PASS", "score": 99.0},
        "state_rates": {"available": True, "source": "tags",
                        "definition": {"min_msp_bp": 85},
                        "all_states": {"aggregate_rate": 0.1},
                        "msp": {"median_per_read_rate": 0.4, "aggregate_rate": 0.41},
                        "outside_msp": {"median_per_read_rate": 0.05,
                                        "aggregate_rate": 0.04},
                        "msp_to_outside_ratio": 10.25, "msp_length_fraction": 0.2},
    }
    path = tmp_path / "s.qc.json"
    path.write_text(json.dumps(report))
    verdicts = qc_verdicts(str(path))
    assert verdicts["efficiency"] == verdicts["background"] == "PASS"
    assert verdicts["efficiency_score"] == 95.0
    assert verdicts["verdict_basis"] == "state-aware"
    assert verdicts["state_rates"]["msp_rate"] == 0.4
    assert verdicts["state_rates"]["outside_msp_rate"] == 0.05
    assert verdicts["state_rates"]["msp_to_outside_ratio"] == 10.25
    # Reports from before state-aware rates keep their verdict keys only.
    legacy = {key: report[key] for key in ("overall", "signal", "periodicity")}
    legacy["overall"] = {"status": "PASS", "score": 90.0}
    path.write_text(json.dumps(legacy))
    assert set(qc_verdicts(str(path))) == {
        "overall", "overall_score", "signal", "signal_score",
        "periodicity", "periodicity_score"}


def test_snp_masked_opposite_strand_events_do_not_make_a_read_chimeric():
    """The QC mask reaches extraction before the strand-swap chimera filter,
    as calling's mask does."""
    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 100_000}],
        "PG": [{"ID": "fiberhmm-call", "PN": "fiberhmm-call",
                "DS": "mode=daf enzyme=dddb coord=molecular daf_run_mask=off"}]})
    molecule = list(_sequence(seed=10))
    cs = [p for p, b in enumerate(molecule) if b == "C"]
    gs = [p for p, b in enumerate(molecule) if b == "G" and p > 2000]
    for p in cs[: len(cs) // 2:3]:
        molecule[p] = "Y"
    for p in gs[:8]:
        molecule[p] = "R"
    read = pysam.AlignedSegment(header)
    read.query_name, read.flag, read.reference_id = "chim", 0, 0
    read.query_sequence = "".join(molecule)
    read.reference_start, read.mapping_quality = 0, 60
    read.cigartuples = [(0, L)]
    read.set_tag("st", "CT")
    read.set_tag("MA", f"{L};msp.:301-400")
    unmasked, _ = compute_state_rates([read], "daf", "dddb", header=header)
    masked, _ = compute_state_rates([read], "daf", "dddb", header=header,
                                    snp_mask={"chr1": set(gs[:8])})
    assert unmasked["reads"]["reads_chimera_skipped"] == 1
    assert masked["reads"]["reads_chimera_skipped"] == 0
    assert masked["reads"]["reads_used"] == 1
