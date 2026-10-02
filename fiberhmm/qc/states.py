"""State-aware modification rates for ``fiberhmm-qc``.

The overall modification rate mixes two things: how efficiently the enzyme
labels accessible DNA, and how much of the DNA is accessible. A targeted
amplicon at an open locus and a genome-wide sample can therefore differ
several-fold in overall rate with the same enzyme efficiency. This module
splits every sampled molecule into two compartments using FiberHMM states:

* **in-MSP**: inside a methylase-sensitive patch (accessible stretch between
  nucleosome-sized footprints) at least ``min_msp_bp`` long. Its rate is the
  enzyme efficiency.
* **outside-MSP**: everything else (nucleosomes, linkers and accessible gaps
  shorter than ``min_msp_bp``). Its rate is the background.

States come from the BAM's own FiberHMM calls (``MA``, fibertools ``Ma`` or
legacy ``as/al`` tags) when present. Otherwise a bounded *light call* runs the
bundled apply HMM of the declared chemistry on the sampled reads only: the
same extraction, encoding and Viterbi code as ``fiberhmm-call``, with no
nucleosome or TF recall.

Opportunities and events are read off the HMM observation encoding itself
(``encode_from_query_sequence``), so they are exactly what calling sees: DAF
targets on the deaminated strand only (with the chemistry's adjacent-run mask
and SNP mask), Hia5 PacBio A/T, Hia5 Nanopore basecalled A, MM ``?``-unlisted
bases excluded, and ``edge_trim`` bases at each read end excluded.
"""
from __future__ import annotations

import math
import time
from contextlib import ExitStack
from typing import Optional, Sequence

import numpy as np

from fiberhmm.core.bam_reader import ContextEncoder

#: QC-side minimum MSP length. FiberHMM's MSP tags include every accessible
#: gap between nucleosome-sized footprints, so linkers are MSPs in the tags.
#: The QC counts short gaps (linkers) as outside-MSP background.
DEFAULT_MIN_MSP_BP = 85
#: Bases masked at each read end, as ``fiberhmm-call --edge-trim``.
DEFAULT_EDGE_TRIM = 10
#: Bounds of the light call on uncalled BAMs.
DEFAULT_LIGHT_CALL_READS = 400
DEFAULT_LIGHT_CALL_SECONDS = 60.0
#: Minimum opportunities in a compartment for a per-read rate.
DEFAULT_MIN_STATE_OPPORTUNITIES = 50
#: Terminal state segments (truncated by a read end) are excluded.
TERMINAL_POLICIES = ("cap", "drop", "keep")
DEFAULT_TERMINAL_POLICY = "cap"

_PLATFORM_FOR_MODE = {"pacbio-fiber": "pacbio", "nanopore-fiber": "nanopore"}


# ---------------------------------------------------------------------------
# MSP intervals
# ---------------------------------------------------------------------------

def tag_frames(header) -> dict:
    """Coordinate frames of the annotation tag families in ``header``."""
    from fiberhmm.io.annotation_frame import legacy_tag_frame, ma_annotation_frame

    try:
        legacy_frame, legacy_reason = legacy_tag_frame(header)
    except Exception as exc:  # malformed @PG chains: treat as unknown
        legacy_frame, legacy_reason = None, f"unreadable @PG history ({exc})"
    try:
        ma_frame = ma_annotation_frame(header)
    except Exception:
        ma_frame = "seq"
    return {"MA": ma_frame, "legacy": legacy_frame, "legacy_reason": legacy_reason}


def _to_seq_frame(intervals, read, frame: Optional[str]):
    """``(start, length, feature_length)`` triples in SEQ frame.

    A wrapped interval (circular molecule) may start before 0 after the frame
    flip; its start is taken modulo the read length so it keeps covering
    ``[start, L) + [0, end - L)``.
    """
    from fiberhmm.io.ma_tags import flip_intervals_to_seq

    starts = [int(item[0]) for item in intervals]
    lengths = [int(item[1]) for item in intervals]
    features = [int(item[2]) if len(item) > 2 else int(item[1]) for item in intervals]
    if frame == "molecular":
        starts, lengths = flip_intervals_to_seq(starts, lengths, read)
    read_length = len(getattr(read, "query_sequence", None) or "")
    if read_length:
        starts = [start % read_length for start in starts]
    return list(zip(starts, lengths, features))


def _ma_msps(parsed: dict, an_names) -> list:
    """MA ``msp`` pieces with the length of the whole feature each belongs to.

    A circular MSP across the origin is written as two pieces sharing an
    ``AN`` name; its length cut-off applies to the joined feature.
    """
    pieces, index = [], 0
    for name, _strand, _qual, intervals in parsed["raw_types"]:
        for start, length in intervals:
            if name == "msp":
                feature = an_names[index] if index < len(an_names) else ""
                pieces.append((start, length, feature))
            index += 1
    totals: dict = {}
    for _start, length, feature in pieces:
        if feature:
            totals[feature] = totals.get(feature, 0) + int(length)
    return [(start, length, totals.get(feature, length) if feature else length)
            for start, length, feature in pieces]


def msp_intervals_from_tags(read, frames: dict) -> Optional[list]:
    """SEQ-frame ``(start, length, feature_length)`` MSPs from a read's calls.

    ``None`` when the read carries no usable footprint annotation (or its
    frame cannot be decided); ``[]`` for a called read without MSPs.
    """
    from fiberhmm.io.ma_tags import parse_an_tag, parse_ma_tag

    for tag, an_tag, frame in (("MA", "AN", frames.get("MA", "seq")),
                               ("Ma", "An", "molecular")):
        if not read.has_tag(tag):
            continue
        try:
            parsed = parse_ma_tag(str(read.get_tag(tag)))
            names = parse_an_tag(str(read.get_tag(an_tag))) if read.has_tag(an_tag) else []
        except (ValueError, TypeError):
            return None
        length = len(read.query_sequence or "")
        if parsed["read_length"] and length and parsed["read_length"] != length:
            return None
        if not (parsed["msp"] or parsed["nuc"]):
            return None  # e.g. an MA carrying only m5C/TF features
        return _to_seq_frame(_ma_msps(parsed, names), read, frame)
    # Legacy footprint tags come in start/length pairs. Other pipelines use
    # the same two-letter names for unrelated scalars (e.g. an integer ``ns``),
    # so only array-valued pairs count as calls.
    has_msps = _array_tag_pair(read, "as", "al")
    if has_msps or _array_tag_pair(read, "ns", "nl"):
        frame = frames.get("legacy")
        if frame is None:
            return None
        if not has_msps:
            return []
        try:
            starts = list(read.get_tag("as"))
            lengths = list(read.get_tag("al"))
        except (KeyError, TypeError):
            return None
        return _to_seq_frame(list(zip(starts, lengths)), read, frame)
    return None


def _array_tag_pair(read, start_tag: str, length_tag: str) -> bool:
    if not (read.has_tag(start_tag) and read.has_tag(length_tag)):
        return False
    starts, lengths = read.get_tag(start_tag), read.get_tag(length_tag)
    return (not isinstance(starts, (int, float, str))
            and not isinstance(lengths, (int, float, str))
            and len(starts) == len(lengths))


def msp_mask(intervals, read_length: int, min_msp_bp: int) -> np.ndarray:
    """Boolean per-base MSP mask; MSPs shorter than ``min_msp_bp`` are dropped.

    Intervals are ``(start, length[, feature_length])``; the cut-off applies
    to ``feature_length`` (a wrapped feature's joined length). Intervals
    running past the read end (a circular wrap) continue at 0.
    """
    mask = np.zeros(read_length, dtype=bool)
    for interval in intervals:
        start, length = int(interval[0]), int(interval[1])
        feature = int(interval[2]) if len(interval) > 2 else length
        if feature < max(1, int(min_msp_bp)) or read_length <= 0 or length <= 0:
            continue
        start %= read_length
        end = start + length
        mask[start:min(end, read_length)] = True
        if end > read_length:
            mask[: min(end - read_length, read_length)] = True
    return mask


def terminal_exclusion(mask: np.ndarray, policy: str = DEFAULT_TERMINAL_POLICY,
                       margin: int = DEFAULT_MIN_MSP_BP) -> np.ndarray:
    """Positions of a molecule's terminal (read-end-truncated) state segments.

    A terminal segment's state is called from a partial feature: a nucleosome
    cut to < 85 bp by the read end does not bound an MSP, so its DNA (and
    any linker beyond it) joins the terminal MSP; an MSP cut below
    ``min_msp_bp`` counts as outside. Excluding ``margin`` (= ``min_msp_bp``)
    bases is a heuristic that removes most of this, not a bound on it.

    ``cap`` (default) excludes each terminal segment up to ``margin`` bases
    from its read end; ``drop`` excludes the whole first and last segment
    (a single-segment molecule entirely); ``keep`` excludes nothing (only the
    encoder's ``edge_trim``).
    """
    n = len(mask)
    excluded = np.zeros(n, dtype=bool)
    if n == 0 or policy == "keep":
        return excluded
    changes = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    first = int(changes[0]) if len(changes) else n
    last = int(changes[-1]) if len(changes) else 0
    if policy == "drop":
        if not len(changes):
            excluded[:] = True
            return excluded
        excluded[:first] = True
        excluded[last:] = True
        return excluded
    margin = max(0, int(margin))
    excluded[: min(first, margin)] = True
    excluded[max(last, n - margin):] = True
    return excluded


_RUN_MASK_RE = None


def declared_run_mask(header) -> Optional[tuple]:
    """``(min_run, policy)`` the BAM's last FiberHMM writer encoded DAF with.

    Read from ``daf_run_mask=off`` / ``daf_run_mask=>=2/keep-one`` in the
    @PG description; ``None`` when no writer recorded it.
    """
    import re

    global _RUN_MASK_RE
    if _RUN_MASK_RE is None:
        _RUN_MASK_RE = re.compile(r"daf_run_mask=(off|>=(\d+)/([a-z-]+))")
    try:
        programs = (header.to_dict() if hasattr(header, "to_dict") else dict(header)).get("PG", [])
    except (TypeError, ValueError):
        return None
    found = None
    for program in programs:
        match = _RUN_MASK_RE.search(str(program.get("DS", "")))
        if match:
            found = (0, "keep-one") if match.group(1) == "off" else (
                int(match.group(2)), match.group(3))
    return found


def _last_fiberhmm_writer(header) -> Optional[dict]:
    """``{program, version}`` of the last FiberHMM footprint writer in @PG."""
    from fiberhmm.io.annotation_frame import FIBERHMM_FOOTPRINT_PROGRAMS

    try:
        programs = (header.to_dict() if hasattr(header, "to_dict") else dict(header)).get("PG", [])
    except (TypeError, ValueError, AttributeError):
        return None
    found = None
    for program in programs:
        if program.get("PN") in FIBERHMM_FOOTPRINT_PROGRAMS:
            found = {"program": program.get("PN"), "version": program.get("VN")}
    return found


def _sample_order(reads, seed: int) -> list:
    """Indices of ``reads`` in a seeded hash order (a random, reproducible
    order, whatever order the sampler returned them in)."""
    import hashlib

    def rank(read):
        key = f"{seed}|{read.query_name}|{read.flag}|{read.reference_id}|{read.reference_start}"
        return hashlib.blake2b(key.encode("utf-8", "replace"), digest_size=8).digest()

    return sorted(range(len(reads)), key=lambda index: (rank(reads[index]), index))


def snp_query_positions(read, snp_mask: Optional[dict]) -> set:
    """Query positions aligned to masked reference sites (DAF SNP mask)."""
    if not snp_mask or read.is_unmapped:
        return set()
    sites = snp_mask.get(read.reference_name)
    if not sites:
        return set()
    from fiberhmm.daf.snps import wrapped_reference_sites

    sites = wrapped_reference_sites(read, sites)
    try:
        return {int(query) for query, reference in read.get_aligned_pairs(matches_only=True)
                if reference in sites}
    except (ValueError, TypeError, IndexError):
        return set()


# ---------------------------------------------------------------------------
# Observations: the calling encoding
# ---------------------------------------------------------------------------

def _encode_like_call(fiber_read: dict, mode: str, context_size: int,
                      edge_trim: int) -> np.ndarray:
    """The observation array ``fiberhmm-call`` builds for this read.

    Mirrors ``engine._process_single_read`` for linear molecules.
    """
    from fiberhmm.core.bam_reader import detect_daf_strand, encode_from_query_sequence

    sequence = fiber_read["query_sequence"]
    mods = fiber_read["m6a_query_positions"]
    if mode == "daf":
        strand = fiber_read.get("_daf_strand") or detect_daf_strand(sequence, mods)
    else:
        strand = "."
    return encode_from_query_sequence(
        sequence, mods, edge_trim,
        mode=mode, strand=strand, context_size=context_size,
        is_reverse=fiber_read.get("is_reverse", False),
        unknown_positions=fiber_read.get("unknown_query_positions"),
    )


def observation_masks(encoded: np.ndarray, context_size: int):
    """``(opportunity, event)`` boolean arrays from an encoded read."""
    n_codes = ContextEncoder.get_n_codes(context_size)
    encoded = np.asarray(encoded)
    event = encoded < n_codes
    unmodified = (encoded > n_codes) & (encoded <= 2 * n_codes)
    return event | unmodified, event


# ---------------------------------------------------------------------------
# Light call
# ---------------------------------------------------------------------------

class LightCaller:
    """The bundled apply HMM of one chemistry, loaded once."""

    def __init__(self, enzyme: Optional[str], mode: str):
        from fiberhmm.core.model_io import (
            freeze_model_for_inference,
            load_model_with_metadata,
        )
        from fiberhmm.models import get_model_path

        if not enzyme:
            raise ValueError("no enzyme is declared, so no bundled model applies")
        platform = _PLATFORM_FOR_MODE.get(mode)
        path = get_model_path(enzyme, tool="apply", seq=platform)
        model, k, model_mode = load_model_with_metadata(path)
        self.model = freeze_model_for_inference(model)
        self.context_size = int(k or 3)
        self.model_path = path
        self.model_mode = model_mode
        import hashlib
        from pathlib import Path

        self.model_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def call(self, fiber_read: dict, mode: str, edge_trim: int):
        """``(msp_intervals, encoded)`` in SEQ frame, or ``None``."""
        from fiberhmm.inference.engine import _process_single_read

        result = _process_single_read(
            fiber_read, self.model, edge_trim, False, mode, self.context_size,
            0, False, nuc_min_size=85, include_encoded=True,
        )
        if result is None:
            return None
        intervals = [(start, length, length) for start, length in zip(
            np.asarray(result["as"]).tolist(), np.asarray(result["al"]).tolist())]
        return intervals, result["encoded"]


def _context_size_for(enzyme: Optional[str], mode: str) -> int:
    try:
        from fiberhmm.core.model_io import load_model_with_metadata
        from fiberhmm.models import get_model_path

        _model, k, _mode = load_model_with_metadata(
            get_model_path(enzyme, tool="apply", seq=_PLATFORM_FOR_MODE.get(mode)))
        return int(k or 3)
    except Exception:
        return 3


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _rate(events: int, opportunities: int) -> Optional[float]:
    return float(events / opportunities) if opportunities else None


def _wilson(events: int, opportunities: int, z: float = 1.96):
    """Binomial 95% Wilson interval (molecules pooled; ignores overdispersion)."""
    if not opportunities:
        return None
    p = events / opportunities
    denominator = 1 + z * z / opportunities
    centre = (p + z * z / (2 * opportunities)) / denominator
    half = z * math.sqrt(p * (1 - p) / opportunities
                         + z * z / (4 * opportunities ** 2)) / denominator
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def _quantiles(values: Sequence[float]):
    if not len(values):
        return [None] * 5
    return np.quantile(np.asarray(values, dtype=float),
                       (0.05, 0.25, 0.5, 0.75, 0.95)).tolist()


def _compartment(events, opportunities, per_read, n_reads_floor):
    q = _quantiles(per_read)
    return {
        "n_events": int(events),
        "n_opportunities": int(opportunities),
        "aggregate_rate": _rate(events, opportunities),
        "aggregate_rate_wilson95": _wilson(events, opportunities),
        "n_rate_reads": int(len(per_read)),
        "p05_per_read_rate": q[0],
        "q25_per_read_rate": q[1],
        "median_per_read_rate": q[2],
        "q75_per_read_rate": q[3],
        "p95_per_read_rate": q[4],
        "mean_per_read_rate": float(np.mean(per_read)) if len(per_read) else None,
    }


def compute_state_rates(
    reads: Sequence,
    mode: str,
    enzyme: Optional[str],
    header=None,
    prob_threshold: int = 125,
    reference_fasta: Optional[str] = None,
    snp_mask: Optional[dict] = None,
    state_source: str = "auto",
    min_msp_bp: int = DEFAULT_MIN_MSP_BP,
    edge_trim: int = DEFAULT_EDGE_TRIM,
    light_call_reads: int = DEFAULT_LIGHT_CALL_READS,
    light_call_seconds: float = DEFAULT_LIGHT_CALL_SECONDS,
    min_state_opportunities: int = DEFAULT_MIN_STATE_OPPORTUNITIES,
    terminal_policy: str = DEFAULT_TERMINAL_POLICY,
    clock=time.monotonic,
    seed: int = 20260824,
) -> tuple[dict, dict]:
    """In-MSP / outside-MSP modification rates of the sampled molecules.

    ``state_source``: ``auto`` (calls when the sample carries them, else a
    light call), ``tags`` (calls only) or ``light-call`` (always re-call).
    Returns ``(report block, per-read arrays)``.
    """
    if terminal_policy not in TERMINAL_POLICIES:
        raise ValueError(f"terminal_policy must be one of {TERMINAL_POLICIES}")
    if state_source not in ("auto", "tags", "light-call"):
        raise ValueError("state_source must be auto, tags or light-call")
    block: dict = {
        "available": False,
        "source": None,
        "definition": {
            "msp": f"FiberHMM MSP of >= {int(min_msp_bp)} bp",
            "outside_msp": "every other base: nucleosomes, linkers and "
                           f"accessible gaps < {int(min_msp_bp)} bp",
            "min_msp_bp": int(min_msp_bp),
            "edge_trim_bp": int(edge_trim),
            "terminal_segments": terminal_policy,
            "opportunities": _opportunity_text(mode),
            "probability_threshold": int(prob_threshold),
            "min_state_opportunities_per_read": int(min_state_opportunities),
        },
    }
    if mode not in ("daf", "pacbio-fiber", "nanopore-fiber"):
        block["note"] = f"state-aware rates are not defined for mode {mode!r}"
        return block, {}

    import pysam

    from fiberhmm.core.bam_reader import daf_run_mask_scope, default_daf_run_mask
    from fiberhmm.inference import engine
    from fiberhmm.inference.read_filters import hard_clipped_mm_unreliable

    frames = tag_frames(header) if header is not None else {
        "MA": "seq", "legacy": None, "legacy_reason": "no header"}
    eligible = [read for read in reads if not getattr(read, "is_duplicate", False)]
    tagged = [msp_intervals_from_tags(read, frames) for read in eligible]
    n_tagged = sum(1 for item in tagged if item is not None)
    # fibertools' own calls (Ma without FiberHMM MA) are a different state
    # caller from the FiberHMM calls the references were built from.
    n_fibertools = sum(1 for read, item in zip(eligible, tagged)
                       if item is not None and not read.has_tag("MA") and read.has_tag("Ma"))

    use_tags = state_source == "tags" or (
        state_source == "auto" and eligible and n_tagged >= 0.5 * len(eligible))
    caller = None
    started = clock()
    run_mask = default_daf_run_mask(enzyme) if mode == "daf" else (0, "keep-one")
    if use_tags:
        if not n_tagged:
            block["note"] = "no sampled read carries FiberHMM MSP/nucleosome calls"
            return block, {}
        source = "fibertools_tags" if n_fibertools > n_tagged / 2 else "tags"
        context_size = _context_size_for(enzyme, mode)
        if mode == "daf":
            # Re-encode tagged reads as their call did.
            run_mask = declared_run_mask(header) or run_mask
    else:
        try:
            caller = LightCaller(enzyme, mode)
        except Exception as exc:
            block["note"] = f"light call unavailable: {exc}"
            return block, {}
        source = "light_call"
        context_size = caller.context_size
        order = _sample_order(eligible, seed)
        eligible = [eligible[index] for index in order]
        tagged = [tagged[index] for index in order]
    block["definition"]["daf_run_mask"] = (
        (f">={run_mask[0]}/{run_mask[1]}" if run_mask[0] else "off")
        if mode == "daf" else None)

    totals = dict(msp_events=0, msp_opps=0, out_events=0, out_opps=0,
                  msp_bp=0, total_bp=0, excluded_opps=0, all_events=0, all_opps=0)
    per_read_msp: list[float] = []
    per_read_out: list[float] = []
    per_read_ratio_inputs: list[tuple] = []
    msp_fraction_per_read: list[float] = []
    counts = dict(reads_considered=0, reads_without_signal=0, reads_without_calls=0,
                  reads_chimera_skipped=0, reads_mm_unreliable=0,
                  reads_single_segment=0, reads_used=0)
    light = {"reads_called": 0, "budget_reads": int(light_call_reads),
             "budget_seconds": float(light_call_seconds), "stopped_by": None,
             "elapsed_seconds": 0.0}
    snp_mask = snp_mask if mode == "daf" else None

    with ExitStack() as stack:
        stack.enter_context(daf_run_mask_scope(*run_mask))
        reference_handle = (
            stack.enter_context(pysam.FastaFile(reference_fasta))
            if reference_fasta else None
        )
        # The QC mask replaces calling's process-wide one in this context only
        # (QC may run inside fiberhmm-call or beside other jobs): extraction
        # drops masked events before the chimera filter, as calling does, and
        # the masked sites are also removed as opportunities below.
        stack.enter_context(engine.daf_snp_mask_scope(snp_mask or {}))

        for read, intervals in zip(eligible, tagged):
            if caller is not None:
                if light["reads_called"] >= light_call_reads:
                    light["stopped_by"] = "read budget"
                    break
                if clock() - started > light_call_seconds:
                    light["stopped_by"] = "time budget"
                    break
            elif intervals is None:
                counts["reads_without_calls"] += 1
                continue
            counts["reads_considered"] += 1
            if hard_clipped_mm_unreliable(read, mode, reference_handle is not None):
                # Calling skips these too: MM/ML no longer lines up with SEQ.
                counts["reads_mm_unreliable"] += 1
                continue
            fiber_read = engine._extract_fiber_read_from_pysam(
                read, mode, prob_threshold, reference_handle)
            if fiber_read is engine.CHIMERA_SKIP:
                counts["reads_chimera_skipped"] += 1
                continue
            if fiber_read is None:
                counts["reads_without_signal"] += 1
                continue
            masked = snp_query_positions(read, snp_mask)
            if masked:
                # Masked sites are neither events nor opportunities.
                fiber_read["m6a_query_positions"] = (
                    set(fiber_read["m6a_query_positions"]) - masked)
            if caller is not None:
                light["reads_called"] += 1
                called = caller.call(fiber_read, mode, edge_trim)
                if called is None:
                    counts["reads_without_calls"] += 1
                    continue
                intervals, encoded = called
            else:
                encoded = _encode_like_call(fiber_read, mode, context_size, edge_trim)
            read_length = len(fiber_read["query_sequence"])
            if not read_length or len(encoded) != read_length:
                counts["reads_without_calls"] += 1
                continue
            opportunity, event = observation_masks(encoded, context_size)
            if masked:
                index = np.fromiter(masked, dtype=np.int64)
                index = index[(index >= 0) & (index < read_length)]
                opportunity[index] = False
                event[index] = False
            in_msp = msp_mask(intervals, read_length, min_msp_bp)
            totals["msp_bp"] += int(in_msp.sum())
            totals["total_bp"] += read_length
            msp_fraction_per_read.append(float(in_msp.mean()))
            totals["all_events"] += int(event.sum())
            totals["all_opps"] += int(opportunity.sum())
            excluded = terminal_exclusion(in_msp, terminal_policy, min_msp_bp)
            if not np.any(in_msp[1:] != in_msp[:-1]):
                counts["reads_single_segment"] += 1
            totals["excluded_opps"] += int((opportunity & excluded).sum())
            keep = opportunity & ~excluded
            m_opps = int((keep & in_msp).sum())
            m_events = int((event & keep & in_msp).sum())
            o_opps = int((keep & ~in_msp).sum())
            o_events = int((event & keep & ~in_msp).sum())
            if m_opps + o_opps:
                counts["reads_used"] += 1
            totals["msp_opps"] += m_opps
            totals["msp_events"] += m_events
            totals["out_opps"] += o_opps
            totals["out_events"] += o_events
            if m_opps >= min_state_opportunities:
                per_read_msp.append(m_events / m_opps)
            if o_opps >= min_state_opportunities:
                per_read_out.append(o_events / o_opps)
            per_read_ratio_inputs.append((m_events, m_opps, o_events, o_opps))
    if caller is not None:
        light["elapsed_seconds"] = round(float(clock() - started), 3)

    msp_rate = _rate(totals["msp_events"], totals["msp_opps"])
    out_rate = _rate(totals["out_events"], totals["out_opps"])
    block.update({
        "available": bool(totals["msp_opps"] and totals["out_opps"]),
        "source": source,
        "reads": {key: int(value) for key, value in counts.items()},
        "all_states": {
            "n_events": int(totals["all_events"]),
            "n_opportunities": int(totals["all_opps"]),
            "aggregate_rate": _rate(totals["all_events"], totals["all_opps"]),
        },
        "msp": _compartment(totals["msp_events"], totals["msp_opps"],
                            per_read_msp, min_state_opportunities),
        "outside_msp": _compartment(totals["out_events"], totals["out_opps"],
                                    per_read_out, min_state_opportunities),
        "msp_to_outside_ratio": (
            float(msp_rate / out_rate) if msp_rate is not None and out_rate else None),
        "msp_length_fraction": (
            float(totals["msp_bp"] / totals["total_bp"]) if totals["total_bp"] else None),
        "median_per_read_msp_length_fraction": (
            float(np.median(msp_fraction_per_read)) if msp_fraction_per_read else None),
        "terminal_excluded_opportunities": int(totals["excluded_opps"]),
    })
    if source in ("tags", "fibertools_tags"):
        block["tag_frames"] = {"MA": frames.get("MA"), "legacy": frames.get("legacy")}
        # Which FiberHMM wrote the calls: graded against this release's calls,
        # so older producers are worth a look when a grade is borderline.
        block["tag_producer"] = _last_fiberhmm_writer(header)
        block["reads"]["reads_with_calls"] = int(n_tagged)
    else:
        block["light_call"] = {
            **light,
            "model": caller.model_path.rsplit("/", 1)[-1],
            "model_sha256": caller.model_sha256,
            "context_size": int(caller.context_size),
            "stages": "apply HMM only (no nucleosome or TF recall)",
            "msp_min_size": 0,
            "nuc_min_size": 85,
        }
    if not block["available"]:
        block["note"] = "too few opportunities in one of the compartments"
    arrays = {
        "msp_rates": np.asarray(per_read_msp, dtype=float),
        "outside_rates": np.asarray(per_read_out, dtype=float),
        "msp_length_fractions": np.asarray(msp_fraction_per_read, dtype=float),
        "per_read_counts": np.asarray(per_read_ratio_inputs, dtype=np.int64).reshape(-1, 4),
    }
    return block, arrays


def _opportunity_text(mode: str) -> str:
    if mode == "daf":
        return ("C (CT reads) or G (GA reads) targets on the deaminated strand, "
                "as encoded for calling (chemistry adjacent-run mask and SNP mask "
                "applied)")
    if mode == "nanopore-fiber":
        return "basecalled-strand A (SEQ T on reverse reads), as encoded for calling"
    return "A and T (both strands), as encoded for calling"


# ---------------------------------------------------------------------------
# Grading against the packaged reference
# ---------------------------------------------------------------------------

#: Definition fields a sample must share with the reference to be graded.
#: Reference per-read quantiles that bound PASS/WARN (adjustable per profile
#: in references.json ``state_rates.grading``).
# Bands relative to the reference median (owner, 3 Oct): efficiency PASS while
# less than 20% below it, WARN 20-30% below, FAIL more than 30% below;
# background mirrors it above the median. A reference may instead give
# pass_quantile/warn_quantile (bounds from its per-read quantiles).
DEFAULT_GRADING = {
    "efficiency": {"pass_relative": 0.20, "warn_relative": 0.30},
    "background": {"pass_relative": 0.20, "warn_relative": 0.30},
}
GRADED_DEFINITION_FIELDS = (
    "min_msp_bp", "edge_trim_bp", "terminal_segments",
    "min_state_opportunities_per_read",
)
_REFERENCE_SOURCE = {"tags": "tags", "light_call": "light_call"}


def _high_is_good(value: float, warn_min: float, pass_min: float,
                  reference: float) -> float:
    """PASS >= pass_min (70-100 toward the reference median), WARN >= warn_min."""
    if value >= pass_min:
        return float(np.clip(70 + 30 * (value - pass_min)
                             / max(reference - pass_min, 1e-12), 70, 100))
    if value >= warn_min:
        return float(35 + 34 * (value - warn_min) / max(pass_min - warn_min, 1e-12))
    return float(34 * np.clip(value / max(warn_min, 1e-12), 0, 1))


def _low_is_good(value: float, pass_max: float, warn_max: float,
                 reference: float) -> float:
    """PASS <= pass_max (70-100 toward the reference median), WARN <= warn_max."""
    if value <= pass_max:
        return float(np.clip(70 + 30 * (pass_max - value)
                             / max(pass_max - reference, 1e-12), 70, 100))
    if value <= warn_max:
        return float(35 + 34 * (warn_max - value) / max(warn_max - pass_max, 1e-12))
    return float(34 * np.clip(1 - (value - warn_max) / max(warn_max - pass_max, 1e-12),
                              0, 1))


def _status(score: Optional[float]) -> str:
    if score is None:
        return "INSUFFICIENT"
    if score >= 70:
        return "PASS"
    if score >= 35:
        return "WARN"
    return "FAIL"


def _component(name: str, metric: str, value, **extra) -> dict:
    return {"metric": metric, "value": value, "score": None,
            "status": "INSUFFICIENT", "note": "", **extra}


def grade_state_rates(block: dict, profile: Optional[dict],
                      prob_threshold: int, min_rate_reads: int = 20) -> tuple[dict, dict]:
    """``(efficiency, background)`` verdict components for a state block.

    Efficiency is the median per-read in-MSP rate, graded one-sided against
    the reference's per-read in-MSP distribution: PASS at or above the
    reference 25th percentile, WARN down to the 5th, FAIL below. Background
    is the median per-read outside-MSP rate: PASS at or below the reference
    75th percentile, WARN up to the 95th, FAIL above. The reference is the
    same state source (calls or light call) under the same definition.
    """
    msp = block.get("msp") or {}
    outside = block.get("outside_msp") or {}
    efficiency = _component(
        "efficiency", "median per-read in-MSP rate", msp.get("median_per_read_rate"),
        n_rate_reads=msp.get("n_rate_reads", 0))
    background = _component(
        "background", "median per-read outside-MSP rate",
        outside.get("median_per_read_rate"), n_rate_reads=outside.get("n_rate_reads", 0))
    reference = (profile or {}).get("state_rates")

    def note(text):
        efficiency["note"] = background["note"] = text
        return efficiency, background

    if not block.get("available"):
        return note(block.get("note") or "state-aware rates unavailable")
    if not reference:
        return note("this QC reference has no state-aware calibration")
    if not reference.get("scoring_enabled", True):
        return note(reference.get("calibration_note", "state-aware calibration pending"))
    definition = block.get("definition", {})
    differing = [field for field in GRADED_DEFINITION_FIELDS
                 if definition.get(field) != reference.get("definition", {}).get(field)]
    if differing:
        return note("not graded: the state definition differs from the reference's "
                    f"({', '.join(differing)})")
    by_source = reference.get("by_source", {})
    source = _REFERENCE_SOURCE.get(block.get("source"))
    ref = by_source.get(source) if source else None
    if ref is None:
        return note(f"not graded: the reference has no calibration for states from "
                    f"{block.get('source')} (available: {', '.join(sorted(by_source)) or 'none'})")
    if block.get("definition", {}).get("daf_run_mask") != reference.get(
            "definition", {}).get("daf_run_mask", block.get("definition", {}).get("daf_run_mask")):
        return note("not graded: the DAF adjacent-run mask differs from the reference's")
    threshold = reference.get("definition", {}).get("probability_threshold")
    cap_note = ""
    if threshold is not None and abs(int(prob_threshold) - int(threshold)) > 5:
        cap_note = (f"reference was calibrated at ML threshold {threshold}; "
                    f"this run used {prob_threshold}")

    def quantile(compartment, probability):
        q = ref[compartment]
        return float(np.interp(probability, q["probabilities"], q["quantiles"]))

    grading = {name: {**DEFAULT_GRADING[name], **reference.get("grading", {}).get(name, {})}
               for name in DEFAULT_GRADING}
    for component, compartment, scorer in (
            (efficiency, "msp", "high"), (background, "outside_msp", "low")):
        median = quantile(compartment, 0.5)
        name = "efficiency" if scorer == "high" else "background"
        rule = {**DEFAULT_GRADING[name], **reference.get("grading", {}).get(name, {})}
        if "pass_quantile" in rule and "warn_quantile" in rule and "pass_relative" not in reference.get(
                "grading", {}).get(name, {}):
            pass_bound = quantile(compartment, rule["pass_quantile"])
            warn_bound = quantile(compartment, rule["warn_quantile"])
        else:
            sign = -1.0 if scorer == "high" else 1.0
            pass_bound = median * (1.0 + sign * float(rule["pass_relative"]))
            warn_bound = median * (1.0 + sign * float(rule["warn_relative"]))
        if scorer == "high":
            component.update(pass_min=pass_bound, warn_min=warn_bound)
        else:
            component.update(pass_max=pass_bound, warn_max=warn_bound)
        component["reference_median"] = median
        component["reference_source"] = source
        notes = [cap_note] if cap_note else []
        value = component["value"]
        if value is None or component["n_rate_reads"] < min_rate_reads:
            component["note"] = "; ".join(
                notes + [f"fewer than {min_rate_reads} reads with "
                         f">= {definition.get('min_state_opportunities_per_read')} "
                         "opportunities in this compartment"])
            continue
        score = (_high_is_good(value, warn_bound, pass_bound, median) if scorer == "high"
                 else _low_is_good(value, pass_bound, warn_bound, median))
        if cap_note:
            score = min(score, 69.0)
        component["score"] = score
        component["status"] = _status(score)
        component["note"] = "; ".join(notes)
    return efficiency, background


def reference_summary(block: dict, arrays: dict) -> dict:
    """Compact per-source reference entry built from a computed state block."""
    probabilities = [0.05, 0.25, 0.5, 0.75, 0.95]

    def compartment(key, values):
        part = block[key]
        return {
            "probabilities": probabilities,
            "quantiles": np.quantile(values, probabilities).tolist() if len(values) else None,
            "n_rate_reads": int(len(values)),
            "aggregate_rate": part["aggregate_rate"],
            "n_events": part["n_events"],
            "n_opportunities": part["n_opportunities"],
        }

    return {
        "msp": compartment("msp", arrays["msp_rates"]),
        "outside_msp": compartment("outside_msp", arrays["outside_rates"]),
        "all_states_rate": block["all_states"]["aggregate_rate"],
        "msp_to_outside_ratio": block["msp_to_outside_ratio"],
        "msp_length_fraction": block["msp_length_fraction"],
        "n_reads_used": block["reads"]["reads_used"],
    }
