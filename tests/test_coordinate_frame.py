"""Tests for SEQ<->molecular coordinate-frame handling + nuc/MSP re-tiling.

FiberHMM works in SEQ (query_sequence) coords internally, but ns/nl/as/al and
MA must be written in molecular (original-fiber) frame for fibertools. For
reverse reads the frames are reverse complements.
"""
from __future__ import annotations

from fiberhmm.inference.nuc_recaller import NucCall, assemble_nuc_msp_tiling
from fiberhmm.io.ma_tags import flip_interval_frame, flip_intervals_to_seq


class _Read:
    """Minimal stand-in for a pysam read."""
    def __init__(self, is_reverse, query_length):
        self.is_reverse = is_reverse
        self.query_length = query_length


def test_flip_interval_frame_is_involution():
    L = 1000
    s, length = 100, 50
    ms, ml = flip_interval_frame(s, length, L)
    assert (ms, ml) == (L - (s + length), length) == (850, 50)
    # applying twice returns the original
    assert flip_interval_frame(ms, ml, L) == (s, length)


def test_flip_intervals_to_seq_forward_is_noop():
    fwd = _Read(is_reverse=False, query_length=1000)
    s, l = flip_intervals_to_seq([100, 300], [50, 40], fwd)
    assert s == [100, 300] and l == [50, 40]


def test_flip_intervals_to_seq_reverse_roundtrip():
    rev = _Read(is_reverse=True, query_length=1000)
    starts, lengths = [100, 300], [50, 40]
    ms, ml = flip_intervals_to_seq(starts, lengths, rev)
    # coords flipped (L-(s+l)); input order preserved (no re-sort on read)
    assert ms == [850, 660] and ml == [50, 40]
    # flipping again recovers the originals (as a set of intervals)
    back_s, back_l = flip_intervals_to_seq(ms, ml, rev)
    assert set(zip(back_s, back_l)) == set(zip(starts, lengths))


def test_assemble_prefers_longer_at_same_start_and_drops_subfloor():
    # Codex regression: a short same-start nuc must NOT clip the promoted
    # full-length nuc into sub-nucleosome pieces. Longer wins; the short one is
    # dropped (its span reverts to MSP), not emitted as a <nuc_min_size call.
    nucs = [
        NucCall(start=0, length=30, nq=100, el=255, er=255),    # short
        NucCall(start=0, length=100, nq=200, el=255, er=255),   # promoted full nuc
    ]
    kept, msps = assemble_nuc_msp_tiling(
        nucs, span_lo=0, span_hi=200, msp_min_size=0, nuc_min_size=85)
    assert [(k.start, k.length) for k in kept] == [(0, 100)]
    assert (100, 100) in [(s, l) for s, l in msps]


def test_assemble_clips_partial_overlap_above_floor():
    # two long nucs with a small overlap -> clip to adjacency; both survive.
    nucs = [
        NucCall(start=0, length=200, nq=200, el=255, er=255),    # [0,200)
        NucCall(start=150, length=200, nq=180, el=255, er=255),  # [150,350) -> clip to [200,350)
    ]
    kept, msps = assemble_nuc_msp_tiling(
        nucs, span_lo=0, span_hi=400, msp_min_size=0, nuc_min_size=85)
    iv = [(k.start, k.length) for k in kept]
    assert iv == [(0, 200), (200, 150)]
    assert kept[1].el == 0  # clipped left edge zeroed
    # tiling: nucs + msps cover [0,400) with no gaps/overlaps
    spans = sorted([(s, s + l) for s, l in iv] + [(s, s + l) for s, l in msps])
    assert spans[0][0] == 0 and spans[-1][1] == 400
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
