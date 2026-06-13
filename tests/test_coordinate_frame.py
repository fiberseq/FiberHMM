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


def test_assemble_tiling_clips_overlaps_and_fills_msps():
    # overlapping nucleosomes (e.g. a promoted nuc overlapping an existing one)
    nucs = [
        NucCall(start=10, length=100, nq=200, el=255, er=255),   # [10,110)
        NucCall(start=90, length=100, nq=180, el=255, er=255),   # overlaps -> clipped to [110,190)
        NucCall(start=250, length=80, nq=150, el=255, er=255),   # [250,330)
    ]
    kept, msps = assemble_nuc_msp_tiling(nucs, span_lo=0, span_hi=400, msp_min_size=0)
    iv = [(k.start, k.start + k.length) for k in kept]
    # non-overlapping, sorted
    assert iv == sorted(iv)
    assert all(iv[i][1] <= iv[i + 1][0] for i in range(len(iv) - 1))
    # clipped nuc lost its (now-meaningless) left edge byte
    assert kept[1].el == 0
    # MSPs are the exact complement within [0, 400)
    msp_iv = [(s, s + l) for s, l in msps]
    assert msp_iv == [(0, 10), (190, 250), (330, 400)]
    # nucs + msps tile [0,400) with no gaps or overlaps
    allspans = sorted(iv + msp_iv)
    assert allspans[0][0] == 0 and allspans[-1][1] == 400
    assert all(allspans[i][1] == allspans[i + 1][0] for i in range(len(allspans) - 1))
