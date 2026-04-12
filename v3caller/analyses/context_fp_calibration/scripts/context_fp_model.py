"""Per-context false positive model for m6A (PacBio) and C→T (Nanopore).

Learns the per-sequence-context false positive rate from an untreated
control BAM (no enzyme, just basecaller noise). The FP rate varies
~15× across 3-mer contexts due to CpG-adjacent kinetic artifacts in
PacBio's m6A model.

Usage:
    # Calibrate from untreated control
    model = ContextFPModel.from_bam('untreated.m6a.bam',
                                      context_size=3, ml_threshold=128)
    model.save('fp_model.json')

    # Load and use in the caller
    model = ContextFPModel.load('fp_model.json')
    fp_rate = model.fp_rate('GAG')  # → 0.024
    expected = model.expected_fp_hits(positions, query_seq)  # → float

Designed for PacBio m6A but the framework is enzyme-agnostic:
- PacBio m6A: target bases are A and T, contexts are trinucleotides
  around A/T positions. Untreated control = no Hia5/DddA enzyme.
- Nanopore C→T: target bases are C, contexts are trinucleotides
  around C positions. Untreated control = no deaminase.
  (Not yet implemented but the data structures are ready.)
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class ContextFPModel:
    """Per-sequence-context false positive rate model.

    Stores a lookup table mapping sequence context strings (e.g.
    'GAG', 'CTC') to their empirical FP rate from an untreated
    control. Contexts not seen in calibration fall back to the
    global mean.
    """

    def __init__(self, context_size: int, rates: Dict[str, float],
                 global_rate: float, target_bases: str = 'AT',
                 n_reads: int = 0, n_opps: int = 0, n_hits: int = 0):
        self.context_size = context_size
        self.rates = rates
        self.global_rate = global_rate
        self.target_bases = target_bases
        self.n_reads = n_reads
        self.n_opps = n_opps
        self.n_hits = n_hits

    def fp_rate(self, context: str) -> float:
        """FP rate for a given sequence context.

        Falls back to global mean for unseen contexts.
        """
        return self.rates.get(context.upper(), self.global_rate)

    def fp_rate_at(self, pos: int, query_seq: str) -> float:
        """FP rate at a specific query position given the full sequence."""
        half = self.context_size // 2
        lo = pos - half
        hi = pos + half + 1
        if lo < 0 or hi > len(query_seq):
            return self.global_rate
        ctx = query_seq[lo:hi].upper()
        if 'N' in ctx:
            return self.global_rate
        return self.fp_rate(ctx)

    def expected_fp_hits(self, positions: np.ndarray,
                          query_seq: str) -> float:
        """Expected number of FP hits across a set of positions.

        Sum of per-position FP rates. Used in the merge step to
        decide whether observed hits in a gap are consistent with
        basecaller noise.
        """
        total = 0.0
        half = self.context_size // 2
        qlen = len(query_seq)
        q_upper = query_seq.upper()
        for p in positions:
            p = int(p)
            lo = p - half
            hi = p + half + 1
            if lo < 0 or hi > qlen:
                total += self.global_rate
            else:
                ctx = q_upper[lo:hi]
                if 'N' in ctx:
                    total += self.global_rate
                else:
                    total += self.rates.get(ctx, self.global_rate)
        return total

    def expected_fp_array(self, query_seq: str,
                            opp_mask: np.ndarray) -> np.ndarray:
        """Per-position FP rate array for all opp positions.

        Returns an array of shape (len(query_seq),) where non-opp
        positions are 0 and opp positions carry their context-specific
        FP rate. Used for per-position scoring in the TF caller.
        """
        qlen = len(query_seq)
        half = self.context_size // 2
        out = np.zeros(qlen, dtype=np.float64)
        q_upper = query_seq.upper()
        for p in range(qlen):
            if not opp_mask[p]:
                continue
            lo = p - half
            hi = p + half + 1
            if lo < 0 or hi > qlen:
                out[p] = self.global_rate
            else:
                ctx = q_upper[lo:hi]
                if 'N' in ctx:
                    out[p] = self.global_rate
                else:
                    out[p] = self.rates.get(ctx, self.global_rate)
        return out

    # ---- serialization ----

    def save(self, path: str):
        """Save to JSON."""
        obj = {
            'context_size': self.context_size,
            'target_bases': self.target_bases,
            'global_rate': self.global_rate,
            'n_reads': self.n_reads,
            'n_opps': self.n_opps,
            'n_hits': self.n_hits,
            'rates': self.rates,
        }
        with open(path, 'w') as f:
            json.dump(obj, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ContextFPModel':
        """Load from JSON."""
        with open(path) as f:
            obj = json.load(f)
        return cls(
            context_size=obj['context_size'],
            rates=obj['rates'],
            global_rate=obj['global_rate'],
            target_bases=obj.get('target_bases', 'AT'),
            n_reads=obj.get('n_reads', 0),
            n_opps=obj.get('n_opps', 0),
            n_hits=obj.get('n_hits', 0),
        )

    # ---- calibration ----

    @classmethod
    def from_bam(cls, bam_path: str, context_size: int = 3,
                  ml_threshold: int = 128,
                  target_bases: str = 'AT',
                  max_reads: int = 10000,
                  min_context_opps: int = 100) -> 'ContextFPModel':
        """Calibrate from an untreated control BAM.

        Parses MM/ML tags for m6A calls (same as Hia5Extractor),
        extracts per-context hit rates. Works on both aligned and
        unmapped reads (uses query sequence context).

        Args:
            bam_path: untreated control BAM with MM/ML tags
            context_size: 1, 3, 5, or 7 (default 3 — sweet spot)
            ml_threshold: ML score threshold for calling a hit
            target_bases: which bases are opportunities ('AT' for
                m6A, 'C' for C→T deamination)
            max_reads: cap on reads to process
            min_context_opps: minimum observations per context to
                include in the model (contexts below this fall back
                to global rate)
        """
        import pysam

        half = context_size // 2
        counts = defaultdict(lambda: [0, 0])  # context → [opps, hits]
        total_opps = total_hits = 0
        n_reads = 0

        bam = pysam.AlignmentFile(bam_path, 'rb', check_sq=False)
        for read in bam.fetch(until_eof=True):
            if read.is_secondary or read.is_supplementary:
                continue
            q = read.query_sequence
            if q is None:
                continue
            qlen = len(q)
            q_upper = q.upper()
            target_set = set(target_bases.upper())

            # Parse m6A calls from MM/ML
            hit_set = set()
            mm_str = ''
            try:
                mm_str = read.get_tag('MM') if read.has_tag('MM') else ''
            except KeyError:
                pass
            if not mm_str:
                try:
                    mm_str = read.get_tag('Mm') if read.has_tag('Mm') else ''
                except KeyError:
                    pass
            ml = None
            try:
                ml = list(read.get_tag('ML')) if read.has_tag('ML') else None
            except KeyError:
                pass
            if mm_str and ml is not None:
                ml_idx = 0
                for section in mm_str.rstrip(';').split(';'):
                    if not section:
                        continue
                    parts = section.split(',')
                    header = parts[0]
                    if len(header) < 3:
                        continue
                    base = header[0]
                    code = header[2]
                    try:
                        skips = [int(x) for x in parts[1:] if x]
                    except ValueError:
                        continue
                    if code != 'a':
                        ml_idx += len(skips)
                        continue
                    target = base.upper()
                    qp = 0
                    for skip in skips:
                        if ml_idx >= len(ml):
                            break
                        score = ml[ml_idx]
                        ml_idx += 1
                        needed = skip + 1
                        while qp < qlen and needed > 0:
                            if q_upper[qp] == target:
                                needed -= 1
                                if needed == 0:
                                    break
                            qp += 1
                        if qp >= qlen:
                            break
                        if score >= ml_threshold:
                            hit_set.add(qp)
                        qp += 1

            # Accumulate per-context counts
            for i in range(qlen):
                if q_upper[i] not in target_set:
                    continue
                is_hit = 1 if i in hit_set else 0
                lo = i - half
                hi = i + half + 1
                if lo < 0 or hi > qlen:
                    ctx = '_EDGE_'
                else:
                    ctx = q_upper[lo:hi]
                    if 'N' in ctx:
                        ctx = '_N_'
                counts[ctx][0] += 1
                counts[ctx][1] += is_hit
                total_opps += 1
                total_hits += is_hit

            n_reads += 1
            if n_reads >= max_reads:
                break
        bam.close()

        # Build rate table
        global_rate = total_hits / max(1, total_opps)
        rates = {}
        for ctx, (opps, hits) in counts.items():
            if ctx.startswith('_'):
                continue  # skip edge/N contexts
            if opps >= min_context_opps:
                rates[ctx] = hits / opps

        return cls(
            context_size=context_size,
            rates=rates,
            global_rate=global_rate,
            target_bases=target_bases,
            n_reads=n_reads,
            n_opps=total_opps,
            n_hits=total_hits,
        )

    # ---- summary ----

    def summary(self) -> str:
        rates = np.array(list(self.rates.values()))
        lines = [
            f'ContextFPModel: {self.context_size}-mer, '
            f'{len(self.rates)} contexts, '
            f'target={self.target_bases}',
            f'  calibrated from {self.n_reads} reads, '
            f'{self.n_opps} opps, {self.n_hits} hits',
            f'  global FP rate: {self.global_rate:.5f}',
        ]
        if len(rates) > 0:
            lines.append(
                f'  per-context: mean={rates.mean():.5f} '
                f'min={rates.min():.5f} max={rates.max():.5f} '
                f'CV={rates.std()/max(1e-9, rates.mean()):.2f}'
            )
            # Top and bottom 3
            sorted_ctx = sorted(self.rates.items(), key=lambda x: -x[1])
            lines.append('  highest FP: ' +
                          ', '.join(f'{c}={r:.4f}' for c, r in sorted_ctx[:3]))
            lines.append('  lowest FP:  ' +
                          ', '.join(f'{c}={r:.4f}' for c, r in sorted_ctx[-3:]))
        return '\n'.join(lines)
