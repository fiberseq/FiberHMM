"""Enzyme-specific extractors that produce (opp, hit) arrays in reference
coordinates for a single read.

The core-find back-end is enzyme-independent — it only cares about the
per-base opportunity and hit indicator arrays. What changes per enzyme
is how those arrays are derived from the read's sequence and/or
modification tags.

Contract
--------
Every extractor exposes:

    read_to_arrays(read, ref_seq) -> (opp, hit)

where
    read      : pysam.AlignedSegment
    ref_seq   : uppercase reference sequence covering
                [read.reference_start, read.reference_end)
    opp       : int8 array of length L = reference_end - reference_start,
                value 1 at reference-coord positions that are opportunities
                for the enzyme, 0 otherwise
    hit       : int8 array of the same length; 1 at opportunity positions
                that were actually called as modified / deaminated

Both arrays are in reference (forward) coordinates. Indels are skipped.

Low hit rate -> protected (nucleosome core) for ALL supported enzymes,
so the same core-find downstream works on any of them.
"""

from __future__ import annotations

import numpy as np


class DAFExtractor:
    """DddA / DddB / scDAF reads — handles BOTH IUPAC-encoded (R/Y)
    and raw-mismatch (C→T / G→A) BAMs.

    All DAF data is strand-specific biologically: deamination happens on
    ONE strand of each duplex. The encoding convention depends on the
    pipeline:

      - IUPAC-encoded BAMs (e.g. scDAF, NAPA "for eitan"): deaminated
        C→T is stored as Y in the query, G→A as R.
      - Raw-mismatch BAMs (e.g. DddB spacetime): deaminated C→T is
        stored as a plain T in the query (just a mismatch vs ref C).

    The extractor auto-detects which encoding is active per-read by
    counting all four hit types (Y, T at C-in-ref; R, A at G-in-ref)
    and picking the majority. This lets a single extractor handle any
    DAF BAM without per-dataset configuration.
    """

    name = 'daf'

    def read_to_arrays(self, read, ref_seq):
        L = read.reference_end - read.reference_start
        opp_c = np.zeros(L, dtype=np.int8)
        opp_g = np.zeros(L, dtype=np.int8)
        hit_y = np.zeros(L, dtype=np.int8)
        hit_t = np.zeros(L, dtype=np.int8)
        hit_r = np.zeros(L, dtype=np.int8)
        hit_a = np.zeros(L, dtype=np.int8)
        q = read.query_sequence
        if q is None:
            return opp_c, hit_y

        n_y = n_t = n_r = n_a = 0
        for qp, rp in read.get_aligned_pairs(matches_only=True):
            rel = rp - read.reference_start
            if rel < 0 or rel >= L:
                continue
            rb = ref_seq[rel]
            qb = q[qp].upper()
            if rb == 'C':
                opp_c[rel] = 1
                if qb == 'Y':
                    hit_y[rel] = 1
                    n_y += 1
                elif qb == 'T':
                    hit_t[rel] = 1
                    n_t += 1
            elif rb == 'G':
                opp_g[rel] = 1
                if qb == 'R':
                    hit_r[rel] = 1
                    n_r += 1
                elif qb == 'A':
                    hit_a[rel] = 1
                    n_a += 1

        # Majority vote across all four encoding styles.
        counts = {'CtoY': (opp_c, hit_y, n_y),
                  'CtoT': (opp_c, hit_t, n_t),
                  'GtoR': (opp_g, hit_r, n_r),
                  'GtoA': (opp_g, hit_a, n_a)}
        best = max(counts, key=lambda k: counts[k][2])
        return counts[best][0], counts[best][1]


class Hia5Extractor:
    """Hia5 m6A reads from fibertools, MM/ML encoded.

    Hia5 methylates accessible adenines on BOTH strands of the duplex
    (fiber-seq). Fibertools reports two mod tracks per read:
      ('A', 0, 'a')  m6A on the forward strand at query-A positions
      ('T', 1, 'a')  m6A on the reverse strand at query-T positions
    Opportunities are every aligned A or T in the query; hits are sites
    whose ML score exceeds --ml-threshold. Both tracks are combined
    because both strands carry independent m6A information.

    Note: the 5mC CpG track ('C', 0, 'm') is IGNORED here — the project
    data is Drosophila, which has essentially no 5mC methylation, so
    this track is uninformative.
    """

    name = 'hia5'

    def __init__(self, ml_threshold: int = 128):
        self.ml_threshold = ml_threshold

    def read_to_arrays(self, read, ref_seq):
        L = read.reference_end - read.reference_start
        opp = np.zeros(L, dtype=np.int8)
        hit = np.zeros(L, dtype=np.int8)
        q = read.query_sequence
        if q is None:
            return opp, hit

        # Build a query-index -> ref-relative map for aligned positions.
        qpos_to_rel = np.full(len(q), -1, dtype=np.int32)
        for qp, rp in read.get_aligned_pairs(matches_only=True):
            rel = rp - read.reference_start
            if 0 <= rel < L:
                qpos_to_rel[qp] = rel

        # Opportunities: aligned A or T in the query (covers both
        # duplex strands — fibertools reports m6A on both).
        qs = q.upper()
        for qp in range(len(qs)):
            rel = qpos_to_rel[qp]
            if rel >= 0 and (qs[qp] == 'A' or qs[qp] == 'T'):
                opp[rel] = 1

        # Hits: m6A calls above threshold. Parse MM/ML tags directly
        # rather than using read.modified_bases — the latter has had
        # memory-stability issues in some pysam versions when iterating
        # over many reads.
        try:
            mm_str = read.get_tag('MM') if read.has_tag('MM') else ''
        except KeyError:
            mm_str = ''
        if not mm_str:
            try:
                mm_str = read.get_tag('Mm') if read.has_tag('Mm') else ''
            except KeyError:
                mm_str = ''
        try:
            ml = read.get_tag('ML') if read.has_tag('ML') else None
        except KeyError:
            ml = None
        if not mm_str or ml is None:
            return opp, hit

        # Snapshot ML into a plain Python list to avoid any lazy
        # buffer issues with pysam's array-like return.
        ml_list = list(ml)

        # Parse MM sections (';'-separated), each 'BASE+STRAND+CODE[.|?],skip1,skip2,...'
        ml_idx = 0
        for section in mm_str.rstrip(';').split(';'):
            if not section:
                continue
            parts = section.split(',')
            if not parts[0]:
                continue
            header = parts[0]
            try:
                skips = [int(x) for x in parts[1:] if x]
            except ValueError:
                continue
            if len(header) < 3:
                continue
            base = header[0]
            # strand = header[1]  # '+' or '-'
            code = header[2]
            # header[3:] may be '.' or '?' (count vs probability convention)
            # We only care about 'a' (m6A)
            if code != 'a':
                ml_idx += len(skips)
                continue

            # Walk the query, counting `base` occurrences along the strand
            # of origin. For the '+' strand the base is in the forward query
            # sequence; for '-' strand we skip occurrences of the complement
            # in the forward query sequence (fibertools uses base='T' for
            # reverse-strand A methylation).
            target_base = base.upper()
            seen = 0
            qp_cursor = 0
            for skip in skips:
                if ml_idx >= len(ml_list):
                    break
                score = ml_list[ml_idx]
                ml_idx += 1
                # Advance `skip` occurrences of target_base, then mark the next
                needed = skip + 1
                while qp_cursor < len(q) and needed > 0:
                    if q[qp_cursor].upper() == target_base:
                        needed -= 1
                        if needed == 0:
                            break
                    qp_cursor += 1
                if qp_cursor >= len(q):
                    break
                if score >= self.ml_threshold:
                    rel = qpos_to_rel[qp_cursor]
                    if rel >= 0:
                        hit[rel] = 1
                qp_cursor += 1

        return opp, hit


def get_extractor(name: str, **kwargs):
    """Factory: return an extractor by name."""
    name = name.lower()
    if name == 'daf':
        return DAFExtractor()
    if name == 'hia5':
        return Hia5Extractor(**kwargs)
    raise ValueError(f'unknown enzyme: {name!r} (known: daf, hia5)')
