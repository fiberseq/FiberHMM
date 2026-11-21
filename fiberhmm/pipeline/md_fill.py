"""MD tags for aligned records that came back from the aligner without one.

DAF-seq calling finds C->T / G->A deamination sites from the ``MD`` tag (or
R/Y codes, or a reference FASTA). ``fiberhmm-pipeline`` asks minimap2 for
``--MD``, but minimap2 2.27 ignores that option: a regression introduced with
``--ds`` writes a ``cs`` tag in place of ``MD`` (fixed in 2.28; 2.27 is the
version Ubuntu 24.04 ships). Every record of such a run lacks ``MD``, so the
aligned BAM, and everything read from it, would have no deamination source.

The pipeline holds the reference, so :class:`MissingMdFiller` computes
``MD`` (and ``NM`` when absent) for any mapped record without ``MD``, from
the reference, exactly as minimap2 2.28+ writes it, and drops a ``cs`` tag
the pipeline never asked for. It works from the record and the reference
alone, so it covers any aligner that leaves ``MD`` out.
"""
from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pysam

from fiberhmm.pipeline.circular import compute_md_nm

M, INS, DEL, REF_SKIP, SOFT, HARD, PAD, EQ, DIFF = range(9)
_N = ord("N")

# minimap2 releases whose ``--MD`` writes no MD tag.
MD_BROKEN_MINIMAP2 = frozenset({(2, 27)})
MD_FIXED_MINIMAP2 = "2.28"


def minimap2_md_broken(version: str) -> bool:
    """Whether a ``minimap2 --version`` string is a release whose ``--MD`` is broken."""
    match = re.match(r"\s*(\d+)\.(\d+)", version or "")
    return bool(match) and (int(match.group(1)), int(match.group(2))) in MD_BROKEN_MINIMAP2


def minimap2_md_bug_note(header) -> str:
    """A note for a BAM whose ``@PG`` shows minimap2 2.27 (empty otherwise).

    ``header`` is a ``pysam.AlignmentHeader`` or its ``to_dict()``. Used where
    a missing MD tag stops DAF-seq calling, so the user learns why ``--MD``
    gave none.
    """
    if hasattr(header, "to_dict"):
        header = header.to_dict()
    for pg in (header or {}).get("PG", []) or []:
        name = str(pg.get("PN") or pg.get("ID") or "")
        if name.split(".")[0] == "minimap2" and minimap2_md_broken(str(pg.get("VN") or "")):
            return (f"  The BAM header shows minimap2 {pg.get('VN')}: that release ignores\n"
                    f"  --MD (it writes a cs tag instead; fixed in minimap2 {MD_FIXED_MINIMAP2}).\n"
                    f"  Pass --reference ref.fa, run 'samtools calmd', or realign with\n"
                    f"  minimap2 {MD_FIXED_MINIMAP2} or newer.\n")
    return ""


def md_nm_linear(query: str, cigar: Sequence[tuple[int, int]], ref: str) -> tuple[str, int]:
    """``MD``/``NM`` of an alignment whose reference span is ``ref``.

    ``ref`` starts at the record's reference start and covers its whole
    reference span. The result equals :func:`compute_md_nm` (and minimap2's
    MD: an ``N`` in the reference is a mismatch); the aligned blocks are
    compared with numpy, so long reads cost little.
    """
    q_arr = np.frombuffer(query.upper().encode("ascii"), dtype=np.uint8)
    ref = ref.upper()
    r_arr = np.frombuffer(ref.encode("ascii"), dtype=np.uint8)
    md: list[str] = []
    matches = 0
    nm = 0
    q = r = 0
    for op, n in cigar:
        if op in (M, EQ, DIFF):
            rb = r_arr[r:r + n]
            mismatch = np.flatnonzero((q_arr[q:q + n] != rb) | (rb == _N))
            previous = 0
            for i in mismatch.tolist():
                md.append(str(matches + i - previous))
                md.append(ref[r + i])
                matches = 0
                previous = i + 1
            matches += n - previous
            nm += len(mismatch)
            q += n
            r += n
        elif op == INS:
            q += n
            nm += n
        elif op == DEL:
            md.append(str(matches))
            md.append("^" + ref[r:r + n])
            matches = 0
            r += n
            nm += n
        elif op == REF_SKIP:
            r += n
        elif op == SOFT:
            q += n
    md.append(str(matches))
    return "".join(md), nm


class MissingMdFiller:
    """Add ``MD`` (and a missing ``NM``) to mapped records that lack ``MD``.

    ``fasta`` is the indexed reference the reads were aligned to; it is
    opened on first use. ``circular`` maps a circular contig's name to its
    sequence, for a record that runs past the contig end (SAM ``TP:circular``).
    """

    def __init__(self, fasta: str, circular: Optional[dict] = None):
        self.fasta_path = fasta
        self.circular = circular or {}
        self._fasta: Optional[pysam.FastaFile] = None
        self.filled = 0

    def __call__(self, read: pysam.AlignedSegment) -> bool:
        if read.is_unmapped or read.has_tag("MD") or not read.cigartuples:
            return False
        seq = read.query_sequence
        if not seq:
            return False
        contig = read.reference_name
        start, end = read.reference_start, read.reference_end
        if self._fasta is None:
            self._fasta = pysam.FastaFile(self.fasta_path)
        length = self._fasta.get_reference_length(contig)
        if end <= length:
            md, nm = md_nm_linear(seq, read.cigartuples,
                                  self._fasta.fetch(contig, start, end))
        else:
            sequence = self.circular.get(contig)
            if sequence is None:
                sequence = self.circular[contig] = self._fasta.fetch(contig).upper()
            md, nm = compute_md_nm(seq.upper(), read.cigartuples, sequence, start)
        read.set_tag("MD", md, "Z")
        if not read.has_tag("NM"):
            read.set_tag("NM", nm, "i")
        # minimap2 2.27 writes cs in place of MD; the pipeline never asks for cs.
        if read.has_tag("cs"):
            read.set_tag("cs", None)
        self.filled += 1
        return True

    def close(self) -> None:
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None
