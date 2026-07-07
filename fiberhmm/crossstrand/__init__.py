"""Cross-strand analysis for DAF-seq (DddA) single-molecule data.

DddA deaminates *both* strands of a duplex; after denaturation each strand is
amplified and sequenced independently, so a molecule appears as two reads of
opposite deamination flavor -- ``CT`` (C->T, forward strand) and ``GA`` (G->A,
reverse strand) -- that overlap in the genome but sample different bases (C vs
G). They therefore cannot be matched by deamination pattern the way same-strand
PCR/PTA copies can.

The pairing here instead matches the two strands of one molecule by their
**nucleosome footprint pattern**, which is a physical property of the duplex and
so is shared by both strands. Concretely, each read's MA ``nuc`` dyads become a
1-D dyad-density signal along the reference; two reads are scored by the
normalized cross-correlation of those signals. Because different homologs have
near-uncorrelated nucleosome arrays (~0.2) while the same molecule's two strands
correlate distinctly higher, the assignment at a locus (<=2 CT + <=2 GA reads,
the diploid ceiling) is a small, well-posed problem: pick the matching whose
correlation clearly beats the alternative, and declare the locus unresolved when
it does not.

See :mod:`fiberhmm.crossstrand.pairing`.
"""
