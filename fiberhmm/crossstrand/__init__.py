"""Cross-strand analysis for DAF-seq (DddA) single-molecule data.

DddA deaminates *both* strands of a duplex; after denaturation each strand is
amplified and sequenced independently, so a molecule appears as two reads of
opposite deamination flavor -- ``CT`` (C->T, forward strand) and ``GA`` (G->A,
reverse strand) -- that overlap in the genome but sample different bases (C vs
G). They therefore cannot be matched by deamination pattern the way same-strand
PCR/PTA copies can.

The public pairer combines two explicit routes. Direct A/T sequence-supported
assignments carry ``pm:S``. Remaining high-confidence assignments from the
frozen sequence-free model carry ``pm:D``; that model uses nucleosome lattice,
alignment geometry, and raw and component-residual DddA protection. Sequence
assignments take precedence on conflicts. Ambiguous cases fail closed rather
than being forced into a chromosome-scale assignment.

See :mod:`fiberhmm.crossstrand.pairing`.
"""
