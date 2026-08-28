"""Cross-strand analysis for DAF-seq (DddA) single-molecule data.

DddA deaminates *both* strands of a duplex; after denaturation each strand is
amplified and sequenced independently, so a molecule appears as two reads of
opposite deamination flavor -- ``CT`` (C->T, forward strand) and ``GA`` (G->A,
reverse strand) -- that overlap in the genome but sample different bases (C vs
G). They therefore cannot be matched by deamination pattern the way same-strand
PCR/PTA copies can.

Pairing is sequence-first. Opposite-flavor reads are compared at shared
reference A/T positions, which are outside the DddA C/G channels, and only
strict reciprocal preferences or constrained local 2x2 assignments are
accepted. Reads that remain sequence-ambiguous can enter a separately labeled
**nucleosome-footprint fallback**: MA ``nuc`` dyads become 1-D density signals
and reciprocal-best pairs must clear correlation and margin gates while gross
sequence conflicts are vetoed. The fallback is useful operationally, but later
footprint-agreement analyses must retain the route label because those pairs
were selected partly through footprint similarity. Ambiguous cases fail closed
rather than being forced into a chromosome-scale assignment.

See :mod:`fiberhmm.crossstrand.pairing`.
"""
