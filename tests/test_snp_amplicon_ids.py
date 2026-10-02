"""SNP report amplicon IDs survive renumbering (core audit 2.4).

Amplicons are numbered in discovery order, then renumbered by read count.
Renaming one ID at a time let a later rename overwrite an earlier one when two
amplicons swapped places, so both SNPs named the same amplicon in
<prefix>.json (masks and amplicons.tsv were unaffected).
"""
from fiberhmm.daf.snps import _renumber_amplicons


def test_swapped_amplicons_keep_their_own_calls():
    # Discovery: right amplicon first (amplicon_1); the left one has more reads.
    amplicons = [
        {"amplicon_id": "amplicon_2", "chrom": "c", "consensus_start_0based": 100},
        {"amplicon_id": "amplicon_1", "chrom": "c", "consensus_start_0based": 20000},
    ]  # already sorted by read count, as the caller does
    calls = [
        {"position_0based": 21000, "amplicon_ids": ["amplicon_1"]},
        {"position_0based": 1000, "amplicon_ids": ["amplicon_2"]},
        {"position_0based": 9999, "amplicon_ids": ["amplicon_1", "amplicon_2"]},
        {"position_0based": 5, "amplicon_ids": []},
    ]
    _renumber_amplicons(amplicons, calls)
    assert [(a["amplicon_id"], a["consensus_start_0based"]) for a in amplicons] == [
        ("amplicon_1", 100), ("amplicon_2", 20000)]
    assert [c["amplicon_ids"] for c in calls] == [
        ["amplicon_2"], ["amplicon_1"], ["amplicon_2", "amplicon_1"], []]

