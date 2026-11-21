"""Consensus DAF duplicate collapse can apply fiberhmm-dedup's endpoint check (audit fh-audit-core 3.6)."""
from types import SimpleNamespace

import numpy as np

from fiberhmm.cli.dedup import cluster_reads
from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.inference.strand_rescue import collapse_amplified_reads


def fibers(*spans):
    shared = np.arange(1200, 1400, 10)  # identical deamination sets inside the shared stretch only
    return [SimpleNamespace(library_id='x', fingerprint_positions=shared, positions=shared,
                            hits=np.ones(len(shared), bool), strand='CT', ref_start=a, ref_end=b)
            for a, b in spans]


def test_default_collapses_on_the_whole_read_fingerprint_alone():
    kept, info = collapse_amplified_reads(fibers((100, 2000), (1000, 3000)))
    assert len(kept) == 1 and info['duplicate_reads_collapsed'] == 1
    assert parse_options({})['input'].molecule_max_end_diff_bp == -1


def test_endpoint_check_matches_fiberhmm_dedup():
    spans = [(100, 2000), (1000, 3000), (120, 2030)]
    kept, _ = collapse_amplified_reads(fibers(*spans), max_end_diff=50)
    sets = [frozenset(range(1200, 1400, 10))] * 3
    labels = cluster_reads(sets, [('CT',)] * 3, 0.95, 32, 8, 7, endpoints=spans, max_end_diff=50)
    assert len(kept) == len(set(labels.tolist())) == 2
    assert sorted((r.ref_start, r.ref_end) for r in kept)[1] == (1000, 3000)


def test_endpoint_check_is_a_consensus_parameter():
    options = parse_options({'input': {'molecule_max_end_diff_bp': 50}})
    assert options['input'].molecule_max_end_diff_bp == 50
