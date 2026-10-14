"""First-pass HMM CpG masking for DddA CpG-aware calling."""
import numpy as np

import fiberhmm.inference.engine as engine
from fiberhmm.core.bam_reader import (
    ContextEncoder,
    daf_cpg_target_mask,
    encode_from_query_sequence,
)


def test_cpg_targets_ct_strand():
    # C->T strand: target C/T followed by G. Positions 4 (CG) and 9 (TG, a
    # deaminated CpG) are CpGs; 1 (CA) and 6 (CC) are not.
    seq = 'ACAACGCCATGA'
    mods = {9}  # the T at 9 is a converted C
    mask = daf_cpg_target_mask(seq, mods, '+')
    assert mask.tolist() == [i in (4, 9) for i in range(len(seq))]


def test_cpg_targets_ga_strand():
    # G->A strand: target G/A preceded by C. 'CG' at 2-3 and 'CA' (converted
    # CpG) at 6-7 are CpGs; the G at 9 follows T and is not.
    seq = 'ATCGTTCAATGA'
    mods = {7}
    mask = daf_cpg_target_mask(seq, mods, '-')
    assert mask.tolist() == [i in (3, 7) for i in range(len(seq))]


def _fiber_read(seq, mods):
    return {'read_id': 'r', 'query_sequence': seq, 'm6a_query_positions': set(mods),
            'query_length': len(seq), 'is_reverse': False, '_daf_strand': '+'}


def _capture(monkeypatch):
    seen = {}

    def fake_predict(model, encoded, *args, **kwargs):
        seen['encoded'] = np.array(encoded, copy=True)
        return {'footprint_starts': np.array([0]), 'footprint_sizes': np.array([5]),
                'msp_starts': np.array([]), 'msp_sizes': np.array([])}

    monkeypatch.setattr(engine, 'predict_footprints_and_msps', fake_predict)
    return seen


def test_hmm_sees_masked_cpgs_recall_keeps_them(monkeypatch):
    seq = 'ATTACGATCAGTTACGTACCATCGATTACA' * 2
    mods = {i for i, b in enumerate(seq) if b == 'T'}
    seen = _capture(monkeypatch)
    plain = encode_from_query_sequence(seq, mods, 0, mode='daf', strand='+', context_size=3)
    fr = _fiber_read(seq, mods)
    fr['_hmm_cpg_mask'] = np.ones(len(seq), dtype=bool)
    res = engine._process_single_read(fr, None, 0, False, 'daf', 3, 1, False,
                                      include_encoded=True)
    cpg = daf_cpg_target_mask(seq, mods, '+')
    non_target = 2 * ContextEncoder.get_n_codes(3) + 1
    assert cpg.any()
    hmm = seen['encoded']
    assert (hmm[cpg] == non_target).all()
    assert (hmm[~cpg] == plain[~cpg]).all()
    # Recall receives the unmasked encoding.
    assert (np.asarray(res['encoded']) == plain).all()


def test_mask_respects_unmethylated_islands(monkeypatch):
    seq = 'ATTACGATCAGTTACGTACCATCGATTACA' * 2
    mods = set()
    seen = _capture(monkeypatch)
    plain = encode_from_query_sequence(seq, mods, 0, mode='daf', strand='+', context_size=3)
    keep = np.ones(len(seq), dtype=bool)
    keep[:30] = False  # an unmethylated island: CpGs here stay visible
    fr = _fiber_read(seq, mods)
    fr['_hmm_cpg_mask'] = keep
    engine._process_single_read(fr, None, 0, False, 'daf', 3, 1, False)
    cpg = daf_cpg_target_mask(seq, mods, '+')
    hmm = seen['encoded']
    assert (hmm[cpg & ~keep] == plain[cpg & ~keep]).all()
    assert (hmm[cpg & keep] != plain[cpg & keep]).all()


def test_no_mask_is_unchanged(monkeypatch):
    seq = 'ATTACGATCAGTTACGTACCATCGATTACA' * 2
    mods = {3, 12}
    seen = _capture(monkeypatch)
    plain = encode_from_query_sequence(seq, mods, 0, mode='daf', strand='+', context_size=3)
    engine._process_single_read(_fiber_read(seq, mods), None, 0, False, 'daf', 3, 1, False)
    assert (seen['encoded'] == plain).all()


# --- the same first-pass mask in every HMM-running tool (D2) -----------------

POLICY = 'unmethylated-only'
EDGE_TRIM = 10


def _ddda_read():
    """A + strand (C->T) DddA read, R/Y-encoded, with a ddda_ucg island over
    SEQ [100, 200): CpGs there stay visible, all others are masked."""
    import pysam

    rng = np.random.default_rng(11)
    bases = rng.choice(list('ACGT'), size=400)
    for start in range(20, 380, 23):           # plenty of CpGs everywhere
        bases[start:start + 2] = ['C', 'G']
    seq = list(''.join(bases))
    cs = [i for i, b in enumerate(seq) if b == 'C']
    for i in cs[::3]:                           # a third of C's deaminated
        seq[i] = 'Y'
    header = pysam.AlignmentHeader.from_dict(
        {'HD': {'VN': '1.6'}, 'SQ': [{'SN': 'chr1', 'LN': 10_000}]})
    read = pysam.AlignedSegment(header)
    read.query_name = 'r1'
    read.flag = 0
    read.reference_id = 0
    read.reference_start = 0
    read.mapping_quality = 60
    read.query_sequence = ''.join(seq)
    read.cigarstring = f'{len(seq)}M'
    read.query_qualities = pysam.qualitystring_to_array('I' * len(seq))
    read.set_tag('MA', f'{len(seq)};ddda_ucg.:101-100', value_type='Z')
    return read


def _model():
    from fiberhmm.core.model_io import freeze_model_for_inference, load_model_with_metadata
    from fiberhmm.models import get_model_path

    model, k, _ = load_model_with_metadata(get_model_path('ddda', tool='apply'))
    return freeze_model_for_inference(model), int(k)


def _recording(monkeypatch, *modules):
    """Record every observation array the HMM decodes (still decoding it)."""
    seen = []
    real = engine.predict_footprints_and_msps

    def wrapper(model, encoded, *args, **kwargs):
        seen.append(np.array(encoded, copy=True))
        return real(model, encoded, *args, **kwargs)

    for module in (engine, *modules):
        monkeypatch.setattr(module, 'predict_footprints_and_msps', wrapper)
    return seen


def _call_first_pass(read, model, k):
    """fiberhmm-call's first pass: payload -> fiber_read + payload_cpg_mask."""
    from fiberhmm.inference.fused_stages import payload_cpg_mask

    payload = engine.make_apply_payload(read, mode='daf')
    fiber_read = engine.extract_fiber_read_from_payload(payload, 'daf', 128)
    fiber_read['_hmm_cpg_mask'] = payload_cpg_mask(
        payload, len(fiber_read['query_sequence']), POLICY)
    return engine._process_single_read(fiber_read, model, EDGE_TRIM, False, 'daf',
                                       k, 0, False)


def test_apply_and_posteriors_mask_the_first_pass_like_call(monkeypatch):
    from fiberhmm.cli.export_posteriors import extract_posteriors_from_read
    from fiberhmm.inference import streaming_workers

    read = _ddda_read()
    model, k = _model()
    seen = _recording(monkeypatch)
    call = _call_first_pass(read, model, k)
    reference = seen[-1]
    assert (reference == 2 * ContextEncoder.get_n_codes(k) + 1).sum() > 0

    # fiberhmm-apply, streaming (payload) workers.
    monkeypatch.setattr(streaming_workers, '_worker_model', model)
    payload = engine.make_apply_payload(read, mode='daf')
    chunk = streaming_workers._process_payload_chunk_worker(
        [payload], EDGE_TRIM, False, 'daf', k, 0, 85, False, False, 128, POLICY)
    assert np.array_equal(seen[-1], reference)
    apply_streaming = chunk.results[0] if hasattr(chunk, 'results') else chunk[0]

    # fiberhmm-apply, chunked / region (pysam read) workers.
    fiber_read = engine._extract_fiber_read_from_pysam(read, 'daf', 128)
    engine.attach_hmm_cpg_mask(fiber_read, POLICY, engine.read_cpg_ma_intervals(read))
    apply_chunked = engine._process_single_read(fiber_read, model, EDGE_TRIM, False,
                                                'daf', k, 0, False)
    assert np.array_equal(seen[-1], reference)

    # fiberhmm-posteriors.
    posteriors = extract_posteriors_from_read(read, model, 'daf', k, EDGE_TRIM,
                                              prob_threshold=128,
                                              cpg_mask_policy=POLICY)
    assert posteriors is not None
    assert np.array_equal(seen[-1], reference)

    # Same segmentation as fiberhmm-call's first pass.
    for result in (apply_streaming, apply_chunked):
        assert list(result['ns']) == list(call['ns'])
        assert list(result['nl']) == list(call['nl'])

    # Without the policy every tool sees the unmasked read.
    extract_posteriors_from_read(read, model, 'daf', k, EDGE_TRIM, prob_threshold=128)
    assert not np.array_equal(seen[-1], reference)


def test_pair_merge_joint_hmm_masks_like_call(monkeypatch):
    """A both-strand consensus whose deam regime is + only is the single read:
    the joint recall's HMM must see exactly fiberhmm-call's first-pass mask,
    while recall still gets the unmasked observation."""
    import pysam

    from fiberhmm.crossstrand import recall as cs_recall
    from fiberhmm.inference import fused_stages

    read = _ddda_read()
    model, k = _model()
    seen = _recording(monkeypatch)
    _call_first_pass(read, model, k)
    reference = seen[-1]

    handed = {}
    real_build = fused_stages.build_fused_recall_result

    def build(fiber_read, apply_result, *args, **kwargs):
        handed['encoded'] = np.array(apply_result['encoded'], copy=True)
        handed['m5c_mask'] = kwargs.get('m5c_mask')
        return real_build(fiber_read, apply_result, *args, **kwargs)

    monkeypatch.setattr(fused_stages, 'build_fused_recall_result', build)
    ctx = cs_recall.RecallContext('ddda')
    assert ctx.cpg_mask_policy == POLICY
    ctx.apply_model = model
    seg = pysam.AlignedSegment(read.header)
    seg.query_name = 'consensus'
    seg.query_sequence = read.query_sequence
    seg.set_tag('MA', f'{read.query_length};deam+:1-{read.query_length}')
    assert cs_recall.recall_consensus_full(
        seg, ctx, edge_trim=EDGE_TRIM, cpg_intervals={'ucg': [(100, 200)], 'mcg': []})
    assert np.array_equal(seen[-1], reference)
    assert not np.array_equal(handed['encoded'], reference)   # recall: unmasked
    assert handed['m5c_mask'] is not None

    # --no-use-m5c: no mask anywhere.
    off = cs_recall.RecallContext('ddda', use_m5c=False)
    off.apply_model = model
    seg.set_tag('MA', f'{read.query_length};deam+:1-{read.query_length}')
    cs_recall.recall_consensus_full(seg, off, edge_trim=EDGE_TRIM)
    assert np.array_equal(seen[-1], handed['encoded'])


def test_both_strand_cpg_targets_cover_both_strands():
    from fiberhmm.crossstrand.recall import both_strand_cpg_targets

    #            0123456789
    conv = 'ACGTTCGAAA'
    plus = np.ones(len(conv), dtype=bool)
    minus = np.ones(len(conv), dtype=bool)
    # C1-G2 and C5-G6 are CpGs: the C's are + targets, the G's - targets.
    mask = both_strand_cpg_targets(conv, set(), set(), plus, minus)
    assert np.flatnonzero(mask).tolist() == [1, 2, 5, 6]
    # A G->A deamination of G2 (R in the consensus) still marks both partners.
    mask = both_strand_cpg_targets('ACATTCGAAA', set(), {2}, plus, minus)
    assert np.flatnonzero(mask).tolist() == [1, 2, 5, 6]
    # Only the strand covering a position contributes it.
    plus_only = both_strand_cpg_targets(conv, set(), set(), plus, ~minus)
    assert np.flatnonzero(plus_only).tolist() == [1, 5]


def test_apply_records_the_first_pass_mask_in_its_pg_record():
    from types import SimpleNamespace

    from fiberhmm.cli.apply import _apply_pg_record

    args = SimpleNamespace(enzyme='ddda', prob_threshold=128, alignments='primary',
                           daf_mask_unaligned=True)
    on = _apply_pg_record(args, 'daf', 3, {}, (2, 'keep-one'), POLICY)['DS']
    off = _apply_pg_record(args, 'daf', 3, {}, (2, 'keep-one'), None)['DS']
    assert f'cpg_mask={POLICY} hmm_cpg_mask=on' in on
    assert 'cpg_mask=off hmm_cpg_mask=off' in off


def test_posteriors_record_the_cpg_policy(tmp_path):
    from fiberhmm.cli.export_posteriors import _export_provenance
    from fiberhmm.models import get_model_path

    model = get_model_path('ddda', tool='apply')
    assert _export_provenance(model, 128, {'cpg_mask_policy': POLICY})['cpg_mask_policy'] == POLICY
    assert _export_provenance(model, 128, {})['cpg_mask_policy'] == 'off'
