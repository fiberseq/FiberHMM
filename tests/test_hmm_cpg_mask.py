"""First-pass HMM CpG masking for DddA CpG-aware calling."""
import array

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


def test_qc_light_call_masks_the_first_pass_like_call(monkeypatch):
    """fiberhmm-qc's light call on an uncalled DddA BAM segments with the same
    first-pass CpG mask as fiberhmm-call (and can turn it off the same way)."""
    from fiberhmm.core.bam_reader import daf_run_mask_scope, default_daf_run_mask
    from fiberhmm.qc.states import compute_state_rates

    read = _ddda_read()
    read.set_tag('MA', None)                    # uncalled: no footprints
    read.set_tag('MA', f'{read.query_length};ddda_ucg.:101-100', value_type='Z')
    model, k = _model()
    seen = _recording(monkeypatch)
    with daf_run_mask_scope(*default_daf_run_mask('ddda')):
        _call_first_pass(read, model, k)
        unmasked = engine._extract_fiber_read_from_pysam(read, 'daf', 128)
        engine._process_single_read(unmasked, model, EDGE_TRIM, False, 'daf', k, 0, False)
    reference, plain = seen[-2], seen[-1]
    assert not np.array_equal(reference, plain)

    block, _ = compute_state_rates([read], 'daf', 'ddda', state_source='light-call',
                                   edge_trim=EDGE_TRIM, prob_threshold=128)
    assert block['source'] == 'light_call'
    assert block['definition']['hmm_cpg_mask'] == POLICY
    assert np.array_equal(seen[-1], reference)

    block, _ = compute_state_rates([read], 'daf', 'ddda', state_source='light-call',
                                   edge_trim=EDGE_TRIM, prob_threshold=128,
                                   use_m5c=False)
    assert block['definition']['hmm_cpg_mask'] == 'off'
    assert np.array_equal(seen[-1], plain)


def test_posteriors_api_takes_the_cli_cpg_options(monkeypatch, tmp_path):
    from fiberhmm.cli import export_posteriors as ep
    from fiberhmm.models import get_model_path

    captured = {}
    monkeypatch.setattr(ep, 'export_posteriors',
                        lambda **kwargs: captured.update(kwargs) or 0)
    model = get_model_path('ddda', tool='apply')
    ep.export_posteriors_from_bam('in.bam', model, str(tmp_path / 'p.tsv.gz'),
                                  enzyme='ddda')
    assert captured['extraction']['cpg_mask_policy'] == POLICY
    assert captured['extraction']['daf_run_mask'] == (2, 'keep-one')
    assert captured['extraction']['min_mapq'] == 0
    ep.export_posteriors_from_bam('in.bam', model, str(tmp_path / 'p.tsv.gz'),
                                  enzyme='ddda', daf_mask_runs=0, min_mapq=30)
    assert captured['extraction']['daf_run_mask'][0] == 0
    assert captured['extraction']['min_mapq'] == 30
    ep.export_posteriors_from_bam('in.bam', model, str(tmp_path / 'p.tsv.gz'),
                                  enzyme='ddda', cpg_mask_policy='methylated-only')
    assert captured['extraction']['cpg_mask_policy'] == 'methylated-only'
    ep.export_posteriors_from_bam('in.bam', model, str(tmp_path / 'p.tsv.gz'),
                                  enzyme='ddda', use_m5c=False)
    assert captured['extraction']['cpg_mask_policy'] is None
    # As on the command line, a custom model without --enzyme is unmasked
    # unless asked, and a non-DAF model refuses the request.
    ep.export_posteriors_from_bam('in.bam', model, str(tmp_path / 'p.tsv.gz'))
    assert captured['extraction']['cpg_mask_policy'] is None
    import pytest
    with pytest.raises(ValueError):
        ep.export_posteriors_from_bam('in.bam', get_model_path('hia5', tool='apply', seq='pacbio'),
                                      str(tmp_path / 'p.tsv.gz'), use_m5c=True)


def _tagged_ddda_read():
    """_ddda_read after a call and tag-m5c: footprints plus a ddda_ucg island."""
    read = _ddda_read()
    n = read.query_length
    read.set_tag('MA', f'{n};nuc.Q:1-150;msp.:151-100;ddda_ucg.:101-100', value_type='Z')
    read.set_tag('AQ', array.array('B', [200]))
    return read


def test_apply_keeps_cpg_island_groups_and_rewrites_footprints():
    from fiberhmm.inference.streaming_drain import strip_for_apply
    from fiberhmm.inference.tagging import set_legacy_apply_tags
    from fiberhmm.io.ma_tags import parse_ma_tag

    read = _tagged_ddda_read()
    strip_for_apply(read)
    assert read.get_tag('MA') == f'{read.query_length};ddda_ucg.:101-100'
    assert not read.has_tag('AQ')
    # 180-219 is a TF-sized footprint inside the MSP: legacy ns keeps it, MA
    # (where nuc and msp tile the read) does not call it a nucleosome.
    result = {'ns': np.array([10, 180, 260]), 'nl': np.array([140, 40, 120]),
              'as': np.array([150]), 'al': np.array([110])}
    set_legacy_apply_tags(read, result, with_scores=False)
    parsed = parse_ma_tag(read.get_tag('MA'))
    assert [tuple(map(int, iv)) for iv in parsed['ddda_ucg']] == [(100, 100)]
    assert sorted(map(tuple, parsed['nuc'])) == [(10, 140), (260, 120)]
    assert list(map(tuple, parsed['msp'])) == [(150, 110)]
    assert list(read.get_tag('ns')) == [10, 180, 260]
    # The kept island still exempts its CpGs in a later recall.
    assert engine.read_cpg_ma_intervals(read)['ucg'] == [(100, 200)]

    # Reads without CpG-island groups keep the legacy-only apply output.
    plain = _ddda_read()
    plain.set_tag('MA', f'{plain.query_length};nuc.Q:1-150', value_type='Z')
    plain.set_tag('AQ', array.array('B', [200]))
    strip_for_apply(plain)
    set_legacy_apply_tags(plain, result, with_scores=False)
    assert not plain.has_tag('MA') and not plain.has_tag('AQ')


def test_region_apply_worker_reads_the_island_mask_before_stripping(monkeypatch, tmp_path):
    """Region-parallel apply sees the same first-pass mask as fiberhmm-call on
    a tag-m5c BAM (the island groups used to be stripped first)."""
    import pysam

    from fiberhmm.inference import region_workers
    from fiberhmm.inference.region_types import RegionBamWorkItem
    from fiberhmm.models import get_model_path

    read = _tagged_ddda_read()
    model, k = _model()
    seen = _recording(monkeypatch)
    _call_first_pass(read, model, k)
    reference = seen[-1]

    bam = tmp_path / 'in.bam'
    with pysam.AlignmentFile(str(bam), 'wb', header=read.header) as out:
        out.write(read)
    pysam.index(str(bam))
    params = {'edge_trim': EDGE_TRIM, 'circular': False, 'mode': 'daf',
              'context_size': k, 'msp_min_size': 0, 'nuc_min_size': 85,
              'min_mapq': 0, 'prob_threshold': 128, 'min_read_length': 0,
              'with_scores': False, 'train_rids': set(), 'primary_only': False,
              'return_posteriors': False, 'write_msps': True, 'io_threads': 1,
              'cpg_mask_policy': POLICY}
    monkeypatch.setattr(region_workers, '_worker_model', model)
    monkeypatch.setattr(region_workers, '_worker_region_params', params)
    item = RegionBamWorkItem(('chr1', 0, 10_000), str(bam), str(tmp_path / 'out.bam'))
    result = region_workers._process_region_to_bam(item)
    assert result.total_reads == 1
    assert np.array_equal(seen[-1], reference)
    with pysam.AlignmentFile(str(tmp_path / 'out.bam'), check_sq=False) as out:
        written = next(out)
    assert 'ddda_ucg.:101-100' in written.get_tag('MA')


def test_posteriors_api_matches_the_cli(monkeypatch, tmp_path):
    """export_posteriors_from_bam(enzyme='ddda') resolves the CpG mask and
    the DddA run thinning as fiberhmm-posteriors --enzyme ddda does, and the
    two write the same posteriors."""
    import gzip
    import sys

    import pysam

    from fiberhmm.cli import export_posteriors as ep
    from fiberhmm.core.bam_reader import configure_daf_run_mask

    read = _tagged_ddda_read()
    bam = tmp_path / 'in.bam'
    with pysam.AlignmentFile(str(bam), 'wb', header=read.header) as out:
        out.write(read)
    pysam.index(str(bam))

    def body(path):
        with gzip.open(path, 'rt') as handle:
            return [line for line in handle if not line.startswith('#')]

    configure_daf_run_mask(0, 'keep-one')   # a stale process setting must not leak
    cli = tmp_path / 'cli.tsv.gz'
    monkeypatch.setattr(sys, 'argv', ['fiberhmm-posteriors', '-i', str(bam), '-o', str(cli),
                                      '--enzyme', 'ddda', '--seq', 'pacbio', '-c', '1',
                                      '--streaming'])
    ep.main()
    configure_daf_run_mask(0, 'keep-one')
    api = tmp_path / 'api.tsv.gz'
    model = __import__('fiberhmm.models', fromlist=['x']).get_model_path('ddda', tool='apply')
    ep.export_posteriors_from_bam(str(bam), model, str(api), mode='daf', enzyme='ddda',
                                  n_workers=1, verbose=False)
    assert body(api) == body(cli) and body(api)
    off = tmp_path / 'off.tsv.gz'
    ep.export_posteriors_from_bam(str(bam), model, str(off), mode='daf', enzyme='ddda',
                                  n_workers=1, verbose=False, daf_mask_runs=0)
    assert body(off) != body(api)
    configure_daf_run_mask(0, 'keep-one')


def test_posteriors_api_honours_min_mapq(tmp_path):
    from fiberhmm.cli.export_posteriors import extract_posteriors_from_read

    read = _ddda_read()
    read.mapping_quality = 0
    model, k = _model()
    stats = {}
    assert extract_posteriors_from_read(read, model, 'daf', k, EDGE_TRIM, prob_threshold=128,
                                        stats=stats, min_mapq=20) is None
    assert stats['filtered'] == 1
    assert extract_posteriors_from_read(read, model, 'daf', k, EDGE_TRIM,
                                        prob_threshold=128) is not None
