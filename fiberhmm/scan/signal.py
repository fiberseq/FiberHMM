"""Explicit signal-BAM inputs: retain producer-scored records before every loader.

Only schema 5 whole-alignment signal annotations are accepted. No signal is
inferred or re-scored here. The four predicates match the browser input gate.
"""
from __future__ import annotations

import json

PREFIX = 'FIBERHMM-SIGNAL:v1:'
INPUT_PREFIX = 'FIBERHMM-SCAN-SIGNAL:v1:'
FILTERS = ('hideBackground', 'excludeUnscored', 'requireDetectable', 'requireStructure')
CODES = {1: 'background', 0: 'detectable', -1: 'unclassified', -2: 'unscored'}


def validate_spec(spec):
    kind = spec.get('input_type', 'called-bam')
    if kind not in ('called-bam', 'signal-bam'):
        raise ValueError('input_type must be called-bam or signal-bam')
    filters = spec.get('signal_filters', {})
    if not isinstance(filters, dict) or set(filters) - set(FILTERS) or any(type(v) is not bool for v in filters.values()):
        raise ValueError('signal_filters must contain boolean browser signal predicates')
    if filters and kind != 'signal-bam':
        raise ValueError('signal_filters requires input_type=signal-bam')
    return {k: filters.get(k, False) for k in FILTERS}


def provenance(header):
    for comment in reversed(header.to_dict().get('CO', [])):
        if comment.startswith(PREFIX):
            try:
                value = json.loads(comment[len(PREFIX):])
            except ValueError:
                break
            if isinstance(value, dict) and value.get('tag_schema') == 5:
                codes = value.get('codes')
                if codes is None or codes == {v: k for k, v in CODES.items()}:
                    return value
            break
    raise ValueError('signal-bam requires FIBERHMM-SIGNAL:v1 provenance with tag_schema=5')


def keeps(read, filters):
    """Unknown/malformed annotations obey the browser's unknown/unscored gate."""
    status, structure = 'unknown', None
    try:
        detail = json.loads(read.get_tag('me')) if read.has_tag('me') else {}
        code = read.get_tag('bg') if read.has_tag('bg') else None
        schema = detail.get('schema') if read.has_tag('me') and isinstance(detail, dict) else 5
        if isinstance(detail, dict) and type(schema) is int and schema == 5:
            status = CODES.get(code, detail.get('signal_status', 'unknown')) if code is None or type(code) is int else 'unknown'
            if status not in CODES.values() or (code is not None and code not in CODES):
                status = 'unknown'
            if code is not None and detail.get('signal_status', status) != status:
                status = 'unknown'
            if read.has_tag('db') and (type(read.get_tag('db')) is not int or CODES.get(read.get_tag('db')) != status):
                status = 'unknown'
            st = detail.get('structure', {})
            score = st.get('score') if isinstance(st, dict) else None
            import math
            if type(score) in (int, float) and math.isfinite(score):
                structure = st.get('status')
    except (ValueError, TypeError, KeyError, OverflowError):
        pass
    return not ((filters['hideBackground'] and status == 'background')
                or (filters['excludeUnscored'] and status in ('unknown', 'unscored'))
                or (filters['requireDetectable'] and status != 'detectable')
                or (filters['requireStructure'] and structure != 'supported'))


def materialize(datasets, folder, cancel=None):
    import pysam
    from .evidence import materialize_evidence
    from .identity import content_id
    specs = []
    labels = {str(d['dataset_id']) for d in datasets if d.get('dataset_id')}
    for i, spec in enumerate(datasets):
        filters = validate_spec(spec)
        paths = spec.get('paths') or ([spec['path']] if spec.get('path') else [])
        if not paths:
            raise ValueError('signal input requires BAM paths')
        paths = [paths] if isinstance(paths, str) else paths
        headers = []
        for path in paths:
            with pysam.AlignmentFile(path, 'rb') as bam:
                h = bam.header.to_dict()
                if spec.get('input_type') == 'signal-bam':
                    provenance(bam.header)
                    h.setdefault('CO', []).append(INPUT_PREFIX + json.dumps(
                        dict(input_type='signal-bam', signal_filters=filters, source_content_id=content_id(path)),
                        sort_keys=True, separators=(',', ':')))
                headers.append(h)
        label = str(spec['dataset_id']) if spec.get('dataset_id') else f'signal_input_{i + 1}'
        if not spec.get('dataset_id'):
            while label in labels:
                label += '_'
        labels.add(label)
        specs.append(dict(spec, dataset_id=label, headers=headers, paths=paths))
    def source():
        for spec in specs:
            filters = validate_spec(spec)
            for i, path in enumerate(spec['paths']):
                with pysam.AlignmentFile(path, 'rb') as bam:
                    for read in bam.fetch(until_eof=True):
                        if cancel is not None and cancel():
                            from .runner import ScanRefused
                            raise ScanRefused('scan cancelled while filtering signal evidence')
                        if spec.get('input_type') != 'signal-bam' or keeps(read, filters):
                            yield spec['dataset_id'], i, read
    # Filtering changes evidence identity. Caller-supplied unfiltered IDs cannot identify it.
    clean = [{k: v for k, v in s.items() if k != 'content_ids'} for s in specs]
    result = materialize_evidence(clean, source, folder, cancel=cancel)
    for before, after in zip(datasets, result):
        after['dataset_id'] = before.get('dataset_id')
    return result


def input_metadata(path):
    import pysam
    with pysam.AlignmentFile(path, 'rb') as bam:
        return [json.loads(c[len(INPUT_PREFIX):]) for c in bam.header.to_dict().get('CO', [])
                if c.startswith(INPUT_PREFIX)]
