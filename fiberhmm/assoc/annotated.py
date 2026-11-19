"""Association rows from the scan's declared MA groups (no source BAM required)."""
from __future__ import annotations


def _region_index(regions):
    from collections import defaultdict
    from .scanio import _union
    by_chrom = defaultdict(list)
    for c, a, b in regions:
        by_chrom[c].append((a, b))
    return {c: _union(ivs) for c, ivs in by_chrom.items()}


def _overlaps(windows, chrom, start, end):
    from bisect import bisect_left
    intervals = windows.get(chrom, ())
    i = bisect_left(intervals, (end,)) - 1
    return i >= 0 and intervals[i][1] > start


def records(scan, regions=None, *, module=None, kind=None):
    """Yield each annotated record once, including duplicate SAM records."""
    import pysam
    from fiberhmm.io.scan_ma import scan_contracts, scan_groups_of
    windows = _region_index(regions) if regions is not None else None
    for source in scan.manifest.get('outputs', {}).get('annotated', []):
        with pysam.AlignmentFile(str(scan.path / source['path']), 'rb') as bam:
            contracts = scan_contracts(bam.header)
            for declaration in scan.manifest.get('modules', {}).values():
                for expected in declaration.get('ma_groups', []):
                    actual = contracts.get(expected['group'])
                    if actual is None or any(actual.get(k) != v for k, v in expected.items()):
                        from .scanio import ScanInputError
                        raise ScanInputError('annotated BAM group contract is missing or differs from the manifest')
            selected = {g: c for g, c in contracts.items()
                        if (module is None or g.startswith(module)) and (kind is None or c.get('kind') == kind)}
            def alignments():
                if regions is None:
                    yield from bam.fetch(until_eof=True)
                    return
                # Header/coordinate order preserves the whole-file record order.
                for c in bam.references:
                    for a, b in windows.get(c, []):
                        yield from bam.fetch(c, max(0, a), b)
            seen = set() if regions is not None else None
            for read in alignments():
                # File offsets retain identical SAM occurrences while deduplicating
                # records fetched in several disjoint requested windows.
                ordinal = bam.tell()
                if seen is not None:
                    if ordinal in seen:
                        continue
                    seen.add(ordinal)
                if read.is_unmapped:
                    continue
                if windows is not None and not _overlaps(windows, read.reference_name,
                                                        read.reference_start, read.reference_end):
                    continue
                try:
                    groups = scan_groups_of(read, selected)
                except (IndexError, TypeError, ValueError) as error:
                    from .scanio import ScanInputError
                    raise ScanInputError('invalid MA scan annotations in annotated BAM') from error
                yield source, ordinal, read, contracts, groups


def reference_span(read, start, length, ma_length):
    """Project a molecular interval through aligned CIGAR blocks, including hard clips."""
    stored = read.query_length or 0
    cigar = read.cigartuples or []
    hard = sum(n for op, n in cigar if op == 5)
    if ma_length not in (stored, stored + hard):
        raise ValueError('MA length matches neither stored sequence nor whole molecule')
    offset = cigar[0][1] if cigar and cigar[0][0] == 5 and ma_length != stored else 0
    a, b = int(start), int(start + length)
    if read.is_reverse:
        a, b = ma_length - b, ma_length - a
    a, b = a - offset, b - offset
    q, r, spans = 0, read.reference_start, []
    for op, n in cigar:
        if op in (0, 7, 8):
            lo, hi = max(a, q), min(b, q + n)
            if hi > lo:
                spans.append((r + lo - q, r + hi - q))
            q += n
            r += n
        elif op in (1, 4):
            q += n
        elif op in (2, 3):
            r += n
    if spans:
        return spans[0][0], spans[-1][1]
    # Insertion-only marks have a zero-width reference junction.
    q, r = 0, read.reference_start
    for op, n in cigar:
        if op == 1 and q <= a < q + n:
            return r, r
        if op in (0, 1, 4, 7, 8):
            q += n
        if op in (0, 2, 3, 7, 8):
            r += n
    return None


def rows(scan, module, kind, regions=None):
    from fiberhmm.scan.identity import molecule_id
    from fiberhmm.io.scan_ma import ma_read_length
    windows = _region_index(regions) if regions is not None else None
    for source, ordinal, read, contracts, groups in records(scan, regions, module=module, kind=kind):
        common = dict(chrom=read.reference_name, start=str(read.reference_start), end=str(read.reference_end),
                      dataset_id=source['dataset_id'], input_index=str(source['input_index']),
                      qname=read.query_name, record=f'annotated:{ordinal}', molecule=molecule_id([read.query_name]))
        length = ma_read_length(read)
        channel = _channel(scan, source['dataset_id'], read) if kind == 'state' and module == 'fp' else None
        for group, items in groups.items():
            decl = contracts[group]
            if decl.get('kind') != kind or not group.startswith(module):
                continue
            for item in items:
                if len(item['q']) != len(decl.get('quality_names', [])):
                    from .scanio import ScanInputError
                    raise ScanInputError(f'annotated group {group} has the wrong quality length')
                if kind == 'scalar':
                    yield dict(common, **item.get('scalars', {}))
                    continue
                span = reference_span(read, item['start'], item['length'], length)
                if span is None:
                    continue
                a, b = span
                if windows is not None and not _overlaps(windows, read.reference_name, a, max(b, a + 1)):
                    continue
                row = dict(common, start=str(a), end=str(b), group=group,
                           q=','.join(str(q) for q in item['q']), name=item['name'])
                if kind == 'state':
                    if channel:
                        row['channel'] = channel
                    row.update(hit_id=item['name'], state={1: '1', 2: '0'}.get(item['q'][0], '.'),
                               call_start=str(a), call_end=str(b))
                yield row


def _channel(scan, dataset, read):
    from fiberhmm.scan.states import header_chemistry
    chem = next((d.get('chemistry') for d in scan.manifest.get('datasets', [])
                 if d['dataset_id'] == dataset), None) or header_chemistry(read.header)
    if chem and chem.startswith('hia5'):
        return f'{dataset}::pooled'
    if chem in ('ddda', 'dddb'):
        if read.has_tag('cs'):
            return f'{dataset}::BOTH'
        from fiberhmm.inference.tf_recaller import extract_modification_calls
        calls = extract_modification_calls(read, 'daf', 1)
        if calls is not None:
            return f"{dataset}::{'CT' if calls[1] == '+' else 'GA'}"
    return None
