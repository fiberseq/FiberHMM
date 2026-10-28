"""Snapshot an iterable record source for the ordinary scan loaders.

Materializing once lets every module, calibration and worker consume precisely
one snapshot, including records outside the plan used by data calibration.
"""
from contextlib import ExitStack
from pathlib import Path


def materialize_evidence(datasets, evidence, folder, *, cancel=None):
    """Records are AlignedSegment (one input) or (dataset_id, input_index, record).

    A callback is called once without arguments and returns that iterable.
    Each dataset supplies headers (one per input) or header (one input). Existing
    paths can supply headers instead. Optional content_ids identify the snapshot
    in the caller's source registry; otherwise the materialized BAM supplies them.
    """
    from tempfile import TemporaryDirectory
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='snapshot_', dir=folder) as snapshot:
        return _snapshot(datasets, evidence, folder, Path(snapshot), cancel)


def _snapshot(datasets, evidence, folder, snapshot, cancel):
    import pysam
    from .runner import ScanRefused
    from .identity import content_id
    result, writers, unsorted = [], {}, {}
    validated_headers = {}
    expected_references = {}
    with ExitStack() as stack:
        for d, spec in enumerate(datasets):
            label = str(spec.get('dataset_id') or f'dataset_{d + 1}')
            if any(key[0] == label for key in writers):
                raise ScanRefused('evidence dataset ids must be unique')
            paths = spec.get('paths') or ([spec['path']] if spec.get('path') else [])
            if isinstance(paths, str):
                paths = [paths]
            headers = spec.get('headers')
            if headers is None and spec.get('header') is not None:
                headers = [spec['header']]
            if headers is None:
                headers = []
                for path in paths:
                    with pysam.AlignmentFile(path, 'rb') as bam:
                        headers.append(bam.header.to_dict())
            if paths and len(paths) != len(headers):
                raise ScanRefused('evidence requires one header per input path')
            if not headers:
                raise ScanRefused('record evidence requires a header per input')
            new_paths = []
            for i, header in enumerate(headers):
                sub = snapshot / f'{d:03d}_{i}'
                sub.mkdir(exist_ok=True)
                name = Path(paths[i]).name if i < len(paths) else f'input_{i}.bam'
                path = sub / name
                raw = sub / 'snapshot.unsorted.bam'
                writers[(label, i)] = stack.enter_context(pysam.AlignmentFile(str(raw), 'wb', header=header))
                hdr = writers[(label, i)].header
                expected_references[(label, i)] = (hdr.references, hdr.lengths)
                unsorted[(label, i)] = (raw, path)
                new_paths.append(str(path))
            result.append(dict(spec, dataset_id=spec.get('dataset_id'), paths=new_paths))
        source = evidence() if callable(evidence) else evidence
        for item in source:
            if cancel is not None and cancel():
                raise ScanRefused('scan cancelled while reading evidence')
            if isinstance(item, pysam.AlignedSegment):
                if len(writers) != 1:
                    raise ScanRefused('multi-input evidence requires (dataset_id, input_index, record)')
                key = next(iter(writers))
                record = item
            else:
                label, i, record = item
                key = (str(label), int(i))
            if key not in writers:
                raise ScanRefused('evidence names an unknown dataset or input_index')
            # Reference ids are header-relative. Reject a mismatched target frame.
            if not isinstance(record, pysam.AlignedSegment):
                raise ScanRefused('evidence must supply AlignedSegment records')
            hdr = record.header
            header_key = (key, id(hdr))
            if header_key not in validated_headers:
                if hdr is None or (hdr.references, hdr.lengths) != expected_references[key]:
                    raise ScanRefused('evidence record reference differs from the supplied header')
                validated_headers[header_key] = hdr  # retain identity; avoid header work per record
            writers[key].write(record)
    for raw, path in unsorted.values():
        pysam.sort('--no-PG', '-@', '1', '-o', str(path), str(raw))
        raw.unlink()
        pysam.index('-c', str(path))
    for spec in result:
        persisted = []
        for name in spec['paths']:
            path = Path(name)
            destination = folder / content_id(path).replace(':', '_') / path.name
            destination.parent.mkdir(exist_ok=True)
            if not destination.exists() or not Path(str(destination) + '.csi').exists():
                path.replace(destination)
                Path(str(path) + '.csi').replace(str(destination) + '.csi')
            persisted.append(str(destination))
        spec['paths'] = persisted
    return result
