"""Evidence-unit identity does not depend on where the BAM lives or what the dataset is called.

unit_id orders the units fed to class discovery and keys its split-half and support folds. It used to
hash the BAM's resolved path and the dataset label, so the same demo BAM opened from three checkouts gave
16, 18 and 20 classes. It now hashes the dataset's ordinal in the run and the file's index in the
dataset's paths, plus the record itself."""
import contextlib
import io
import shutil
from pathlib import Path

import numpy as np
import pysam

from fiberhmm.inference.consensus.bam import load_bam_payload
from fiberhmm.inference.consensus.parameters import parse_options
from fiberhmm.inference.consensus.workflow import run_workflow

PARAMETERS = {'cr': {'engine': 'lattice_recaller'}, 'recaller': {'learned_spots': False}, 'compute': {'cores': 1}}
REGION = dict(chrom='chr1', start=600, end=1000)


def planted_bam(path, n=120, seed=11):
    """DddA BAM, both strands: a 30-bp footprint at chr1:780-810 protects 40% of molecules."""
    from fiberhmm.io.bam_header import append_chemistry
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    reference = ''.join(rng.choice(list('ACGT'), 600))
    header = pysam.AlignmentHeader.from_dict(dict(HD={'VN': '1.6', 'SO': 'coordinate'}, SQ=[{'SN': 'chr1', 'LN': 3000}]))
    header = append_chemistry(header, dict(assay='daf', enzyme='ddda', platform='pacbio', mode='daf'))
    with pysam.AlignmentFile(str(path), 'wb', header=header) as out:
        for i in range(n):
            strand = 'CT' if i % 2 == 0 else 'GA'
            target, mark = ('C', 'Y') if strand == 'CT' else ('G', 'R')
            bound = rng.random() < .4
            sequence = ''.join(mark if base == target and rng.random() < (0. if bound and 280 <= j < 310 else .95) else base
                               for j, base in enumerate(reference))
            r = pysam.AlignedSegment(header)
            r.query_name = f'm{i:04d}'; r.query_sequence = sequence
            r.reference_id = 0; r.reference_start = 500; r.mapping_quality = 60; r.cigarstring = '600M'
            r.set_tag('st', strand); r.set_tag('MA', '600;msp.:1-600' + (';tf.:281-30' if bound else ''))
            out.write(r)
    pysam.index(str(path))
    return path


def copy_bam(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, target); shutil.copy(str(source) + '.bai', str(target) + '.bai')
    return target


def unit_ids(payload):
    return [[u['unit_id'] for u in s['units']] for s in payload['strata']]


def run(payload, out):
    with contextlib.redirect_stdout(io.StringIO()):
        run_workflow(payload, PARAMETERS, out)
    return (out/'classes.tsv').read_text()


def test_same_bam_bytes_at_two_paths_give_identical_units_and_classes(tmp_path):
    first = planted_bam(tmp_path/'a'/'demo.bam')
    second = copy_bam(first, tmp_path/'elsewhere'/'deeper'/'renamed.bam')
    payloads = [load_bam_payload([dict(dataset_id='d', paths=[str(p)])], REGION, parse_options(PARAMETERS)) for p in (first, second)]
    assert unit_ids(payloads[0]) == unit_ids(payloads[1]) and payloads[0]['strata'][0]['units']
    # The real path is kept where outputs need it (BAM export maps members back to their source file).
    for payload, path in zip(payloads, (first, second)):
        member = payload['strata'][0]['units'][0]['source_members'][0]
        assert member['library_id'] == str(path.resolve())
        assert [f['path'] for f in payload['input_files']] == [str(path.resolve())]
    classes = [run(p, tmp_path/f'out{i}') for i, p in enumerate(payloads)]
    assert len(classes[0].splitlines()) > 1          # the planted class is found
    assert classes[0] == classes[1]


def test_dataset_label_does_not_change_units_or_classes(tmp_path):
    path = planted_bam(tmp_path/'a'/'demo.bam')
    payloads = [load_bam_payload([dict(dataset_id=label, paths=[str(path)])], REGION, parse_options(PARAMETERS))
                for label in ('first', 'second')]
    assert unit_ids(payloads[0]) == unit_ids(payloads[1])
    classes = [run(p, tmp_path/f'out{i}') for i, p in enumerate(payloads)]
    assert len(classes[0].splitlines()) > 1
    # Only the label columns (dataset, channel) differ.
    assert classes[0].replace('first', 'second') == classes[1]


def test_datasets_and_files_sharing_a_bam_keep_distinct_units(tmp_path):
    path = planted_bam(tmp_path/'a'/'demo.bam', n=40)
    alone = unit_ids(load_bam_payload([dict(dataset_id='x', paths=[str(path)])], REGION, parse_options(PARAMETERS)))[0]
    # The same BAM as two datasets of one run: the dataset ordinal separates them; the first keeps its units.
    a, b = unit_ids(load_bam_payload([dict(dataset_id='x', paths=[str(path)]), dict(dataset_id='y', paths=[str(path)])],
                                     REGION, parse_options(PARAMETERS)))
    assert a == alone and not set(a) & set(b)
    # Two copies inside one dataset: the file index separates them.
    copy = copy_bam(path, tmp_path/'b'/'demo.bam')
    both, = unit_ids(load_bam_payload([dict(dataset_id='x', paths=[str(path), str(copy)])], REGION, parse_options(PARAMETERS)))
    assert len(set(both)) == len(both) == 2*len(alone) and set(alone) <= set(both)


def test_cli_classes_do_not_depend_on_bam_path(tmp_path):
    from fiberhmm.inference.consensus.cli import main
    first = planted_bam(tmp_path/'a'/'demo.bam')
    second = copy_bam(first, tmp_path/'checkout2'/'data'/'demo.bam')
    tables = []
    for i, path in enumerate((first, second)):
        out = tmp_path/f'cli{i}'
        with contextlib.redirect_stdout(io.StringIO()):
            main(['--bam', str(path), '--region', 'chr1:600-1000', '--cores', '1', '--no-bam', '--output', str(out)])
        found = sorted(out.rglob('classes.tsv'))
        assert len(found) == 1
        tables.append(found[0].read_text())
    assert len(tables[0].splitlines()) > 1 and tables[0] == tables[1]
