"""The default DAF chimera detector is unchanged from FiberHMM 3.2.1.

``--chimera-detector strict`` is opt-in (3.3.0). Without it, ``fiberhmm-call``
(with its in-process QC) must write what 3.2.1 wrote. The golden file
``data/chimera_default_golden_321.json`` was generated once from the 3.2.1
code (commit 3ef10bd2) on the synthetic library built below::

    PYTHONPATH=<3.2.1 checkout> python tests/test_chimera_default_golden.py --write-golden

Compared: every output record (SHA-256 of its full SAM text: all fields and
tags, in order), the BAM header, the chimeras/dedup/DAF SNP sidecars (the SNP
screen's background site profile as a SHA-256 of its normalised JSON), the
QC JSON and TSV, and the list of QC files written.

Excluded, because they name the code or the run rather than the result:
  * ``VN`` of FiberHMM ``@PG`` lines and ``CL`` of every ``@PG`` line
    (version, command line with temporary paths);
  * the ``fiberhmm_version`` / ``fiberhmm_commit`` tokens of the chemistry
    ``@CO`` line;
  * JSON keys holding versions, commits, timings and the output BAM's
    SHA-256 (it covers the header, hence the version);
  * the temporary directory, replaced by ``<tmp>`` in paths, and random
    temporary file names;
  * floats are compared at 9 significant digits.

All data are synthetic: a random reference and reads whose C->T / G->A
conversions are placed here. The library holds clean reads of both strands,
strong strand swaps, weak swaps (five to nine events per segment: flagged by
the standard detector, not by the strict one), duplicate copies (marked by
dedup) and enough depth for the automatic DAF SNP screen.
"""
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pysam

GOLDEN = Path(__file__).parent / 'data' / 'chimera_default_golden_321.json'
CONTIG = 'chrS'
REF_LEN = 12_000
READ_LEN = 3000

VOLATILE_KEYS = {'fiberhmm_version', 'version', 'elapsed_seconds', 'seconds',
                 'output_sha256', 'fiberhmm_commit', 'commit', 'git_commit',
                 'created', 'timestamp', 'generated_at', 'wall_seconds'}


# --- synthetic library (pysam only; identical under any FiberHMM version) ----

def _reference(seed=23):
    rng = random.Random(seed)
    return ''.join(rng.choice('ACGT') for _ in range(REF_LEN))


def _read(hdr, ref, name, start, strand_at, *, rate=0.35, seed=0, flag=0,
          forced=()):
    """A 3-kb read from ``start``; ``strand_at(ref_pos)`` is 'CT', 'GA' or None.

    Target bases convert with probability ``rate``; ``forced`` reference
    positions always convert (C->T or G->A by their base). MD/NM computed.
    """
    rng = random.Random(seed)
    forced = set(forced)
    seq, md, run, nm = [], [], 0, 0
    for r in range(start, start + READ_LEN):
        base, out = ref[r], ref[r]
        strand = strand_at(r)
        if r in forced:
            out = {'C': 'T', 'G': 'A'}.get(base, base)
        elif rng.random() < rate:
            if strand == 'CT' and base == 'C':
                out = 'T'
            elif strand == 'GA' and base == 'G':
                out = 'A'
        if out != base:
            md += [str(run), base]
            run, nm = 0, nm + 1
        else:
            run += 1
        seq.append(out)
    md.append(str(run))
    read = pysam.AlignedSegment(hdr)
    read.query_name, read.flag, read.reference_id = name, flag, 0
    read.reference_start, read.mapping_quality = start, 60
    read.cigartuples = [(0, READ_LEN)]
    read.query_sequence = ''.join(seq)
    read.query_qualities = pysam.qualitystring_to_array(
        ''.join(chr(33 + 12 + (i * 7) % 28) for i in range(READ_LEN)))
    read.set_tag('NM', nm)
    read.set_tag('MD', ''.join(md))
    return read


def build_library(path):
    ref = _reference()
    hdr = pysam.AlignmentHeader.from_dict({
        'HD': {'VN': '1.6', 'SO': 'coordinate'},
        'SQ': [{'SN': CONTIG, 'LN': REF_LEN}],
        'PG': [{'ID': 'minimap2', 'PN': 'minimap2', 'VN': '2.30'}]})
    reads = []
    for i in range(80):
        strand = 'CT' if i % 2 else 'GA'
        reads.append(_read(hdr, ref, f'clean{i:03d}', 1000 + (i % 9) * 40,
                           lambda r, s=strand: s, seed=i))
    for i in range(4):                       # PCR copies: dedup marks them
        copy = pysam.AlignedSegment.fromstring(reads[i].to_string(), hdr)
        copy.query_name = f'copy{i:03d}'
        reads.append(copy)
    for i in range(10):                      # strong swaps: both detectors
        start = 1200 + (i % 5) * 30
        bp = start + 1500
        reads.append(_read(hdr, ref, f'strong{i:03d}', start,
                           lambda r, bp=bp: 'CT' if r < bp else 'GA', seed=500 + i))
    for i in range(8):                       # weak swaps: standard detector only
        start = 1100 + i * 25
        cs = [p for p in range(start, start + 2600) if ref[p] == 'C'][::35][:20]
        gs = [p for p in range(start + 2700, start + READ_LEN) if ref[p] == 'G']
        gs = gs[::max(1, len(gs) // (5 + i % 5))][:5 + i % 5]
        reads.append(_read(hdr, ref, f'weak{i:03d}', start, lambda r: None,
                           rate=0.0, seed=900 + i, forced=cs + gs))
    reads.sort(key=lambda r: (r.reference_start, r.query_name))
    with pysam.AlignmentFile(str(path), 'wb', header=hdr) as out:
        for read in reads:
            out.write(read)
    pysam.index(str(path))
    return str(path)


# --- normalised outputs ------------------------------------------------------

def _normalise(value, tmp):
    if isinstance(value, dict):
        return {k: _normalise(v, tmp) for k, v in sorted(value.items())
                if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_normalise(v, tmp) for v in value]
    if isinstance(value, float):
        return float(f'{value:.9g}')
    if isinstance(value, str):
        for prefix in sorted({str(Path(tmp).resolve()), str(tmp)}, key=len, reverse=True):
            value = value.replace(prefix, '<tmp>')
        value = re.sub(r'\.fiberhmm_[a-z]+_[A-Za-z0-9_]+', '.fiberhmm_<random>', value)
        return value
    return value


def _header(path, tmp):
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        header = bam.header.to_dict()
    for pg in header.get('PG', []):
        pg.pop('CL', None)
        if str(pg.get('PN', '')).startswith('fiberhmm'):
            pg.pop('VN', None)
    header['CO'] = [re.sub(r'fiberhmm_(commit|version)=[^;]*', r'fiberhmm_\1=<code>', co)
                    for co in header.get('CO', [])]
    return _normalise(header, tmp)


def _records(path):
    with pysam.AlignmentFile(str(path), check_sq=False) as bam:
        lines = [read.to_string() for read in bam.fetch(until_eof=True)]
    marked = sorted({line.split('\t')[0] for line in lines if '\txc:Z:strand_swap' in line})
    duplicates = sorted({line.split('\t')[0] for line in lines
                         if int(line.split('\t')[1]) & 0x400})
    return {'n': len(lines), 'marked_chimeras': marked, 'duplicates': duplicates,
            'sha256': [hashlib.sha256(line.encode()).hexdigest() for line in lines]}


def default_outputs(tmp):
    """Run the default fiberhmm-call (with its QC) and collect normalised outputs."""
    tmp = Path(tmp)
    source = build_library(tmp / 'lib.bam')
    target = tmp / 'calls.bam'
    command = [sys.executable, '-m', 'fiberhmm.cli.call', '-i', source, '-o', str(target),
               '--enzyme', 'dddb', '--seq', 'nanopore', '-c', '1', '--io-threads', '1',
               '--qc-sample-reads', '500']
    run = subprocess.run(command, capture_output=True, text=True, timeout=900, cwd=tmp)
    assert run.returncode == 0, run.stderr
    qc = tmp / 'qc'
    out = {'header': _header(target, tmp), 'records': _records(target)}
    for name in ('calls.chimeras.json', 'calls.dedup.json', 'calls.daf_snps.json',
                 'calls.qc.json'):
        out[name] = _normalise(json.loads((qc / name).read_text()), tmp)
    # The SNP screen's background site profile is large: kept as a digest.
    profile = out['calls.daf_snps.json'].get('site_distribution')
    if profile is not None:
        out['calls.daf_snps.json']['site_distribution'] = {'sha256': hashlib.sha256(
            json.dumps(profile, sort_keys=True).encode()).hexdigest()}
    out['calls.qc.tsv'] = _normalise((qc / 'calls.qc.tsv').read_text().splitlines(), tmp)
    out['qc_files'] = sorted(p.name for p in qc.iterdir())
    return out


def test_default_call_and_qc_match_321_golden(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")  # the golden QC includes the PDF/PNG report
    golden = json.loads(GOLDEN.read_text())
    got = default_outputs(tmp_path)
    # The fixture exercises what the strict detector would change.
    assert golden['records']['marked_chimeras'] and golden['records']['duplicates']
    assert any(name.startswith('weak') for name in golden['records']['marked_chimeras'])
    assert golden['calls.daf_snps.json']['method'] == 'bidirectional_recurrent_daf_snp'
    ds = next(pg['DS'] for pg in got['header']['PG'] if pg['PN'] == 'fiberhmm-call')
    assert 'chimera_detector' not in ds and 'chimera_min_baseq' not in ds
    for key in golden:
        assert got[key] == golden[key], key


if __name__ == '__main__':
    import tempfile
    if sys.argv[1:] != ['--write-golden']:
        sys.exit(__doc__)
    with tempfile.TemporaryDirectory() as tmp:
        outputs = default_outputs(tmp)
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(outputs, indent=1, sort_keys=True) + '\n')
    print(f'wrote {GOLDEN} ({GOLDEN.stat().st_size:,} bytes)')
