"""Hash-prefix sampling before sequence, annotation or native-call decoding.

BAM decompression still scans the region. Only cheap names/flags/MAPQ/spans
are retained. All records of a selected name travel together, preserving the
post-collapse unit-id tie break. DAF channel is not an alignment orientation;
without decoding its tags we advance one common frontier for CT and GA.
"""
from bisect import bisect_left
from collections import Counter
import math

from .. import depth_cap as Cap

OVERSAMPLE = 1.5
FULL_LOAD_FRACTION = .9


def policy(options):
    if options['cr'].engine != 'lattice_recaller':
        return None
    opt = options['recaller']
    caps = (opt.max_discovery_molecules, opt.max_quantification_molecules)
    # An uncapped active phase requires the complete evidence population.
    if not all(caps) or opt.efficiency_calibration:
        return None
    return dict(cap=max(caps), discovery_cap=caps[0], quantification_cap=caps[1], seed=opt.sample_seed, mode='recaller',
                tile_bp=opt.tile_bp, tile_step_bp=opt.tile_step_bp)


def scan(paths, region, minimum_mapq):
    """Counts and cheap candidates; never access auxiliary tags or sequence."""
    import pysam
    from .bam import _resolve_bam_fetch_region
    rows = []; seen = 0
    for path in paths:
        chrom, lo, hi, _ = _resolve_bam_fetch_region(path, region['chrom'], region['start'], region['end'])
        with pysam.AlignmentFile(path, 'rb') as bam:
            for read in bam.fetch(chrom, lo, hi):
                seen += 1
                if read.is_unmapped or read.is_secondary or read.is_supplementary or read.mapping_quality < minimum_mapq:
                    continue
                rows.append((read.query_name, read.reference_start, read.reference_end))
    return seen, rows


class Frontier:
    """One dataset/window's ordered names, with boundary ties retained in full."""
    def __init__(self, rows, window, cap, seed, margin):
        self.window = window
        a, b = window
        self.names = sorted({name for name, lo, hi in rows if lo < b and hi > a},
                            key=lambda name: Cap.rank(name, seed))
        self.cap = cap
        self.step = max(cap, math.ceil(cap * margin))
        self.n = min(len(self.names), self.step)

    def advance(self):
        old = self.n
        self.n = min(len(self.names), self.n + self.step)
        return self.n > old

    def selected(self):
        return self.names[:self.n]


def eligible(units, window, mode):
    a, b = window
    if mode in ('nfr', 'nfr_detection'):
        from ..accessibility.gaps import nucleosomes
        return [u for u in units if (n := nucleosomes(u))
                and n[0][0] < (b if mode == 'nfr_detection' else a)
                and n[-1][1] > (a if mode == 'nfr_detection' else b)]
    from .lattice_recaller.units import MIN_OVERLAP, SITE_PAD
    return [u for u in units if u['reference_start'] < b - MIN_OVERLAP
            and u['reference_end'] > a + MIN_OVERLAP
            and bisect_left(u['positions'], b + SITE_PAD) - bisect_left(u['positions'], a - SITE_PAD) >= 3]


class SampledPayload(dict):
    """Serializable evidence with an in-process top-up hook; no callable enters artifacts."""
    def __init__(self, body, refine):
        super().__init__(body)
        self._refine = refine

    def refine(self, requirements):
        body = self._refine(requirements)
        original = {s['dataset_id']: s for s in self['strata']}
        for source in body['strata']:
            resolution = (original[source['dataset_id']].get('model_manifest') or {}).get('chemistry_resolution')
            if resolution is not None:
                source['model_manifest']['chemistry_resolution'] = resolution
        self.update(body)


def frontier_units(source, frontier):
    units = eligible(source['units'], frontier.window, frontier.mode)
    predicate = getattr(frontier, 'predicate', None)
    return [u for u in units if predicate(source, u)] if predicate else units


def load_sampled(state, request, options, progress, sampling, load):
    cap, seed = sampling['cap'], sampling['seed']
    margin = sampling.get('margin', OVERSAMPLE)
    if not math.isfinite(margin) or margin < 1:
        raise ValueError('Loader oversampling margin must be finite and at least one')
    mode = sampling.get('mode', 'recaller')
    from .lattice_recaller.workflow import _tiles
    windows = sampling.get('windows')
    if not windows:
        windows = (_tiles(request['start'], request['end'],
                   sampling.get('tile_bp', options['recaller'].tile_bp),
                   sampling.get('tile_step_bp', options['recaller'].tile_step_bp))
                   if mode == 'recaller' else [])
    scopes = [(tuple(w), mode) for w in windows]
    if mode == 'nfr':
        scopes.insert(0, ((request['start'], request['end']), 'nfr_detection'))
    scans = {}; frontiers = {}; margins = {}; attempts = Counter(); rounds = 0
    for dsid in request['dataset_ids']:
        ds = state.get_dataset(dsid)
        chemistry = request.get('chemistry_overrides', {}).get(dsid) or getattr(ds, 'family_scan_chemistry', None)
        margins[dsid] = margin * (2 if chemistry in ('ddda', 'dddb') else 1)
        scans[dsid] = scan(ds.paths, request, options['input'].minimum_mapq)
        if any(not name for name, _, _ in scans[dsid][1]):
            progress('loading', 'An unnamed BAM record requires full loading for stable unit identities')
            return load(state, request, options, progress)
        frontiers[dsid] = []
        for w, kind in scopes:
            budget = sampling.get('discovery_cap', cap) if kind in ('recaller', 'nfr_detection') else cap
            f = Frontier(scans[dsid][1], w, budget or cap, seed, margins[dsid]); f.mode = kind
            frontiers[dsid].append(f)
    # Near the budget, caching and extra passes cost more than the few skipped
    # decodes. Use the complete population; the final hash cap is unchanged.
    full = set()
    for ds, fs in frontiers.items():
        projected = {name for f in fs for name in f.names[:math.ceil(cap * margins[ds])]}
        rows = scans[ds][1]
        if rows and sum(n in projected for n, _, _ in rows) >= FULL_LOAD_FRACTION * len(rows):
            full.add(ds)
            for f in fs:
                f.n = len(f.names)
    decoded_names = {ds: set() for ds in frontiers}
    decode_cache = {ds: {} for ds in frontiers if ds not in full}
    replay_cache = {ds: {} for ds in frontiers if ds not in full}
    alignment_cache = {ds: {} for ds in frontiers if ds not in full}
    last_names = None; last_payload = None
    def materialize():
        nonlocal rounds, last_names, last_payload
        while True:
            names = {ds: ({n for n, _, _ in scans[ds][1]} if ds in full else
                          set(n for f in fs for n in f.selected())) for ds, fs in frontiers.items()}
            for ds, selected in names.items():
                attempts[ds] += sum(name in selected and name not in decoded_names[ds] for name, _, _ in scans[ds][1])
                decoded_names[ds].update(selected)
            if names != last_names:
                progress('loading', f'Hash sample: decoding tranche {rounds + 1}')
                last_payload = load(state, dict(request, _read_names=names, _decode_cache=decode_cache,
                                               _replay_cache=replay_cache, _alignment_cache=alignment_cache), options, progress)
                last_names = names
                rounds += 1
            payload = last_payload
            more = False
            for source in payload['strata']:
                ds = source['dataset_id']; hia5 = source['chemistry'].startswith('hia5')
                channels = ['pooled'] if hia5 else sorted({'CT', 'GA'} | {u['strand'] for u in source['units']})
                for f in frontiers[ds]:
                    prefix = set(f.selected())
                    counts = Counter('pooled' if hia5 else u['strand'] for u in frontier_units(source, f)
                                     if u['read_name'] in prefix)
                    if any(counts[ch] < f.cap for ch in getattr(f, 'channels', channels)):
                        more = f.advance() or more
            if mode == 'nfr' and sampling.get('nfr_options') is not None:
                from ..accessibility import gaps as G
                from ..accessibility.workflow import _nfr_regions
                opt = sampling['nfr_options']
                detection = eligible(G.units_of(payload), (request['start'], request['end']), 'nfr_detection')
                detection = Cap.select(detection, opt.max_discovery_molecules, seed, G.sampling_channel, Cap.identity)
                regions, _ = _nfr_regions(detection, request, opt)
                for ds, fs in frontiers.items():
                    known = {f.window for f in fs if f.mode == 'nfr'}
                    for reg in regions:
                        w = (reg['start'], reg['end'])
                        if w not in known:
                            f = Frontier(scans[ds][1], w, cap, seed, margins[ds]); f.mode = 'nfr'
                            if ds in full: f.n = len(f.names)
                            fs.append(f); known.add(w); more = True
            if not more:
                break
        for source in payload['strata']:
            ds = source['dataset_id']; hia5 = source['chemistry'].startswith('hia5')
            channel = lambda u: 'pooled' if hia5 else u['strand']
            counts = Counter(channel(u) for u in source['units'])
            channels = ['pooled'] if hia5 else sorted({'CT', 'GA'} | set(counts))
            seen, rows = scans[ds]
            decoded = Counter()
            for file in source['load'].get('files', []):
                decoded.update(file.get('decoded_channels', {}))
            records = []
            for f in frontiers[ds]:
                usable = frontier_units(source, f)
                used = Cap.select(usable, f.cap, seed, channel, Cap.identity)
                available, retained = Counter(map(channel, usable)), Counter(map(channel, used))
                records.append(dict(window=list(f.window), purpose=getattr(f, 'purpose', f.mode), cap=f.cap, strata=[dict(stratum=f'{ds}::{ch}',
                    usable_after_collapse=available[ch], used=retained[ch], short=available[ch] < f.cap,
                    exhausted=f.n == len(f.names)) for ch in channels]))
            source['load']['loader_sampling'] = dict(dataset_id=ds, cap=cap, seed=seed, margin=margin, passes=rounds,
                strategy='all_candidates_near_budget' if ds in full else 'hash_prefix',
                reads_seen=seen, eligible_reads=len(rows), reads_decoded=sum(n in names[ds] for n, _, _ in rows),
                decode_attempts=attempts[ds], unknown_decoded_channel=0 if hia5 else decoded['unknown'],
                usable_after_collapse=len(source['units']),
                strata=[dict(stratum=f'{ds}::{ch}', reads_seen=seen if hia5 else None,
                    reads_decoded=sum(n in names[ds] for n, _, _ in rows) if hia5 else decoded[ch],
                    usable_after_collapse=counts[ch]) for ch in channels], windows=records,
                channel_counting='pooled' if hia5 else 'chemical channel unknown before tag decoding; shared candidate counts',
                duplicate_scope='sample only; independent read-name hashes, no collapse against unsampled aliases')
        return payload

    def refine(requirements):
        for fs_ds, fs in frontiers.items():
            for window, predicate in requirements:
                f = Frontier(scans[fs_ds][1], tuple(window), sampling.get('quantification_cap', cap), seed, margins[fs_ds])
                f.mode = 'recaller'; f.predicate = predicate; f.purpose = 'quantification'
                if hasattr(predicate, 'loader_channels'):
                    f.channels = [ch.split('::', 1)[1] for ch in predicate.loader_channels if ch.split('::', 1)[0] == fs_ds]
                    if not f.channels:
                        f.names = []; f.n = 0
                if fs_ds in full: f.n = len(f.names)
                fs.append(f)
        return materialize()

    return SampledPayload(materialize(), refine)
