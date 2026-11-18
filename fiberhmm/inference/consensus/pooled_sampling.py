"""Molecule caps that limit what a pooled (--pool-loci) run LOADS.

Without this, each BED window is loaded with the run's caps applied per window,
so a pooled run over hundreds of windows decodes every molecule of every window
(e.g. 563 windows x 300 reads = 170k molecules) and only then keeps
``max_discovery_molecules`` of them per pooled tile.

Discovery and quantification keep, per pooled tile (or scoring window) and
channel, the eligible molecules with the lowest read-name hash
(``depth_cap.rank``). That rank depends only on the read name and the seed, so
one global hash threshold serves every window: decode, in every window, only
the names ranked below it. The threshold starts at ``cap x margin`` candidate
names per pooled tile (counted from a cheap name/span scan, as the per-window
sampler does) and is raised, and the windows that gain names are decoded
again, whenever a tile, or after discovery a scoring window, has fewer than
``cap`` eligible loaded molecules in a channel while unloaded names remain.

Exactness: when a tile has at least ``cap`` eligible loaded molecules in a
channel, every eligible molecule ranked below the cap-th one is loaded (all
names under the threshold are), so the selection equals the full-load one; a
name is loaded in every window it overlaps, so the one-view-per-molecule choice
of ``pool_payloads`` is unchanged. As with the per-window sampler, DAF
duplicate collapse sees only decoded reads.
"""
from __future__ import annotations

import math
from collections import Counter

from .. import depth_cap as Cap


def enabled(options):
    from .loader_sampling import policy
    return policy(options) is not None


def _genomic(window, a, b):
    """Pooled (oriented) [a, b) -> genomic [lo, hi) for one BED window."""
    if window['strand'] == '-':
        return window['end'] - b, window['end'] - a
    return window['start'] + a, window['start'] + b


class PooledLoader:
    def __init__(self, datasets, windows, options, progress, load_window):
        from .loader_sampling import OVERSAMPLE, policy, scan
        from .lattice_recaller.workflow import _tiles
        from .lattice_recaller.units import MIN_OVERLAP
        self.datasets, self.windows, self.options, self.progress = datasets, windows, options, progress
        self.load_window = load_window
        sampling = policy(options); opt = options['recaller']
        self.discovery_cap, self.quantification_cap, self.seed = sampling['discovery_cap'], sampling['quantification_cap'], sampling['seed']
        width = windows[0]['end'] - windows[0]['start']
        self.tiles = _tiles(0, width, opt.tile_bp, opt.tile_step_bp)
        self.min_overlap = MIN_OVERLAP
        # Cheap scan (names, flags, MAPQ, spans; no tags or sequence) of every window and dataset.
        self.rows = {}           # (window index, dataset) -> [(name, start, end)]
        names = {}
        for i, w in enumerate(windows):
            if i % 25 == 0:
                progress('loading', f'Pooled hash sample: scanning windows {i}/{len(windows)}')
            for item in datasets:
                ds = item['dataset_id']
                from .bam import _flatten_paths
                paths = _flatten_paths(item['paths'])
                _, rows = scan(paths, dict(chrom=w['chrom'], start=w['start'], end=w['end']), options['input'].minimum_mapq)
                rows = [r for r in rows if r[1] < w['end'] and r[2] > w['start']]
                if any(not n for n, _, _ in rows):
                    raise _Unsupported('An unnamed BAM record requires full loading for stable unit identities')
                self.rows[i, ds] = rows
                names.setdefault(ds, set()).update(n for n, _, _ in rows)
        self.ranked = {ds: sorted(v, key=lambda n: Cap.rank(n, self.seed)) for ds, v in names.items()}
        self.position = {ds: {n: k for k, n in enumerate(v)} for ds, v in self.ranked.items()}
        from .bam import _dataset_chemistry, _flatten_paths
        chem = {item['dataset_id']: _dataset_chemistry(item, _flatten_paths(item['paths']))[0] for item in datasets}
        self.margin = {ds: OVERSAMPLE * (2 if chem.get(ds) in ('ddda', 'dddb') else 1) for ds in self.ranked}
        self.prefix = {ds: 0 for ds in self.ranked}
        self.loaded = {}          # window index -> (frozen name set per dataset, payload)
        self.passes = 0
        for ds in self.ranked:
            for a, b in self.tiles:
                self._cover(ds, (a, b), math.ceil(self.discovery_cap * self.margin[ds]))

    # ------------------------------------------------------------------ threshold
    def _covering(self, ds, span):
        """Rank positions of names whose scanned span overlaps pooled span in some window (sorted)."""
        a, b = span; out = set(); pos = self.position[ds]
        for i, w in enumerate(self.windows):
            lo, hi = _genomic(w, a, b)
            for name, s, e in self.rows.get((i, ds), ()):
                if s < hi - self.min_overlap and e > lo + self.min_overlap:
                    out.add(pos[name])
        return sorted(out)

    def _cover(self, ds, span, count):
        """Raise ds's threshold until ``count`` names covering span are under it (or all are)."""
        covering = self._covering(ds, span)
        if not covering:
            return False
        need = covering[min(count, len(covering)) - 1] + 1
        if need > self.prefix[ds]:
            self.prefix[ds] = need
            return True
        return False

    def planned_molecules(self):
        """Window views the first pass will decode (names under the threshold, per window they overlap)."""
        selected = {ds: set(v[:self.prefix[ds]]) for ds, v in self.ranked.items()}
        return sum(1 for (i, ds), rows in self.rows.items() for n, _, _ in rows if n in selected[ds])

    def exhausted(self, ds):
        return self.prefix[ds] >= len(self.ranked[ds])

    # ------------------------------------------------------------------ loading
    def materialize(self):
        from .regions import pool_payloads
        selected = {ds: set(v[:self.prefix[ds]]) for ds, v in self.ranked.items()}
        payloads = []
        for i, w in enumerate(self.windows):
            want = {ds: frozenset(n for n, _, _ in self.rows.get((i, ds), ()) if n in selected[ds]) for ds in self.ranked}
            cached = self.loaded.get(i)
            if cached is None or cached[0] != want:
                if i % 25 == 0:
                    self.progress('loading', f'Pooled hash sample (pass {self.passes + 1}): decoding window {i + 1}/{len(self.windows)}')
                self.loaded[i] = (want, self.load_window(i, w, {ds: set(v) for ds, v in want.items()}))
            payloads.append(self.loaded[i][1])
        self.passes += 1
        pooled = pool_payloads(payloads, self.windows)
        self._receipt(pooled)
        return pooled

    def _receipt(self, pooled):
        record = dict(strategy='pooled_hash_threshold', seed=self.seed, discovery_cap=self.discovery_cap,
                      quantification_cap=self.quantification_cap, passes=self.passes, datasets=[
                          dict(dataset_id=ds, margin=self.margin[ds], candidate_names=len(v), names_decoded=self.prefix[ds],
                               exhausted=self.exhausted(ds)) for ds, v in self.ranked.items()])
        pooled['pooling']['pooled_loader_sampling'] = record
        for source in pooled['strata']:
            ds = source['dataset_id']
            # run_lattice_recaller marks all_units False when a loader decoded fewer reads than were eligible.
            source['load']['loader_sampling'] = dict(dataset_id=ds, strategy='pooled_hash_threshold',
                eligible_reads=len(self.ranked.get(ds, ())), reads_decoded=self.prefix.get(ds, 0), seed=self.seed)

    def _short(self, pooled, span_of, predicate, cap, channels=None):
        """Datasets with a pooled span/channel holding fewer than cap eligible loaded molecules and names left."""
        from .lattice_recaller.units import channel_of
        grow = set()
        for source in pooled['strata']:
            ds = source['dataset_id']
            if self.exhausted(ds):
                continue
            counts = Counter(channel_of(source, u) for u in source['units'] if predicate(source, u))
            wanted = channels if channels is not None else (
                {f'{ds}::pooled'} if source['chemistry'].startswith('hia5') else {f'{ds}::CT', f'{ds}::GA'})
            for ch in wanted:
                if ch.split('::', 1)[0] == ds and counts[ch] < cap:
                    grow.add(ds)
        return grow

    def load(self):
        from .lattice_recaller.units import eligible_unit
        while True:
            pooled = self.materialize()
            grew = False
            for a, b in self.tiles:
                for ds in self._short(pooled, (a, b), lambda s, u, a=a, b=b: eligible_unit(u, a, b), self.discovery_cap):
                    grew |= self._extend(ds, (a, b), self.discovery_cap)
            if not grew:
                return pooled

    def _extend(self, ds, span, cap):
        """Double the covering-name target of span (at least cap more names)."""
        covering = self._covering(ds, span)
        under = sum(1 for k in covering if k < self.prefix[ds])
        return self._cover(ds, span, max(under + cap, 2 * under))

    def refine(self, pooled, requirements):
        """Quantification top-up: requirements = [((a, b), accepts(source, unit))] in pooled coordinates."""
        changed = False
        while True:
            grew = False
            for (a, b), accepts in requirements:
                channels = getattr(accepts, 'loader_channels', None)
                for ds in self._short(pooled, (a, b), accepts, self.quantification_cap, channels):
                    grew |= self._extend(ds, (a, b), self.quantification_cap)
            if not grew:
                return pooled, changed
            pooled = self.materialize(); changed = True


class _Unsupported(Exception):
    pass


def planned_loader(datasets, windows, options, progress, load_window):
    """A scanned PooledLoader (nothing decoded yet), or None when caps are off/unsupported.

    load_window(i, window, read_names) -> one window's payload decoding only read_names {dataset: names}."""
    if not enabled(options):
        return None
    try:
        return PooledLoader(datasets, windows, options, progress, load_window)
    except _Unsupported as error:
        progress('loading', str(error))
        return None


def load_pooled(datasets, windows, options, progress, load_window, loader=None):
    """Pooled payload whose loads are limited by the run's molecule caps; None when caps are off/unsupported."""
    loader = loader or planned_loader(datasets, windows, options, progress, load_window)
    if loader is None:
        return None
    return PooledSampledPayload(loader.load(), loader)


def scanned_molecules(datasets, windows, options):
    """Alignments overlapping each window, summed over windows and datasets (cheap scan, no tag decoding)."""
    from .bam import _flatten_paths
    from .loader_sampling import scan
    total = 0
    for w in windows:
        for item in datasets:
            _, rows = scan(_flatten_paths(item['paths']), dict(chrom=w['chrom'], start=w['start'], end=w['end']),
                           options['input'].minimum_mapq)
            total += sum(1 for _, s, e in rows if s < w['end'] and e > w['start'])
    return total


class PooledSampledPayload(dict):
    """Pooled evidence with an in-process top-up hook (the SampledPayload contract of run_lattice_recaller:
    ``refine(requirements)`` updates the payload in place and returns whether its molecules changed)."""
    def __init__(self, body, loader):
        super().__init__(body)
        self._loader = loader

    def refine(self, requirements):
        body, changed = self._loader.refine(dict(self), requirements)
        if changed:
            self.clear(); self.update(body)
        return changed
