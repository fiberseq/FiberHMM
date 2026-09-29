"""Frozen class catalogs for the lattice recaller: freeze a run's classes, score new molecules against them.

Freezing reads a completed ``lattice_recaller`` run (manifest.json, result.json.gz, evidence.json.gz) and writes a
versioned, digest-checked JSON catalog (schema ``fiberhmm.frozen_classes.lattice_recaller.v1``) holding everything the
quantification stage needs, so nothing is rediscovered:

* the frame (analysis region; pooled BED windows when the run was an oriented CL-CR pool), discovery tiles and the
  full recaller/preparation parameters of the run;
* every discovered class (id, span, pooled L/R edge boxes, calls, stability), supported or not: all of them are
  components of the EM mixture, so dropping one would change every other class's prevalence;
* per source channel (dataset::strand, with its chemistry): each class's final edge boxes (chemistry jitter and any
  kept edge contraction applied), its learned internal spots (positions and protected-state mark rates) and the
  source estimates (prevalence tiers, support, resolution, unknown-state accessible fraction) for reference;
* hashed identities of the training molecules and provenance (source manifest digest, input digest, parameters,
  code version and hashes at freeze time).

Applying runs the recaller's own quantification (``workflow.quantify``) with discovery disabled: classes, tiles,
per-channel edge boxes and spot rates are fixed; EM weights, posteriors, per-molecule labels and edges, prevalence
tiers, held-out support gain, resolution and the channel's unknown-state accessible fraction (and efficiency, when the
source run enabled it) are computed from the target molecules. The output is an ordinary lattice-recaller run with an
added ``transfer`` block.
"""
from __future__ import annotations

import copy
import hashlib
import math
import subprocess
from pathlib import Path

import numpy as np

from ..artifacts import digest, read_json, write_json
from . import model as Mo

SCHEMA = 'fiberhmm.frozen_classes.lattice_recaller.v1'
TRANSFER_SCHEMA = 'fiberhmm.frozen_transfer.lattice_recaller.v1'
MODE = 'lattice_recaller'
CATALOG_NAME = 'frozen_classes.json.gz'
CHEMISTRIES = ('ddda', 'dddb', 'hia5-pacbio', 'hia5-nanopore')
# Parameter groups a transfer may change: molecule preparation and execution. Everything else is fixed by the catalog.
OVERRIDABLE_GROUPS = ('input', 'families', 'compute')


def code_identity():
    """Version, git commit (when run from a checkout) and hashes of the recaller sources of the running code."""
    import fiberhmm
    here = Path(__file__).parent
    commit = None
    try:
        out = subprocess.run(['git', '-C', str(here), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5)
        commit = out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    files = sorted(here.glob('*.py')) + [here.parent/'parameters.py']
    return dict(fiberhmm_version=getattr(fiberhmm, '__version__', None), git_commit=commit,
                implementation_sha256={str(p.relative_to(here.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def _parse_spots(row):
    """Learned spots of one class x channel row: full precision when the run stored them, else the 3-decimal string."""
    if isinstance(row.get('spot_rates'), dict):
        return {int(p): float(v) for p, v in row['spot_rates'].items()}, 'full'
    out = {}
    for item in filter(None, (row.get('spots') or '').split(';')):
        p, v = item.split(':'); out[int(p)] = float(v)
    return out, ('3_decimals' if out else 'full')


def freeze_run(run_dir, output):
    """Freeze a completed lattice-recaller run into a catalog at ``output``; returns the loaded catalog."""
    from ..transfer import molecule_keys
    root = Path(run_dir)
    for name in ('manifest.json', 'result.json.gz', 'evidence.json.gz'):
        if not (root/name).is_file():
            raise ValueError(f'{root} is not a complete lattice-recaller run directory (missing {name})')
    manifest = read_json(root/'manifest.json')
    if manifest.get('cr_mode') != MODE:
        raise ValueError(f"Not a lattice_recaller run (cr_mode={manifest.get('cr_mode')!r})")
    if manifest.get('status') != 'complete':
        raise ValueError('Only completed lattice-recaller runs can be frozen')
    if manifest.get('transfer'):
        raise ValueError('This run is itself a frozen-class transfer; freeze the discovery run it came from')
    result = read_json(root/'result.json.gz'); evidence = read_json(root/'evidence.json.gz')
    rec = result.get('recaller') or {}
    if not rec.get('classes'):
        raise ValueError('The run has no discovered classes to freeze')
    chem = {d['dataset_id']: d['chemistry'] for d in manifest['datasets']}
    classes = [dict(id=c['id'], span=[float(c['start']), float(c['end'])], L=[int(x) for x in c['L']], R=[int(x) for x in c['R']],
                    calls=int(c['calls']), stability=float(c['stability']), supported_channels=int(c.get('supported_channels', 0)))
               for c in rec['classes']]
    channels = {}; precision = set()
    for r in rec.get('rows', []):
        spots, p = _parse_spots(r); precision.add(p)
        ch = channels.setdefault(r['channel'], dict(dataset=r['dataset'], strand=r['strand'], chemistry=chem[r['dataset']],
                                                     unknown_accessible_fraction=r.get('unknown_accessible_fraction'),
                                                     efficiency=r.get('efficiency'), classes={}))
        ch['classes'][r['class_id']] = dict(
            L=[int(r['L0']), int(r['L1'])], R=[int(r['R0']), int(r['R1'])], spots={str(k): v for k, v in sorted(spots.items())},
            edge_contraction=r.get('edge_contraction') or '',
            source=dict({k: r.get(k) for k in ('prevalence', 'prevalence_edge', 'prevalence_loose', 'prevalence_lower_bound', 'supported',
                                               'support_gain_nats', 'resolved', 'resolution_nats', 'molecules')}))
    units = [u for s in evidence['strata'] for u in s['units']]
    hashes = sorted(set().union(*(molecule_keys(u) for u in units))) if units else []
    pooling = manifest.get('pooling') or evidence.get('pooling')
    body = dict(schema=SCHEMA, engine=MODE,
                frame=dict(region=dict(manifest['region']), pooled=bool(pooling),
                           windows=(pooling or {}).get('windows'), coordinate_system=(pooling or {}).get('coordinate_system', 'reference')),
                parameters=manifest['parameters'], tiles=[[int(a), int(b)] for a, b in manifest['recaller']['tiles']],
                classes=classes, datasets=[dict(dataset_id=d['dataset_id'], chemistry=d['chemistry'], units=d.get('units'), model=d.get('model'))
                                           for d in manifest['datasets']],
                channels=channels, spot_precision='3_decimals' if '3_decimals' in precision else 'full',
                training_molecule_hashes=hashes,
                provenance=dict(source_run=str(root.resolve()), source_manifest_sha256=digest(manifest), input_digest=manifest.get('input_digest'),
                                source_mode=manifest.get('mode'), source_seconds=manifest.get('seconds'),
                                source_implementation_sha256=manifest.get('implementation_sha256'),   # the recaller receipt does not record it
                                freeze_code=code_identity()),   # no timestamp: re-freezing a run gives the same digest
                score_semantics='EM prevalence over fixed classes, broader protection, other shape and accessible; three-way per-molecule labels; '
                                'geometry and learned spots fixed, prevalence re-estimated on the target molecules')
    body = dict(body, content_sha256=digest(body))
    write_json(output, body)
    return load_catalog(output)


def load_catalog(path_or_body):
    """Read and validate a catalog (path or already-parsed JSON); spots come back keyed by int position."""
    from ..parameters import parse_options
    body = copy.deepcopy(path_or_body) if isinstance(path_or_body, dict) else read_json(path_or_body)
    schema = body.get('schema')
    if schema != SCHEMA:
        raise ValueError(f'Unsupported frozen-class catalog schema {schema!r}; this FiberHMM reads {SCHEMA!r}')
    expected = body.pop('content_sha256', None)
    if expected != digest(body):
        raise ValueError('Frozen-class catalog content digest does not match (edited or truncated file)')
    if body.get('engine') != MODE:
        raise ValueError('Frozen-class catalog engine must be lattice_recaller')
    options = parse_options(body['parameters'])
    if options['cr'].engine != MODE:
        raise ValueError('Frozen-class catalog parameters are not lattice-recaller parameters')
    region = body['frame']['region']
    if not region['end'] > region['start']:
        raise ValueError('Invalid frozen frame')
    if not body['tiles'] or any(not b > a for a, b in body['tiles']):
        raise ValueError('Invalid frozen discovery tiles')
    ids = [c['id'] for c in body['classes']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Frozen catalog needs nonempty, unique class IDs')
    for c in body['classes']:
        if not (c['span'][0] < c['span'][1] and c['L'][0] <= c['L'][1] and c['R'][0] <= c['R'][1]):
            raise ValueError(f"Invalid geometry for frozen class {c['id']}")
    for name, ch in body['channels'].items():
        if ch['chemistry'] not in CHEMISTRIES or name != f"{ch['dataset']}::{ch['strand']}":
            raise ValueError(f'Invalid frozen channel {name}')
        for cid, e in ch['classes'].items():
            if cid not in ids:
                raise ValueError(f'Frozen channel {name} refers to unknown class {cid}')
            if any(not 0 < v < 1 or not math.isfinite(v) for v in e['spots'].values()):
                raise ValueError(f'Invalid learned spot rate for {cid} on {name}')
            e['spots'] = {int(p): float(v) for p, v in e['spots'].items()}
    body['content_sha256'] = expected
    return body


# ------------------------------------------------------------------------------------------------ scoring
def score_channel(units, gs, f, opt, fixed):
    """fit_channel with nothing fitted but the mixture: fixed per-channel boxes and spots, EM weights and posteriors,
    held-out support gain, resolution, tiers and per-molecule calls (same return shape as model.fit_channel).

    gs: the class geometries quantify built for this channel (catalog class boxes with the target chemistry's jitter);
    fixed: FrozenClasses.fixed(...) (source channel boxes, spots and contraction records, None where absent)."""
    info = [Mo.expected_evidence(units, g) for g in gs]      # as fit_channel: resolution before any contraction
    gs = [dict(g, L=list(b['L']), R=list(b['R'])) if b else g for g, b in zip(gs, fixed['boxes'])]
    prof = [dict(s) for s in fixed['spots']]
    sc = Mo.Scorer(units, gs, f, opt)
    M, keep = sc.rows(prof)
    if not len(M):
        return None
    w, P = Mo.em(M)
    fold = np.array([Mo._hash2(u['uid']) for u in keep]); gains = []
    for c in range(len(gs)):
        gain = 0.
        for kf in (0, 1):
            tr, te = M[fold != kf], M[fold == kf]
            if len(tr) < 10 or len(te) < 1:
                gain = float('nan'); break
            wf, _ = Mo.em(tr); wd, _ = Mo.em(np.delete(tr, c, axis=1))
            gain += Mo.loglik(te, wf) - Mo.loglik(np.delete(te, c, axis=1), wd)
        gains.append(gain)
    return dict(w=w, P=P, units=keep, spots=prof, support_gain=gains, resolution=info, n=len(M), gs=gs, edges=list(fixed['edges']),
                tiers=Mo.prevalence_tiers(sc.keep, P, w, gs, prof),
                calls=[dict(classes=[Mo.molecule_edges(it, c, gs[c], prof[c]) for c in range(len(gs))], broader=it['broader'],
                            tiers=Mo.molecule_tiers(u, gs, prof))
                       for it, u in zip(sc.items, sc.keep)])


class FrozenClasses:
    """The catalog as the recaller workflow consumes it (see workflow.run_lattice_recaller / quantify ``frozen=``)."""
    score = staticmethod(score_channel)

    def __init__(self, catalog, chemistry, dataset_map=None, exclusions=None, window=None):
        self.catalog = catalog; self.chemistry = dict(chemistry); self.dataset_map = dict(dataset_map or {})
        self.exclusions = exclusions or []; self.window = window; self.channel_map = {}
        for target, source in self.dataset_map.items():
            if not any(ch['dataset'] == source for ch in catalog['channels'].values()):
                raise ValueError(f'--dataset-map {target}={source}: no frozen channel of source dataset {source}')

    def discovery(self):
        """(classes, dropped, tiles, diagnostics) in the shape workflow.discover returns them."""
        classes = [dict(id=c['id'], span=(c['span'][0], c['span'][1]), L=list(c['L']), R=list(c['R']), calls=c['calls'],
                        stability=c['stability']) for c in self.catalog['classes']]
        return classes, [], [tuple(t) for t in self.catalog['tiles']], []

    def resolve(self, channel, chemistry):
        """The frozen source channel whose boxes and spots a target channel uses, or None (class geometry only)."""
        ds, strand = channel.split('::', 1); src = self.catalog['channels']
        if ds in self.dataset_map:
            cand = f'{self.dataset_map[ds]}::{strand}'
            if cand in src and src[cand]['chemistry'] != chemistry:
                raise ValueError(f"--dataset-map {ds}={self.dataset_map[ds]}: chemistry {chemistry} differs from the source's {src[cand]['chemistry']}")
            return cand if cand in src else None
        if channel in src and src[channel]['chemistry'] == chemistry:
            return channel
        cands = sorted(c for c, v in src.items() if v['chemistry'] == chemistry and v['strand'] == strand)
        if len(cands) > 1:
            raise ValueError(f'Target channel {channel} ({chemistry}) matches several frozen source channels ({", ".join(cands)}); '
                             f'choose one with --dataset-map {ds}=SOURCE_DATASET')
        return cands[0] if cands else None

    def fixed(self, classes, grp, channel, chemistry):
        src = self.resolve(channel, chemistry); self.channel_map[channel] = src
        per = self.catalog['channels'][src]['classes'] if src else {}
        entries = [per.get(classes[x]['id']) for x in grp]
        return dict(source_channel=src, boxes=[dict(L=e['L'], R=e['R']) if e else None for e in entries],
                    spots=[dict(e['spots']) if e else {} for e in entries],
                    # The source channel's contraction record; prefixed with the source when it is another channel.
                    edges=[(e['edge_contraction'] if src == channel or not e['edge_contraction'] else f"from {src}: {e['edge_contraction']}")
                           if e else '' for e in entries])

    def provenance(self):
        cat = self.catalog
        return dict(schema=TRANSFER_SCHEMA, catalog_schema=cat['schema'], catalog_sha256=cat['content_sha256'],
                    source=dict(cat['provenance'], frame=cat['frame']), window=self.window,
                    discovery='disabled: classes, tiles, per-channel edge boxes and learned spots are fixed by the catalog',
                    recomputed='EM prevalence and posteriors, per-molecule labels, edges and tiers, held-out support, resolution, '
                               'unknown-state accessible fraction (and efficiency when enabled) from the target molecules',
                    spot_precision=cat['spot_precision'], dataset_map=self.dataset_map,
                    channel_map={k: self.channel_map[k] for k in sorted(self.channel_map)},
                    excluded_training_molecules=len(self.exclusions), apply_code=code_identity())


# ------------------------------------------------------------------------------------------------ apply
def transfer_options(catalog, strata, overrides=None, cores=None):
    """The source run's parameters; only preparation/execution groups may be overridden."""
    from ..parameters import parse_options
    values = copy.deepcopy(catalog['parameters'])
    for group, settings in (overrides or {}).items():
        if group not in OVERRIDABLE_GROUPS:
            raise ValueError(f"Parameter group '{group}' is fixed by the frozen catalog; a transfer may change only "
                             + ', '.join(OVERRIDABLE_GROUPS))
        values.setdefault(group, {}).update(settings)
    values['families']['stop_after'] = 'resolved'
    if cores is not None:
        values['compute']['cores'] = int(cores)
    elif 'cores' not in (overrides or {}).get('compute', {}):
        from ..regions import machine_compute_defaults       # the source machine's worker count is not this machine's
        values['compute']['cores'] = machine_compute_defaults()['cores']
    values['sr']['enabled'] = any(s['chemistry'] in ('ddda', 'dddb') for s in strata)
    values['cross']['enabled'] = len({s['dataset_id'] for s in strata}) > 1
    return parse_options(values)


def exclude_training(payload, catalog):
    """A copy of the payload without molecules the catalog was trained on, and the exclusion ledger."""
    from ..transfer import molecule_keys
    training = set(catalog['training_molecule_hashes']); excluded = []; strata = []
    for s in payload['strata']:
        keep = []
        for u in s['units']:
            if molecule_keys(u) & training:
                excluded.append(dict(dataset=s['dataset_id'], unit_id=u['unit_id'], reason='training_molecule'))
            else:
                keep.append(u)
        strata.append(dict(s, units=keep))
    return dict(payload, strata=strata), excluded


def apply_catalog(catalog, payload, output_dir, *, parameters=None, cores=None, dataset_map=None, include_training=False,
                  progress=None, window=None):
    """Score ``payload`` against the frozen classes and write a normal lattice-recaller run to ``output_dir``.

    Returns (result, analysed payload). The payload must be in the catalog's frame (same region start/end)."""
    from .workflow import run_lattice_recaller
    from . import units as Un
    frame = catalog['frame']['region']
    if [payload['region']['start'], payload['region']['end']] != [frame['start'], frame['end']]:
        raise ValueError(f"Target evidence frame {payload['region']['start']}-{payload['region']['end']} differs from the frozen frame "
                         f"{frame['start']}-{frame['end']}; supply BAMs with oriented BED windows of width {frame['end'] - frame['start']}")
    target, excluded = (payload, []) if include_training else exclude_training(payload, catalog)
    options = transfer_options(catalog, target['strata'], parameters, cores)
    chem = {s['dataset_id']: s['chemistry'] for s in target['strata']}
    frozen = FrozenClasses(catalog, chem, dataset_map, excluded, window)
    for ds in frozen.dataset_map:
        if ds not in chem:
            raise ValueError(f'--dataset-map names target dataset {ds}, which is not in the target evidence')
    for s in target['strata']:                     # fail fast on ambiguous channel mappings, before any scoring
        for ch in sorted({Un.channel_of(s, u) for u in s['units']}):
            frozen.resolve(ch, s['chemistry'])
    result = run_lattice_recaller(target, options, output_dir, progress, frozen=frozen)
    if excluded:
        write_json(Path(output_dir)/'transfer_exclusions.json', excluded)
    return result, target


def shift_payload(payload, origin, region):
    """Move a pooled one-window payload (frame 0..width) to the catalog frame starting at ``origin``, keeping unit IDs
    and genomic provenance (with coordinate_origin, which BAM export subtracts)."""
    from ..regions import orient_unit
    width = payload['region']['end'] - payload['region']['start']
    out = dict(payload, region=region, strata=[])
    if origin == 0:
        out['strata'] = payload['strata']; return out
    ident = dict(chrom='oriented', start=0, end=width, strand='+', name='frame')
    for s in payload['strata']:
        units = []
        for u in s['units']:
            v = orient_unit(u, ident, origin=origin)
            v['unit_id'] = u['unit_id']; v['genomic_provenance'] = dict(u['genomic_provenance'], coordinate_origin=origin)
            units.append(v)
        out['strata'].append(dict(s, units=units))
    return out
