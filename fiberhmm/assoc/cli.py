"""``fiberhmm-scan assoc``: association statistics from a scan directory (feature hub design 7.10).

    fiberhmm-scan assoc SCAN_DIR --mode cooccur --a nfr.region --b fp.class -o out/
    fiberhmm-scan assoc SCAN_DIR --mode condition --a fp.class --response nuc:nrl -o out/
    fiberhmm-scan assoc SCAN_DIR --mode rate --a nuc.seg.regular --b tf -o out/
    fiberhmm-scan assoc SCAN_DIR --mode overlap --a peaks.peak --b nfr.region -o out/

Modes: ``cooccur`` (M1), ``condition`` (M2), ``rate`` (M3), ``overlap`` (M4). M1-M3 read tables or
annotated BAMs declared by the scan. Annotated scalars use their declared codec precision.
M4 reads hits and the callable territory. Writes ``result.json`` (and ``pairs.tsv`` / ``anchors.tsv``
where the mode has per-pair or per-anchor rows) to the output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

MODES = ('cooccur', 'condition', 'rate', 'overlap')


def _parser():
    p = argparse.ArgumentParser(prog='fiberhmm-scan assoc', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('scan_dir', help='A finished fiberhmm-scan directory')
    p.add_argument('--mode', required=True, choices=MODES, help='cooccur (M1), condition (M2), rate (M3), overlap (M4)')
    p.add_argument('--a', required=True, help='Feature type A (e.g. nfr.region, fp.class, nuc.seg.regular, peaks.peak)')
    p.add_argument('--b', help='Feature type B (cooccur, overlap); rate: tf (TF footprint centres from nuc marks) or a '
                               'state type whose member calls are the points (fp.class)')
    p.add_argument('--response', help='condition: per-molecule response MODULE:SCALAR (e.g. nuc:nrl, nfr:openness)')
    p.add_argument('--bed', help='condition: anchors are the --a hits whose anchor lies in these intervals (e.g. an '
                                 'enhancer BED); other --a hits are their matched backgrounds. Without it every hit '
                                 'is an anchor and only Delta_anchor and per-anchor rows are reported')
    p.add_argument('--backgrounds', choices=['hits', 'random', 'none'], default='hits',
                   help='condition: other hits outside --bed (default), random loci with a transferable state '
                        'rule (currently nfr.region), or none')
    p.add_argument('--background-k', type=int, default=5, help='condition: backgrounds per anchor (default 5)')
    p.add_argument('--background-step', type=int, default=500, help='condition: random candidate grid step in bp')
    p.add_argument('--background-min-sep', type=int, default=3000, help='condition: minimum separation in bp')
    p.add_argument('--background-width', type=int, help='condition: random interval width in bp; default median '
                   'anchor hit width; the scan state rule is evaluated over this interval')
    p.add_argument('--fasta', help='Indexed reference FASTA: condition matching covariate or overlap density strata')
    p.add_argument('--site-bases', default='AT', help='Opportunity bases counted in FASTA (default AT; C/G for '
                   'strand-specific deaminase contexts); density excludes ambiguous bases')
    p.add_argument('--site-window', type=int, default=1000, help='condition: site-density window in bp')
    p.add_argument('--site-bin-bp', type=int, default=10000, help='overlap: site-density stratum window in bp')
    p.add_argument('--region', action='append', metavar='CHROM:START-END', help='Restrict to windows (repeatable)')
    p.add_argument('--group', choices=['dataset', 'haplotype'], help='Group contrast: dataset or haplotype '
                                                                     '(haplotype from the input BAMs\' HP tags; '
                                                                     'unphased molecules are reported apart)')
    p.add_argument('--haplotype-tag', default='HP', help='BAM tag holding the haplotype (default HP)')
    p.add_argument('--labels', help='TSV with a molecule column and label columns (other groups, replicates)')
    p.add_argument('--group-label', help='With --labels: the label column to contrast')
    p.add_argument('--density', default='rho_read', choices=['rho_read', 'lambda_acc'],
                   help='nuc molecule scalar used as the mark-density covariate (default rho_read)')
    p.add_argument('--max-distance', type=int, default=5000, help='cooccur: maximum anchor distance (default 5000)')
    p.add_argument('--test', default='adjusted', choices=['adjusted', 'strata'], help='cooccur: pooled test')
    p.add_argument('--strata', help='overlap: BED of strata (chrom start end label); default one stratum per contig')
    p.add_argument('--n-perm', type=int, default=1000, help='Permutations / shifts / relocations (default 1000)')
    p.add_argument('--n-boot', type=int, default=1000, help='Bootstrap draws (default 1000)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('-o', '--output', required=True, help='Output directory')
    return p


def _regions(values):
    out = []
    for v in values or []:
        chrom, _, span = v.rpartition(':')
        a, _, b = span.replace(',', '').partition('-')
        out.append((chrom, int(a), int(b)))
    return out or None


def _write_tsv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _with_defaults(args):
    """Programmatic callers may pass a Namespace without newer options: fill them with the parser's defaults."""
    for action in _parser()._actions:
        if action.dest != 'help' and not hasattr(args, action.dest):
            setattr(args, action.dest, action.default)
    return args


def run(args) -> dict:
    from fiberhmm.assoc import scanio as S
    args = _with_defaults(args)
    if min(args.background_k, args.background_step, args.background_min_sep, args.site_window, args.site_bin_bp) <= 0:
        raise S.ScanInputError('background and site-density sizes must be positive')
    if args.background_width is not None and args.background_width <= 0:
        raise S.ScanInputError('--background-width must be positive')
    if args.fasta and args.strata:
        raise S.ScanInputError('choose --strata or --fasta for overlap strata')
    scan = S.open_scan(args.scan_dir)
    regions = _regions(args.region)
    group, gkind = None, 'label'
    if args.group == 'dataset':
        group = 'dataset'
    elif args.group == 'haplotype':
        group, gkind = 'haplotype', 'haplotype'
    elif args.group_label:
        group = args.group_label
    meta = {'scan_id': scan.manifest.get('scan_id'), 'mode': args.mode, 'a': args.a, 'b': args.b,
            'regions': regions, 'group': group,
            'scalar_precision': 'MA codecs' if scan.manifest.get('outputs', {}).get('annotate') else 'exact tables'}
    if args.fasta:
        meta['site_density'] = dict(bases=args.site_bases, window=args.site_window, bin_bp=args.site_bin_bp)
    if args.mode == 'cooccur':
        from fiberhmm.assoc.cooccur import cooccurrence
        if not args.b:
            raise S.ScanInputError('cooccur needs --b')
        mi = S.MoleculeIndex()
        A = S.read_states(scan, args.a, mi, regions)
        B = A if args.b == args.a else S.read_states(scan, args.b, mi, regions)
        ch = dict(getattr(A, 'channel_of', {}))
        ch.update(getattr(B, 'channel_of', {}))
        mt = S.molecule_table(scan, mi, density=args.density,
                              haplotype_tag=args.haplotype_tag if gkind == 'haplotype' else None,
                              channel_of=ch or None, regions=regions, labels_tsv=args.labels)
        res = cooccurrence(A, B, mt, max_distance=args.max_distance, group=group, group_kind=gkind, test=args.test,
                           n_boot=args.n_boot, seed=args.seed)
    elif args.mode == 'condition':
        res = _condition_from_scan(scan, args, regions, group, gkind)
    elif args.mode == 'rate':
        from fiberhmm.assoc.rate import rate_ratio
        d = S.rate_data_from_marks(scan, args.a, args.b or 'tf', regions)
        if gkind == 'haplotype' or args.labels:
            mi = S.MoleculeIndex()
            idx = []
            for m, ds, qs in zip(d.labels['molecule'], d.labels['dataset'], d.labels['qname']):
                qs = (qs,) if isinstance(qs, str) else (tuple(qs) or ('',))
                idx.append(mi.add(m, ds, qs[0]))
                for q in qs[1:]:                  # every record of the molecule, so conflicting HP tags unphase it
                    mi.add(m, ds, q)
            mt = S.molecule_table(scan, mi, haplotype_tag=args.haplotype_tag if gkind == 'haplotype' else None,
                                  regions=regions, labels_tsv=args.labels)
            for k, v in mt.labels.items():
                d.labels[k] = v[idx]
        res = rate_ratio(d, n_shift=args.n_perm, n_boot=args.n_boot, group=group, group_kind=gkind, seed=args.seed)
        res['coordinate'] = ('admissible reference bp (HUB 7.5): nucseg segments minus low-information; scan tables '
                             'carry no per-site informativeness')
    else:
        from fiberhmm.assoc.overlap import overlap_enrichment
        if not args.b:
            raise S.ScanInputError('overlap needs --b')
        terr = S.territory(scan, regions=regions)
        strata = None
        if args.strata:
            strata = [(c, int(s), int(e), lab) for c, s, e, lab in
                      (line.split()[:4] for line in Path(args.strata).read_text().splitlines() if line.strip())]
        if args.fasta:
            from .sites import density_strata
            strata = density_strata(scan, args.fasta, terr, bases=args.site_bases, bin_bp=args.site_bin_bp)
        A = S.read_hits(scan, args.a, regions)
        B = S.read_hits(scan, args.b, regions)
        res = {'A->B': overlap_enrichment(A, B, terr, strata, n_relocations=args.n_perm, seed=args.seed),
               'B->A': overlap_enrichment(B, A, terr, strata, n_relocations=args.n_perm, seed=args.seed + 1),
               'territory_bp': int(sum(e - s for _c, s, e in terr))}
    res['input'] = meta
    return res



def _in_bed(chrom, positions, intervals):
    """Vectorized interval membership, including nested/overlapping BED rows."""
    import numpy as np
    from collections import defaultdict
    by_chrom, hits = defaultdict(list), defaultdict(list)
    for c, a, b in intervals:
        by_chrom[c].append((a, b))
    for i, c in enumerate(chrom):
        hits[c].append(i)
    out = np.zeros(len(positions), bool)
    positions = np.asarray(positions)
    for c in by_chrom.keys() & hits.keys():
        rows = sorted(by_chrom[c])
        starts = np.array([a for a, b in rows])
        ends = np.maximum.accumulate([b for a, b in rows])
        indices = np.array(hits[c])
        points = positions[indices]
        j = np.searchsorted(starts, points, side='right')-1
        out[indices] = (j >= 0) & (points < ends[np.maximum(j, 0)])
    return out

def _condition_from_scan(scan, args, regions, group, gkind):
    """M2 from scan files: anchors are --a hits with their states, the response a per-molecule scalar
    (--response module:scalar); backgrounds are other hits, random loci evaluated by a transferable rule, or none.
    Matching uses depth, optional reference density and (for hit backgrounds) prevalence."""
    import numpy as np
    from fiberhmm.assoc import scanio as S
    from fiberhmm.assoc import strata as ST
    from fiberhmm.assoc.background import matched_backgrounds
    from fiberhmm.assoc.condition import Appearances, LocusTable, condition, mean_metric
    if not args.response or ':' not in args.response:
        raise S.ScanInputError('condition needs --response MODULE:SCALAR (e.g. nuc:nrl)')
    rmod, rcol = args.response.split(':', 1)
    S.require_tables(scan, [f'{rmod}/molecules.tsv.bgz'], 'condition')
    mi = S.MoleculeIndex()
    A = S.read_states(scan, args.a, mi, regions)
    n = A.n_hits
    depth = np.bincount(A.hit, minlength=n).astype(float)
    prev = np.bincount(A.hit, weights=A.state, minlength=n) / np.maximum(depth, 1)
    order = np.arange(n)
    if args.bed:
        bed = [(c, int(a), int(b)) for c, a, b in (line.split()[:3] for line in Path(args.bed).read_text().splitlines()
                                                   if line.strip() and not line.startswith(('#', 'track')))]
        inside = _in_bed(A.chrom, A.anchor, bed)
        anchors, cands = order[inside], order[~inside]
        if not len(anchors):
            raise S.ScanInputError(f'no {args.a} hit lies in --bed')
    else:
        anchors, cands = order, order[:0]
    random_positions = None
    if args.backgrounds == 'random':
        from fiberhmm.assoc.background import candidate_positions
        from fiberhmm.assoc.cooccur import FeatureStates
        width = args.background_width or max(1, int(np.median(A.end[anchors] - A.start[anchors])))
        random_positions = candidate_positions(S.territory(scan, regions=regions), step=args.background_step,
                                                margin=(width + 1) // 2, seed=args.seed)
        C = S.transferable_states(scan, args.a, random_positions, width, mi, regions)
        offset = A.n_hits
        A = FeatureStates(A.type_id, np.r_[A.hit_ids, C.hit_ids], np.r_[A.chrom, C.chrom],
                          np.r_[A.start, C.start], np.r_[A.end, C.end], np.r_[A.anchor, C.anchor],
                          np.r_[A.hit, C.hit + offset], np.r_[A.molecule, C.molecule], np.r_[A.state, C.state])
        cands = np.arange(offset, A.n_hits)
        depth = np.bincount(A.hit, minlength=A.n_hits).astype(float)
        prev = np.bincount(A.hit, weights=A.state, minlength=A.n_hits) / np.maximum(depth, 1)
    elif args.backgrounds == 'none':
        cands = order[:0]
    acov, ccov = {'depth': depth[anchors]}, {'depth': depth[cands]}
    if args.backgrounds == 'hits':
        acov['prevalence'], ccov['prevalence'] = prev[anchors], prev[cands]
    if args.fasta:
        from .sites import locus_density
        sd = locus_density(scan, args.fasta, A.chrom, A.anchor, bases=args.site_bases, window=args.site_window)
        acov['site_density'], ccov['site_density'] = sd[anchors], sd[cands]
    mb = matched_backgrounds([(A.chrom[i], int(A.anchor[i])) for i in anchors],
                             [(A.chrom[j], int(A.anchor[j])) for j in cands],
                             anchor_covariates=acov, candidate_covariates=ccov,
                             k=args.background_k, n_bins=10, min_sep=args.background_min_sep, seed=args.seed)
    y = S.molecule_scalars(scan, rmod, [rcol], mi, regions)[rcol]
    dens = S.molecule_scalars(scan, 'nuc', [args.density], mi, regions)[args.density]
    hap = S.haplotypes_from_bams(scan, mi, args.haplotype_tag, regions) if gkind == 'haplotype' else None
    extra = {}
    if args.labels:
        mt = S.molecule_table(scan, mi, regions=regions, labels_tsv=args.labels)
        extra = {k: v for k, v in mt.labels.items() if k != 'dataset'}
    if group is not None and group not in ('dataset', 'haplotype') and group not in extra:
        raise S.ScanInputError(f'--group-label {group!r} is not a column of --labels')
    loci = LocusTable.from_sets([(A.chrom[i], int(A.anchor[i])) for i in anchors],
                                [(A.chrom[j], int(A.anchor[j])) for j in cands], mb['sets'])
    hit_of_locus = list(anchors) + [cands[j] for _a, js in mb['sets'] for j in js]
    rows_l, rows_m, rows_s = [], [], []
    for l, h in enumerate(hit_of_locus):
        mols, st = A.states_of(int(h))
        rows_l += [l] * len(mols)
        rows_m += mols.tolist()
        rows_s += st.tolist()
    rows_m = np.array(rows_m, np.int64)
    ok = np.isfinite(y[rows_m])
    yy = np.where(ok, y[rows_m], 0.0)
    stats = np.stack([yy, ok.astype(float)], -1)[:, None, :]
    stratum = ST.density_bins(dens[rows_m], 12)
    labels = {'dataset': np.array(mi.dataset, dtype=object)[rows_m]}
    if hap is not None:
        labels['haplotype'] = hap[rows_m]
    for k, v in extra.items():
        labels[k] = v[rows_m]
    app = Appearances(np.array(rows_l), rows_m, np.array(rows_s), stratum, ok.astype(np.int64), stats, labels,
                      density=dens[rows_m])
    res = condition(app, loci, mean_metric(f'mean {rcol}'), bins=['molecule'], n_perm=args.n_perm,
                    n_role=args.n_perm, n_boot=args.n_boot, group=group, group_kind=gkind, seed=args.seed)
    res['backgrounds'] = {k: v for k, v in mb.items() if k != 'sets'}
    res['backgrounds']['rule'] = args.backgrounds
    if random_positions is not None:
        res['backgrounds']['width_bp'] = width
        res['backgrounds']['state_rule'] = args.a
    return res


def main(argv=None):
    from fiberhmm.cli.common import run_reporting_input_errors
    return run_reporting_input_errors('fiberhmm-scan assoc', lambda: _main(argv))


def _main(argv=None):
    from fiberhmm.assoc.scanio import ScanInputError
    p = _parser()
    args = p.parse_args(argv)
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        res = run(args)
    except ScanInputError as error:
        p.exit(2, f'fiberhmm-scan assoc: error: {error}\n')
    pairs = res.pop('pairs', None)
    anchors = res.pop('per_anchor', None)
    (out / 'result.json').write_text(json.dumps(res, indent=1, sort_keys=True, default=_json_default))
    if pairs:
        _write_tsv(out / 'pairs.tsv', pairs)
    if anchors:
        _write_tsv(out / 'anchors.tsv', anchors)
    print(json.dumps({'output': str(out), 'mode': args.mode, 'pairs': len(pairs or []),
                      'anchors': len(anchors or [])}, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
