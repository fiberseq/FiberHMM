"""Calibration by thinning, and the absolute-control path (NUC 4).

**Thinning is a degradation model** (D2). It gives the edge noise
sigma_loc(lambda), the structural outcome rates and the overshoot shift as
mark density falls, the supported lambda range and the low-information floor.
It never gives true lengths or class offsets: the dense reference is itself
biased. Named classes need an absolute control (:func:`calibrate_absolute`).

1. **Population**: up to ``n_population`` molecules by stable name-hash rank
   (never a viewport), restricted to the central 90% of informative-site
   density so thinned and natural molecules share base composition. The
   population is reported in lambda_acc strata.
2. **Thin**: molecules in the top lambda_acc quartile are thinned to target
   lambda values below their own, from 0.2 x the quartile up (each mark kept with probability
   lambda_t / lambda_acc,m), and re-called by the dataset's caller with the
   emission table rescaled by the same factor (:mod:`fiberhmm.nuc.recall`);
   ``seeds`` repeats. Retention factors are quantised to a log grid so
   workers reuse callers. The unthinned dense molecules and the natural
   molecules are re-called by the same caller, so every comparison is within
   one caller version.
3. **Outcomes**: each dense footprint is matched to the thinned calls by
   interval overlap: *kept* (one thinned call, reciprocal overlap >= 50%),
   *merged* (a thinned call covering >= 50% of two or more dense calls),
   *split* (two or more thinned calls inside it) or *dropped*. Rates are
   modelled per nominal-lambda bin (retention x the dense molecule's
   lambda_acc; the rate re-measured on merged calls is biased upward); sigma_loc (RMS of the central 95% of
   edge shifts) and the overshoot (median length change) come from kept
   footprints, merges counting as censored.
4. **Validate**: 20% of the dense molecules (by hash) are held out. The
   training thinning predicts mu_mono(lambda); the mismatch against the
   natural curve, and against the held-out thinning, is reported per bin.
5. **Restrict**: only lambda covered by both natural and thinning data
   (2.5-97.5 percentiles; the thinning series ends at the dense references)
   is supported; outside it footprints are can't-tell.
6. **Floor (D8)**: the lambda_acc below which held-out thinned molecules have
   a coarse can't-tell rate above 50%.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fiberhmm.nuc.calibration import (NucCalibration, calibrate_self, footprint_table,
                                      hash_rank_sample, molecule_hash)
from fiberhmm.nuc.density import accessible_counts
from fiberhmm.nuc.params import DEFAULT_PARAMS, NucParams
from fiberhmm.nuc.records import DecodeContext, NucMolecule, decode_record
from fiberhmm.nuc.sizes import CANT_TELL, EDGE_TRUNCATED, AbsoluteClasses, fit_reference_curve

KEEP_GRID = np.exp(np.linspace(np.log(0.02), 0.0, 60))


def quantise_keep(f: float) -> float:
    return float(KEEP_GRID[int(np.argmin(np.abs(np.log(KEEP_GRID) - np.log(max(f, 1e-3)))))])


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------

_W: Dict = {}


def _winit(settings):
    _W['settings'] = settings
    _W['callers'] = {}


def _caller(f: float):
    from fiberhmm.nuc.recall import Caller
    c = _W['callers'].get(f)
    if c is None:
        if len(_W['callers']) > 16:
            _W['callers'].clear()
        c = Caller(_W['settings'], thin=f)
        _W['callers'][f] = c
    return c


def _wcall(task):
    """task = (key, fiber_read, keep, seed) -> (key, calls or None)."""
    from fiberhmm.nuc.recall import thin_fiber_read
    key, fr, keep, seed = task
    if keep < 1.0:
        fr = thin_fiber_read(fr, keep, np.random.default_rng(seed))
    try:
        return key, _caller(keep).call(fr), sorted(fr['m6a_query_positions'])
    except Exception as exc:  # recorded, never silently mixed into results
        return key, {'error': repr(exc)}, None


def _run_tasks(tasks, settings, cores):
    if cores <= 1:
        _winit(settings)
        return [_wcall(t) for t in tasks]
    from multiprocessing import get_context
    with get_context('fork').Pool(cores, initializer=_winit, initargs=(settings,)) as pool:
        return pool.map(_wcall, tasks, chunksize=8)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _recalled_molecule(mol: NucMolecule, calls: dict, mod_seq_positions=None) -> NucMolecule:
    """``mol`` with calls (SEQ frame) replaced and, for thinned runs, marks
    restricted to the kept ones."""
    L = mol.length

    def m(iv):
        return [(L - e, L - s) for s, e in iv] if mol.is_reverse else list(iv)
    hits = mol.hits
    if mod_seq_positions is not None:
        keep = np.zeros(L, dtype=bool)
        p = np.asarray(mod_seq_positions, dtype=np.int64)
        p = p[(p >= 0) & (p < L)]
        if mol.is_reverse:
            p = L - 1 - p
        keep[p] = True
        hits = hits & keep
    return NucMolecule(name=mol.name, length=L, is_reverse=mol.is_reverse, nucs=m(calls['nucs']),
                       tfs=m(calls['tfs']), msps=m(calls['msps']), hits=hits,
                       informative=mol.informative, no_call=mol.no_call, chrom=mol.chrom,
                       ref_start=mol.ref_start, ref_end=mol.ref_end, flag=mol.flag, hp=mol.hp,
                       blocks=mol.blocks, meta=dict(mol.meta))


def match_outcomes(dense: np.ndarray, thin: np.ndarray, min_recip: float = 0.5):
    """Classify each dense footprint against the thinned calls.

    Returns a list of ``(outcome, m, d_left, d_right, d_len)`` per dense
    footprint; ``m`` is the number of dense calls a merged call covers."""
    out = []
    if len(dense) == 0:
        return out
    ts, te = (thin[:, 0], thin[:, 1]) if len(thin) else (np.zeros(0), np.zeros(0))
    for s, e in dense:
        L = e - s
        ov = np.maximum(0, np.minimum(te, e) - np.maximum(ts, s))
        cand = np.flatnonzero(ov > 0)
        if len(cand) == 0:
            out.append(('dropped', 0, np.nan, np.nan, np.nan))
            continue
        big = [j for j in cand if ov[j] >= min_recip * L]
        inside = [j for j in cand if ov[j] >= min_recip * (te[j] - ts[j])]
        if len(inside) >= 2 and not big:
            out.append(('split', len(inside), np.nan, np.nan, np.nan))
            continue
        if not big:
            out.append(('dropped', 0, np.nan, np.nan, np.nan))
            continue
        j = max(big, key=lambda k: ov[k])
        tl = te[j] - ts[j]
        # how many dense calls does this thinned call cover by >= 50%?
        dov = np.maximum(0, np.minimum(dense[:, 1], te[j]) - np.maximum(dense[:, 0], ts[j]))
        n_cov = int((dov >= min_recip * (dense[:, 1] - dense[:, 0])).sum())
        if n_cov >= 2:
            out.append(('merged', n_cov, np.nan, np.nan, np.nan))
        elif ov[j] >= min_recip * tl:
            out.append(('kept', 1, float(ts[j] - s), float(te[j] - e), float(tl - L)))
        else:
            out.append(('dropped', 0, np.nan, np.nan, np.nan))
    return out


def edge_sigma(shifts) -> float:
    """Per-edge noise: RMS of edge shifts over their central 95%. (Most kept
    edges do not move at all, so a MAD would read 0.)"""
    v = np.asarray(shifts, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float('nan')
    lo, hi = np.quantile(v, [0.025, 0.975])
    v = v[(v >= lo) & (v <= hi)]
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else 0.0


def supported_range(lam_natural, lam_thinned, lam_dense, q: float = 0.025):
    """[lo, hi] of lambda covered by natural and thinning data (NUC 4.1 step 5).

    The thinned series runs from the lowest target up to the unthinned dense
    references, so its low end is the thinned footprints' and its high end the
    dense references' (pooling them would let the far more numerous thinned
    footprints pull the top quantile down and cut off the densest natural
    footprints, which are measured directly)."""
    nat = np.asarray(lam_natural, float)
    thin = np.asarray(lam_thinned, float)
    dense = np.asarray(lam_dense, float)
    if not (len(nat) and len(thin) and len(dense)):
        return None
    lo = max(np.quantile(nat, q), np.quantile(thin, q))
    hi = min(np.quantile(nat, 1 - q), np.quantile(dense, 1 - q))
    return [float(lo), float(hi)] if hi > lo else None


def _bin_edges(values: np.ndarray, n: int) -> np.ndarray:
    v = np.asarray(values, float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < n * 10:
        n = max(1, len(v) // 10)
    e = np.quantile(np.log(v), np.linspace(0, 1, n + 1))
    e[0] -= 1e-9
    e[-1] += 1e-9
    return e


def _curve_mismatch(a, b, lam_grid) -> List[float]:
    return [float(x) for x in (a.mu_at(lam_grid) - b.mu_at(lam_grid))]


# ---------------------------------------------------------------------------
# calibrate_thin
# ---------------------------------------------------------------------------

def load_population(bam_path: str, params: NucParams = DEFAULT_PARAMS, n_population: int = 2000,
                    region: Optional[str] = None, stats: Optional[dict] = None):
    """(molecules, fiber_reads, caller settings, chemistry) for the
    hash-ranked calibration population of one BAM (primary records)."""
    import pysam
    from fiberhmm.nuc.recall import CallerSettings, fiber_read_from_record
    with pysam.AlignmentFile(bam_path, 'rb') as bam:
        ctx = DecodeContext(bam.header, params)
        settings = CallerSettings.from_header(bam.header)
        it = bam.fetch(region=region) if region else bam.fetch(until_eof=True)
        names = sorted({r.query_name for r in it if not (r.is_secondary or r.is_supplementary)},
                       key=molecule_hash)
        keep = set(names[:n_population * 2])           # density filter trims later
        bam.reset() if region is None else None
        it = bam.fetch(region=region) if region else bam.fetch(until_eof=True)
        mols, frs = [], {}
        for r in it:
            if r.is_secondary or r.is_supplementary or r.query_name not in keep:
                continue
            m, why = decode_record(r, ctx)
            if stats is not None:
                stats[why] = stats.get(why, 0) + 1
            if m is None:
                continue
            fr = fiber_read_from_record(r, settings)
            if fr is None:
                continue
            mols.append(m)
            frs[m.name] = fr
    return mols, frs, settings, ctx.chemistry


def calibrate_thin(bam_path: str, params: NucParams = DEFAULT_PARAMS, *, label: str = '',
                   n_population: int = 2000, seeds: int = 5, n_targets: int = 8,
                   holdout: float = 0.2, cores: int = 1, region: Optional[str] = None,
                   settings_overrides: Optional[dict] = None, progress=None) -> NucCalibration:
    """Thinning calibration of one dataset (NUC 4.1)."""
    def say(msg):
        if progress is not None:
            progress(msg)
    stats: dict = {}
    mols, frs, settings, chem = load_population(bam_path, params, n_population, region, stats)
    if settings_overrides:
        settings = replace(settings, **settings_overrides)
    # informative-density matching: central 90%
    dens = np.array([m.n_informative / max(m.length, 1) for m in mols])
    lo_d, hi_d = np.quantile(dens, [0.05, 0.95]) if len(dens) >= 20 else (0, 1)
    mols = [m for m, d in zip(mols, dens) if lo_d <= d <= hi_d]
    mols = hash_rank_sample(mols, n_population)
    say(f'population {len(mols)} molecules')
    cal = calibrate_self(mols, params, label=label, chemistry=chem, fit_composition=False)
    cal.source = 'thin'
    cal.caller = settings.to_dict()
    lam = np.array([float(cal.prior.shrink(*accessible_counts(m, params))) for m in mols])
    strata_edges = np.quantile(lam, np.linspace(0, 1, 7))
    q75 = float(np.quantile(lam, 0.75))
    dense_idx = [i for i in range(len(mols)) if lam[i] >= q75]
    natural_idx = list(range(len(mols)))
    hold = {i for i in dense_idx if int(molecule_hash(mols[i].name + ':holdout'), 16) / 16 ** 16 < holdout}
    # targets reach well below the natural range so the floor (D8) is found;
    # only lambda covered by natural data too is supported (step 5)
    lam_lo = min(float(np.quantile(lam, 0.05)), 0.2 * q75)
    targets = np.exp(np.linspace(np.log(max(lam_lo, 0.02)), np.log(q75 * 0.9), n_targets))
    say(f'dense {len(dense_idx)} (held out {len(hold)}), targets {np.round(targets, 3).tolist()}')
    tasks = []
    for i in natural_idx:
        tasks.append((('nat', i, 0, 0), frs[mols[i].name], 1.0, 0))
    for i in dense_idx:
        for t_i, t in enumerate(targets):
            if t >= lam[i]:
                continue
            f = quantise_keep(t / lam[i])
            for s in range(seeds):
                seed = int(molecule_hash(f'{mols[i].name}:{t_i}:{s}'), 16) % (2 ** 31)
                tasks.append((('thin', i, t_i, s), frs[mols[i].name], f, seed))
    say(f'{len(tasks)} caller runs on {cores} cores')
    import time
    t0 = time.time()
    res = _run_tasks(tasks, settings, cores)
    t_call = time.time() - t0
    errors = [r for r in res if isinstance(r[1], dict) and 'error' in r[1]]
    recalled: Dict[tuple, NucMolecule] = {}
    for key, calls, mods in res:
        if calls is None or (isinstance(calls, dict) and 'error' in calls):
            continue
        i = key[1]
        recalled[key] = _recalled_molecule(mols[i], calls, mods if key[0] == 'thin' else None)
    # --- outcomes and edge noise on thinned dense molecules --------------------
    rows = []   # (achieved lambda_i, outcome, m, dl, dr, dlen, held_out)
    thin_ft = {'train': ([], []), 'hold': ([], [])}
    mol_ct = []  # (achieved lambda_acc, cant_tell_rate, held_out) filled after sigma_loc
    nat_curve_mols = [recalled[('nat', i, 0, 0)] for i in natural_idx if ('nat', i, 0, 0) in recalled]
    for key, tm in recalled.items():
        if key[0] != 'thin':
            continue
        i = key[1]
        ref = recalled.get(('nat', i, 0, 0))
        if ref is None:
            continue
        ft_ref = footprint_table([ref], cal, params)
        keep_ref = ~ft_ref['truncated']
        dense_iv = ref.nucs[keep_ref]
        # outcomes are binned by the nominal thinned rate (retention x the dense
        # molecule's lambda_acc): the rate measured on the re-called structure is
        # biased upward once merges remove weakly marked linkers
        lam_t = quantise_keep(targets[key[2]] / lam[i]) * lam[i]
        oc = match_outcomes(dense_iv, tm.nucs)
        for o in oc:
            rows.append((lam_t,) + o + (i in hold,))
        ft = footprint_table([tm], cal, params)
        kk = ~ft['truncated']
        part = 'hold' if i in hold else 'train'
        thin_ft[part][0].append(ft['length'][kk])
        thin_ft[part][1].append(ft['lambda'][kk])
    R = rows
    lam_rows = np.array([r[0] for r in R], float)
    edges = _bin_edges(lam_rows, 8)
    outcomes = {'log_lambda': [], 'n': [], 'kept': [], 'merged': [], 'split': [], 'dropped': []}
    sl_knots, sl_vals, os_vals = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [r for r in R if a <= np.log(r[0]) < b and not r[-1]]
        if len(sel) < 30:
            continue
        outcomes['log_lambda'].append(float(np.median([np.log(r[0]) for r in sel])))
        outcomes['n'].append(len(sel))
        for o in ('kept', 'merged', 'split', 'dropped'):
            outcomes[o].append(float(np.mean([r[1] == o for r in sel])))
        kept = np.array([(r[3], r[4], r[5]) for r in sel if r[1] == 'kept'], float)
        if len(kept) >= 20:
            shifts = np.concatenate([kept[:, 0], kept[:, 1]])
            sl_knots.append(outcomes['log_lambda'][-1])
            sl_vals.append(edge_sigma(shifts))
            os_vals.append(float(np.median(kept[:, 2])))
    if sl_knots:
        cal.sigma_loc = {'log_lambda': sl_knots, 'sigma': sl_vals}
        cal.overshoot = {'log_lambda': sl_knots, 'shift': os_vals}
    cal.outcomes = outcomes
    # --- curves: natural (same caller) vs training and held-out thinning ---------
    ft_nat = footprint_table(nat_curve_mols, cal, params)
    kn = ~ft_nat['truncated']
    curve_nat = fit_reference_curve(ft_nat['length'][kn], ft_nat['lambda'][kn], cal.nrl_pooled, params)
    out_val = {'natural_recalled': curve_nat.to_dict()}
    curves = {}
    for part in ('train', 'hold'):
        if thin_ft[part][0]:
            Lp = np.concatenate(thin_ft[part][0])
            lp = np.concatenate(thin_ft[part][1])
            curves[part] = fit_reference_curve(Lp, lp, cal.nrl_pooled, params)
            out_val[f'thinned_{part}'] = curves[part].to_dict()
    lam_nat = ft_nat['lambda'][kn]
    lam_thin = np.concatenate([np.concatenate(thin_ft[p][1]) for p in thin_ft if thin_ft[p][1]]) if any(thin_ft[p][1] for p in thin_ft) else np.zeros(0)
    if len(lam_thin) and len(lam_nat):
        dense_lams = np.concatenate([footprint_table([recalled[('nat', i, 0, 0)]], cal, params)['lambda']
                                     for i in dense_idx if ('nat', i, 0, 0) in recalled])
        cal.supported = supported_range(lam_nat, lam_thin, dense_lams)
    if 'train' in curves and cal.supported:
        grid = np.exp(np.linspace(np.log(cal.supported[0]), np.log(cal.supported[1]), 8))
        out_val['lambda_grid'] = grid.tolist()
        out_val['train_minus_natural_bp'] = _curve_mismatch(curves['train'], curve_nat, grid)
        if 'hold' in curves:
            out_val['holdout_minus_train_bp'] = _curve_mismatch(curves['hold'], curves['train'], grid)
            out_val['holdout_minus_natural_bp'] = _curve_mismatch(curves['hold'], curve_nat, grid)
        out_val['stored_minus_recalled_natural_bp'] = _curve_mismatch(cal.curve, curve_nat, grid)
    cal.validation = out_val
    # --- low-information floor (D8) on held-out thinned molecules ---------------
    from fiberhmm.nuc.molecule import footprint_calls
    ct_rows = []
    for key, tm in recalled.items():
        if key[0] != 'thin' or key[1] not in hold:
            continue
        fp = footprint_calls(tm, params, cal)
        if fp is None:
            continue
        c = fp['coarse'][~fp['is_tf']]
        c = c[c != EDGE_TRUNCATED]
        if len(c) == 0:
            continue
        ct_rows.append((fp['lambda_acc'], float((c == CANT_TELL).mean()), len(c)))
    floor = None
    floor_curve = {'log_lambda': [], 'cant_tell': [], 'n': []}
    if len(ct_rows) >= 30:
        la = np.array([r[0] for r in ct_rows])
        ct = np.array([r[1] for r in ct_rows])
        e2 = _bin_edges(la, 8)
        for a, b in zip(e2[:-1], e2[1:]):
            s = (np.log(la) >= a) & (np.log(la) < b)
            if s.sum() >= 5:
                floor_curve['log_lambda'].append(float(np.median(np.log(la[s]))))
                floor_curve['cant_tell'].append(float(np.mean(ct[s])))
                floor_curve['n'].append(int(s.sum()))
        xs, ys = np.array(floor_curve['log_lambda']), np.array(floor_curve['cant_tell'])
        above = ys > 0.5
        if above.any() and not above.all():
            # highest-lambda crossing from >0.5 (sparse) to <=0.5 (dense)
            k = int(np.max(np.flatnonzero(above)))
            if k + 1 < len(xs):
                x0, x1, y0, y1 = xs[k], xs[k + 1], ys[k], ys[k + 1]
                floor = float(np.exp(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)))
        elif above.all():
            floor = float(np.exp(xs.max()))
    cal.lambda_floor = floor
    cal.validation['cant_tell_by_lambda_acc'] = floor_curve
    # --- composition with the final calibration ------------------------------------
    from fiberhmm.nuc.segment import fit_composition
    cal.composition = fit_composition(mols, cal, params)
    cal.notes.append(
        f'thinning: {len(dense_idx)} dense molecules (top lambda_acc quartile >= {q75:.3f}), '
        f'{len(hold)} held out, {len(targets)} targets x {seeds} seeds, {len(tasks)} caller runs, '
        f'{len(errors)} caller errors, {t_call:.0f} s on {cores} cores')
    cal.validation['population_strata_lambda_acc'] = [float(x) for x in strata_edges]
    cal.validation['decode'] = stats
    cal.validation['caller_seconds'] = t_call
    cal.validation['caller_runs'] = len(tasks)
    cal.validation['caller_errors'] = len(errors)
    return cal


# ---------------------------------------------------------------------------
# absolute controls (NUC 4.2)
# ---------------------------------------------------------------------------

DEFAULT_T = {'small': 70.0, 'hexasome': 110.0, 'octasome': 147.0, 'chromatosome': 167.0}


def calibrate_absolute(base: NucCalibration, control_molecules: Sequence[NucMolecule],
                       control_class: str = 'octasome', T: Optional[Dict[str, float]] = None,
                       params: NucParams = DEFAULT_PARAMS, sigma_struct: float = 5.0,
                       naked_molecules: Sequence[NucMolecule] = ()) -> NucCalibration:
    """Absolute calibration from a control of known core length.

    ``control_molecules`` carry footprints of one known class (601 arrays:
    octasomes, T = 147 bp). The bias b(lambda) = observed mode - T is fitted
    on the control alone (monotone, over log lambda) and shared by every
    class: mu_k(lambda) = T_k + b(lambda), sigma_k^2 = sigma_struct^2 +
    sigma_b(lambda)^2. Class locations are never fitted jointly with the
    weights. ``naked_molecules`` (naked DNA) give the false-footprint length
    distribution per lambda bin, the null for short calls.
    """
    T = dict(DEFAULT_T if T is None else T)
    if control_class not in T:
        raise ValueError(f'control class {control_class!r} has no T')
    ft = footprint_table(control_molecules, base, params)
    keep = ~ft['truncated']
    nrl = base.nrl_pooled
    curve = fit_reference_curve(ft['length'][keep], ft['lambda'][keep], nrl, params)
    b = [m - T[control_class] for m in curve.mu]
    cal = NucCalibration.from_dict(base.to_dict())
    cal.source = 'absolute'
    cal.absolute = AbsoluteClasses(T=T, b_log_lambda=list(curve.log_lambda), b=b,
                                   sigma_b=list(curve.sigma), sigma_struct={k: sigma_struct for k in T})
    if naked_molecules:
        ftn = footprint_table(naked_molecules, base, params)
        Ln, ln = ftn['length'], ftn['lambda']
        total_bp = sum(m.length for m in naked_molecules)
        e = _bin_edges(ln, 4) if len(ln) >= 40 else np.array([-np.inf, np.inf])
        rows = []
        for a, c in zip(e[:-1], e[1:]):
            s = (np.log(np.maximum(ln, 1e-9)) >= a) & (np.log(np.maximum(ln, 1e-9)) < c)
            if s.sum():
                rows.append({'log_lambda': float(np.median(np.log(ln[s]))), 'n': int(s.sum()),
                             'length_q': [float(x) for x in np.percentile(Ln[s], [10, 50, 90])]})
        cal.naked = {'footprints_per_kb': float(len(Ln) / max(total_bp, 1) * 1000), 'bins': rows}
    cal.notes.append(f'absolute: control class {control_class} (T = {T[control_class]:g} bp), '
                     f'{int(keep.sum())} control footprints')
    return cal
