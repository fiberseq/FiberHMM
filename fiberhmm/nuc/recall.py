"""The dataset's caller, in process, for thinning and prior checks.

:class:`CallerSettings` reconstructs a ``fiberhmm-call`` configuration from a
BAM header (the fiberhmm-call @PG command line and its DS ``phase_nrl``),
falling back to the current defaults. :class:`Caller` runs exactly the fused
path a region worker runs (``run_hmm_apply_stage`` then
``build_fused_recall_result``, recall included), with two calibration knobs:

``thin``
    a mark-retention factor f. Thinning drops marks whatever the state, so
    P(hit | context) is multiplied by f in **both** HMM states and the
    removed mass moves to the miss code of the same context (the context
    marginal is unchanged). The apply model and the recall/nucleosome LLR
    tables are rebuilt from the rescaled table; the preset emission uplift is
    applied after the rescale, as the CLI applies it to the bundled table.
``phase_nrl``
    the Pass-2 phase prior (0 = off), for the prior-off check.
"""
from __future__ import annotations

import copy
import re
import shlex
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

import numpy as np

N_CTX = 4096
UNMETH_OFFSET = 4097


@dataclass(frozen=True)
class CallerSettings:
    enzyme: str = 'hia5'
    seq: Optional[str] = 'pacbio'
    mode: str = 'pacbio-fiber'
    apply_model: Optional[str] = None
    recall_model: Optional[str] = None
    nuc_model: Optional[str] = None
    nuc_profile: Optional[str] = None
    min_llr: float = 5.0
    min_opps: int = 3
    unify_threshold: int = 90
    emission_uplift: float = 1.0
    recall_nucs: bool = True
    split_min_llr: float = 4.0
    split_min_opps: int = 3
    nuc_recall_policy: str = 'conservative'
    nuc_min_size: int = 85
    msp_min_size: int = 0
    phase_nrl: int = 0
    edge_trim: int = 10
    prob_threshold: int = 128
    derived_tf_max_edge_ambiguity: Optional[int] = None
    source: str = 'defaults'
    notes: tuple = field(default_factory=tuple)

    def to_dict(self):
        d = asdict(self)
        d['notes'] = list(self.notes)
        return d

    @classmethod
    def from_header(cls, header, **overrides) -> "CallerSettings":
        """Settings of the header's last fiberhmm-call run; bundled models for
        the recorded enzyme (a custom ``-m`` path is not reused: it is not
        portable; noted in ``notes``)."""
        from fiberhmm.inference.tf_recaller import ENZYME_PRESETS
        from fiberhmm.models import get_model_path, get_observation_mode, resolve_prob_threshold
        hd = header.to_dict() if hasattr(header, 'to_dict') else dict(header or {})
        pg = None
        for rec in reversed(hd.get('PG', [])):
            if 'fiberhmm-call' in str(rec.get('PN') or rec.get('ID') or ''):
                pg = rec
                break
        kw: dict = {}
        notes = []
        if pg is not None:
            toks = shlex.split(str(pg.get('CL', '')))

            def opt(*names):
                for i, t in enumerate(toks):
                    for n in names:
                        if t == n and i + 1 < len(toks):
                            return toks[i + 1]
                        if t.startswith(n + '='):
                            return t.split('=', 1)[1]
                return None
            enzyme = opt('--enzyme')
            seq = opt('--seq')
            if enzyme:
                kw['enzyme'] = enzyme.lower()
            if seq:
                kw['seq'] = seq.lower()
            for flag, key, typ in (('--min-llr', 'min_llr', float), ('--min-opps', 'min_opps', int),
                                   ('--unify-threshold', 'unify_threshold', int),
                                   ('--emission-uplift', 'emission_uplift', float),
                                   ('--split-min-llr', 'split_min_llr', float),
                                   ('--split-min-opps', 'split_min_opps', int),
                                   ('--nuc-min-size', 'nuc_min_size', int),
                                   ('--msp-min-size', 'msp_min_size', int),
                                   ('--edge-trim', 'edge_trim', int),
                                   ('--prob-threshold', 'prob_threshold', int)):
                v = opt(flag)
                if v is not None:
                    kw[key] = typ(v)
            if '--no-recall-nucs' in toks:
                kw['recall_nucs'] = False
            if opt('-m', '--model'):
                notes.append('custom apply model in the header was not reused; bundled model used')
            m = re.search(r'phase_nrl=(\d+)', str(pg.get('DS', '')))
            if m:
                kw['phase_nrl'] = int(m.group(1))
            kw['source'] = 'header'
        kw.update(overrides)
        base = cls(**{k: v for k, v in kw.items() if k in cls.__dataclass_fields__})
        enzyme, seq = base.enzyme, base.seq
        preset = ENZYME_PRESETS.get(enzyme, {})
        upd = {}
        if 'min_llr' not in kw:
            upd['min_llr'] = float(preset.get('min_llr', 5.0))
        if 'emission_uplift' not in kw:
            upd['emission_uplift'] = float(preset.get('emission_uplift', 1.0))
        if 'prob_threshold' not in kw:
            upd['prob_threshold'] = int(resolve_prob_threshold(None, enzyme, seq))
        upd['mode'] = get_observation_mode(enzyme, seq, warn_missing_seq=False)
        upd['apply_model'] = base.apply_model or get_model_path(enzyme, tool='apply', seq=seq)
        try:
            upd['recall_model'] = base.recall_model or get_model_path(enzyme, tool='recall', seq=seq)
        except Exception:
            upd['recall_model'] = upd['apply_model']
        if enzyme == 'ddda' and base.recall_nucs:
            from fiberhmm.models import _bundled_model_path
            upd['nuc_profile'] = base.nuc_profile or _bundled_model_path('ddda_nuc_profile.json')
            upd['nuc_model'] = base.nuc_model or get_model_path(enzyme, tool='nuc_refine', seq=seq)
            if base.derived_tf_max_edge_ambiguity is None:
                upd['derived_tf_max_edge_ambiguity'] = 12
        upd['notes'] = tuple(notes)
        return replace(base, **upd)


def rescale_emissions(model, f: float):
    """Copy of ``model`` with P(hit | context) multiplied by ``f`` in every
    state; the removed probability moves to the same context's miss code."""
    if not (0.0 < f <= 1.0):
        raise ValueError('thinning factor must be in (0, 1]')
    out = copy.deepcopy(model)
    if f == 1.0:
        return out
    EP = np.array(out.emissionprob_, dtype=float)
    hit = EP[:, :N_CTX].copy()
    EP[:, :N_CTX] = f * hit
    EP[:, UNMETH_OFFSET:UNMETH_OFFSET + N_CTX] += (1.0 - f) * hit
    out.emissionprob_ = EP
    for attr in ('_log_emissionprob', '_frozen_log_emission', '_log_emit'):
        if hasattr(out, attr):
            try:
                delattr(out, attr)
            except AttributeError:
                pass
    return out


class Caller:
    """Fused apply + recall on one fiber_read dict, as a region worker does."""

    def __init__(self, settings: CallerSettings, thin: float = 1.0):
        from fiberhmm.core.model_io import (freeze_model_for_inference, load_model,
                                            load_model_with_metadata)
        from fiberhmm.inference.tf_recaller import (apply_emission_uplift,
                                                    build_conditional_hit_tables,
                                                    build_llr_tables)
        self.settings = settings
        self.thin = float(thin)
        s = settings
        apply_raw = load_model(s.apply_model)
        _, self.context_size, _ = load_model_with_metadata(s.apply_model)
        self.model = freeze_model_for_inference(rescale_emissions(apply_raw, self.thin))
        r_model, _, _ = load_model_with_metadata(s.recall_model or s.apply_model)
        r_model = rescale_emissions(r_model, self.thin)
        llr_hit, llr_miss = build_llr_tables(r_model)
        if abs(s.emission_uplift - 1.0) > 1e-9:
            llr_hit, llr_miss = apply_emission_uplift(llr_hit, llr_miss, r_model, s.emission_uplift)
        self.llr_hit, self.llr_miss = llr_hit, llr_miss
        n_model = r_model
        nuc_uplift = s.emission_uplift
        if s.nuc_model:
            n_model, _, _ = load_model_with_metadata(s.nuc_model)
            n_model = rescale_emissions(n_model, self.thin)
            nuc_uplift = 1.0
        nh, nm = build_llr_tables(n_model)
        if abs(nuc_uplift - 1.0) > 1e-9:
            nh, nm = apply_emission_uplift(nh, nm, n_model, nuc_uplift)
        self.nuc_llr_hit, self.nuc_llr_miss = nh, nm
        self.nuc_profile = None
        if s.nuc_profile:
            from fiberhmm.inference.nuc_recaller import (attach_nuc_profile_emissions,
                                                         load_nuc_profile)
            prot, acc = build_conditional_hit_tables(n_model, emission_uplift=nuc_uplift)
            self.nuc_profile = attach_nuc_profile_emissions(load_nuc_profile(s.nuc_profile), prot, acc)

    def call(self, fiber_read: dict) -> Optional[dict]:
        """Calls for one read in SEQ frame: ``{'nucs', 'msps', 'tfs'}`` as
        ``[(start, end)]`` lists, or None when the HMM finds nothing."""
        from fiberhmm.inference.fused_stages import (apply_result_has_footprints,
                                                     build_fused_recall_result,
                                                     run_hmm_apply_stage)
        s = self.settings
        ar = run_hmm_apply_stage(fiber_read, self.model, s.edge_trim, False, s.mode,
                                 self.context_size, s.msp_min_size, s.nuc_min_size, False)
        if not apply_result_has_footprints(ar):
            return None
        res = build_fused_recall_result(
            fiber_read, ar, self.llr_hit, self.llr_miss, s.min_llr, s.min_opps,
            s.unify_threshold, False, recall_nucs=s.recall_nucs,
            split_min_llr=s.split_min_llr, split_min_opps=s.split_min_opps,
            nuc_recall_policy=s.nuc_recall_policy, nuc_min_size=s.nuc_min_size,
            msp_min_size=s.msp_min_size, phase_nrl=s.phase_nrl, nuc_profile=self.nuc_profile,
            derived_tf_max_edge_ambiguity=s.derived_tf_max_edge_ambiguity,
            nuc_llr_hit=self.nuc_llr_hit, nuc_llr_miss=self.nuc_llr_miss)
        nucs = [(int(a), int(a) + int(b)) for a, b in zip(res['ns'], res['nl'])]
        msps = [(int(a), int(a) + int(b)) for a, b in zip(res['as'], res['al'])]
        tfs = [(int(c.start), int(c.start) + int(c.length)) for c in res.get('tf_calls', [])]
        return {'nucs': nucs, 'msps': msps, 'tfs': tfs}


def fiber_read_from_record(read, settings: CallerSettings) -> Optional[dict]:
    """The caller's input for one pysam record (None if no usable evidence)."""
    from fiberhmm.inference.engine import CHIMERA_SKIP, _extract_fiber_read_from_pysam
    fr = _extract_fiber_read_from_pysam(read, settings.mode, settings.prob_threshold)
    if fr is None or fr is CHIMERA_SKIP:
        return None
    return fr


def thin_fiber_read(fr: dict, keep: float, rng: np.random.Generator) -> dict:
    """Copy of ``fr`` with each mark kept with probability ``keep``."""
    out = dict(fr)
    pos = np.fromiter(fr['m6a_query_positions'], dtype=np.int64)
    pos.sort()
    kept = pos[rng.random(len(pos)) < keep]
    out['m6a_query_positions'] = set(int(p) for p in kept)
    return out
