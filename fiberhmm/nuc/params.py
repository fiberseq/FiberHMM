"""Parameters of the nucleosome module (``fiberhmm.nuc``).

One frozen dataclass holds every setting that can change an output row. Its
``digest()`` goes into calibration records and, later, scan unit keys, so two
runs with the same parameters, calibration and inputs produce the same rows.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Optional, Tuple


@dataclass(frozen=True)
class NucParams:
    # --- evidence -----------------------------------------------------------
    #: ML threshold for a mark; None = the chemistry default (128, Hia5 ONT 248).
    mark_threshold: Optional[int] = None
    #: observation mode; None = from the BAM's chemistry declaration.
    mode: Optional[str] = None
    #: bases masked at each read end by the encoder (as the caller does).
    edge_trim: int = 10

    # --- density (NUC 3.1) ----------------------------------------------------
    #: MSP sites dropped at each MSP end when measuring lambda_acc: the caller
    #: places MSP boundaries on marks, so boundary sites are marks by
    #: construction and would inflate the rate.
    msp_boundary_sites: int = 1
    #: flank (bp) around a footprint for its local lambda_i.
    lambda_local_flank: int = 300
    #: lambda_i falls back to lambda_acc below this many informative sites.
    lambda_local_min_sites: int = 20
    #: bp outside each footprint edge counted for edge informativeness.
    edge_window: int = 20
    #: anchor +/- this many bp is excluded from lambda_acc when conditioning.
    anchor_exclusion_bp: int = 2000

    # --- dyads and spacing (NUC 3.4) -----------------------------------------
    #: bootstrap mono range (bp) used before NRL_pooled is known.
    bootstrap_mono_bp: Tuple[int, int] = (90, 230)
    #: mono range as fractions of NRL_pooled once it is known.
    mono_frac: Tuple[float, float] = (0.45, 1.25)
    #: MSPs longer than this break gaps and segments (NFR module territory).
    long_msp_bp: int = 300
    min_gaps_nrl: int = 4
    min_gaps_regularity: int = 6
    snr_min_bp: int = 2000
    snr_min_dyads: int = 8
    acf_halfwidth: int = 10
    acf_lag: Tuple[int, int] = (150, 260)
    snr_band_bp: Tuple[int, int] = (100, 400)

    # --- phasing (NUC 3.5) ---------------------------------------------------
    phase_x0: int = 150
    phase_bin_periods: int = 2
    phase_summary_periods: int = 5
    phase_n_null: int = 200
    phase_period_window: int = 10
    theta_min_r: float = 0.3
    theta_min_dyads: int = 4
    #: molecule frame: an MSP of at least this length covering anchor +/- slop.
    barrier_min_msp: int = 80
    barrier_slop: int = 30

    # --- sizes (NUC 3.2) -----------------------------------------------------
    z_short: float = -2.0
    z_long: float = 2.0
    coarse_min_posterior: float = 0.8
    named_min_posterior: float = 0.7
    min_edge_sites: int = 3
    #: knots of the reference curve over log(lambda).
    curve_knots: int = 8
    curve_min_per_bin: int = 200

    # --- window mode (NUC 3.3, 3.7) ------------------------------------------
    window_min_cover: float = 0.8
    window_insufficient_frac: float = 0.5

    # --- segmentation (NUC 3.6) ----------------------------------------------
    seg_q_grid: Tuple[int, ...] = tuple(range(165, 216, 5))
    #: regular-state gap SD (bp); None = the calibration's gap model
    seg_sigma_r: Optional[float] = None
    #: structural gap SD of a second, tight regular state (the design's 8 bp);
    #: None = one regular state only
    seg_sigma_tight: Optional[float] = 8.0
    #: weights of gaps spanning 1, 2, 3 repeats (missing dyads); None = calibration
    seg_grid_weights: Optional[Tuple[float, ...]] = None
    #: irregular-state gap density: 'broad' (the gap model's broad component)
    #: or 'empirical' (every gap of the dataset, NUC 3.6 as first written)
    seg_irregular: str = 'broad'
    #: whether a can't-tell footprint flags its step low-information
    seg_cant_tell_low_info: bool = False
    #: per-step order decision: 'threshold' (default; regular where the
    #: posterior P(regular), summed over repeat and tightness, is at least
    #: ``seg_posterior_min``, else the likelier of disordered / low-information),
    #: 'viterbi' (the Viterbi path) or 'posterior' (the class with the highest
    #: marginal, i.e. P(regular) >= 0.5)
    seg_decode: str = 'threshold'
    seg_posterior_min: float = 0.7
    #: the model-free regular rule, OR-ed with the decoder: runs of at least
    #: ``seg_rule_min_spacings`` consecutive eligible mono-dyad spacings each
    #: within ``seg_rule_tol_bp`` of k x the molecule's NRL (k = 1 ..
    #: ``seg_rule_max_k``) are regular from their first to their last dyad
    seg_rule: bool = True
    seg_rule_tol_bp: float = 25.0
    seg_rule_min_spacings: int = 6
    seg_rule_max_k: int = 2
    #: a run is disordered only when its spacing evidence beats every regular
    #: repeat by this many nats; weaker runs are undetermined
    seg_disorder_llr: float = 3.0
    seg_min_steps_regular: int = 5
    seg_min_steps_sub: int = 5
    seg_min_steps_irregular: int = 3
    #: probability per step of an order or composition switch (design 1/200;
    #: 1/50 from planted-array sensitivity at realistic densities)
    seg_switch_prob: float = 1.0 / 50.0
    seg_dq_max: int = 10
    seg_merge_llr: float = 3.0
    seg_min_dyads: int = 5
    seg_min_bp: int = 800
    seg_min_molecule_bp: int = 2000
    seg_composition_flank: int = 400
    seg_max_gap: int = 300

    # --- disrupted nucleosomes (NUC 13.1) -----------------------------------
    #: a footprint (nucleosome call, or TF call of at least
    #: ``undersized_min_bp``) is undersized at density-matched z <= this;
    #: shorter TF calls (linker-sized; recall turns footprints under 85 bp
    #: into TF calls) never count
    undersized_z: float = -2.0
    undersized_min_bp: int = 60
    #: undersized rate inside a run (the alternative of the run score); at
    #: least twice the density-matched background
    undersized_p1: float = 0.5
    #: minimum log-likelihood-ratio score (nats) of an undersized run and its
    #: minimum count of undersized footprints (7 nats: about one chance run
    #: per 8,000 footprints, the same for background rates 0.05 to 0.45)
    undersized_min_llr: float = 7.0
    undersized_min_count: int = 2
    #: missing nucleosome: open (MSP) stretch length between two nucleosome
    #: calls whose dyads sit about two repeats apart
    missing_len_bp: Tuple[int, int] = (120, 300)
    missing_grid_tol_bp: float = 30.0
    #: accessible-versus-protected log-likelihood ratio of the stretch's marks
    missing_min_llr: float = 3.0

    extra: dict = field(default_factory=dict, compare=False, hash=False)

    def to_dict(self) -> dict:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, tuple):
                out[key] = list(value)
        return out

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "NucParams":
        if not data:
            return cls()
        known = {f.name: f for f in fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key not in known:
                raise ValueError(f"unknown nuc parameter: {key!r}")
            if isinstance(value, list):
                value = tuple(value)
            kwargs[key] = value
        return cls(**kwargs)

    def digest(self) -> str:
        payload = {k: v for k, v in self.to_dict().items() if k != 'extra'}
        blob = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(blob.encode()).hexdigest()


DEFAULT_PARAMS = NucParams()


def parameter_schema() -> dict:
    """Every parameter with its default, as JSON-ready data."""
    return {'module': 'nuc', 'parameters': DEFAULT_PARAMS.to_dict()}
