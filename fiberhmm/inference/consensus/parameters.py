"""One parameter contract shared by the CLI and Browser Footprint panel.

Inference parameters deliberately exclude display thresholds. Nothing here caps
the number of reads, families, or hits; compute limits fail explicitly instead.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import math


def control(default, label, help, minimum=None, maximum=None, step=None, choices=None):
    return field(default=default, metadata=dict(label=label, help=help,
        minimum=minimum, maximum=maximum, step=step, choices=choices))


@dataclass
class InputOptions:
    legacy_hia5_annotation_frame: str = control('disabled', 'Legacy Hia5 MSP/nucleosome tag frame', 'Only for BAMs without MA/Ma: explicitly declare whether as/al and ns/nl coordinates use stored SEQ or molecular orientation. Disabled uses molecular when the header shows fibertools wrote the tags (a fibertools nucleosome command with no later FiberHMM caller), otherwise stops; it never guesses or flips REV from the data. The verified ind 2–4 h input uses seq.', choices=['disabled','seq','molecular'])
    correct_native: bool = control(True, "Replay corrected native TF decoder", "Re-score fixed MSPs with the multi-interval decoder. Original BAM calls remain untouched in the native layer.")
    ddda_m5c_correction: bool = control(True, "Honor DddA mCG substrate annotations", "Use the production DddA mCG rate correction at tagged CpGs, in native replay and every downstream lattice score. Untagged observations are unchanged.")
    minimum_mapq: int = control(20, "Minimum alignment MAPQ", "Input evidence filter, applied before within-strand duplicate collapse.", 0, 255, 1)
    minimum_nfr_length: int = control(0, "Minimum MSP length (bp; 0 = all)", "Search every supplied MSP by default. The native LLR and opportunity minimum, not a 150-bp accessible-patch gate, determine detection. A positive value explicitly restricts the search.", 0, 100000, 1)
    native_minimum_opportunities: int = control(3, "Native decoder minimum opportunities", "Actual observed opportunities required for corrected native seeds; independent of the CR and strict-rescue gates.", 1, 100, 1)
    native_maximum_alignment_gap_bp: int = control(0, "Native call maximum alignment gap (bp)", "Largest total length of unaligned reference (deletions or skips) a corrected native call may span. 0 keeps the frozen rule that discards any call crossing an alignment gap, under which a one-base deletion inside a footprint vetoes it regardless of evidence; the bases inside a permitted gap contribute nothing and the call records the gap.", 0, 1000, 1)
    ddda_minimum_llr: float = control(-1., "DddA native LLR (−1 = model preset)", "Native nomination threshold in nats, not the CR display Q. Changing this reruns the native decoder.", -1, 10000, .5)
    dddb_minimum_llr: float = control(-1., "DddB native LLR (−1 = model preset)", "Native nomination threshold in nats, not the CR display Q.", -1, 10000, .5)
    hia5_pacbio_minimum_llr: float = control(-1., "Hia5 PacBio native LLR (−1 = preset)", "Native nomination threshold in nats. This is not a scan-length-calibrated false discovery rate.", -1, 10000, .5)
    hia5_nanopore_minimum_llr: float = control(-1., "Hia5 ONT native LLR (−1 = preset)", "Uses the ONT native emission model; a user threshold cannot add unobserved modification opportunities.", -1, 10000, .5)
    molecule_min_jaccard: float = control(.95, "Same-strand duplicate Jaccard", "Collapse amplification copies within a chemical strand. Does not pair complementary strands.", 0, 1, .01)
    molecule_min_deam: int = control(10, "Minimum duplicate fingerprint hits", "Below this number, do not assert amplification identity from a sparse pattern.", 1, 100000, 1)


@dataclass
class SROptions:
    enabled: bool = control(True, "Normalize boundaries with SR", "Current harmonization changes only measurement-equivalent native edges. Historical fitted engines retain their original strand-specific rules.")
    loss_odds: float = control(3., "Maximum native edge-loss odds", "Each edge and their joint change must pass this native-likelihood loss budget.", 1, 10000, 1)
    projection_only: bool = control(False, "Only projection-equivalent changes", "Do not change which recipient opportunities are protected.")
    minimum_source_support: int = control(3, "Minimum source units", "Independent within-strand evidence units supporting a source boundary cell.", 1, 100000, 1)
    minimum_projection_mass: float = control(.8, "Minimum projection rank mass", "Required only when changing the recipient's observed projection; not calibrated accuracy.", 0, 1, .01)
    maximum_diffuse_odds: float = control(100., "Maximum diffuse edge odds", "Reject a proposed edge if the diffuse alternative exceeds both endpoint models by these odds.", 1, 1000000, 1)


@dataclass
class CROptions:
    engine: str = control('lattice_recaller', 'Consensus algorithm', 'Lattice recaller (the default) finds class geometries by k-means on confident calls and scores every molecule\'s lattice against them with EM (no Monte Carlo). Staged native families (deprecated) uses lattice-aware Monte Carlo fits and shared-geometry consolidation. Call harmonization is the separate lightweight overlap method. Saved historical engines remain available.', choices=['lattice_recaller', 'staged_native_families', 'call_harmonization', 'native_family_distribution', 'legacy_lattice'])
    harmonization_cut: float = control(.5, 'Native-call clustering distance', 'Average-linkage cut on overlap of the shorter footprint. Larger values merge more similar geometries.', 0, .99, .05)
    harmonization_fold_overlap: float = control(.7, 'Family folding overlap', 'Direct non-transitive overlap required to fold a recurrent family into an anchor.', .01, 1, .05)
    harmonization_max_call_bp: int = control(100, 'Maximum class footprint length (bp)', 'Longer native calls remain visible but are not used to nominate TF-sized classes.', 1, 1000, 1)
    harmonization_edge_tolerance_bp: int = control(10, 'Class edge harmonization allowance (bp)', 'Bounds stratum-specific class edges. Individual source calls remain immutable.', 0, 100, 1)
    enabled: bool = control(True, "Discover and assign CR classes", "Group existing native footprints into recurrent classes using all supplied units. Original calls are retained; no new-call rescue.")
    ambiguity_bp: int = control(10, "Molecular boundary variation (±bp/edge)", "Uniform integer variation around each class edge, integrated on each native opportunity lattice. Not SR measurement uncertainty.", 0, 100, 1)
    neighborhood_bandwidth: int = control(0, "Nomination bandwidth (bp; 0 = edge variation)", "Organize overlapping calls into local work neighborhoods; spans remain intact and compete across neighborhood boundaries.", 0, 200, 1)
    selection_se: float = control(1., "Complexity selection tolerance (SE)", "Fewest classes within this paired standard-error tolerance of the best internal predictive fit. Higher values favor broader grouping.", 0, 10, .25)
    predictive_loss_tolerance: float = control(0., "Coarse grouping loss allowance (nats/call-bearing unit)", "Method-independent granularity: tolerate this native predictive loss per validation unit with an upstream TF in the fixed neighborhood. Unlike the SE term, this allowance does not shrink with depth. Use the larger of this allowance and the SE tolerance; 0 preserves legacy selection. Not a confidence/FDR threshold.", 0, 20, .05)
    minimum_opportunities: int = control(3, "Minimum assignment opportunities", "A displayed boundary proposal requires this many actual observations, positive native evidence, and physical eligibility.", 1, 100, 1)
    gmm_regularization: float = control(1., "Endpoint mixture covariance regularization", "Regularizes GMM nomination only; native modification likelihood decides complexity and assignments.", .000001, 1000, .25)
    gmm_restarts: int = control(4, "Endpoint mixture restarts", "Deterministic initialization restarts for each tested complexity.", 1, 100, 1)
    gmm_iterations: int = control(400, "Endpoint mixture iterations", "Numerical iteration budget, not a family-count cap.", 1, 10000, 1)
    local_iterations: int = control(50, "Local likelihood fit iterations", "Iteration budget for native activity fitting during nomination.", 1, 10000, 1)
    global_iterations: int = control(800, "Whole-region fit iterations", "Fit normalized configuration activities using every evidence unit. Stops early at convergence; the 800-iteration default covers the measured full NAPA/UBA1/ind fits. Unconverged results remain explicitly provisional.", 1, 10000, 1)
    seed: int = control(20260905, "Nomination seed", "Deterministic endpoint mixture seed; included in the run receipt.", 0, 2147483647, 1)
    minimum_edge_tolerance_bp: int = control(-1, 'Additional minimum edge tolerance (bp; −1 = chemistry default)', 'Additional to inherent native lattice/probability variability: DddA 10 bp, Hia5/DddB 0 bp. Never a maximum search radius or substituted modification probability.', -1, 100, 1)
    edge_tolerance_mode: str = control('legacy_profile', 'Fitted CR edge allowance', 'Bounded uses a joint endpoint-cell profile within the stated bp allowance. Legacy profile retains the historical reference-distance trigger for unrestricted single-edge profiling. Simulation uses the same score as observations; neither mode alters fitted geometry or generative mass.', choices=['legacy_profile', 'bounded'])
    family_fit_iterations: int = control(100, 'Native shape fit iterations', 'Independent group-excluded latent distribution fits. Nonconvergence remains explicitly flagged.', 1, 10000, 1)
    scoring_folds: int = control(10, 'Native shape scoring folds', 'A whole evidence group is excluded from the model used to classify its own calls; nomination remains caller-conditioned.', 2, 100, 1)
    membership_loss_odds: float = control(1., 'Tie-set membership margin (loss odds)', '1 keeps winner-take-all nearest-centre membership. Above 1, a call is additionally a member of every predictively compatible family whose floor-adjusted loss is within ln(odds) of the best compatible loss. The primary family label never changes; this only widens cohort membership, and it is a versioned statistical change, not a display filter.', 1, 1000, .5)
    predictive_replicates: int = control(4095, 'Native CR predictive reference draws', 'Within-dataset classification reference simulations. Independent of XCR reference draws. At least 4095 draws are needed even to resolve a zero-exceedance upper bound below 0.001. This is not FDR.', 31, 65535, 1)
    residual_nomination: bool = control(True, 'Nominate unexplained existing calls', 'One frozen support-one residual proposal pass at 99.9 reference. Adds classes, never new footprints, and never reruns from a display slider.')
    residual_update_policy: str = control('refit_catalog', 'Residual catalog update policy', 'Refit catalog keeps the current behavior: refit all models after residual nomination. Append frozen preserves every initial model and score, appends only new nominated models, and recomputes downstream XCR/auxiliary results. Shared training observations remain explicit; neither policy creates new native footprints.', choices=['refit_catalog', 'append_frozen'])


@dataclass
class RescueOptions:
    enabled: bool = control(False, "Add strictly supported uncalled footprints", "Separate opt-in layer. Existing TFs and nucleosomes are obstacles. Fold-excluded source geometry must support each addition.")
    proposal_edge_radius_bp: int = control(10, "Strict auxiliary proposal radius (±bp/edge)", "Bounded proposal grid for the explicitly named strict-detection auxiliary only. Not native CR variation, assay resolution, or an SR tolerance. Every addition also needs an independently trained full-grid native shape check with zero extra bp allowance.", 0, 100, 1)
    source_mode: str = control("opposite_strand", "Rescue source population", "Opposite strand is meaningful only for strand-resolved chemistries; pooled enables within-dataset CR recall.", choices=["opposite_strand", "same_strand", "pooled"])
    model: str = control("legacy_strict", "Rescue evidence model", "Population-assisted fits joint recurrence from every source pattern and retains low-information hypotheses for display. Legacy strict reproduces the previous three-opportunity implementation.", choices=["legacy_strict", "population_assisted"])
    population_prior_mean: float = control(-2., "Population baseline log activity", "Mean of the proper Gaussian activity hyperprior; not log marginal occupancy. The joint configuration normalizer handles overlapping families.", -12, 12, .25)
    population_prior_sd: float = control(2., "Population activity prior SD", "Regularizes weak source families while retaining uncertain activities. Applies only to population-assisted rescue.", .1, 6, .1)
    population_transfer_sd: float = control(.25, "Cross-strand activity uncertainty", "Additional recipient log-activity SD around source activities; 0 assumes shared activities. It is separate from edge variability and display confidence.", 0, 3, .05)
    population_geometry_floor: float = control(.02, "Original geometry robustness fraction", "Mix this fraction of the original bounded joint geometry into the frozen learned source distribution; never alter native emissions.", 0, 1, .01)
    population_draws: int = control(512, "Population integration draws", "Power of two. Exact-target corrected population sampling with ESS/convergence diagnostics; Laplace preconditions the sampler. Not calibrated FDR.", 8, 4096, 8)
    population_iterations: int = control(250, "Population activity fit iterations", "Full-source joint fit; no source call-count threshold or read subsampling.", 1, 10000, 10)
    minimum_source_units: int = control(3, "Minimum source detections", "Recipient fold excluded; source detections must also have physical eligibility.", 1, 100000, 1)
    prior_scale: float = control(1., "Source activity scale", "Scales detection/eligible activity, not occupancy odds. Native evidence and adequacy veto remain mandatory.", .0001, 100, .1)
    minimum_protection_mass: float = control(.9, "Minimum new-call protection mass", "Candidate protection-event mass before the strict geometry check; not accuracy or FDR.", .01, .99999, .01)
    minimum_probability: float = control(.9, "Minimum learned-family rescue mass", "Family inclusion after fold-excluded geometry validation; separate from the initial protection-event mass.", .01, .99999, .01)
    credible_mass: float = control(.9, "Population geometry credible mass", "Candidate's actual recipient projection must be typical of source geometries.", .01, .99999, .01)
    loss_odds: float = control(3., "Maximum population geometry loss odds", "A generic arbitrary gap must not explain the candidate substantially better than its nominated family.", 1, 10000, 1)
    geometry_prior_units: float = control(1., "Geometry regularization units", "Regularization toward the original bounded geometry prior in each held-fold fit.", .001, 1000, .25)
    maximum_diffuse_odds: float = control(100., "Maximum diffuse core odds", "Recipient protected-core adequacy veto; never overridden by source depth.", 1, 1000000, 1)
    null_replicates: int = control(1, "Accessible-null replicates", "Conditional native-emission controls repeat candidate selection. These are not empirical FDR estimates.", 0, 10000, 1)


@dataclass
class CrossOptions:
    enabled: bool = control(False, "Build XCR relationship graph", "All plausible cross-dataset pairs; native nodes and both intervals stay unchanged. At least two datasets required.")
    coarse_native_events: bool = control(True, 'Report coarser ANY-child comparisons', 'Keep fine native classes and additionally report bounded, non-transitive broad-anchor groups. Count each molecule once, even if it carries multiple children. No detection is added.')
    lattice_capable_fraction: float = control(.8, 'Minimum lattice-capable fraction', 'Flag comparison as lattice-limited below this fraction of span-covered units able to reach the native threshold. Necessary reachability only, not sensitivity or a fit to observed agreement.', 0, 1, .05)
    lattice_minimum_covered_units: int = control(20, 'Minimum comparison coverage', 'Flag low-coverage comparisons; retain their counts and all native calls.', 1, 100000, 1)
    tolerance_odds: float = control(10., "Reciprocal shape likelihood tolerance", "Integrate foreign and native geometry using each recipient's measured emissions and lattice.", 1, 1000000, 1)
    minimum_support: float = control(3., "Minimum effective native support", "Interpretation threshold on frozen native membership; rescued calls cannot manufacture shared support.", 0, 100000, .5)
    minimum_fraction: float = control(.5, "Minimum reciprocal compatible fraction", "Required on both sides for shape compatibility; untestable is distinct from disagreement.", 0, 1, .01)
    joint_validation: str = control("compatible", "Full-model replacement tests", "Replace one geometry while retaining all native competitors and recomputing both partition functions. Unchecked edges stay provisional.", choices=["compatible", "all"])
    minimum_joint_fraction: float = control(.8, "Minimum full-model retained mass fraction", "Comparability mask additionally requires native-weighted full-model loss within the likelihood tolerance on both sides.", 0, 1, .01)
    minimum_target_retention: float = control(.8, "Minimum individual-family inclusion retention", "Unique-family correspondence additionally retains the target's inclusion on its original units. A shape-compatible edge can remain while its class identity is unresolved; coarse counts and native calls are not deleted.", 0, 1, .01)
    native_reference_percent: float = control(99.9, 'Native XCR predictive reference (%)', 'Reciprocal foreign-shape compatibility on recipient native observations; larger is more permissive. Does not force population rate agreement or calibrate FDR.', 50, 99.999, .1)
    agreement_summary: bool = control(False, 'Export the per-family agreement summary', 'Adds a per-dataset summary to the XCR graph: how many families had a cohort, how many reached a verdict, and how many matched. The per-pair ledger is unchanged. Off by default only so a default run stays byte-identical to the September 2026 frozen reference; it should become the default at the next schema bump.')
    minimum_node_source_units: int = control(0, 'XCR node minimum source units', 'Families fitted from fewer independent source units than this are not XCR nodes. 0 keeps the frozen behaviour. A family below cross.minimum_support can never pass the reciprocal cohort gate, so excluding it removes pairs that could never reach a verdict and makes the agreement denominators honest.', 0, 100, 1)
    pair_nomination_rule: str = control('all_overlapping_frozen_model_domains', 'XCR pair nomination rule', 'Which family pairs are assessed. The frozen rule assesses every pair whose model domains overlap, most of which are families at different places. reference_interval_gap assesses only pairs whose native reference intervals lie within pair_nomination_gap_bp of each other.', choices=['all_overlapping_frozen_model_domains', 'reference_interval_gap'])
    pair_nomination_gap_bp: int = control(20, 'XCR pair nomination gap (bp)', 'Used only by pair_nomination_rule=reference_interval_gap. On the September 2026 amplicon bundle 20 bp retained every compatible and every incompatible verdict while assessing 39% of the pairs.', 0, 10000, 1)
    native_predictive_replicates: int = control(4095, 'Native XCR predictive reference draws', 'Independent of native CR draws. More draws narrow Monte Carlo uncertainty, not biological uncertainty. The zero-exceedance upper bound must be below the selected XCR tail threshold. Changing this is an inference rerun, not a display filter; the current workflow also refits CR.', 31, 65535, 1)
    native_minimum_call_attribution_mass: float = control(.05, 'Minimum full geometry mass overlapping this call', 'Positive fraction of the original normalized donor geometry distribution overlapping the unchanged recipient call. Uses all geometry, not a majority core; a model located elsewhere cannot be attributed to this call.', 0, 1, .01)
    native_minimum_geometry_retention: float = control(.05, 'Minimum physically retained geometry mass', 'Positive fraction of original donor geometry surviving this call, frozen neighbors and domain constraints. Prevents renormalizing a tiny surviving tail without reporting its loss.', 0, 1, .01)
    native_minimum_visible_geometry_mass: float = control(.05, 'Minimum observable retained geometry mass', 'Positive fraction of the physically retained geometry with at least one actual recipient opportunity. Conditional on retained geometry; not a majority-core rule, detection power or missingness imputation.', 0, 1, .01)
    native_minimum_testable_fraction: float = control(.5, 'Minimum assessable source-call fraction', 'For each reciprocal direction, scored plus core-contradicted calls divided by all native primary calls. Below this fraction is incomplete assessment, not proof of sequence-resolution limitation.', 0, 1, .01)


@dataclass
class ComparabilityOptions:
    enabled: bool = control(False, "Score CT/GA comparability", "Conditional population comparison from native evidence only; no balancing or deletion of calls.")
    equivalence_margin: float = control(.10, "Allowed population fraction difference", "Absolute CT–GA margin for comparability, separate from existence and testability.", .001, .99999, .01)
    existence_floor: float = control(.01, "Population existence floor", "Both conditional fractions must exceed this floor for comparative support.", .000001, .99, .005)
    maximum_population_ci_width: float = control(.20, "Maximum population CI width", "Wider intervals yield untestable/underpowered, not low-quality absence.", .001, 1, .01)
    minimum_population_units: int = control(20, "Minimum informative units/strand", "Units must carry direct native information on the family.", 1, 100000, 1)
    quadrature_size: int = control(3072, "Population quadrature points", "Continuous spike-and-slab integration; 6144 can be used as a numerical convergence check.", 64, 32768, 64)
    simulation_units: int = control(8, "Power simulation contexts/strand", "Deterministic real opportunity contexts, balanced by fold; 0 skips power and leaves the molecule mask unavailable.", 0, 1000, 1)
    draws_per_state: int = control(8, "Power draws/state/context", "Exact whole-configuration simulations, not isolated-footprint simulations.", 1, 10000, 1)
    evidence_odds: float = control(20., "Molecule detection evidence odds", "Native include/exclude BF threshold in the model-power test.", 1.0001, 1000000, 1)
    minimum_detection_power: float = control(.8, "Minimum power lower bound", "Required 95% Monte Carlo lower bound for molecule-level comparative quantification.", .001, .99999, .01)
    q_cap: float = control(40., "Comparability model-Q ceiling", "Numerical display cap only. CQ is not a calibrated FDR q value.", 1, 120, 1)


@dataclass
class SplitOptions:
    enabled: bool = control(False, "Test CR-supported nucleosome splits", "Retain original outer spans. Compare intact spans with internal accessible separators and CR-like protected pieces.")
    ambiguity_bp: int = control(10, "CR matching edge allowance (±bp)", "Both ends of a proposed protected piece must match a supported CR class.", 0, 100, 1)
    maximum_gap_bp: int = control(30, "Maximum internal separator width", "No cap on the number of compatible separators or protected pieces.", 1, 1000, 1)
    minimum_source_units: int = control(3, "Minimum native CR source units", "Count native CR assignments, never rescue additions.", 1, 100000, 1)
    activity: float = control(.1, "Internal-gap activity", "Normalized prior over all nonoverlapping internal-gap configurations.", .000001, 1000, .1)
    minimum_split_bf: float = control(100., "Minimum split/intact BF", "Evidence for any allowed internal split, not evidence for a biological nucleosome identity.", 1, 1000000000, 1)
    minimum_separator_bf: float = control(10., "Minimum accessible separator LR", "Each selected gap must show native accessible-versus-protected evidence.", 1, 1000000, 1)
    minimum_separator_opportunities: int = control(3, "Minimum separator observations", "Actual observed opportunities, not base-pair length.", 1, 100, 1)
    require_nuc_model: bool = control(True, "Require DddA nucleosome-model agreement", "The same CR class must pass native TF and installed DddA nucleosome emission sensitivity tests.")
    null_replicates: int = control(1, "Intact-null replicates", "Simulate an intact span and repeat full split selection; model-based diagnostic only.", 0, 10000, 1)


@dataclass
class ComputeOptions:
    require_native_cache: bool = control(False, "Require saved native fits", "Start at consolidation: fail on a missing or incompatible native-fit checkpoint instead of fitting again.")
    predictive_backend: str = control('cpu', 'Predictive scoring backend (vectorized = counter-based generator, declared versioned kernel)', 'CPU reference or opt-in CUDA/MPS interval scoring. Same CPU random draws and probabilities; numerically borderline device results are rechecked on CPU. Unavailable devices fall back explicitly.', choices=['cpu', 'vectorized', 'cuda', 'mps', 'auto'])
    predictive_tilt: str = control('0', 'Vectorized predictive importance tilt', 'Only with predictive_backend=vectorized. 0 = plain Monte Carlo with the counter-based generator; a fraction in (0,1) mixes the generating-projection law with uniform; threshold:M spends draws on projections whose penalty could reach the observed loss within M nats. Unbiased for the same tail in every case; only the variance changes.')
    fit_backend: str = control('cpu', 'Native fitting objective / device', 'separable: same optimizer and data on a cache-resident factorized objective, several times faster on large families; not bit-identical to cpu (declared versioned fitter). nonparametric: penalized maximum-likelihood mass over the projection grid, smoothed toward its Gaussian seed by nonparametric_pseudo_units; monotone MM fit, unique from every start on the September 2026 bundle, +0.7 to +1.0 nats per unit held-out over the Gaussian; transferred by exact cell refinement. CUDA is validation-only: compare full GPU fits but always retain CPU models after a real-locus optimizer divergence. This is slower, not a production accelerator. Auto and MPS retain CPU fitting; MPS predictive scoring is separately supported.', choices=['cpu', 'separable', 'nonparametric', 'cuda', 'mps', 'auto'])
    nonparametric_pseudo_units: float = control(4., 'Nonparametric geometry smoothing (pseudo-units)', 'Only with fit_backend=nonparametric. The fitted projection mass is smoothed toward its Gaussian seed as if this many extra source units followed the seed; cells no source unit could produce keep the seed mass exactly. 4 is the held-out optimum on Hia5 (small families) and within 0.01 nats per unit of the best value on the largest DddA families. Bound in the result as part of the model identity.', 0, 1000, .5)
    accelerator_mb: int = control(256, 'Accelerator scratch budget (MiB)', 'Bounded per-workflow accelerator buffers; never reserves all GPU memory or reduces reads/draws. Oversized predictive experiments run on CPU.', 16, 16384, 16)
    maximum_region_bp: int = control(50000, "Maximum analysis span (bp)", "Explicit safety budget. A larger requested region fails, never silently truncates.", 1, 10000000, 1)
    maximum_matrix_mb: int = control(2048, "Observation/evidence matrix budget (MiB)", "Explicit allocation budget; all units are used or the run fails with an actionable message.", 16, 262144, 16)
    maximum_nodes: int = control(100000, "Maximum exact-DAG nodes", "Computational guard, not a cap on unique footprints. Exceeding it fails the run.", 100, 10000000, 100)
    maximum_edges: int = control(2000000, "Maximum exact-DAG edges", "Computational guard with no approximate fallback.", 100, 100000000, 100)
    neighborhood_seconds: float = control(300., "Seconds per adaptive neighborhood", "Abort with a visible incomplete result if nomination exceeds its budget; never drop later classes.", 1, 86400, 1)
    batch_size: int = control(24, "Evidence batch size", "Memory/performance control; must not change numerical results.", 1, 2048, 1)
    cores: int = control(4, "Numerical CPU budget", "Concurrency for exact numerical work: nomination threads, single-threaded native-fit processes, or predictive-scoring workers. Memory limits can reduce concurrency without dropping evidence.", 1, 128, 1)
    predictive_stopping: str = control('full', 'Shared-state predictive stopping', 'Staged families only. decision: a shared-state or foreign-state predictive run stops once it has enough exceedances to pass the families.assignment_reference_percent gate (one at 99.9%). The draws run are the identical prefix of the full experiment, so every compatible/rejected decision is identical to full; stored tails and intervals become lower bounds at the stop. Zero-exceedance rejections still use every draw. full: always run every draw. Native-stage scoring always runs every draw.', choices=['full', 'decision'])
    fit_cache_dir: str = control('', "Persistent native-fit cache directory (blank = off)", "Content-addressed reuse of identical native boundary-distribution fits across runs. Keys cover the complete fit inputs, fold rows, fitter source and numerical-library versions; a hit is bit-identical to a fresh fit. Execution only; no model or evidence changes.")


@dataclass
class FamilyStageOptions:
    assignment_reference_percent: float = control(99.9, 'Assignment compatibility reference (%)', '50 is strict; 99.9 accepts any state compatible under the native predictive test. Uses the upper predictive-tail interval bound, not a posterior probability. Never forces a rejected or unassessed match. Does not refit models.', 50, 99.9, .1)
    minimum_retention_groups: int = control(2, 'Alternative retention: witnesses per direction', 'Retain a tested alternative when recurring physical molecules distinguish it from every proposed replacement in both directions. Does not veto promotion or rerun Monte Carlo. 0 reproduces historical retirement.', 0, 100000, 1)
    minimum_display_primary_units: int = control(3, 'Minimum recurrent-state molecules', 'Show a fitted population state only when it is the primary explanation for at least this many independent evidence units. All fitted alternatives remain in the frozen audit evidence.', 1, 100000, 1)
    minimum_display_primary_fraction: float = control(.05, 'Minimum recurrent-state fraction', 'Show a fitted population state only when its primary molecules are at least this fraction of the molecules eligible at its fitted span. This is a recurrence/display gate, not an FDR or a refit.', 0, 1, .01)
    recall_hia5_nucleosomes: bool = control(True, 'Replay Hia5 nucleosome/TF boundaries first', 'With native replay enabled, use the existing query-coordinate nuc/gap and TF caller before classifying Hia5 footprints. Preserve original BAM layers and recompute the analysis MSP scaffold. Does not use motifs or family evidence and does not change DddA detection.')
    nuc_split_minimum_llr: float = control(4., 'Hia5 nuc split minimum LLR', 'Native accessible-gap evidence for upstream nucleosome recall. Three opportunities required; TF LLR remains the separate input setting.', 0, 10000, .5)
    nomination_radius_bp: int = control(2, 'Initial proposal edge allowance (±bp)', 'Nominate common native-edge-cell cohorts of actually overlapping LLR calls. This does not assign calls or pad overlap. Each hypothesis is subsequently evaluated by native predictive Monte Carlo.', 0, 100, 1)
    physical_radius_bp: int = control(10, 'Shared-family edge variation (±bp)', 'Physical variation around each shared parent edge, paid once in the bounded parent fit. 5 bp gives finer grouping; 10 bp is the accepted broader setting. No additional matching tolerance.', 0, 100, 1)
    stop_after: str = control('resolved', 'Last stage to compute', 'Keep every completed stage for inspection. Native: separate measurement-source fits; parents: joint bounded fits; consolidated: cached common explanations; resolved: final representative/alias resolution with no extra fits.', choices=['native', 'parents', 'consolidated', 'resolved'])


@dataclass
class RecallerOptions:
    stringency: float = control(0.9, 'k-means stringency (prediction strength)', 'Discovery chooses the largest k whose split-half prediction strength reaches this value and keeps classes at least this stable. Higher = fewer, coarser classes; lower (e.g. 0.6-0.7) finds more and finer classes, such as composite footprints or dense Hia5 data, at the cost of near-duplicates.', 0.05, 1.0, 0.05)
    kmax: int = control(40, 'Maximum k per tile', 'Upper limit on clusters per discovery tile.', 1, 200, 1)
    prediction_splits: int = control(6, 'Prediction-strength split-halves', 'Molecule split-halves used to score each k.', 2, 50, 1)
    seed: int = control(1, 'Discovery seed', 'k-means seed; fixed for reproducibility.', 0, 1000000, 1)
    minimum_candidate_calls: int = control(5, 'Minimum calls per candidate', 'k-means clusters with fewer calls are not carried forward.', 1, 10000, 1)
    call_min_llr: float = control(5.0, 'Discovery call LLR', 'Native calls at or above this LLR define geometries (and are the calls assigned in the records).', 0, 1000, 0.5)
    call_max_bp: int = control(100, 'Discovery call maximum width (bp)', 'Longer calls are not used for discovery.', 5, 1000, 1)
    censor_bp: int = control(15, 'Edge censoring cap (bp)', 'A call edge is uncertain up to the nearest mark, capped at this distance.', 0, 200, 1)
    identity_nats: float = control(5.0, 'Identity test threshold (nats)', 'Overlapping candidates merge while two geometries beat one by less than this held-out log-likelihood.', 0, 10000, 0.5)
    identity_folds: int = control(3, 'Identity test folds', 'Cross-validation folds for the identity test.', 2, 20, 1)
    identity_pad_bp: int = control(6, 'Identity grid padding (bp)', 'Edge-grid padding around the two candidates.', 0, 100, 1)
    identity_wide_bp: int = control(40, 'Identity broad-protection reach (bp)', 'Reach of the broad-protection alternative in the identity test.', 0, 500, 1)
    identity_max_width_bp: int = control(150, 'Identity grid maximum width (bp)', 'Widest footprint on the identity grid.', 10, 2000, 1)
    edge_quantile_low: float = control(10., 'Edge box lower quantile (%)', 'Edge boxes span these quantiles of the class calls\' censored edge ranges. 25/75 gives tighter boxes for positionally variable data.', 0, 50, 1)
    edge_quantile_high: float = control(90., 'Edge box upper quantile (%)', 'See the lower quantile.', 50, 100, 1)
    minimum_core_bp: int = control(3, 'Minimum protected core (bp)', 'A class needs this much DNA between its left and right core-rule boxes; classes whose boxes overlap are dropped as imprecise. -1000 disables the rule.', -1000, 1000, 1)
    core_quantile_low: float = control(25., 'Core-rule box lower quantile (%)', 'The core rule uses edge boxes at these quantiles of the class calls (25/75: the middle half of calls must share a protected core), independent of the scoring boxes.', 0, 50, 1)
    core_quantile_high: float = control(75., 'Core-rule box upper quantile (%)', 'See the lower quantile.', 50, 100, 1)
    jitter_ddda_bp: int = control(0, 'Edge jitter, DddA (bp)', 'Widen DddA edge boxes outward (never into the core) by this much: tolerance for enzyme processivity or binding jitter.', 0, 100, 1)
    jitter_dddb_bp: int = control(0, 'Edge jitter, DddB (bp)', 'As for DddA.', 0, 100, 1)
    jitter_hia5_bp: int = control(0, 'Edge jitter, Hia5 (bp)', 'As for DddA; about 10 bp helped Hia5 at NAPA E-box 1.', 0, 100, 1)
    linker: str = control('both', 'Accessible linker', 'both: a class needs accessible DNA just beyond both edges; either: beyond at least one (footprints abutting a nucleosome, sparse lattices such as DddB).', choices=['both', 'either'])
    linker_bp: int = control(5, 'Linker width (bp)', 'Sites within this distance beyond an edge (and always the nearest site) must be accessible.', 0, 50, 1)
    flank_bp: int = control(25, 'Scoring flank (bp)', 'Molecules are scored over the class edge boxes plus this flank.', 5, 200, 1)
    edge_contraction: bool = control(False, 'Per-channel edge contraction', 'On each channel, pull a class edge box inward past sites that class members almost always mark (the class is drawn wider than that strand\'s footprint, e.g. a CT site marked ~99% inside a GA-defined box). Kept only when held-out likelihood improves.')
    edge_contraction_rate: float = control(0.5, 'Edge contraction: marked rate', 'Sites at the class edge that class members mark more often than this are moved outside the box.', 0.05, 0.99, 0.05)
    edge_minimum_members: float = control(20., 'Edge contraction: minimum members', 'Posterior-weighted class members a site needs before its marked rate is used.', 1, 100000, 1)
    edge_gain_nats: float = control(5., 'Edge contraction: held-out gain (nats)', 'A contraction is kept only if it is proposed in both halves and raises held-out likelihood by this much, summed over both.', 0, 10000, 0.5)
    class_weighting: str = control('bp', 'Class configuration weighting', 'bp: each configuration weighted by the edge positions it covers inside the class boxes (widening boxes adds tolerance without diluting); configurations: uniform over lattice configurations.', choices=['bp', 'configurations'])
    learned_spots: bool = control(True, 'Learned internal spots', 'Learn a class-specific mark rate at interior positions that are sometimes marked while bound (e.g. CTCF +7/+8 on Hia5); kept only when held-out likelihood improves.')
    spot_minimum_evidence_nats: float = control(10., 'Spot channel resolution (nats)', 'Spots are learned only on channels whose expected evidence over the class reaches this.', 0, 1000, 0.5)
    spot_gain_nats: float = control(5., 'Spot held-out gain (nats)', 'A spot is kept only if it raises held-out likelihood by this much, summed over both folds.', 0, 1000, 0.5)
    spot_cap: float = control(0.35, 'Spot rate cap', 'Maximum learned mark rate; spots that reach it are rejected.', 0.01, 0.99, 0.01)
    spot_pseudo_units: float = control(30., 'Spot shrinkage (pseudo-molecules)', 'Learned rates are shrunk toward the model rate by this many pseudo-molecules.', 0, 10000, 1)
    spot_edge_bp: int = control(5, 'Spot interior margin (bp)', 'Spots are learned only this far inside the class span.', 0, 50, 1)
    spot_iterations: int = control(10, 'Spot EM rounds', 'Rounds of EM alternating with spot updates.', 1, 100, 1)
    report_unsupported_classes: bool = control(False, 'Show classes no channel supports', 'Discovered classes that no channel supports (held-out gain and prevalence bound) are left out of the class catalog and call labels, e.g. short calls inside a nucleosome that the scoring assigns to broader protection. They stay in classes.tsv and are listed in the manifest; turn this on to show them anyway.')
    bf_threshold: float = control(3., 'Per-molecule BF threshold', 'Member if the posterior odds exceed the prior odds by this factor, non-member if below its inverse, otherwise abstain. Prevalence comes from EM and does not use it.', 1, 1000, 0.5)
    support_gain_nats: float = control(5., 'Support: held-out gain (nats)', 'A channel supports a class when including it raises held-out likelihood by at least this much (class weight fixed at 0 in the null).', 0, 10000, 0.5)
    support_minimum_lower_bound: float = control(0.02, 'Support: minimum prevalence lower bound', 'And its prevalence Wilson lower bound reaches this.', 0, 1, 0.005)
    resolution_nats: float = control(5., 'Resolution threshold (nats)', 'A channel resolves a class when its expected evidence per molecule over the class reaches this; below it the fraction is reported but flagged unresolved.', 0, 1000, 0.5)
    efficiency_calibration: bool = control(False, 'Per-channel efficiency calibration', 'Scale each channel\'s accessible rate by the observed/expected rate of its most-marked molecules. Off by default: use calibrated emission tables instead.')
    tile_bp: int = control(350, 'Discovery tile (bp)', 'Regions longer than this are discovered in overlapping tiles.', 100, 5000, 10)
    tile_step_bp: int = control(250, 'Discovery tile step (bp)', 'Tile stride; classes found twice are deduplicated.', 50, 5000, 10)
    minimum_channel_units: int = control(20, 'Minimum molecules per channel', 'Channels with fewer molecules in a tile are not quantified.', 1, 100000, 1)


GROUPS = dict(input=InputOptions, sr=SROptions, cr=CROptions, families=FamilyStageOptions, recaller=RecallerOptions, rescue=RescueOptions,
              cross=CrossOptions, comparability=ComparabilityOptions,
              split=SplitOptions, compute=ComputeOptions)


def parameter_schema():
    return {group: [dict(name=f.name, default=getattr(cls(), f.name),
                        type=type(getattr(cls(), f.name)).__name__, **dict(f.metadata))
                    for f in fields(cls)] for group, cls in GROUPS.items()}


# Controls each engine reads, per group; a group not listed is read in full. A control outside these sets must stay at
# its default, so a setting the chosen engine would ignore is rejected instead of being recorded as if it were used.
ACTIVE_CONTROLS = dict(
    staged_native_families=dict(
        cr={'engine','enabled','edge_tolerance_mode','minimum_edge_tolerance_bp'}, sr={'enabled'}, cross={'enabled'},
        rescue={'enabled'}, split={'enabled'}, comparability={'enabled'}, recaller=set(),
        compute={'require_native_cache','cores','maximum_matrix_mb','maximum_region_bp','fit_cache_dir','fit_backend','predictive_backend','predictive_stopping'}),
    # The recaller reads the input group (native replay and loading), Hia5 nucleosome replay and the compute budget.
    # sr/cross.enabled are accepted but change nothing (the recaller's classes are shared by construction; its mode is
    # CR). fit_cache_dir is accepted and unused (no native fits are cached).
    lattice_recaller=dict(
        cr={'engine','enabled'}, sr={'enabled'}, cross={'enabled'}, rescue={'enabled'}, split={'enabled'},
        comparability={'enabled'}, families={'recall_hia5_nucleosomes','nuc_split_minimum_llr','stop_after'},
        compute={'cores','maximum_region_bp','maximum_matrix_mb','fit_cache_dir','require_native_cache','predictive_stopping'}),
)


def _reject_inactive(result, active, engine, advice):
    for group, names in active.items():
        default = GROUPS[group]()
        for f in fields(default):
            if f.name not in names and getattr(result[group],f.name) != getattr(default,f.name):
                raise ValueError(f'{group}.{f.name} is not used by {engine}; reset it to its default ({advice})')


def parse_options(values=None):
    values = values or {}
    if not isinstance(values, dict) or set(values) - set(GROUPS):
        raise ValueError("Unknown consensus parameter group")
    result = {}
    for group, cls in GROUPS.items():
        supplied = values.get(group, {})
        if group == 'recaller' and isinstance(supplied, dict) and 'abutting' in supplied:
            # Removed in 3.0 (its configuration weights were not a normalized prior). Manifests, frozen catalogs and
            # sessions from before the removal carry abutting=false; accept and drop that, refuse a request for it.
            if supplied['abutting'] is not False:
                raise ValueError('recaller.abutting was removed in FiberHMM 3.0: it overweighted classes against '
                                 'wider protection. Molecules whose protected run lines up with one class edge are '
                                 'reported in the "+ edge" prevalence tier; for footprints against a nucleosome, '
                                 'use recaller.linker=either.')
            supplied = {k: v for k, v in supplied.items() if k != 'abutting'}
        definitions = {f.name: f for f in fields(cls)}
        if not isinstance(supplied, dict) or set(supplied) - set(definitions):
            raise ValueError(f"Unknown parameter in {group}")
        obj = cls()
        for key, value in supplied.items():
            default = getattr(obj, key)
            if isinstance(default, bool):
                valid = isinstance(value, bool)
            elif isinstance(default, int):
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif isinstance(default, float):
                valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            else:
                valid = isinstance(value, str)
            meta = definitions[key].metadata
            if not valid or (meta['choices'] is not None and value not in meta['choices']):
                raise ValueError(f"Invalid {group}.{key}: {value!r}")
            if meta['minimum'] is not None and value < meta['minimum'] or meta['maximum'] is not None and value > meta['maximum']:
                raise ValueError(f"{group}.{key} outside [{meta['minimum']}, {meta['maximum']}]")
            setattr(obj, key, value)
        result[group] = obj
    if any(result[g].enabled for g in ('rescue', 'cross', 'comparability', 'split')) and not result['cr'].enabled:
        raise ValueError("Rescue, XCR, comparability and splitting require CR")
    if result['cr'].residual_update_policy=='append_frozen' and result['cr'].engine!='native_family_distribution':
        raise ValueError('cr.residual_update_policy=append_frozen requires the native family-distribution engine')
    draws=result['rescue'].population_draws
    if draws&(draws-1):
        raise ValueError('rescue.population_draws must be a power of two')
    cross=result['cross']
    if result['cr'].engine=='staged_native_families':
        if not result['cr'].enabled or any(result[g].enabled for g in ('rescue', 'split', 'comparability')):
            raise ValueError('Staged families classifies existing LLR calls; enable CR and disable rescue, nucleosome splitting and legacy comparability')
        if result['compute'].fit_backend!='cpu' or result['compute'].predictive_backend!='cpu':
            raise ValueError('Staged families currently requires the reference CPU fitter and predictive scorer')
        for name,fixed in [('edge_tolerance_mode','bounded'),('minimum_edge_tolerance_bp',2)]:
            value=getattr(result['cr'],name)
            if value not in (getattr(CROptions(),name),fixed):
                raise ValueError(f'cr.{name} is fixed to {fixed!r} for staged families; use families.physical_radius_bp for consolidation')
            setattr(result['cr'],name,fixed)
        _reject_inactive(result, ACTIVE_CONTROLS['staged_native_families'], 'staged families',
                         'use the families controls')
    elif result['cr'].engine=='lattice_recaller':
        if not result['cr'].enabled or any(result[g].enabled for g in ('rescue', 'split', 'comparability')):
            raise ValueError('The lattice recaller classifies molecules against discovered classes; enable CR and disable rescue, nucleosome splitting and legacy comparability')
        if not result['input'].correct_native:
            raise ValueError('input.correct_native=false is not supported by the lattice recaller: it discovers and scores '
                             'classes from the replayed native calls and their LLRs. Keep native replay on, or use '
                             'cr.engine=staged_native_families to classify the original BAM calls')
        if result['families'].stop_after!='resolved':
            raise ValueError('families.stop_after (--stop-after) applies to staged_native_families only: the lattice recaller '
                             'discovers and scores classes in one pass. Every run saves evidence.json.gz for replay with '
                             '--evidence or --resume')
        if result['compute'].require_native_cache:
            raise ValueError('compute.require_native_cache (--start-at consolidation) applies to staged_native_families only')
        r = result['recaller']
        if r.edge_quantile_low >= r.edge_quantile_high or r.core_quantile_low >= r.core_quantile_high:
            raise ValueError('recaller edge/core quantile pairs must have low < high')
        if r.tile_step_bp > r.tile_bp:
            raise ValueError('recaller.tile_step_bp must not exceed recaller.tile_bp (tiles must overlap or abut)')
        if r.call_max_bp > r.tile_bp - r.tile_step_bp:
            # A call is used for discovery only inside one tile; a wider call straddling a tile overlap would be
            # invisible to every tile, depending on its phase relative to the tile starts.
            raise ValueError(f'recaller.call_max_bp ({r.call_max_bp}) must not exceed the tile overlap, recaller.tile_bp - '
                             f'recaller.tile_step_bp ({r.tile_bp} - {r.tile_step_bp} = {r.tile_bp - r.tile_step_bp}): '
                             'raise tile_bp or lower tile_step_bp, or a wide footprint straddling a tile overlap is never discovered')
        if result['compute'].predictive_stopping!='full':
            raise ValueError('compute.predictive_stopping applies only to staged native families')
        _reject_inactive(result, ACTIVE_CONTROLS['lattice_recaller'], 'the lattice recaller',
                         'the recaller\'s own controls are in the recaller group')
    else:
        if result['compute'].predictive_stopping!='full':
            raise ValueError('compute.predictive_stopping applies only to staged native families')
        _reject_inactive(result, dict(recaller=set()), f"cr.engine={result['cr'].engine}",
                         'the recaller group applies to cr.engine=lattice_recaller only')
    for name in ('native_minimum_call_attribution_mass', 'native_minimum_geometry_retention',
                 'native_minimum_visible_geometry_mass'):
        if not 0 < getattr(cross,name) <= 1:
            raise ValueError(f'cross.{name} must be in (0,1]')
    if not result['cr'].membership_loss_odds >= 1:
        raise ValueError('cr.membership_loss_odds must be at least 1 (1 = winner-take-all membership)')
    if cross.pair_nomination_gap_bp < 0:
        raise ValueError('cross.pair_nomination_gap_bp must be nonnegative')
    if cross.enabled and result['cr'].engine=='native_family_distribution':
        z2=3.841458820694124
        zero_upper=z2/(cross.native_predictive_replicates+z2)
        cut=1.-cross.native_reference_percent/100.
        if zero_upper >= cut:
            required=math.floor(z2*(1.-cut)/cut)+1
            raise ValueError(f'Native XCR predictive gate unresolved: {cross.native_predictive_replicates} draws '
                f'give zero-exceedance Monte Carlo upper bound {zero_upper:.6g}, not below {cut:.6g}. '
                f'Use at least {required} cross.native_predictive_replicates within the 65535-draw budget, '
                'or lower cross.native_reference_percent. Native CR draws do not change this gate.')
    return result


def options_dict(options):
    return {name: asdict(value) for name, value in options.items()}
