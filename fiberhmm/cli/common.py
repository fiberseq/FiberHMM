"""Shared argparse argument factories for FiberHMM CLI tools.

Each function adds a group of related arguments to an ArgumentParser.
Default values can be overridden per-script where needed.
"""

import argparse
import sys
from typing import Collection, Optional

OBSERVATION_MODES = ('pacbio-fiber', 'nanopore-fiber', 'daf', 'gpc', 'cpg')


def add_mode_args(parser: argparse.ArgumentParser,
                  default: str = 'pacbio-fiber',
                  required: bool = False) -> None:
    """Add --mode argument."""
    parser.add_argument(
        '--mode',
        choices=OBSERVATION_MODES,
        default=None if required else default,
        required=required,
        help=f"Analysis mode (default: {default})"
    )


def add_legacy_mode_override(parser: argparse.ArgumentParser) -> None:
    """Accept the old high-level --mode override without advertising it.

    High-level commands infer the observation mode from the selected bundled
    enzyme/platform model, or from custom-model metadata. The hidden option is
    retained so existing scripts and models with incorrect metadata can still
    override that inference explicitly.
    """
    parser.add_argument(
        '--mode',
        choices=OBSERVATION_MODES,
        default=None,
        help=argparse.SUPPRESS,
    )


def resolve_observation_mode(
    model_mode: Optional[str],
    *,
    inferred_mode: Optional[str] = None,
    explicit_mode: Optional[str] = None,
    source_label: str = 'selected model',
    metadata_mode_aliases: Collection[str] = (),
) -> str:
    """Resolve a high-level command's observation mode.

    Precedence is explicit legacy override, bundled enzyme/platform inference,
    then custom-model metadata. Bundled inference is authoritative because
    packaged compatibility model files can contain stale mode metadata.
    """
    valid_model_mode = (
        model_mode if model_mode in OBSERVATION_MODES else None
    )
    valid_inferred_mode = (
        inferred_mode if inferred_mode in OBSERVATION_MODES else None
    )

    if inferred_mode is not None and valid_inferred_mode is None:
        raise ValueError(
            f"internal error: inferred unsupported observation mode "
            f"{inferred_mode!r}"
        )
    if explicit_mode is not None and explicit_mode not in OBSERVATION_MODES:
        raise ValueError(f"unsupported observation mode {explicit_mode!r}")

    expected_mode = valid_inferred_mode or valid_model_mode
    if explicit_mode is not None:
        if expected_mode and explicit_mode != expected_mode:
            print(
                f"WARNING: legacy --mode {explicit_mode!r} overrides the "
                f"{source_label} mode {expected_mode!r}. This is allowed for "
                "compatibility and recovery from incorrect model metadata; "
                "verify that the override is intentional.",
                file=sys.stderr,
            )
        else:
            print(
                "WARNING: high-level --mode is deprecated and normally "
                "unnecessary; mode is inferred from --enzyme/--seq or custom-"
                "model metadata.",
                file=sys.stderr,
            )
        return explicit_mode

    if valid_inferred_mode is not None:
        if (
            valid_model_mode is not None
            and valid_model_mode != valid_inferred_mode
            and valid_model_mode not in metadata_mode_aliases
        ):
            print(
                f"WARNING: {source_label} metadata declares mode "
                f"{valid_model_mode!r}, but --enzyme/--seq selects "
                f"{valid_inferred_mode!r}; using the enzyme/platform mode.",
                file=sys.stderr,
            )
        elif model_mode not in (None, '', 'unknown') and valid_model_mode is None:
            print(
                f"WARNING: {source_label} metadata contains unsupported mode "
                f"{model_mode!r}; using inferred mode {valid_inferred_mode!r}.",
                file=sys.stderr,
            )
        return valid_inferred_mode

    if valid_model_mode is not None:
        return valid_model_mode

    detail = (
        "does not declare an observation mode"
        if model_mode in (None, '', 'unknown')
        else f"declares unsupported observation mode {model_mode!r}"
    )
    raise ValueError(
        f"{source_label} {detail}. Add valid 'mode' metadata "
        f"({', '.join(OBSERVATION_MODES)}) to the custom model. For a legacy "
        "model, --mode remains available as a temporary explicit override."
    )


def ml_threshold(value) -> int:
    """argparse type for an ML probability threshold: an integer 0-255.

    ML bytes are 0-255, so a larger threshold silently turns every
    modification call off.
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected an integer 0-255, got {value!r}")
    if not 0 <= number <= 255:
        raise argparse.ArgumentTypeError(
            f"must be 0-255 (ML probabilities are bytes), got {number}")
    return number


def non_negative_int(value) -> int:
    """argparse type for counts and thresholds that cannot be negative."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected an integer >= 0, got {value!r}")
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {number}")
    return number


def refuse_model_enzyme_assay_conflict(model_mode, enzyme, seq, model_path, *,
                                       tool: str, explicit_mode=None) -> None:
    """Exit 2 when a custom ``-m`` and ``--enzyme`` name different assays.

    A DAF-seq (deamination) table with an m6A enzyme, or the reverse, used to
    run in the model's mode and fail later with a misleading message ("DAF-seq
    calling needs deamination calls"). Platform differences within an assay
    are left to the existing mode resolution; an explicit legacy ``--mode``
    override is honoured.
    """
    if not enzyme or explicit_mode or model_mode not in OBSERVATION_MODES:
        return
    from fiberhmm.models import get_observation_mode

    try:
        expected = get_observation_mode(enzyme, seq, warn_missing_seq=False)
    except KeyError:
        return
    if (model_mode == 'daf') == (expected == 'daf'):
        return

    def assay(mode):
        return 'DAF-seq (deamination)' if mode == 'daf' else 'Fiber-seq (m6A)'

    print(
        f"error: {tool}: -m {model_path} is a {assay(model_mode)} model "
        f"(mode {model_mode}), but --enzyme {enzyme} is {assay(expected)}. "
        f"Drop --enzyme to use the model as it is, or give a {enzyme} model.",
        file=sys.stderr,
    )
    sys.exit(2)


def add_filter_args(parser: argparse.ArgumentParser,
                    min_mapq: int = 0,
                    prob_threshold: Optional[int] = 128,
                    min_read_length: int = 1000) -> None:
    """Add read filtering arguments (--min-mapq, --prob-threshold, --min-read-length).

    ``prob_threshold=None`` makes the default chemistry-dependent (resolved by
    :func:`fiberhmm.models.resolve_prob_threshold` once --enzyme/--seq are
    known): 248 for Hia5 Nanopore, 128 otherwise.
    """
    if prob_threshold is None:
        threshold_help = ("Minimum MM/ML probability (0-255) to call a "
                          "modification. Default: chemistry preset -- 248 for "
                          "Hia5 Nanopore (--seq nanopore, given or detected), "
                          "128 otherwise. R/Y- and MD-encoded DAF input is "
                          "binary and ignores it.")
    else:
        threshold_help = (f"Minimum MM/ML probability (0-255) to call "
                          f"modification (default: {prob_threshold})")
    parser.add_argument(
        '--min-mapq', '-q', type=non_negative_int, default=min_mapq,
        help="Minimum mapping quality; reads below this are written to output "
             "unchanged without footprint/nucleosome tags. Default 0 (call on "
             "all mapped reads). Pass a positive value to filter."
    )
    parser.add_argument(
        '--prob-threshold', type=ml_threshold, default=prob_threshold,
        help=threshold_help,
    )
    parser.add_argument(
        '--min-read-length', type=non_negative_int, default=min_read_length,
        help=f"Minimum aligned read length in bp; shorter reads are written to "
             f"output unchanged without footprint/nucleosome tags. Set to 0 to "
             f"attempt calling on all reads regardless of length (default: {min_read_length})"
    )


def add_context_args(parser: argparse.ArgumentParser,
                     default=3,
                     multiple: bool = False) -> None:
    """Add context size argument.

    Args:
        default: Default value (int or list of ints)
        multiple: If True, accept multiple values (--context-sizes -k 3 4 5 6)
    """
    if multiple:
        if not isinstance(default, list):
            default = [default]
        parser.add_argument(
            '--context-sizes', '-k', type=int, nargs='+', default=default,
            help=f"Context sizes (k-mer) to compute (default: {default})"
        )
    else:
        parser.add_argument(
            '--context-size', '-k', type=int, default=default,
            help=f"Context size (k-mer) for HMM (default: {default})"
        )


def add_edge_trim_args(parser: argparse.ArgumentParser,
                       default: int = 10) -> None:
    """Add --edge-trim argument."""
    parser.add_argument(
        '--edge-trim', '-e', type=int, default=default,
        help=f"Bases to trim from read edges (default: {default})"
    )


def add_parallel_args(parser: argparse.ArgumentParser,
                      default_cores: int = 1,
                      default_region_size: int = 10_000_000) -> None:
    """Add parallelization arguments (--cores, --region-size, --skip-scaffolds, --chroms)."""
    parser.add_argument(
        '--cores', '-c', type=int, default=default_cores,
        help=f"Number of CPU cores (0=auto, default: {default_cores})"
    )
    parser.add_argument(
        '--region-size', type=int, default=default_region_size,
        help=f"Region size in bp for parallel processing (default: {default_region_size:,})"
    )
    parser.add_argument(
        '--skip-scaffolds', action='store_true',
        help="Skip scaffold/contig chromosomes"
    )
    parser.add_argument(
        '--chroms', nargs='+', default=None,
        help="Only process these chromosomes"
    )
    parser.add_argument(
        '--io-threads', type=int, default=4,
        help="Number of htslib decompression/compression threads for BAM I/O (default: 4)"
    )
    parser.add_argument(
        '--streaming', action='store_true',
        help="Use streaming pipeline mode (works with unaligned/unindexed BAMs and stdin). "
             "Recommended for unaligned data or when reading from pipes."
    )
    parser.add_argument(
        '--chunk-size', type=int, default=500,
        help="Reads per compute chunk in streaming mode (default: 500)"
    )


def add_output_args(parser: argparse.ArgumentParser,
                    required: bool = True,
                    help_text: str = "Output directory") -> None:
    """Add -o/--output argument."""
    parser.add_argument(
        '-o', '--output', required=required,
        help=help_text
    )


def add_stats_args(parser: argparse.ArgumentParser) -> None:
    """Add --stats flag."""
    parser.add_argument(
        '--stats', action='store_true',
        help="Generate summary statistics and plots"
    )


def add_verbose_args(parser: argparse.ArgumentParser) -> None:
    """Add --verbose flag."""
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help="Verbose output"
    )


def resolve_input_prob_threshold(explicit, enzyme, seq, input_path) -> int:
    """ML threshold for a run on ``input_path``: explicit, else chemistry preset.

    ``enzyme``/``seq`` are the run's resolved chemistry (``--enzyme``/``--seq``
    after platform detection). Without an enzyme (a custom ``--model``) the
    input header's chemistry declaration decides, so a custom table on a
    declared Hia5 Nanopore BAM also reads ML at 248. Resolved by
    :func:`fiberhmm.models.resolve_prob_threshold`, shared with call/apply.
    """
    from fiberhmm.models import (
        declared_prob_threshold_chemistry,
        resolve_prob_threshold,
    )

    if explicit is not None:
        return int(explicit)
    if not enzyme and input_path and input_path != '-':
        import pysam
        try:
            with pysam.AlignmentFile(input_path, 'rb', check_sq=False) as bam:
                declared_enzyme, declared_seq = (
                    declared_prob_threshold_chemistry(bam.header))
        except (OSError, ValueError):
            declared_enzyme = declared_seq = None
        enzyme = declared_enzyme
        seq = seq or declared_seq
    return resolve_prob_threshold(None, enzyme, seq)


def add_version_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--version``: prints ``fiberhmm <version>`` and exits.

    Every console script registers it, so each reports the same package
    version string.
    """
    from fiberhmm import __version__
    parser.add_argument(
        '--version', action='version',
        version=f'fiberhmm {__version__}',
        help="Print the FiberHMM version and exit",
    )


# ---------------------------------------------------------------------------
# Sequencing-platform inference for a missing --seq
# ---------------------------------------------------------------------------

PLATFORM_SNIFF_READS = 200
# Share of informative reads the minority MM pattern may reach before the
# input is treated as mixed/conflicting.
_PLATFORM_MINORITY_FRACTION = 0.10
_PACBIO_PROGRAMS = {"ccs", "pbmm2", "primrose", "jasmine", "lima", "pbindex",
                    "pbccs"}
_ONT_PROGRAMS = {"dorado", "guppy", "guppy_basecaller", "minknow", "bonito"}


class PlatformEvidence:
    """Result of :func:`sniff_sequencing_platform`."""

    def __init__(self, platform=None, source="", conflict=None, *,
                 reads_inspected=0, m6a_reads=0, mm_platform=None,
                 mm_source="", declared_platform=None):
        self.platform = platform      # 'pacbio' | 'nanopore' | None
        self.source = source          # human-readable evidence summary
        self.conflict = conflict      # explanation when evidence disagrees
        # Read-level evidence (only when records were inspected): primary
        # records read, how many carry an m6A MM spec (A+a / T-a), and the
        # platform their MM specs alone indicate.
        self.reads_inspected = reads_inspected
        self.m6a_reads = m6a_reads
        self.mm_platform = mm_platform
        self.mm_source = mm_source
        # The header's single FIBERHMM-CHEMISTRY platform, if any.
        self.declared_platform = declared_platform

    def __repr__(self):  # pragma: no cover - debugging aid
        return (f"PlatformEvidence(platform={self.platform!r}, "
                f"source={self.source!r}, conflict={self.conflict!r})")


def _mm_spec_platform(mm_tag: str):
    """'pacbio' for a T-a (bottom-strand m6A) spec, 'nanopore' for A+a only."""
    specs = [item.split(",", 1)[0] for item in str(mm_tag).split(";") if item]
    bases = {spec.rstrip(".?") for spec in specs}
    if "T-a" in bases:
        return "pacbio"
    if "A+a" in bases:
        return "nanopore"
    return None


def _mm_has_m6a(mm_tag: str) -> bool:
    """Whether an MM tag carries an m6A spec on A or T.

    Codes follow the SAM spec, as the MM parser reads them: a run of
    single-letter codes (``A+a``, ``A+ab``) or one ChEBI number (``A+21839``).
    """
    for item in str(mm_tag).split(";"):
        spec = item.split(",", 1)[0].strip().rstrip(".?")
        if len(spec) < 3 or spec[0] not in "ATN" or spec[1] not in "+-":
            continue
        codes = spec[2:]
        if codes == "21839" or (codes.isalpha() and "a" in codes):
            return True
    return False


def _header_platform(header_dict):
    votes = set()
    for group in header_dict.get("RG", []):
        platform = str(group.get("PL", "")).upper()
        if platform in {"PACBIO", "PACBIO_SMRT"}:
            votes.add("pacbio")
        elif platform in {"ONT", "NANOPORE", "OXFORD_NANOPORE"}:
            votes.add("nanopore")
    for program in header_dict.get("PG", []):
        name = str(program.get("PN") or program.get("ID") or "").lower()
        name = name.split(".", 1)[0]
        command = str(program.get("CL", ""))
        if name.startswith("fiberhmm"):
            continue
        if name in _PACBIO_PROGRAMS or "map-hifi" in command or "map-pb" in command:
            votes.add("pacbio")
        elif name in _ONT_PROGRAMS or "map-ont" in command or "lr:hq" in command:
            votes.add("nanopore")
    return votes


def sniff_sequencing_platform(bam_path, n_reads: int = PLATFORM_SNIFF_READS,
                              *, inspect_reads: bool = True,
                              inspect_declared: bool = False):
    """Infer PacBio vs Nanopore from a BAM's own evidence.

    Evidence, strongest first: a FiberHMM chemistry declaration in the header;
    the MM modification specs of the first reads (PacBio fiber-seq reports
    bottom-strand m6A as ``T-a``, Nanopore reports only ``A+a``); @RG ``PL`` and
    @PG program names. Disagreement between sources, or a mix of MM patterns,
    is reported as a conflict rather than resolved silently. Returns a
    :class:`PlatformEvidence`; ``platform`` is None when nothing is known
    (including stdin, which cannot be peeked without consuming it).

    At most ``n_reads`` records are read, tagged or not (an MM-less DAF BAM
    is never scanned to EOF). Records are not read at all when
    ``inspect_reads`` is False, or when the header's chemistry declaration
    names a single platform unless ``inspect_declared`` is True. The
    declaration still decides ``platform`` then; the read-level evidence is
    reported in ``mm_platform``/``m6a_reads``/``reads_inspected``.
    """
    if not bam_path or bam_path == "-":
        return PlatformEvidence(source="stdin (not inspected)")
    import pysam

    from fiberhmm.io.bam_header import declared_chemistries

    try:
        with pysam.AlignmentFile(bam_path, "rb", check_sq=False) as bam:
            header = bam.header.to_dict()
            declared = {
                str(item.get("platform", "")).lower()
                for item in declared_chemistries(bam.header)
            } & {"pacbio", "nanopore"}
            counts = {"pacbio": 0, "nanopore": 0}
            primary = m6a_reads = 0
            mapped_primary = mapped_m6a = 0
            aligned_input = bool(getattr(bam, "references", None) or ())
            if inspect_reads and (len(declared) != 1 or inspect_declared):
                for inspected, read in enumerate(bam.fetch(until_eof=True), 1):
                    if not (read.is_secondary or read.is_supplementary):
                        primary += 1
                        tag = ("MM" if read.has_tag("MM")
                               else ("Mm" if read.has_tag("Mm") else None))
                        mm = read.get_tag(tag) if tag else None
                        platform = _mm_spec_platform(mm) if tag else None
                        if platform:
                            counts[platform] += 1
                        has_m6a = bool(tag) and _mm_has_m6a(mm)
                        m6a_reads += has_m6a
                        if not read.is_unmapped:
                            mapped_primary += 1
                            mapped_m6a += has_m6a
                    if inspected >= n_reads:
                        break
    except (OSError, ValueError) as exc:
        return PlatformEvidence(source=f"unreadable input ({exc})")

    if aligned_input:
        # Aligned input: unmapped records are passed through uncalled, so
        # only mapped reads are evidence (none sampled: no evidence).
        primary, m6a_reads = mapped_primary, mapped_m6a
    read_evidence = {"reads_inspected": primary, "m6a_reads": m6a_reads}
    informative = counts["pacbio"] + counts["nanopore"]
    if informative:
        majority = max(counts, key=counts.get)
        if informative - counts[majority] <= _PLATFORM_MINORITY_FRACTION * informative:
            pattern = "T-a present" if majority == "pacbio" else "A+a only, no T-a"
            read_evidence["mm_platform"] = majority
            read_evidence["mm_source"] = (
                f"MM specs of {informative} read(s) ({pattern})")

    sources = []
    if len(declared) > 1:
        return PlatformEvidence(conflict=(
            "the input header declares several platforms "
            f"({', '.join(sorted(declared))})"), **read_evidence)
    declared_platform = next(iter(declared), None)
    if declared_platform:
        sources.append((declared_platform, "FIBERHMM-CHEMISTRY declaration"))
        if inspect_declared:
            # The declaration settles the platform; reads were read only for
            # the read-level evidence above.
            return PlatformEvidence(
                platform=declared_platform,
                source="FIBERHMM-CHEMISTRY declaration",
                declared_platform=declared_platform, **read_evidence)

    mm_platform = None
    if informative:
        majority = max(counts, key=counts.get)
        minority = informative - counts[majority]
        if minority > _PLATFORM_MINORITY_FRACTION * informative:
            return PlatformEvidence(conflict=(
                f"MM specs are mixed: {counts['pacbio']} read(s) carry PacBio "
                f"T-a calls and {counts['nanopore']} read(s) only Nanopore-style "
                "A+a calls"), **read_evidence)
        mm_platform = majority
        pattern = "T-a present" if majority == "pacbio" else "A+a only, no T-a"
        sources.append((majority, f"MM specs of {informative} read(s) ({pattern})"))

    header_votes = _header_platform(header)
    if len(header_votes) == 1:
        sources.append((next(iter(header_votes)), "@RG/@PG header records"))
    elif len(header_votes) > 1 and mm_platform is None and not declared_platform:
        return PlatformEvidence(conflict=(
            "@RG/@PG header records name both PacBio and Nanopore tools"),
            **read_evidence)

    platforms = {platform for platform, _ in sources}
    if len(platforms) > 1:
        detail = "; ".join(f"{src} -> {platform}" for platform, src in sources)
        return PlatformEvidence(conflict=f"evidence disagrees ({detail})",
                                **read_evidence)
    if not sources:
        return PlatformEvidence(source="no platform evidence in the first reads",
                                **read_evidence)
    return PlatformEvidence(
        platform=sources[0][0],
        source="; ".join(src for _, src in sources),
        **read_evidence,
    )


def add_force_seq_arg(parser: argparse.ArgumentParser) -> None:
    """Add ``--force-seq``: run with an explicit ``--seq`` the reads contradict."""
    parser.add_argument(
        '--force-seq', action='store_true',
        help="Use the given --seq even when the input's MM specs or header "
             "say the reads come from the other platform (normally refused: "
             "the wrong platform model changes the calls, e.g. ~100x more "
             "TF calls for Nanopore reads called as PacBio), and run Hia5 "
             "even when the first reads carry no m6A calls.",
    )


def _refuse_mismatched_seq(args, evidence, *, tool, enzyme, explicit):
    """Exit 2 when the reads contradict an explicit ``--seq`` (unless forced)."""
    # A FIBERHMM-CHEMISTRY declaration is authoritative, as when --seq is
    # omitted: a sample without T-a calls does not prove Nanopore origin.
    # Otherwise MM specs are direct evidence, and header records back them up
    # when no read carries an informative spec.
    if evidence.declared_platform:
        if evidence.declared_platform != explicit:
            # The chemistry reconciliation (ChemistryConflictError, with
            # --replace-chemistry as the way out) handles a declared conflict.
            print(
                f"WARNING: --seq {explicit} was given, but the input declares "
                f"--seq {evidence.declared_platform} (FIBERHMM-CHEMISTRY "
                "declaration).",
                file=sys.stderr,
            )
        return
    if evidence.mm_platform:
        observed, source = evidence.mm_platform, evidence.mm_source
    elif evidence.platform and not evidence.conflict:
        observed, source = evidence.platform, evidence.source
    else:
        return
    if observed == explicit:
        return
    if getattr(args, 'force_seq', False):
        print(
            f"WARNING: --seq {explicit} was given, but the input looks like "
            f"{observed} ({source}). Using --seq {explicit} because of "
            "--force-seq.",
            file=sys.stderr,
        )
        return
    print(
        f"error: {tool}: --seq {explicit} was given, but the input looks like "
        f"{observed} ({source}). Calling {observed} reads with the "
        f"{explicit} model changes the calls (for example, Nanopore reads "
        f"called as PacBio give ~100x more TF calls). Pass --seq {observed}, "
        f"omit --seq "
        f"to detect it, or add --force-seq to use --seq {explicit} anyway.",
        file=sys.stderr,
    )
    sys.exit(2)


def _refuse_reads_without_m6a(args, evidence, *, tool, enzyme):
    """Exit 2 when an m6A enzyme meets reads without any m6A MM calls.

    Only the first records are read, so ``--force-seq`` runs anyway (a BAM
    whose first reads lack m6A calls but later ones carry them); an explicit
    legacy ``--mode`` override decides the observation mode itself.
    """
    if evidence.reads_inspected == 0 or evidence.m6a_reads > 0:
        return
    if getattr(args, 'mode', None):
        return
    if getattr(args, 'force_seq', False):
        print(
            f"WARNING: none of the first {evidence.reads_inspected} primary "
            "reads carries an m6A MM/ML tag; running anyway because of "
            "--force-seq.",
            file=sys.stderr,
        )
        return
    print(
        f"error: {tool}: --enzyme {enzyme} calls footprints from m6A "
        f"modification calls, but none of the first {evidence.reads_inspected} "
        "primary reads carries an m6A MM/ML tag (MM A+a or T-a), so every read "
        "would get no footprints. Is this DAF-seq? Use --enzyme dddb or "
        "--enzyme ddda. For Fiber-seq, call m6A first (ft predict-m6a for "
        "PacBio, dorado with an m6A model for Nanopore). If later reads do "
        "carry m6A calls, add --force-seq to run anyway.",
        file=sys.stderr,
    )
    sys.exit(2)


def resolve_platform_argument(args, input_path, *, tool: str,
                              enzyme_attr: str = "enzyme") -> None:
    """Fill a missing ``args.seq`` from the input BAM, or exit with a fix.

    Only enzymes whose bundled model or observation frame depends on the
    platform (Hia5) are sniffed; for DAF enzymes a missing ``--seq`` is filled
    only from explicit header evidence (it then affects just the declared
    platform). For Hia5 the first reads are always inspected: an explicit
    ``--seq`` that the reads' MM specs (or, without informative specs, the
    header) contradict is refused unless ``--force-seq`` is given, and reads
    with no m6A MM calls at all are refused (DAF-seq run with Hia5). For DAF
    enzymes a disagreeing header only warns.
    """
    from fiberhmm.models import enzyme_requires_platform

    enzyme = getattr(args, enzyme_attr, None)
    if not enzyme:
        return
    requires = enzyme_requires_platform(enzyme)
    explicit = getattr(args, "seq", None)
    # MM specs decide only for Hia5: a platform-independent (DAF) enzyme takes
    # header evidence alone, and its reads carry no m6A specs to read.
    evidence = sniff_sequencing_platform(
        input_path, inspect_reads=bool(requires),
        inspect_declared=bool(requires))
    if requires:
        _refuse_reads_without_m6a(args, evidence, tool=tool, enzyme=enzyme)
    if explicit:
        if requires:
            _refuse_mismatched_seq(args, evidence, tool=tool, enzyme=enzyme,
                                   explicit=explicit)
        elif evidence.platform and evidence.platform != explicit:
            print(
                f"WARNING: --seq {explicit} was given, but the input looks like "
                f"{evidence.platform} ({evidence.source}). Using --seq {explicit} "
                "as requested.",
                file=sys.stderr,
            )
        return
    if evidence.conflict:
        if requires:
            print(
                f"error: {tool}: cannot infer the sequencing platform for "
                f"--enzyme {enzyme}: {evidence.conflict}. Pass --seq pacbio or "
                "--seq nanopore.",
                file=sys.stderr,
            )
            sys.exit(2)
        return
    if evidence.platform:
        if requires or "header" in evidence.source or "declaration" in evidence.source:
            args.seq = evidence.platform
            print(
                f"NOTE: --seq not given; using --seq {evidence.platform} "
                f"(detected from {evidence.source}).",
                file=sys.stderr,
            )
        return
    if requires:
        print(
            f"WARNING: --seq not given for --enzyme {enzyme} and the platform "
            f"could not be detected ({evidence.source}); assuming PacBio "
            "(--seq pacbio). Pass --seq nanopore for Nanopore data.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Input/output path aliasing guard (shared by every CLI that writes files)
# ---------------------------------------------------------------------------
#
# A mistyped path must never destroy data: an output that names an input (or
# an input's index), two outputs that name one file, or an output/input inside
# a directory the tool deletes when it finishes. Paths are compared by the
# file they reach, not by spelling: symlinks are resolved, hard links compare
# equal (same device and inode), and on case-insensitive filesystems (macOS
# APFS/HFS+ default, Windows) names that differ only in case are the same.

# Sidecars that describe an input and must not be overwritten either.
_INPUT_SIDECAR_SUFFIXES = ('.bai', '.csi', '.crai', '.tbi', '.fai', '.gzi')


class PathAliasError(ValueError):
    """An output path would overwrite or delete an input or another output."""


def _labelled_paths(paths):
    """``[(label, path)]`` from ``{label: path-or-paths}`` or ``(label, path)``
    pairs; ``None``, ``''`` and ``'-'`` (stdin/stdout) are skipped."""
    import os

    items = paths.items() if hasattr(paths, 'items') else (paths or ())
    out = []
    for label, value in items:
        values = value if isinstance(value, (list, tuple)) else [value]
        for path in values:
            if path is None:
                continue
            path = os.fspath(path)
            if path in ('', '-'):
                continue
            out.append((label, path))
    return out


def _case_insensitive_directory(directory: str) -> bool:
    """Whether names in existing ``directory``'s filesystem ignore case.

    Probed without writing: the case-swapped spelling of the nearest path
    component that has letters is looked up. Undecidable paths count as
    case-sensitive.
    """
    import os

    probe = directory
    while True:
        parent, name = os.path.split(probe)
        if name and name.swapcase() != name:
            try:
                return os.path.samefile(probe, os.path.join(parent, name.swapcase()))
            except OSError:
                return False
        if not name or parent == probe:
            return False
        probe = parent


def _location_key(path: str):
    """Identity of the file ``path`` reaches (or would create).

    ``(st_dev, st_ino, rest)``: the nearest existing ancestor of the fully
    resolved path, plus the not-yet-existing components below it (case-folded
    on a case-insensitive filesystem). Two paths with equal keys are the same
    file now or will be once written.
    """
    import os

    current = os.path.realpath(os.path.abspath(os.path.expanduser(path)))
    rest = []
    while not os.path.exists(current):
        parent, name = os.path.split(current)
        if parent == current:
            break
        rest.append(name)
        current = parent
    st = os.stat(current)
    if rest and _case_insensitive_directory(current):
        rest = [name.casefold() for name in rest]
    return (st.st_dev, st.st_ino, tuple(reversed(rest)))


def _traversed_directories(path: str, max_links: int = 40):
    """Every directory whose entries resolving ``path`` passes through.

    The path is resolved one component at a time, following each symlink
    (``..`` is physical, as the kernel does it), so a link anywhere on the
    way -- including a chain of links -- contributes the directory holding
    it. The fully resolved path itself is included.
    """
    import os

    absolute = os.path.abspath(os.path.expanduser(path))
    drive, rest = os.path.splitdrive(absolute)
    root = drive + os.sep
    pending = [part for part in rest.split(os.sep) if part]
    current = root
    seen = []
    links = 0
    while pending:
        name = pending.pop(0)
        if name == '.':
            continue
        if name == '..':
            current = os.path.dirname(current) or root
            continue
        seen.append(current)
        candidate = os.path.join(current, name)
        if os.path.islink(candidate) and links < max_links:
            links += 1
            target = os.readlink(candidate)
            target_drive, target_rest = os.path.splitdrive(target)
            parts = [part for part in target_rest.split(os.sep) if part]
            if os.path.isabs(target):
                current = (target_drive or drive) + os.sep
            pending = parts + pending
        else:
            current = candidate
    seen.append(current)
    return seen


def _is_inside(path: str, dir_key) -> bool:
    """Whether ``path`` is, lies below, or reaches through the directory
    ``dir_key``: resolving it passes an entry (file, directory or symlink)
    of that directory, which disappears with it."""
    return any(_location_key(directory) == dir_key
               for directory in _traversed_directories(path))


def find_path_aliases(*, inputs=(), outputs=(), deleted_dirs=(),
                      protect_input_sidecars: bool = True):
    """Problems (one sentence each) with these paths; empty when they are safe.

    ``inputs``/``outputs``/``deleted_dirs`` map an option label (``'--input'``)
    to a path or a list of paths, or are ``(label, path)`` pairs. An output is
    refused when it is the same file as an input, as an index beside an input
    (``.bai/.csi/.crai/.tbi/.fai/.gzi``, when ``protect_input_sidecars``), or
    as another output -- including the index (``.bai/.csi/...``) that
    publishing an output writes or removes beside it. Any input (or input index)
    or output inside a directory in ``deleted_dirs`` (or naming it) is refused
    too.
    """
    import os

    from fiberhmm.inference.bam_output import index_paths_for

    inputs = _labelled_paths(inputs)
    outputs = _labelled_paths(outputs)
    deleted_dirs = _labelled_paths(deleted_dirs)

    protected = [(label, path, _location_key(path), None) for label, path in inputs]
    if protect_input_sidecars:
        for label, path in inputs:
            sidecars = list(index_paths_for(path))
            sidecars += [path + suffix for suffix in ('.fai', '.gzi')]
            for sidecar in dict.fromkeys(sidecars):
                if os.path.exists(sidecar):
                    protected.append((label, sidecar, _location_key(sidecar), path))

    # Indexes an output's publication writes or deletes beside it (BAM
    # writers publish whatever the file is called, so every output counts).
    implied = []
    for label, path in outputs:
        for index in dict.fromkeys(index_paths_for(path)):
            implied.append((label, path, index, _location_key(index)))

    problems = []
    for in_label, in_path, key, owner in protected:
        hit = next((i for i in implied if i[3] == key), None)
        if hit is not None:
            problems.append(
                f"{in_label} {in_path} is where {hit[0]} {hit[1]} keeps its index "
                f"({hit[2]}); publishing the output would replace or delete it. "
                "Choose a different output path")

    seen_outputs = []
    for out_label, out_path in outputs:
        key = _location_key(out_path)
        clash = next((p for p in protected if p[2] == key), None)
        if clash is not None:
            in_label, in_path, _key, owner = clash
            if owner is None:
                problems.append(
                    f"{out_label} {out_path} is the same file as {in_label} "
                    f"{in_path}; writing it would destroy the input. Choose a "
                    "different output path")
            else:
                problems.append(
                    f"{out_label} {out_path} is the index {in_path} of {in_label} "
                    f"{owner}; writing it would destroy that index. Choose a "
                    "different output path")
            continue
        earlier = next((o for o in seen_outputs if o[2] == key), None)
        if earlier is not None:
            problems.append(
                f"{earlier[0]} and {out_label} name the same file ({earlier[1]}, "
                f"{out_path}); each output needs its own path")
            continue
        hit = next((i for i in implied if i[3] == key and i[1] != out_path), None)
        if hit is not None:
            problems.append(
                f"{out_label} {out_path} is the index ({hit[2]}) that publishing "
                f"{hit[0]} {hit[1]} replaces or deletes; choose a different path")
            continue
        seen_outputs.append((out_label, out_path, key))

    for dir_label, directory in deleted_dirs:
        dir_key = _location_key(directory)
        candidates = ([('input', label, path) for label, path, _k, _o in protected]
                      + [('output', label, path) for label, path in outputs])
        for kind, label, path in candidates:
            if _is_inside(path, dir_key):
                problems.append(
                    f"{kind} {label} {path} is inside {dir_label} {directory}, "
                    "which is deleted when the run finishes; choose a path "
                    "outside it")
    return problems


def check_path_aliases(**kwargs) -> None:
    """Raise :class:`PathAliasError` for the problems :func:`find_path_aliases`
    finds."""
    problems = find_path_aliases(**kwargs)
    if problems:
        raise PathAliasError('; '.join(problems))


def refuse_path_aliases(tool: str, **kwargs) -> None:
    """Exit with status 2 and a one-line error naming the clashing options
    when :func:`find_path_aliases` finds a problem. Call it before any file is
    opened for writing."""
    try:
        check_path_aliases(**kwargs)
    except PathAliasError as exc:
        print(f"{tool}: error: {exc}", file=sys.stderr)
        # The bundled-model lookup would otherwise repeat this warning.
        args.seq = "pacbio"


# ---------------------------------------------------------------------------
# User-error reporting for console scripts and model/input validation
# ---------------------------------------------------------------------------

# pysam/htslib errors that mean "this input is not a readable BAM/CRAM/SAM".
_INPUT_FORMAT_MESSAGES = (
    'file does not contain alignment data',
    'file has no sequences defined',
    'could not open alignment file',
    'no bgzf eof marker',
    'file may be truncated',
    'truncated file',
    'error while reading file',
)
_DEBUG_ENV = 'FIBERHMM_DEBUG'


def is_input_error(exc: BaseException) -> bool:
    """Whether ``exc`` is an ordinary input problem rather than a bug.

    Missing paths, directories and permission problems (``FileNotFoundError``,
    ``IsADirectoryError``, ``NotADirectoryError``, ``PermissionError``), and
    the pysam/htslib errors for a file that is not, or no longer, a readable
    BAM/CRAM/SAM.
    """
    if isinstance(exc, (FileNotFoundError, IsADirectoryError,
                        NotADirectoryError, PermissionError)):
        return True
    if isinstance(exc, (ValueError, OSError)):
        text = str(exc).lower()
        return any(message in text for message in _INPUT_FORMAT_MESSAGES)
    return False


def _input_error_text(exc: BaseException) -> str:
    import re

    text = re.sub(r'^\[Errno \d+\]\s*', '', str(exc)).strip()
    lowered = text.lower()
    if '\n' not in text and (isinstance(exc, ValueError)
                             or 'eof marker' in lowered or 'truncated' in lowered):
        text += ' (is it a valid, complete BAM/CRAM/SAM file?)'
    return text


def run_reporting_input_errors(tool: str, main):
    """Run ``main()``; an input error exits 2 with one line instead of a traceback.

    Errors :func:`is_input_error` does not recognise propagate unchanged.
    ``FIBERHMM_DEBUG=1`` re-raises input errors too, with their traceback.
    """
    import os

    try:
        return main()
    except (OSError, ValueError) as exc:
        if not is_input_error(exc) or os.environ.get(_DEBUG_ENV):
            raise
        print(f"error: {tool}: {_input_error_text(exc)}", file=sys.stderr)
        sys.exit(2)


_MODEL_JSON_KEYS = ('n_states', 'startprob', 'transmat', 'emissionprob')


def require_model_files(tool: str, *flag_paths) -> None:
    """Exit 2 with one line unless each given model path is a loadable model.

    ``flag_paths`` are ``(flag, path)`` pairs; ``None`` paths are skipped. A
    ``.json`` file must parse and carry the HMM keys
    ``fiberhmm.core.model_io.load_model_with_metadata`` reads, so a JSON that
    is not a model is reported by name instead of as ``KeyError: 'n_states'``.
    """
    import json
    import os

    for flag, path in flag_paths:
        if not path:
            continue
        if not os.path.exists(path):
            problem = 'does not exist'
        elif os.path.isdir(path):
            problem = 'is a directory, not a model file'
        elif str(path).endswith('.json'):
            try:
                with open(path) as handle:
                    data = json.load(handle)
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                problem = f'is not valid JSON ({exc})'
            else:
                missing = ([key for key in _MODEL_JSON_KEYS if key not in data]
                           if isinstance(data, dict) else list(_MODEL_JSON_KEYS))
                problem = (
                    'is not a FiberHMM model (missing '
                    + ', '.join(missing) + ')' if missing else None)
        else:
            problem = None
        if problem:
            print(f"error: {tool}: {flag} {path} {problem}", file=sys.stderr)
            sys.exit(2)


def refuse_non_bam_output(path, tool: str) -> None:
    """Exit 2 when a BAM-writing tool is given a ``.sam``/``.cram`` output name.

    The tools always write BGZF BAM; a ``.sam`` name gave BAM bytes under a SAM
    name (plus ``x.sam.bai``).
    """
    if not path or path == '-':
        return
    suffix = str(path).lower().rsplit('.', 1)[-1] if '.' in str(path) else ''
    if suffix in ('sam', 'cram'):
        print(
            f"error: {tool}: writes BAM, but the output is named {path!r}. Name "
            "it .bam (convert afterwards with samtools view if you need "
            f"{suffix.upper()}).",
            file=sys.stderr,
        )
        sys.exit(2)


def refuse_region_options_without_region_parallel(args, tool: str) -> None:
    """Exit 2 when --chroms/--skip-scaffolds are given to a streaming run.

    They select regions of the region-parallel pipeline; the streaming path
    used to ignore them and call every read.
    """
    given = [flag for flag, value in (('--chroms', getattr(args, 'chroms', None)),
                                      ('--skip-scaffolds', getattr(args, 'skip_scaffolds', False)))
             if value]
    if given and not getattr(args, 'region_parallel', False):
        print(
            f"error: {tool}: {' and '.join(given)} "
            f"{'select' if len(given) > 1 else 'selects'} regions of "
            "--region-parallel runs; add --region-parallel (indexed, "
            "coordinate-sorted input), or use --region for a streaming run.",
            file=sys.stderr,
        )
        sys.exit(2)
