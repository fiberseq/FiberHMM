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
        '--min-mapq', '-q', type=int, default=min_mapq,
        help="Minimum mapping quality; reads below this are written to output "
             "unchanged without footprint/nucleosome tags. Default 0 (call on "
             "all mapped reads). Pass a positive value to filter."
    )
    parser.add_argument(
        '--prob-threshold', type=int, default=prob_threshold,
        help=threshold_help,
    )
    parser.add_argument(
        '--min-read-length', type=int, default=min_read_length,
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

    def __init__(self, platform=None, source="", conflict=None):
        self.platform = platform      # 'pacbio' | 'nanopore' | None
        self.source = source          # human-readable evidence summary
        self.conflict = conflict      # explanation when evidence disagrees

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
                              *, inspect_reads: bool = True):
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
    ``inspect_reads`` is False or when the header's chemistry declaration
    names a single platform.
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
            if inspect_reads and len(declared) != 1:
                for inspected, read in enumerate(bam.fetch(until_eof=True), 1):
                    if not (read.is_secondary or read.is_supplementary):
                        tag = ("MM" if read.has_tag("MM")
                               else ("Mm" if read.has_tag("Mm") else None))
                        platform = (_mm_spec_platform(read.get_tag(tag))
                                    if tag else None)
                        if platform:
                            counts[platform] += 1
                    if inspected >= n_reads:
                        break
    except (OSError, ValueError) as exc:
        return PlatformEvidence(source=f"unreadable input ({exc})")

    sources = []
    if len(declared) > 1:
        return PlatformEvidence(conflict=(
            "the input header declares several platforms "
            f"({', '.join(sorted(declared))})"))
    declared_platform = next(iter(declared), None)
    if declared_platform:
        sources.append((declared_platform, "FIBERHMM-CHEMISTRY declaration"))

    informative = counts["pacbio"] + counts["nanopore"]
    mm_platform = None
    if informative:
        majority = max(counts, key=counts.get)
        minority = informative - counts[majority]
        if minority > _PLATFORM_MINORITY_FRACTION * informative:
            return PlatformEvidence(conflict=(
                f"MM specs are mixed: {counts['pacbio']} read(s) carry PacBio "
                f"T-a calls and {counts['nanopore']} read(s) only Nanopore-style "
                "A+a calls"))
        mm_platform = majority
        pattern = "T-a present" if majority == "pacbio" else "A+a only, no T-a"
        sources.append((majority, f"MM specs of {informative} read(s) ({pattern})"))

    header_votes = _header_platform(header)
    if len(header_votes) == 1:
        sources.append((next(iter(header_votes)), "@RG/@PG header records"))
    elif len(header_votes) > 1 and mm_platform is None and not declared_platform:
        return PlatformEvidence(conflict=(
            "@RG/@PG header records name both PacBio and Nanopore tools"))

    platforms = {platform for platform, _ in sources}
    if len(platforms) > 1:
        detail = "; ".join(f"{src} -> {platform}" for platform, src in sources)
        return PlatformEvidence(conflict=f"evidence disagrees ({detail})")
    if not sources:
        return PlatformEvidence(source="no platform evidence in the first reads")
    return PlatformEvidence(
        platform=sources[0][0],
        source="; ".join(src for _, src in sources),
    )


def resolve_platform_argument(args, input_path, *, tool: str,
                              enzyme_attr: str = "enzyme") -> None:
    """Fill a missing ``args.seq`` from the input BAM, or exit with a fix.

    Only enzymes whose bundled model or observation frame depends on the
    platform (Hia5) are sniffed; for DAF enzymes a missing ``--seq`` is filled
    only from explicit header evidence (it then affects just the declared
    platform). An explicit ``--seq`` is authoritative; a disagreeing sniff
    only warns.
    """
    from fiberhmm.models import enzyme_requires_platform

    enzyme = getattr(args, enzyme_attr, None)
    if not enzyme:
        return
    requires = enzyme_requires_platform(enzyme)
    explicit = getattr(args, "seq", None)
    # Read MM specs only when they can decide something: an explicit --seq is
    # authoritative (header evidence still backs the mismatch warning), and a
    # platform-independent (DAF) enzyme takes only header evidence below.
    evidence = sniff_sequencing_platform(
        input_path, inspect_reads=bool(requires and not explicit))
    if explicit:
        if evidence.platform and evidence.platform != explicit:
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
            f"could not be detected ({evidence.source}); assuming PacBio. Pass "
            "--seq nanopore for Nanopore data.",
            file=sys.stderr,
        )
