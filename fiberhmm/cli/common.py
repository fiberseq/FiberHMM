"""Shared argparse argument factories for FiberHMM CLI tools.

Each function adds a group of related arguments to an ArgumentParser.
Default values can be overridden per-script where needed.
"""

import argparse
import sys
from typing import Optional

OBSERVATION_MODES = ('pacbio-fiber', 'nanopore-fiber', 'daf')


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
) -> str:
    """Resolve a high-level command's observation mode.

    Precedence is explicit legacy override, bundled enzyme/platform inference,
    then custom-model metadata. Bundled inference is authoritative because
    historical model files can contain stale mode metadata.
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
        if valid_model_mode is not None and valid_model_mode != valid_inferred_mode:
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
                    prob_threshold: int = 128,
                    min_read_length: int = 1000) -> None:
    """Add read filtering arguments (--min-mapq, --prob-threshold, --min-read-length)."""
    parser.add_argument(
        '--min-mapq', '-q', type=int, default=min_mapq,
        help="Minimum mapping quality; reads below this are written to output "
             "unchanged without footprint/nucleosome tags. Default 0 (call on "
             "all mapped reads). Pass a positive value to filter."
    )
    parser.add_argument(
        '--prob-threshold', type=int, default=prob_threshold,
        help=f"Minimum MM/ML probability (0-255) to call modification (default: {prob_threshold})"
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


def add_version_args(parser: argparse.ArgumentParser) -> None:
    """Add --version flag."""
    from fiberhmm import __version__
    parser.add_argument(
        '--version', action='version',
        version=f'%(prog)s {__version__}'
    )
