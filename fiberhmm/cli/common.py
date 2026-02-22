"""Shared argparse argument factories for FiberHMM CLI tools.

Each function adds a group of related arguments to an ArgumentParser.
Default values can be overridden per-script where needed.
"""

import argparse


def add_mode_args(parser: argparse.ArgumentParser,
                  default: str = 'pacbio-fiber',
                  required: bool = False) -> None:
    """Add --mode argument."""
    parser.add_argument(
        '--mode',
        choices=['pacbio-fiber', 'nanopore-fiber', 'daf'],
        default=None if required else default,
        required=required,
        help=f"Analysis mode (default: {default})"
    )


def add_filter_args(parser: argparse.ArgumentParser,
                    min_mapq: int = 20,
                    prob_threshold: int = 128,
                    min_read_length: int = 1000) -> None:
    """Add read filtering arguments (--min-mapq, --prob-threshold, --min-read-length)."""
    parser.add_argument(
        '--min-mapq', '-q', type=int, default=min_mapq,
        help=f"Minimum mapping quality (default: {min_mapq})"
    )
    parser.add_argument(
        '--prob-threshold', type=int, default=prob_threshold,
        help=f"Minimum MM/ML probability (0-255) to call modification (default: {prob_threshold})"
    )
    parser.add_argument(
        '--min-read-length', type=int, default=min_read_length,
        help=f"Minimum aligned read length in bp (default: {min_read_length})"
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
