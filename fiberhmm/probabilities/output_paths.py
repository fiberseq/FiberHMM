"""Output path helpers for probability-generation tables."""

import os


def _probability_filename(*parts, suffix: str) -> str:
    return "_".join(str(part) for part in parts) + suffix


def probability_counter_path(
    output_dir: str,
    base_name: str,
    sample_name: str,
    base: str,
    *,
    temporary: bool = False,
) -> str:
    suffix = ".probs.pkl.tmp" if temporary else ".probs.pkl"
    return os.path.join(
        output_dir,
        _probability_filename(base_name, sample_name, base, suffix=suffix),
    )


def probability_table_path(
    tables_dir: str,
    base_name: str,
    sample_name: str,
    base: str,
    context_size: int,
) -> str:
    return os.path.join(
        tables_dir,
        _probability_filename(
            base_name,
            sample_name,
            base,
            f"k{context_size}",
            suffix=".tsv",
        ),
    )


def combined_probability_table_path(
    tables_dir: str,
    base_name: str,
    base: str,
    context_size: int,
) -> str:
    return os.path.join(
        tables_dir,
        _probability_filename(
            base_name,
            base,
            f"k{context_size}",
            "probs",
            suffix=".tsv",
        ),
    )
