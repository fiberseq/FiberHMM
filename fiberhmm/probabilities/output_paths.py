"""Output path helpers for probability-generation tables."""

import os


def _probability_filename(*parts, suffix: str) -> str:
    return "_".join(str(part) for part in parts) + suffix


def _probability_path(directory: str, *parts, suffix: str) -> str:
    return os.path.join(directory, _probability_filename(*parts, suffix=suffix))


def _probability_counter_suffix(temporary: bool) -> str:
    return ".probs.pkl.tmp" if temporary else ".probs.pkl"


def probability_counter_path(
    output_dir: str,
    base_name: str,
    sample_name: str,
    base: str,
    *,
    temporary: bool = False,
) -> str:
    suffix = _probability_counter_suffix(temporary)
    return _probability_path(output_dir, base_name, sample_name, base, suffix=suffix)


def probability_table_path(
    tables_dir: str,
    base_name: str,
    sample_name: str,
    base: str,
    context_size: int,
) -> str:
    return _probability_path(
        tables_dir,
        base_name,
        sample_name,
        base,
        f"k{context_size}",
        suffix=".tsv",
    )


def combined_probability_table_path(
    tables_dir: str,
    base_name: str,
    base: str,
    context_size: int,
) -> str:
    return _probability_path(
        tables_dir,
        base_name,
        base,
        f"k{context_size}",
        "probs",
        suffix=".tsv",
    )
