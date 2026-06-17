"""Output path helpers for probability-generation tables."""

import os


def probability_counter_path(
    output_dir: str,
    base_name: str,
    sample_name: str,
    base: str,
    *,
    temporary: bool = False,
) -> str:
    suffix = ".probs.pkl.tmp" if temporary else ".probs.pkl"
    return os.path.join(output_dir, f"{base_name}_{sample_name}_{base}{suffix}")


def probability_table_path(
    tables_dir: str,
    base_name: str,
    sample_name: str,
    base: str,
    context_size: int,
) -> str:
    return os.path.join(
        tables_dir,
        f"{base_name}_{sample_name}_{base}_k{context_size}.tsv",
    )


def combined_probability_table_path(
    tables_dir: str,
    base_name: str,
    base: str,
    context_size: int,
) -> str:
    return os.path.join(tables_dir, f"{base_name}_{base}_k{context_size}_probs.tsv")
