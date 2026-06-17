"""Helpers for BAM index discovery and creation."""

from __future__ import annotations

import os


def bam_index_paths(bam_path: str) -> tuple[str, str]:
    return bam_path + '.bai', bam_path.replace('.bam', '.bai')


def bam_index_exists(bam_path: str) -> bool:
    return any(os.path.exists(path) for path in bam_index_paths(bam_path))


def ensure_bam_index(bam_path: str, message: str) -> None:
    if bam_index_exists(bam_path):
        return
    print(message)
    import pysam
    pysam.index(bam_path)
