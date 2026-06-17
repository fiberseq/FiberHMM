"""Helpers for BAM index discovery and creation."""

from __future__ import annotations

import os


def bam_index_exists(bam_path: str) -> bool:
    return (
        os.path.exists(bam_path + '.bai') or
        os.path.exists(bam_path.replace('.bam', '.bai'))
    )


def ensure_bam_index(bam_path: str, message: str) -> None:
    if bam_index_exists(bam_path):
        return
    print(message)
    import pysam
    pysam.index(bam_path)
